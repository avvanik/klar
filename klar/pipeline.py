"""Pipeline orchestrator: ingest -> transcribe -> extract -> persist + observe.

Dependencies (Transcriber, LLMClient, Store) are injected, so the whole pipeline
runs in tests with fakes and in production with the real ElevenLabs + Anthropic
clients. Each stage is timed; a RunRecord is written whether the run succeeds or
fails, so failures are as observable as successes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .customers import CustomerConfig, default_config
from .errors import KlarError
from .extract import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_TRANSCRIPT_CHARS,
    ExtractionMeta,
    LLMClient,
    extract_brief,
)
from .ingest import ingest
from .models import Brief
from .observe import (
    RunRecord,
    StageTimer,
    estimate_llm_cost,
    estimate_stt_cost,
    new_run_id,
    utcnow_iso,
)
from .store import Store
from .transcribe import Transcriber, TranscriptDoc


@dataclass
class RunResult:
    run_id: str
    status: str  # ok | skipped | failed
    content_hash: str
    version: int
    brief: Brief | None
    transcript: TranscriptDoc | None
    record: RunRecord
    reused: bool = False


@dataclass
class CostRates:
    stt_usd_per_hour: float = 0.40
    llm_usd_per_mtok_input: float = 3.00
    llm_usd_per_mtok_output: float = 15.00


def run_pipeline(
    audio_path: str | Path,
    *,
    store: Store,
    transcriber: Transcriber,
    llm: LLMClient,
    customer: CustomerConfig | None = None,
    rates: CostRates | None = None,
    force: bool = False,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    max_transcript_chars: int = DEFAULT_MAX_TRANSCRIPT_CHARS,
) -> RunResult:
    """Process a single audio file end to end.

    Idempotency:
      * If an unchanged file (same content hash) already has a successful run and
        `force` is False, we SKIP - no re-transcription, no duplicate brief.
      * With `force=True` we create a NEW VERSION rather than overwriting, so
        history is preserved.
    """
    customer = customer or default_config()
    rates = rates or CostRates()
    timer = StageTimer()

    # --- Stage 1: ingest (cheap, fails loudly before any paid call) -------
    with timer.stage("ingest"):
        ing = ingest(audio_path)

    # --- Idempotency check ------------------------------------------------
    existing = store.latest_successful_run(ing.content_hash)
    if existing is not None and not force:
        record = _record_from_existing(existing, reused=True)
        return RunResult(
            run_id=existing["run_id"],
            status="skipped",
            content_hash=ing.content_hash,
            version=int(existing["version"]),
            brief=None,
            transcript=None,
            record=record,
            reused=True,
        )

    version = store.register_input(ing.content_hash, str(ing.path), utcnow_iso())
    run_id = new_run_id()

    record = RunRecord(
        run_id=run_id,
        created_at=utcnow_iso(),
        input_path=str(ing.path),
        content_hash=ing.content_hash,
        version=version,
        customer=customer.name,
        status="failed",  # optimistic flip to ok at the end
    )

    try:
        # --- Stage 2: transcribe -----------------------------------------
        with timer.stage("transcribe"):
            transcript = transcriber.transcribe(ing.path, **customer.stt_kwargs())
        record.language_code = transcript.language_code
        record.stt_model = transcript.stt_model
        record.duration_seconds = transcript.duration_seconds

        # --- Stage 3: extract --------------------------------------------
        with timer.stage("extract"):
            brief, meta = extract_brief(
                transcript, llm, customer,
                max_output_tokens=max_output_tokens,
                max_transcript_chars=max_transcript_chars,
            )

        # --- Stage 4: persist artifacts ----------------------------------
        with timer.stage("persist"):
            transcript_path = store.write_json(ing.content_hash, version, "transcript.json", transcript)
            brief_path = store.write_json(ing.content_hash, version, "brief.json", brief)

        _finalise_ok(record, transcript, meta, rates, timer, transcript_path, brief_path, llm)
        store.insert_run(record)
        return RunResult(
            run_id=run_id,
            status="ok",
            content_hash=ing.content_hash,
            version=version,
            brief=brief,
            transcript=transcript,
            record=record,
        )

    except Exception as exc:  # record the failure, then re-raise loudly
        record.status = "failed"
        record.error = f"{type(exc).__name__}: {exc}"
        record.latency_ms = dict(timer.latencies_ms)
        record.total_latency_ms = timer.total_ms
        store.insert_run(record)
        if isinstance(exc, KlarError):
            raise
        raise KlarError(f"Pipeline failed on {Path(audio_path).name}: {exc}") from exc


def _finalise_ok(
    record: RunRecord,
    transcript: TranscriptDoc,
    meta: ExtractionMeta,
    rates: CostRates,
    timer: StageTimer,
    transcript_path: Path,
    brief_path: Path,
    llm: LLMClient,
) -> None:
    record.status = "ok"
    record.llm_model = getattr(llm, "model", None)
    record.llm_input_tokens = meta.input_tokens
    record.llm_output_tokens = meta.output_tokens
    record.llm_chunks = meta.chunks
    record.stt_cost_usd = estimate_stt_cost(transcript.duration_seconds, rates.stt_usd_per_hour)
    record.llm_cost_usd = estimate_llm_cost(
        meta.input_tokens, meta.output_tokens,
        rates.llm_usd_per_mtok_input, rates.llm_usd_per_mtok_output,
    )
    record.latency_ms = dict(timer.latencies_ms)
    record.total_latency_ms = timer.total_ms
    record.transcript_path = str(transcript_path)
    record.brief_path = str(brief_path)


def _record_from_existing(row, *, reused: bool) -> RunRecord:
    return RunRecord(
        run_id=row["run_id"],
        created_at=row["created_at"],
        input_path=row["input_path"],
        content_hash=row["content_hash"],
        version=int(row["version"]),
        customer=row["customer"],
        status="skipped",
        language_code=row["language_code"],
        stt_model=row["stt_model"],
        llm_model=row["llm_model"],
        duration_seconds=row["duration_seconds"],
        transcript_path=row["transcript_path"],
        brief_path=row["brief_path"],
    )
