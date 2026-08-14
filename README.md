# Klar: Call Intelligence on the ElevenLabs API

**Klar** (German for *clear*) turns a call or meeting recording into a structured
brief: **summary, action items with owners, decisions, open questions, per-speaker
sentiment**, in the recording's own language.

It transcribes with **ElevenLabs Scribe v2** (batch, diarized, word-level timestamps)
and extracts the brief in one validated LLM pass. Long recordings are split on speaker
turns and merged, so a full meeting doesn't overflow the context. Tested in English,
German, and Spanish. It's idempotent, observable, retries transient API failures, and
ships with an eval harness that fails loudly on bad input.

```
python -m klar run samples/en_sales_call.mp3 --customer acme_sales
```

---

## What it does

Drop in an audio/video recording; get back a strict-schema JSON brief plus a row in
a run log. Re-running the same file doesn't duplicate work. A per-customer YAML file
changes how both transcription and extraction behave: the same pipeline, adapted per
account without code changes.

Two supporting docs: [`docs/deployment-one-pager.md`](docs/deployment-one-pager.md)
covers how Klar is configured and rolled out per customer;
[`docs/elevenagents-qa.md`](docs/elevenagents-qa.md) is the realtime voice Q&A design.

## 30-second quickstart

```bash
# 1. Install
pip install -r requirements.txt          # or: pip install -e .

# 2. Secrets (never committed)
cp .env.example .env                      # then fill in ELEVENLABS_API_KEY + ANTHROPIC_API_KEY

# 3. Make sample audio (English / German / Spanish, two speakers each)
python scripts/generate_samples.py

# 4. Run the pipeline
python -m klar run samples/en_sales_call.mp3 --customer acme_sales

# 5. See the observability log
python -m klar runs

# 6. Run the eval harness (exits non-zero on failure)
python -m klar eval
```

No keys yet? `python -m klar eval --negative-only` and `pytest` both run fully
offline. The robustness checks and unit tests mock the SDKs.

## Architecture

```
                       customers/<name>.yaml
                (language hints · keyterms · emphasis · glossary)
                                   │
      audio ─▶ ┌─────────┐  ┌────────────┐  ┌──────────┐  ┌──────────┐
   file/folder │ ingest  │─▶│ transcribe │─▶│ extract  │─▶│ persist  │
               │ hash +  │  │ Scribe v2  │  │ 1 LLM    │  │ transcript│
               │ validate│  │ diarize +  │  │ pass →   │  │ + brief   │
               └────┬────┘  │ timestamps │  │ STRICT   │  │ (files)   │
                    │       └─────┬──────┘  │ JSON     │  └────┬─────┘
        idempotency │             │         │ (retry×1)│       │
        check: run  │             │         └────┬─────┘       │
        already ok? │             │              │            │
                    ▼             ▼              ▼            ▼
               ┌────────────────────────────────────────────────┐
               │  SQLite: inputs (hash→version) + runs (metrics) │
               │  hash · language · model versions · latency/stage
               │  · token/char cost estimate · pass|fail          │
               └────────────────────────────────────────────────┘
                                   │
                  python -m klar runs   │   python -m klar eval
                 (run-log table)        │   (fails loudly on bad input)

Interfaces (swappable, mockable):  Transcriber  ·  LLMClient
Default implementations:           ElevenLabs Scribe v2  ·  Anthropic Claude
```

### Package layout

```
klar/
  ingest.py      validate input + content hash (fails loudly, pre-API)
  transcribe.py  Transcriber protocol + ElevenLabsTranscriber (Scribe v2 batch)
  extract.py     LLMClient protocol + AnthropicClient; strict-JSON extract + retry
  synth.py       Speaker protocol + ElevenLabsSpeaker (TTS spoken summary)
  retry.py       transient-failure backoff shared by both API calls
  models.py      TranscriptDoc + the strict Brief schema (pydantic, extra=forbid)
  customers.py   per-customer YAML → STT kwargs + extraction prompt controls
  store.py       SQLite (idempotency + run log) and on-disk artifacts
  observe.py     stage timing, cost estimates, RunRecord
  pipeline.py    orchestration: ingest→transcribe→extract→persist, timed + logged
  evaluate.py    eval harness (offline robustness + keyed positive checks)
  config.py      env-only settings;  __main__.py  CLI (run / runs / eval)
customers/  acme_sales.yaml · medtech_support.yaml
samples/    expectations.yaml (+ generated audio)
scripts/    generate_samples.py (ElevenLabs TTS)
fixtures/   real committed transcript+brief output (see fixtures/README.md)
docs/       deployment-one-pager.md · elevenagents-qa.md
tests/      unit tests + eval harness (SDKs mocked, run offline)
```

## Sample brief output

`python -m klar run samples/en_sales_call.mp3 --customer acme_sales` →

```json
{
  "language": "eng",
  "summary": "Acme followed up on the customer's proof of concept. Pricing and a security review are the remaining blockers; Acme committed to sending the order form and pricing by Friday and to sharing the security document this week. The customer will sign off if the numbers work.",
  "action_items": [
    { "owner": "agent",    "text": "Send the order form and pricing by Friday." },
    { "owner": "agent",    "text": "Loop in the security team and share the security review document this week." },
    { "owner": "customer", "text": "Sign off on the proof of concept once pricing is confirmed." }
  ],
  "decisions": [
    "Proceed with the proof of concept, contingent on acceptable pricing."
  ],
  "open_questions": [
    "What are the final contract numbers?",
    "Does the security review meet the customer's requirements?"
  ],
  "sentiment_by_speaker": [
    { "speaker": "agent",    "sentiment": "positive", "rationale": "Confident, committed to clear next steps." },
    { "speaker": "customer", "sentiment": "mixed",    "rationale": "Impressed but withholding until pricing and security are resolved." }
  ]
}
```

The same call under a different customer config produces a different transcript
request (language, boosted keyterms, role labels) and a differently-weighted brief.

### Per-customer adaptability

`customers/acme_sales.yaml` (English sales) boosts brand/product keyterms
(`Acme`, `proof of concept`, `order form`), labels speakers `agent`/`customer`, and
tells the extractor to emphasise **action items** and **decisions**.

`customers/medtech_support.yaml` (German support) switches Scribe to German, boosts
clinical device terms (`CardioScan Pro`, `Sensor-Kalibrierung`), enables
`no_verbatim` for a cleaner clinical transcript, and emphasises **open questions**
and **per-speaker sentiment**. Swapping the `--customer` flag visibly changes both
the STT call and the resulting brief.

## Behaviour

**Idempotency.** A recording's identity is the **SHA-256 of its bytes**, not its
filename, so the same file (even renamed) is recognised. If a successful run exists we
**skip** (no re-transcription, no duplicate brief, no second bill); `--force` writes a
**new version** instead of overwriting. *Verified:* an identical rerun yields
`status=skipped` and exactly one `ok` run; `--force` yields `v2`. The check is
per-process; two processes racing on the same brand-new file would both bill, so a real
deployment adds a per-hash lock.

**Resilient, and fails loudly.** Bad input raises a typed error **before any paid call**
(`MissingFileError`, `EmptyAudioError`, `UnsupportedFormatError`); an empty Scribe
result raises `TranscriptionError`; malformed LLM output is retried once with the
validation error fed back, then raises `SchemaValidationError`. Transient API failures
(429/5xx, dropped connections) get bounded backoff retries; client errors (400/401/404)
don't. A typo'd `KLAR_LLM_MODEL` is rejected *before* transcription, so it can't run up
a Scribe bill and then crash at extraction. Long calls that would overflow the context
are split on speaker turns and merged in a reduce pass (`llm_chunks` shows how many).

**Eval that gates.** `python -m klar eval` exits non-zero if either half fails.
(A) **Robustness** (offline, no keys). Ingest raises the right typed error on empty,
truncated, wrong-format, and missing input. (B) **Quality** runs the pipeline over
`samples/expectations.yaml` and asserts each brief validates and contains that clip's
known entities (`Acme`, `Friday`, `CardioScan`, `presupuesto`). Corrupt a sample and it
goes red.

**Observability.** Every run, success *or* failure, writes a `RunRecord`: input hash,
language, model versions, per-stage latency, token counts, a cost estimate, and
`pass|fail`. `python -m klar runs` prints it as a table.

**Swappable by interface.** Transcription sits behind a `Transcriber` protocol,
extraction behind an `LLMClient`; SDKs are lazy-imported, so the whole test suite runs
offline with fakes and you can swap providers with one adapter.

## Configuration

All secrets and tunables come from the environment (see `.env.example`):
`ELEVENLABS_API_KEY`, `ANTHROPIC_API_KEY`, `KLAR_LLM_PROVIDER` (default `anthropic`),
`KLAR_LLM_MODEL` (default `claude-sonnet-4-6`), `KLAR_STT_MODEL` (default `scribe_v2`),
`KLAR_HOME`, and the reliability knobs (`KLAR_LLM_MAX_TOKENS`,
`KLAR_MAX_TRANSCRIPT_CHARS`, `KLAR_REQUEST_TIMEOUT_S`, `KLAR_MAX_ATTEMPTS`,
`KLAR_ALLOW_UNKNOWN_MODEL`). Cost-estimate rates are defaults in code.

## Tests

```bash
pip install -r requirements.txt
pytest                     # 57 unit tests: hashing, ingest, STT mapping, extract
                           # retry, store idempotency, pipeline, eval, reliability
                           # paths (API retries, model fail-fast, long-transcript
                           # map-reduce), and golden fixtures
```

Tests mock the ElevenLabs and Anthropic SDKs, so they need no network and no keys.

## ElevenLabs Scribe API shape

Klar targets this call shape:

`client.speech_to_text.convert(file=…, model_id="scribe_v2", diarize=True,
tag_audio_events=True, timestamps_granularity="word", language_code="eng",
keyterms=[…], num_speakers=…, detect_speaker_roles=…, no_verbatim=…)`.
Response: `language_code`, `language_probability`, `text`, and `words[]`
(`text`, `start`, `end`, `type` ∈ {word, spacing, audio_event}, `speaker_id`).
Language codes are ISO-639-3 (`eng`, `deu`, `spa`).

> **Verify before a live run.** `model_id` and the exact parameter names are pinned to
> what was current when this was written. Confirm them against the ElevenLabs docs for
> your SDK version and set `KLAR_STT_MODEL` accordingly. The normalisation layer
> (`transcribe.py`) reads response fields defensively so minor shape drift won't crash
> the pipeline, but the model id must be one your account can call.

## Voice loop

`python -m klar run <file> --speak` reads the brief's summary and top next-steps back
aloud via **Text-to-Speech**, saving a `summary.mp3` next to the brief. The pipeline
uses two ElevenLabs surfaces: Scribe (STT) in and Text-to-Speech out.

## Stretch (scoped, not built)

An **ElevenAgents** voice agent that answers questions over the stored briefs and
replies in voice ("what did we commit to Acme last week?"). The `Brief` artifacts are a
knowledge base for it. Realtime design (latency budget, streaming transcription, eval
approach) is in [`docs/elevenagents-qa.md`](docs/elevenagents-qa.md).

## License

MIT.
