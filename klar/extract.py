"""Extract stage: one LLM pass -> STRICT JSON -> validated Brief.

Design points that matter for production:
  * The LLM is behind an `LLMClient` Protocol so the provider/model is swappable
    (default: Anthropic). Tests inject a fake client - no network, deterministic.
  * The prompt asks for JSON only; we still parse defensively (strip fences,
    slice to the outermost braces) because models occasionally add prose.
  * Output is validated against the `Brief` pydantic schema. On failure we retry
    exactly once, feeding the validation error back to the model. Second failure
    raises SchemaValidationError - loud, not lossy.
  * Token usage is captured for the cost estimate in the run log.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from .customers import CustomerConfig, default_config
from .errors import ConfigError, SchemaValidationError
from .models import Brief
from .retry import retry_call
from .transcribe import TranscriptDoc

# Extraction defaults. Output budget is generous enough for a full brief; the
# transcript budget bounds a single LLM pass so long meetings map-reduce instead
# of silently truncating (see extract_brief).
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_MAX_TRANSCRIPT_CHARS = 24000

# Model families we know resolve today. This is a typo guard, not a hard
# allowlist: unknown ids are rejected with a clear error unless the caller opts
# out (allow_unknown=True / KLAR_ALLOW_UNKNOWN_MODEL=1), so a genuinely new model
# is one env var away rather than a silent 404 after a paid transcription.
KNOWN_MODEL_PREFIXES = (
    "claude-opus-4",
    "claude-sonnet-4",
    "claude-haiku-4",
    "claude-fable-5",
)


def validate_anthropic_model(model: str, *, allow_unknown: bool = False) -> None:
    """Fail fast on an obviously-invalid model id, BEFORE any paid API call."""
    if allow_unknown or any(model.startswith(p) for p in KNOWN_MODEL_PREFIXES):
        return
    raise ConfigError(
        f"KLAR_LLM_MODEL={model!r} is not a recognised Anthropic model. "
        f"Known families: {', '.join(KNOWN_MODEL_PREFIXES)} (e.g. 'claude-sonnet-4-6'). "
        "If this is a new model, set KLAR_ALLOW_UNKNOWN_MODEL=1 to bypass this check."
    )


@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    truncated: bool = False  # True if the model hit max_tokens (output cut off)


class LLMClient(Protocol):
    """Minimal single-call chat interface. Provider-agnostic on purpose."""

    def complete(self, system: str, user: str, *, max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS, temperature: float = 0.0) -> LLMResult:
        ...


@dataclass
class ExtractionMeta:
    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    chunks: int = 1  # >1 when a long transcript was map-reduced
    errors: list[str] = field(default_factory=list)


SYSTEM_PROMPT = (
    "You are Klar, a precise meeting-analysis engine. You read a diarized "
    "transcript and extract a structured brief. You output ONLY a single JSON "
    "object, no prose, no markdown fences. Write all human-readable strings in "
    "the SAME LANGUAGE as the transcript. Do not invent facts; if a field has no "
    "content, use an empty list. Owners of action items must be a named speaker "
    "or 'unassigned'."
)

SCHEMA_HINT = """Return JSON with EXACTLY these keys and no others:
{
  "language": "<ISO code of the transcript language>",
  "summary": "<2-4 sentence summary>",
  "action_items": [{"owner": "<name or 'unassigned'>", "text": "<action>"}],
  "decisions": ["<decision>"],
  "open_questions": ["<question>"],
  "sentiment_by_speaker": [
    {"speaker": "<speaker id/name>",
     "sentiment": "positive|neutral|negative|mixed",
     "rationale": "<short reason>"}
  ]
}"""


def build_user_prompt(
    transcript: TranscriptDoc,
    customer: CustomerConfig,
    *,
    body: str | None = None,
    note: str | None = None,
) -> str:
    """Assemble the extraction prompt.

    `body` overrides the transcript text (used to pass a single chunk during
    map-reduce); `note` is a leading instruction (e.g. "this is part 2 of 4").
    """
    parts: list[str] = []
    if note:
        parts.append(note)
    parts.append(f"Transcript language (detected): {transcript.language_code}")
    if transcript.speakers:
        parts.append(f"Speakers present: {', '.join(transcript.speakers)}")

    if customer.glossary:
        gloss = "\n".join(f"- {term}: {desc}" for term, desc in customer.glossary.items())
        parts.append("Domain glossary (interpret these terms correctly):\n" + gloss)

    if customer.emphasis:
        parts.append(
            "Pay special attention to these fields; be thorough and specific in them: "
            + ", ".join(customer.emphasis)
        )

    if customer.extraction_instructions:
        parts.append("Extra instructions: " + customer.extraction_instructions)

    parts.append(SCHEMA_HINT)
    # Transcript content is untrusted input. Its blast radius is bounded by strict
    # schema validation of the output (a hostile transcript can at worst yield a
    # wrong-but-valid brief, not arbitrary keys or actions), but a hardened
    # deployment would additionally sandbox instructions and flag injection attempts.
    parts.append("Transcript:\n" + (body if body is not None else transcript.diarized_text()))
    return "\n\n".join(parts)


def _parse_json_object(text: str) -> dict[str, Any]:
    """Best-effort extraction of a single JSON object from model output."""
    s = text.strip()
    if s.startswith("```"):
        # strip ```json ... ``` fences
        s = s.split("```", 2)[1] if s.count("```") >= 2 else s.strip("`")
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:]
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in model output")
    return json.loads(s[start : end + 1])


def extract_brief(
    transcript: TranscriptDoc,
    llm: LLMClient,
    customer: CustomerConfig | None = None,
    *,
    max_retries: int = 1,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    max_transcript_chars: int = DEFAULT_MAX_TRANSCRIPT_CHARS,
) -> tuple[Brief, ExtractionMeta]:
    """Transcript -> validated Brief.

    Short transcripts run in a single validated pass (plus up to `max_retries`
    corrective retries). Long transcripts that would overflow the model context
    are split on speaker turns, extracted per chunk, then merged in one reduce
    pass - so a long meeting produces a real brief instead of truncated JSON.
    """
    customer = customer or default_config()
    diarized = transcript.diarized_text()
    meta = ExtractionMeta()

    if len(diarized) <= max_transcript_chars:
        user = build_user_prompt(transcript, customer, body=diarized)
        brief = _complete_to_brief(
            llm, user, meta, max_retries=max_retries, max_output_tokens=max_output_tokens
        )
        return brief, meta

    return _extract_map_reduce(
        transcript, diarized, llm, customer, meta,
        max_retries=max_retries,
        max_output_tokens=max_output_tokens,
        max_transcript_chars=max_transcript_chars,
    )


def _complete_to_brief(
    llm: LLMClient,
    user: str,
    meta: ExtractionMeta,
    *,
    max_retries: int,
    max_output_tokens: int,
) -> Brief:
    """One extraction call with corrective retries; validates against Brief."""
    last_error = ""
    for attempt in range(max_retries + 1):
        prompt = user
        if attempt > 0:
            prompt = (
                user
                + f"\n\nYour previous response was INVALID: {last_error}\n"
                "Return ONLY the corrected JSON object matching the schema exactly."
            )

        result = llm.complete(SYSTEM_PROMPT, prompt, max_tokens=max_output_tokens, temperature=0.0)
        meta.attempts += 1
        meta.input_tokens += result.input_tokens
        meta.output_tokens += result.output_tokens

        try:
            data = _parse_json_object(result.text)
            return Brief(**data)
        except Exception as exc:  # JSON error or pydantic ValidationError
            last_error = str(exc)
            if result.truncated:
                # Surface the real cause instead of a confusing JSON parse error.
                last_error = (
                    f"output was cut off at max_tokens={max_output_tokens} "
                    f"(raise KLAR_LLM_MAX_TOKENS); {last_error}"
                )
            meta.errors.append(f"attempt {attempt + 1}: {last_error}")

    raise SchemaValidationError(
        f"LLM output failed schema validation after {meta.attempts} attempt(s). "
        f"Last error: {last_error}"
    )


def _chunk_turns(diarized: str, max_chars: int) -> list[str]:
    """Split a diarized transcript into <=max_chars chunks on turn boundaries.

    A single turn longer than the budget is hard-split so we never exceed it.
    """
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for line in diarized.split("\n"):
        while len(line) > max_chars:  # pathological single turn
            if buf:
                chunks.append("\n".join(buf))
                buf, size = [], 0
            chunks.append(line[:max_chars])
            line = line[max_chars:]
        add = len(line) + 1
        if size + add > max_chars and buf:
            chunks.append("\n".join(buf))
            buf, size = [], 0
        buf.append(line)
        size += add
    if buf:
        chunks.append("\n".join(buf))
    return chunks


REDUCE_INSTRUCTION = (
    "You are merging {n} partial briefs, each extracted from a consecutive part "
    "of ONE transcript, into a SINGLE final brief. Deduplicate action items, "
    "decisions, and open questions; give one consolidated sentiment per speaker; "
    "write one cohesive summary of the whole call. Output ONLY the JSON object."
)


def _extract_map_reduce(
    transcript: TranscriptDoc,
    diarized: str,
    llm: LLMClient,
    customer: CustomerConfig,
    meta: ExtractionMeta,
    *,
    max_retries: int,
    max_output_tokens: int,
    max_transcript_chars: int,
) -> tuple[Brief, ExtractionMeta]:
    chunks = _chunk_turns(diarized, max_transcript_chars)
    meta.chunks = len(chunks)

    # --- map: a partial brief per chunk ----------------------------------
    partials: list[Brief] = []
    for i, chunk in enumerate(chunks, start=1):
        note = (
            f"This is part {i} of {len(chunks)} of a longer transcript. "
            "Extract a brief for THIS part only; do not speculate about other parts."
        )
        user = build_user_prompt(transcript, customer, body=chunk, note=note)
        partials.append(
            _complete_to_brief(llm, user, meta, max_retries=max_retries, max_output_tokens=max_output_tokens)
        )

    # --- reduce: merge the partials into one final brief -----------------
    payload = json.dumps([p.model_dump() for p in partials], ensure_ascii=False)
    reduce_user = "\n\n".join([
        REDUCE_INSTRUCTION.format(n=len(partials)),
        f"Transcript language (detected): {transcript.language_code}",
        SCHEMA_HINT,
        "Partial briefs (JSON array):\n" + payload,
    ])
    final = _complete_to_brief(
        llm, reduce_user, meta, max_retries=max_retries, max_output_tokens=max_output_tokens
    )
    return final, meta


class AnthropicClient:
    """Default LLM adapter (Anthropic). Lazy-imports the SDK.

    Transient failures (429/5xx/timeouts) are retried with backoff; a per-request
    timeout bounds how long a hung call can stall a batch. The SDK's own retries
    are disabled so backoff isn't applied twice.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "claude-sonnet-4-6",
        client: Any | None = None,
        timeout_s: float = 120.0,
        max_attempts: int = 3,
    ):
        self.model = model
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        if client is not None:
            self._client = client
        else:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "The 'anthropic' package is not installed. Run: pip install anthropic"
                ) from exc
            self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s, max_retries=0)

    def complete(self, system: str, user: str, *, max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS, temperature: float = 0.0) -> LLMResult:
        msg = retry_call(
            lambda: self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            ),
            attempts=self.max_attempts,
        )
        text = "".join(
            block.text for block in msg.content if getattr(block, "type", None) == "text"
        )
        usage = getattr(msg, "usage", None)
        return LLMResult(
            text=text,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            truncated=getattr(msg, "stop_reason", None) == "max_tokens",
        )


def make_llm_client(settings: Any) -> LLMClient:
    """Factory: build the configured LLM client. Extend here to add providers."""
    provider = settings.llm_provider
    if provider == "anthropic":
        validate_anthropic_model(
            settings.llm_model, allow_unknown=getattr(settings, "allow_unknown_model", False)
        )
        return AnthropicClient(
            settings.require_anthropic(),
            model=settings.llm_model,
            timeout_s=getattr(settings, "request_timeout_s", 120.0),
            max_attempts=getattr(settings, "max_attempts", 3),
        )
    raise ValueError(
        f"Unknown KLAR_LLM_PROVIDER={provider!r}. Supported: 'anthropic'. "
        "Add an adapter in klar/extract.py to support more."
    )
