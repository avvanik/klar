"""Transcribe stage: ElevenLabs Scribe v2 (batch) with diarization + timestamps.

The transcriber is behind a small `Transcriber` Protocol so the ElevenLabs call
is swappable and, crucially, mockable in tests. Only the real ElevenLabs
implementation ships (per project decision) - no fake is baked into the product.

Confirmed against the current docs (Aug 2026):
    client.speech_to_text.convert(
        file=..., model_id="scribe_v2", diarize=True,
        tag_audio_events=True, timestamps_granularity="word",
        language_code="eng", keyterms=[...], num_speakers=..., ...)
Response: language_code, language_probability, text, words[] where each word has
    text, start, end, type (word|spacing|audio_event), speaker_id.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from .errors import TranscriptionError
from .models import TranscriptDoc, Word
from .retry import retry_call


class Transcriber(Protocol):
    """Anything that turns an audio path into a normalised TranscriptDoc."""

    def transcribe(self, audio_path: str | Path, **options: Any) -> TranscriptDoc:  # noqa: D401
        ...


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a dict or an SDK model object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def normalise_response(payload: Any, *, stt_model: str) -> TranscriptDoc:
    """Convert a Scribe response (object or dict) into a TranscriptDoc.

    Kept as a pure function so tests can exercise the mapping without any SDK.
    """
    language_code = _get(payload, "language_code") or "unknown"
    language_probability = _get(payload, "language_probability")
    text = _get(payload, "text") or ""

    raw_words = _get(payload, "words") or []
    words: list[Word] = []
    speakers: list[str] = []
    max_end = 0.0
    for w in raw_words:
        word = Word(
            text=_get(w, "text", "") or "",
            start=_get(w, "start"),
            end=_get(w, "end"),
            type=_get(w, "type", "word") or "word",
            speaker_id=_get(w, "speaker_id"),
        )
        words.append(word)
        if word.speaker_id and word.speaker_id not in speakers:
            speakers.append(word.speaker_id)
        if isinstance(word.end, (int, float)):
            max_end = max(max_end, float(word.end))

    if not text.strip() and not words:
        raise TranscriptionError(
            "Scribe returned an empty transcript (no text, no words). "
            "The audio may be silent or corrupt."
        )

    # Best-effort duration from the last word's end timestamp.
    duration = max_end or None

    raw = payload if isinstance(payload, dict) else _to_dict(payload)
    return TranscriptDoc(
        language_code=language_code,
        language_probability=language_probability,
        duration_seconds=duration,
        stt_model=stt_model,
        text=text,
        words=words,
        speakers=speakers,
        raw=raw,
    )


def _to_dict(obj: Any) -> dict:
    for attr in ("model_dump", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:  # pragma: no cover - defensive
                pass
    return {}


class ElevenLabsTranscriber:
    """Real Scribe v2 batch transcriber. Lazy-imports the SDK.

    Import is deferred to construction so the rest of Klar (and the unit tests)
    can be imported without the `elevenlabs` package installed.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model_id: str = "scribe_v2",
        client: Any | None = None,
        timeout_s: float = 120.0,
        max_attempts: int = 3,
    ):
        self.model_id = model_id
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        if client is not None:
            self._client = client
        else:
            try:
                from elevenlabs.client import ElevenLabs
            except ImportError as exc:  # pragma: no cover
                raise TranscriptionError(
                    "The 'elevenlabs' package is not installed. Run: pip install elevenlabs"
                ) from exc
            try:  # bound how long a hung request can stall a batch, if supported
                self._client = ElevenLabs(api_key=api_key, timeout=timeout_s)
            except TypeError:  # pragma: no cover - older SDK signature
                self._client = ElevenLabs(api_key=api_key)

    def transcribe(self, audio_path: str | Path, **options: Any) -> TranscriptDoc:
        p = Path(audio_path)
        # Sensible defaults; per-customer options override where provided.
        call_kwargs: dict[str, Any] = {
            "model_id": self.model_id,
            "diarize": True,
            "tag_audio_events": True,
            "timestamps_granularity": "word",
        }
        call_kwargs.update({k: v for k, v in options.items() if v is not None})

        try:
            # Re-open the file each attempt: a consumed stream can't be re-sent.
            def _call():
                with open(p, "rb") as fh:
                    return self._client.speech_to_text.convert(file=fh, **call_kwargs)

            response = retry_call(_call, attempts=self.max_attempts)
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"Scribe transcription failed for {p.name}: {exc}") from exc

        return normalise_response(response, stt_model=self.model_id)
