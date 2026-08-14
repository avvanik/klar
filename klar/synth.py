"""Optional voice loop: read a brief's summary back aloud via ElevenLabs TTS.

This is Klar's second ElevenLabs surface (Text-to-Speech) alongside Scribe (STT):
the brief you just extracted, spoken. It's behind a `Speaker` protocol so it's
swappable and testable, and the real client is lazy-imported, so nothing here loads
unless you pass `--speak`. The `convert(...)` shape mirrors scripts/generate_samples.py
so there's no new API surface to verify.
"""

from __future__ import annotations

from typing import Any, Protocol

from .models import Brief

DEFAULT_TTS_MODEL = "eleven_multilingual_v2"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"


class Speaker(Protocol):
    """Anything that turns text into audio bytes."""

    def synthesize(self, text: str) -> bytes:
        ...


def spoken_summary(brief: Brief, *, max_items: int = 3) -> str:
    """The utterance to read aloud: the summary plus the top few action items.

    Kept short on purpose: a spoken recap is a headline, not the whole brief.
    """
    parts = [brief.summary.strip()]
    if brief.action_items:
        parts.append("Key next steps:")
        for a in brief.action_items[:max_items]:
            parts.append(f"{a.owner}, {a.text}")
    return " ".join(p for p in parts if p)


def first_available_voice(client: Any) -> str:
    """Pick any voice on the account, so --speak works without extra config."""
    resp = client.voices.get_all()
    voices = getattr(resp, "voices", resp)
    for v in voices:
        vid = getattr(v, "voice_id", None) or (v.get("voice_id") if isinstance(v, dict) else None)
        if vid:
            return vid
    raise RuntimeError("No voices available on this ElevenLabs account.")


class ElevenLabsSpeaker:
    """Real TTS speaker. Lazy-imports the SDK; picks a default voice if unset."""

    def __init__(
        self,
        api_key: str,
        *,
        voice_id: str | None = None,
        model_id: str = DEFAULT_TTS_MODEL,
        output_format: str = DEFAULT_OUTPUT_FORMAT,
        client: Any | None = None,
    ):
        self.model_id = model_id
        self.output_format = output_format
        if client is not None:
            self._client = client
        else:
            try:
                from elevenlabs.client import ElevenLabs
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "The 'elevenlabs' package is not installed. Run: pip install elevenlabs"
                ) from exc
            self._client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id or first_available_voice(self._client)

    def synthesize(self, text: str) -> bytes:
        audio = self._client.text_to_speech.convert(
            voice_id=self.voice_id,
            text=text,
            model_id=self.model_id,
            output_format=self.output_format,
        )
        if isinstance(audio, (bytes, bytearray)):
            return bytes(audio)
        return b"".join(audio)  # SDK returns a byte-chunk iterator
