"""Test doubles. These are the ONLY fakes in the codebase - the product ships
real ElevenLabs/Anthropic clients only. Fakes live under tests/ by design.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from klar.extract import LLMResult
from klar.models import TranscriptDoc, Word


# --- A fake Scribe response object (mimics the SDK's attribute access) -----
class FakeWord:
    def __init__(self, text, start, end, type="word", speaker_id=None):
        self.text = text
        self.start = start
        self.end = end
        self.type = type
        self.speaker_id = speaker_id


class FakeScribeResponse:
    def __init__(self, language_code="eng", text="", words=None, language_probability=0.99):
        self.language_code = language_code
        self.language_probability = language_probability
        self.text = text
        self.words = words or []

    def model_dump(self):
        return {
            "language_code": self.language_code,
            "language_probability": self.language_probability,
            "text": self.text,
            "words": [vars(w) for w in self.words],
        }


def make_scribe_response() -> FakeScribeResponse:
    words = [
        FakeWord("Let's", 0.0, 0.4, speaker_id="speaker_0"),
        FakeWord(" ", 0.4, 0.45, type="spacing", speaker_id="speaker_0"),
        FakeWord("ship", 0.45, 0.8, speaker_id="speaker_0"),
        FakeWord(" ", 0.8, 0.85, type="spacing", speaker_id="speaker_0"),
        FakeWord("Friday.", 0.85, 1.3, speaker_id="speaker_0"),
        FakeWord("I", 1.5, 1.7, speaker_id="speaker_1"),
        FakeWord(" ", 1.7, 1.75, type="spacing", speaker_id="speaker_1"),
        FakeWord("need", 1.75, 2.0, speaker_id="speaker_1"),
        FakeWord(" ", 2.0, 2.05, type="spacing", speaker_id="speaker_1"),
        FakeWord("QA.", 2.05, 2.5, speaker_id="speaker_1"),
    ]
    return FakeScribeResponse(text="Let's ship Friday. I need QA.", words=words)


# --- Fake transcriber (implements the Transcriber Protocol) ----------------
class FakeTranscriber:
    def __init__(self, transcript: TranscriptDoc | None = None):
        self.last_options: dict[str, Any] = {}
        self._transcript = transcript or default_transcript()

    def transcribe(self, audio_path, **options) -> TranscriptDoc:
        self.last_options = options
        return self._transcript


def default_transcript() -> TranscriptDoc:
    return TranscriptDoc(
        language_code="eng",
        language_probability=0.99,
        duration_seconds=2.5,
        stt_model="scribe_v2",
        text="Let's ship Friday. I need one more day for QA.",
        words=[
            Word(text="Let's", start=0.0, end=0.4, speaker_id="speaker_0"),
            Word(text="ship", start=0.45, end=0.8, speaker_id="speaker_0"),
            Word(text="Friday.", start=0.85, end=1.3, speaker_id="speaker_0"),
            Word(text="I", start=1.5, end=1.7, speaker_id="speaker_1"),
            Word(text="need", start=1.75, end=2.0, speaker_id="speaker_1"),
            Word(text="QA.", start=2.05, end=2.5, speaker_id="speaker_1"),
        ],
        speakers=["speaker_0", "speaker_1"],
    )


# --- Fake LLM clients (implement the LLMClient Protocol) -------------------
VALID_BRIEF = {
    "language": "eng",
    "summary": "The team agreed to ship on Friday pending one more day of QA.",
    "action_items": [
        {"owner": "speaker_1", "text": "Complete QA before Friday."},
        {"owner": "speaker_0", "text": "Ship the release on Friday."},
    ],
    "decisions": ["Ship on Friday."],
    "open_questions": ["Is one day enough for QA?"],
    "sentiment_by_speaker": [
        {"speaker": "speaker_0", "sentiment": "positive", "rationale": "Eager to ship."},
        {"speaker": "speaker_1", "sentiment": "neutral", "rationale": "Wants more QA time."},
    ],
}


class FakeLLM:
    """Returns a fixed valid brief. `model` attr mirrors the real adapter."""

    def __init__(self, payload: dict | None = None, model: str = "fake-model"):
        self.model = model
        self._payload = payload or VALID_BRIEF
        self.calls = 0

    def complete(self, system, user, *, max_tokens=4096, temperature=0.0) -> LLMResult:
        self.calls += 1
        return LLMResult(text=json.dumps(self._payload), input_tokens=100, output_tokens=50)


class SequenceLLM:
    """Returns a scripted sequence of raw strings - to test retry behaviour."""

    def __init__(self, responses: list[str], model: str = "fake-model", truncated: bool = False):
        self.model = model
        self._responses = list(responses)
        self.truncated = truncated
        self.calls = 0

    def complete(self, system, user, *, max_tokens=4096, temperature=0.0) -> LLMResult:
        text = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return LLMResult(text=text, input_tokens=100, output_tokens=50, truncated=self.truncated)


def write_fake_audio(path: Path, size: int = 2048) -> Path:
    """Create a file that passes ingest validation (right ext, >= MIN_BYTES)."""
    path.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * size)
    return path
