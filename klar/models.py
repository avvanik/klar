"""Pydantic models: the transcript and the strict Brief schema.

The Brief schema is the contract the LLM must satisfy. It is `extra="forbid"`
on purpose: if the model invents keys or omits required ones, validation fails
and the extract step retries once, then raises. Loud, not lossy.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Sentiment = Literal["positive", "neutral", "negative", "mixed"]


# --- Transcript (output of the transcribe stage) --------------------------
class Word(BaseModel):
    text: str
    start: float | None = None
    end: float | None = None
    type: str = "word"  # word | spacing | audio_event
    speaker_id: str | None = None


class TranscriptDoc(BaseModel):
    """Normalised Scribe v2 output plus the metadata we persist."""

    language_code: str
    language_probability: float | None = None
    duration_seconds: float | None = None
    stt_model: str = "scribe_v2"
    text: str
    words: list[Word] = Field(default_factory=list)
    speakers: list[str] = Field(default_factory=list)
    # The untouched provider payload, kept for auditability/replay.
    raw: dict = Field(default_factory=dict)

    def diarized_text(self) -> str:
        """Speaker-labelled transcript for the LLM prompt.

        Groups consecutive words by speaker so the LLM sees turns, e.g.:
            speaker_0: Let's ship Friday.
            speaker_1: I need one more day for QA.
        Falls back to flat text if there is no diarization.
        """
        if not self.words or not any(w.speaker_id for w in self.words):
            return self.text.strip()

        lines: list[str] = []
        cur_speaker: str | None = None
        buf: list[str] = []

        def flush() -> None:
            if buf and cur_speaker is not None:
                line = "".join(buf).strip()
                if line:
                    lines.append(f"{cur_speaker}: {line}")

        for w in self.words:
            spk = w.speaker_id or cur_speaker or "speaker_0"
            if spk != cur_speaker:
                flush()
                buf = []
                cur_speaker = spk
            buf.append(w.text)
        flush()
        return "\n".join(lines)


# --- Brief (output of the extract stage: the STRICT schema) ---------------
class ActionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    owner: str = Field(description="Who is responsible. Use 'unassigned' if unclear.")
    text: str = Field(min_length=1, description="The action, imperative and specific.")


class SpeakerSentiment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    speaker: str
    sentiment: Sentiment
    rationale: str = ""


class Brief(BaseModel):
    """The structured brief. This is the schema the LLM must match exactly."""

    model_config = ConfigDict(extra="forbid")

    language: str = Field(description="ISO code of the recording's language.")
    summary: str = Field(min_length=1)
    action_items: list[ActionItem] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    sentiment_by_speaker: list[SpeakerSentiment] = Field(default_factory=list)

    def searchable_text(self) -> str:
        """Flatten the brief to one string for keyword-based eval checks."""
        parts = [self.summary]
        parts += [f"{a.owner}: {a.text}" for a in self.action_items]
        parts += self.decisions
        parts += self.open_questions
        parts += [f"{s.speaker} {s.sentiment} {s.rationale}" for s in self.sentiment_by_speaker]
        return "\n".join(parts)
