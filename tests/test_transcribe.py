from __future__ import annotations

import pytest

from klar.errors import TranscriptionError
from klar.transcribe import ElevenLabsTranscriber, normalise_response

from .fakes import FakeScribeResponse, make_scribe_response, write_fake_audio


def test_normalise_maps_words_speakers_duration():
    doc = normalise_response(make_scribe_response(), stt_model="scribe_v2")
    assert doc.language_code == "eng"
    assert doc.speakers == ["speaker_0", "speaker_1"]
    assert doc.duration_seconds == pytest.approx(2.5)
    # diarized text groups consecutive speakers into turns
    dt = doc.diarized_text()
    assert "speaker_0:" in dt and "speaker_1:" in dt


def test_normalise_accepts_plain_dict():
    payload = {
        "language_code": "deu",
        "text": "Hallo Welt",
        "words": [{"text": "Hallo", "start": 0, "end": 0.5, "type": "word", "speaker_id": "speaker_0"}],
    }
    doc = normalise_response(payload, stt_model="scribe_v2")
    assert doc.language_code == "deu"
    assert doc.text == "Hallo Welt"


def test_normalise_empty_raises():
    with pytest.raises(TranscriptionError):
        normalise_response(FakeScribeResponse(text="", words=[]), stt_model="scribe_v2")


def test_transcriber_passes_customer_kwargs(tmp_path):
    """The injected client should receive scribe_v2 + diarize + customer keyterms."""
    captured = {}

    class DummySTT:
        def convert(self, **kwargs):
            captured.update(kwargs)
            return make_scribe_response()

    class DummyClient:
        speech_to_text = DummySTT()

    t = ElevenLabsTranscriber("fake-key", client=DummyClient())
    audio = write_fake_audio(tmp_path / "call.mp3")
    t.transcribe(audio, language_code="eng", keyterms=["Acme"], num_speakers=2)

    assert captured["model_id"] == "scribe_v2"
    assert captured["diarize"] is True
    assert captured["timestamps_granularity"] == "word"
    assert captured["language_code"] == "eng"
    assert captured["keyterms"] == ["Acme"]
    assert captured["num_speakers"] == 2
