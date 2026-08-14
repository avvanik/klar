from __future__ import annotations

import types

from klar.models import Brief
from klar.synth import ElevenLabsSpeaker, first_available_voice, spoken_summary

from .fakes import VALID_BRIEF


class _FakeTTS:
    def __init__(self):
        self.kwargs = {}

    def convert(self, **kwargs):
        self.kwargs = kwargs
        return b"AUDIO_BYTES"


class _FakeVoices:
    def get_all(self):
        return types.SimpleNamespace(voices=[{"voice_id": "voice_1"}, {"voice_id": "voice_2"}])


class _FakeElevenClient:
    def __init__(self):
        self.text_to_speech = _FakeTTS()
        self.voices = _FakeVoices()


def test_spoken_summary_includes_summary_and_action_items():
    brief = Brief(**VALID_BRIEF)
    text = spoken_summary(brief)
    assert brief.summary in text
    assert "Ship the release on Friday." in text  # a top action item


def test_spoken_summary_caps_action_items():
    brief = Brief(**VALID_BRIEF)
    text = spoken_summary(brief, max_items=1)
    # only the first action item is spoken
    assert "Complete QA before Friday." in text
    assert "Ship the release on Friday." not in text


def test_first_available_voice_picks_one():
    assert first_available_voice(_FakeElevenClient()) == "voice_1"


def test_speaker_synthesizes_with_expected_kwargs():
    client = _FakeElevenClient()
    speaker = ElevenLabsSpeaker("key", voice_id="voice_1", client=client)
    out = speaker.synthesize("hello")
    assert out == b"AUDIO_BYTES"
    assert client.text_to_speech.kwargs["voice_id"] == "voice_1"
    assert client.text_to_speech.kwargs["text"] == "hello"
    assert "model_id" in client.text_to_speech.kwargs


def test_speaker_auto_picks_voice_when_unset():
    speaker = ElevenLabsSpeaker("key", client=_FakeElevenClient())
    assert speaker.voice_id == "voice_1"
