from __future__ import annotations

import pytest

from klar.store import Store

from .fakes import (  # noqa: F401  (re-exported for tests)
    FakeLLM,
    FakeTranscriber,
    SequenceLLM,
    default_transcript,
    write_fake_audio,
)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "klar.sqlite", tmp_path / "artifacts")
    yield s
    s.close()


@pytest.fixture
def audio_file(tmp_path):
    return write_fake_audio(tmp_path / "call.mp3")
