from __future__ import annotations

import pytest

from klar.customers import CustomerConfig
from klar.errors import EmptyAudioError
from klar.models import Brief
from klar.pipeline import run_pipeline

from .fakes import FakeLLM, FakeTranscriber, write_fake_audio


def test_pipeline_happy_path(store, audio_file):
    result = run_pipeline(audio_file, store=store, transcriber=FakeTranscriber(), llm=FakeLLM())
    assert result.status == "ok"
    assert isinstance(result.brief, Brief)
    # a run row and artifacts were written
    runs = store.list_runs()
    assert len(runs) == 1
    assert runs[0]["status"] == "ok"
    assert runs[0]["llm_input_tokens"] == 100
    assert result.record.brief_path and result.record.transcript_path


def test_pipeline_is_idempotent(store, audio_file):
    t, llm = FakeTranscriber(), FakeLLM()
    first = run_pipeline(audio_file, store=store, transcriber=t, llm=llm)
    second = run_pipeline(audio_file, store=store, transcriber=t, llm=llm)  # same bytes
    assert first.status == "ok"
    assert second.status == "skipped"
    assert second.reused is True
    # exactly ONE successful run recorded; no duplicate state
    ok_runs = [r for r in store.list_runs() if r["status"] == "ok"]
    assert len(ok_runs) == 1


def test_pipeline_force_creates_new_version(store, audio_file):
    t, llm = FakeTranscriber(), FakeLLM()
    run_pipeline(audio_file, store=store, transcriber=t, llm=llm)
    forced = run_pipeline(audio_file, store=store, transcriber=t, llm=llm, force=True)
    assert forced.status == "ok"
    assert forced.version == 2


def test_pipeline_customer_kwargs_reach_transcriber(store, audio_file):
    t = FakeTranscriber()
    cfg = CustomerConfig(name="acme", language_hints=["eng"], keyterms=["Acme"], num_speakers=2)
    run_pipeline(audio_file, store=store, transcriber=t, llm=FakeLLM(), customer=cfg)
    assert t.last_options.get("language_code") == "eng"
    assert t.last_options.get("keyterms") == ["Acme"]
    assert t.last_options.get("num_speakers") == 2


def test_pipeline_records_failure_and_raises(store, tmp_path):
    bad = tmp_path / "empty.mp3"
    bad.write_bytes(b"")
    with pytest.raises(EmptyAudioError):
        run_pipeline(bad, store=store, transcriber=FakeTranscriber(), llm=FakeLLM())
    # ingest fails before a run row is created (no version registered)
    assert store.list_runs() == []


def test_pipeline_records_failure_after_ingest(store, audio_file):
    class BoomTranscriber:
        def transcribe(self, *a, **k):
            from klar.errors import TranscriptionError
            raise TranscriptionError("scribe exploded")

    from klar.errors import TranscriptionError
    with pytest.raises(TranscriptionError):
        run_pipeline(audio_file, store=store, transcriber=BoomTranscriber(), llm=FakeLLM())
    runs = store.list_runs()
    assert len(runs) == 1 and runs[0]["status"] == "failed"
    assert "scribe exploded" in runs[0]["error"]
