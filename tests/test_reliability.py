"""Tests for the reliability fixes: model fail-fast, retries, long transcripts."""

from __future__ import annotations

import json

import pytest

from klar.errors import ConfigError, SchemaValidationError
from klar.extract import extract_brief, validate_anthropic_model
from klar.models import TranscriptDoc, Word
from klar.retry import RETRYABLE_STATUS, is_retryable, retry_call

from .fakes import VALID_BRIEF, FakeLLM, SequenceLLM, default_transcript


# --- Fix 1: fail fast on an unknown model, before any paid call -----------
def test_validate_model_accepts_known_family():
    validate_anthropic_model("claude-sonnet-4-6")  # must not raise


def test_validate_model_rejects_bogus_id():
    with pytest.raises(ConfigError):
        validate_anthropic_model("claude-sonnet-5")  # the old, invalid default


def test_validate_model_override_bypasses_check():
    validate_anthropic_model("some-future-model", allow_unknown=True)  # must not raise


# --- Fix 2: retry with backoff on transient failures ----------------------
class _HTTPError(Exception):
    def __init__(self, status_code):
        super().__init__(f"http {status_code}")
        self.status_code = status_code


def test_is_retryable_classification():
    assert all(is_retryable(_HTTPError(s)) for s in RETRYABLE_STATUS)
    assert is_retryable(TimeoutError())
    assert is_retryable(ConnectionError())
    # client errors are NOT retryable - fail fast instead of burning attempts
    assert not is_retryable(_HTTPError(400))
    assert not is_retryable(_HTTPError(401))
    assert not is_retryable(_HTTPError(404))


def test_retry_call_recovers_after_transient_error():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _HTTPError(529)  # overloaded
        return "ok"

    slept: list[float] = []
    out = retry_call(flaky, attempts=3, sleep=slept.append)
    assert out == "ok"
    assert calls["n"] == 3
    assert len(slept) == 2  # backed off before each of the 2 retries


def test_retry_call_does_not_retry_client_error():
    calls = {"n": 0}

    def bad():
        calls["n"] += 1
        raise _HTTPError(400)

    with pytest.raises(_HTTPError):
        retry_call(bad, attempts=5, sleep=lambda _: None)
    assert calls["n"] == 1  # no retries on a 400


def test_retry_call_gives_up_after_attempts():
    def always():
        raise _HTTPError(503)

    with pytest.raises(_HTTPError):
        retry_call(always, attempts=3, sleep=lambda _: None)


# --- Fix 3: long transcripts map-reduce instead of truncating -------------
def _long_transcript(n_turns: int = 8) -> TranscriptDoc:
    words = []
    for i in range(n_turns):
        spk = f"speaker_{i % 2}"
        for tok in ("This", "is", "turn", f"number-{i}", "with", "several", "words."):
            words.append(Word(text=tok, start=float(i), end=float(i) + 0.1, speaker_id=spk))
    return TranscriptDoc(language_code="eng", text="…", words=words, speakers=["speaker_0", "speaker_1"])


def test_short_transcript_uses_single_pass():
    llm = FakeLLM()
    _, meta = extract_brief(default_transcript(), llm, max_transcript_chars=10_000)
    assert meta.chunks == 1
    assert llm.calls == 1


def test_long_transcript_map_reduces():
    llm = FakeLLM()
    transcript = _long_transcript()
    diarized = transcript.diarized_text()
    # Force multiple chunks by setting the budget well under the transcript size.
    budget = len(diarized) // 3
    brief, meta = extract_brief(transcript, llm, max_transcript_chars=budget)
    assert meta.chunks >= 2
    # one call per chunk (map) + one reduce call
    assert llm.calls == meta.chunks + 1
    assert brief.summary  # valid merged brief


def test_truncated_output_surfaces_clear_error():
    # Model returns invalid JSON AND signals it was cut off at max_tokens.
    llm = SequenceLLM(["{ this is cut off", "{ still cut off"], truncated=True)
    with pytest.raises(SchemaValidationError) as exc:
        extract_brief(default_transcript(), llm)
    assert "max_tokens" in str(exc.value)


def test_map_reduce_chunk_boundaries_respect_budget():
    from klar.extract import _chunk_turns

    text = "\n".join(f"speaker_{i%2}: line {i} " + "x" * 40 for i in range(10))
    chunks = _chunk_turns(text, max_chars=100)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


# --- Small robustness gaps closed in the polish pass ----------------------
def test_make_llm_client_rejects_unknown_provider():
    import types

    from klar.extract import make_llm_client

    settings = types.SimpleNamespace(llm_provider="openai", llm_model="x")
    with pytest.raises(ValueError):
        make_llm_client(settings)


def test_parse_json_object_ignores_surrounding_prose():
    from klar.extract import _parse_json_object

    text = 'Sure! Here is the brief:\n{"summary": "hi"}\nHope that helps.'
    assert _parse_json_object(text) == {"summary": "hi"}


def test_stt_cost_zero_when_duration_unknown():
    from klar.observe import estimate_stt_cost

    assert estimate_stt_cost(None, 0.40) == 0.0
    assert estimate_stt_cost(3600, 0.40) == pytest.approx(0.40)
