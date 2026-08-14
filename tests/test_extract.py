from __future__ import annotations

import json

import pytest

from klar.customers import CustomerConfig
from klar.errors import SchemaValidationError
from klar.extract import build_user_prompt, extract_brief
from klar.models import Brief

from .fakes import VALID_BRIEF, FakeLLM, SequenceLLM, default_transcript


def test_extract_valid_first_try():
    brief, meta = extract_brief(default_transcript(), FakeLLM())
    assert isinstance(brief, Brief)
    assert meta.attempts == 1
    assert meta.input_tokens == 100 and meta.output_tokens == 50
    assert len(brief.action_items) == 2


def test_extract_strips_code_fences():
    fenced = "```json\n" + json.dumps(VALID_BRIEF) + "\n```"
    brief, meta = extract_brief(default_transcript(), SequenceLLM([fenced]))
    assert brief.summary


def test_extract_retries_once_then_succeeds():
    bad = "not json at all"
    good = json.dumps(VALID_BRIEF)
    llm = SequenceLLM([bad, good])
    brief, meta = extract_brief(default_transcript(), llm)
    assert meta.attempts == 2
    assert llm.calls == 2
    assert brief.summary


def test_extract_malformed_twice_raises():
    llm = SequenceLLM(["nope", "still nope"])
    with pytest.raises(SchemaValidationError):
        extract_brief(default_transcript(), llm)
    assert llm.calls == 2  # original + one retry


def test_extract_rejects_extra_keys():
    payload = dict(VALID_BRIEF)
    payload["hallucinated_field"] = "oops"  # extra=forbid should reject
    llm = SequenceLLM([json.dumps(payload), json.dumps(payload)])
    with pytest.raises(SchemaValidationError):
        extract_brief(default_transcript(), llm)


def test_extract_rejects_bad_sentiment_label():
    payload = json.loads(json.dumps(VALID_BRIEF))
    payload["sentiment_by_speaker"][0]["sentiment"] = "euphoric"  # not in enum
    llm = SequenceLLM([json.dumps(payload), json.dumps(payload)])
    with pytest.raises(SchemaValidationError):
        extract_brief(default_transcript(), llm)


def test_prompt_includes_customer_glossary_and_emphasis():
    cfg = CustomerConfig(
        name="acme", emphasis=["action_items"], glossary={"POC": "proof of concept"},
        extraction_instructions="Treat commitments as action items.",
    )
    prompt = build_user_prompt(default_transcript(), cfg)
    assert "POC" in prompt
    assert "action_items" in prompt
    assert "commitments" in prompt
