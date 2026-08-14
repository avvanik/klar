from __future__ import annotations

import pytest

from klar.customers import load_customer
from klar.errors import ConfigError
from klar.evaluate import normalize_lang, run_negative_checks, run_positive_checks


def test_normalize_lang_matches_iso1_and_iso3():
    # the case that was failing live: spa (ISO-639-3) vs es (ISO-639-1)
    assert normalize_lang("spa") == normalize_lang("es")
    assert normalize_lang("eng") == normalize_lang("en")
    assert normalize_lang("deu") == normalize_lang("de")
    # genuinely different languages must not collide
    assert normalize_lang("spa") != normalize_lang("deu")


def test_negative_checks_all_pass():
    """The robustness half of the eval must be green offline (no keys)."""
    report = run_negative_checks()
    assert report.ok, report.render()
    # it actually ran several checks, not zero
    assert len(report.results) >= 5


def test_positive_checks_fail_loudly_when_samples_missing(tmp_path, store):
    from .fakes import FakeLLM, FakeTranscriber
    # expectations.yaml present but audio absent -> should report failures, not crash
    (tmp_path / "expectations.yaml").write_text(
        "samples:\n  - file: missing.mp3\n    expect_keywords: ['x']\n"
    )
    report = run_positive_checks(
        tmp_path, store=store, transcriber=FakeTranscriber(), llm=FakeLLM()
    )
    assert not report.ok
    assert any("missing audio" in r.detail for r in report.failed)


def test_positive_checks_pass_with_fakes(tmp_path, store):
    """Wire fakes + a real audio file to prove the positive path end to end."""
    from .fakes import FakeLLM, FakeTranscriber, write_fake_audio
    write_fake_audio(tmp_path / "call.mp3")
    (tmp_path / "expectations.yaml").write_text(
        "samples:\n"
        "  - file: call.mp3\n"
        "    language: eng\n"
        "    expect_keywords: ['Friday', 'QA']\n"
        "    min_action_items: 2\n"
    )
    report = run_positive_checks(
        tmp_path, store=store, transcriber=FakeTranscriber(), llm=FakeLLM()
    )
    assert report.ok, report.render()


def test_load_customer_default():
    cfg = load_customer(None)
    assert cfg.name == "default"
    assert cfg.stt_kwargs() == {}  # no hints -> auto-detect, no keyterms


def test_load_customer_from_repo_configs():
    cfg = load_customer("acme_sales")
    assert cfg.primary_language() == "eng"
    kw = cfg.stt_kwargs()
    assert kw["language_code"] == "eng"
    assert "Acme" in kw["keyterms"]
    assert kw["detect_speaker_roles"] is True


def test_load_customer_rejects_unknown_emphasis(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: bad\nemphasis: ['not_a_field']\n")
    with pytest.raises(ConfigError):
        load_customer(str(p))
