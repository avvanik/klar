"""Golden-fixture checks over REAL committed output (see fixtures/README.md).

These skip cleanly until someone runs the pipeline once with real keys and commits
`fixtures/<stem>/brief.json`. After that they prove the committed output is genuine,
schema-valid, and in the expected language.

They deliberately do NOT assert on brief wording/keywords: extraction is
non-deterministic run to run (one run says "Acme", another says "the customer"), so
gating a frozen snapshot on content would be flaky. Content quality is gated live by
`python -m klar eval`, which re-runs the pipeline against samples/expectations.yaml.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from klar.evaluate import normalize_lang
from klar.models import Brief

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = _ROOT / "fixtures"


def _fixture_dirs() -> dict[str, Path]:
    if not _FIXTURES.exists():
        return {}
    return {d.name: d for d in _FIXTURES.iterdir() if d.is_dir() and (d / "brief.json").exists()}


def test_committed_briefs_validate():
    dirs = _fixture_dirs()
    if not dirs:
        pytest.skip("no fixtures yet - run `klar run <sample> --save-fixtures` with keys")
    for name, d in dirs.items():
        brief = Brief.model_validate_json((d / "brief.json").read_text())
        assert brief.summary.strip(), f"{name}: empty summary"
        assert (d / "transcript.txt").read_text().strip(), f"{name}: empty transcript"


def test_committed_briefs_are_in_expected_language():
    dirs = _fixture_dirs()
    if not dirs:
        pytest.skip("no fixtures yet - run `klar run <sample> --save-fixtures` with keys")
    expectations = yaml.safe_load((_ROOT / "samples" / "expectations.yaml").read_text()) or {}
    for spec in expectations.get("samples", []):
        stem = Path(spec["file"]).stem
        expected = spec.get("language")
        if stem not in dirs or not expected:
            continue
        brief = Brief.model_validate_json((dirs[stem] / "brief.json").read_text())
        assert normalize_lang(brief.language) == normalize_lang(expected), (
            f"{stem}: expected {expected}, got {brief.language}"
        )
