"""Eval harness. Two halves, both required to be green:

  A. Negative / robustness checks (offline, no keys): the pipeline MUST raise the
     right typed error on empty, corrupt, wrong-format, and missing input.
  B. Positive checks (needs ELEVENLABS + ANTHROPIC keys and real sample audio):
     each sample in samples/expectations.yaml transcribes + extracts to a valid
     Brief, and the brief contains the known keywords for that sample.

`run_eval` returns an exit code: 0 = all pass, non-zero = at least one failure.
`python -m klar eval` calls this; so do the pytest tests (negative half).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .customers import default_config, load_customer
from .errors import (
    EmptyAudioError,
    IngestError,
    MissingFileError,
    UnsupportedFormatError,
)
from .ingest import ingest


# ISO-639-3 / -1 aliases so "spa" and "es" (etc.) compare equal.
_LANG_ALIASES = {
    "eng": "en", "en": "en",
    "deu": "de", "ger": "de", "de": "de",
    "spa": "es", "es": "es",
    "fra": "fr", "fre": "fr", "fr": "fr",
    "ita": "it", "it": "it",
    "por": "pt", "pt": "pt",
    "nld": "nl", "dut": "nl", "nl": "nl",
}


def normalize_lang(code: str) -> str:
    """Canonicalise a language code to ISO-639-1, falling back to its first two letters."""
    c = (code or "").strip().lower()
    return _LANG_ALIASES.get(c, c[:2])


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class EvalReport:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.results.append(CheckResult(name, passed, detail))

    @property
    def failed(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed]

    @property
    def ok(self) -> bool:
        return not self.failed

    def render(self) -> str:
        lines = []
        for r in self.results:
            mark = "PASS" if r.passed else "FAIL"
            line = f"[{mark}] {r.name}"
            if r.detail:
                line += f"  -- {r.detail}"
            lines.append(line)
        passed = len(self.results) - len(self.failed)
        lines.append("")
        lines.append(f"{passed}/{len(self.results)} checks passed.")
        return "\n".join(lines)


# --- A. Negative / robustness checks (offline) ----------------------------
def run_negative_checks(report: EvalReport | None = None) -> EvalReport:
    report = report or EvalReport()

    def expect_raises(name: str, exc_type, make_path):
        with tempfile.TemporaryDirectory() as td:
            try:
                path = make_path(Path(td))
                ingest(path)
            except exc_type as e:
                report.add(name, True, f"raised {type(e).__name__}")
            except Exception as e:  # wrong exception type is still a failure
                report.add(name, False, f"raised {type(e).__name__}, expected {exc_type.__name__}")
            else:
                report.add(name, False, "did not raise")

    def make_empty(d: Path) -> Path:
        p = d / "empty.mp3"
        p.write_bytes(b"")
        return p

    def make_corrupt(d: Path) -> Path:
        # A tiny non-audio blob with a valid extension: below MIN_BYTES -> loud fail.
        p = d / "corrupt.wav"
        p.write_bytes(b"\x00\x01\x02")
        return p

    def make_wrong_format(d: Path) -> Path:
        p = d / "notes.txt"
        p.write_bytes(b"this is not audio, it is a text file" * 10)
        return p

    def make_missing(d: Path) -> Path:
        return d / "does_not_exist.mp3"

    expect_raises("empty audio raises loudly", EmptyAudioError, make_empty)
    expect_raises("corrupt/tiny audio raises loudly", EmptyAudioError, make_corrupt)
    expect_raises("wrong format raises loudly", UnsupportedFormatError, make_wrong_format)
    expect_raises("missing file raises loudly", MissingFileError, make_missing)
    # And a broad check that all of the above are subclasses of IngestError.
    report.add(
        "ingest errors share IngestError base",
        all(issubclass(t, IngestError) for t in (EmptyAudioError, UnsupportedFormatError, MissingFileError)),
    )
    return report


# --- B. Positive checks (needs keys + real audio) -------------------------
def load_expectations(samples_dir: str | Path) -> list[dict]:
    path = Path(samples_dir) / "expectations.yaml"
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    return data.get("samples", [])


def run_positive_checks(
    samples_dir: str | Path,
    *,
    store,
    transcriber,
    llm,
    customers_dir: str | Path = "customers",
    report: EvalReport | None = None,
) -> EvalReport:
    """Run the pipeline over each declared sample and assert schema + keywords."""
    from .pipeline import CostRates, run_pipeline  # local import avoids cycle

    report = report or EvalReport()
    samples = load_expectations(samples_dir)
    if not samples:
        report.add("positive checks", False, f"no expectations found in {samples_dir}/expectations.yaml")
        return report

    for spec in samples:
        fname = spec["file"]
        audio = Path(samples_dir) / fname
        name = f"sample '{fname}'"
        if not audio.exists():
            report.add(name, False, f"missing audio file (generate it: scripts/generate_samples.py)")
            continue

        customer = load_customer(spec.get("customer"), customers_dir=customers_dir) if spec.get("customer") else default_config()
        try:
            result = run_pipeline(
                audio, store=store, transcriber=transcriber, llm=llm,
                customer=customer, rates=CostRates(), force=True,
            )
        except Exception as e:
            report.add(name, False, f"pipeline raised {type(e).__name__}: {e}")
            continue

        brief = result.brief
        if brief is None:
            report.add(name, False, "no brief produced")
            continue

        # Schema is guaranteed valid by construction (pydantic), assert non-empty.
        if not brief.summary.strip():
            report.add(name + " summary", False, "empty summary")
        else:
            report.add(name + " summary", True)

        # Language check (if declared). Normalise first: Scribe and the LLM may
        # report ISO-639-1 ("es") or -3 ("spa"), so compare canonical codes, not
        # a substring (which wrongly fails "spa" vs "es").
        expected_lang = spec.get("language")
        if expected_lang:
            got = (brief.language or result.transcript.language_code or "").lower()
            passed = normalize_lang(expected_lang) == normalize_lang(got)
            report.add(name + " language", passed, f"expected ~{expected_lang}, got {got}")

        # Keyword recall over the whole brief.
        haystack = brief.searchable_text().lower()
        for kw in spec.get("expect_keywords", []):
            report.add(name + f" keyword '{kw}'", kw.lower() in haystack)

        min_ai = spec.get("min_action_items")
        if min_ai is not None:
            report.add(
                name + " action items",
                len(brief.action_items) >= min_ai,
                f"got {len(brief.action_items)}, expected >= {min_ai}",
            )

    return report
