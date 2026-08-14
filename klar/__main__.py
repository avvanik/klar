"""Klar command-line interface.

    python -m klar run <path> [--customer NAME] [--force]
    python -m klar runs [--limit N]
    python -m klar eval [--samples DIR] [--negative-only]

Real ElevenLabs + Anthropic clients are constructed from environment variables.
The pipeline itself is provider-agnostic; this module is the only place that
wires in the concrete SDKs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Settings
from .customers import load_customer
from .errors import KlarError
from .ingest import discover_audio
from .observe import RunRecord
from .store import Store


def _build_clients(settings: Settings):
    from .extract import make_llm_client
    from .transcribe import ElevenLabsTranscriber

    transcriber = ElevenLabsTranscriber(
        settings.require_elevenlabs(),
        model_id=settings.stt_model,
        timeout_s=settings.request_timeout_s,
        max_attempts=settings.max_attempts,
    )
    llm = make_llm_client(settings)
    return transcriber, llm


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import run_pipeline

    settings = Settings.from_env()
    customer = load_customer(args.customer)
    store = Store(settings.db_path, settings.artifacts_dir)
    try:
        try:
            transcriber, llm = _build_clients(settings)
        except KlarError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

        files = discover_audio(args.path)
        if not files:
            print(f"No audio files found at {args.path}", file=sys.stderr)
            return 2

        exit_code = 0
        for audio in files:
            print(f"\n=== {audio.name} ===")
            try:
                result = run_pipeline(
                    audio, store=store, transcriber=transcriber, llm=llm,
                    customer=customer, force=args.force,
                    max_output_tokens=settings.llm_max_tokens,
                    max_transcript_chars=settings.max_transcript_chars,
                )
            except KlarError as e:
                print(f"  FAILED: {e}", file=sys.stderr)
                exit_code = 1
                continue

            if result.status == "skipped":
                print(f"  already processed (run {result.run_id}, v{result.version}) - skipped. "
                      f"Use --force to re-process as a new version.")
                print(f"  brief: {result.record.brief_path}")
                continue

            _print_brief(result)
            if args.speak:
                _speak_summary(result, store, settings)
            if args.save_fixtures:
                _save_fixtures(result, args.save_fixtures, audio)
        return exit_code
    finally:
        store.close()


def _save_fixtures(result, dest: str, audio) -> None:
    """Persist a real transcript + brief as committable golden fixtures.

    Run this once with real keys so a reviewer *without* keys sees genuine output
    (and tests/test_fixtures.py validates it). See fixtures/README.md.
    """
    d = Path(dest) / Path(audio).stem
    d.mkdir(parents=True, exist_ok=True)
    (d / "brief.json").write_text(result.brief.model_dump_json(indent=2), encoding="utf-8")
    (d / "transcript.txt").write_text(result.transcript.diarized_text(), encoding="utf-8")
    print(f"  saved fixtures -> {d}")


def _speak_summary(result, store, settings) -> None:
    """Synthesize a spoken recap of the brief and save it beside the brief."""
    from .synth import ElevenLabsSpeaker, spoken_summary

    try:
        speaker = ElevenLabsSpeaker(
            settings.require_elevenlabs(), voice_id=settings.tts_voice_id
        )
        audio = speaker.synthesize(spoken_summary(result.brief))
        path = store.write_bytes(result.content_hash, result.version, "summary.mp3", audio)
    except Exception as e:  # never let TTS failure fail the run
        print(f"  (spoken summary unavailable: {e})", file=sys.stderr)
        return
    print(f"\n  ▶ Spoken summary: {path}  (play it: afplay {path})")


def _print_brief(result) -> None:
    b = result.brief
    r = result.record
    print(f"  run {result.run_id}  v{result.version}  lang={b.language}  "
          f"({r.total_latency_ms:.0f} ms, ~${r.stt_cost_usd + r.llm_cost_usd:.4f})")
    print(f"\n  Summary:\n    {b.summary}")
    if b.action_items:
        print("\n  Action items:")
        for a in b.action_items:
            print(f"    - [{a.owner}] {a.text}")
    if b.decisions:
        print("\n  Decisions:")
        for d in b.decisions:
            print(f"    - {d}")
    if b.open_questions:
        print("\n  Open questions:")
        for q in b.open_questions:
            print(f"    - {q}")
    if b.sentiment_by_speaker:
        print("\n  Sentiment:")
        for s in b.sentiment_by_speaker:
            print(f"    - {s.speaker}: {s.sentiment} ({s.rationale})")
    print(f"\n  Saved: {r.brief_path}")


def cmd_runs(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    store = Store(settings.db_path, settings.artifacts_dir)
    rows = store.list_runs(limit=args.limit)
    store.close()
    if not rows:
        print("No runs yet. Try: python -m klar run samples/<file>")
        return 0
    print(_render_runs_table(rows))
    return 0


def _render_runs_table(rows: list[dict]) -> str:
    headers = ["created_at", "run_id", "hash", "v", "customer", "lang", "status", "ms", "cost$"]
    table = []
    for r in rows:
        cost = (r.get("stt_cost_usd") or 0) + (r.get("llm_cost_usd") or 0)
        table.append([
            (r.get("created_at") or "")[:19],
            r.get("run_id") or "",
            (r.get("content_hash") or "")[:8],
            str(r.get("version") or ""),
            r.get("customer") or "",
            r.get("language_code") or "-",
            r.get("status") or "",
            f"{(r.get('total_latency_ms') or 0):.0f}",
            f"{cost:.4f}",
        ])
    widths = [len(h) for h in headers]
    for row in table:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    def fmt(cells):
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))
    out = [fmt(headers), "  ".join("-" * w for w in widths)]
    out += [fmt(r) for r in table]
    return "\n".join(out)


def cmd_eval(args: argparse.Namespace) -> int:
    from .evaluate import EvalReport, run_negative_checks, run_positive_checks

    report = EvalReport()
    run_negative_checks(report)

    if not args.negative_only:
        settings = Settings.from_env()
        try:
            transcriber, llm = _build_clients(settings)
        except KlarError as e:
            print(f"note: skipping positive checks - {e}", file=sys.stderr)
            print("(run with --negative-only to suppress this, or set your API keys)\n")
        else:
            store = Store(settings.db_path, settings.artifacts_dir)
            run_positive_checks(
                args.samples, store=store, transcriber=transcriber, llm=llm, report=report
            )
            store.close()

    print(report.render())
    return 0 if report.ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="klar", description="Call intelligence pipeline (ElevenLabs Scribe v2 + LLM).")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="Transcribe + extract a brief for a file or folder.")
    pr.add_argument("path", help="Audio file or folder of audio files.")
    pr.add_argument("--customer", help="Customer config name (customers/<name>.yaml) or path.")
    pr.add_argument("--force", action="store_true", help="Re-process even if unchanged (new version).")
    pr.add_argument("--speak", action="store_true", help="Also synthesize a spoken summary (ElevenLabs TTS).")
    pr.add_argument("--save-fixtures", nargs="?", const="fixtures", metavar="DIR",
                    help="Save the real transcript + brief as golden fixtures (default dir: fixtures/).")
    pr.set_defaults(func=cmd_run)

    pl = sub.add_parser("runs", help="Print the run log as a table.")
    pl.add_argument("--limit", type=int, default=50)
    pl.set_defaults(func=cmd_runs)

    pe = sub.add_parser("eval", help="Run the eval harness (exits non-zero on failure).")
    pe.add_argument("--samples", default="samples", help="Samples directory.")
    pe.add_argument("--negative-only", action="store_true", help="Only run offline robustness checks.")
    pe.set_defaults(func=cmd_eval)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KlarError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
