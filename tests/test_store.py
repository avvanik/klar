from __future__ import annotations

from klar.observe import RunRecord, utcnow_iso


def _record(run_id, content_hash, version, status="ok"):
    return RunRecord(
        run_id=run_id, created_at=utcnow_iso(), input_path="samples/x.mp3",
        content_hash=content_hash, version=version, customer="default", status=status,
        language_code="eng", stt_model="scribe_v2", llm_model="fake",
        latency_ms={"ingest": 1.0, "transcribe": 2.0}, total_latency_ms=3.0,
    )


def test_register_input_versions(store):
    h = "abc123"
    assert store.register_input(h, "samples/x.mp3", utcnow_iso()) == 1
    assert store.register_input(h, "samples/x.mp3", utcnow_iso()) == 2
    row = store.get_input(h)
    assert row["latest_version"] == 2


def test_insert_and_list_runs_roundtrip(store):
    store.insert_run(_record("run1", "hash1", 1))
    rows = store.list_runs()
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run1"
    assert rows[0]["latency_ms"] == {"ingest": 1.0, "transcribe": 2.0}


def test_latest_successful_run_ignores_failed(store):
    store.insert_run(_record("run_ok", "h", 1, status="ok"))
    store.insert_run(_record("run_fail", "h", 2, status="failed"))
    latest = store.latest_successful_run("h")
    assert latest["run_id"] == "run_ok"


def test_write_json_artifact(store):
    from klar.models import Brief
    b = Brief(language="eng", summary="hi")
    p = store.write_json("deadbeefcafe", 1, "brief.json", b)
    assert p.exists()
    assert "summary" in p.read_text()
