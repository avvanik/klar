"""Storage: SQLite for run records + local files for transcripts and briefs.

Idempotency lives here. The `inputs` table tracks every content hash we have
seen and its current version. `latest_successful_run` lets the pipeline skip
re-processing an unchanged file instead of duplicating state.

No ORM on purpose - one small stdlib sqlite3 module is easier to audit and has
zero deploy surface.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .observe import RunRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS inputs (
    content_hash   TEXT PRIMARY KEY,
    first_path     TEXT NOT NULL,
    first_seen_at  TEXT NOT NULL,
    latest_version INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    input_path        TEXT NOT NULL,
    content_hash      TEXT NOT NULL,
    version           INTEGER NOT NULL,
    customer          TEXT NOT NULL,
    status            TEXT NOT NULL,
    language_code     TEXT,
    stt_model         TEXT,
    llm_model         TEXT,
    duration_seconds  REAL,
    latency_ms        TEXT,
    total_latency_ms  REAL,
    stt_cost_usd      REAL,
    llm_input_tokens  INTEGER,
    llm_output_tokens INTEGER,
    llm_chunks        INTEGER,
    llm_cost_usd      REAL,
    transcript_path   TEXT,
    brief_path        TEXT,
    error             TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_hash ON runs(content_hash);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);
"""


class Store:
    def __init__(self, db_path: str | Path, artifacts_dir: str | Path):
        self.db_path = Path(db_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # --- idempotency ------------------------------------------------------
    def get_input(self, content_hash: str) -> sqlite3.Row | None:
        cur = self._conn.execute(
            "SELECT * FROM inputs WHERE content_hash = ?", (content_hash,)
        )
        return cur.fetchone()

    def register_input(self, content_hash: str, path: str, seen_at: str) -> int:
        """Insert the input if new, else bump its version. Returns the version."""
        row = self.get_input(content_hash)
        if row is None:
            self._conn.execute(
                "INSERT INTO inputs(content_hash, first_path, first_seen_at, latest_version) "
                "VALUES (?, ?, ?, 1)",
                (content_hash, path, seen_at),
            )
            self._conn.commit()
            return 1
        version = int(row["latest_version"]) + 1
        self._conn.execute(
            "UPDATE inputs SET latest_version = ? WHERE content_hash = ?",
            (version, content_hash),
        )
        self._conn.commit()
        return version

    def latest_successful_run(self, content_hash: str) -> sqlite3.Row | None:
        cur = self._conn.execute(
            "SELECT * FROM runs WHERE content_hash = ? AND status = 'ok' "
            "ORDER BY version DESC, created_at DESC LIMIT 1",
            (content_hash,),
        )
        return cur.fetchone()

    # --- runs -------------------------------------------------------------
    def insert_run(self, record: RunRecord) -> None:
        row = record.to_row()
        row["latency_ms"] = json.dumps(row.get("latency_ms") or {})
        cols = ", ".join(row.keys())
        placeholders = ", ".join(["?"] * len(row))
        self._conn.execute(
            f"INSERT INTO runs ({cols}) VALUES ({placeholders})", tuple(row.values())
        )
        self._conn.commit()

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC, run_id DESC LIMIT ?", (limit,)
        )
        out: list[dict[str, Any]] = []
        for row in cur.fetchall():
            d = dict(row)
            try:
                d["latency_ms"] = json.loads(d.get("latency_ms") or "{}")
            except json.JSONDecodeError:
                d["latency_ms"] = {}
            out.append(d)
        return out

    # --- artifacts (transcript + brief on disk) ---------------------------
    def artifact_dir(self, content_hash: str, version: int) -> Path:
        d = self.artifacts_dir / content_hash[:12] / f"v{version}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_json(self, content_hash: str, version: int, name: str, payload: Any) -> Path:
        d = self.artifact_dir(content_hash, version)
        path = d / name
        text = payload.model_dump_json(indent=2) if hasattr(payload, "model_dump_json") else json.dumps(payload, indent=2, ensure_ascii=False)
        path.write_text(text, encoding="utf-8")
        return path

    def write_bytes(self, content_hash: str, version: int, name: str, data: bytes) -> Path:
        path = self.artifact_dir(content_hash, version) / name
        path.write_bytes(data)
        return path
