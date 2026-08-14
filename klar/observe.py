"""Observability primitives: stage timing, cost estimates, run records.

Every run produces one RunRecord. It captures the input hash, detected language,
model versions, per-stage latency, a token/character cost estimate, and pass/fail
- everything you need to debug a bad brief or explain a bill.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Iterator


def new_run_id() -> str:
    return uuid.uuid4().hex[:16]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class StageTimer:
    """Accumulates per-stage wall-clock latency in milliseconds."""

    latencies_ms: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.latencies_ms[name] = round((time.perf_counter() - start) * 1000, 1)

    @property
    def total_ms(self) -> float:
        return round(sum(self.latencies_ms.values()), 1)


def estimate_stt_cost(duration_seconds: float | None, usd_per_hour: float) -> float:
    if not duration_seconds:
        return 0.0
    return round((duration_seconds / 3600.0) * usd_per_hour, 6)


def estimate_llm_cost(
    input_tokens: int, output_tokens: int, usd_per_mtok_in: float, usd_per_mtok_out: float
) -> float:
    cost = (input_tokens / 1_000_000.0) * usd_per_mtok_in
    cost += (output_tokens / 1_000_000.0) * usd_per_mtok_out
    return round(cost, 6)


@dataclass
class RunRecord:
    run_id: str
    created_at: str
    input_path: str
    content_hash: str
    version: int
    customer: str
    status: str  # "ok" | "failed" | "skipped"
    language_code: str | None = None
    stt_model: str | None = None
    llm_model: str | None = None
    duration_seconds: float | None = None
    latency_ms: dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    stt_cost_usd: float = 0.0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_chunks: int = 1  # >1 when a long transcript was map-reduced
    llm_cost_usd: float = 0.0
    transcript_path: str | None = None
    brief_path: str | None = None
    error: str | None = None

    def to_row(self) -> dict:
        return asdict(self)
