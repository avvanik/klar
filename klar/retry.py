"""Transient-failure retry with exponential backoff.

Both ElevenLabs and Anthropic rate-limit and occasionally return 429/500/503/529
or drop the connection. Without retries a single transient blip kills a file in a
batch run - the most common real-world failure. This wraps a call with bounded
exponential backoff, and deliberately does NOT retry on client errors (400/401/
404) so a bad key or an unknown model fails fast instead of burning four attempts.

Provider-agnostic on purpose: it classifies by HTTP status code (read off the
exception when present) and by exception family, so it works for either SDK and
for the fakes in tests without importing any SDK.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable, TypeVar

T = TypeVar("T")

# Statuses worth retrying: rate limit, overload, and transient server errors.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

# Substrings we look for in an exception's class name when no status is exposed.
_RETRYABLE_NAME_TOKENS = (
    "timeout",
    "overload",
    "ratelimit",
    "rate_limit",
    "serviceunavailable",
    "connection",
    "apiconnection",
    "internalserver",
)


def _status_of(exc: BaseException) -> int | None:
    for attr in ("status_code", "status", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    return None


def is_retryable(exc: BaseException) -> bool:
    """True if `exc` looks like a transient failure worth retrying."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    status = _status_of(exc)
    if status is not None:
        return status in RETRYABLE_STATUS
    name = type(exc).__name__.lower()
    return any(tok in name for tok in _RETRYABLE_NAME_TOKENS)


def retry_call(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retryable: Callable[[BaseException], bool] = is_retryable,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    _multiplier: Iterable[float] | None = None,
) -> T:
    """Call `fn`, retrying transient failures with exponential backoff.

    `attempts` is the total number of tries (so 3 => original + 2 retries). A
    non-retryable exception, or the final attempt, re-raises immediately.
    """
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except BaseException as exc:  # noqa: BLE001 - we re-raise unless retryable
            last = exc
            if attempt >= attempts or not retryable(exc):
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            sleep(delay)
    assert last is not None  # unreachable; keeps type-checkers happy
    raise last
