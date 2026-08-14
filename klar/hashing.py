"""Content hashing for idempotency.

The hash is computed over the *bytes of the audio file*, not its name or path,
so the same recording dropped in twice (even renamed) is recognised as already
processed. This is the anchor of the idempotency guarantee.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 1 << 20  # 1 MiB


def hash_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file's contents (streamed)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def short_hash(digest: str) -> str:
    """A short, human-friendly prefix for logs and directory names."""
    return digest[:12]
