"""Ingest stage: validate the input and compute its content hash.

This runs *before* any paid API call, so it is the cheapest place to fail loudly
on garbage input. It rejects missing files, empty files, and unsupported formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .errors import EmptyAudioError, MissingFileError, UnsupportedFormatError
from .hashing import hash_file, short_hash

# Formats Scribe v2 accepts (audio + video), per the ElevenLabs docs.
SUPPORTED_EXTS = {
    # audio
    ".aac", ".aiff", ".aif", ".ogg", ".mp3", ".mpeg", ".mpga", ".opus",
    ".wav", ".webm", ".flac", ".m4a", ".mp4",
    # video
    ".avi", ".mkv", ".mov", ".wmv", ".flv", ".3gp", ".3gpp",
}

# ~100ms of audio is Scribe's stated minimum; anything smaller is certainly junk.
MIN_BYTES = 256

AUDIO_EXTS = tuple(sorted(SUPPORTED_EXTS))


@dataclass(frozen=True)
class Ingested:
    path: Path
    content_hash: str
    size_bytes: int

    @property
    def short(self) -> str:
        return short_hash(self.content_hash)


def ingest(path: str | Path) -> Ingested:
    """Validate a single audio file and return its hash + metadata.

    Raises MissingFileError / EmptyAudioError / UnsupportedFormatError.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise MissingFileError(f"No such audio file: {p}")

    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise UnsupportedFormatError(
            f"Unsupported format '{ext}' for {p.name}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTS))}"
        )

    size = p.stat().st_size
    if size < MIN_BYTES:
        raise EmptyAudioError(
            f"{p.name} is {size} bytes, below the {MIN_BYTES}-byte minimum "
            f"(empty or truncated recording)."
        )

    return Ingested(path=p, content_hash=hash_file(p), size_bytes=size)


def discover_audio(path: str | Path) -> list[Path]:
    """Expand a file or folder into a sorted list of candidate audio files."""
    p = Path(path)
    if p.is_dir():
        return sorted(
            f for f in p.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
        )
    return [p]
