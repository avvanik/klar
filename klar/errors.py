"""Typed errors for Klar.

The pipeline is designed to *fail loudly*: bad input raises a specific,
catchable exception rather than silently producing an empty or wrong brief.
Every exception here carries enough context to end up in the run log.
"""

from __future__ import annotations


class KlarError(Exception):
    """Base class for all Klar errors."""


# --- Ingest ---------------------------------------------------------------
class IngestError(KlarError):
    """Something is wrong with the input before we ever call an API."""


class MissingFileError(IngestError):
    """The audio path does not exist."""


class EmptyAudioError(IngestError):
    """The audio file exists but is empty (0 bytes) or below the minimum size."""


class UnsupportedFormatError(IngestError):
    """The file extension is not an audio/video format Scribe accepts."""


# --- Transcribe -----------------------------------------------------------
class TranscriptionError(KlarError):
    """The speech-to-text step failed or returned an unusable transcript."""


# --- Extract --------------------------------------------------------------
class ExtractionError(KlarError):
    """The LLM extraction step failed."""


class SchemaValidationError(ExtractionError):
    """The LLM output did not validate against the Brief schema after retry."""


# --- Config ---------------------------------------------------------------
class ConfigError(KlarError):
    """Missing/invalid configuration (env vars, customer YAML, etc.)."""
