"""Runtime configuration, loaded from environment variables only.

Secrets never live in code. `python-dotenv` loads a local .env if present, but
that file is git-ignored. Everything here is a plain dataclass so it is trivial
to construct in tests without touching the real environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigError

try:  # dotenv is optional at import time; real runs will have it installed.
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - only hit if dependency missing

    def load_dotenv(*_a, **_k):  # type: ignore
        return False


@dataclass(frozen=True)
class Settings:
    """All tunables for a run. Construct via `Settings.from_env()`."""

    elevenlabs_api_key: str | None
    anthropic_api_key: str | None
    llm_provider: str
    llm_model: str
    stt_model: str
    tts_voice_id: str | None  # optional voice for `run --speak`; auto-picks if None
    home: Path
    # Reliability / scaling knobs.
    llm_max_tokens: int
    max_transcript_chars: int
    request_timeout_s: float
    max_attempts: int
    allow_unknown_model: bool

    @property
    def db_path(self) -> Path:
        return self.home / "klar.sqlite"

    @property
    def artifacts_dir(self) -> Path:
        return self.home / "artifacts"

    @classmethod
    def from_env(cls, *, load_dotenv_file: bool = True) -> "Settings":
        if load_dotenv_file:
            load_dotenv()
        home = Path(os.getenv("KLAR_HOME", ".klar")).expanduser()
        return cls(
            elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            llm_provider=os.getenv("KLAR_LLM_PROVIDER", "anthropic").lower(),
            llm_model=os.getenv("KLAR_LLM_MODEL", "claude-sonnet-4-6"),
            stt_model=os.getenv("KLAR_STT_MODEL", "scribe_v2"),
            tts_voice_id=os.getenv("KLAR_TTS_VOICE_ID") or None,
            home=home,
            llm_max_tokens=_int_env("KLAR_LLM_MAX_TOKENS", 4096),
            max_transcript_chars=_int_env("KLAR_MAX_TRANSCRIPT_CHARS", 24000),
            request_timeout_s=_float_env("KLAR_REQUEST_TIMEOUT_S", 120.0),
            max_attempts=_int_env("KLAR_MAX_ATTEMPTS", 3),
            allow_unknown_model=_bool_env("KLAR_ALLOW_UNKNOWN_MODEL", False),
        )

    def require_elevenlabs(self) -> str:
        if not self.elevenlabs_api_key:
            raise ConfigError(
                "ELEVENLABS_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        return self.elevenlabs_api_key

    def require_anthropic(self) -> str:
        if not self.anthropic_api_key:
            raise ConfigError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        return self.anthropic_api_key


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
