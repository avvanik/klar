"""Per-customer configuration.

This is the Forward-Deployed signal: the *same* pipeline behaves differently for
each customer, driven by a small YAML file. The config feeds two stages:

  transcribe: language hint, keyterms to boost, speaker count, role detection,
              verbatim cleanup.
  extract:    which brief fields to emphasise, a glossary, and any extra
              instructions injected into the LLM prompt.

Switching the file visibly changes the STT request and the brief.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .errors import ConfigError

VALID_FIELDS = {
    "summary",
    "action_items",
    "decisions",
    "open_questions",
    "sentiment_by_speaker",
}


class CustomerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "default"
    # First hint is used as the STT language_code; None/empty => auto-detect.
    language_hints: list[str] = Field(default_factory=list)
    # Terms to bias Scribe towards (product names, jargon). Max 1000, <=50 chars.
    keyterms: list[str] = Field(default_factory=list)
    num_speakers: int | None = None
    detect_speaker_roles: bool = False
    no_verbatim: bool = False
    # Brief fields to emphasise in extraction (must be a subset of VALID_FIELDS).
    emphasis: list[str] = Field(default_factory=list)
    # term -> definition, injected so the LLM interprets domain language right.
    glossary: dict[str, str] = Field(default_factory=dict)
    extraction_instructions: str = ""

    def primary_language(self) -> str | None:
        return self.language_hints[0] if self.language_hints else None

    def stt_kwargs(self) -> dict[str, Any]:
        """Kwargs merged into ElevenLabs speech_to_text.convert()."""
        kwargs: dict[str, Any] = {}
        lang = self.primary_language()
        if lang:
            kwargs["language_code"] = lang
        if self.keyterms:
            kwargs["keyterms"] = self.keyterms[:1000]
        if self.num_speakers:
            kwargs["num_speakers"] = self.num_speakers
        if self.detect_speaker_roles:
            kwargs["detect_speaker_roles"] = True
        if self.no_verbatim:
            kwargs["no_verbatim"] = True
        return kwargs


def default_config() -> CustomerConfig:
    return CustomerConfig(name="default")


def load_customer(name_or_path: str | None, *, customers_dir: str | Path = "customers") -> CustomerConfig:
    """Load a customer config by name (customers/<name>.yaml) or explicit path.

    Returns the default config when name_or_path is falsy.
    """
    if not name_or_path:
        return default_config()

    p = Path(name_or_path)
    if p.suffix in {".yaml", ".yml"} and p.exists():
        path = p
    else:
        path = Path(customers_dir) / f"{name_or_path}.yaml"

    if not path.exists():
        raise ConfigError(f"Customer config not found: {path}")

    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Customer config {path} must be a YAML mapping.")
    data.setdefault("name", path.stem)

    try:
        cfg = CustomerConfig(**data)
    except Exception as exc:  # pydantic ValidationError -> ConfigError
        raise ConfigError(f"Invalid customer config {path}: {exc}") from exc

    bad = set(cfg.emphasis) - VALID_FIELDS
    if bad:
        raise ConfigError(
            f"Customer config {path} emphasises unknown field(s): {sorted(bad)}. "
            f"Valid fields: {sorted(VALID_FIELDS)}"
        )
    return cfg
