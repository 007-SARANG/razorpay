"""Runtime configuration, loaded from environment / ``.env``.

Two principles here, both of which exist to keep the submission reproducible:

* **No secret ever has a default.** ``api_key`` defaults to ``None``, and every code
  path checks :meth:`Settings.llm_enabled` before attempting a call. There is no
  hardcoded key anywhere in this repository.
* **Absence of a key is a supported mode, not an error.** With no key configured the
  pipeline runs deterministically end to end and escalates ambiguous candidates to the
  review queue. Every metric in the evaluation report is still produced.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Environment-driven settings. All LLM fields are optional by design."""

    model_config = SettingsConfigDict(
        env_prefix="TRIKON_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Provider -------------------------------------------------------------------
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str | None = None
    llm_model: str = "glm-5.2"
    llm_fallback_models: str = ""

    # --- Budget guards --------------------------------------------------------------
    llm_max_calls_per_run: int = Field(default=25, ge=0)
    llm_requests_per_minute: int = Field(default=15, ge=1)
    llm_timeout_seconds: float = Field(default=90.0, gt=0)
    llm_adjudication_batch_size: int = Field(default=5, ge=1, le=25)

    # --- Cache ----------------------------------------------------------------------
    llm_cache_dir: Path = Path("data/cache")
    llm_cache_enabled: bool = True

    # --- Paths ----------------------------------------------------------------------
    batch_dir: Path = Path("data/batches")
    report_dir: Path = Path("data/reports")

    @field_validator("llm_api_key")
    @classmethod
    def _blank_key_is_none(cls, value: str | None) -> str | None:
        """Treat an empty or placeholder key as absent.

        ``.env`` files routinely carry ``TRIKON_LLM_API_KEY=`` with nothing after it;
        that must mean "no key", not "key is the empty string", or we would attempt
        doomed HTTP calls and report them as model failures.
        """
        if value is None:
            return None
        stripped = value.strip()
        if not stripped or stripped.lower() in {"none", "null", "changeme", "your-key-here"}:
            return None
        return stripped

    @property
    def llm_enabled(self) -> bool:
        """True only if a usable key is present and a call budget remains."""
        return self.llm_api_key is not None and self.llm_max_calls_per_run > 0

    @property
    def fallback_models(self) -> tuple[str, ...]:
        """Parsed fallback model chain, primary excluded."""
        raw = (m.strip() for m in self.llm_fallback_models.split(","))
        return tuple(m for m in raw if m and m != self.llm_model)

    def redacted(self) -> dict[str, object]:
        """Settings safe to print or log -- the key is reduced to a fingerprint.

        Used by ``trikon doctor`` so a user can confirm *which* key is loaded without
        the value ever reaching a terminal, a log file, or a demo recording.
        """
        key = self.llm_api_key
        fingerprint = "not set"
        if key:
            fingerprint = f"{key[:6]}...{key[-4:]} (len {len(key)})" if len(key) > 12 else "set"
        return {
            "llm_base_url": self.llm_base_url,
            "llm_api_key": fingerprint,
            "llm_model": self.llm_model,
            "llm_fallback_models": list(self.fallback_models),
            "llm_enabled": self.llm_enabled,
            "llm_max_calls_per_run": self.llm_max_calls_per_run,
            "llm_cache_enabled": self.llm_cache_enabled,
            "llm_cache_dir": str(self.llm_cache_dir),
        }

    def resolve(self, path: Path) -> Path:
        """Resolve a possibly-relative configured path against the repository root."""
        return path if path.is_absolute() else REPO_ROOT / path


def load_settings() -> Settings:
    """Load settings from the environment.

    Deliberately a function rather than a module-level singleton so that tests can
    construct isolated ``Settings`` instances without monkeypatching global state.
    """
    return Settings()
