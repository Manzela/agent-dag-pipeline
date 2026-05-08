"""
Pipeline Configuration — Typed, Validated Settings.

Loads pipeline configuration from JSON files, environment variables,
or programmatic defaults. All settings are validated via Pydantic
for type safety and documented constraints.

Loading Priority:
    1. Explicit ``PipelineConfig(...)`` constructor arguments
    2. JSON config file (``--config path/to/config.json``)
    3. Environment variables (``PIPELINE_*`` prefix)
    4. Built-in defaults

Usage::

    from agent_dag.config import PipelineConfig, load_config

    # From defaults (zero config)
    config = PipelineConfig()

    # From JSON file
    config = load_config("examples/config.json")

    # From environment
    import os
    os.environ["PIPELINE_LLM_PROVIDER"] = "openai"
    os.environ["PIPELINE_LLM_MODEL"] = "gpt-4o-mini"
    config = load_config()
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class ORAVThresholds(BaseModel):
    """O-R-A-V quality gate thresholds."""
    model_config = ConfigDict(frozen=True)

    originality: float = Field(default=0.6, ge=0.0, le=1.0)
    relevance: float = Field(default=0.7, ge=0.0, le=1.0)
    accuracy: float = Field(default=0.8, ge=0.0, le=1.0)
    value: float = Field(default=0.6, ge=0.0, le=1.0)


class FlywheelConfig(BaseModel):
    """Data flywheel tier routing thresholds."""
    model_config = ConfigDict(frozen=True)

    approved_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    failure_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_pairs_for_training: int = Field(default=50, ge=1)
    min_margin_for_training: float = Field(default=0.2, ge=0.0, le=1.0)


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    model_config = ConfigDict(frozen=True)

    provider: str = Field(default="mock", description="LLM provider: mock, openai, google")
    model: str = Field(default="", description="Model identifier (provider-specific)")
    api_key: str = Field(default="", description="API key (or use env var)")
    api_key_env: str = Field(default="LLM_API_KEY", description="Env var name for API key")
    base_url: str = Field(default="", description="Custom API base URL (OpenAI-compatible)")

    def resolve_api_key(self) -> str:
        """Resolve API key from explicit value or environment variable."""
        if self.api_key:
            return self.api_key
        return os.environ.get(self.api_key_env, "")


class PipelineConfig(BaseModel):
    """Complete pipeline configuration.

    All pipeline behavior is controlled through this single configuration
    object. Defaults produce a fully functional pipeline using the mock
    LLM client with no external dependencies.
    """
    model_config = ConfigDict(frozen=True)

    # ── Locale ──
    supported_locales: list[str] = Field(
        default=["ES", "PT", "IL", "GB", "DE", "FR", "IT", "PL", "CZ"],
        description="ISO 3166-1 alpha-2 codes for supported locales",
    )

    # ── Demand gate ──
    min_search_volume: int = Field(default=10, ge=0)
    max_monthly_quota: int = Field(default=5000, ge=0)

    # ── Quality gate ──
    orav: ORAVThresholds = Field(default_factory=ORAVThresholds)

    # ── Flywheel ──
    flywheel: FlywheelConfig = Field(default_factory=FlywheelConfig)

    # ── LLM ──
    llm: LLMConfig = Field(default_factory=LLMConfig)

    # ── Output ──
    output_dir: str = Field(default="./output", description="Directory for output files")


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """Load pipeline configuration from file and/or environment.

    Parameters
    ----------
    config_path : str, optional
        Path to a JSON configuration file. If None, uses defaults
        with environment variable overrides.

    Returns
    -------
    PipelineConfig
        Validated, frozen configuration object.
    """
    data: dict[str, Any] = {}

    # ── Load from JSON file if provided ──
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            logger.info("Loaded config from %s", path)
        else:
            logger.warning("Config file not found: %s, using defaults", path)

    # ── Override from environment variables ──
    env_overrides: dict[str, Any] = {}
    if os.environ.get("PIPELINE_LLM_PROVIDER"):
        env_overrides.setdefault("llm", {})["provider"] = os.environ["PIPELINE_LLM_PROVIDER"]
    if os.environ.get("PIPELINE_LLM_MODEL"):
        env_overrides.setdefault("llm", {})["model"] = os.environ["PIPELINE_LLM_MODEL"]
    if os.environ.get("PIPELINE_OUTPUT_DIR"):
        env_overrides["output_dir"] = os.environ["PIPELINE_OUTPUT_DIR"]
    if os.environ.get("PIPELINE_MIN_SEARCH_VOLUME"):
        env_overrides["min_search_volume"] = int(os.environ["PIPELINE_MIN_SEARCH_VOLUME"])

    # Merge: file < env
    for key, val in env_overrides.items():
        if isinstance(val, dict) and key in data and isinstance(data[key], dict):
            data[key].update(val)
        else:
            data[key] = val

    return PipelineConfig(**data)
