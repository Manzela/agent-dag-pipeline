"""
Score Configurations — Multi-Dimensional Evaluation Metrics Registry.

Defines 15+ typed evaluation metrics used by the O-R-A-V evaluation framework,
DEMAS JIT auditor, and the data flywheel for dataset curation decisions.

Metric Types:
    NUMERIC (0.0–1.0): Continuous quality scores with configurable thresholds
    BOOLEAN (pass/fail): Binary compliance gates
    CATEGORICAL (PASS/RETRY/FAIL): Multi-class decision outcomes

Architecture Notes:
    - All configs are registered at startup via ensure_score_configs()
    - Anti-thundering-herd: Redis SET NX leader election at 100+ container scale
    - Each metric has a unique config_id for Langfuse correlation
    - Metrics are partitioned by node for granular performance analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class ScoreDataType(StrEnum):
    """Data type for a score configuration."""
    NUMERIC = "NUMERIC"
    BOOLEAN = "BOOLEAN"
    CATEGORICAL = "CATEGORICAL"


@dataclass(frozen=True)
class ScoreConfig:
    """Type-safe score configuration for a single evaluation metric.

    Each config defines the schema for one metric: its name, data type,
    valid range, threshold, and metadata for observability correlation.

    Attributes
    ----------
    config_id : str
        Unique identifier for this metric (e.g., "orav_quality").
    name : str
        Human-readable metric name.
    data_type : ScoreDataType
        NUMERIC, BOOLEAN, or CATEGORICAL.
    description : str
        What this metric measures and why it matters.
    min_value : float
        Minimum valid value (NUMERIC only).
    max_value : float
        Maximum valid value (NUMERIC only).
    threshold : float
        Minimum acceptable value for quality gate decisions.
    categories : tuple
        Valid categories (CATEGORICAL only).
    node : str
        The pipeline node that produces this metric.
    """
    config_id: str
    name: str
    data_type: ScoreDataType
    description: str
    min_value: float = 0.0
    max_value: float = 1.0
    threshold: float = 0.7
    categories: tuple[str, ...] = ()
    node: str = ""


# ════════════════════════════════════════════════════════════════════════════
# METRIC REGISTRY — All 15+ evaluation metrics
# ════════════════════════════════════════════════════════════════════════════

SCORE_CONFIGS: dict[str, ScoreConfig] = {
    # ── O-R-A-V Quality Dimensions ──
    "orav_quality": ScoreConfig(
        config_id="orav_quality",
        name="O-R-A-V Composite Quality",
        data_type=ScoreDataType.NUMERIC,
        description=(
            "Composite quality score across all 4 O-R-A-V dimensions. "
            "Weighted average of Originality, Relevance, Accuracy, and Value."
        ),
        threshold=0.7,
        node="6",
    ),
    "orav_decision": ScoreConfig(
        config_id="orav_decision",
        name="O-R-A-V Decision",
        data_type=ScoreDataType.CATEGORICAL,
        description="Final O-R-A-V gate decision: PASS, RETRY, or FAIL.",
        categories=("PASS", "RETRY", "FAIL"),
        node="6",
    ),
    "orav_originality": ScoreConfig(
        config_id="orav_originality",
        name="Originality Score",
        data_type=ScoreDataType.NUMERIC,
        description="Measures content uniqueness vs. template repetition and existing corpus overlap.",
        threshold=0.6,
        node="6",
    ),
    "orav_relevance": ScoreConfig(
        config_id="orav_relevance",
        name="Relevance Score",
        data_type=ScoreDataType.NUMERIC,
        description="Measures alignment between generated content and product/locale context.",
        threshold=0.7,
        node="6",
    ),
    "orav_accuracy": ScoreConfig(
        config_id="orav_accuracy",
        name="Accuracy Score",
        data_type=ScoreDataType.NUMERIC,
        description="Factual correctness of product attributes, pricing, and specifications.",
        threshold=0.8,
        node="6",
    ),
    "orav_value": ScoreConfig(
        config_id="orav_value",
        name="Value Score",
        data_type=ScoreDataType.NUMERIC,
        description="Cross-model consensus on content's commercial and informational value.",
        threshold=0.6,
        node="6",
    ),

    # ── Node 2: Normalizer Metrics ──
    "n2_taxonomy_confidence": ScoreConfig(
        config_id="n2_taxonomy_confidence",
        name="Taxonomy Confidence",
        data_type=ScoreDataType.NUMERIC,
        description="Confidence of semantic category classification by the Normalizer.",
        threshold=0.8,
        node="2",
    ),
    "n2_schema_valid": ScoreConfig(
        config_id="n2_schema_valid",
        name="Schema Validation",
        data_type=ScoreDataType.BOOLEAN,
        description="Whether the raw input passed Pydantic schema validation.",
        node="2",
    ),

    # ── Node 5: Content Generator Metrics ──
    "n5_generation_success": ScoreConfig(
        config_id="n5_generation_success",
        name="Generation Success",
        data_type=ScoreDataType.BOOLEAN,
        description="Whether content generation completed without LLM errors.",
        node="5",
    ),
    "n5_block_quality": ScoreConfig(
        config_id="n5_block_quality",
        name="Per-Block Quality",
        data_type=ScoreDataType.NUMERIC,
        description="Individual content block quality from DEMAS JIT audit.",
        threshold=0.65,
        node="5",
    ),
    "n5_trigger_term_inclusion": ScoreConfig(
        config_id="n5_trigger_term_inclusion",
        name="Trigger Term Inclusion",
        data_type=ScoreDataType.BOOLEAN,
        description="Whether the primary trigger term appears in the generated content.",
        node="5",
    ),

    # ── Node 6: Validator Metrics ──
    "n6_label_leak_free": ScoreConfig(
        config_id="n6_label_leak_free",
        name="Label Leak Free",
        data_type=ScoreDataType.BOOLEAN,
        description="No internal pipeline labels leaked into generated content.",
        node="6",
    ),
    "n6_transactional_free": ScoreConfig(
        config_id="n6_transactional_free",
        name="Transactional Language Free",
        data_type=ScoreDataType.BOOLEAN,
        description="No transactional/sales language in content.",
        node="6",
    ),
    "n6_template_var_clean": ScoreConfig(
        config_id="n6_template_var_clean",
        name="Template Variable Clean",
        data_type=ScoreDataType.BOOLEAN,
        description="No unreplaced template variables ({var}) in content.",
        node="6",
    ),

    # ── DEMAS Metrics ──
    "demas_jit_verdict": ScoreConfig(
        config_id="demas_jit_verdict",
        name="DEMAS JIT Verdict",
        data_type=ScoreDataType.CATEGORICAL,
        description="Per-block JIT audit verdict from DEMAS evaluator.",
        categories=("PASS", "RETRY", "FAIL"),
        node="5",
    ),
    "demas_provenance_coverage": ScoreConfig(
        config_id="demas_provenance_coverage",
        name="Provenance Coverage",
        data_type=ScoreDataType.NUMERIC,
        description="Percentage of ground-truth variables covered by generated content.",
        threshold=0.8,
        node="6",
    ),
}


def get_config_id(metric_name: str) -> str:
    """Get the config ID for a named metric."""
    config = SCORE_CONFIGS.get(metric_name)
    if config is None:
        logger.warning("Unknown metric: %s", metric_name)
        return metric_name
    return config.config_id


def ensure_score_configs(
    *,
    observability_client: Any | None = None,
    leader_election: Any | None = None,
) -> int:
    """Register all score configs with the observability backend.

    Anti-thundering-herd: Uses Redis SET NX leader election at 100+
    container scale to ensure only one container registers configs.

    Parameters
    ----------
    observability_client : optional
        Injectable Langfuse/observability client for config registration.
    leader_election : optional
        Injectable leader election mechanism (Redis SET NX, etc.).

    Returns
    -------
    int
        Number of configs registered (0 if not leader).
    """
    if leader_election is not None:
        is_leader = leader_election.try_acquire("score_config_registration", ttl_seconds=60)
        if not is_leader:
            logger.debug("Not leader — skipping score config registration")
            return 0

    registered = 0
    for config_id, config in SCORE_CONFIGS.items():
        try:
            if observability_client is not None:
                observability_client.create_score_config(
                    name=config.name,
                    data_type=config.data_type.value,
                    min_value=config.min_value if config.data_type == ScoreDataType.NUMERIC else None,
                    max_value=config.max_value if config.data_type == ScoreDataType.NUMERIC else None,
                    categories=list(config.categories) if config.categories else None,
                    description=config.description,
                )
            registered += 1
        except Exception as exc:
            logger.debug("Config %s already registered or error: %s", config_id, exc)

    logger.info("Registered %d/%d score configs", registered, len(SCORE_CONFIGS))
    return registered
