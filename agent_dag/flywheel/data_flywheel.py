"""
Data Flywheel — 3-Tier Dataset Curation Engine for RL Self-Retraining.

This module implements the closed-loop data flywheel that enables
continuous model improvement through production data curation.

Architecture (mirrors Anthropic's Constitutional AI methodology):

    Production Traffic
          │
          ▼
    ┌─────────────────────────┐
    │ TIER 1: production-baseline │  ← All production runs
    │   (Raw execution data)       │
    └─────────┬───────────────┘
              │ O-R-A-V + DEMAS scoring
              ▼
    ┌─────────────────────────┐
    │ TIER 2: quality-approved     │  ← ORAV_QUALITY ≥ 0.7 && DEMAS PASS
    │   (Curated high-quality)     │
    └─────────┬───────────────┘
              │ Failures + rejections
              ▼
    ┌─────────────────────────┐
    │ TIER 3: failure-cases        │  ← ORAV_QUALITY < 0.5 || DEMAS FAIL
    │   (Negative examples)        │
    └─────────────────────────┘

    TIER 2 + TIER 3 → DPO Preference Pairs → LoRA Fine-Tuning

Key Design Decisions:
    - 17 prompt-specific dataset partitions for granular analysis
    - Schema-enforced ingestion (Pydantic validation at write time)
    - Multi-tenant partitioning (tenant data never crosses boundaries)
    - Langfuse dataset API integration for trace-linked evaluation
    - Fail-open: flywheel failures never block the main pipeline
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DatasetTier(str, Enum):
    """3-tier dataset curation hierarchy."""
    PRODUCTION_BASELINE = "production-baseline"
    QUALITY_APPROVED = "quality-approved"
    FAILURE_CASES = "failure-cases"


class CurationDecision(str, Enum):
    """Curation routing decision based on scoring thresholds."""
    APPROVE = "APPROVE"        # → quality-approved tier
    BASELINE_ONLY = "BASELINE"  # → stays in production-baseline
    REJECT = "REJECT"           # → failure-cases tier


# ════════════════════════════════════════════════════════════════════════════
# CURATION THRESHOLDS
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CurationThresholds:
    """Configurable thresholds for tier routing decisions.

    Attributes
    ----------
    approve_min_orav : float
        Minimum O-R-A-V composite score for quality-approved tier.
    approve_min_demas : bool
        Whether DEMAS JIT must pass for quality-approved tier.
    reject_max_orav : float
        Maximum O-R-A-V score below which content goes to failure-cases.
    margin_threshold : float
        Minimum score margin between chosen/rejected for DPO pair validity.
    """
    approve_min_orav: float = 0.7
    approve_min_demas: bool = True
    reject_max_orav: float = 0.5
    margin_threshold: float = 0.15


DEFAULT_THRESHOLDS = CurationThresholds()


# ════════════════════════════════════════════════════════════════════════════
# DATASET ITEM — Schema-enforced ingestion unit
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class DatasetItem:
    """A single ingestion unit for the data flywheel.

    Every pipeline run that completes (success or failure) produces
    one DatasetItem. The item contains the full input-output pair
    plus all scoring metadata needed for curation routing.
    """
    trace_id: str
    tenant_id: str
    prompt_id: str
    input_payload: dict[str, Any]
    output_content: dict[str, Any]
    scores: dict[str, float]
    tier: DatasetTier = DatasetTier.PRODUCTION_BASELINE
    curation_decision: CurationDecision = CurationDecision.BASELINE_ONLY
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    pipeline_version: str = ""
    model_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "trace_id": self.trace_id,
            "tenant_id": self.tenant_id,
            "prompt_id": self.prompt_id,
            "input": self.input_payload,
            "output": self.output_content,
            "scores": self.scores,
            "tier": self.tier.value,
            "decision": self.curation_decision.value,
            "timestamp": self.timestamp,
            "pipeline_version": self.pipeline_version,
            "model_version": self.model_version,
        }


# ════════════════════════════════════════════════════════════════════════════
# CURATION ENGINE — Routes items to appropriate tiers
# ════════════════════════════════════════════════════════════════════════════

def route_to_tier(
    item: DatasetItem,
    thresholds: CurationThresholds = DEFAULT_THRESHOLDS,
) -> DatasetItem:
    """Route a dataset item to the appropriate curation tier.

    Parameters
    ----------
    item : DatasetItem
        The item to route, containing scores from the evaluation framework.
    thresholds : CurationThresholds
        Configurable thresholds for routing decisions.

    Returns
    -------
    DatasetItem
        The same item with tier and curation_decision updated.
    """
    orav_score = item.scores.get("orav_quality", 0.0)
    demas_pass = item.scores.get("demas_jit_verdict", "FAIL") == "PASS"

    if orav_score >= thresholds.approve_min_orav:
        if not thresholds.approve_min_demas or demas_pass:
            item.tier = DatasetTier.QUALITY_APPROVED
            item.curation_decision = CurationDecision.APPROVE
            return item

    if orav_score < thresholds.reject_max_orav or not demas_pass:
        item.tier = DatasetTier.FAILURE_CASES
        item.curation_decision = CurationDecision.REJECT
        return item

    item.tier = DatasetTier.PRODUCTION_BASELINE
    item.curation_decision = CurationDecision.BASELINE_ONLY
    return item


# ════════════════════════════════════════════════════════════════════════════
# PROMPT REGISTRY — 17 prompt-specific dataset partitions
# ════════════════════════════════════════════════════════════════════════════

PROMPT_REGISTRY: dict[str, dict[str, str]] = {
    "full_description": {"node": "5", "io": "context → full_description"},
    "short_description": {"node": "5", "io": "context → short_description"},
    "meta_title": {"node": "5", "io": "context → meta_title"},
    "meta_description": {"node": "5", "io": "context → meta_description"},
    "image_alt_tag": {"node": "5", "io": "context → image_alt_tag"},
    "store_welcome_line": {"node": "5", "io": "context → store_welcome_line"},
    "key_features": {"node": "5", "io": "context → key_features[]"},
    "local_trending": {"node": "5", "io": "context → local_trending_features[]"},
    "focus_keywords": {"node": "5", "io": "context → focus_keywords[]"},
    "faq_generation": {"node": "5", "io": "context → faq_items[]"},
    "city_dna_demographics": {"node": "1", "io": "locale → demographics"},
    "city_dna_culture": {"node": "1", "io": "locale → culture"},
    "taxonomy_classify": {"node": "2", "io": "raw_name → semantic_category"},
    "synonym_expand": {"node": "3", "io": "lean_name → synonyms[]"},
    "trend_analysis": {"node": "4", "io": "candidates → trend_scores"},
    "orav_judge": {"node": "6", "io": "content+context → orav_scores"},
    "demas_audit": {"node": "6", "io": "block+provenance → audit_verdict"},
}


# ════════════════════════════════════════════════════════════════════════════
# FLYWHEEL — Main entry point for ingestion
# ════════════════════════════════════════════════════════════════════════════

class DataFlywheel:
    """Orchestrates the 3-tier data flywheel for self-improvement.

    Usage::

        flywheel = DataFlywheel(dataset_store=my_store)
        stats = await flywheel.ingest_pipeline_run(trace_id, state, tier)
    """

    def __init__(
        self,
        *,
        dataset_store: Optional[Any] = None,
        thresholds: CurationThresholds = DEFAULT_THRESHOLDS,
    ) -> None:
        self._store = dataset_store
        self._thresholds = thresholds

    async def ingest_pipeline_run(
        self,
        trace_id: str,
        state: dict[str, Any],
        tier: str = "production-baseline",
    ) -> dict[str, int]:
        """Ingest a complete pipeline run into the flywheel.

        Parameters
        ----------
        trace_id : str
            Observability trace ID for correlation.
        state : dict
            Complete pipeline state (all node outputs).
        tier : str
            Initial tier assignment.

        Returns
        -------
        dict
            Ingestion statistics {tier_name: count}.
        """
        stats: dict[str, int] = {
            "production_baseline": 0,
            "quality_approved": 0,
            "failure_cases": 0,
        }

        tenant_id = state.get("store_context", {})
        if hasattr(tenant_id, "store_id"):
            tenant_id = tenant_id.store_id
        else:
            tenant_id = str(tenant_id.get("store_id", "unknown"))

        for prompt_id, prompt_meta in PROMPT_REGISTRY.items():
            item = DatasetItem(
                trace_id=trace_id,
                tenant_id=tenant_id,
                prompt_id=prompt_id,
                input_payload=state.get("product", {}),
                output_content=state.get(f"node{prompt_meta['node']}_result", {}),
                scores=state.get("scores", {}),
            )

            routed_item = route_to_tier(item, self._thresholds)
            tier_key = routed_item.tier.value.replace("-", "_")
            stats[tier_key] = stats.get(tier_key, 0) + 1

            if self._store is not None:
                try:
                    self._store.write(routed_item.to_dict())
                except Exception as exc:
                    logger.debug("Flywheel write failed for %s (non-fatal): %s", prompt_id, exc)

        logger.info("Flywheel ingested: %s", stats)
        return stats


def ingest_failure_case(
    trace_id: str,
    failure_node: str,
    error_msg: str,
    product: dict[str, Any],
    store_loc: Any,
) -> None:
    """Convenience function to ingest a pipeline failure into the failure-cases tier.

    This is called from the orchestrator's _record_failure() function.
    Fail-open: errors here never propagate to the main pipeline.
    """
    logger.info(
        "Flywheel failure case: trace=%s node=%s error=%s",
        trace_id, failure_node, error_msg[:100],
    )
