"""
Node 4 — Demand Gatekeeper (Volume Filtering).

Role: VOLUME FILTERING
Causal Trace: Synonym Set → Volume Lookup → Threshold Gate → Qualified Terms

Architecture (Gate-Agent Pattern):
    ┌─────────────────────────────────────────────┐
    │ DETERMINISTIC GATE: Threshold Comparator     │
    │   Applies configurable numeric thresholds    │
    │   against search volume data. Purely math.   │
    │   Tools: Threshold Config, Data Store        │
    ├─────────────────────────────────────────────┤
    │ PROBABILISTIC AGENT: Trend Analysis          │
    │   For borderline candidates, applies trend   │
    │   detection to identify emerging terms.       │
    │   Tools: LLM, Time-Series Decomposition      │
    └─────────────────────────────────────────────┘

Gate Decisions:
    PASS   — Term meets or exceeds demand threshold → proceed
    REJECT — Zero demand detected → permanent skip (retry_eligible=False)
    DEFER  — Quota exceeded for this cycle → retry next cycle
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..shared.data_contracts import GateDecision, IntentLockOutput

logger = logging.getLogger(__name__)

# Configurable thresholds (injected via environment or config)
DEFAULT_MIN_SEARCH_VOLUME: int = 10
DEFAULT_MAX_MONTHLY_QUOTA: int = 5000


def gate_threshold_comparator(
    search_volume: int,
    *,
    min_threshold: int = DEFAULT_MIN_SEARCH_VOLUME,
) -> GateDecision:
    """Apply numeric threshold gate against search volume data.

    This is purely mathematical — zero probabilistic uncertainty.
    """
    if search_volume <= 0:
        return GateDecision.REJECT
    if search_volume < min_threshold:
        return GateDecision.REJECT
    return GateDecision.PASS


async def run_node4(state: dict[str, Any]) -> dict[str, Any]:
    """Execute Node 4: Demand Gatekeeper."""
    node3_result = state.get("node3_result", {})
    trigger_term = node3_result.get("lean_product_name", "")
    synonyms = node3_result.get("synonyms", [])

    # In production, search volume is looked up from a data store
    search_volume = 0  # Placeholder — injected from demand data source

    decision = gate_threshold_comparator(search_volume)

    return {
        "decision": decision,
        "trigger_term": trigger_term,
        "search_volume": search_volume,
        "reason": f"SV={search_volume}, threshold={DEFAULT_MIN_SEARCH_VOLUME}",
    }
