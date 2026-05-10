"""Node 4 — Demand Gatekeeper (ADK-Native)."""

from __future__ import annotations

from typing import Any

from ...nodes.node4_demand_gatekeeper import gate_threshold_comparator
from ...shared.data_contracts import GateDecision as ContractGateDecision
from ..gate_agent import GateAgent, GateDecision, GateResult


class DemandGatekeeper(GateAgent):
    """ADK agent: Search volume threshold filtering."""

    name: str = "demand_gatekeeper"
    description: str = (
        "Applies configurable numeric thresholds against search volume data. "
        "Rejects zero-demand products to save downstream LLM costs."
    )

    def gate(self, state: dict[str, Any]) -> GateResult:
        node3 = state.get("synonym_generator:result", {})
        if not node3.get("lean_product_name"):
            return GateResult(GateDecision.REJECT, "No product name from Node 3")
        return GateResult(GateDecision.PASS)

    async def agent(self, state: dict[str, Any], ctx: Any) -> dict[str, Any]:
        node3 = state["synonym_generator:result"]
        # In production, search volume comes from a data source
        search_volume = 0

        decision = gate_threshold_comparator(search_volume)

        if decision == ContractGateDecision.REJECT:
            raise ValueError(f"REJECT_ZERO_DEMAND: SV={search_volume}")

        return {
            "decision": decision.value,
            "trigger_term": node3["lean_product_name"],
            "search_volume": search_volume,
        }
