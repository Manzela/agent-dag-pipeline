"""Node 1 — Context Researcher (ADK-Native)."""

from __future__ import annotations

from typing import Any

from ...nodes.node1_context_researcher import (
    agent_cultural_context,
    gate_locale_resolver,
)
from ..gate_agent import GateAgent, GateDecision, GateResult


class ContextResearcher(GateAgent):
    """ADK agent: City DNA cultural enrichment via locale validation."""

    name: str = "context_researcher"
    description: str = (
        "Validates locale codes and enriches product context with "
        "city-level cultural signals (demographics, geography, culture)."
    )

    def gate(self, state: dict[str, Any]) -> GateResult:
        store_context = state.get("store_context")
        if not store_context:
            return GateResult(GateDecision.REJECT, "No store context")
        if not gate_locale_resolver(store_context):
            return GateResult(
                GateDecision.REJECT,
                f"Invalid locale: {store_context.country_code}",
            )
        return GateResult(GateDecision.PASS)

    async def agent(self, state: dict[str, Any], ctx: Any) -> dict[str, Any]:
        from dataclasses import asdict

        store_context = state["store_context"]
        profile = await agent_cultural_context(store_context)
        return asdict(profile)
