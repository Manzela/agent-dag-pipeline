"""Node 3 — Synonym Generator (ADK-Native)."""

from __future__ import annotations

from typing import Any

from ..gate_agent import GateAgent, GateDecision, GateResult
from ...nodes.node3_synonym_generator import (
    agent_synonym_generation,
    gate_dedup_filter,
)
from ...shared.data_contracts import SynonymOutput


class SynonymGenerator(GateAgent):
    """ADK agent: Locale-aware synonym expansion."""

    name: str = "synonym_generator"
    description: str = (
        "Generates locale-specific synonym expansions for product terms "
        "with brand-aware terminology constraints."
    )

    def gate(self, state: dict[str, Any]) -> GateResult:
        node2 = state.get("input_normalizer:result", {})
        lean_name = node2.get("lean_product_name", "")
        if not lean_name:
            return GateResult(GateDecision.REJECT, "No lean product name from Node 2")
        deduped = gate_dedup_filter([lean_name])
        if not deduped:
            return GateResult(GateDecision.REJECT, "No valid terms after dedup")
        return GateResult(GateDecision.PASS)

    async def agent(self, state: dict[str, Any], ctx: Any) -> dict[str, Any]:
        node2 = state["input_normalizer:result"]
        store_context = state["store_context"]
        synonyms = await agent_synonym_generation(
            node2["lean_product_name"],
            node2.get("semantic_category", ""),
            node2.get("target_language", ""),
            store_context.country_code,
        )
        return SynonymOutput(
            lean_product_name=node2["lean_product_name"],
            synonyms=synonyms,
            target_language=node2.get("target_language", ""),
            country_code=store_context.country_code,
        ).model_dump()
