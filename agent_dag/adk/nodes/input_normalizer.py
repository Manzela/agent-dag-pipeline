"""Node 2 — Input Normalizer (ADK-Native)."""

from __future__ import annotations

from typing import Any

from ...nodes.node2_input_normalizer import (
    agent_semantic_extraction,
    gate_schema_validator,
)
from ..gate_agent import GateAgent, GateDecision, GateResult


class InputNormalizer(GateAgent):
    """ADK agent: Schema validation and semantic extraction."""

    name: str = "input_normalizer"
    description: str = (
        "Validates raw product data schema and extracts lean product name, "
        "semantic category, and target language."
    )

    def gate(self, state: dict[str, Any]) -> GateResult:
        product = state.get("product")
        if not product:
            return GateResult(GateDecision.REJECT, "No product data")
        if not gate_schema_validator(product):
            return GateResult(
                GateDecision.REJECT,
                f"Invalid schema: missing name field. Keys: {list(product.keys())[:5]}",
            )
        return GateResult(GateDecision.PASS)

    async def agent(self, state: dict[str, Any], ctx: Any) -> dict[str, Any]:
        product = state["product"]
        store_context = state["store_context"]
        result = await agent_semantic_extraction(product, store_context)
        return result.model_dump()
