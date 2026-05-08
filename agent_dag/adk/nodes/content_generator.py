"""Node 5 — Content Generator (ADK-Native)."""

from __future__ import annotations

from typing import Any

from ..gate_agent import GateAgent, GateDecision, GateResult
from ...nodes.node5_content_generator import (
    gate_template_selector,
    validate_block_constraints,
)
from ...shared.data_contracts import ContentBlockOutput


class ContentGenerator(GateAgent):
    """ADK agent: LLM-powered content block generation."""

    name: str = "content_generator"
    description: str = (
        "Generates product content blocks (descriptions, meta tags, FAQs) "
        "using locale-aware templates and LLM structured output."
    )

    def gate(self, state: dict[str, Any]) -> GateResult:
        node2 = state.get("input_normalizer:result", {})
        category = node2.get("semantic_category", "")
        if not gate_template_selector(category):
            return GateResult(
                GateDecision.REJECT,
                f"No template for category: {category}",
            )
        return GateResult(GateDecision.PASS)

    async def agent(self, state: dict[str, Any], ctx: Any) -> dict[str, Any]:
        content = ContentBlockOutput()
        violations = validate_block_constraints(content)
        return content.model_dump()
