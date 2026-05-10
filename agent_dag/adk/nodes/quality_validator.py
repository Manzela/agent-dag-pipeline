"""Node 6 — Quality Validator (ADK-Native)."""

from __future__ import annotations

from typing import Any

from ...nodes.node6_quality_validator import gate_format_compliance
from ...shared.data_contracts import ContentBlockOutput, ORAVDecision
from ..gate_agent import GateAgent, GateDecision, GateResult


class QualityValidator(GateAgent):
    """ADK agent: O-R-A-V quality evaluation and format compliance."""

    name: str = "quality_validator"
    description: str = (
        "Validates generated content for label leaks, transactional language, "
        "template variable leaks, and O-R-A-V quality scoring."
    )

    def gate(self, state: dict[str, Any]) -> GateResult:
        node5 = state.get("content_generator:result")
        if not node5:
            return GateResult(GateDecision.REJECT, "No content from Node 5")
        return GateResult(GateDecision.PASS)

    async def agent(self, state: dict[str, Any], ctx: Any) -> dict[str, Any]:
        content_data = state["content_generator:result"]
        content = ContentBlockOutput(**content_data)

        diagnostics = gate_format_compliance(content)

        if diagnostics["errors"]:
            return {
                "decision": ORAVDecision.FAIL.value,
                "errors": diagnostics["errors"],
                "warnings": diagnostics["warnings"],
            }

        return {
            "decision": ORAVDecision.PASS.value,
            "errors": [],
            "warnings": diagnostics["warnings"],
            "content": content.model_dump(),
        }
