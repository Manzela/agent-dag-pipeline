"""Node 7 — Metadata Extractor (ADK-Native)."""

from __future__ import annotations

from typing import Any

from ..gate_agent import GateAgent, GateDecision, GateResult


class MetadataExtractor(GateAgent):
    """ADK agent: Embedding vectorization and metadata extraction."""

    name: str = "metadata_extractor"
    description: str = (
        "Generates dense vector representations from validated content "
        "for similarity search and clustering."
    )

    def gate(self, state: dict[str, Any]) -> GateResult:
        node6 = state.get("quality_validator:result", {})
        if not node6:
            return GateResult(GateDecision.REJECT, "No validated content from Node 6")
        decision = node6.get("decision", "")
        if decision != "PASS":
            return GateResult(GateDecision.REJECT, f"Quality gate: {decision}")
        return GateResult(GateDecision.PASS)

    async def agent(self, state: dict[str, Any], ctx: Any) -> dict[str, Any]:
        return {
            "embeddings": [],
            "metadata": {"content_hash": "", "model_version": ""},
        }
