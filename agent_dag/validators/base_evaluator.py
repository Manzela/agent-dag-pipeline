"""
DEMAS Evaluator Framework — Base Classes and Registry.

The DEMAS (Deterministic Evaluation, Multi-model Assessment System)
framework provides the JIT audit layer that intercepts at every node
boundary in the pipeline.

Core Components:
    - BaseEvaluator: Abstract interface for all evaluators
    - EvaluatorRegistry: Discovers and loads evaluators at startup
    - ProvenanceMatrix: Maps content blocks to ground-truth variables
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """Abstract base class for all DEMAS evaluators.

    Every evaluator must implement evaluate() which takes a content
    block and its provenance context, returning a typed verdict.
    """

    @abstractmethod
    async def evaluate(
        self,
        content: str,
        provenance: dict[str, Any],
        *,
        llm_client: Any | None = None,
    ) -> dict[str, Any]:
        """Evaluate a content block against its provenance context.

        Parameters
        ----------
        content : str
            The generated content block to evaluate.
        provenance : dict
            Ground-truth variables from the Provenance Matrix
            relevant to this specific block.

        Returns
        -------
        dict
            Evaluation result with keys: verdict, score, reasons.
        """
        ...


# ════════════════════════════════════════════════════════════════════════════
# PROVENANCE MATRIX — Prevents attention dilution in LLM-as-Judge
# ════════════════════════════════════════════════════════════════════════════

PROVENANCE_MATRIX: dict[str, list[str]] = {
    "full_description": [
        "lean_product_name", "semantic_category", "brand", "trigger_term",
        "city_dna", "style_seed", "visual_context",
    ],
    "short_description": [
        "lean_product_name", "semantic_category", "trigger_term",
    ],
    "meta_title": [
        "lean_product_name", "trigger_term", "city",
    ],
    "meta_description": [
        "lean_product_name", "semantic_category", "trigger_term",
        "secondary_keyword",
    ],
    "store_welcome_line": [
        "city", "store_name", "lean_product_name",
    ],
    "image_alt_tag": [
        "lean_product_name", "brand", "color", "visual_context",
    ],
    "key_features": [
        "lean_product_name", "semantic_category", "visual_context",
    ],
    "faq_items": [
        "lean_product_name", "semantic_category", "trigger_term",
        "relevant_search_queries",
    ],
}


def get_provenance_context(
    block_name: str,
    full_context: dict[str, Any],
) -> dict[str, Any]:
    """Extract block-specific provenance variables from full context.

    This prevents attention dilution in LLM-as-Judge evaluations by
    giving the judge model ONLY the ground-truth variables relevant
    to the specific content block being evaluated.
    """
    allowed_vars = PROVENANCE_MATRIX.get(block_name, [])
    return {k: full_context.get(k) for k in allowed_vars if k in full_context}


class EvaluatorRegistry:
    """Registry for discovering and loading DEMAS evaluators."""

    def __init__(self) -> None:
        self._evaluators: dict[str, BaseEvaluator] = {}

    def register(self, name: str, evaluator: BaseEvaluator) -> None:
        """Register an evaluator by name."""
        self._evaluators[name] = evaluator
        logger.info("Registered evaluator: %s", name)

    def get(self, name: str) -> BaseEvaluator | None:
        """Get a registered evaluator by name."""
        return self._evaluators.get(name)

    @property
    def registered_names(self) -> list[str]:
        """List all registered evaluator names."""
        return list(self._evaluators.keys())
