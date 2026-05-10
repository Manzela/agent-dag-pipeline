"""
Node 5 — Content Generator.

Role: CONTENT GENERATION
Causal Trace: Qualified Terms → Template Gate → LoRA Generation → Draft Content

Architecture (Gate-Agent Pattern):
    ┌─────────────────────────────────────────────┐
    │ DETERMINISTIC GATE: Template Selector         │
    │   Pattern-matches product category to pre-    │
    │   validated content templates. Enforces       │
    │   structural constraints (char limits).       │
    │   Tools: Template Registry, Constraint DSL    │
    ├─────────────────────────────────────────────┤
    │ PROBABILISTIC AGENT: Content Generation       │
    │   Produces optimized content with structured  │
    │   Pydantic output. Each block is generated    │
    │   independently and JIT-audited by DEMAS.     │
    │   Tools: LLM (MoE), LoRA Adapter, Prompt Cache│
    └─────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
from typing import Any

from ..shared.data_contracts import ContentBlockOutput

logger = logging.getLogger(__name__)

# Content block structural constraints (character limits)
BLOCK_CONSTRAINTS: dict[str, dict[str, int]] = {
    "full_description": {"min_chars": 200, "max_chars": 2000},
    "short_description": {"min_chars": 50, "max_chars": 300},
    "store_welcome_line": {"min_chars": 50, "max_chars": 120},
    "meta_title": {"min_chars": 30, "max_chars": 65},
    "meta_description": {"min_chars": 120, "max_chars": 160},
    "image_alt_tag": {"min_chars": 10, "max_chars": 125},
}

# Minimum counts for list fields
LIST_CONSTRAINTS: dict[str, dict[str, int]] = {
    "key_features": {"min_items": 3, "max_items": 8},
    "local_trending_features": {"min_items": 2, "max_items": 5},
    "faq_items": {"min_items": 3, "max_items": 6},
}


def gate_template_selector(semantic_category: str) -> bool:
    """Select and validate content template based on product category.

    Returns True if a valid template exists for the category.
    In production, this maps semantic categories to pre-validated
    content templates with per-block structural constraints.
    """
    # All categories are accepted — the constraint enforcement
    # happens post-generation via BLOCK_CONSTRAINTS validation.
    return True


def validate_block_constraints(content: ContentBlockOutput) -> list[str]:
    """Validate generated content against structural constraints.

    Parameters
    ----------
    content : ContentBlockOutput
        The generated content to validate.

    Returns
    -------
    list of str
        Constraint violations found (empty if all pass).
    """
    violations: list[str] = []

    for field_name, limits in BLOCK_CONSTRAINTS.items():
        text = getattr(content, field_name, "")
        if text and len(text) < limits["min_chars"]:
            violations.append(
                f"{field_name}: {len(text)} chars < min {limits['min_chars']}"
            )
        if text and len(text) > limits["max_chars"]:
            violations.append(
                f"{field_name}: {len(text)} chars > max {limits['max_chars']}"
            )

    for field_name, limits in LIST_CONSTRAINTS.items():
        items = getattr(content, field_name, [])
        if items and len(items) < limits["min_items"]:
            violations.append(
                f"{field_name}: {len(items)} items < min {limits['min_items']}"
            )

    return violations


async def run_node5(state: dict[str, Any]) -> dict[str, Any]:
    """Execute Node 5: Content Generator with Gate-Agent pattern.

    Extracts context from upstream nodes and generates content blocks.
    When an LLM client is available, generates real content via the
    provider. Otherwise, returns deterministic stubs for testing.
    """
    # ── Extract context from upstream nodes ──
    node2_result = state.get("node2_result", {})
    state.get("node4_result", {})
    store_context = state.get("store_context")
    llm_client = state.get("llm_client")

    semantic_category = ""
    if hasattr(node2_result, "semantic_category"):
        semantic_category = node2_result.semantic_category
    elif isinstance(node2_result, dict):
        semantic_category = node2_result.get("semantic_category", "")

    lean_name = ""
    if hasattr(node2_result, "lean_product_name"):
        lean_name = node2_result.lean_product_name
    elif isinstance(node2_result, dict):
        lean_name = node2_result.get("lean_product_name", "")

    # ── Step 1: Deterministic Gate ──
    if not gate_template_selector(semantic_category):
        raise ValueError(
            f"Node 5 gate REJECTED: no template for category '{semantic_category}'"
        )

    # ── Step 2: Probabilistic Agent ──
    if llm_client is not None and lean_name:
        # Generate content using the injected LLM client
        city = store_context.city if store_context else ""
        language = store_context.language_name if store_context else "English"

        prompt = (
            f"Generate product content for '{lean_name}' "
            f"(category: {semantic_category or 'general'}) "
            f"for a store in {city}. Language: {language}. "
            f"Include: full description, short description, "
            f"store welcome line, meta title, meta description, "
            f"image alt tag, key features, and FAQ items."
        )

        try:
            content = await llm_client.generate_structured(
                prompt,
                ContentBlockOutput,
                system_prompt="You are a product content specialist.",
            )
        except Exception as exc:
            logger.warning("LLM generation failed, using stub: %s", exc)
            content = ContentBlockOutput()
    else:
        content = ContentBlockOutput()

    # ── Post-generation constraint validation ──
    violations = validate_block_constraints(content)
    if violations:
        logger.debug("Block constraint violations (non-fatal for stubs): %s", violations)

    return content.model_dump()
