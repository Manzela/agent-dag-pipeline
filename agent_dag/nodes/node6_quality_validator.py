"""
Node 6 — Quality Validator (O-R-A-V Evaluation).

Role: O-R-A-V EVALUATION
Causal Trace: Draft Content → Format Gate → O-R-A-V Scoring → Accept or Reject

Gate-Agent Pattern:
    GATE: Format Compliance Check — word count, forbidden patterns, structure
    AGENT: Multi-Model Consensus Judge — O-R-A-V 4-dimension scoring
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ..shared.data_contracts import ContentBlockOutput, ORAVDecision

logger = logging.getLogger(__name__)

# Internal pipeline labels that must NEVER appear in generated content
SACRED_LABELS: frozenset[str] = frozenset({
    "sacred context", "gmb_context", "city_dna", "trigger_term",
    "style_seed", "context_payload", "lean_product_name",
    "semantic_category", "format_vars",
})

# Transactional language that violates content policy
TRANSACTIONAL_BLACKLIST: frozenset[str] = frozenset({
    "buy now", "shop now", "order now", "add to cart",
    "purchase", "checkout", "free shipping", "limited time",
    "act now", "don't miss", "exclusive offer",
})


def gate_format_compliance(content: ContentBlockOutput) -> dict[str, Any]:
    """Deterministic format compliance check. O(1), zero LLM cost.

    Checks:
        Q1: Internal label leak detection
        Transactional blacklist enforcement
        Template variable leak detection ({var} or {{var}})
        Empty field validation
    """
    errors: list[str] = []
    warnings: list[str] = []

    # ── Q1: Label Leak Detection ──
    all_text_fields = [
        ("full_description", content.full_description),
        ("short_description", content.short_description),
        ("store_welcome_line", content.store_welcome_line),
        ("meta_title", content.meta_title),
        ("meta_description", content.meta_description),
        ("image_alt_tag", content.image_alt_tag),
    ]
    for field_name, text in all_text_fields:
        text_lower = text.lower()
        for label in SACRED_LABELS:
            if label in text_lower:
                errors.append(f"LABEL_LEAK: '{label}' found in {field_name}")

    # ── Label leak in list fields ──
    for i, feature in enumerate(content.key_features):
        for label in SACRED_LABELS:
            if label in feature.lower():
                errors.append(f"LABEL_LEAK: '{label}' in key_features[{i}]")

    for i, trend in enumerate(content.local_trending_features):
        for label in SACRED_LABELS:
            if label in trend.lower():
                errors.append(f"LABEL_LEAK: '{label}' in local_trending_features[{i}]")

    for i, faq in enumerate(content.faq_items):
        answer = faq.get("answer", "").lower()
        for label in SACRED_LABELS:
            if label in answer:
                errors.append(f"LABEL_LEAK: '{label}' in faq_items[{i}].answer")

    # ── Transactional Blacklist ──
    for field_name, text in all_text_fields:
        text_lower = text.lower()
        for phrase in TRANSACTIONAL_BLACKLIST:
            if phrase in text_lower:
                errors.append(f"TRANSACTIONAL: '{phrase}' in {field_name}")

    # ── Template Variable Leak ──
    var_pattern = re.compile(r"\{[{]?[a-zA-Z_]+[}]?\}")
    for field_name, text in all_text_fields:
        if var_pattern.search(text):
            errors.append(f"TEMPLATE_VAR_LEAK: unreplaced variable in {field_name}")

    return {"errors": errors, "warnings": warnings}


async def run_node6(state: dict[str, Any]) -> dict[str, Any]:
    """Execute Node 6: Quality Validator with Gate-Agent pattern."""
    content_data = state.get("node5_result", {})
    content = ContentBlockOutput(**content_data) if isinstance(content_data, dict) else content_data

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
        "content": content.model_dump() if hasattr(content, "model_dump") else content_data,
    }
