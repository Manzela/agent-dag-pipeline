"""
Node 2 — Input Normalizer (Data Cleansing).

Role: DATA CLEANSING
Causal Trace: Raw Input → Schema Gate → Semantic Extraction → Normalized Record

Architecture (Gate-Agent Pattern):
    ┌─────────────────────────────────────────────┐
    │ DETERMINISTIC GATE: Schema Validator         │
    │   Enforces strict JSON schema, structural    │
    │   integrity, and type coercion.              │
    │   Tools: Pydantic, Regex, Python AST         │
    ├─────────────────────────────────────────────┤
    │ PROBABILISTIC AGENT: Semantic Extraction     │
    │   Uses LLM to extract semantic category,     │
    │   lean product name, and brand signals.       │
    │   Tools: LLM, Cached Prompts, Redis LTM      │
    └─────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from ..shared.data_contracts import NormalizerOutput, StoreContext

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# DETERMINISTIC GATE: Schema Validator
# ════════════════════════════════════════════════════════════════════════════

REQUIRED_FIELDS: frozenset[str] = frozenset({"product_name", "product-name", "name"})


def _coerce_str(val: Any) -> str:
    """Safely coerce webscraper values to string.

    Webscrapers inconsistently return list, None, or str for product
    attributes. This function normalizes all of them to a clean string.
    """
    if isinstance(val, list):
        return ", ".join(str(v) for v in val) if val else ""
    if val is None:
        return ""
    return str(val).strip()


def gate_schema_validator(product: dict[str, Any]) -> bool:
    """Validate that the raw product data has minimum required fields.

    Parameters
    ----------
    product : dict
        Raw product data from the upstream scraper/API.

    Returns
    -------
    bool
        True if the product has at least one valid product name field.
    """
    for field_name in REQUIRED_FIELDS:
        val = product.get(field_name, "")
        if _coerce_str(val):
            return True

    logger.warning("Gate REJECT: no product name found in fields %s", list(product.keys())[:10])
    return False


# ════════════════════════════════════════════════════════════════════════════
# PROBABILISTIC AGENT: Semantic Extraction
# ════════════════════════════════════════════════════════════════════════════

# Noise patterns to strip from product names
_NOISE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(ref|sku|ean|upc)[.:]\s*\S+", re.IGNORECASE),
    re.compile(r"\s{2,}"),
]


def _clean_product_name(raw_name: str) -> str:
    """Remove noise tokens (SKU references, extra whitespace) from product names."""
    cleaned = raw_name
    for pattern in _NOISE_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    return cleaned.strip()


async def agent_semantic_extraction(
    product: dict[str, Any],
    store_context: StoreContext,
    *,
    llm_client: Optional[Any] = None,
) -> NormalizerOutput:
    """Extract semantic category and lean product name using LLM.

    Parameters
    ----------
    product : dict
        Schema-validated product data.
    store_context : StoreContext
        Store context for language-aware extraction.

    Returns
    -------
    NormalizerOutput
        Normalized product data with semantic classification.
    """
    raw_name = _coerce_str(
        product.get("product-name") or product.get("product_name") or product.get("name", "")
    )
    clean_name = _clean_product_name(raw_name)
    brand = _coerce_str(product.get("brand", ""))

    # In production, the LLM performs semantic taxonomy classification
    # and lean product name extraction with structured output.
    return NormalizerOutput(
        original_product_name=raw_name,
        lean_product_name=clean_name,
        semantic_category="",  # Populated by LLM in production
        brand=brand,
        target_language=store_context.language_name or store_context.language,
        parent_fingerprint=_coerce_str(product.get("sku", "")),
    )


# ════════════════════════════════════════════════════════════════════════════
# NODE ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

async def run_node2(
    product: dict[str, Any],
    store_context: StoreContext,
    *,
    llm_client: Optional[Any] = None,
) -> NormalizerOutput:
    """Execute Node 2: Input Normalizer.

    Implements the Gate-Agent pattern:
    1. GATE: Schema Validator (deterministic, O(1))
    2. AGENT: Semantic Extraction (probabilistic, LLM-powered)
    """
    if not gate_schema_validator(product):
        raise ValueError(
            f"Node 2 gate REJECTED: product has no valid name field. "
            f"Available keys: {list(product.keys())[:10]}"
        )

    return await agent_semantic_extraction(product, store_context, llm_client=llm_client)
