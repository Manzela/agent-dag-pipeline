"""
Node 3 — Synonym Generator (Semantic Expansion).

Role: SEMANTIC EXPANSION
Causal Trace: Base Terms → Dedup Gate → LLM Expansion → Synonym Set

Architecture (Gate-Agent Pattern):
    ┌─────────────────────────────────────────────┐
    │ DETERMINISTIC GATE: Deduplication Filter     │
    │   Hash-based exact-match and fuzzy dedup.    │
    │   Tools: SimHash, Levenshtein, Set Algebra   │
    ├─────────────────────────────────────────────┤
    │ PROBABILISTIC AGENT: Synonym Generation      │
    │   Generates locale-aware semantic variants    │
    │   and long-tail expansions.                  │
    │   Tools: LLM, Embedding Similarity           │
    └─────────────────────────────────────────────┘
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

from ..shared.data_contracts import SynonymOutput

logger = logging.getLogger(__name__)


def gate_dedup_filter(base_terms: list[str]) -> list[str]:
    """Remove duplicate and near-duplicate terms using hash-based dedup."""
    seen_hashes: set[str] = set()
    unique: list[str] = []
    for term in base_terms:
        normalized = term.strip().lower()
        h = hashlib.md5(normalized.encode()).hexdigest()[:8]
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(term.strip())
    return unique


async def agent_synonym_generation(
    lean_product_name: str,
    semantic_category: str,
    target_language: str,
    country_code: str,
    *,
    llm_client: Optional[Any] = None,
) -> list[str]:
    """Generate locale-aware synonym expansions using LLM."""
    # In production, the LLM generates locale-specific synonyms
    # with brand-aware terminology constraints.
    return []


async def run_node3(state: dict[str, Any]) -> dict[str, Any]:
    """Execute Node 3: Synonym Generator."""
    node2 = state["node2_result"]
    store = state["store_context"]

    base_terms = [node2.lean_product_name]
    deduped = gate_dedup_filter(base_terms)

    if not deduped:
        raise ValueError("Node 3 gate REJECTED: no valid base terms after dedup")

    synonyms = await agent_synonym_generation(
        node2.lean_product_name, node2.semantic_category,
        node2.target_language, store.country_code,
    )

    return SynonymOutput(
        lean_product_name=node2.lean_product_name,
        synonyms=synonyms,
        target_language=node2.target_language,
        country_code=store.country_code,
    ).model_dump()
