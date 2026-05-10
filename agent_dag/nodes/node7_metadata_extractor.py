"""
Node 7 — Metadata Extractor (Vectorization).

Role: VECTORIZATION
Causal Trace: Validated Content → Dimension Gate → Embedding Generation → Feature Vector

Gate-Agent Pattern:
    GATE: Embedding Dimension Check — NaN detection, magnitude normalization
    AGENT: Feature Extraction — dense vector representations for similarity search
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


def gate_embedding_dimension_check(
    vector: list[float],
    expected_dim: int = 768,
) -> bool:
    """Validate vector dimensionality and numeric integrity.

    Checks:
        - Correct dimensionality
        - No NaN values
        - Magnitude normalization within bounds
    """
    if len(vector) != expected_dim:
        logger.warning("Gate REJECT: dim=%d expected=%d", len(vector), expected_dim)
        return False

    for i, val in enumerate(vector):
        if math.isnan(val) or math.isinf(val):
            logger.warning("Gate REJECT: NaN/Inf at index %d", i)
            return False

    magnitude = math.sqrt(sum(v * v for v in vector))
    if magnitude < 1e-6:
        logger.warning("Gate REJECT: zero-magnitude vector")
        return False

    return True


async def run_node7(state: dict[str, Any]) -> dict[str, Any]:
    """Execute Node 7: Metadata Extractor with Gate-Agent pattern."""
    state.get("node6_result", {}).get("content", {})

    # In production, generates embedding vectors from validated content
    return {
        "embeddings": [],
        "metadata": {"content_hash": "", "model_version": ""},
    }
