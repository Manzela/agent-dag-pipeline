"""
Orchestrator — Master DAG Controller.

Wires all 7 nodes into the complete pipeline DAG with fail-closed safety
guarantees at every boundary.

DAG Topology:
    Phase 1 (Parallel):   Node 1 (Context Research) | Node 2 (Input Normalizer)
    Phase 2 (Sequential): Node 3 (Synonyms) → Node 4 (Demand Gate) →
                           Node 5 (Content Generator) → Node 6 (Quality Validator) →
                           Node 7 (Metadata Extractor)

Fail-Closed Policy:
    - Any node failure, rejection, or integrity violation is recorded
      to a persistent failure store and halts downstream propagation
    - Store context integrity: asserted at entry AND before export
    - REJECT_ZERO_DEMAND: permanent skip — recorded, retry_eligible=False
    - DEFERRED_QUOTA: queued for next cycle — recorded, retry_eligible=True

Gate-Agent Pattern:
    Every node internally implements a dual-layer architecture:
    1. DETERMINISTIC GATE — fires first, zero-LLM, hard pass/fail
    2. PROBABILISTIC AGENT — fires only if gate passes, LLM-powered
    See docs/architecture.md for the full Gate-Agent specification.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from .shared.data_contracts import (
    ContentBlockOutput,
    ContextPayload,
    GateDecision,
    PipelineResult,
    StoreContext,
)

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# CAUSAL TRACE — Per-node transformation paths for mechanistic interpretability
# ════════════════════════════════════════════════════════════════════════════

CAUSAL_TRACES: dict[int, str] = {
    1: "Locale ID → Geo Validation → Cultural Enrichment → Context Object",
    2: "Raw Input → Schema Gate → Semantic Extraction → Normalized Record",
    3: "Base Terms → Dedup Gate → LLM Expansion → Synonym Set",
    4: "Synonym Set → Volume Lookup → Threshold Gate → Qualified Terms",
    5: "Qualified Terms → Template Gate → LoRA Generation → Draft Content",
    6: "Draft Content → Format Gate → O-R-A-V Scoring → Accept or Reject",
    7: "Validated Content → Dimension Gate → Embedding Generation → Feature Vector",
}


class FailureReason(str, Enum):
    """Enumeration of all pipeline exit reasons for auditability."""
    NODE_FAILURE = "NODE_FAILURE"
    INTEGRITY_VIOLATION = "STORE_INTEGRITY_VIOLATION"
    REJECT_ZERO_DEMAND = "REJECT_ZERO_DEMAND"
    DEFERRED_QUOTA = "DEFERRED_QUOTA"
    ORAV_FAIL = "ORAV_QUALITY_FAIL"
    PIPELINE_EXCEPTION = "PIPELINE_EXCEPTION"


@dataclass(frozen=True)
class FailureRecord:
    """Immutable record of a pipeline exit for audit trail.

    Every product that exits the pipeline for any reason (failure, rejection,
    deferral) produces one of these records. This is the atomic unit of the
    fail-closed audit trail.
    """
    product_id: str
    store_id: str
    failure_node: str
    failure_reason: FailureReason
    error_message: str
    retry_eligible: bool
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    trace_id: str = ""
    causal_trace: str = ""


def record_failure(
    product_id: str,
    store_id: str,
    failure_node: str,
    reason: FailureReason,
    error_msg: str,
    *,
    retry_eligible: bool = True,
    trace_id: str = "",
    failure_store: Optional[Any] = None,
) -> FailureRecord:
    """Record a pipeline exit to the persistent failure store.

    FAIL-CLOSED POLICY: Every product exit (node failure, rejection,
    deferral, integrity violation) is recorded here for auditability.
    Downstream propagation is blocked for all exit paths.

    Parameters
    ----------
    product_id : str
        Unique identifier for the product being processed.
    store_id : str
        Identifier for the store/tenant context.
    failure_node : str
        The node where the pipeline exited (e.g., "NODE_4_DEMAND_GATE").
    reason : FailureReason
        Categorized exit reason from the FailureReason enum.
    error_msg : str
        Human-readable description of the failure.
    retry_eligible : bool
        False for permanent rejections, True for retryable failures.
    trace_id : str
        Observability trace ID for cross-referencing with Langfuse.
    failure_store : optional
        Injectable failure store backend (Firestore, file, etc.).
        If None, logs to structured logger only.

    Returns
    -------
    FailureRecord
        Immutable audit record of the pipeline exit.
    """
    node_id = int(failure_node.split("_")[1]) if "_" in failure_node else 0
    record = FailureRecord(
        product_id=product_id,
        store_id=store_id,
        failure_node=failure_node,
        failure_reason=reason,
        error_message=error_msg[:500],
        retry_eligible=retry_eligible,
        trace_id=trace_id,
        causal_trace=CAUSAL_TRACES.get(node_id, ""),
    )

    logger.warning(
        "PIPELINE_EXIT: node=%s reason=%s product=%s store=%s retry=%s",
        failure_node, reason.value, product_id, store_id, retry_eligible,
    )

    if failure_store is not None:
        try:
            failure_store.write(record)
        except Exception as exc:
            logger.error("Failure store write failed (non-fatal): %s", exc)

    return record


# ════════════════════════════════════════════════════════════════════════════
# STORE CONTEXT INTEGRITY — Entry and exit assertions
# ════════════════════════════════════════════════════════════════════════════

def assert_store_integrity(context: StoreContext, checkpoint: str) -> None:
    """Assert that the store context has not been mutated between checkpoints.

    This is a critical fail-closed mechanism. The store context (locale,
    timezone, cultural signals) is established at pipeline entry and must
    remain immutable throughout execution. Any mutation indicates a
    cross-contamination bug that would produce incorrect content.

    Parameters
    ----------
    context : StoreContext
        The frozen store context to verify.
    checkpoint : str
        Human-readable label for the assertion point (e.g., "ENTRY", "PRE_EXPORT").

    Raises
    ------
    AssertionError
        If the context has been mutated since the entry checkpoint.
    """
    if not context.is_frozen():
        raise AssertionError(
            f"STORE_INTEGRITY_VIOLATION at {checkpoint}: "
            f"Store context for {context.store_id} has been mutated. "
            f"This indicates cross-contamination and MUST be investigated."
        )
    logger.debug("Store integrity PASSED at checkpoint=%s store=%s", checkpoint, context.store_id)


# ════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

async def run_pipeline(
    product: dict[str, Any],
    store_context: StoreContext,
    *,
    llm_client: Optional[Any] = None,
    node_registry: Optional[dict[str, Any]] = None,
    failure_store: Optional[Any] = None,
    flywheel: Optional[Any] = None,
    trace_id: str = "",
) -> PipelineResult:
    """Execute the complete 7-node DAG pipeline for a single product.

    This is the main entry point. It orchestrates all 7 nodes in the
    defined topology (Phase 1 parallel, Phase 2 sequential), enforces
    fail-closed safety at every boundary, and feeds results into the
    data flywheel for self-improvement.

    Parameters
    ----------
    product : dict
        Raw product data from the upstream data source.
    store_context : StoreContext
        Immutable store/locale context established at pipeline entry.
    llm_client : optional
        Injectable LLM client implementing the LLMClient protocol.
        When None, nodes use deterministic stubs (no LLM calls).
        See ``agent_dag.shared.llm_protocol.LLMClient`` for the interface.
    node_registry : dict, optional
        Injectable node implementations for testing. Keys are node names
        (e.g., "node1", "node2"), values are async callables.
    failure_store : optional
        Injectable failure store backend for audit trail persistence.
    flywheel : optional
        Injectable data flywheel for self-improvement loop integration.
    trace_id : str
        Observability trace ID for end-to-end tracing.

    Returns
    -------
    PipelineResult
        Contains the final content, metadata, audit trail, and flywheel stats.

    Architecture
    ------------
    Phase 1 — Parallel:
        Node 1 (Context Research) and Node 2 (Input Normalizer) execute
        concurrently. They have no data dependency on each other.

    Phase 2 — Sequential:
        Nodes 3-7 execute in strict order. Each node's output feeds
        directly into the next node's input.

    Fail-Closed:
        Any node failure at any phase triggers record_failure() and
        returns a PipelineResult with success=False. No partial content
        is ever propagated downstream.
    """
    pipeline_start = time.monotonic()
    product_id = product.get("id", product.get("sku", "unknown"))

    # ── Entry Checkpoint: Verify store context integrity ──
    try:
        assert_store_integrity(store_context, "ENTRY")
    except AssertionError as exc:
        record_failure(
            product_id, store_context.store_id,
            "NODE_0_ENTRY", FailureReason.INTEGRITY_VIOLATION,
            str(exc), retry_eligible=False, trace_id=trace_id,
            failure_store=failure_store,
        )
        return PipelineResult(
            success=False, product_id=product_id,
            failure_node="NODE_0_ENTRY",
            failure_reason=FailureReason.INTEGRITY_VIOLATION.value,
            duration_ms=_elapsed_ms(pipeline_start),
        )

    nodes = node_registry or _default_node_registry()

    # ════════════════════════════════════════════════════════════════════
    # PHASE 1 — Parallel Execution (Node 1 + Node 2)
    # ════════════════════════════════════════════════════════════════════
    try:
        node1_result, node2_result = await asyncio.gather(
            nodes["node1"](product, store_context),
            nodes["node2"](product, store_context),
        )
    except Exception as exc:
        failure = record_failure(
            product_id, store_context.store_id,
            "PHASE_1_PARALLEL", FailureReason.NODE_FAILURE,
            f"Phase 1 parallel execution failed: {exc}",
            trace_id=trace_id, failure_store=failure_store,
        )
        return PipelineResult(
            success=False, product_id=product_id,
            failure_node="PHASE_1_PARALLEL",
            failure_reason=FailureReason.NODE_FAILURE.value,
            duration_ms=_elapsed_ms(pipeline_start),
        )

    # ════════════════════════════════════════════════════════════════════
    # PHASE 2 — Sequential Execution (Nodes 3 → 4 → 5 → 6 → 7)
    # ════════════════════════════════════════════════════════════════════
    sequential_nodes = [
        ("NODE_3_SYNONYMS", "node3"),
        ("NODE_4_DEMAND_GATE", "node4"),
        ("NODE_5_CONTENT_GENERATOR", "node5"),
        ("NODE_6_QUALITY_VALIDATOR", "node6"),
        ("NODE_7_METADATA_EXTRACTOR", "node7"),
    ]

    state: dict[str, Any] = {
        "product": product,
        "store_context": store_context,
        "llm_client": llm_client,
        "node1_result": node1_result,
        "node2_result": node2_result,
    }

    for node_label, node_key in sequential_nodes:
        try:
            node_fn = nodes[node_key]
            result = await node_fn(state)

            # ── Node 4 special handling: demand gating ──
            if node_key == "node4":
                gate_decision = result.get("decision", GateDecision.PASS)
                if gate_decision == GateDecision.REJECT:
                    record_failure(
                        product_id, store_context.store_id,
                        node_label, FailureReason.REJECT_ZERO_DEMAND,
                        f"Demand gate rejected: {result.get('reason', 'no demand')}",
                        retry_eligible=False, trace_id=trace_id,
                        failure_store=failure_store,
                    )
                    return PipelineResult(
                        success=False, product_id=product_id,
                        failure_node=node_label,
                        failure_reason=FailureReason.REJECT_ZERO_DEMAND.value,
                        duration_ms=_elapsed_ms(pipeline_start),
                    )
                if gate_decision == GateDecision.DEFER:
                    record_failure(
                        product_id, store_context.store_id,
                        node_label, FailureReason.DEFERRED_QUOTA,
                        f"Demand gate deferred: {result.get('reason', 'quota exceeded')}",
                        retry_eligible=True, trace_id=trace_id,
                        failure_store=failure_store,
                    )
                    return PipelineResult(
                        success=False, product_id=product_id,
                        failure_node=node_label,
                        failure_reason=FailureReason.DEFERRED_QUOTA.value,
                        duration_ms=_elapsed_ms(pipeline_start),
                    )

            # ── Node 6 special handling: O-R-A-V quality gate ──
            if node_key == "node6":
                orav_decision = result.get("decision", "FAIL")
                if orav_decision != "PASS":
                    record_failure(
                        product_id, store_context.store_id,
                        node_label, FailureReason.ORAV_FAIL,
                        f"O-R-A-V quality gate failed: {result.get('errors', [])}",
                        retry_eligible=True, trace_id=trace_id,
                        failure_store=failure_store,
                    )
                    return PipelineResult(
                        success=False, product_id=product_id,
                        failure_node=node_label,
                        failure_reason=FailureReason.ORAV_FAIL.value,
                        duration_ms=_elapsed_ms(pipeline_start),
                    )

            state[f"{node_key}_result"] = result

        except Exception as exc:
            record_failure(
                product_id, store_context.store_id,
                node_label, FailureReason.NODE_FAILURE,
                f"{node_label} failed: {exc}",
                trace_id=trace_id, failure_store=failure_store,
            )
            return PipelineResult(
                success=False, product_id=product_id,
                failure_node=node_label,
                failure_reason=FailureReason.NODE_FAILURE.value,
                duration_ms=_elapsed_ms(pipeline_start),
            )

    # ── Exit Checkpoint: Verify store context integrity ──
    try:
        assert_store_integrity(store_context, "PRE_EXPORT")
    except AssertionError as exc:
        record_failure(
            product_id, store_context.store_id,
            "NODE_7_EXIT", FailureReason.INTEGRITY_VIOLATION,
            str(exc), retry_eligible=False, trace_id=trace_id,
            failure_store=failure_store,
        )
        return PipelineResult(
            success=False, product_id=product_id,
            failure_node="NODE_7_EXIT",
            failure_reason=FailureReason.INTEGRITY_VIOLATION.value,
            duration_ms=_elapsed_ms(pipeline_start),
        )

    # ════════════════════════════════════════════════════════════════════
    # DATA FLYWHEEL — Ingest successful run (fail-open, never blocks)
    # ════════════════════════════════════════════════════════════════════
    flywheel_stats: dict[str, int] = {}
    if flywheel is not None:
        try:
            flywheel_stats = await flywheel.ingest_pipeline_run(
                trace_id=trace_id,
                state=state,
                tier="production-baseline",
            )
        except Exception as exc:
            logger.debug("Data flywheel ingestion failed (non-fatal): %s", exc)

    duration_ms = _elapsed_ms(pipeline_start)
    logger.info(
        "PIPELINE_SUCCESS: product=%s store=%s duration=%.1fms flywheel=%s",
        product_id, store_context.store_id, duration_ms, flywheel_stats,
    )

    return PipelineResult(
        success=True,
        product_id=product_id,
        content=state.get("node5_result"),
        metadata=state.get("node7_result"),
        duration_ms=duration_ms,
        flywheel_stats=flywheel_stats,
    )


# ════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _elapsed_ms(start: float) -> float:
    """Calculate elapsed time in milliseconds since start."""
    return (time.monotonic() - start) * 1000


def _default_node_registry() -> dict[str, Any]:
    """Return the default node registry with lazy imports.

    Each node is imported lazily to allow the orchestrator to be
    instantiated without requiring all node dependencies.
    """
    from .nodes.node1_context_researcher import run_node1
    from .nodes.node2_input_normalizer import run_node2
    from .nodes.node3_synonym_generator import run_node3
    from .nodes.node4_demand_gatekeeper import run_node4
    from .nodes.node5_content_generator import run_node5
    from .nodes.node6_quality_validator import run_node6
    from .nodes.node7_metadata_extractor import run_node7

    return {
        "node1": run_node1,
        "node2": run_node2,
        "node3": run_node3,
        "node4": run_node4,
        "node5": run_node5,
        "node6": run_node6,
        "node7": run_node7,
    }
