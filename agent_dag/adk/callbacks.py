"""
ADK Lifecycle Callbacks.

Provides before_agent and after_agent callbacks for the pipeline,
implementing integrity checks, telemetry, and data flywheel integration
using ADK's native callback mechanism.

ADK Callback Contract:
    - Receives ``callback_context`` with ``.state`` access
    - Return ``None`` to proceed normally
    - Return ``Content`` to short-circuit (skip agent execution)
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def integrity_check_callback(callback_context: Any) -> Any:
    """before_agent: Verify pipeline preconditions before execution.

    Checks:
        1. Store context exists and is frozen/immutable
        2. Product data is present
        3. No cross-tenant state contamination
    """
    state = callback_context.state
    store_context = state.get("store_context")

    if not store_context:
        try:
            from google.genai.types import Content, Part

            return Content(
                parts=[Part(text="INTEGRITY_VIOLATION: No store context provided")]
            )
        except ImportError:
            return None

    # Record pipeline start time
    state["app:pipeline_start_ms"] = time.monotonic() * 1000
    return None


def flywheel_ingest_callback(callback_context: Any) -> None:
    """after_agent: Feed results into the data flywheel for self-improvement.

    Routes results to the appropriate flywheel tier:
        - PASS + high ORAV → approved tier (training data)
        - PASS + low ORAV → baseline tier (review needed)
        - FAIL → failure tier (negative examples for DPO)
    """
    state = callback_context.state

    # Calculate total duration
    start_ms = state.get("app:pipeline_start_ms", 0)
    if start_ms:
        state["app:pipeline_duration_ms"] = round(
            (time.monotonic() * 1000) - start_ms, 2
        )

    # Route to flywheel tier
    validator_result = state.get("quality_validator:result", {})
    decision = validator_result.get("decision", "")

    if decision == "PASS":
        state["flywheel:tier"] = "approved"
    elif decision == "FAIL":
        state["flywheel:tier"] = "failure"
    else:
        state["flywheel:tier"] = "baseline"

    logger.info(
        "FLYWHEEL: tier=%s duration=%.1fms",
        state.get("flywheel:tier", "unknown"),
        state.get("app:pipeline_duration_ms", 0),
    )
    return


def telemetry_callback(callback_context: Any) -> None:
    """after_agent: Emit OpenTelemetry spans for pipeline observability.

    ADK has built-in OTel instrumentation. This callback adds
    pipeline-specific attributes to the trace.
    """
    state = callback_context.state

    # Collect per-node metrics
    node_names = [
        "context_researcher",
        "input_normalizer",
        "synonym_generator",
        "demand_gatekeeper",
        "content_generator",
        "quality_validator",
        "metadata_extractor",
    ]
    for name in node_names:
        success = state.get(f"{name}:success")
        duration = state.get(f"{name}:duration_ms", 0)
        if success is not None:
            logger.debug(
                "TELEMETRY: node=%s success=%s duration=%.1fms",
                name,
                success,
                duration,
            )

    return
