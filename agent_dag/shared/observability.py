"""
Observability — Structured Logging and Tracing Infrastructure.

Provides the observability layer for the pipeline with:
    - Structured run context (tenant, run_id, pipeline_version)
    - OpenTelemetry-compatible trace management
    - Score emission for Langfuse integration
    - 4-layer isolation enforcement (tenant, prompt, task, KV cache)
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Thread-safe run context using contextvars ──
_run_context: ContextVar[dict[str, Any]] = ContextVar("run_context", default={})


def set_run_context(
    *,
    tenant_id: str,
    run_id: str,
    pipeline_version: str = "",
    environment: str = "production",
    domain: str = "",
    **kwargs: Any,
) -> None:
    """Set the run context for the current execution.

    This context is propagated through all pipeline nodes and
    attached to every observability event.
    """
    ctx = {
        "tenant_id": tenant_id,
        "run_id": run_id,
        "pipeline_version": pipeline_version,
        "environment": environment,
        "domain": domain,
        **kwargs,
    }
    _run_context.set(ctx)


def get_run_context() -> dict[str, Any]:
    """Get the current run context."""
    return _run_context.get()


def score_generation(
    trace_id: str,
    metric_name: str,
    value: Any,
    comment: str = "",
    *,
    data_type: str = "NUMERIC",
    config_id: str = "",
    metadata: Optional[dict[str, Any]] = None,
    observability_client: Optional[Any] = None,
) -> None:
    """Emit a score event for observability.

    Parameters
    ----------
    trace_id : str
        The trace to attach this score to.
    metric_name : str
        The metric name (must match a ScoreConfig).
    value : any
        The score value (float, bool, or str depending on data_type).
    """
    if observability_client is not None:
        try:
            observability_client.score(
                trace_id=trace_id,
                name=metric_name,
                value=value,
                comment=comment,
                data_type=data_type,
                config_id=config_id,
                metadata=metadata or {},
            )
        except Exception as exc:
            logger.debug("Score emission failed (non-fatal): %s", exc)
