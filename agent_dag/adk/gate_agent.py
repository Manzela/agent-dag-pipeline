"""
GateAgent — ADK-Native Gate-Agent Pattern.

Maps the pipeline's dual-layer architecture (deterministic gate +
probabilistic agent) onto Google ADK's ``BaseAgent`` class hierarchy.

Every pipeline node extends ``GateAgent`` and implements:
    - ``gate(state)`` → O(1) deterministic check, zero LLM cost
    - ``agent(state, ctx)`` → Probabilistic LLM execution

The ADK runtime calls ``_run_async_impl`` which orchestrates
gate → agent → Event yield, with full lifecycle hook support.

ADK Integration Points:
    - ``before_agent_callback`` fires before gate()
    - ``after_agent_callback`` fires after agent()
    - ``InvocationContext.session.state`` provides shared state
    - ``Event`` objects stream results to the Runner
"""

from __future__ import annotations

import enum
import logging
import time
from abc import abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Conditional import: ADK is optional (standalone mode works without it)
try:
    from google.adk.agents import BaseAgent
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.events.event import Event
    from google.genai.types import Content, Part

    _ADK_AVAILABLE = True
except ImportError:
    # Fallback: define minimal stubs so the module can be imported
    # without google-adk installed (for type checking, testing, etc.)
    from pydantic import BaseModel as BaseAgent  # type: ignore[assignment]

    InvocationContext = Any  # type: ignore[misc,assignment]
    Event = Any  # type: ignore[misc,assignment]
    _ADK_AVAILABLE = False


class GateDecision(enum.Enum):
    """Deterministic gate decisions."""

    PASS = "PASS"
    REJECT = "REJECT"
    DEFER = "DEFER"


@dataclass(frozen=True)
class GateResult:
    """Result of a deterministic gate check."""

    decision: GateDecision
    reason: str = ""
    metadata: dict[str, Any] | None = None


class GateAgent(BaseAgent):
    """ADK-native base class for all pipeline nodes.

    Implements the Gate-Agent pattern on top of Google ADK's BaseAgent:

    1. ``gate(state)`` — Deterministic O(1) check. Returns GateResult.
       If REJECT or DEFER, the agent step is skipped entirely (zero LLM cost).

    2. ``agent(state, ctx)`` — Probabilistic LLM-powered step. Only called
       when the gate returns PASS.

    3. ``_run_async_impl(ctx)`` — ADK entry point. Orchestrates
       gate → agent → Event yield with state management.

    Subclass Contract::

        class MyNode(GateAgent):
            name: str = "my_node"
            description: str = "What this node does"

            def gate(self, state: dict) -> GateResult:
                if not valid(state):
                    return GateResult(GateDecision.REJECT, "reason")
                return GateResult(GateDecision.PASS)

            async def agent(self, state: dict, ctx) -> dict:
                return {"output": await llm_call(...)}
    """

    # Subclasses MUST override these
    @abstractmethod
    def gate(self, state: dict[str, Any]) -> GateResult:
        """Deterministic gate — O(1), no LLM cost.

        Parameters
        ----------
        state : dict
            The shared session state (``ctx.session.state`` in ADK,
            or the orchestrator state dict in standalone mode).

        Returns
        -------
        GateResult
            PASS to proceed to agent(), REJECT or DEFER to skip.
        """
        ...

    @abstractmethod
    async def agent(
        self,
        state: dict[str, Any],
        ctx: Any,
    ) -> dict[str, Any]:
        """Probabilistic agent — LLM-powered step.

        Only called when gate() returns PASS.

        Parameters
        ----------
        state : dict
            Shared session state.
        ctx : InvocationContext
            ADK invocation context (or None in standalone mode).

        Returns
        -------
        dict
            Node output to store in state.
        """
        ...

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """ADK entry point — orchestrates gate → agent → Event.

        This method is called by the ADK Runner. It:
        1. Reads shared state from ``ctx.session.state``
        2. Executes the deterministic gate
        3. If PASS, executes the probabilistic agent
        4. Writes results back to state
        5. Yields an Event for the ADK runtime
        """
        if not _ADK_AVAILABLE:
            raise RuntimeError(
                "Google ADK is not installed. Install with: pip install google-adk"
            )

        state = ctx.session.state
        node_name = self.name
        start_ms = time.monotonic() * 1000

        # ── Step 1: Deterministic Gate ──
        gate_result = self.gate(state)

        state[f"{node_name}:gate_decision"] = gate_result.decision.value
        state[f"{node_name}:gate_reason"] = gate_result.reason

        if gate_result.decision != GateDecision.PASS:
            state[f"{node_name}:success"] = False
            duration_ms = (time.monotonic() * 1000) - start_ms
            state[f"{node_name}:duration_ms"] = round(duration_ms, 2)

            logger.info(
                "GATE_%s: node=%s reason=%s (%.1fms)",
                gate_result.decision.value,
                node_name,
                gate_result.reason,
                duration_ms,
            )

            yield Event(
                invocation_id=ctx.invocation_id,
                author=node_name,
                content=Content(
                    parts=[
                        Part(
                            text=f"GATE_{gate_result.decision.value}: {gate_result.reason}"
                        )
                    ]
                ),
            )
            return

        # ── Step 2: Probabilistic Agent ──
        try:
            result = await self.agent(state, ctx)
            state[f"{node_name}:result"] = result
            state[f"{node_name}:success"] = True
        except Exception as exc:
            state[f"{node_name}:success"] = False
            state[f"{node_name}:error"] = str(exc)[:500]
            logger.error("AGENT_ERROR: node=%s error=%s", node_name, exc)
            raise

        duration_ms = (time.monotonic() * 1000) - start_ms
        state[f"{node_name}:duration_ms"] = round(duration_ms, 2)

        logger.info(
            "NODE_COMPLETE: node=%s success=True (%.1fms)",
            node_name,
            duration_ms,
        )

        yield Event(
            invocation_id=ctx.invocation_id,
            author=node_name,
        )
