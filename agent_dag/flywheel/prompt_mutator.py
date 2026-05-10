"""
Prompt Mutator — Runtime Prompt Adaptation (Hebbian Feedback Loop).

Implements the fast feedback mechanism where O-R-A-V evaluation
signals mutate the per-tenant prompt cache WITHOUT waiting for a
full LoRA retraining cycle.

This is the "Self-Improving RL Feedback" loop shown in the architecture:
    Ingest → Generate → Evaluate → Score → **Mutate** → Recycle

The mutation mechanism mimics Hebbian learning:
    - Connections that fire together (good prompts + high scores) strengthen
    - Connections that fail together (bad prompts + low scores) weaken
    - Mutations are scoped per-tenant to prevent cross-contamination

Mutation Types:
    1. APPEND_CONSTRAINT: Add a new negative constraint to the prompt
       (e.g., "Do not use transactional language")
    2. BOOST_EXAMPLE: Inject a high-scoring example as a few-shot
    3. ADJUST_TEMPERATURE: Reduce temperature for high-failure-rate prompts
    4. SWAP_TEMPLATE: Switch to an alternative template variant
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class MutationType(StrEnum):
    """Types of prompt mutations."""
    APPEND_CONSTRAINT = "APPEND_CONSTRAINT"
    BOOST_EXAMPLE = "BOOST_EXAMPLE"
    ADJUST_TEMPERATURE = "ADJUST_TEMPERATURE"
    SWAP_TEMPLATE = "SWAP_TEMPLATE"


@dataclass
class PromptMutation:
    """Record of a single prompt mutation applied to the cache.

    Attributes
    ----------
    tenant_id : str
        The tenant whose prompt was mutated.
    prompt_id : str
        The specific prompt that was mutated.
    mutation_type : MutationType
        The type of mutation applied.
    mutation_detail : str
        Human-readable description of the mutation.
    trigger_score : float
        The O-R-A-V score that triggered this mutation.
    trigger_trace_id : str
        The trace ID of the evaluation that triggered the mutation.
    timestamp : str
        When the mutation was applied.
    """
    tenant_id: str
    prompt_id: str
    mutation_type: MutationType
    mutation_detail: str
    trigger_score: float = 0.0
    trigger_trace_id: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "prompt_id": self.prompt_id,
            "mutation_type": self.mutation_type.value,
            "detail": self.mutation_detail,
            "trigger_score": self.trigger_score,
            "trigger_trace_id": self.trigger_trace_id,
            "timestamp": self.timestamp,
        }


class PromptMutator:
    """Applies evaluation-driven mutations to the per-tenant prompt cache.

    Usage::

        mutator = PromptMutator(prompt_cache=redis_cache)
        mutation = mutator.apply_feedback(
            tenant_id="store_123",
            prompt_id="full_description",
            orav_score=0.42,
            failure_reasons=["TRANSACTIONAL_LANGUAGE"],
            trace_id="abc-123",
        )
    """

    # Failure patterns → mutation rules (deterministic mapping)
    FAILURE_MUTATION_MAP: dict[str, tuple[MutationType, str]] = {
        "LABEL_LEAK": (
            MutationType.APPEND_CONSTRAINT,
            "CRITICAL: Never include internal variable names, field labels, "
            "or system context in generated content."
        ),
        "TRANSACTIONAL": (
            MutationType.APPEND_CONSTRAINT,
            "Do NOT use transactional or sales language (buy now, shop now, "
            "order today, limited time, etc.)."
        ),
        "TEMPLATE_VAR_LEAK": (
            MutationType.APPEND_CONSTRAINT,
            "Ensure ALL template variables are replaced. Never output "
            "{variable_name} patterns in final content."
        ),
        "LOW_ORIGINALITY": (
            MutationType.ADJUST_TEMPERATURE,
            "Increase creativity. Avoid repeating common patterns from "
            "previous generations."
        ),
        "LOW_RELEVANCE": (
            MutationType.BOOST_EXAMPLE,
            "Focus tightly on the product's specific attributes and the "
            "local context. Avoid generic descriptions."
        ),
    }

    def __init__(
        self,
        *,
        prompt_cache: Any | None = None,
        mutation_history: Any | None = None,
        max_constraints_per_prompt: int = 5,
    ) -> None:
        self._cache = prompt_cache
        self._history = mutation_history
        self._max_constraints = max_constraints_per_prompt

    def apply_feedback(
        self,
        tenant_id: str,
        prompt_id: str,
        orav_score: float,
        failure_reasons: list[str],
        trace_id: str = "",
    ) -> list[PromptMutation]:
        """Apply evaluation feedback as prompt mutations.

        Parameters
        ----------
        tenant_id : str
            The tenant whose prompt to mutate.
        prompt_id : str
            The specific prompt template to mutate.
        orav_score : float
            The O-R-A-V composite score that triggered the mutation.
        failure_reasons : list of str
            Specific failure reasons from Node 6 diagnostics.
        trace_id : str
            The trace ID for observability correlation.

        Returns
        -------
        list of PromptMutation
            All mutations applied in this feedback cycle.
        """
        mutations: list[PromptMutation] = []

        for reason in failure_reasons:
            # ── Normalize reason to mutation rule ──
            rule_key = None
            for key in self.FAILURE_MUTATION_MAP:
                if key in reason.upper():
                    rule_key = key
                    break

            if rule_key is None:
                continue

            mutation_type, mutation_detail = self.FAILURE_MUTATION_MAP[rule_key]

            mutation = PromptMutation(
                tenant_id=tenant_id,
                prompt_id=prompt_id,
                mutation_type=mutation_type,
                mutation_detail=mutation_detail,
                trigger_score=orav_score,
                trigger_trace_id=trace_id,
            )

            # ── Apply mutation to cache ──
            if self._cache is not None:
                cache_key = f"prompt:{tenant_id}:{prompt_id}:constraints"
                try:
                    self._cache.append(cache_key, mutation_detail)
                except Exception as exc:
                    logger.debug("Prompt cache mutation failed (non-fatal): %s", exc)

            # ── Record mutation history ──
            if self._history is not None:
                try:
                    self._history.write(mutation.to_dict())
                except Exception as exc:
                    logger.debug("Mutation history write failed (non-fatal): %s", exc)

            mutations.append(mutation)
            logger.info(
                "PROMPT_MUTATION: tenant=%s prompt=%s type=%s score=%.2f",
                tenant_id, prompt_id, mutation_type.value, orav_score,
            )

        return mutations
