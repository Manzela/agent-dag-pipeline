"""
Preference Pairs — DPO-Compatible Training Data Generation.

Generates (chosen, rejected) preference pairs from the data flywheel's
quality-approved vs. failure-cases tiers for Direct Preference Optimization.

This is the bridge between "we collect scored data" and "we use it for
alignment" — implementing the RLAIF methodology used in Anthropic's
Constitutional AI for continuous model improvement.

Algorithm:
    1. Query quality-approved tier for high-scoring examples (chosen)
    2. Query failure-cases tier for matching low-scoring examples (rejected)
    3. Match pairs by prompt_id (same prompt, different quality outputs)
    4. Apply margin-maximization filtering (score gap ≥ threshold)
    5. Output DPO-compatible training pairs with metadata

Margin Maximization:
    Not all (chosen, rejected) pairs are equally informative. Pairs with
    a larger score margin provide stronger training signal. We filter
    pairs below a configurable margin threshold to optimize training
    data quality over quantity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreferencePair:
    """A single DPO-compatible preference pair for training.

    Attributes
    ----------
    prompt : str
        The shared prompt/context that produced both outputs.
    chosen : str
        The high-quality output (from quality-approved tier).
    rejected : str
        The low-quality output (from failure-cases tier).
    chosen_score : float
        O-R-A-V composite score of the chosen output.
    rejected_score : float
        O-R-A-V composite score of the rejected output.
    margin : float
        Score difference (chosen_score - rejected_score).
    prompt_id : str
        The specific prompt template that generated both outputs.
    tenant_id : str
        The tenant context for this pair.
    metadata : dict
        Additional training metadata (model version, pipeline version).
    """
    prompt: str
    chosen: str
    rejected: str
    chosen_score: float
    rejected_score: float
    margin: float
    prompt_id: str = ""
    tenant_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dpo_format(self) -> dict[str, str]:
        """Convert to standard DPO training format.

        Returns a dictionary compatible with TRL's DPOTrainer:
            {"prompt": ..., "chosen": ..., "rejected": ...}
        """
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
        }


class PreferencePairGenerator:
    """Generates DPO preference pairs from flywheel datasets.

    Usage::

        generator = PreferencePairGenerator(margin_threshold=0.2)
        pairs = generator.generate_pairs(approved_items, failure_items)
    """

    def __init__(
        self,
        *,
        margin_threshold: float = 0.15,
        max_pairs_per_prompt: int = 10,
    ) -> None:
        self._margin_threshold = margin_threshold
        self._max_pairs_per_prompt = max_pairs_per_prompt

    def generate_pairs(
        self,
        approved_items: Sequence[dict[str, Any]],
        failure_items: Sequence[dict[str, Any]],
    ) -> list[PreferencePair]:
        """Generate preference pairs from approved and failure datasets.

        Parameters
        ----------
        approved_items : sequence of dicts
            Items from the quality-approved tier (chosen candidates).
        failure_items : sequence of dicts
            Items from the failure-cases tier (rejected candidates).

        Returns
        -------
        list of PreferencePair
            Filtered, margin-maximized preference pairs.
        """
        # ── Index failure items by prompt_id for O(1) lookup ──
        failures_by_prompt: dict[str, list[dict[str, Any]]] = {}
        for item in failure_items:
            pid = item.get("prompt_id", "")
            failures_by_prompt.setdefault(pid, []).append(item)

        pairs: list[PreferencePair] = []

        for approved in approved_items:
            prompt_id = approved.get("prompt_id", "")
            matched_failures = failures_by_prompt.get(prompt_id, [])

            for failure in matched_failures[:self._max_pairs_per_prompt]:
                chosen_score = approved.get("scores", {}).get("orav_quality", 0.0)
                rejected_score = failure.get("scores", {}).get("orav_quality", 0.0)
                margin = chosen_score - rejected_score

                if margin < self._margin_threshold:
                    continue

                pair = PreferencePair(
                    prompt=str(approved.get("input", {}).get("context", "")),
                    chosen=str(approved.get("output", "")),
                    rejected=str(failure.get("output", "")),
                    chosen_score=chosen_score,
                    rejected_score=rejected_score,
                    margin=margin,
                    prompt_id=prompt_id,
                    tenant_id=approved.get("tenant_id", ""),
                    metadata={
                        "pipeline_version": approved.get("pipeline_version", ""),
                        "model_version": approved.get("model_version", ""),
                    },
                )
                pairs.append(pair)

        # ── Sort by margin (descending) for strongest training signal first ──
        pairs.sort(key=lambda p: p.margin, reverse=True)

        logger.info(
            "Generated %d preference pairs from %d approved × %d failures "
            "(margin threshold=%.2f)",
            len(pairs), len(approved_items), len(failure_items),
            self._margin_threshold,
        )

        return pairs


def export_dpo_dataset(
    pairs: Sequence[PreferencePair],
    *,
    output_format: str = "jsonl",
) -> list[dict[str, str]]:
    """Export preference pairs to DPO-compatible training format.

    Parameters
    ----------
    pairs : sequence of PreferencePair
        Generated preference pairs.
    output_format : str
        Output format. Currently supports "jsonl".

    Returns
    -------
    list of dicts
        DPO-formatted training examples.
    """
    return [pair.to_dpo_format() for pair in pairs]
