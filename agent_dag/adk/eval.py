"""
ADK Evaluation Integration — Trajectory + O-R-A-V Quality.

Maps the pipeline's O-R-A-V scoring to ADK's evaluation framework,
enabling ``adk eval`` and pytest-based CI/CD evaluation.

Evaluation Modes:
    1. Trajectory — Verify correct node execution order
    2. Response Quality — LLM-as-Judge O-R-A-V scoring
    3. Golden Dataset — Deterministic assertions against reference data
"""

from __future__ import annotations

from typing import Any

# Expected node execution trajectory for a successful pipeline run
EXPECTED_TRAJECTORY_FULL: list[str] = [
    "context_researcher",
    "input_normalizer",
    "synonym_generator",
    "demand_gatekeeper",
    "content_generator",
    "quality_validator",
    "metadata_extractor",
]

# Minimum trajectory for a demand-gated rejection (valid early exit)
EXPECTED_TRAJECTORY_DEMAND_REJECT: list[str] = [
    "context_researcher",
    "input_normalizer",
    "synonym_generator",
    "demand_gatekeeper",
]

# O-R-A-V Judge prompt for LLM-as-Judge evaluation
ORAV_JUDGE_PROMPT: str = """You are an expert content quality evaluator.

Evaluate the following generated product content on 4 dimensions.
Score each dimension from 0.0 to 1.0.

Dimensions:
- **Originality** (O): Novel phrasing, avoids generic templates, creative language
- **Relevance** (R): Matches the product category and locale context accurately
- **Accuracy** (A): Factually correct, no hallucinations, no made-up features
- **Value** (V): Actionable for the end user, improves purchase decision

Product: {product_name}
Locale: {city}, {country}
Category: {category}

Generated Content:
{content}

Respond ONLY with JSON:
{{"originality": 0.0, "relevance": 0.0, "accuracy": 0.0, "value": 0.0}}
"""


def extract_trajectory(session_state: dict[str, Any]) -> list[str]:
    """Extract the actual node execution trajectory from session state.

    Parameters
    ----------
    session_state : dict
        ADK session state after pipeline execution.

    Returns
    -------
    list[str]
        Ordered list of node names that executed successfully.
    """
    trajectory: list[str] = []

    for node_name in EXPECTED_TRAJECTORY_FULL:
        gate_decision = session_state.get(f"{node_name}:gate_decision")
        success = session_state.get(f"{node_name}:success")

        if gate_decision is not None:
            trajectory.append(node_name)
            # Stop if node failed (fail-closed)
            if not success:
                break

    return trajectory


def validate_trajectory(
    actual: list[str],
    expected: list[str],
) -> dict[str, Any]:
    """Validate that the actual trajectory matches expected.

    Parameters
    ----------
    actual : list[str]
        Actual node execution order.
    expected : list[str]
        Expected node execution order.

    Returns
    -------
    dict
        Validation result with pass/fail and details.
    """
    passed = actual == expected
    return {
        "passed": passed,
        "expected": expected,
        "actual": actual,
        "missing": [n for n in expected if n not in actual],
        "extra": [n for n in actual if n not in expected],
    }


def compute_orav_score(scores: dict[str, float]) -> float:
    """Compute composite O-R-A-V score.

    Uses weighted average: Accuracy weighted highest (0.35),
    Relevance (0.30), Value (0.20), Originality (0.15).
    """
    weights = {
        "originality": 0.15,
        "relevance": 0.30,
        "accuracy": 0.35,
        "value": 0.20,
    }
    total = sum(
        scores.get(dim, 0.0) * weight
        for dim, weight in weights.items()
    )
    return round(total, 4)
