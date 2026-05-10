"""Tests for ADK evaluation utilities."""

from __future__ import annotations

from agent_dag.adk.eval import (
    EXPECTED_TRAJECTORY_DEMAND_REJECT,
    EXPECTED_TRAJECTORY_FULL,
    compute_orav_score,
    extract_trajectory,
    validate_trajectory,
)


class TestExtractTrajectory:
    """Test trajectory extraction from session state."""

    def test_full_trajectory(self):
        state = {
            "context_researcher:gate_decision": "PASS",
            "context_researcher:success": True,
            "input_normalizer:gate_decision": "PASS",
            "input_normalizer:success": True,
            "synonym_generator:gate_decision": "PASS",
            "synonym_generator:success": True,
            "demand_gatekeeper:gate_decision": "PASS",
            "demand_gatekeeper:success": True,
            "content_generator:gate_decision": "PASS",
            "content_generator:success": True,
            "quality_validator:gate_decision": "PASS",
            "quality_validator:success": True,
            "metadata_extractor:gate_decision": "PASS",
            "metadata_extractor:success": True,
        }
        trajectory = extract_trajectory(state)
        assert trajectory == EXPECTED_TRAJECTORY_FULL

    def test_demand_rejection(self):
        state = {
            "context_researcher:gate_decision": "PASS",
            "context_researcher:success": True,
            "input_normalizer:gate_decision": "PASS",
            "input_normalizer:success": True,
            "synonym_generator:gate_decision": "PASS",
            "synonym_generator:success": True,
            "demand_gatekeeper:gate_decision": "REJECT",
            "demand_gatekeeper:success": False,
        }
        trajectory = extract_trajectory(state)
        assert trajectory == EXPECTED_TRAJECTORY_DEMAND_REJECT

    def test_empty_state(self):
        trajectory = extract_trajectory({})
        assert trajectory == []


class TestValidateTrajectory:
    """Test trajectory validation."""

    def test_matching(self):
        result = validate_trajectory(
            EXPECTED_TRAJECTORY_FULL,
            EXPECTED_TRAJECTORY_FULL,
        )
        assert result["passed"] is True
        assert result["missing"] == []
        assert result["extra"] == []

    def test_missing_nodes(self):
        result = validate_trajectory(
            EXPECTED_TRAJECTORY_FULL[:3],
            EXPECTED_TRAJECTORY_FULL,
        )
        assert result["passed"] is False
        assert len(result["missing"]) == 4


class TestComputeORAVScore:
    """Test O-R-A-V composite scoring."""

    def test_perfect_score(self):
        score = compute_orav_score({
            "originality": 1.0,
            "relevance": 1.0,
            "accuracy": 1.0,
            "value": 1.0,
        })
        assert score == 1.0

    def test_zero_score(self):
        score = compute_orav_score({
            "originality": 0.0,
            "relevance": 0.0,
            "accuracy": 0.0,
            "value": 0.0,
        })
        assert score == 0.0

    def test_accuracy_weighted_highest(self):
        """Accuracy has highest weight (0.35)."""
        high_accuracy = compute_orav_score({
            "originality": 0.0,
            "relevance": 0.0,
            "accuracy": 1.0,
            "value": 0.0,
        })
        high_originality = compute_orav_score({
            "originality": 1.0,
            "relevance": 0.0,
            "accuracy": 0.0,
            "value": 0.0,
        })
        assert high_accuracy > high_originality

    def test_partial_scores(self):
        score = compute_orav_score({
            "originality": 0.7,
            "relevance": 0.8,
            "accuracy": 0.9,
            "value": 0.6,
        })
        # 0.7*0.15 + 0.8*0.30 + 0.9*0.35 + 0.6*0.20 = 0.105+0.24+0.315+0.12 = 0.78
        assert score == 0.78
