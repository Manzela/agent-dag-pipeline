"""
Test Suite — Pipeline Orchestrator and Node Integration Tests.

Tests cover:
    - Fail-closed policy enforcement
    - Gate-Agent pattern for each node
    - O-R-A-V quality gate decisions
    - Data flywheel tier routing
    - Store context integrity assertions
    - Preference pair generation
    - Prompt mutation application
"""

from __future__ import annotations

import pytest

from agent_dag.flywheel.data_flywheel import (
    CurationDecision,
    DatasetItem,
    DatasetTier,
    route_to_tier,
)
from agent_dag.flywheel.preference_pairs import PreferencePairGenerator
from agent_dag.flywheel.prompt_mutator import MutationType, PromptMutator
from agent_dag.flywheel.score_configs import SCORE_CONFIGS, ScoreDataType
from agent_dag.nodes.node1_context_researcher import gate_locale_resolver
from agent_dag.nodes.node2_input_normalizer import _coerce_str, gate_schema_validator
from agent_dag.nodes.node3_synonym_generator import gate_dedup_filter
from agent_dag.nodes.node4_demand_gatekeeper import gate_threshold_comparator
from agent_dag.nodes.node6_quality_validator import gate_format_compliance
from agent_dag.nodes.node7_metadata_extractor import gate_embedding_dimension_check
from agent_dag.orchestrator import (
    CAUSAL_TRACES,
    FailureReason,
    assert_store_integrity,
    record_failure,
)
from agent_dag.shared.data_contracts import (
    ContentBlockOutput,
    GateDecision,
    StoreContext,
)

# ════════════════════════════════════════════════════════════════════════════
# STORE CONTEXT INTEGRITY
# ════════════════════════════════════════════════════════════════════════════

class TestStoreContextIntegrity:
    """Tests for fail-closed store context integrity assertions."""

    def test_frozen_context_passes(self) -> None:
        ctx = StoreContext(store_id="store_001", city="Madrid", country_code="ES")
        assert_store_integrity(ctx, "TEST")

    def test_context_immutability(self) -> None:
        ctx = StoreContext(store_id="store_001", city="Madrid", country_code="ES")
        with pytest.raises(Exception):
            ctx.city = "Barcelona"  # type: ignore[misc]


# ════════════════════════════════════════════════════════════════════════════
# NODE GATES (Deterministic, O(1))
# ════════════════════════════════════════════════════════════════════════════

class TestNodeGates:
    """Tests for all deterministic gates in the Gate-Agent pattern."""

    def test_node1_gate_valid_locale(self) -> None:
        ctx = StoreContext(store_id="s1", city="Madrid", country_code="ES")
        assert gate_locale_resolver(ctx) is True

    def test_node1_gate_missing_country(self) -> None:
        ctx = StoreContext(store_id="s1", city="Madrid", country_code="")
        assert gate_locale_resolver(ctx) is False

    def test_node1_gate_unsupported_locale(self) -> None:
        ctx = StoreContext(store_id="s1", city="Tokyo", country_code="JP")
        assert gate_locale_resolver(ctx) is False

    def test_node2_gate_valid_product(self) -> None:
        product = {"product-name": "Samsung Galaxy S24"}
        assert gate_schema_validator(product) is True

    def test_node2_gate_empty_product(self) -> None:
        product = {"sku": "12345"}
        assert gate_schema_validator(product) is False

    def test_node3_dedup_removes_duplicates(self) -> None:
        terms = ["smartphone", "Smartphone", "SMARTPHONE", "tablet"]
        result = gate_dedup_filter(terms)
        assert len(result) == 2
        assert "tablet" in [t.lower() for t in result]

    def test_node4_gate_pass(self) -> None:
        assert gate_threshold_comparator(100) == GateDecision.PASS

    def test_node4_gate_reject_zero(self) -> None:
        assert gate_threshold_comparator(0) == GateDecision.REJECT

    def test_node4_gate_reject_below_threshold(self) -> None:
        assert gate_threshold_comparator(5) == GateDecision.REJECT

    def test_node7_gate_valid_vector(self) -> None:
        vector = [0.1] * 768
        assert gate_embedding_dimension_check(vector) is True

    def test_node7_gate_wrong_dimension(self) -> None:
        vector = [0.1] * 512
        assert gate_embedding_dimension_check(vector) is False

    def test_node7_gate_nan_values(self) -> None:
        vector = [float("nan")] + [0.1] * 767
        assert gate_embedding_dimension_check(vector) is False


# ════════════════════════════════════════════════════════════════════════════
# NODE 6: O-R-A-V QUALITY GATE
# ════════════════════════════════════════════════════════════════════════════

class TestORAVQualityGate:
    """Tests for the O-R-A-V quality validation gate."""

    def test_clean_content_passes(self) -> None:
        content = ContentBlockOutput(
            full_description="A great product for your home.",
            meta_title="Great Product Title",
        )
        result = gate_format_compliance(content)
        assert len(result["errors"]) == 0

    def test_label_leak_detected(self) -> None:
        content = ContentBlockOutput(
            full_description="The sacred context of this product is amazing.",
        )
        result = gate_format_compliance(content)
        assert any("LABEL_LEAK" in e for e in result["errors"])

    def test_transactional_language_detected(self) -> None:
        content = ContentBlockOutput(
            full_description="Buy now and get free shipping on this amazing product!",
        )
        result = gate_format_compliance(content)
        assert any("TRANSACTIONAL" in e for e in result["errors"])

    def test_template_variable_leak_detected(self) -> None:
        content = ContentBlockOutput(
            meta_title="Best {product_name} in {city}",
        )
        result = gate_format_compliance(content)
        assert any("TEMPLATE_VAR_LEAK" in e for e in result["errors"])


# ════════════════════════════════════════════════════════════════════════════
# DATA FLYWHEEL: 3-Tier Curation
# ════════════════════════════════════════════════════════════════════════════

class TestDataFlywheel:
    """Tests for the 3-tier dataset curation engine."""

    def test_high_quality_routes_to_approved(self) -> None:
        item = DatasetItem(
            trace_id="t1", tenant_id="s1", prompt_id="full_description",
            input_payload={}, output_content={},
            scores={"orav_quality": 0.85, "demas_jit_verdict": "PASS"},
        )
        routed = route_to_tier(item)
        assert routed.tier == DatasetTier.QUALITY_APPROVED
        assert routed.curation_decision == CurationDecision.APPROVE

    def test_low_quality_routes_to_failures(self) -> None:
        item = DatasetItem(
            trace_id="t2", tenant_id="s1", prompt_id="meta_title",
            input_payload={}, output_content={},
            scores={"orav_quality": 0.3, "demas_jit_verdict": "FAIL"},
        )
        routed = route_to_tier(item)
        assert routed.tier == DatasetTier.FAILURE_CASES
        assert routed.curation_decision == CurationDecision.REJECT

    def test_medium_quality_stays_baseline(self) -> None:
        item = DatasetItem(
            trace_id="t3", tenant_id="s1", prompt_id="short_description",
            input_payload={}, output_content={},
            scores={"orav_quality": 0.6, "demas_jit_verdict": "PASS"},
        )
        routed = route_to_tier(item)
        assert routed.tier == DatasetTier.PRODUCTION_BASELINE


# ════════════════════════════════════════════════════════════════════════════
# SCORE CONFIGS
# ════════════════════════════════════════════════════════════════════════════

class TestScoreConfigs:
    """Tests for the multi-dimensional scoring engine."""

    def test_minimum_15_metrics(self) -> None:
        assert len(SCORE_CONFIGS) >= 15

    def test_all_orav_dimensions_present(self) -> None:
        orav_ids = [k for k in SCORE_CONFIGS if k.startswith("orav_")]
        assert "orav_quality" in orav_ids
        assert "orav_originality" in orav_ids
        assert "orav_relevance" in orav_ids
        assert "orav_accuracy" in orav_ids
        assert "orav_value" in orav_ids

    def test_all_data_types_used(self) -> None:
        types_used = {c.data_type for c in SCORE_CONFIGS.values()}
        assert ScoreDataType.NUMERIC in types_used
        assert ScoreDataType.BOOLEAN in types_used
        assert ScoreDataType.CATEGORICAL in types_used


# ════════════════════════════════════════════════════════════════════════════
# PREFERENCE PAIRS
# ════════════════════════════════════════════════════════════════════════════

class TestPreferencePairs:
    """Tests for DPO preference pair generation."""

    def test_generates_valid_pairs(self) -> None:
        approved = [
            {"prompt_id": "full_description", "scores": {"orav_quality": 0.9},
             "input": {"context": "test"}, "output": "Good content"},
        ]
        failures = [
            {"prompt_id": "full_description", "scores": {"orav_quality": 0.3},
             "input": {"context": "test"}, "output": "Bad content"},
        ]
        gen = PreferencePairGenerator(margin_threshold=0.15)
        pairs = gen.generate_pairs(approved, failures)
        assert len(pairs) == 1
        assert pairs[0].margin == pytest.approx(0.6)
        assert pairs[0].chosen_score > pairs[0].rejected_score

    def test_filters_low_margin_pairs(self) -> None:
        approved = [
            {"prompt_id": "meta_title", "scores": {"orav_quality": 0.55},
             "input": {}, "output": "OK"},
        ]
        failures = [
            {"prompt_id": "meta_title", "scores": {"orav_quality": 0.45},
             "input": {}, "output": "Meh"},
        ]
        gen = PreferencePairGenerator(margin_threshold=0.15)
        pairs = gen.generate_pairs(approved, failures)
        assert len(pairs) == 0  # margin 0.1 < threshold 0.15

    def test_dpo_format_output(self) -> None:
        approved = [
            {"prompt_id": "full_description", "scores": {"orav_quality": 0.9},
             "input": {"context": "ctx"}, "output": "chosen"},
        ]
        failures = [
            {"prompt_id": "full_description", "scores": {"orav_quality": 0.2},
             "input": {"context": "ctx"}, "output": "rejected"},
        ]
        gen = PreferencePairGenerator()
        pairs = gen.generate_pairs(approved, failures)
        dpo = pairs[0].to_dpo_format()
        assert "prompt" in dpo
        assert "chosen" in dpo
        assert "rejected" in dpo


# ════════════════════════════════════════════════════════════════════════════
# PROMPT MUTATOR
# ════════════════════════════════════════════════════════════════════════════

class TestPromptMutator:
    """Tests for the Hebbian prompt mutation engine."""

    def test_label_leak_triggers_constraint(self) -> None:
        mutator = PromptMutator()
        mutations = mutator.apply_feedback(
            tenant_id="store_001",
            prompt_id="full_description",
            orav_score=0.4,
            failure_reasons=["LABEL_LEAK: 'sacred context' in full_description"],
        )
        assert len(mutations) == 1
        assert mutations[0].mutation_type == MutationType.APPEND_CONSTRAINT

    def test_transactional_triggers_constraint(self) -> None:
        mutator = PromptMutator()
        mutations = mutator.apply_feedback(
            tenant_id="store_001",
            prompt_id="meta_description",
            orav_score=0.35,
            failure_reasons=["TRANSACTIONAL: 'buy now' in meta_description"],
        )
        assert len(mutations) == 1
        assert mutations[0].mutation_type == MutationType.APPEND_CONSTRAINT

    def test_unknown_failure_no_mutation(self) -> None:
        mutator = PromptMutator()
        mutations = mutator.apply_feedback(
            tenant_id="store_001",
            prompt_id="full_description",
            orav_score=0.5,
            failure_reasons=["UNKNOWN_ERROR: something weird"],
        )
        assert len(mutations) == 0


# ════════════════════════════════════════════════════════════════════════════
# CAUSAL TRACES
# ════════════════════════════════════════════════════════════════════════════

class TestCausalTraces:
    """Tests for per-node causal trace strings."""

    def test_all_7_nodes_have_traces(self) -> None:
        for node_id in range(1, 8):
            assert node_id in CAUSAL_TRACES
            assert "→" in CAUSAL_TRACES[node_id]

    def test_failure_record_includes_trace(self) -> None:
        record = record_failure(
            "prod_001", "store_001",
            "NODE_5_CONTENT", FailureReason.NODE_FAILURE,
            "Test failure", trace_id="trace_001",
        )
        assert record.causal_trace == CAUSAL_TRACES[5]


# ════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

class TestUtilities:
    """Tests for shared utility functions."""

    def test_coerce_str_list(self) -> None:
        assert _coerce_str(["Red", "Blue"]) == "Red, Blue"

    def test_coerce_str_none(self) -> None:
        assert _coerce_str(None) == ""

    def test_coerce_str_passthrough(self) -> None:
        assert _coerce_str("hello") == "hello"

    def test_coerce_str_number(self) -> None:
        assert _coerce_str(42) == "42"
