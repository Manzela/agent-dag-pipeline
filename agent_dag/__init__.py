"""
Agent DAG Pipeline — Autonomous Multi-Agent Orchestration Framework.

A production-grade 7-node Directed Acyclic Graph (DAG) for autonomous
content generation with:

    - Fail-closed safety guarantees at every node boundary
    - Dual-layer architecture: deterministic gates + probabilistic agents
    - O-R-A-V semantic evaluation with multi-dimensional scoring
    - DEMAS JIT audit framework with Provenance Matrix
    - Self-improving RL data flywheel with 3-tier dataset curation
    - DPO preference pair generation for continuous alignment

Architecture:
    Phase 1 (Parallel):   Node 1 (Context) | Node 2 (Normalizer)
    Phase 2 (Sequential): Node 3 → Node 4 → Node 5 → Node 6 → Node 7

Each node implements the Gate-Agent pattern:
    1. DETERMINISTIC GATE fires first (zero-LLM, O(1), hard pass/fail)
    2. PROBABILISTIC AGENT fires only if gate passes (LLM-powered)

See: docs/architecture.md for the full topology diagram.
"""

from .__version__ import __version__
from .config import PipelineConfig, load_config
from .orchestrator import FailureReason, FailureRecord, run_pipeline
from .shared.data_contracts import (
    ContentBlockOutput,
    ContextPayload,
    FAQItem,
    GateDecision,
    IntentLockOutput,
    LinguisticContext,
    NormalizerOutput,
    ORAVDecision,
    PipelineResult,
    StoreContext,
    SynonymOutput,
)

from .shared.llm_protocol import LLMClient, MockLLMClient, create_llm_client

__all__ = [
    "__version__",
    # Orchestrator
    "run_pipeline",
    "FailureReason",
    "FailureRecord",
    # LLM
    "LLMClient",
    "MockLLMClient",
    "create_llm_client",
    # Config
    "PipelineConfig",
    "load_config",
    # Data contracts
    "StoreContext",
    "LinguisticContext",
    "NormalizerOutput",
    "SynonymOutput",
    "IntentLockOutput",
    "ContentBlockOutput",
    "FAQItem",
    "ContextPayload",
    "PipelineResult",
    "GateDecision",
    "ORAVDecision",
]
