# Roadmap

## Current: v3.0.0 — Open-Source Foundation

The initial open-source release establishes the complete pipeline architecture with all 7 nodes, fail-closed safety, and the data flywheel framework.

### ✅ Shipped

- [x] 7-node DAG with Gate-Agent dual-layer pattern
- [x] Phase 1 parallel + Phase 2 sequential execution
- [x] Fail-closed safety with immutable audit trail
- [x] O-R-A-V 4-dimension semantic evaluation framework
- [x] DEMAS JIT audit with Provenance Matrix
- [x] 3-tier data flywheel (production-baseline → quality-approved → failure-cases)
- [x] 15+ multi-dimensional scoring metrics (NUMERIC, BOOLEAN, CATEGORICAL)
- [x] DPO preference pair generation with margin-maximization
- [x] Hebbian prompt mutation for runtime feedback
- [x] LoRA training trigger with curriculum-based versioning
- [x] 36-test suite covering all critical paths
- [x] CI pipeline (ruff + mypy + pytest + security scan)
- [x] Complete documentation (architecture, flywheel, evaluation methodology)

---

## v3.1.0 — Observability & Production Hardening

**Target:** Q3 2026

### Observability
- [ ] Production Langfuse integration (replace injectable stubs)
- [ ] OpenTelemetry distributed tracing with span-per-node granularity
- [ ] Real-time scoring dashboard via Langfuse Prompt Playground
- [ ] Cost-per-token telemetry with per-node attribution

### Evaluator Expansion
- [ ] `evaluators/structural_evaluator.py` — Full implementation of structural compliance checks
- [ ] `evaluators/seo_policy_evaluator.py` — RAG-grounded policy compliance with injected rubrics
- [ ] `evaluators/factual_evaluator.py` — Product attribute factual accuracy verification
- [ ] Pluggable evaluator discovery via `importlib.metadata` entry points

### Infrastructure
- [ ] S-LoRA hot-reload integration for zero-downtime adapter swaps
- [ ] Redis-backed prompt mutation cache (replace injectable stubs)
- [ ] Firestore state management for training coordination
- [ ] Health check endpoint for container orchestration

---

## v3.2.0 — Multi-Model & Advanced RL

**Target:** Q4 2026

### Multi-Model Consensus
- [ ] Cross-model O-R-A-V scoring (gemini-flash + gemma-4 agreement gate)
- [ ] Model-disagreement detection with automatic escalation
- [ ] Confidence-weighted ensemble scoring

### Advanced Flywheel
- [ ] Online preference optimization (streaming DPO updates)
- [ ] Active learning: identify maximally informative training samples
- [ ] Prompt version rollback with automatic A/B comparison
- [ ] Drift detection on production-baseline tier (statistical process control)

### Scaling
- [ ] Batch pipeline mode for bulk product processing
- [ ] Rate limiting and backpressure handling
- [ ] Multi-tenant adapter management (per-tenant LoRA selection)

---

## v4.0.0 — Autonomous Self-Improvement

**Target:** 2027

### Full Autonomy
- [ ] Closed-loop retraining without human intervention
- [ ] Automatic quality threshold calibration based on distribution shift
- [ ] Self-healing prompt mutation with rollback on regression
- [ ] Constitutional AI-style critique-and-revise chains

### Benchmarks & Research
- [ ] Public evaluation benchmark for content generation pipelines
- [ ] Ablation study: Gate-Agent vs. monolithic node architecture
- [ ] Paper: "Provenance Matrices for Attention Dilution Prevention in LLM-as-Judge"

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to propose features or report issues.
Feature requests use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.yml).
