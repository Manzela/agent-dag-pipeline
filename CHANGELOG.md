# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] — 2026-05-08

### Added
- Open-source extraction from production ADK v2.6
- 7-node DAG with Gate-Agent dual-layer architecture per node
- Per-node causal trace strings for mechanistic interpretability
- `flywheel/prompt_mutator.py` — Hebbian runtime prompt adaptation
- `flywheel/preference_pairs.py` — DPO (chosen, rejected) pair generation
- `flywheel/data_flywheel.py` — 3-tier dataset curation engine
- `flywheel/score_configs.py` — 15+ multi-dimensional scoring metrics
- `flywheel/training_trigger.py` — LoRA fine-tuning with curriculum triggers
- `validators/base_evaluator.py` — DEMAS framework with Provenance Matrix
- 4-layer isolation architecture (tenant, prompt, task, KV cache)
- Comprehensive test suite (36 tests)
- GitHub Actions CI with lint, type-check, test, and security scan

### Removed
- All proprietary references (company names, client data, internal secrets)
- Vendor-specific infrastructure bindings (graceful degradation)

## Model Migration History

| Version | Model | Calls | Period |
|---|---|---|---|
| v1.0 | gemma-3-4b-it | 25,847 | Mar 24–31, 2026 |
| v2.0 | gemma-4-26b-a4b | 1,438 | Apr 5–6, 2026 |
| v2.5 | gemini-2.5-flash-lite | 5,595 | Apr 13–14, 2026 |
| v2.6 | gemini-3.1-flash-lite | 164+ | May 7+, 2026 |
