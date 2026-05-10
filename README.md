# Agent DAG Pipeline

> A production-grade 7-node autonomous agent DAG with fail-closed safety, O-R-A-V semantic evaluation, and a self-improving RL data flywheel.

[![CI](https://github.com/Manzela/agent-dag-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Manzela/agent-dag-pipeline/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Google ADK](https://img.shields.io/badge/Google_ADK-Native-4285F4?logo=google)](https://google.github.io/adk-docs/)
[![Vertex AI](https://img.shields.io/badge/Vertex_AI-Deployable-34A853?logo=googlecloud)](https://cloud.google.com/vertex-ai)
[![Observatory](https://img.shields.io/badge/Live_Demo-Pipeline_Observatory-0A84FF)](https://manzela.github.io/pipeline-observatory/)

---

## Architecture

```
Phase 1 — Parallel          Phase 2 — Sequential
┌──────────┐ ┌──────────┐   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  Node 1  │ │  Node 2  │   │  Node 3  │→ │  Node 4  │→ │  Node 5  │→ │  Node 6  │→ │  Node 7  │
│ City DNA │ │Normalizer│   │ Synonyms │  │ SV Gate  │  │  Writer  │  │Validator │  │ Features │
│ Context  │ │ Cleanse  │   │  Expand  │  │  Filter  │  │ Generate │  │  O-R-A-V │  │Vectorize │
└──────────┘ └──────────┘   └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
                                                              │              │
                                                              ▼              ▼
                                                   ┌─────────────────────────────────┐
                                                   │     DEMAS — JIT Audit Layer     │
                                                   │   Provenance Matrix Filtering   │
                                                   │   68.9% pass rate · Fail-Closed │
                                                   └─────────────────────────────────┘
```

### Gate-Agent Pattern (Every Node)

Every node implements a dual-layer architecture:

1. **DETERMINISTIC GATE** — fires first, zero-LLM, O(1), hard pass/fail
2. **PROBABILISTIC AGENT** — fires only if gate passes, LLM-powered

This ensures that expensive LLM calls are never wasted on invalid inputs.

## Key Differentiators

### 1. Fail-Closed Safety

Every pipeline exit — node failure, rejection, deferral, integrity violation — is recorded to an immutable audit trail. No partial content ever propagates downstream.

```python
# Store context integrity verified at ENTRY and EXIT
assert_store_integrity(store_context, "ENTRY")
# ... entire pipeline ...
assert_store_integrity(store_context, "PRE_EXPORT")
```

### 2. O-R-A-V Multi-Model Evaluation

Four-dimensional semantic evaluation with typed scoring:

| Dimension | Model Tier | Threshold |
|---|---|---|
| **O**riginality | gemini-flash-lite | 0.6 |
| **R**elevance | gemini-flash | 0.7 |
| **A**ccuracy | deterministic + LLM | 0.8 |
| **V**alue | cross-model consensus | 0.6 |

### 3. Self-Improving Data Flywheel (RLAIF)

A 3-tier dataset curation engine that mirrors Anthropic's Constitutional AI methodology:

```
Production Traffic → [T1: production-baseline] → All runs
                          │
                   O-R-A-V ≥ 0.7 && DEMAS PASS
                          │
                     [T2: quality-approved] → Chosen examples
                          │
                   O-R-A-V < 0.5 || DEMAS FAIL
                          │
                     [T3: failure-cases] → Rejected examples
                          │
                   T2 + T3 → DPO Preference Pairs → LoRA Fine-Tuning
```

### 4. Provenance Matrix

Prevents attention dilution in LLM-as-Judge evaluations by mapping each content block to its **exact ground-truth variables**. The judge model sees ONLY the context relevant to the block being scored.

### 5. 15+ Multi-Dimensional Scoring Metrics

Not a single scalar — three metric types power the flywheel:

- **NUMERIC** (0.0–1.0): `orav_quality`, `orav_originality`, `orav_relevance`, `orav_accuracy`, `orav_value`, `demas_provenance_coverage`
- **BOOLEAN** (pass/fail): `n2_schema_valid`, `n5_generation_success`, `n6_label_leak_free`, `n6_transactional_free`
- **CATEGORICAL** (PASS/RETRY/FAIL): `orav_decision`, `demas_jit_verdict`

## Quick Start

```bash
# Clone
git clone https://github.com/Manzela/agent-dag-pipeline.git
cd agent-dag-pipeline

# Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Test (61 tests, 0.04s)
pytest tests/ -v
```

## Running the Pipeline

The pipeline is **LLM-agnostic**. Bring your own provider or use the built-in mock for testing:

```bash
# Zero-config demo (mock LLM, no API key needed)
python -m agent_dag run \
    --products examples/data/products.json \
    --stores examples/data/stores.json \
    --output ./output

# With OpenAI
pip install -e ".[openai]"
export OPENAI_API_KEY="sk-..."
python -m agent_dag run \
    --products data/products.json \
    --stores data/stores.json \
    --llm openai --model gpt-4o-mini

# With Google Gemini
pip install -e ".[google]"
export GOOGLE_API_KEY="..."
python -m agent_dag run \
    --products data/products.json \
    --stores data/stores.json \
    --llm google --model gemini-2.5-flash-lite
```

**Output:** `output/products_per_location.json` — all results grouped by store location.

### Google ADK Integration (GCP-Native)

The pipeline includes a first-class **Google ADK integration layer** that enables
deployment to Vertex AI Agent Engine:

```bash
# Install ADK dependencies
pip install -e ".[adk]"

# Run via ADK dev server (interactive web UI)
adk run agent.py
adk web agent.py

# Evaluate with golden dataset
adk eval agent.py --eval-set examples/eval/golden_set.json

# Deploy to Vertex AI Agent Engine
export GOOGLE_GENAI_USE_VERTEXAI=1
adk deploy agent.py --project $PROJECT_ID --region us-central1
```

The ADK layer wraps each node as a `GateAgent(BaseAgent)` with:
- Native `before_agent_callback` / `after_agent_callback` lifecycle hooks
- `SequentialAgent` + `ParallelAgent` orchestration (matching the DAG topology)
- `SessionService` for state persistence (in-memory or Vertex AI managed)
- Built-in OpenTelemetry instrumentation
- Model Armor guardrails integration
- A2A Agent Card for protocol-based discovery

> **Note:** The standalone CLI (`python -m agent_dag run`) remains fully functional
> without ADK installed. The ADK layer is additive, not a replacement.

### LLM Client Protocol

Implement your own provider by satisfying the `LLMClient` protocol:

```python
from agent_dag import LLMClient, run_pipeline, StoreContext

class MyCustomLLM:
    async def generate(self, prompt, *, system_prompt="", temperature=0.7, max_tokens=2048, **kw) -> str:
        return my_api.call(prompt)

    async def generate_structured(self, prompt, schema, *, system_prompt="", temperature=0.3, **kw):
        data = json.loads(my_api.call(prompt))
        return schema(**data)

# Use it
result = await run_pipeline(product, store_context, llm_client=MyCustomLLM())
```

## Project Structure

```
agent_dag/
├── orchestrator.py           # DAG controller — Phase 1 parallel + Phase 2 sequential
├── nodes/
│   ├── node1_context_researcher.py    # City DNA — locale gate + cultural agent
│   ├── node2_input_normalizer.py      # Data cleansing — schema gate + semantic agent
│   ├── node3_synonym_generator.py     # Semantic expansion — dedup gate + LLM agent
│   ├── node4_demand_gatekeeper.py     # Volume filtering — threshold gate + trend agent
│   ├── node5_content_generator.py     # Content generation — template gate + LoRA agent
│   ├── node6_quality_validator.py     # O-R-A-V evaluation — format gate + consensus agent
│   └── node7_metadata_extractor.py    # Vectorization — dimension gate + embedding agent
├── adk/                               # Google ADK integration layer
│   ├── gate_agent.py                  # GateAgent(BaseAgent) — ADK-native Gate-Agent ABC
│   ├── pipeline.py                    # SequentialAgent + ParallelAgent composition
│   ├── runner.py                      # ADK Runner with SessionService
│   ├── callbacks.py                   # Lifecycle hooks (integrity, flywheel, telemetry)
│   ├── tools.py                       # Tool callbacks (guardrails, sanitization)
│   ├── eval.py                        # Trajectory + O-R-A-V evaluation
│   ├── guardrails.py                  # Google Cloud Model Armor integration
│   ├── agent_card.py                  # A2A protocol Agent Card
│   ├── deploy.py                      # Vertex AI Agent Engine deployment
│   └── nodes/                         # 7 GateAgent subclasses wrapping existing nodes
├── validators/
│   ├── base_evaluator.py              # DEMAS framework + Provenance Matrix
│   └── evaluators/                    # Pluggable evaluator implementations
├── flywheel/
│   ├── data_flywheel.py               # 3-tier dataset curation engine
│   ├── score_configs.py               # 15+ multi-dimensional scoring metrics
│   ├── preference_pairs.py            # DPO (chosen, rejected) pair generation
│   ├── prompt_mutator.py              # Hebbian runtime prompt adaptation
│   └── training_trigger.py            # LoRA fine-tuning orchestration
├── shared/
│   ├── data_contracts.py              # Pydantic models (all frozen)
│   └── observability.py               # Structured tracing + score emission
│
agent.py                               # ADK entry point (adk run/web/deploy)
tests/
├── test_pipeline.py                   # 36 core tests
├── test_adk_gate_agent.py             # 8 GateAgent tests
├── test_adk_eval.py                   # 7 evaluation tests
└── test_adk_tools.py                  # 8 guardrail tests
```

## Model Evolution

| Period | Model | Calls | Notes |
|---|---|---|---|
| Mar 24–31 | gemma-3-4b-it | 25,847 | On-premise, single GPU |
| Apr 5–6 | gemma-4-26b-a4b | 1,438 | MoE sparse routing, Multi-LoRA |
| Apr 13–14 | gemini-2.5-flash-lite | 5,595 | Cloud migration |
| May 7+ | gemini-3.1-flash-lite | 164+ | Current production |

## Live Demos

- **[Pipeline Operation Demo (YouTube)](https://youtu.be/czZsPbylC1M)** — Visual proof of the pipeline executing autonomously
- **[Architecture Deep-Dive (Loom)](https://www.loom.com/share/be8a97a64f80414eae11176843c559bb)** — Mechanistic interpretability and system walk-through
- **[Pipeline Observatory](https://manzela.github.io/pipeline-observatory/)** — Live execution telemetry
- **[Architecture Deep View](https://manzela.github.io/pipeline-observatory/architecture.html)** — System topology

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full development roadmap, including:
- **v3.1** — Google ADK native integration, Vertex AI deployment, Model Armor guardrails
- **v3.2** — Production Langfuse integration, pluggable evaluators, S-LoRA hot-reload
- **v3.3** — Multi-model consensus scoring, active learning, drift detection
- **v4.0** — Fully autonomous self-improvement with closed-loop retraining

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and workflow.
This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
