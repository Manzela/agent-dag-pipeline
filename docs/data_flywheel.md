# Data Flywheel — RL Self-Retraining Methodology

## Overview

The data flywheel implements a closed-loop self-improvement system that mirrors Anthropic's Constitutional AI / RLAIF methodology. Every pipeline execution — success or failure — feeds into a 3-tier dataset curation engine that continuously generates training data for model improvement.

## 3-Tier Architecture

```
Production Traffic
      │
      ▼
┌─────────────────────────────┐
│ TIER 1: production-baseline │ ← All production runs
│   17 prompt-specific        │
│   dataset partitions        │
└─────────┬───────────────────┘
          │ O-R-A-V ≥ 0.7 && DEMAS PASS
          ▼
┌─────────────────────────────┐
│ TIER 2: quality-approved    │ ← "Chosen" examples
│   High-quality outputs      │
│   for DPO training          │
└─────────┬───────────────────┘
          │ O-R-A-V < 0.5 || DEMAS FAIL
          ▼
┌─────────────────────────────┐
│ TIER 3: failure-cases       │ ← "Rejected" examples
│   Negative examples         │
│   for DPO training          │
└─────────────────────────────┘
          │
          ▼
    DPO Preference Pairs → LoRA Fine-Tuning → Improved Model
```

## Scoring Engine (15+ Metrics)

| Metric | Type | Threshold | Node |
|---|---|---|---|
| orav_quality | NUMERIC | 0.7 | 6 |
| orav_originality | NUMERIC | 0.6 | 6 |
| orav_relevance | NUMERIC | 0.7 | 6 |
| orav_accuracy | NUMERIC | 0.8 | 6 |
| orav_value | NUMERIC | 0.6 | 6 |
| n2_taxonomy_confidence | NUMERIC | 0.8 | 2 |
| n5_block_quality | NUMERIC | 0.65 | 5 |
| demas_provenance_coverage | NUMERIC | 0.8 | 6 |
| n2_schema_valid | BOOLEAN | — | 2 |
| n5_generation_success | BOOLEAN | — | 5 |
| n5_trigger_term_inclusion | BOOLEAN | — | 5 |
| n6_label_leak_free | BOOLEAN | — | 6 |
| n6_transactional_free | BOOLEAN | — | 6 |
| n6_template_var_clean | BOOLEAN | — | 6 |
| orav_decision | CATEGORICAL | — | 6 |
| demas_jit_verdict | CATEGORICAL | — | 5 |

## Preference Pair Generation (DPO)

The DPO generator matches quality-approved vs. failure-cases items by `prompt_id` and applies margin-maximization filtering:

```python
# Only pairs with score gap ≥ threshold survive
margin = chosen_score - rejected_score
if margin < margin_threshold:  # default: 0.15
    continue  # Too close — weak training signal
```

Pairs are sorted by margin (descending) so the strongest training signals come first.

## Hebbian Prompt Mutation

Before a full LoRA retraining cycle, the prompt mutator provides fast runtime feedback:

| Failure Pattern | Mutation Type | Effect |
|---|---|---|
| LABEL_LEAK | APPEND_CONSTRAINT | Adds "never include internal labels" to prompt |
| TRANSACTIONAL | APPEND_CONSTRAINT | Adds "no sales language" to prompt |
| TEMPLATE_VAR_LEAK | APPEND_CONSTRAINT | Adds "replace all variables" to prompt |
| LOW_ORIGINALITY | ADJUST_TEMPERATURE | Increases LLM creativity |
| LOW_RELEVANCE | BOOST_EXAMPLE | Injects high-scoring few-shot example |

## Training Trigger

LoRA training is curriculum-based:
1. Monitor prompt version changes in the registry
2. When version changes → check if sufficient preference pairs exist (min: 50)
3. Validate mean margin ≥ 0.2 for training data quality
4. Submit LoRA fine-tuning job (rank=16, alpha=32, lr=2e-5)
5. Hot-reload via S-LoRA serving (zero downtime)
