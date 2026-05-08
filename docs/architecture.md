# Architecture

## DAG Topology

```mermaid
graph LR
    subgraph Phase 1 — Parallel
        N1[Node 1<br/>Context Researcher]
        N2[Node 2<br/>Input Normalizer]
    end

    subgraph Phase 2 — Sequential
        N3[Node 3<br/>Synonym Generator]
        N4[Node 4<br/>Demand Gatekeeper]
        N5[Node 5<br/>Content Generator]
        N6[Node 6<br/>Quality Validator]
        N7[Node 7<br/>Metadata Extractor]
    end

    N1 --> N3
    N2 --> N3
    N3 --> N4
    N4 --> N5
    N5 --> N6
    N6 --> N7

    N5 -.-> DEMAS[DEMAS JIT Audit]
    DEMAS -.-> N6
    N6 -.-> FW[Data Flywheel]
```

## Gate-Agent Pattern

Every node implements a dual-layer architecture:

```
┌─────────────────────────────────────┐
│ DETERMINISTIC GATE                   │
│   Fires first. Zero LLM cost.       │
│   O(1) complexity. Hard pass/fail.   │
├─────────────────────────────────────┤
│ PROBABILISTIC AGENT                  │
│   Fires ONLY if gate passes.         │
│   LLM-powered. Structured output.   │
└─────────────────────────────────────┘
```

### Per-Node Gates

| Node | Gate | Agent |
|---|---|---|
| 1 — Context Researcher | Locale Resolver (ISO 3166-1) | Cultural Context Agent (5-pillar City DNA) |
| 2 — Input Normalizer | Schema Validator (Pydantic) | Semantic Extraction Agent (taxonomy) |
| 3 — Synonym Generator | Deduplication Filter (SimHash) | Synonym Generation Agent (locale-aware) |
| 4 — Demand Gatekeeper | Threshold Comparator (numeric) | Trend Analysis Agent (time-series) |
| 5 — Content Generator | Template Selector (category→template) | Content Generation Agent (LoRA + MoE) |
| 6 — Quality Validator | Format Compliance (label leak, blacklist) | Multi-Model Consensus Judge (O-R-A-V) |
| 7 — Metadata Extractor | Embedding Dimension Check (NaN, magnitude) | Feature Extraction Agent (dense vectors) |

## Fail-Closed Policy

Every pipeline exit is recorded to an immutable `FailureRecord`:

```python
@dataclass(frozen=True)
class FailureRecord:
    product_id: str
    store_id: str
    failure_node: str
    failure_reason: FailureReason
    error_message: str
    retry_eligible: bool
    trace_id: str
    causal_trace: str  # Per-node transformation path
```

### Causal Trace Strings

Each node has a human-readable causal trace for mechanistic interpretability:

| Node | Causal Trace |
|---|---|
| 1 | Locale ID → Geo Validation → Cultural Enrichment → Context Object |
| 2 | Raw Input → Schema Gate → Semantic Extraction → Normalized Record |
| 3 | Base Terms → Dedup Gate → LLM Expansion → Synonym Set |
| 4 | Synonym Set → Volume Lookup → Threshold Gate → Qualified Terms |
| 5 | Qualified Terms → Template Gate → LoRA Generation → Draft Content |
| 6 | Draft Content → Format Gate → O-R-A-V Scoring → Accept or Reject |
| 7 | Validated Content → Dimension Gate → Embedding Generation → Feature Vector |

## Store Context Integrity

The store context (locale, timezone, cultural signals) is:
1. Frozen at pipeline `ENTRY` via `assert_store_integrity(ctx, "ENTRY")`
2. Verified at pipeline `PRE_EXPORT` via `assert_store_integrity(ctx, "PRE_EXPORT")`
3. Any mutation triggers `STORE_INTEGRITY_VIOLATION` → fail-closed halt

## Provenance Matrix

Maps each content block to its exact ground-truth variables, preventing attention dilution in LLM-as-Judge evaluations:

| Block | Ground-Truth Variables |
|---|---|
| `full_description` | lean_product_name · semantic_category · brand · trigger_term · city_dna · style_seed · visual_context |
| `short_description` | lean_product_name · semantic_category · trigger_term |
| `meta_title` | lean_product_name · trigger_term · city |
| `meta_description` | lean_product_name · semantic_category · trigger_term · secondary_keyword |
| `store_welcome_line` | city · store_name · lean_product_name |
| `image_alt_tag` | lean_product_name · brand · color · visual_context |
| `key_features` | lean_product_name · semantic_category · visual_context |
| `faq_items` | lean_product_name · semantic_category · trigger_term · relevant_search_queries |

## 4-Layer Isolation Stack

| Layer | Mechanism | Purpose |
|---|---|---|
| Tenant | Adapter-level binding (`store_context.store_id`) | No cross-tenant data leakage |
| Prompt | Per-tenant prompt cache with mutation history | No prompt cross-contamination |
| Task | `contextvars.ContextVar` run context | No async task cross-contamination |
| KV Cache | Namespace-prefixed cache keys | No cache poisoning between tenants |
