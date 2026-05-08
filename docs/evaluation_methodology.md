# Evaluation Methodology — O-R-A-V + DEMAS

## O-R-A-V Framework

O-R-A-V is a 4-dimension semantic evaluation framework that scores every piece of generated content:

| Dimension | What It Measures | Model Tier | Threshold |
|---|---|---|---|
| **O**riginality | Content uniqueness vs. template repetition and existing corpus overlap | gemini-flash-lite | 0.6 |
| **R**elevance | Alignment between generated content and product/locale context | gemini-flash | 0.7 |
| **A**ccuracy | Factual correctness of product attributes, pricing, and specifications | deterministic + LLM | 0.8 |
| **V**alue | Cross-model consensus on content's commercial and informational value | multi-model | 0.6 |

### Deterministic Checks (Node 6 Gate)

Before any LLM-powered evaluation, Node 6 applies deterministic checks at O(1) cost:

1. **Label Leak Detection**: Scans all content fields for internal pipeline labels (`sacred context`, `city_dna`, `trigger_term`, etc.)
2. **Transactional Blacklist**: Rejects content with sales language (`buy now`, `shop now`, `free shipping`, etc.)
3. **Template Variable Leak**: Catches unreplaced `{variable_name}` patterns
4. **Empty Field Validation**: Ensures content blocks are non-empty

## DEMAS Framework

DEMAS (Deterministic Evaluation, Multi-model Assessment System) provides the JIT audit layer:

### Architecture

```
Content Block → Provenance Matrix Filter → Block-Specific Context → LLM-as-Judge → Verdict
```

### Provenance Matrix

The Provenance Matrix prevents **attention dilution** in LLM-as-Judge evaluations. Instead of sending the judge model the entire pipeline context (dozens of variables), it sends ONLY the ground-truth variables relevant to the specific content block being evaluated.

### Evaluator Registry

Evaluators are registered at startup and dispatched via the `EvaluatorRegistry`:

```python
registry = EvaluatorRegistry()
registry.register("structural", StructuralEvaluator())
registry.register("seo_policy", SEOPolicyEvaluator())

evaluator = registry.get("structural")
result = await evaluator.evaluate(content, provenance)
```

Each evaluator implements `BaseEvaluator.evaluate()` returning a typed verdict with score, reasons, and chain-of-thought reasoning.
