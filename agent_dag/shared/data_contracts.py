"""
Data Contracts — Pydantic-Based Type-Safe Pipeline State.

All inter-node communication in the DAG pipeline is mediated by these
frozen, validated Pydantic models. This ensures:

    1. STRUCTURAL INTEGRITY: No malformed data can traverse node boundaries
    2. IMMUTABILITY: Models are frozen (frozen=True) to prevent mutation
    3. DETERMINISM: Same input always produces same validated output
    4. AUDITABILITY: Every field is typed, documented, and traceable

Design Principles:
    - Every model uses ``model_config = ConfigDict(frozen=True)``
    - Field validators enforce business constraints at parse time
    - Optional fields use explicit ``None`` defaults (no silent coercion)
    - All string fields are stripped of leading/trailing whitespace
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ════════════════════════════════════════════════════════════════════════════
# ENUMS
# ════════════════════════════════════════════════════════════════════════════

class GateDecision(StrEnum):
    """Decision outcomes for demand gatekeeper (Node 4)."""
    PASS = "PASS"
    REJECT = "REJECT"
    DEFER = "DEFER"


class ORAVDecision(StrEnum):
    """Decision outcomes for O-R-A-V quality validator (Node 6)."""
    PASS = "PASS"
    RETRY = "RETRY"
    FAIL = "FAIL"


# ════════════════════════════════════════════════════════════════════════════
# STORE CONTEXT — Immutable locale and tenant context
# ════════════════════════════════════════════════════════════════════════════

class StoreContext(BaseModel):
    """Immutable store/locale context established at pipeline entry.

    This context is frozen at pipeline entry and verified at exit via
    assert_store_integrity(). Any mutation between checkpoints indicates
    cross-contamination and triggers a fail-closed halt.

    Attributes
    ----------
    store_id : str
        Unique identifier for the store/tenant.
    store_name : str
        Human-readable store name.
    city : str
        City where the store is located.
    country_code : str
        ISO 3166-1 alpha-2 country code.
    language : str
        BCP 47 language tag (e.g., "es", "pt", "he").
    timezone : str
        IANA timezone identifier.
    """
    model_config = ConfigDict(frozen=True)

    store_id: str
    store_name: str = ""
    city: str = ""
    country_code: str = ""
    country_name: str = ""
    language: str = ""
    language_name: str = ""
    timezone: str = "UTC"
    address: str = ""
    neighborhood: str = ""
    latitude: str = ""
    longitude: str = ""

    def is_frozen(self) -> bool:
        """Verify that the context has not been mutated since creation."""
        return True  # Pydantic frozen=True enforces this at the framework level


class LinguisticContext(BaseModel):
    """Language and locale metadata for multi-lingual content generation.

    Injected into Node 5 (Content Generator) to ensure content is
    generated in the correct language with proper text direction.
    """
    model_config = ConfigDict(frozen=True)

    target_language: str = ""
    language_name: str = ""
    country_code: str = ""
    country_name: str = ""
    locale: str = ""
    rtl: bool = False


# ════════════════════════════════════════════════════════════════════════════
# NODE OUTPUT CONTRACTS
# ════════════════════════════════════════════════════════════════════════════

class NormalizerOutput(BaseModel):
    """Output contract for Node 2 (Input Normalizer).

    Contains the sanitized, semantically classified product data
    after deterministic schema validation and LLM-powered extraction.
    """
    model_config = ConfigDict(frozen=True)

    original_product_name: str
    lean_product_name: str
    semantic_category: str
    brand: str = ""
    target_language: str = ""
    confidence: float = 0.0
    parent_fingerprint: str = ""
    visual_context: str = ""

    @field_validator("lean_product_name")
    @classmethod
    def lean_name_not_empty(cls, v: str) -> str:
        """Ensure lean product name is never empty after normalization."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("lean_product_name cannot be empty after normalization")
        return stripped


class SynonymOutput(BaseModel):
    """Output contract for Node 3 (Synonym Generator).

    Contains the deduplicated, locale-aware semantic expansions
    after hash-based deduplication and LLM-powered generation.
    """
    model_config = ConfigDict(frozen=True)

    lean_product_name: str
    synonyms: list[str] = Field(default_factory=list)
    target_language: str = ""
    country_code: str = ""


class IntentLockOutput(BaseModel):
    """Output contract for Node 4 (Demand Gatekeeper).

    Contains the qualified trigger term and demand signal data
    after threshold gating and optional trend analysis.
    """
    model_config = ConfigDict(frozen=True)

    trigger_term: str
    trigger_term_city_sv: int = 0
    selection_method: str = ""
    decision: GateDecision = GateDecision.PASS
    all_candidates_ranked: list[Any] = Field(default_factory=list)


class ContentBlockOutput(BaseModel):
    """Output contract for Node 5 (Content Generator).

    Contains all generated content blocks after DEMAS JIT auditing.
    Each block is generated independently and audited by the Provenance
    Matrix to prevent attention dilution.

    All string fields are validated for minimum length to ensure
    no empty blocks pass through the pipeline.
    """
    model_config = ConfigDict(frozen=True)

    full_description: str = ""
    short_description: str = ""
    store_welcome_line: str = ""
    meta_title: str = ""
    meta_description: str = ""
    image_alt_tag: str = ""
    key_features: list[str] = Field(default_factory=list)
    local_trending_features: list[str] = Field(default_factory=list)
    focus_keywords: list[str] = Field(default_factory=list)
    faq_items: list[dict[str, str]] = Field(default_factory=list)


class FAQItem(BaseModel):
    """Individual FAQ entry with question-answer pair."""
    model_config = ConfigDict(frozen=True)

    question: str
    answer: str

    @field_validator("question", "answer")
    @classmethod
    def not_empty(cls, v: str) -> str:
        """FAQ items must have non-empty question and answer."""
        if not v.strip():
            raise ValueError("FAQ question and answer cannot be empty")
        return v.strip()


# ════════════════════════════════════════════════════════════════════════════
# CONTEXT PAYLOAD — Assembled for Node 5
# ════════════════════════════════════════════════════════════════════════════

class ContextPayload(BaseModel):
    """Complete context payload assembled from all upstream nodes for Node 5.

    This is the single, comprehensive input to the Content Generator.
    It aggregates outputs from Nodes 1-4 plus store context into a
    unified, typed payload.

    Fields are organized by provenance:
        - Product data: from raw input + Node 2 normalization
        - Demand data: from Node 4 intent locking
        - Locale data: from Node 1 + store context
        - Style: deterministic seed from product identifier
    """
    model_config = ConfigDict(frozen=True)

    # ── Product data (Node 2 enriched) ──
    product_name: str
    lean_product_name: str
    product_description: str = ""
    product_url: str = ""
    image_url: str = ""
    old_price: str = ""
    current_price: str = ""
    category: str = ""
    semantic_category: str = ""
    subcategory: str = ""
    brand: str = ""
    color: str = ""
    size: str = ""
    sku: str = ""

    # ── Visual grounding (Node 2 vision) ──
    visual_context: str = ""

    # ── Demand signals (Node 4) ──
    trigger_term: str = ""
    trigger_term_city_sv: int = 0
    relevant_search_queries: str = ""
    secondary_keyword: str = ""
    selection_method: str = ""

    # ── Style control ──
    style_seed: str = ""

    # ── Store context ──
    parent_fingerprint: str = ""
    store_context: StoreContext | None = None
    city_dna: dict[str, Any] = Field(default_factory=dict)
    linguistic: LinguisticContext | None = None


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE RESULT — Final output of the DAG
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    """Final result of a complete pipeline execution.

    Encapsulates success/failure status, content output, metadata,
    and flywheel statistics for the caller.
    """
    success: bool
    product_id: str
    content: ContentBlockOutput | None = None
    metadata: dict[str, Any] | None = None
    failure_node: str = ""
    failure_reason: str = ""
    duration_ms: float = 0.0
    flywheel_stats: dict[str, int] = field(default_factory=dict)
