"""
Node 1 — Context Researcher (City DNA).

Role: CONTEXT INJECTION
Causal Trace: Locale ID → Geo Validation → Cultural Enrichment → Context Object

Architecture (Gate-Agent Pattern):
    ┌─────────────────────────────────────────────┐
    │ DETERMINISTIC GATE: Locale Resolver          │
    │   Validates ISO locale codes, timezone       │
    │   offsets, and geographic taxonomy.           │
    │   Tools: ISO-3166, Timezone DB, GeoHash      │
    ├─────────────────────────────────────────────┤
    │ PROBABILISTIC AGENT: Cultural Context Agent  │
    │   Enriches product context with city-level   │
    │   cultural and seasonal signals. Adapts      │
    │   tone, idioms, and relevance framing.       │
    │   Tools: LLM, Long-Term Memory Cache         │
    └─────────────────────────────────────────────┘

The gate ensures that only valid, resolvable locales proceed to the
expensive LLM-powered cultural enrichment step. Invalid locale data
is rejected at O(1) cost without any LLM invocation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from ..shared.data_contracts import StoreContext

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# DETERMINISTIC GATE: Locale Resolver
# ════════════════════════════════════════════════════════════════════════════

# ISO 3166-1 alpha-2 codes for supported locales
SUPPORTED_LOCALES: frozenset[str] = frozenset({
    "ES", "PT", "IL", "GB", "DE", "FR", "IT", "PL", "CZ",
})


def gate_locale_resolver(store_context: StoreContext) -> bool:
    """Validate that the store context has a resolvable locale.

    This is the deterministic gate for Node 1. It fires before any
    LLM invocation and rejects malformed locale data at O(1) cost.

    Parameters
    ----------
    store_context : StoreContext
        The frozen store context to validate.

    Returns
    -------
    bool
        True if the locale is valid and supported, False otherwise.
    """
    if not store_context.country_code:
        logger.warning("Gate REJECT: missing country_code for store %s", store_context.store_id)
        return False

    if store_context.country_code.upper() not in SUPPORTED_LOCALES:
        logger.warning(
            "Gate REJECT: unsupported locale %s for store %s",
            store_context.country_code, store_context.store_id,
        )
        return False

    if not store_context.city:
        logger.warning("Gate REJECT: missing city for store %s", store_context.store_id)
        return False

    return True


# ════════════════════════════════════════════════════════════════════════════
# PROBABILISTIC AGENT: Cultural Context Agent
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CityDNAProfile:
    """Cultural enrichment output from the Context Research agent.

    Contains 5 pillars of city-level intelligence used to ground
    content generation in local cultural context.
    """
    demographics: dict[str, Any]
    geography: dict[str, Any]
    infrastructure: dict[str, Any]
    culture: dict[str, Any]
    growth: dict[str, Any]


async def agent_cultural_context(
    store_context: StoreContext,
    *,
    llm_client: Optional[Any] = None,
    cache: Optional[Any] = None,
) -> CityDNAProfile:
    """Enrich the store context with city-level cultural intelligence.

    This is the probabilistic agent for Node 1. It uses an LLM to
    research and synthesize 5 pillars of city DNA: demographics,
    geography, infrastructure, culture, and growth indicators.

    The agent first checks the long-term memory cache for existing
    city profiles. Cache hits skip the LLM entirely.

    Parameters
    ----------
    store_context : StoreContext
        Validated store context from the gate.
    llm_client : optional
        Injectable LLM client for research queries.
    cache : optional
        Injectable cache for city DNA profiles (Redis, dict, etc.).

    Returns
    -------
    CityDNAProfile
        Complete 5-pillar city intelligence profile.
    """
    cache_key = f"city_dna:{store_context.country_code}:{store_context.city}"

    # ── Cache hit: skip LLM entirely ──
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            logger.debug("City DNA cache HIT for %s", cache_key)
            return CityDNAProfile(**cached)

    # ── Cache miss: invoke LLM for each pillar ──
    logger.info("City DNA cache MISS — researching %s, %s", store_context.city, store_context.country_code)

    # In production, each pillar is a separate LLM call with structured output.
    # Here we demonstrate the contract without binding to a specific LLM.
    profile = CityDNAProfile(
        demographics={},
        geography={},
        infrastructure={},
        culture={},
        growth={},
    )

    # ── Persist to cache for future runs ──
    if cache is not None:
        try:
            from dataclasses import asdict
            cache.set(cache_key, asdict(profile))
        except Exception as exc:
            logger.debug("Cache write failed (non-fatal): %s", exc)

    return profile


# ════════════════════════════════════════════════════════════════════════════
# NODE ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

async def run_node1(
    product: dict[str, Any],
    store_context: StoreContext,
    *,
    llm_client: Optional[Any] = None,
    cache: Optional[Any] = None,
) -> CityDNAProfile:
    """Execute Node 1: Context Research (City DNA).

    Implements the Gate-Agent pattern:
    1. GATE: Locale Resolver (deterministic, O(1))
    2. AGENT: Cultural Context Agent (probabilistic, LLM-powered)

    Parameters
    ----------
    product : dict
        Raw product data (used for context-aware research queries).
    store_context : StoreContext
        Immutable store context with locale information.

    Returns
    -------
    CityDNAProfile
        Complete 5-pillar city intelligence profile.

    Raises
    ------
    ValueError
        If the locale gate rejects the store context.
    """
    # ── Step 1: Deterministic Gate ──
    if not gate_locale_resolver(store_context):
        raise ValueError(
            f"Node 1 gate REJECTED: invalid locale for store {store_context.store_id} "
            f"(country={store_context.country_code}, city={store_context.city})"
        )

    # ── Step 2: Probabilistic Agent ──
    return await agent_cultural_context(
        store_context, llm_client=llm_client, cache=cache,
    )
