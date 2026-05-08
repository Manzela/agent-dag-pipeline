"""
ADK Pipeline Composition — SequentialAgent + ParallelAgent.

Composes the 7 GateAgent nodes into the pipeline's DAG topology
using ADK's native workflow agents.

Topology::

    ┌─────────────────────────────────────┐
    │ Phase 1: ParallelAgent              │
    │   ├── ContextResearcher  (Node 1)   │
    │   └── InputNormalizer    (Node 2)   │
    └─────────────┬───────────────────────┘
                  │
    ┌─────────────▼───────────────────────┐
    │ Phase 2: SequentialAgent            │
    │   ├── SynonymGenerator   (Node 3)   │
    │   ├── DemandGatekeeper   (Node 4)   │
    │   ├── ContentGenerator   (Node 5)   │
    │   ├── QualityValidator   (Node 6)   │
    │   └── MetadataExtractor  (Node 7)   │
    └─────────────────────────────────────┘
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Conditional imports — ADK is optional
try:
    from google.adk.agents import ParallelAgent, SequentialAgent

    _ADK_AVAILABLE = True
except ImportError:
    _ADK_AVAILABLE = False

from .nodes.content_generator import ContentGenerator
from .nodes.context_researcher import ContextResearcher
from .nodes.demand_gatekeeper import DemandGatekeeper
from .nodes.input_normalizer import InputNormalizer
from .nodes.metadata_extractor import MetadataExtractor
from .nodes.quality_validator import QualityValidator
from .nodes.synonym_generator import SynonymGenerator


def build_pipeline(
    *,
    before_agent: Optional[Callable[..., Any]] = None,
    after_agent: Optional[Callable[..., Any]] = None,
) -> Any:
    """Build the 7-node content pipeline using ADK workflow agents.

    Parameters
    ----------
    before_agent : callable, optional
        ADK before_agent_callback for the root pipeline agent.
    after_agent : callable, optional
        ADK after_agent_callback for the root pipeline agent.

    Returns
    -------
    SequentialAgent
        The composed pipeline, ready for use with an ADK Runner.

    Raises
    ------
    ImportError
        If google-adk is not installed.
    """
    if not _ADK_AVAILABLE:
        raise ImportError(
            "Google ADK is required for pipeline composition. "
            "Install with: pip install google-adk"
        )

    # Phase 1: Parallel (no data dependency between nodes 1 and 2)
    phase1 = ParallelAgent(
        name="phase1_context_and_normalize",
        sub_agents=[
            ContextResearcher(),
            InputNormalizer(),
        ],
    )

    # Phase 2: Sequential (strict data flow: 3 → 4 → 5 → 6 → 7)
    phase2 = SequentialAgent(
        name="phase2_generate_and_validate",
        sub_agents=[
            SynonymGenerator(),
            DemandGatekeeper(),
            ContentGenerator(),
            QualityValidator(),
            MetadataExtractor(),
        ],
    )

    # Root pipeline
    kwargs: dict[str, Any] = {
        "name": "content_pipeline",
        "sub_agents": [phase1, phase2],
    }
    if before_agent:
        kwargs["before_agent_callback"] = before_agent
    if after_agent:
        kwargs["after_agent_callback"] = after_agent

    return SequentialAgent(**kwargs)


# Default pipeline instance — used by agent.py entry point
try:
    root_agent = build_pipeline()
except ImportError:
    root_agent = None  # type: ignore[assignment]
