"""
Google ADK Integration Layer.

Provides first-class integration with Google's Agent Development Kit,
enabling the pipeline to run via ``adk run``, ``adk web``, and
``adk deploy`` while preserving standalone CLI operation.

Usage (ADK)::

    adk run agent.py          # Local dev server
    adk web agent.py          # Interactive debugging UI
    adk deploy agent.py       # Deploy to Vertex AI Agent Engine

Usage (Standalone — unchanged)::

    python -m agent_dag run --products data/products.json --stores data/stores.json
"""

from .callbacks import flywheel_ingest_callback, integrity_check_callback
from .gate_agent import GateAgent, GateDecision, GateResult
from .pipeline import build_pipeline, root_agent
from .runner import create_runner

__all__ = [
    "GateAgent",
    "GateDecision",
    "GateResult",
    "build_pipeline",
    "root_agent",
    "create_runner",
    "integrity_check_callback",
    "flywheel_ingest_callback",
]
