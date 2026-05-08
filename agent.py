"""
ADK Entry Point — ``adk run`` / ``adk web`` / ``adk deploy``.

This file is the standard ADK entry point that exposes ``root_agent``
for the ADK CLI tools.

Usage::

    # Local development with ADK web UI
    adk run agent.py
    adk web agent.py

    # Deploy to Vertex AI Agent Engine
    adk deploy agent.py --project $PROJECT_ID --region us-central1

The pipeline can still be run standalone without ADK::

    python -m agent_dag run --products data/products.json --stores data/stores.json
"""

from agent_dag.adk.pipeline import root_agent  # noqa: F401
