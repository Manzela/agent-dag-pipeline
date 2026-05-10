"""
Vertex AI Deployment Helper.

Provides utilities for deploying the pipeline to Vertex AI Agent Engine
via the ``adk deploy`` CLI or programmatic API.

Usage (CLI)::

    adk deploy agent.py \\
        --project $PROJECT_ID \\
        --region us-central1

Usage (Programmatic)::

    from agent_dag.adk.deploy import deploy_to_vertex

    remote = deploy_to_vertex(
        project_id="my-gcp-project",
        region="us-central1",
    )

Requires::

    pip install google-cloud-aiplatform
    export GOOGLE_GENAI_USE_VERTEXAI=1
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def deploy_to_vertex(
    *,
    project_id: str,
    region: str = "us-central1",
    display_name: str = "agent-dag-pipeline",
    enable_tracing: bool = True,
    requirements: list[str] | None = None,
) -> Any:
    """Deploy the pipeline to Vertex AI Agent Engine.

    Parameters
    ----------
    project_id : str
        GCP project ID.
    region : str
        GCP region (default: us-central1).
    display_name : str
        Display name for the deployed agent.
    enable_tracing : bool
        Enable Cloud Trace integration.
    requirements : list[str], optional
        Additional pip requirements for the deployment.

    Returns
    -------
    RemoteAgent
        The deployed agent engine instance.
    """
    try:
        import vertexai
        from vertexai import agent_engines
    except ImportError as exc:
        raise ImportError(
            "Vertex AI SDK is required for deployment. "
            "Install with: pip install google-cloud-aiplatform"
        ) from exc

    from .pipeline import root_agent

    if root_agent is None:
        raise RuntimeError("Pipeline could not be built. Is google-adk installed?")

    # Initialize Vertex AI
    vertexai.init(project=project_id, location=region)

    # Wrap in AdkApp
    app = agent_engines.AdkApp(
        agent=root_agent,
        enable_tracing=enable_tracing,
    )

    # Deploy
    default_requirements = [
        "google-adk>=1.0,<2",
        "pydantic>=2.6",
    ]
    if requirements:
        default_requirements.extend(requirements)

    remote_agent = agent_engines.create(
        agent_engine=app,
        display_name=display_name,
        requirements=default_requirements,
    )

    logger.info(
        "Deployed to Vertex AI: project=%s region=%s name=%s",
        project_id,
        region,
        remote_agent.resource_name,
    )

    return remote_agent
