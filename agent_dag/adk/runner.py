"""
ADK Runner — Session Management and Execution.

Wraps the pipeline in an ADK Runner with configurable SessionService
for development (InMemory) and production (Vertex AI) use.

Usage::

    from agent_dag.adk.runner import create_runner

    # Development (in-memory sessions)
    runner = create_runner()

    # Production (Vertex AI managed sessions)
    from google.adk.sessions import VertexAiSessionService
    runner = create_runner(session_service=VertexAiSessionService())
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def create_runner(
    *,
    session_service: Optional[Any] = None,
    app_name: str = "agent_dag_pipeline",
    agent: Optional[Any] = None,
) -> Any:
    """Create an ADK Runner for the pipeline.

    Parameters
    ----------
    session_service : SessionService, optional
        ADK session service for state persistence.
        Defaults to InMemorySessionService for development.
    app_name : str
        Application name for session scoping.
    agent : BaseAgent, optional
        Override the default pipeline agent.

    Returns
    -------
    Runner
        Configured ADK Runner ready for execution.
    """
    try:
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
    except ImportError as exc:
        raise ImportError(
            "Google ADK is required for the Runner. "
            "Install with: pip install google-adk"
        ) from exc

    if agent is None:
        from .pipeline import root_agent

        if root_agent is None:
            raise RuntimeError("Pipeline could not be built. Is google-adk installed?")
        agent = root_agent

    return Runner(
        agent=agent,
        app_name=app_name,
        session_service=session_service or InMemorySessionService(),
    )
