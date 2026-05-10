"""
A2A Agent Card — Agent-to-Agent Protocol Discovery.

Defines the pipeline's Agent Card for the A2A protocol,
enabling other agents to discover and delegate tasks to
this pipeline via standardized capability declarations.
"""

from __future__ import annotations

from typing import Any

AGENT_CARD: dict[str, Any] = {
    "name": "content_pipeline",
    "description": (
        "7-node autonomous content generation pipeline with O-R-A-V "
        "semantic evaluation, fail-closed safety, and self-improving "
        "RL data flywheel. Generates locale-specific product content "
        "from raw product data and store context."
    ),
    "version": "3.0.0",
    "url": "https://github.com/Manzela/agent-dag-pipeline",
    "provider": {
        "organization": "Daniel Manzela",
    },
    "capabilities": {
        "streaming": False,
        "pushNotifications": False,
        "stateTransitionHistory": True,
    },
    "skills": [
        {
            "id": "product_content_generation",
            "name": "Product Content Generation",
            "description": (
                "Generate locale-specific product descriptions, meta tags, "
                "FAQs, and key features from raw product data."
            ),
            "tags": ["content", "product", "localization", "seo"],
        },
        {
            "id": "quality_evaluation_orav",
            "name": "O-R-A-V Quality Evaluation",
            "description": (
                "Evaluate generated content on Originality, Relevance, "
                "Accuracy, and Value dimensions using LLM-as-Judge."
            ),
            "tags": ["evaluation", "quality", "llm-judge"],
        },
    ],
    "defaultInputModes": ["application/json"],
    "defaultOutputModes": ["application/json"],
}
