"""
Model Armor Guardrails — GCP-Native Input/Output Protection.

Integrates with Google Cloud's Model Armor service to provide
runtime security for the pipeline's LLM interactions.

Usage::

    from agent_dag.adk.guardrails import create_model_armor_callback

    pipeline = build_pipeline(
        before_agent=create_model_armor_callback(
            template_name="projects/my-project/locations/us-central1/templates/pipeline-guard"
        ),
    )

Requires::

    pip install google-cloud-modelarmor
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def create_model_armor_callback(
    template_name: str,
) -> Callable[..., Any]:
    """Factory for Model Armor guardrail callbacks.

    Parameters
    ----------
    template_name : str
        Full resource name of the Model Armor template.
        Format: projects/{project}/locations/{location}/templates/{template}

    Returns
    -------
    Callable
        ADK before_agent_callback that screens inputs via Model Armor.
    """

    def guardrail(callback_context: Any) -> Any:
        """Screen pipeline inputs through Model Armor."""
        try:
            from google.cloud import modelarmor_v1
        except ImportError:
            logger.debug("Model Armor SDK not installed, skipping guardrail")
            return None

        user_input = callback_context.state.get("product", {})
        if not user_input:
            return None

        # Serialize product data for screening
        import json

        prompt_text = json.dumps(user_input, default=str)

        try:
            client = modelarmor_v1.ModelArmorClient()
            request = modelarmor_v1.SanitizeUserPromptRequest(
                name=template_name,
                user_prompt_data=modelarmor_v1.UserPromptData(
                    text=prompt_text,
                ),
            )
            response = client.sanitize_user_prompt(request=request)

            filter_state = response.sanitize_result.filter_match_state
            if filter_state.name == "MATCH_FOUND":
                logger.warning(
                    "MODEL_ARMOR_BLOCK: input matched filter in template %s",
                    template_name,
                )
                try:
                    from google.genai.types import Content, Part

                    return Content(
                        parts=[
                            Part(
                                text="BLOCKED: Model Armor policy violation detected."
                            )
                        ]
                    )
                except ImportError:
                    return None

        except Exception as exc:
            # Model Armor is non-blocking — log but don't fail the pipeline
            logger.debug("Model Armor check failed (non-fatal): %s", exc)

        return None

    return guardrail


def create_output_armor_callback(
    template_name: str,
) -> Callable[..., Any]:
    """Factory for Model Armor output screening.

    Screens LLM-generated content for policy violations before
    it's stored in the output dataset.

    Parameters
    ----------
    template_name : str
        Full resource name of the Model Armor template.
    """

    def output_guardrail(callback_context: Any) -> None:
        """Screen pipeline outputs through Model Armor."""
        try:
            from google.cloud import modelarmor_v1
        except ImportError:
            return None

        content = callback_context.state.get("content_generator:result", {})
        if not content:
            return None

        import json

        try:
            client = modelarmor_v1.ModelArmorClient()
            request = modelarmor_v1.SanitizeModelResponseRequest(
                name=template_name,
                model_response_data=modelarmor_v1.ModelResponseData(
                    text=json.dumps(content, default=str),
                ),
            )
            response = client.sanitize_model_response(request=request)
            filter_state = response.sanitize_result.filter_match_state

            if filter_state.name == "MATCH_FOUND":
                callback_context.state["quality_validator:result"] = {
                    "decision": "FAIL",
                    "errors": ["Model Armor output filter matched"],
                }
                logger.warning("MODEL_ARMOR_OUTPUT_BLOCK: content filtered")

        except Exception as exc:
            logger.debug("Model Armor output check failed (non-fatal): %s", exc)

        return None

    return output_guardrail
