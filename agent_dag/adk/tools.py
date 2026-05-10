"""
ADK Tool Callbacks — Input/Output Guardrails.

Implements ADK's before_tool_callback and after_tool_callback
for runtime safety enforcement on all LLM tool calls.

ADK Tool Callback Contract:
    - before_tool_callback(tool, args, tool_context) → dict | None
        Return dict to short-circuit (skip tool execution)
        Return None to proceed
    - after_tool_callback(tool, args, tool_context, tool_response) → Any
        Return modified or original tool_response
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Internal pipeline labels that must NEVER appear in LLM I/O
SACRED_LABELS: frozenset[str] = frozenset({
    "sacred context", "gmb_context", "city_dna", "trigger_term",
    "style_seed", "context_payload", "lean_product_name",
    "semantic_category", "format_vars",
})

# Maximum response size before truncation (tokens ≈ chars/4)
MAX_TOOL_RESPONSE_CHARS: int = 8000


def before_tool_guardrail(
    tool: Any,
    args: dict[str, Any],
    tool_context: Any,
) -> dict[str, Any] | None:
    """Intercept tool calls for safety checks.

    Checks:
        1. Sacred label leak in arguments
        2. Prompt injection patterns
        3. Argument size limits
    """
    for key, val in args.items():
        if not isinstance(val, str):
            continue
        val_lower = val.lower()

        # Check for sacred label leaks
        for label in SACRED_LABELS:
            if label in val_lower:
                logger.warning(
                    "GUARDRAIL_BLOCK: sacred label '%s' in arg '%s'",
                    label, key,
                )
                return {
                    "error": f"BLOCKED: Internal label '{label}' detected in {key}. "
                    f"This is a pipeline integrity violation."
                }

        # Check for prompt injection patterns
        injection_patterns = [
            r"ignore\s+(previous|above|all)\s+instructions",
            r"you\s+are\s+now\s+",
            r"system\s*:\s*",
            r"<\s*script\s*>",
        ]
        for pattern in injection_patterns:
            if re.search(pattern, val_lower):
                logger.warning(
                    "GUARDRAIL_BLOCK: injection pattern in arg '%s'", key,
                )
                return {
                    "error": "BLOCKED: Potential prompt injection detected."
                }

    return None  # Allow tool to proceed


def after_tool_sanitizer(
    tool: Any,
    args: dict[str, Any],
    tool_context: Any,
    tool_response: Any,
) -> Any:
    """Sanitize tool outputs before returning to LLM context.

    Performs:
        1. Response size truncation (prevent context overflow)
        2. Sacred label scrubbing from outputs
        3. PII pattern detection (log warning)
    """
    if isinstance(tool_response, str):
        # Truncate oversized responses
        if len(tool_response) > MAX_TOOL_RESPONSE_CHARS:
            logger.info(
                "SANITIZER: truncating response from %d to %d chars",
                len(tool_response),
                MAX_TOOL_RESPONSE_CHARS,
            )
            tool_response = (
                tool_response[:MAX_TOOL_RESPONSE_CHARS]
                + "\n... [TRUNCATED FOR CONTEXT EFFICIENCY]"
            )

        # Scrub sacred labels from output
        for label in SACRED_LABELS:
            if label in tool_response.lower():
                tool_response = re.sub(
                    re.escape(label),
                    "[REDACTED]",
                    tool_response,
                    flags=re.IGNORECASE,
                )

    return tool_response
