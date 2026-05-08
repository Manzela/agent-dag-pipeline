"""
LLM Client Protocol — Provider-Agnostic Interface.

Defines the typed contract that any LLM provider must implement to plug
into the pipeline. Ships with a ``MockLLMClient`` for deterministic
testing and reference implementations for major providers.

Usage (mock — zero config)::

    from agent_dag.shared.llm_protocol import MockLLMClient
    client = MockLLMClient()
    result = await client.generate("Describe this product")

Usage (OpenAI)::

    from agent_dag.shared.llm_protocol import OpenAILLMClient
    client = OpenAILLMClient(model="gpt-4o-mini", api_key="sk-...")
    result = await client.generate("Describe this product")

Usage (Google GenAI)::

    from agent_dag.shared.llm_protocol import GoogleGenAILLMClient
    client = GoogleGenAILLMClient(model="gemini-2.5-flash-lite")
    result = await client.generate("Describe this product")
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional, Protocol, runtime_checkable

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# PROTOCOL — The contract any LLM provider must satisfy
# ════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class LLMClient(Protocol):
    """Provider-agnostic LLM client interface.

    Any LLM provider (OpenAI, Google, Anthropic, local Ollama, etc.)
    can be used with the pipeline by implementing this protocol.

    The two methods cover all pipeline use cases:
        - ``generate``: Free-form text generation
        - ``generate_structured``: Structured output with Pydantic schema
    """

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        """Generate free-form text from a prompt.

        Parameters
        ----------
        prompt : str
            The user prompt to generate from.
        system_prompt : str
            Optional system-level instructions.
        temperature : float
            Sampling temperature (0.0 = deterministic, 1.0 = creative).
        max_tokens : int
            Maximum output token count.

        Returns
        -------
        str
            The generated text content.
        """
        ...

    async def generate_structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        *,
        system_prompt: str = "",
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> BaseModel:
        """Generate structured output conforming to a Pydantic schema.

        Parameters
        ----------
        prompt : str
            The user prompt.
        schema : type[BaseModel]
            The Pydantic model class to parse the output into.
        system_prompt : str
            Optional system-level instructions.
        temperature : float
            Sampling temperature (lower for structured output).

        Returns
        -------
        BaseModel
            An instance of the provided schema, populated with generated data.
        """
        ...


# ════════════════════════════════════════════════════════════════════════════
# MOCK CLIENT — Deterministic, zero-config, no API key needed
# ════════════════════════════════════════════════════════════════════════════

class MockLLMClient:
    """Deterministic mock LLM client for testing and demo purposes.

    Returns plausible template-based content so the full pipeline can be
    exercised end-to-end without any external API calls or credentials.
    """

    def __init__(self, *, latency_ms: float = 0) -> None:
        self.latency_ms = latency_ms
        self.call_count = 0

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        """Return deterministic template-based text."""
        self.call_count += 1

        if self.latency_ms > 0:
            import asyncio
            await asyncio.sleep(self.latency_ms / 1000)

        # Extract product name from prompt if present
        product_hint = _extract_hint(prompt, "product")
        city_hint = _extract_hint(prompt, "city")

        return (
            f"This is a high-quality {product_hint} available at our {city_hint} location. "
            f"Featuring premium materials and expert craftsmanship, this product combines "
            f"modern design with exceptional functionality. Perfect for discerning customers "
            f"who value quality and style in their everyday lives."
        )

    async def generate_structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        *,
        system_prompt: str = "",
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> BaseModel:
        """Return a schema instance populated with plausible mock data."""
        self.call_count += 1

        # Build mock data based on schema fields
        mock_data: dict[str, Any] = {}
        for field_name, field_info in schema.model_fields.items():
            annotation = field_info.annotation
            if annotation is str or annotation == Optional[str]:
                mock_data[field_name] = f"Mock {field_name.replace('_', ' ')}"
            elif annotation is int:
                mock_data[field_name] = 42
            elif annotation is float:
                mock_data[field_name] = 0.85
            elif annotation is bool:
                mock_data[field_name] = True
            elif annotation is list or str(annotation).startswith("list"):
                mock_data[field_name] = []

        return schema(**mock_data)


# ════════════════════════════════════════════════════════════════════════════
# REFERENCE IMPLEMENTATIONS — Optional, require provider SDKs
# ════════════════════════════════════════════════════════════════════════════

class OpenAILLMClient:
    """Reference implementation for OpenAI-compatible APIs.

    Requires: ``pip install openai``

    Works with OpenAI, Azure OpenAI, and any OpenAI-compatible endpoint
    (Ollama, vLLM, LiteLLM, etc.) by setting ``base_url``.

    Parameters
    ----------
    model : str
        Model identifier (e.g., "gpt-4o-mini", "gpt-4o").
    api_key : str, optional
        API key. Falls back to ``OPENAI_API_KEY`` env var.
    base_url : str, optional
        Custom API base URL for compatible endpoints.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI client requires the 'openai' package. "
                "Install with: pip install openai"
            ) from exc

        self.model = model
        self._client = AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    async def generate_structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        *,
        system_prompt: str = "",
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> BaseModel:
        text = await self.generate(
            prompt + f"\n\nRespond ONLY with valid JSON matching this schema:\n{schema.model_json_schema()}",
            system_prompt=system_prompt,
            temperature=temperature,
        )
        # Parse JSON from response
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            import re
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
            if match:
                data = json.loads(match.group(1))
            else:
                raise
        return schema(**data)


class GoogleGenAILLMClient:
    """Reference implementation for Google Generative AI (Gemini).

    Requires: ``pip install google-genai``

    Parameters
    ----------
    model : str
        Model identifier (e.g., "gemini-2.5-flash-lite").
    api_key : str, optional
        API key. Falls back to ``GOOGLE_API_KEY`` env var.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash-lite",
        *,
        api_key: Optional[str] = None,
    ) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "Google GenAI client requires the 'google-genai' package. "
                "Install with: pip install google-genai"
            ) from exc

        self.model = model
        self._client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system_prompt:
            config.system_instruction = system_prompt

        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        return response.text or ""

    async def generate_structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        *,
        system_prompt: str = "",
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> BaseModel:
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=schema,
        )
        if system_prompt:
            config.system_instruction = system_prompt

        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        data = json.loads(response.text) if isinstance(response.text, str) else {}
        return schema(**data)


# ════════════════════════════════════════════════════════════════════════════
# FACTORY — Create client from config
# ════════════════════════════════════════════════════════════════════════════

def create_llm_client(
    provider: str = "mock",
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> LLMClient:
    """Factory function to create an LLM client from a provider name.

    Parameters
    ----------
    provider : str
        One of "mock", "openai", "google". Default is "mock".
    model : str, optional
        Model identifier for the provider.
    api_key : str, optional
        API key for the provider.
    base_url : str, optional
        Custom API base URL (OpenAI-compatible endpoints only).

    Returns
    -------
    LLMClient
        An instance satisfying the LLMClient protocol.
    """
    provider = provider.lower().strip()

    if provider == "mock":
        return MockLLMClient()

    if provider == "openai":
        return OpenAILLMClient(
            model=model or "gpt-4o-mini",
            api_key=api_key,
            base_url=base_url,
        )

    if provider in ("google", "gemini"):
        return GoogleGenAILLMClient(
            model=model or "gemini-2.5-flash-lite",
            api_key=api_key,
        )

    raise ValueError(
        f"Unknown LLM provider: '{provider}'. "
        f"Supported: mock, openai, google"
    )


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _extract_hint(prompt: str, keyword: str) -> str:
    """Extract a hint word near a keyword from a prompt for mock responses."""
    lower = prompt.lower()
    idx = lower.find(keyword)
    if idx == -1:
        return keyword
    # Grab a few words after the keyword
    after = prompt[idx:idx + 60].split()
    return " ".join(after[1:4]) if len(after) > 1 else keyword
