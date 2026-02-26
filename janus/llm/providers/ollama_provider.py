"""
Ollama LLM provider for locally-hosted models.

Ollama exposes an OpenAI-compatible REST API.
Default endpoint: http://localhost:11434/v1
Override via OLLAMA_BASE_URL env var or the base_url constructor argument.
"""

import os
from typing import Any

from janus.llm.base import BaseLLMProvider

_DEFAULT_BASE_URL = "http://localhost:11434/v1"


class OllamaProvider(BaseLLMProvider):
    """Ollama provider — returns native OpenAI ChatCompletion objects."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, **kwargs):
        from openai import OpenAI

        effective_url = base_url or os.environ.get("OLLAMA_BASE_URL", _DEFAULT_BASE_URL)
        self.client = OpenAI(api_key=api_key or "ollama", base_url=effective_url)

    def generate(
        self,
        model_name: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.1,
        **kwargs: Any,
    ) -> Any:
        params: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            **kwargs,
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice
        return self.client.chat.completions.create(**params)
