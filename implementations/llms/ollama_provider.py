"""
Ollama LLM Provider for locally-hosted models.

Ollama exposes an OpenAI-compatible REST API, so this provider reuses the
standard `openai` SDK pointed at the local Ollama server.

Default endpoint: http://localhost:11434/v1
Override via OLLAMA_BASE_URL env var or the `base_url` constructor argument.

No API key is required; we pass the string "ollama" as a placeholder because
the OpenAI client requires a non-empty value.

Usage (model string format):
    ollama/llama3.2
    ollama/mistral
    ollama/qwen2.5-coder:7b
"""

import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from implementations.llms.base_provider import BaseLLMProvider

_DEFAULT_BASE_URL = "http://localhost:11434/v1"


class OllamaProvider(BaseLLMProvider):
    """Ollama provider — returns native OpenAI ChatCompletion objects."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", _DEFAULT_BASE_URL)
        # Ollama does not require auth; placeholder satisfies the SDK validation
        self.client = OpenAI(api_key=api_key or "ollama", base_url=self.base_url)

    def generate(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.1,
        **kwargs,
    ) -> Any:
        params: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            **kwargs,
        }

        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        return self.client.chat.completions.create(**params)
