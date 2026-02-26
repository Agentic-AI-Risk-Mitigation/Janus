"""
vLLM LLM provider for self-hosted models.

vLLM's OpenAI-compatible server exposes the same API surface as OpenAI.
Default endpoint: http://localhost:8000/v1
Override via VLLM_BASE_URL env var or the base_url constructor argument.
"""

import os
from typing import Any

from janus.llm.base import BaseLLMProvider

_DEFAULT_BASE_URL = "http://localhost:8000/v1"


class VLLMProvider(BaseLLMProvider):
    """vLLM provider — returns native OpenAI ChatCompletion objects."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, **kwargs):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key or os.environ.get("VLLM_API_KEY", "vllm"),
            base_url=base_url or os.environ.get("VLLM_BASE_URL", _DEFAULT_BASE_URL),
        )

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
