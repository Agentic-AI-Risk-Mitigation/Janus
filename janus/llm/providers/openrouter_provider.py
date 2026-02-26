"""OpenRouter LLM provider (OpenAI-compatible API)."""

import os
from typing import Any

from janus.llm.base import BaseLLMProvider


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter provider — returns native OpenAI ChatCompletion objects."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, **kwargs):
        from openai import OpenAI

        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url or "https://openrouter.ai/api/v1",
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
