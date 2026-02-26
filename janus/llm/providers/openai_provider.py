"""OpenAI LLM provider."""

import os
from typing import Any

from janus.llm.base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """
    Provider for OpenAI's chat completions API.

    Environment variable: OPENAI_API_KEY
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None, **kwargs):
        from openai import OpenAI

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

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
