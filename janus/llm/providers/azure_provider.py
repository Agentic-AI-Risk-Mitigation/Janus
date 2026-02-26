"""
Azure OpenAI LLM provider.

Environment variables:
    AZURE_OPENAI_API_KEY     – Azure resource API key
    AZURE_OPENAI_ENDPOINT    – e.g. https://<resource>.openai.azure.com/
    AZURE_OPENAI_API_VERSION – API version (default: 2024-02-01)

The model_name passed to generate() must be the deployment name.
"""

import os
from typing import Any

from janus.llm.base import BaseLLMProvider

_DEFAULT_API_VERSION = "2024-02-01"


class AzureOpenAIProvider(BaseLLMProvider):
    """Azure OpenAI provider — returns native OpenAI ChatCompletion objects."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        api_version: str | None = None,
        **kwargs,
    ):
        from openai import AzureOpenAI

        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is not set.")

        endpoint = base_url or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is not set.")

        self.api_version = (
            api_version
            or os.environ.get("AZURE_OPENAI_API_VERSION")
            or _DEFAULT_API_VERSION
        )

        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=endpoint,
            api_version=self.api_version,
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
