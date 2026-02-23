"""
Azure OpenAI LLM Provider.

Uses Microsoft Azure's hosted OpenAI endpoints via the `openai` SDK's
AzureOpenAI client. Authentication uses an API key + endpoint URL.

Required environment variables:
    AZURE_OPENAI_API_KEY      – Azure resource API key
    AZURE_OPENAI_ENDPOINT     – e.g. https://<resource>.openai.azure.com/
    AZURE_OPENAI_API_VERSION  – e.g. 2024-02-01 (optional, has a default)

The `model_name` passed to `generate()` must be the **deployment name** as
configured in your Azure resource, not the underlying model name.
"""

import os
from typing import Any, Dict, List, Optional

from openai import AzureOpenAI

from implementations.llms.base_provider import BaseLLMProvider

_DEFAULT_API_VERSION = "2024-02-01"


class AzureOpenAIProvider(BaseLLMProvider):
    """Azure OpenAI provider — returns native OpenAI ChatCompletion objects."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        **kwargs,
    ):
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not set for AzureOpenAIProvider")

        endpoint = base_url or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT not set for AzureOpenAIProvider")

        self.api_version = (
            api_version or os.environ.get("AZURE_OPENAI_API_VERSION") or _DEFAULT_API_VERSION
        )

        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=endpoint,
            api_version=self.api_version,
        )

    def generate(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.1,
        **kwargs,
    ) -> Any:
        """Generate using Azure OpenAI. `model_name` is the deployment name."""
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
