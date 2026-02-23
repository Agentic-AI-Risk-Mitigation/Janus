import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from implementations.llms.base_provider import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI LLM Provider.
    Handles calls to OpenAI API.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the OpenAI provider.

        Args:
            api_key: Optional API key. Defaults to OPENAI_API_KEY env var.
            base_url: Optional base URL. Defaults to OpenAI's default.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set for OpenAIProvider")

        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

    def generate(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.1,
        **kwargs,
    ) -> Any:
        """
        Generate a response using OpenAI.
        """
        # Prepare parameters
        params = {"model": model_name, "messages": messages, "temperature": temperature, **kwargs}

        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        return self.client.chat.completions.create(**params)
