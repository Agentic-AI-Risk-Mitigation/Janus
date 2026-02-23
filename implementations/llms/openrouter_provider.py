import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from implementations.llms.base_provider import BaseLLMProvider


class OpenRouterProvider(BaseLLMProvider):
    """
    OpenRouter LLM Provider.
    Handles calls to OpenRouter API (OpenAI-compatible).
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the OpenRouter provider.

        Args:
            api_key: Optional API key. Defaults to OPENROUTER_API_KEY env var.
            base_url: Optional base URL. Defaults to OpenRouter's default.
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set for OpenRouterProvider")

        self.base_url = base_url or "https://openrouter.ai/api/v1"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

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
        Generate a response using OpenRouter.
        """
        # Prepare parameters
        params = {"model": model_name, "messages": messages, "temperature": temperature, **kwargs}

        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        # OpenRouter specific headers can be added via extra_headers in client or per request,
        # but basic usage works with standard OpenAI client.
        return self.client.chat.completions.create(**params)
