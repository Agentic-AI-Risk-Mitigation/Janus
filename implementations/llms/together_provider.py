import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from implementations.llms.base_provider import BaseLLMProvider


class TogetherProvider(BaseLLMProvider):
    """
    Together AI LLM Provider using OpenAI-compatible client.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the Together provider.

        Args:
            api_key: Optional API key. Defaults to TOGETHER_API_KEY env var.
            base_url: Optional base URL. Defaults to 'https://api.together.xyz/v1'.
        """
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY not set for TogetherProvider")

        self.base_url = base_url or "https://api.together.xyz/v1"
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
        Generate a response using Together AI via OpenAI client.
        """
        # Prepare parameters
        params = {"model": model_name, "messages": messages, "temperature": temperature, **kwargs}

        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        return self.client.chat.completions.create(**params)
