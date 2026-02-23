"""
vLLM LLM Provider for self-hosted models.

vLLM's OpenAI-compatible server exposes the same API surface as OpenAI, so
this provider reuses the standard `openai` SDK pointed at the vLLM endpoint.

Default endpoint: http://localhost:8000/v1
Override via VLLM_BASE_URL env var or the `base_url` constructor argument.

An optional API key can be set via VLLM_API_KEY (defaults to "vllm") — useful
when the vLLM server is deployed with `--api-key` authentication.

Usage (model string format):
    vllm/meta-llama/Llama-3.3-70B-Instruct
    vllm/Qwen/Qwen2.5-Coder-32B-Instruct
    vllm/mistralai/Mistral-7B-Instruct-v0.3
"""

import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from implementations.llms.base_provider import BaseLLMProvider

_DEFAULT_BASE_URL = "http://localhost:8000/v1"


class VLLMProvider(BaseLLMProvider):
    """vLLM provider — returns native OpenAI ChatCompletion objects."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        self.base_url = base_url or os.environ.get("VLLM_BASE_URL", _DEFAULT_BASE_URL)
        self.api_key = api_key or os.environ.get("VLLM_API_KEY", "vllm")
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
