"""
Abstract base class for LLM providers.

All providers must implement ``generate()`` and return a response that
conforms to the unified OpenAI-compatible interface (via ``UnifiedResponse``
or the native OpenAI ChatCompletion object, which is structurally identical).
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseLLMProvider(ABC):
    """Abstract interface for an LLM provider."""

    @abstractmethod
    def generate(
        self,
        model_name: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.1,
        **kwargs: Any,
    ) -> Any:
        """
        Send a chat completion request and return the response.

        Args:
            model_name: Provider-specific model identifier.
            messages: Conversation history in OpenAI message format.
            tools: Optional list of tool schemas in OpenAI function format.
            tool_choice: Tool selection strategy ("auto", "none", or a specific tool).
            temperature: Sampling temperature.
            **kwargs: Additional provider-specific parameters.

        Returns:
            A response object with ``choices[0].message.content`` and
            ``choices[0].message.tool_calls`` accessible.
        """
        ...
