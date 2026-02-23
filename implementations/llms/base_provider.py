from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
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
        Generate a response from the LLM.

        Args:
            model_name: The name of the model to use.
            messages: List of message dictionaries (role, content).
            tools: Optional list of tool definitions.
            tool_choice: Tool choice strategy.
            temperature: Sampling temperature.
            **kwargs: Additional provider-specific arguments.

        Returns:
            The raw response object from the provider, or a standardized response wrapper.
        """
        pass
