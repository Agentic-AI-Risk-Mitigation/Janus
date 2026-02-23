"""
Anthropic (Claude) LLM Provider.

Anthropic uses a fundamentally different API shape from OpenAI:
- System prompt is a top-level parameter, not a message
- Tool definitions use `input_schema` instead of `parameters`
- Tool calls in history are `tool_use` content blocks (assistant turn)
- Tool results are `tool_result` content blocks inside a *user* turn
- Multiple consecutive tool results must be merged into one user message

This provider translates OpenAI-format messages → Anthropic format on the way
in, calls the Anthropic API, then normalises the response back to the unified
OpenAI-compatible shape expected by raw_sdk_agent.py.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from implementations.llms.base_provider import BaseLLMProvider
from implementations.llms.response_types import Message, ToolCall, UnifiedResponse

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider with OpenAI-format message translation."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed. Install with: uv add anthropic")

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set for AnthropicProvider")

        client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = anthropic.Anthropic(**client_kwargs)

    # ------------------------------------------------------------------
    # Message translation: OpenAI format → Anthropic format
    # ------------------------------------------------------------------

    def _translate_messages(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert OpenAI-format message list to (system_str, anthropic_messages).

        Key differences handled:
        - `role: system` is extracted as a top-level `system` parameter
        - `role: assistant` with tool_calls becomes assistant content blocks of
          type `text` + `tool_use`
        - `role: tool` results are merged into a single `role: user` message
          containing `tool_result` content blocks
        """
        system: Optional[str] = None
        translated: List[Dict[str, Any]] = []

        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg["role"]

            if role == "system":
                system = msg["content"]
                i += 1
                continue

            if role == "user":
                translated.append({"role": "user", "content": msg["content"]})
                i += 1
                continue

            if role == "assistant":
                content_blocks: List[Dict[str, Any]] = []
                if msg.get("content"):
                    content_blocks.append({"type": "text", "text": msg["content"]})
                for tc in msg.get("tool_calls", []):
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": json.loads(tc["function"]["arguments"]),
                        }
                    )
                translated.append({"role": "assistant", "content": content_blocks})
                i += 1
                continue

            if role == "tool":
                # Anthropic requires all tool results for a single assistant turn
                # to be batched inside one user message.
                tool_results: List[Dict[str, Any]] = []
                while i < len(messages) and messages[i]["role"] == "tool":
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": messages[i]["tool_call_id"],
                            "content": messages[i]["content"],
                        }
                    )
                    i += 1
                translated.append({"role": "user", "content": tool_results})
                continue

            i += 1

        return system, translated

    def _translate_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI tool format → Anthropic tool format."""
        return [
            {
                "name": t["function"]["name"],
                "description": t["function"].get("description", ""),
                "input_schema": t["function"]["parameters"],
            }
            for t in tools
        ]

    # ------------------------------------------------------------------
    # Response normalisation: Anthropic response → UnifiedResponse
    # ------------------------------------------------------------------

    def _normalize_response(self, response) -> UnifiedResponse:
        """Map an Anthropic Messages response to the unified OpenAI-compatible shape."""
        content: Optional[str] = None
        tool_calls: List[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(call_id=block.id, name=block.name, arguments=block.input)
                )

        message = Message(content=content, tool_calls=tool_calls if tool_calls else None)
        return UnifiedResponse(message)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.1,
        **kwargs,
    ) -> UnifiedResponse:
        system, translated_messages = self._translate_messages(messages)

        params: Dict[str, Any] = {
            "model": model_name,
            "messages": translated_messages,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            "temperature": temperature,
            **kwargs,
        }

        if system:
            params["system"] = system

        if tools:
            params["tools"] = self._translate_tools(tools)
            if tool_choice == "none":
                params["tool_choice"] = {"type": "none"}
            else:
                params["tool_choice"] = {"type": "auto"}

        response = self.client.messages.create(**params)
        return self._normalize_response(response)
