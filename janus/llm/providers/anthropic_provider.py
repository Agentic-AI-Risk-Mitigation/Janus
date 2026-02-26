"""
Anthropic (Claude) LLM provider.

Translates OpenAI-format messages/tools to Anthropic's API format and
normalizes responses back to the unified OpenAI-compatible shape.

Key differences handled:
- System prompt is a top-level parameter (not a message)
- Tool definitions use ``input_schema`` instead of ``parameters``
- Tool calls in history are ``tool_use`` content blocks (assistant turn)
- Tool results are ``tool_result`` blocks inside a *user* turn
- Multiple consecutive tool results must be batched into one user message
"""

import json
import os
from typing import Any

from janus.llm.base import BaseLLMProvider
from janus.llm.response_types import Message, ToolCall, UnifiedResponse

try:
    import anthropic
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider with OpenAI-format message translation."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, **kwargs):
        if not _AVAILABLE:
            raise ImportError("Install anthropic: uv add anthropic")

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")

        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = anthropic.Anthropic(**client_kwargs)

    def generate(
        self,
        model_name: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.1,
        **kwargs: Any,
    ) -> UnifiedResponse:
        system, translated = self._translate_messages(messages)

        params: dict[str, Any] = {
            "model": model_name,
            "messages": translated,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            "temperature": temperature,
            **kwargs,
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = self._translate_tools(tools)
            params["tool_choice"] = {"type": "none" if tool_choice == "none" else "auto"}

        response = self.client.messages.create(**params)
        return self._normalize(response)

    def _translate_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        system: str | None = None
        translated: list[dict[str, Any]] = []
        i = 0

        while i < len(messages):
            msg = messages[i]
            role = msg["role"]

            if role == "system":
                system = msg["content"]
                i += 1

            elif role == "user":
                translated.append({"role": "user", "content": msg["content"]})
                i += 1

            elif role == "assistant":
                blocks: list[dict] = []
                if msg.get("content"):
                    blocks.append({"type": "text", "text": msg["content"]})
                for tc in msg.get("tool_calls", []):
                    blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": json.loads(tc["function"]["arguments"]),
                    })
                translated.append({"role": "assistant", "content": blocks})
                i += 1

            elif role == "tool":
                tool_results: list[dict] = []
                while i < len(messages) and messages[i]["role"] == "tool":
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": messages[i]["tool_call_id"],
                        "content": messages[i]["content"],
                    })
                    i += 1
                translated.append({"role": "user", "content": tool_results})

            else:
                i += 1

        return system, translated

    def _translate_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "name": t["function"]["name"],
                "description": t["function"].get("description", ""),
                "input_schema": t["function"]["parameters"],
            }
            for t in tools
        ]

    def _normalize(self, response) -> UnifiedResponse:
        content: str | None = None
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(call_id=block.id, name=block.name, arguments=block.input)
                )

        return UnifiedResponse(Message(content, tool_calls or None))
