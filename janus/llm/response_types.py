"""
Unified response types.

Providers whose API shapes differ from OpenAI (Anthropic, Google, Bedrock)
normalize their responses to these lightweight wrappers so the agent loop
can access ``response.choices[0].message.content`` and
``response.choices[0].message.tool_calls`` identically for every provider.
"""

import json
from typing import Any


class ToolFunction:
    """Function details within a tool call (mirrors OpenAI's ChatCompletionMessageToolCallFunction)."""

    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments  # JSON-encoded string


class ToolCall:
    """Unified tool call (mirrors OpenAI's ChatCompletionMessageToolCall)."""

    def __init__(self, call_id: str, name: str, arguments: dict[str, Any]):
        self.id = call_id
        self.type = "function"
        self.function = ToolFunction(name=name, arguments=json.dumps(arguments))


class Message:
    """Unified message (mirrors OpenAI's ChatCompletionMessage)."""

    def __init__(self, content: str | None, tool_calls: list[ToolCall] | None = None):
        self.content = content
        self.tool_calls = tool_calls or None


class Choice:
    """Single response choice (mirrors OpenAI's Choice)."""

    def __init__(self, message: Message):
        self.message = message


class UnifiedResponse:
    """
    OpenAI-compatible response envelope.

    Non-OpenAI providers wrap their response in this class so that the
    agent loop can treat every provider identically.
    """

    def __init__(self, message: Message):
        self.choices = [Choice(message)]
