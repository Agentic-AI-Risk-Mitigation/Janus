"""
Unified response types that normalize different LLM provider response formats
to a common OpenAI-compatible interface consumed by raw_sdk_agent.py.

Providers like Anthropic, Google, and AWS Bedrock use completely different
response structures. This module provides lightweight wrappers that mirror
the OpenAI ChatCompletion response shape so the agent loop doesn't need to
branch on provider type.
"""

import json
from typing import Any, Dict, List, Optional


class ToolFunction:
    """Function portion of a tool call (mirrors openai.types.chat.ChatCompletionMessageToolCallFunction)."""

    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments  # JSON-encoded string, matching OpenAI's format


class ToolCall:
    """Unified tool call (mirrors openai.types.chat.ChatCompletionMessageToolCall)."""

    def __init__(self, call_id: str, name: str, arguments: Dict[str, Any]):
        self.id = call_id
        self.type = "function"
        self.function = ToolFunction(name=name, arguments=json.dumps(arguments))


class Message:
    """Unified message (mirrors openai.types.chat.ChatCompletionMessage)."""

    def __init__(self, content: Optional[str], tool_calls: Optional[List[ToolCall]] = None):
        self.content = content
        # None when no tool calls — matches OpenAI behaviour (falsy either way)
        self.tool_calls = tool_calls if tool_calls else None


class Choice:
    """Single response choice (mirrors openai.types.chat.Choice)."""

    def __init__(self, message: Message):
        self.message = message


class UnifiedResponse:
    """
    OpenAI-compatible response envelope.

    Non-OpenAI providers normalise their responses to this type so that
    raw_sdk_agent.py can consume every provider identically via:
        response.choices[0].message.content
        response.choices[0].message.tool_calls
    """

    def __init__(self, message: Message):
        self.choices = [Choice(message)]
