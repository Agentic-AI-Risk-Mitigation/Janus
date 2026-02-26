"""
Google Gemini LLM provider (google-genai SDK).

Translates OpenAI-format messages/tools to Google's Content/Part objects
and normalizes responses back to the unified shape.

Key differences:
- System instruction is part of GenerateContentConfig
- Assistant role is "model" (not "assistant")
- Tool calls are ``function_call`` Parts in a "model" Content
- Tool results are ``function_response`` Parts in a "user" Content
- Multiple tool results for one model turn go in one user Content
"""

import json
import os
import uuid
from typing import Any

from janus.llm.base import BaseLLMProvider
from janus.llm.response_types import Message, ToolCall, UnifiedResponse

try:
    from google import genai
    from google.genai import types as gtypes
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class GoogleProvider(BaseLLMProvider):
    """Google Gemini provider with OpenAI-format message translation."""

    def __init__(self, api_key: str | None = None, **kwargs):
        if not _AVAILABLE:
            raise ImportError("Install google-genai: uv add google-genai")

        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY is not set.")

        self.client = genai.Client(api_key=self.api_key)

    def generate(
        self,
        model_name: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.1,
        **kwargs: Any,
    ) -> UnifiedResponse:
        system, contents = self._translate_messages(messages)

        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "automatic_function_calling": gtypes.AutomaticFunctionCallingConfig(disable=True),
        }
        if system:
            config_kwargs["system_instruction"] = system
        if tools:
            config_kwargs["tools"] = self._translate_tools(tools)

        response = self.client.models.generate_content(
            model=model_name,
            contents=contents,
            config=gtypes.GenerateContentConfig(**config_kwargs),
        )
        return self._normalize(response)

    def _build_id_map(self, messages: list[dict]) -> dict[str, str]:
        """Map tool_call_id → function_name (Google needs the name for function_response)."""
        mapping: dict[str, str] = {}
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls", []):
                    mapping[tc["id"]] = tc["function"]["name"]
        return mapping

    def _translate_messages(self, messages: list[dict]) -> tuple[str | None, list]:
        system: str | None = None
        contents = []
        id_map = self._build_id_map(messages)
        i = 0

        while i < len(messages):
            msg = messages[i]
            role = msg["role"]

            if role == "system":
                system = msg["content"]
                i += 1

            elif role == "user":
                contents.append(gtypes.Content(role="user", parts=[gtypes.Part(text=msg["content"])]))
                i += 1

            elif role == "assistant":
                parts = []
                if msg.get("content"):
                    parts.append(gtypes.Part(text=msg["content"]))
                for tc in msg.get("tool_calls", []):
                    parts.append(gtypes.Part.from_function_call(
                        name=tc["function"]["name"],
                        args=json.loads(tc["function"]["arguments"]),
                    ))
                contents.append(gtypes.Content(role="model", parts=parts))
                i += 1

            elif role == "tool":
                parts = []
                while i < len(messages) and messages[i]["role"] == "tool":
                    tc_id = messages[i]["tool_call_id"]
                    func_name = id_map.get(tc_id, "unknown_function")
                    parts.append(gtypes.Part.from_function_response(
                        name=func_name,
                        response={"result": messages[i]["content"]},
                    ))
                    i += 1
                contents.append(gtypes.Content(role="user", parts=parts))

            else:
                i += 1

        return system, contents

    def _translate_tools(self, tools: list[dict]) -> list:
        declarations = []
        for t in tools:
            f = t["function"]
            declarations.append({
                "name": f["name"],
                "description": f.get("description", ""),
                "parameters": f["parameters"],
            })
        return [gtypes.Tool(function_declarations=declarations)]

    def _normalize(self, response) -> UnifiedResponse:
        content: str | None = None
        tool_calls: list[ToolCall] = []

        if response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    content = part.text
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_calls.append(ToolCall(
                        call_id=f"call_{fc.name}_{uuid.uuid4().hex[:8]}",
                        name=fc.name,
                        arguments=dict(fc.args),
                    ))

        return UnifiedResponse(Message(content, tool_calls or None))
