"""
Google Gemini LLM Provider using the google-genai SDK.

Google's API differs from OpenAI in several ways:
- System instruction is part of GenerateContentConfig, not a message
- Assistant role is called "model" (not "assistant")
- Tool calls in history are `function_call` Parts in a "model" Content
- Tool results are `function_response` Parts in a "user" Content
- Multiple tool results for one model turn go in one user Content object
- function_response Parts require the *function name*, not just the call id,
  so we build a tool_call_id → function_name map from assistant messages

This provider translates OpenAI-format messages → Google Content objects on the
way in, calls models.generate_content (stateless), then normalises the response
back to the unified OpenAI-compatible shape expected by raw_sdk_agent.py.
"""

import json
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

from implementations.llms.base_provider import BaseLLMProvider
from implementations.llms.response_types import Message, ToolCall, UnifiedResponse

try:
    from google import genai
    from google.genai import types

    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


class GoogleProvider(BaseLLMProvider):
    """Google Gemini provider with OpenAI-format message translation."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        if not GOOGLE_AVAILABLE:
            raise ImportError(
                "google-genai package not installed. Install with: uv add google-genai"
            )

        self.api_key = (
            api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        )
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not set for GoogleProvider")

        self.client = genai.Client(api_key=self.api_key)

    # ------------------------------------------------------------------
    # Message translation: OpenAI format → Google Content objects
    # ------------------------------------------------------------------

    def _build_tool_id_map(self, messages: List[Dict[str, Any]]) -> Dict[str, str]:
        """Return {tool_call_id: function_name} from all assistant messages.

        Google's function_response Part requires the function *name*, not the
        call id, so we pre-scan the history to build this reverse mapping.
        """
        mapping: Dict[str, str] = {}
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls", []):
                    mapping[tc["id"]] = tc["function"]["name"]
        return mapping

    def _translate_messages(self, messages: List[Dict[str, Any]]) -> Tuple[Optional[str], List]:
        """Convert OpenAI-format message list to (system_str, google_contents).

        Returns typed Content / Part objects from google.genai.types so that
        the SDK does not have to infer types from plain dicts at call time.
        """
        system: Optional[str] = None
        translated = []
        tool_id_map = self._build_tool_id_map(messages)

        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg["role"]

            if role == "system":
                system = msg["content"]
                i += 1
                continue

            if role == "user":
                translated.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(text=msg["content"])],
                    )
                )
                i += 1
                continue

            if role == "assistant":
                parts = []
                if msg.get("content"):
                    parts.append(types.Part(text=msg["content"]))
                for tc in msg.get("tool_calls", []):
                    # Reconstruct a function_call Part from serialised history
                    parts.append(
                        types.Part.from_function_call(
                            name=tc["function"]["name"],
                            args=json.loads(tc["function"]["arguments"]),
                        )
                    )
                translated.append(types.Content(role="model", parts=parts))
                i += 1
                continue

            if role == "tool":
                # All tool results for one model turn go in one user Content.
                parts = []
                while i < len(messages) and messages[i]["role"] == "tool":
                    tc_id = messages[i]["tool_call_id"]
                    func_name = tool_id_map.get(tc_id, "unknown_function")
                    parts.append(
                        types.Part.from_function_response(
                            name=func_name,
                            response={"result": messages[i]["content"]},
                        )
                    )
                    i += 1
                translated.append(types.Content(role="user", parts=parts))
                continue

            i += 1

        return system, translated

    def _translate_tools(self, tools: List[Dict[str, Any]]) -> List:
        """Convert OpenAI tool format → Google Tool with FunctionDeclarations."""
        declarations = []
        for t in tools:
            f = t["function"]
            declarations.append(
                {
                    "name": f["name"],
                    "description": f.get("description", ""),
                    "parameters": f["parameters"],
                }
            )
        return [types.Tool(function_declarations=declarations)]

    # ------------------------------------------------------------------
    # Response normalisation: Google response → UnifiedResponse
    # ------------------------------------------------------------------

    def _normalize_response(self, response) -> UnifiedResponse:
        """Map a Google GenerateContent response to the unified OpenAI-compatible shape."""
        content: Optional[str] = None
        tool_calls: List[ToolCall] = []

        if response.candidates:
            candidate = response.candidates[0]
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    content = part.text
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_calls.append(
                        ToolCall(
                            call_id=f"call_{fc.name}_{uuid.uuid4().hex[:8]}",
                            name=fc.name,
                            arguments=dict(fc.args),
                        )
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
        system, translated_contents = self._translate_messages(messages)

        config_kwargs: Dict[str, Any] = {
            "temperature": temperature,
            # Disable automatic function calling so we drive the loop ourselves
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }

        if system:
            config_kwargs["system_instruction"] = system

        if tools:
            config_kwargs["tools"] = self._translate_tools(tools)

        response = self.client.models.generate_content(
            model=model_name,
            contents=translated_contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return self._normalize_response(response)
