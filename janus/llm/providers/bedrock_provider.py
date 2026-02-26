"""
AWS Bedrock LLM provider (Converse API).

Authentication follows the standard AWS credential chain:
    1. Constructor kwargs (aws_access_key_id / aws_secret_access_key)
    2. Environment variables: AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY
    3. IAM role attached to the compute instance
    4. ~/.aws/credentials profile

Bedrock Converse API differences from OpenAI:
- System prompt is a separate top-level list
- Tool definitions use toolSpec / inputSchema.json nesting
- Tool calls/results use typed content blocks
"""

import json
import os
from typing import Any

from janus.llm.base import BaseLLMProvider
from janus.llm.response_types import Message, ToolCall, UnifiedResponse

try:
    import boto3
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class BedrockProvider(BaseLLMProvider):
    """AWS Bedrock provider using the Converse API."""

    def __init__(
        self,
        region: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        if not _AVAILABLE:
            raise ImportError("Install boto3: uv add boto3")

        self.region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

        session_kwargs: dict[str, Any] = {}
        if aws_access_key_id:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key
        if aws_session_token:
            session_kwargs["aws_session_token"] = aws_session_token

        client_kwargs: dict[str, Any] = {"region_name": self.region}
        if base_url:
            client_kwargs["endpoint_url"] = base_url

        session = boto3.Session(**session_kwargs)
        self.client = session.client("bedrock-runtime", **client_kwargs)

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
            "modelId": model_name,
            "messages": translated,
            "inferenceConfig": {
                "temperature": temperature,
                "maxTokens": kwargs.pop("max_tokens", 4096),
            },
        }
        if system:
            params["system"] = system
        if tools:
            params["toolConfig"] = self._translate_tools(tools)

        response = self.client.converse(**params)
        return self._normalize(response)

    def _translate_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict] | None, list[dict[str, Any]]]:
        system: list[dict] | None = None
        translated: list[dict[str, Any]] = []
        i = 0

        while i < len(messages):
            msg = messages[i]
            role = msg["role"]

            if role == "system":
                system = [{"text": msg["content"]}]
                i += 1

            elif role == "user":
                translated.append({"role": "user", "content": [{"text": msg["content"]}]})
                i += 1

            elif role == "assistant":
                blocks: list[dict] = []
                if msg.get("content"):
                    blocks.append({"text": msg["content"]})
                for tc in msg.get("tool_calls", []):
                    blocks.append({
                        "toolUse": {
                            "toolUseId": tc["id"],
                            "name": tc["function"]["name"],
                            "input": json.loads(tc["function"]["arguments"]),
                        }
                    })
                translated.append({"role": "assistant", "content": blocks})
                i += 1

            elif role == "tool":
                result_blocks: list[dict] = []
                while i < len(messages) and messages[i]["role"] == "tool":
                    result_blocks.append({
                        "toolResult": {
                            "toolUseId": messages[i]["tool_call_id"],
                            "content": [{"text": messages[i]["content"]}],
                        }
                    })
                    i += 1
                translated.append({"role": "user", "content": result_blocks})

            else:
                i += 1

        return system, translated

    def _translate_tools(self, tools: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "tools": [
                {
                    "toolSpec": {
                        "name": t["function"]["name"],
                        "description": t["function"].get("description", ""),
                        "inputSchema": {"json": t["function"]["parameters"]},
                    }
                }
                for t in tools
            ]
        }

    def _normalize(self, response: dict[str, Any]) -> UnifiedResponse:
        content: str | None = None
        tool_calls: list[ToolCall] = []

        for block in response.get("output", {}).get("message", {}).get("content", []):
            if "text" in block:
                content = block["text"]
            elif "toolUse" in block:
                tu = block["toolUse"]
                tool_calls.append(ToolCall(
                    call_id=tu["toolUseId"],
                    name=tu["name"],
                    arguments=tu["input"],
                ))

        return UnifiedResponse(Message(content, tool_calls or None))
