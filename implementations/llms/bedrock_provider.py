"""
AWS Bedrock LLM Provider.

Uses the Bedrock Converse API — a single unified interface that works across
all Bedrock foundation models (Claude, Llama, Mistral, Titan, etc.) and
supports tool use natively.

Authentication follows the standard AWS credential chain:
    1. Explicit kwargs (aws_access_key_id / aws_secret_access_key)
    2. Environment variables: AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY
    3. IAM role attached to the compute instance
    4. AWS profile in ~/.aws/credentials

Bedrock Converse API differences from OpenAI:
- System prompt is a separate top-level list, not a message
- Tool definitions use `toolSpec` / `inputSchema.json` nesting
- Tool calls in history are `toolUse` content blocks (assistant turn)
- Tool results are `toolResult` content blocks inside a *user* turn
- Messages use `content` as a list of typed blocks, not plain strings

Usage (model string format):
    bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
    bedrock/meta.llama3-70b-instruct-v1:0
    bedrock/mistral.mistral-large-2402-v1:0
    bedrock/amazon.titan-text-express-v1
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from implementations.llms.base_provider import BaseLLMProvider
from implementations.llms.response_types import Message, ToolCall, UnifiedResponse

try:
    import boto3

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class BedrockProvider(BaseLLMProvider):
    """AWS Bedrock provider using the Converse API."""

    def __init__(
        self,
        region: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        base_url: Optional[str] = None,  # endpoint_url override for VPC / LocalStack
        **kwargs,
    ):
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 package not installed. Install with: uv add boto3")

        self.region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

        session_kwargs: Dict[str, Any] = {}
        if aws_access_key_id:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key
        if aws_session_token:
            session_kwargs["aws_session_token"] = aws_session_token

        client_kwargs: Dict[str, Any] = {"region_name": self.region}
        if base_url:
            client_kwargs["endpoint_url"] = base_url

        session = boto3.Session(**session_kwargs)
        self.client = session.client("bedrock-runtime", **client_kwargs)

    # ------------------------------------------------------------------
    # Message translation: OpenAI format → Bedrock Converse format
    # ------------------------------------------------------------------

    def _translate_messages(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Optional[List[Dict]], List[Dict[str, Any]]]:
        """
        Convert OpenAI-format message list to (system_blocks, bedrock_messages).

        Bedrock Converse content blocks:
          Text      – {"text": "..."}
          Tool use  – {"toolUse": {"toolUseId": ..., "name": ..., "input": {...}}}
          Tool res  – {"toolResult": {"toolUseId": ..., "content": [{"text": "..."}]}}
        """
        system: Optional[List[Dict]] = None
        translated: List[Dict[str, Any]] = []

        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg["role"]

            if role == "system":
                system = [{"text": msg["content"]}]
                i += 1
                continue

            if role == "user":
                translated.append({"role": "user", "content": [{"text": msg["content"]}]})
                i += 1
                continue

            if role == "assistant":
                content_blocks: List[Dict[str, Any]] = []
                if msg.get("content"):
                    content_blocks.append({"text": msg["content"]})
                for tc in msg.get("tool_calls", []):
                    content_blocks.append(
                        {
                            "toolUse": {
                                "toolUseId": tc["id"],
                                "name": tc["function"]["name"],
                                "input": json.loads(tc["function"]["arguments"]),
                            }
                        }
                    )
                translated.append({"role": "assistant", "content": content_blocks})
                i += 1
                continue

            if role == "tool":
                # Batch consecutive tool results into a single user message
                result_blocks: List[Dict[str, Any]] = []
                while i < len(messages) and messages[i]["role"] == "tool":
                    result_blocks.append(
                        {
                            "toolResult": {
                                "toolUseId": messages[i]["tool_call_id"],
                                "content": [{"text": messages[i]["content"]}],
                            }
                        }
                    )
                    i += 1
                translated.append({"role": "user", "content": result_blocks})
                continue

            i += 1

        return system, translated

    def _translate_tools(self, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert OpenAI tool format → Bedrock toolConfig dict."""
        tool_specs = []
        for t in tools:
            f = t["function"]
            tool_specs.append(
                {
                    "toolSpec": {
                        "name": f["name"],
                        "description": f.get("description", ""),
                        "inputSchema": {"json": f["parameters"]},
                    }
                }
            )
        return {"tools": tool_specs}

    # ------------------------------------------------------------------
    # Response normalisation: Bedrock response → UnifiedResponse
    # ------------------------------------------------------------------

    def _normalize_response(self, response: Dict[str, Any]) -> UnifiedResponse:
        """Map a Bedrock Converse response dict to the unified OpenAI-compatible shape."""
        content: Optional[str] = None
        tool_calls: List[ToolCall] = []

        output_message = response.get("output", {}).get("message", {})
        for block in output_message.get("content", []):
            if "text" in block:
                content = block["text"]
            elif "toolUse" in block:
                tu = block["toolUse"]
                tool_calls.append(
                    ToolCall(
                        call_id=tu["toolUseId"],
                        name=tu["name"],
                        arguments=tu["input"],
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
        system, translated_messages = self._translate_messages(messages)

        params: Dict[str, Any] = {
            "modelId": model_name,
            "messages": translated_messages,
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
        return self._normalize_response(response)
