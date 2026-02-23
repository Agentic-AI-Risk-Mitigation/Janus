"""
Google ADK (Agent Development Kit) based agent implementation.

This is a THIN ADAPTER that:
1. Converts unified tool definitions to Gemini FunctionDeclaration format
2. Initializes the Gemini model and handles function calling

Tool definitions and policy enforcement are handled by core modules.
DO NOT define tools or policy logic here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from google import genai
    from google.genai import types

    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False

from implementations.core.progent_enforcer import init_progent
from implementations.core.secured_executor import create_secured_handler
from implementations.core.tool_definitions import TOOL_DEFINITIONS
from implementations.frameworks.base_agent import BaseAgent


class ADKAgent(BaseAgent):
    """
    Agent implementation using Google's Gen AI SDK.

    Uses Gemini models with function calling capabilities.
    """

    def __init__(
        self,
        config: dict[str, Any],
        workspace: str | Path,
        policies_path: Optional[str | Path] = None,
    ):
        if not ADK_AVAILABLE:
            raise ImportError(
                "Google Gen AI SDK not installed. Install with: pip install google-genai"
            )

        super().__init__(config, workspace)

        # Get API key from environment
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")

        self.client = genai.Client(api_key=api_key)

        # Get model from config
        llm_config = config.get("llm", {})
        self.model_name = llm_config.get("model", "gemini-2.0-flash")

        # Convert OpenRouter model names to Gemini
        if "/" in self.model_name:
            self.model_name = "gemini-2.0-flash"

        # Initialize Progent policies
        if policies_path:
            tool_defs = [
                {"name": t.name, "description": t.description, "args": {}} for t in TOOL_DEFINITIONS
            ]
            init_progent(policies_path, tool_defs)

        # Create tools (converted from unified definitions)
        self.tool_handlers = self._create_handlers()
        self.tools = self._create_tools()

        # Build config for chat sessions
        self.genai_config = types.GenerateContentConfig(
            tools=self.tools,
            system_instruction=config.get("agent", {}).get("system_prompt", ""),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        # Start chat session
        self.chat = self.client.chats.create(model=self.model_name, config=self.genai_config)

    def _create_handlers(self) -> dict[str, Any]:
        """Create secured handlers for all tools."""
        return {tool_def.name: create_secured_handler(tool_def) for tool_def in TOOL_DEFINITIONS}

    def _create_tools(self) -> list[types.Tool]:
        """Convert unified tool definitions to Gemini FunctionDeclaration format."""
        declarations = []

        for tool_def in TOOL_DEFINITIONS:
            declarations.append(
                {
                    "name": tool_def.name,
                    "description": tool_def.description,
                    "parameters": tool_def.to_json_schema(),
                }
            )

        return [types.Tool(function_declarations=declarations)]

    def _execute_function_call(self, function_call) -> str:
        """Execute a function call from the model."""
        name = function_call.name
        args = dict(function_call.args)

        if name in self.tool_handlers:
            return self.tool_handlers[name](**args)
        return f"Unknown function: {name}"

    def run(self, user_input: str) -> str:
        """Process user input and return response."""
        try:
            response = self.chat.send_message(user_input)

            # Process function calls
            while response.function_calls:
                fc = response.function_calls[0]
                result = self._execute_function_call(fc)

                function_response_part = types.Part.from_function_response(
                    name=fc.name,
                    response={"result": result},
                )
                response = self.chat.send_message(function_response_part)

            if response.text:
                return response.text
            return "No response generated."

        except Exception as e:
            self.logger.error("ADK agent execution error", e)
            return f"Error: {e}"

    def get_tools(self) -> list:
        """Get the list of tools."""
        return self.tools

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.chat = self.client.chats.create(model=self.model_name, config=self.genai_config)
