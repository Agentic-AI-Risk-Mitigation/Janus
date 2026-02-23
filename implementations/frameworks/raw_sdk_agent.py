"""
Raw SDK agent implementation using abstracted LLM providers.

This is a THIN ADAPTER that:
1. Converts unified tool definitions to OpenAI function-calling format
2. Handles the tool-calling loop manually
3. Delegates LLM calls to the appropriate provider based on model name

Tool definitions and policy enforcement are handled by core modules.
DO NOT define tools or policy logic here.

Supported model string format: "<provider>/<model-name>"

    openai/gpt-4o
    anthropic/claude-3-5-sonnet-20241022
    google/gemini-2.0-flash          (also: gemini/...)
    azure/<deployment-name>
    ollama/llama3.2
    vllm/meta-llama/Llama-3.3-70B-Instruct
    bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0   (also: aws/...)
    together/meta-llama/Llama-3-70b-chat-hf
    openrouter/anthropic/claude-3.5-sonnet

Provider-specific config keys (under llm: in config.yaml):
    api_key      – override env-var API key
    api_base     – override API endpoint URL (base_url)
    api_version  – Azure API version (default: 2024-02-01)
    region       – AWS region for Bedrock (default: us-east-1)
"""

import json
import sys
from pathlib import Path
from typing import Any, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from implementations.core.progent_enforcer import init_progent
from implementations.core.secured_executor import create_secured_handler
from implementations.core.tool_definitions import TOOL_DEFINITIONS
from implementations.frameworks.base_agent import BaseAgent
from implementations.llms import get_llm_provider


class RawSDKAgent(BaseAgent):
    """
    Agent implementation using abstracted LLM providers.

    Accepts any provider registered in implementations.llms.
    Model must be specified as 'provider/model-name'.
    """

    def __init__(
        self,
        config: dict[str, Any],
        workspace: str | Path,
        policies_path: Optional[str | Path] = None,
    ):
        super().__init__(config, workspace)

        # Get LLM config
        llm_config = config.get("llm", {})
        full_model_name = llm_config.get("model", "openai/gpt-4o")

        # Parse provider and model name
        if "/" in full_model_name:
            provider_parts = full_model_name.split("/", 1)
            # Handle case where split might result in unexpected parts, though 2 is expected
            self.provider_name = provider_parts[0]
            self.model_name = provider_parts[1]
        else:
            # print a helpful error message
            print("Error: Invalid model name. Please use the format 'provider/model-name'.")

        # Provider configuration — collect all optional provider-specific keys
        provider_kwargs = {}
        if llm_config.get("api_base"):
            provider_kwargs["base_url"] = llm_config["api_base"]
        if llm_config.get("api_key"):
            provider_kwargs["api_key"] = llm_config["api_key"]
        # Azure: deployment API version
        if llm_config.get("api_version"):
            provider_kwargs["api_version"] = llm_config["api_version"]
        # Bedrock: AWS region
        if llm_config.get("region"):
            provider_kwargs["region"] = llm_config["region"]

        # Initialize the provider
        try:
            self.provider = get_llm_provider(self.provider_name, **provider_kwargs)
        except ValueError as e:
            # If the provider name extracted (e.g. from meta-llama/...) is not supported,
            # and it was likely the old default, we might want to warn or fail.
            # For now, we fail as the new system requires explicit supported providers.
            raise ValueError(f"Failed to initialize LLM provider: {e}")

        # Initialize Progent policies
        if policies_path:
            tool_defs = [
                {"name": t.name, "description": t.description, "args": {}} for t in TOOL_DEFINITIONS
            ]
            init_progent(policies_path, tool_defs)

        # Create tools (converted from unified definitions)
        self.tool_handlers = self._create_handlers()
        self.tools_schema = self._create_tools_schema()

        # System prompt
        self.system_prompt = config.get("agent", {}).get(
            "system_prompt", "You are a helpful coding assistant."
        )

        # Conversation history
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def _create_handlers(self) -> dict[str, Any]:
        """Create secured handlers for all tools."""
        return {tool_def.name: create_secured_handler(tool_def) for tool_def in TOOL_DEFINITIONS}

    def _create_tools_schema(self) -> list[dict[str, Any]]:
        """Convert unified tool definitions to OpenAI function calling format."""
        tools = []

        for tool_def in TOOL_DEFINITIONS:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_def.name,
                        "description": tool_def.description,
                        "parameters": tool_def.to_json_schema(),
                    },
                }
            )

        return tools

    def _execute_tool_call(self, tool_call) -> str:
        """Execute a tool call and return the result."""
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return f"Error: Invalid JSON arguments: {tool_call.function.arguments}"

        if name in self.tool_handlers:
            return self.tool_handlers[name](**args)
        return f"Unknown function: {name}"

    def run(self, user_input: str) -> str:
        """Process user input and return response."""
        try:
            self.messages.append({"role": "user", "content": user_input})

            response = self.provider.generate(
                model_name=self.model_name,
                messages=self.messages,
                tools=self.tools_schema,
                tool_choice="auto",
                temperature=0.1,
            )

            assistant_message = response.choices[0].message

            # Process tool calls in a loop
            max_iterations = 10
            iteration = 0

            while assistant_message.tool_calls and iteration < max_iterations:
                iteration += 1

                self.messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in assistant_message.tool_calls
                        ],
                    }
                )

                for tool_call in assistant_message.tool_calls:
                    result = self._execute_tool_call(tool_call)
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                    )

                response = self.provider.generate(
                    model_name=self.model_name,
                    messages=self.messages,
                    tools=self.tools_schema,
                    tool_choice="auto",
                    temperature=0.1,
                )

                assistant_message = response.choices[0].message

            final_response = assistant_message.content or "No response generated."
            self.messages.append({"role": "assistant", "content": final_response})

            return final_response

        except Exception as e:
            self.logger.error("Raw SDK agent execution error", e)
            return f"Error: {e}"

    def get_tools(self) -> list:
        """Get the list of tools."""
        return self.tools_schema

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.messages = [{"role": "system", "content": self.system_prompt}]
