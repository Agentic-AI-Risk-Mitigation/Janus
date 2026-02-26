"""
LLM conversation runner.

The LLMRunner manages a stateful conversation loop with an LLM provider:
  1. Maintains the message history.
  2. Calls the provider for completions.
  3. Detects tool calls in the response and dispatches them to the
     ToolRegistry for execution (with policy enforcement baked in).
  4. Feeds tool results back into the conversation and continues until
     the model produces a final text response.

This module is intentionally decoupled from JanusAgent so it can be
tested independently or embedded in other frameworks.
"""

import json
from typing import Any, Callable

from janus.exceptions import PolicyViolation
from janus.llm.base import BaseLLMProvider
from janus.logger import get_logger
from janus.tools.registry import ToolRegistry


class LLMRunner:
    """
    Stateful conversation runner.

    Maintains message history across multiple ``run()`` calls, so follow-up
    queries can reference prior context.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        model_name: str,
        registry: ToolRegistry,
        system_prompt: str = "You are a helpful assistant.",
        max_tool_iterations: int = 10,
        temperature: float = 0.1,
    ):
        """
        Args:
            provider: Initialized LLM provider instance.
            model_name: Provider-specific model identifier.
            registry: ToolRegistry used to execute tool calls.
            system_prompt: System message prepended to every conversation.
            max_tool_iterations: Maximum number of tool-call → response cycles
                per ``run()`` call (guards against infinite loops).
            temperature: Sampling temperature passed to the provider.
        """
        self._provider = provider
        self._model = model_name
        self._registry = registry
        self._system_prompt = system_prompt
        self._max_iterations = max_tool_iterations
        self._temperature = temperature
        self._logger = get_logger()

        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> str:
        """
        Send a user message and return the model's final text response.

        The runner drives the full tool-calling loop internally:
        - Appends the user message.
        - Calls the LLM.
        - If tool calls are returned, executes them via the registry,
          appends results, and calls the LLM again.
        - Repeats until a text response is produced or max iterations reached.

        Args:
            user_input: The user's message.

        Returns:
            The model's final text response string.
        """
        self._logger.agent_event("USER_INPUT", user_input[:120])
        self._messages.append({"role": "user", "content": user_input})

        try:
            return self._drive_loop()
        except Exception as exc:
            error_msg = f"Agent error: {type(exc).__name__}: {exc}"
            self._logger.error(error_msg)
            return error_msg

    def clear_history(self) -> None:
        """Reset the conversation to the initial system prompt only."""
        self._messages = [{"role": "system", "content": self._system_prompt}]
        self._logger.agent_event("HISTORY_CLEARED")

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Read-only view of the current conversation history."""
        return list(self._messages)

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _drive_loop(self) -> str:
        tools_schema = self._registry.to_openai_schema()

        response = self._provider.generate(
            model_name=self._model,
            messages=self._messages,
            tools=tools_schema or None,
            tool_choice="auto",
            temperature=self._temperature,
        )

        assistant_msg = response.choices[0].message
        iteration = 0

        while assistant_msg.tool_calls and iteration < self._max_iterations:
            iteration += 1
            self._append_assistant(assistant_msg)

            for tool_call in assistant_msg.tool_calls:
                result = self._execute(tool_call)
                self._messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            response = self._provider.generate(
                model_name=self._model,
                messages=self._messages,
                tools=tools_schema or None,
                tool_choice="auto",
                temperature=self._temperature,
            )
            assistant_msg = response.choices[0].message

        final = assistant_msg.content or "No response generated."
        self._messages.append({"role": "assistant", "content": final})
        self._logger.agent_event("RESPONSE", final[:120])
        return final

    def _append_assistant(self, msg) -> None:
        """Serialize an assistant message with tool calls into history."""
        self._messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in (msg.tool_calls or [])
            ],
        })

    def _execute(self, tool_call) -> str:
        """Execute a single tool call, handling parse errors and policy violations."""
        name = tool_call.function.name

        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as exc:
            return f"Error: Could not parse arguments for '{name}': {exc}"

        try:
            return self._registry.execute(name, **args)
        except PolicyViolation as exc:
            return f"[Janus] Tool '{name}' was blocked by policy: {exc.reason}"
        except Exception as exc:
            return f"Error executing '{name}': {type(exc).__name__}: {exc}"
