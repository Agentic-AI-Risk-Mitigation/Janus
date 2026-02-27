"""
Janus × Google ADK (Gemini) integration adapter.

Provides two integration depths:

┌─────────────────────────────────────────────────────────────────────┐
│  Depth 1 — secure_adk_tools()                                        │
│  Convert Janus ToolDef list →                                        │
│    • Gemini FunctionDeclaration list  (for GenerateContentConfig)   │
│    • Guarded handler dict             (for your function-call loop)  │
│  Use when: you build your own Gemini chat loop.                      │
├─────────────────────────────────────────────────────────────────────┤
│  Depth 2 — JanusADKAgent                                             │
│  Turnkey agent: provide model + tools + policy, call .run().         │
│  The Gemini chat session and function-calling loop are managed       │
│  internally; Janus guards every tool execution.                      │
└─────────────────────────────────────────────────────────────────────┘

Installation requirement:
    uv add google-genai
"""

from __future__ import annotations

from typing import Any, Callable

from janus.adapters._base import PolicySource, make_guarded_handler, resolve_enforcer
from janus.logger import get_logger
from janus.policy.enforcer import PolicyEnforcer
from janus.tools.base import ToolDef

try:
    from google import genai
    from google.genai import types as gtypes

    _ADK_AVAILABLE = True
except ImportError:
    _ADK_AVAILABLE = False

_logger = get_logger()


def _require_adk() -> None:
    if not _ADK_AVAILABLE:
        raise ImportError(
            "google-genai is required for the ADK adapter.\n"
            "Install with: uv add google-genai"
        )


# =============================================================================
# Depth 1 — Convert ToolDef list → ADK-native types + guarded handlers
# =============================================================================


def secure_adk_tools(
    tools: list[ToolDef],
    policy: PolicySource,
) -> tuple[list, dict[str, Callable]]:
    """
    Convert a Janus ``ToolDef`` list into the two things a Gemini function-
    calling loop needs, with Janus policy enforcement built into the handlers.

    Returns
    -------
    declarations : list
        A list of ``{"name": ..., "description": ..., "parameters": ...}``
        dicts ready to be passed to ``types.Tool(function_declarations=...)``.
    handlers : dict[str, Callable]
        ``{tool_name: guarded_handler}`` — call ``handlers[name](**args)``
        inside your function-calling loop.  Blocked calls return a
        descriptive error string instead of raising.

    Example — minimal Gemini chat loop with Janus::

        from janus.adapters.adk import secure_adk_tools
        from google import genai
        from google.genai import types

        declarations, handlers = secure_adk_tools(my_tools, "policies.json")

        client = genai.Client(api_key="...")
        config = types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=declarations)],
            system_instruction="You are helpful.",
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        chat = client.chats.create(model="gemini-2.0-flash", config=config)

        response = chat.send_message("List the files")
        while response.function_calls:
            fc = response.function_calls[0]
            result = handlers[fc.name](**dict(fc.args))
            response = chat.send_message(
                types.Part.from_function_response(fc.name, {"result": result})
            )
        print(response.text)
    """
    _require_adk()
    enforcer = resolve_enforcer(policy)

    declarations: list[dict] = []
    handlers: dict[str, Callable] = {}

    for tool_def in tools:
        declarations.append({
            "name": tool_def.name,
            "description": tool_def.description,
            "parameters": tool_def._parameters_schema(),
        })
        handlers[tool_def.name] = make_guarded_handler(
            tool_def.name, tool_def.handler, enforcer
        )

    _logger.info(f"[adk adapter] Secured {len(declarations)} ADK tools.")
    return declarations, handlers


# =============================================================================
# Depth 2 — JanusADKAgent (turnkey)
# =============================================================================


class JanusADKAgent:
    """
    A policy-enforced Google ADK (Gemini) agent.

    Janus handles the security layer; the ``google-genai`` SDK drives the
    Gemini chat session and function-calling loop.

    Parameters
    ----------
    model : str
        Gemini model name, e.g. ``"gemini-2.0-flash"``, ``"gemini-1.5-pro"``.
        Only the model name is needed (no provider prefix).
    tools : list[ToolDef]
        Janus tool definitions.
    policy : str | Path | dict | PolicyEnforcer | None
        Policy source.  Pass a JSON file path, dict, enforcer, or ``None``.
    system_prompt : str
        System instruction passed to the Gemini model.
    api_key : str | None
        Google API key.  Falls back to GOOGLE_API_KEY / GEMINI_API_KEY env var.
    max_tool_iterations : int
        Maximum function-call cycles per ``run()`` call.

    Attributes
    ----------
    enforcer : PolicyEnforcer
        The active policy enforcer.
    declarations : list
        The raw Gemini FunctionDeclaration dicts.
    handlers : dict[str, Callable]
        The guarded handler dict.
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        tools: list[ToolDef] | None = None,
        policy: PolicySource = None,
        system_prompt: str = "You are a helpful assistant.",
        api_key: str | None = None,
        max_tool_iterations: int = 10,
    ):
        _require_adk()
        import os

        self.enforcer = resolve_enforcer(policy)
        self._max_iterations = max_tool_iterations
        self._logger = get_logger()

        # Resolve API key
        resolved_key = (
            api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        if not resolved_key:
            raise ValueError(
                "A Google API key is required. Set GOOGLE_API_KEY or GEMINI_API_KEY, "
                "or pass api_key= to JanusADKAgent."
            )

        # Build secured tool primitives
        self.declarations, self.handlers = secure_adk_tools(
            tools or [], self.enforcer
        )

        # Gemini client + chat config
        self._client = genai.Client(api_key=resolved_key)
        self._model = model
        self._system_prompt = system_prompt

        self._genai_config = gtypes.GenerateContentConfig(
            tools=[gtypes.Tool(function_declarations=self.declarations)] if self.declarations else [],
            system_instruction=system_prompt,
            automatic_function_calling=gtypes.AutomaticFunctionCallingConfig(disable=True),
        )

        # Start the chat session
        self._chat = self._client.chats.create(
            model=self._model,
            config=self._genai_config,
        )

        self._logger.agent_event(
            "INIT",
            f"framework=adk model={model} tools={list(self.handlers.keys())}",
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> str:
        """
        Send a user message and return the agent's final text response.

        Drives the Gemini function-calling loop: if the model returns tool
        calls, they are enforced and executed, results are fed back, and the
        loop continues until a text response or the iteration limit is reached.

        Args:
            user_input: The user's message or task.

        Returns:
            The model's final text response.
        """
        self._logger.agent_event("USER_INPUT", user_input[:120])

        try:
            response = self._chat.send_message(user_input)
            iteration = 0

            while response.function_calls and iteration < self._max_iterations:
                iteration += 1

                # Execute all function calls in this turn
                function_responses: list = []
                for fc in response.function_calls:
                    result = self._execute(fc)
                    function_responses.append(
                        gtypes.Part.from_function_response(
                            name=fc.name,
                            response={"result": result},
                        )
                    )

                # Feed all results back at once
                response = self._chat.send_message(function_responses)

            text = response.text or "No response generated."
            self._logger.agent_event("RESPONSE", text[:120])
            return text

        except Exception as exc:
            error = f"Agent error: {type(exc).__name__}: {exc}"
            self._logger.error(error)
            return error

    def clear_history(self) -> None:
        """Reset the chat session (starts a fresh conversation)."""
        self._chat = self._client.chats.create(
            model=self._model,
            config=self._genai_config,
        )
        self._logger.agent_event("HISTORY_CLEARED")

    def list_tools(self) -> list[str]:
        """Return the names of all registered tools."""
        return list(self.handlers.keys())

    def get_policy(self) -> dict | None:
        """Return the current policy dict."""
        return self.enforcer.policy

    def allow_tools(self, tools: list[str]) -> None:
        """Unconditionally allow the given tools."""
        self.enforcer.allow_tools(tools)

    def block_tools(self, tools: list[str]) -> None:
        """Unconditionally block the given tools."""
        self.enforcer.block_tools(tools)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute(self, fc) -> str:
        """Execute a single Gemini function call through the guarded handler."""
        name = fc.name
        args = dict(fc.args) if fc.args else {}

        handler = self.handlers.get(name)
        if handler is None:
            return f"Unknown tool '{name}'."
        return handler(**args)
