"""
Tool registry — register, look up, and securely execute tools.

The ToolRegistry acts as the single runtime container for all tools an agent
can use. When a PolicyEnforcer is attached, every tool execution is
automatically gated by policy enforcement, logging, and error handling.

Usage::

    from janus.tools import ToolRegistry, ToolDef, ToolParam

    registry = ToolRegistry(enforcer=my_enforcer)

    registry.register(ToolDef(
        name="read_file",
        description="Read a file",
        params=[ToolParam("file_path", "string", "Path to file")],
        handler=read_file_fn,
    ))

    result = registry.execute("read_file", file_path="data.csv")
"""

from typing import Any, Callable

from janus.exceptions import PolicyViolation, ToolNotFoundError
from janus.logger import get_logger
from janus.tools.base import ToolDef


class ToolRegistry:
    """
    Container for tool definitions with optional policy enforcement.

    All tools executed through the registry are automatically:
    1. Looked up (raises ``ToolNotFoundError`` if unknown).
    2. Policy-checked (raises ``PolicyViolation`` if blocked).
    3. Logged (DEBUG level before/after execution).
    4. Error-wrapped (exceptions from the handler are caught and returned
       as error strings so the LLM can see them).
    """

    def __init__(self, enforcer: "PolicyEnforcer | None" = None):
        """
        Args:
            enforcer: Optional ``PolicyEnforcer`` instance. If None, no
                      policy enforcement is applied — useful for testing
                      or for frameworks that handle enforcement externally.
        """
        self._tools: dict[str, ToolDef] = {}
        self._enforcer = enforcer
        self._logger = get_logger()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: ToolDef) -> None:
        """
        Register a single tool.

        Overwrites any existing registration with the same name.
        """
        self._tools[tool.name] = tool
        self._logger.debug(f"Registered tool '{tool.name}'.")

    def register_many(self, tools: list[ToolDef]) -> None:
        """Register a list of tools."""
        for tool in tools:
            self.register(tool)

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry."""
        self._tools.pop(name, None)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> ToolDef:
        """
        Retrieve a tool definition by name.

        Raises:
            ToolNotFoundError: If no tool with the given name is registered.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(name)
        return tool

    def names(self) -> list[str]:
        """Return a sorted list of all registered tool names."""
        return sorted(self._tools.keys())

    def all(self) -> list[ToolDef]:
        """Return all registered tool definitions."""
        return list(self._tools.values())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    # ------------------------------------------------------------------
    # Schema conversion (for LLM APIs)
    # ------------------------------------------------------------------

    def to_openai_schema(self) -> list[dict]:
        """
        Build the tool list for the OpenAI chat completions API.

        Returns a list of ``{"type": "function", "function": {...}}`` dicts.
        """
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def to_janus_specs(self) -> list[dict]:
        """
        Build the tool spec list used by the Janus policy engine.

        Returns ``[{"name": ..., "description": ..., "args": {...}}, ...]``
        """
        return [tool.to_janus_tool_spec() for tool in self._tools.values()]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, tool_name: str, **kwargs: Any) -> str:
        """
        Execute a registered tool by name, applying policy enforcement.

        This is the primary execution path called by the agent loop.

        Args:
            tool_name: Name of the tool to call.
            **kwargs: Arguments forwarded to the tool handler.

        Returns:
            The tool's return value as a string. If the handler raises an
            exception, an error message string is returned (so the LLM can
            reason about the failure).

        Raises:
            ToolNotFoundError: If the tool is not registered.
            PolicyViolation: If the tool call is blocked by the enforcer.
        """
        tool = self.get(tool_name)

        # Redact large argument values in logs
        log_args = {
            k: (f"[{len(v)} chars]" if isinstance(v, str) and len(v) > 100 else v)
            for k, v in kwargs.items()
        }
        self._logger.tool_call(tool_name, log_args)

        # Policy enforcement (may raise PolicyViolation)
        if self._enforcer is not None:
            self._enforcer.enforce(tool_name, kwargs)

        # Execute the tool handler
        try:
            result = tool.handler(**kwargs)
            result_str = str(result) if not isinstance(result, str) else result
            self._logger.tool_result(tool_name, result_str, success=True)
            return result_str
        except Exception as exc:
            error_msg = f"Tool '{tool_name}' raised an error: {type(exc).__name__}: {exc}"
            self._logger.tool_result(tool_name, error_msg, success=False)
            return error_msg

    def make_handler(self, tool_name: str) -> Callable[..., str]:
        """
        Return a callable that executes the named tool through this registry.

        Useful for building the ``tool_handlers`` dict expected by agent loops::

            handlers = {name: registry.make_handler(name) for name in registry.names()}
        """
        def _handler(**kwargs: Any) -> str:
            return self.execute(tool_name, **kwargs)

        _handler.__name__ = tool_name
        tool_def = self._tools.get(tool_name)
        if tool_def:
            _handler.__doc__ = tool_def.description
        return _handler

    def make_all_handlers(self) -> dict[str, Callable[..., str]]:
        """Return a ``{name: handler}`` dict for every registered tool."""
        return {name: self.make_handler(name) for name in self._tools}
