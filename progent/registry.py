"""
Progent Tool Registry.

Provides a mechanism to register tools and automatically wrap them with
Progent policy enforcement.
"""

import inspect
from pathlib import Path
from typing import Any, Callable, Optional, TypedDict

from progent.core import (
    check_tool_call,
    update_available_tools,
    update_security_policy,
)
from progent.logger import get_logger
from progent.policy import load_policies

_logger = get_logger()


class ToolDefinition(TypedDict):
    name: str
    description: str
    args: dict


class ProgentRegistry:
    """
    A tool registry that automatically enforces Progent policies.
    """

    def __init__(self, policies_path: str | Path | None = None):
        """
        Initialize the registry.

        Args:
            policies_path: Optional path to load policies from on initialization.
                           If None, policies must be loaded manually or via
                           load_policies_from_file later.
        """
        self._tools: dict[str, Callable] = {}
        self._tool_definitions: list[ToolDefinition] = []
        self._initialized = False

        if policies_path:
            self.load_policies_from_file(policies_path)

    def register(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Callable:
        """
        Register a tool definition.
        NOTE: This does NOT wrap the function yet. The wrapping happens
        when you retrieve the tool or execute it, ensuring the latest
        policies are applied.
        """
        tool_name = name or func.__name__
        tool_desc = description or (func.__doc__ or "").strip().split("\n")[0]

        # Check for conflicts
        if tool_name in self._tools:
            # warn or error? For now, overwrite but maybe log warning
            pass

        self._tools[tool_name] = func

        # Build tool definition for Progent
        sig = inspect.signature(func)
        params = {}
        for pname, param in sig.parameters.items():
            # Basic type inference (could be improved)
            params[pname] = {"type": "string", "description": f"The {pname} parameter"}
            if param.annotation is int:
                params[pname]["type"] = "integer"
            elif param.annotation is float:
                params[pname]["type"] = "number"
            elif param.annotation is bool:
                params[pname]["type"] = "boolean"
            elif param.annotation is list:
                params[pname]["type"] = "array"
            elif param.annotation is dict:
                params[pname]["type"] = "object"

        self._tool_definitions.append(
            {
                "name": tool_name,
                "description": tool_desc,
                "args": params,
            }
        )

        _logger.info(f"Registered tool: {tool_name}")
        return func

    def load_policies_from_file(self, policies_path: str | Path) -> None:
        """Load policies from a JSON file."""
        policy_dict = load_policies(policies_path)
        update_security_policy(policy_dict)

    def initialize(self) -> None:
        """
        Finalize registration and update Progent core.

        This pushes the tool definitions to progent.core so validation
        and generation can work correctly.
        """
        if self._initialized:
            return

        update_available_tools(self._tool_definitions)
        self._initialized = True

    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a tool by name, wrapped with enforcement."""
        func = self._tools.get(name)
        if func is None:
            return None

        return self._wrap_tool(func, name)

    def get_all_tools(self) -> dict[str, Callable]:
        """Get all registered tools, wrapped with enforcement."""
        return {name: self._wrap_tool(func, name) for name, func in self._tools.items()}

    def execute(self, name: str, **kwargs) -> Any:
        """Execute a tool by name with enforcement."""
        if not self._initialized:
            self.initialize()

        tool = self.get_tool(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")

        return tool(**kwargs)

    def _wrap_tool(self, func: Callable, tool_name: str) -> Callable:
        """Internal method to wrap a tool with policy check."""

        def wrapper(*args, **kwargs):
            # 1. Map args to kwargs if necessary (simple version)
            # For now, we assume tools are called with kwargs as per agent convention
            # If args are present, we might loose policy check on them if logic depends on kwargs

            # 2. Check policy
            # Note: This raises ProgentBlockedError on failure
            check_tool_call(tool_name, kwargs)

            # 3. Execute
            return func(*args, **kwargs)

        # Preserve metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__

        return wrapper
