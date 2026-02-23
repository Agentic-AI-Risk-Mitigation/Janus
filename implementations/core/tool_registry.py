"""Tool registration and discovery system."""

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, get_type_hints


@dataclass
class ToolDefinition:
    """Definition of a tool that can be used by the agent."""

    name: str
    description: str
    function: Callable
    parameters: dict[str, Any]
    required_params: list[str] = field(default_factory=list)

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required_params,
                },
            },
        }

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON schema for Progent."""
        return {
            "name": self.name,
            "description": self.description,
            "args": self.parameters,
        }


class ToolRegistry:
    """Registry for all available tools."""

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        func: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Callable:
        """
        Register a tool function. Can be used as a decorator.

        Usage:
            @registry.register
            def my_tool(arg: str) -> str:
                '''Tool description.'''
                ...

            @registry.register(name="custom_name", description="Custom description")
            def another_tool(arg: str) -> str:
                ...
        """

        def decorator(fn: Callable) -> Callable:
            tool_name = name or fn.__name__
            tool_desc = description or (fn.__doc__ or "").strip().split("\n")[0]

            # Extract parameter info from type hints and docstring
            params, required = self._extract_parameters(fn)

            tool_def = ToolDefinition(
                name=tool_name,
                description=tool_desc,
                function=fn,
                parameters=params,
                required_params=required,
            )

            self._tools[tool_name] = tool_def
            return fn

        if func is not None:
            return decorator(func)
        return decorator

    def _extract_parameters(self, func: Callable) -> tuple[dict[str, Any], list[str]]:
        """Extract parameter definitions from function signature and docstring."""
        sig = inspect.signature(func)
        type_hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
        docstring = func.__doc__ or ""

        # Parse docstring for parameter descriptions
        param_descriptions = self._parse_docstring_params(docstring)

        parameters = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            # Determine type
            python_type = type_hints.get(param_name, str)
            json_type = self._python_type_to_json(python_type)

            param_def = {
                "type": json_type,
                "description": param_descriptions.get(param_name, f"The {param_name} parameter"),
            }

            parameters[param_name] = param_def

            # Check if required (no default value)
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return parameters, required

    def _parse_docstring_params(self, docstring: str) -> dict[str, str]:
        """Parse parameter descriptions from docstring."""
        descriptions = {}
        lines = docstring.split("\n")

        for i, line in enumerate(lines):
            line = line.strip()
            # Support :param name: description format
            if line.startswith(":param "):
                parts = line[7:].split(":", 1)
                if len(parts) == 2:
                    param_name = parts[0].strip()
                    param_desc = parts[1].strip()
                    descriptions[param_name] = param_desc
            # Support Args: section format
            elif line.lower() == "args:":
                for j in range(i + 1, len(lines)):
                    arg_line = lines[j].strip()
                    if not arg_line or arg_line.endswith(":"):
                        break
                    if ":" in arg_line:
                        parts = arg_line.split(":", 1)
                        param_name = parts[0].strip()
                        param_desc = parts[1].strip()
                        descriptions[param_name] = param_desc

        return descriptions

    def _python_type_to_json(self, python_type: Any) -> str:
        """Convert Python type to JSON schema type."""
        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        # Handle Optional types
        origin = getattr(python_type, "__origin__", None)
        if origin is not None:
            # For Optional[X], get the inner type
            args = getattr(python_type, "__args__", ())
            for arg in args:
                if arg is not type(None):
                    return self._python_type_to_json(arg)

        return type_mapping.get(python_type, "string")

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all(self) -> list[ToolDefinition]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_names(self) -> list[str]:
        """Get all tool names."""
        return list(self._tools.keys())

    def execute(self, name: str, **kwargs) -> Any:
        """Execute a tool by name with given arguments."""
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")
        return tool.function(**kwargs)

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Convert all tools to OpenAI function calling format."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def to_progent_tools(self) -> list[dict[str, Any]]:
        """Convert all tools to Progent format."""
        return [tool.to_json_schema() for tool in self._tools.values()]


# Global registry instance
_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable:
    """
    Decorator to register a tool with the global registry.

    Usage:
        @tool
        def my_tool(arg: str) -> str:
            '''Tool description.'''
            ...
    """
    registry = get_registry()
    return registry.register(func, name=name, description=description)
