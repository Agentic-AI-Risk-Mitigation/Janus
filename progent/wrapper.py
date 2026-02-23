"""
Progent tool wrapper utilities.

Provides the @secure decorator and tool wrapping functionality.
"""

import builtins
import inspect
import types
from functools import wraps
from typing import Any, Callable, TypeVar, overload

from docstring_parser import parse as parse_docstring
from pydantic import Field, create_model

from progent.core import (
    Tool,
    check_tool_call,
    get_available_tools,
)

# For typing
F = TypeVar("F", bound=Callable)


@overload
def apply_secure_tool_wrapper(func_or_tools: F) -> F: ...


@overload
def apply_secure_tool_wrapper(func_or_tools: list) -> list: ...


def apply_secure_tool_wrapper(func_or_tools):
    """
    Wrap a function or list of tools with Progent policy enforcement.

    Can be used as:
    1. Decorator: @secure
    2. Function wrapper: secure(my_func)
    3. List wrapper: secure([tool1, tool2])

    Example:
        @secure
        def my_tool(arg: str) -> str:
            return f"Executed with {arg}"

        # Or for existing tools:
        tools = secure([existing_tool1, existing_tool2])
    """
    if isinstance(func_or_tools, list):
        return [secure_tool_wrapper(t) for t in func_or_tools]
    else:
        return secure_tool_wrapper(func_or_tools)


def secure_tool_wrapper(tool: Any) -> Any:
    """
    Wrap a single tool with policy enforcement.

    Supports:
    - Plain Python functions
    - LangChain BaseTool instances (if langchain is installed)

    Args:
        tool: The tool to wrap

    Returns:
        Wrapped tool with policy enforcement
    """
    # Check for LangChain tool
    if _is_langchain_tool(tool):
        return _wrap_langchain_tool(tool)

    # Plain function
    return _wrap_function(tool)


def _wrap_function(func: Callable) -> Callable:
    """Wrap a plain Python function with policy enforcement."""
    # Extract tool metadata for registration
    tool_def = _extract_tool_definition(func)

    # Register tool
    _available_tools = get_available_tools()
    # Only add if not already registered
    if not any(t["name"] == tool_def["name"] for t in _available_tools):
        _available_tools.append(tool_def)
        from progent.core import update_available_tools

        update_available_tools(_available_tools)

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Bind arguments to parameter names
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # Check policy
        check_tool_call(func.__name__, bound.arguments)

        # Execute
        return func(*args, **kwargs)

    return wrapper


def _extract_tool_definition(func: Callable) -> Tool:
    """Extract tool definition from a function."""
    # Parse docstring
    doc = parse_docstring(func.__doc__ or "")
    description = doc.short_description or func.__name__

    # Get signature and type hints
    sig = inspect.signature(func)
    hints = {}
    try:
        from typing import get_type_hints

        hints = get_type_hints(func)
    except Exception:
        pass

    # Build parameter schema
    fields = {}
    for name, param in sig.parameters.items():
        type_hint = hints.get(name, Any)

        # Skip non-builtin types
        if not _is_builtin_type(type_hint):
            continue

        # Find description from docstring
        param_desc = ""
        for p in doc.params:
            if p.arg_name == name:
                param_desc = p.description or ""
                break

        # Create field
        if param.default is inspect.Parameter.empty:
            fields[name] = (type_hint, Field(..., description=param_desc))
        else:
            fields[name] = (type_hint, Field(default=param.default, description=param_desc))

    # Create Pydantic model for JSON schema
    if fields:
        model = create_model(func.__name__, **fields)
        args_schema = model.model_json_schema().get("properties", {})
    else:
        args_schema = {}

    return Tool(
        name=func.__name__,
        description=description,
        args=args_schema,
    )


def _is_builtin_type(type_hint) -> bool:
    """Check if a type hint is a builtin type."""
    from typing import Any, get_args, get_origin

    # Any is acceptable
    if type_hint is Any:
        return True

    def is_builtin(t):
        return t in vars(builtins).values() or t in vars(types).values()

    if is_builtin(type_hint):
        return True

    origin = get_origin(type_hint)
    if origin is not None:
        return all(_is_builtin_type(arg) for arg in get_args(type_hint))

    return False


def _is_langchain_tool(tool: Any) -> bool:
    """Check if an object is a LangChain tool."""
    try:
        from langchain_core.tools import BaseTool

        return isinstance(tool, BaseTool)
    except ImportError:
        return False


def _wrap_langchain_tool(tool: Any) -> Any:
    """Wrap a LangChain tool with policy enforcement."""
    from langchain_core.tools import BaseTool

    if not isinstance(tool, BaseTool):
        raise TypeError("Expected LangChain BaseTool")

    # Store original method
    original_to_args_and_kwargs = tool._to_args_and_kwargs

    @wraps(original_to_args_and_kwargs)
    def wrapped_to_args_and_kwargs(*args, **kwargs):
        result = original_to_args_and_kwargs(*args, **kwargs)
        tool_args, tool_kwargs = result

        if tool_args:
            raise NotImplementedError(
                "Progent does not support positional arguments in LangChain tools"
            )

        check_tool_call(tool.name, tool_kwargs)
        return result

    tool._to_args_and_kwargs = wrapped_to_args_and_kwargs

    # Register tool
    args = tool.args.copy()
    # Remove $ref entries
    for arg in list(args.keys()):
        if "$ref" in args.get(arg, {}):
            del args[arg]

    _available_tools = get_available_tools()
    if not any(t["name"] == tool.name for t in _available_tools):
        _available_tools.append(
            Tool(
                name=tool.name,
                description=tool.description,
                args=args,
            )
        )
        from progent.core import update_available_tools

        update_available_tools(_available_tools)

    return tool
