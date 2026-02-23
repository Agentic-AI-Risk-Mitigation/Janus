"""
Progent LangChain adapter.

Provides integration with LangChain tools and agents.

Usage:
    from progent.adapters.langchain import secure_langchain_tools

    # Wrap existing LangChain tools
    secured_tools = secure_langchain_tools(original_tools)

    # Or create a secured tool from a function
    from progent.adapters.langchain import create_secured_tool
    tool = create_secured_tool(my_function, name="my_tool", description="...")
"""

from functools import wraps
from typing import Any, Callable, Type

try:
    from langchain_core.tools import BaseTool, StructuredTool

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseTool = None
    StructuredTool = None

from progent.core import Tool, check_tool_call, get_available_tools, update_available_tools


def _check_langchain() -> None:
    """Verify LangChain is available."""
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "langchain-core is required for LangChain integration. "
            "Install with: pip install progent[langchain]"
        )


def secure_langchain_tools(tools: list) -> list:
    """
    Wrap a list of LangChain tools with Progent policy enforcement.

    Args:
        tools: List of LangChain BaseTool instances

    Returns:
        List of wrapped tools with policy enforcement
    """
    _check_langchain()

    return [secure_langchain_tool(tool) for tool in tools]


def secure_langchain_tool(tool: Any) -> Any:
    """
    Wrap a single LangChain tool with Progent policy enforcement.

    The tool is modified in-place by wrapping its _to_args_and_kwargs method
    to intercept argument processing and apply policy checks.

    Args:
        tool: A LangChain BaseTool instance

    Returns:
        The same tool, modified with policy enforcement
    """
    _check_langchain()

    if not isinstance(tool, BaseTool):
        raise TypeError(f"Expected BaseTool, got {type(tool)}")

    # Store original method
    original_method = tool._to_args_and_kwargs

    @wraps(original_method)
    def wrapped_to_args_and_kwargs(*args, **kwargs):
        result = original_method(*args, **kwargs)
        tool_args, tool_kwargs = result

        if tool_args:
            raise NotImplementedError(
                "Progent does not support positional arguments in LangChain tools. "
                "Please use keyword arguments only."
            )

        # Check policy
        check_tool_call(tool.name, tool_kwargs)

        return result

    tool._to_args_and_kwargs = wrapped_to_args_and_kwargs

    # Register tool with Progent
    args = tool.args.copy()

    # Remove $ref entries (internal LangChain references)
    for arg in list(args.keys()):
        if "$ref" in args.get(arg, {}):
            del args[arg]

    # Add to available tools
    _available = get_available_tools()
    if not any(t["name"] == tool.name for t in _available):
        _available.append(
            Tool(
                name=tool.name,
                description=tool.description,
                args=args,
            )
        )
        update_available_tools(_available)

    return tool


def create_secured_tool(
    func: Callable,
    name: str = None,
    description: str = None,
    args_schema: Type = None,
) -> Any:
    """
    Create a secured LangChain StructuredTool from a function.

    This creates a LangChain tool that automatically enforces
    Progent policies on every call.

    Args:
        func: The function to wrap
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)
        args_schema: Optional Pydantic model for arguments

    Returns:
        A LangChain StructuredTool with policy enforcement
    """
    _check_langchain()

    tool_name = name or func.__name__
    tool_description = description or (func.__doc__ or "").strip().split("\n")[0]

    @wraps(func)
    def secured_func(**kwargs):
        # Policy check
        check_tool_call(tool_name, kwargs)
        # Execute
        return func(**kwargs)

    # Create StructuredTool
    if args_schema:
        tool = StructuredTool.from_function(
            func=secured_func,
            name=tool_name,
            description=tool_description,
            args_schema=args_schema,
        )
    else:
        tool = StructuredTool.from_function(
            func=secured_func,
            name=tool_name,
            description=tool_description,
        )

    # Register with Progent
    _available = get_available_tools()
    if not any(t["name"] == tool_name for t in _available):
        _available.append(
            Tool(
                name=tool_name,
                description=tool_description,
                args=tool.args,
            )
        )
        update_available_tools(_available)

    return tool


def tools_to_progent_format(tools: list) -> list[dict]:
    """
    Convert LangChain tools to Progent tool format.

    Args:
        tools: List of LangChain BaseTool instances

    Returns:
        List of tool definitions in Progent format
    """
    _check_langchain()

    result = []

    for tool in tools:
        if not isinstance(tool, BaseTool):
            continue

        args = tool.args.copy()
        for arg in list(args.keys()):
            if "$ref" in args.get(arg, {}):
                del args[arg]

        result.append(
            {
                "name": tool.name,
                "description": tool.description,
                "args": args,
            }
        )

    return result
