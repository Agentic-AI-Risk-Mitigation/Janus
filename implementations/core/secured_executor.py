"""
Unified secured tool executor with policy enforcement.

Wraps tool handlers with:
- Logging (before and after)
- Policy enforcement (Progent)
- Error handling

DO NOT implement policy enforcement in individual agent files.
"""

from typing import Callable

from implementations.core.logging_utils import get_logger
from implementations.core.progent_enforcer import enforce_policy
from implementations.core.tool_definitions import ToolDefinition


def create_secured_handler(tool_def: ToolDefinition) -> Callable:
    """
    Create a secured version of a tool handler.

    Wraps the original handler with:
    1. Logging of tool calls
    2. Policy enforcement check
    3. Error handling
    4. Logging of results

    Args:
        tool_def: The tool definition to wrap

    Returns:
        A wrapped function with security and logging
    """
    original_handler = tool_def.handler
    tool_name = tool_def.name

    def secured_handler(**kwargs) -> str:
        logger = get_logger()

        # Log the call (redact large content for readability)
        log_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, str) and len(v) > 100:
                log_kwargs[k] = f"[{len(v)} chars]"
            else:
                log_kwargs[k] = v
        logger.tool_call(tool_name, log_kwargs)

        # Policy enforcement
        allowed, reason = enforce_policy(tool_name, kwargs)
        if not allowed:
            result = f"Policy blocked: {reason}"
            logger.tool_result(tool_name, result, success=False)
            return result

        # Execute the tool
        try:
            result = original_handler(**kwargs)

            # Log success (truncate long results)
            log_result = str(result)
            if len(log_result) > 200:
                log_result = log_result[:200] + "..."
            logger.tool_result(tool_name, log_result, success=True)

            return result

        except Exception as e:
            result = f"Error: {e}"
            logger.tool_result(tool_name, result, success=False)
            return result

    # Preserve metadata for frameworks that inspect it
    secured_handler.__name__ = tool_name
    secured_handler.__doc__ = tool_def.description

    return secured_handler


def get_secured_handlers(tool_definitions: list[ToolDefinition]) -> dict[str, Callable]:
    """
    Create secured handlers for all tool definitions.

    Returns:
        Dict mapping tool name to secured handler function
    """
    return {tool_def.name: create_secured_handler(tool_def) for tool_def in tool_definitions}
