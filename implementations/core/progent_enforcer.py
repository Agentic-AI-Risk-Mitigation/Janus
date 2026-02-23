"""Progent policy enforcement wrapper for implementations.

This module provides a simplified interface to the progent library,
adding logging integration specific to this demo application.
"""

from pathlib import Path
from typing import Any, Callable

# Import from the new progent library
from progent import (
    ProgentBlockedError,
    check_tool_call,
    update_available_tools,
)
from progent import (
    load_policies as progent_load_policies,
)
from progent.registry import ProgentRegistry

from .logging_utils import get_logger


def load_policies(policies_path: str | Path) -> dict:
    """
    Load policies from a JSON file.

    This is a thin wrapper around progent.load_policies that returns
    the loaded policies dict.
    """
    return progent_load_policies(policies_path)


def init_progent(
    policies_path: str | Path,
    tools: list[dict[str, Any]],
) -> None:
    """
    Initialize Progent with policies and available tools.

    :param policies_path: Path to the policies JSON file
    :param tools: List of tool definitions in Progent format
    """
    # Update available tools
    update_available_tools(tools)

    # Load and apply policies
    policies = load_policies(policies_path)

    logger = get_logger()
    logger.info(f"Progent initialized with {len(tools)} tools and {len(policies)} policies")


def enforce_policy(tool_name: str, kwargs: dict[str, Any]) -> tuple[bool, str]:
    """
    Check if a tool call is allowed by the current policy.

    :param tool_name: Name of the tool being called
    :param kwargs: Arguments to the tool
    :return: Tuple of (allowed: bool, reason: str)
    """
    logger = get_logger()

    try:
        check_tool_call(tool_name, kwargs)
        logger.progent_decision(tool_name, kwargs, allowed=True)
        return True, ""
    except ProgentBlockedError as e:
        reason = str(e)
        logger.progent_decision(tool_name, kwargs, allowed=False, reason=reason)
        return False, reason
    except Exception as e:
        reason = f"Policy check error: {type(e).__name__}: {e}"
        logger.progent_decision(tool_name, kwargs, allowed=False, reason=reason)
        return False, reason


def wrap_tool_with_enforcement(func: Callable, tool_name: str) -> Callable:
    """
    Wrap a tool function with Progent policy enforcement.

    :param func: The tool function to wrap
    :param tool_name: Name of the tool (for policy lookup)
    :return: Wrapped function that enforces policies
    """

    def wrapper(*args, **kwargs):
        logger = get_logger()

        # Log the tool call
        logger.tool_call(tool_name, kwargs)

        # Check policy
        allowed, reason = enforce_policy(tool_name, kwargs)

        if not allowed:
            error_msg = f"Policy blocked: {reason}"
            logger.tool_result(tool_name, error_msg, success=False)
            return error_msg

        # Execute the tool
        try:
            result = func(*args, **kwargs)
            result_str = str(result)
            logger.tool_result(tool_name, result_str, success=True)
            return result
        except Exception as e:
            error_msg = f"Tool error: {type(e).__name__}: {e}"
            logger.tool_result(tool_name, error_msg, success=False)
            raise

    # Preserve function metadata
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__annotations__ = getattr(func, "__annotations__", {})

    return wrapper


class ProgentEnforcedRegistry(ProgentRegistry):
    """
    A tool registry that automatically enforces Progent policies
    and logs actions using the agent logger.
    """

    def __init__(self, policies_path: str | Path):
        super().__init__(policies_path=policies_path)

    def _wrap_tool(self, func: Callable, tool_name: str) -> Callable:
        """
        Wrap tool with the logging-enabled enforcement wrapper.
        """
        return wrap_tool_with_enforcement(func, tool_name)
