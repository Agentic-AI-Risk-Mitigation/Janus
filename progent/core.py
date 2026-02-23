"""
Progent core functionality.

Handles policy storage, tool registration, and policy enforcement.
"""

import sys
from typing import Any, TypedDict

from progent.exceptions import PolicyValidationError, ProgentBlockedError
from progent.logger import get_logger
from progent.validation import check_argument

_logger = get_logger()

# =============================================================================
# Global State
# =============================================================================

# Tool registry: list of {"name": str, "description": str, "args": dict}
_available_tools: list[dict] = []

# Security policy: {"tool_name": [(priority, effect, conditions, fallback), ...]}
# effect: 0 = allow, 1 = deny
# fallback: 0 = raise error, 1 = terminate, 2 = ask user
_security_policy: dict | None = None

# User query context (for dynamic policy updates)
_user_query: str | None = None


class Tool(TypedDict):
    """Tool definition format."""

    name: str
    description: str
    args: dict


# =============================================================================
# Tool Registration
# =============================================================================


def update_available_tools(tools: list[Tool]) -> None:
    """
    Register the available tools.

    Args:
        tools: List of tool definitions with name, description, and args schema
    """
    global _available_tools
    _available_tools = tools


def get_available_tools() -> list[Tool]:
    """Get the list of registered tools."""
    return _available_tools.copy()


# =============================================================================
# Policy Management
# =============================================================================


def update_security_policy(policy: dict) -> None:
    """
    Set the security policy.

    Args:
        policy: Dict mapping tool names to list of policy rules.
                Each rule is a tuple: (priority, effect, conditions, fallback)
    """
    global _security_policy
    _security_policy = policy
    _sort_policy()


def get_security_policy() -> dict | None:
    """Get the current security policy."""
    if _security_policy is None:
        return None
    return _security_policy.copy()


def reset_security_policy(include_manual: bool = False) -> None:
    """
    Reset the security policy.

    Args:
        include_manual: If True, clears all policies. If False, only clears
                       auto-generated policies (priority >= 100).
    """
    global _security_policy

    if include_manual:
        _security_policy = None
    else:
        _delete_generated_policies()


def update_always_allowed_tools(
    tools: list[str],
    allow_no_arg_tools: bool = False,
) -> None:
    """
    Mark tools as always allowed (highest priority allow rule).

    Args:
        tools: List of tool names to always allow
        allow_no_arg_tools: If True, also allow all tools with no arguments
    """
    global _security_policy

    if _security_policy is None:
        _security_policy = {}

    always_allowed = set(tools)

    if allow_no_arg_tools:
        for tool in _available_tools:
            if len(tool.get("args", {})) == 0:
                always_allowed.add(tool["name"])

    for tool_name in always_allowed:
        if tool_name not in _security_policy:
            _security_policy[tool_name] = [(1, 0, {}, 0)]
        else:
            # Insert at beginning (highest priority)
            _security_policy[tool_name].insert(0, (1, 0, {}, 0))

    _sort_policy()


def update_always_blocked_tools(tools: list[str]) -> None:
    """
    Mark tools as always blocked (highest priority deny rule).

    Args:
        tools: List of tool names to always block
    """
    global _security_policy

    if _security_policy is None:
        _security_policy = {}

    for tool_name in tools:
        if tool_name not in _security_policy:
            _security_policy[tool_name] = [(1, 1, {}, 0)]
        else:
            _security_policy[tool_name].insert(0, (1, 1, {}, 0))

    _sort_policy()


# =============================================================================
# Policy Enforcement
# =============================================================================


def check_tool_call(tool_name: str, kwargs: dict[str, Any]) -> None:
    """
    Check if a tool call is allowed by the current policy.

    Args:
        tool_name: Name of the tool being called
        kwargs: Arguments to the tool

    Raises:
        ProgentBlockedError: If the tool call is not allowed
    """
    global _security_policy

    if _security_policy is None:
        # No policy set - allow by default (with warning)
        return

    policies = _security_policy.get(tool_name)
    _logger.tool_call(tool_name, kwargs)

    if policies is None or len(policies) == 0:
        _logger.progent_decision(tool_name, allowed=False, reason="Tool not in allowlist")
        raise ProgentBlockedError(
            tool_name=tool_name,
            arguments=kwargs,
            reason=f"Tool '{tool_name}' is not in the allowed tools list.",
        )

    try:
        _check_tool_call_impl(tool_name, kwargs, policies)
        _logger.progent_decision(tool_name, allowed=True)
    except ProgentBlockedError as e:
        _logger.progent_decision(tool_name, allowed=False, reason=e.reason)
        raise e


def _check_tool_call_impl(
    tool_name: str,
    kwargs: dict[str, Any],
    policies: list[tuple],
) -> None:
    """
    Internal implementation of policy checking.

    Policies are checked in priority order. For each policy:
    - effect=0 (allow): If all conditions pass, tool is allowed
    - effect=1 (deny): If all conditions match, tool is blocked
    """
    failed_reasons = []

    for policy in policies:
        # Unpack policy tuple
        if len(policy) >= 4:
            priority, effect, conditions, fallback = policy[:4]
        else:
            raise ValueError(f"Invalid policy format: {policy}")

        if effect == 0:  # Allow rule
            try:
                # Check all conditions
                for arg_name, restriction in conditions.items():
                    if arg_name in kwargs:
                        check_argument(arg_name, kwargs[arg_name], restriction)

                # All conditions passed - tool is allowed
                return

            except PolicyValidationError as e:
                # This allow rule doesn't match, record why and try next rule
                failed_reasons.append(f"Policy (priority {priority}) skipped: {str(e)}")
                continue

        elif effect == 1:  # Deny rule
            try:
                # Check if all conditions match
                for arg_name, restriction in conditions.items():
                    if arg_name in kwargs:
                        check_argument(arg_name, kwargs[arg_name], restriction)

                # All conditions matched - tool is blocked
                _handle_block(
                    tool_name,
                    kwargs,
                    fallback,
                    policy_rule=policy,
                    failed_condition=f"Matched deny rule: {conditions}",
                )

            except PolicyValidationError:
                # Deny rule doesn't match, continue
                continue

    # No rule matched - block by default
    if failed_reasons:
        reason = (
            f"Tool '{tool_name}' blocked. No matching allow rule found. Reasons:\n- "
            + "\n- ".join(failed_reasons)
        )
    else:
        reason = f"Tool '{tool_name}' blocked: no policy rule matched the provided arguments."

    raise ProgentBlockedError(
        tool_name=tool_name,
        arguments=kwargs,
        reason=reason,
    )


def _handle_block(
    tool_name: str,
    kwargs: dict[str, Any],
    fallback: int,
    policy_rule: tuple | None = None,
    failed_condition: str | None = None,
) -> None:
    """Handle a blocked tool call based on fallback setting."""
    if fallback == 0:  # Return error message
        raise ProgentBlockedError(
            tool_name=tool_name,
            arguments=kwargs,
            policy_rule=policy_rule,
            failed_condition=failed_condition,
        )
    elif fallback == 1:  # Terminate
        sys.exit(1)
    elif fallback == 2:  # Ask user
        print(
            f"[Progent] Agent wants to call {tool_name} with args {kwargs}. Allow? [y/N]: ",
            end="",
            flush=True,
        )
        response = input().strip().lower()
        if response != "y":
            raise ProgentBlockedError(
                tool_name=tool_name,
                arguments=kwargs,
                reason="Tool call rejected by user.",
                policy_rule=policy_rule,
                failed_condition=failed_condition,
            )
    else:
        raise ProgentBlockedError(
            tool_name=tool_name,
            arguments=kwargs,
            policy_rule=policy_rule,
            failed_condition=failed_condition,
        )


# =============================================================================
# Internal Helpers
# =============================================================================


def _sort_policy() -> None:
    """Sort policies by priority for consistent evaluation order."""
    global _security_policy

    if _security_policy is None:
        return

    for tool_name in _security_policy:
        # Sort by (priority, -effect) so allow rules come before deny at same priority
        _security_policy[tool_name] = sorted(
            _security_policy[tool_name],
            key=lambda x: (x[0], -x[1]),
        )


def _delete_generated_policies() -> None:
    """Remove auto-generated policies (priority >= 100)."""
    global _security_policy

    if _security_policy is None:
        return

    for tool_name in list(_security_policy.keys()):
        _security_policy[tool_name] = [p for p in _security_policy[tool_name] if p[0] < 100]
        if len(_security_policy[tool_name]) == 0:
            del _security_policy[tool_name]


def set_user_query(query: str) -> None:
    """Set the user query context for dynamic policy generation."""
    global _user_query
    _user_query = query


def get_user_query() -> str | None:
    """Get the current user query context."""
    return _user_query
