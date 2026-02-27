"""
Shared utilities for all Janus framework adapters.

Every adapter (LangChain, ADK, …) needs to:
  1. Accept a policy from multiple source types and produce a PolicyEnforcer.
  2. Wrap a tool handler so that policy enforcement runs before execution,
     returning a safe error string if the call is blocked (rather than raising
     into the framework and breaking the agent loop).

These two helpers live here so each adapter stays thin.
"""

from pathlib import Path
from typing import Any, Callable

from janus.exceptions import PolicyViolation
from janus.logger import get_logger
from janus.policy.enforcer import PolicyEnforcer

PolicySource = "str | Path | dict | PolicyEnforcer"


def resolve_enforcer(policy: PolicySource) -> PolicyEnforcer:
    """
    Produce a ready-to-use PolicyEnforcer from multiple source types.

    Accepts:
    - ``PolicyEnforcer`` instance → returned as-is (shared state).
    - ``str`` / ``Path``         → treated as a path to a JSON policy file.
    - ``dict``                   → inline policy dict.
    - ``None``                   → empty enforcer (all tools allowed).

    Args:
        policy: The policy source.

    Returns:
        An initialized ``PolicyEnforcer``.
    """
    if isinstance(policy, PolicyEnforcer):
        return policy

    enforcer = PolicyEnforcer()
    if policy is not None:
        enforcer.load(policy)
    return enforcer


def make_guarded_handler(
    tool_name: str,
    handler: Callable,
    enforcer: PolicyEnforcer,
) -> Callable:
    """
    Return a new callable that enforces policy before invoking ``handler``.

    If the call is blocked, an informative error string is returned (not
    raised) so the LLM can reason about the blockage and explain it to the
    user.  Handler-side exceptions are also caught and returned as strings.

    Args:
        tool_name: Name used for policy lookup and log messages.
        handler:   Original tool implementation (keyword-args callable).
        enforcer:  PolicyEnforcer instance to consult before each call.

    Returns:
        A guarded callable with the same signature as ``handler``.
    """
    _logger = get_logger()

    def guarded(**kwargs: Any) -> str:
        # Log + enforce
        _logger.tool_call(tool_name, kwargs)
        try:
            enforcer.enforce(tool_name, kwargs)
        except PolicyViolation as exc:
            _logger.policy_decision(tool_name, allowed=False, reason=exc.reason)
            return f"[Janus] Tool '{tool_name}' was blocked by policy: {exc.reason}"

        _logger.policy_decision(tool_name, allowed=True)

        # Execute
        try:
            result = handler(**kwargs)
            return str(result) if not isinstance(result, str) else result
        except Exception as exc:
            error = f"Tool '{tool_name}' raised an error: {type(exc).__name__}: {exc}"
            _logger.error(error)
            return error

    guarded.__name__ = tool_name
    guarded.__doc__ = handler.__doc__ or ""
    return guarded
