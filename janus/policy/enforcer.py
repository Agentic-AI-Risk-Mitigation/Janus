"""
Policy enforcement engine — the heart of Janus.

PolicyEnforcer is a stateful, instance-based object that holds a security
policy and evaluates every tool call against it. Unlike a global-state
approach, each PolicyEnforcer instance is fully independent, making it
safe for concurrent use and easy to plug into different agentic frameworks.

Policy format (internal):
    {
        "tool_name": [
            (priority: int, effect: int, conditions: dict, fallback: int),
            ...
        ]
    }

    effect:   0 = allow,  1 = deny
    fallback: 0 = raise PolicyViolation,  1 = sys.exit,  2 = ask user
    priority: lower value = evaluated first
"""

import sys
from typing import Any

from janus.exceptions import ArgumentValidationError, PolicyViolation
from janus.logger import get_logger
from janus.policy.validator import validate_argument


# Type alias for a single policy rule tuple
PolicyRule = tuple[int, int, dict, int]
# Type alias for the full policy dict
PolicyDict = dict[str, list[PolicyRule]]


class PolicyEnforcer:
    """
    Evaluates tool calls against a security policy.

    Usage::

        enforcer = PolicyEnforcer()
        enforcer.load({"read_file": [{"priority": 1, "effect": 0, "conditions": {}, "fallback": 0}]})
        enforcer.enforce("read_file", {"file_path": "data.csv"})  # OK
        enforcer.enforce("run_command", {"command": "rm -rf /"})   # raises PolicyViolation

    The enforcer can also be used standalone by other frameworks — just call
    ``enforce()`` before executing any tool.
    """

    def __init__(self, policy: PolicyDict | None = None):
        self._policy: PolicyDict | None = None
        self._logger = get_logger()

        if policy is not None:
            self._policy = _sort_policy(policy)

    # ------------------------------------------------------------------
    # Policy management
    # ------------------------------------------------------------------

    def load(self, source: "str | PathLike | dict") -> None:
        """
        Load and apply a policy from a file path or dict.

        The source can be:
        - A ``str`` or ``Path`` pointing to a JSON policy file.
        - A ``dict`` in either the full Progent/Janus format or the simple
          conditions-only shorthand.

        Replaces any previously loaded policy.
        """
        from janus.policy.loader import parse_policy
        self._policy = _sort_policy(parse_policy(source))

    def update(self, additional: PolicyDict) -> None:
        """
        Merge additional policy rules into the current policy.

        Useful for augmenting a base manual policy with LLM-generated rules,
        or for adding always-allow / always-block overrides at runtime.
        """
        if self._policy is None:
            self._policy = {}

        for tool_name, rules in additional.items():
            if tool_name not in self._policy:
                self._policy[tool_name] = []
            self._policy[tool_name].extend(rules)

        self._policy = _sort_policy(self._policy)

    def allow_tools(self, tools: list[str], *, allow_no_arg_tools: bool = False) -> None:
        """
        Grant unconditional allow (highest priority) to the specified tools.

        Args:
            tools: Tool names to always allow.
            allow_no_arg_tools: If True, also allow all registered no-argument
                tools (useful for safe read-only operations).
        """
        if self._policy is None:
            self._policy = {}

        for name in tools:
            rules = self._policy.setdefault(name, [])
            rules.insert(0, (1, 0, {}, 0))

        self._policy = _sort_policy(self._policy)

    def block_tools(self, tools: list[str]) -> None:
        """
        Apply an unconditional deny (highest priority) to the specified tools.
        """
        if self._policy is None:
            self._policy = {}

        for name in tools:
            rules = self._policy.setdefault(name, [])
            rules.insert(0, (1, 1, {}, 0))

        self._policy = _sort_policy(self._policy)

    def reset(self, *, keep_manual: bool = True) -> None:
        """
        Reset the policy.

        Args:
            keep_manual: If True (default), only removes LLM-generated rules
                (priority >= 100). If False, clears the entire policy.
        """
        if self._policy is None:
            return

        if not keep_manual:
            self._policy = None
            return

        for tool_name in list(self._policy.keys()):
            self._policy[tool_name] = [r for r in self._policy[tool_name] if r[0] < 100]
            if not self._policy[tool_name]:
                del self._policy[tool_name]

    @property
    def policy(self) -> PolicyDict | None:
        """Return a copy of the current policy dict."""
        if self._policy is None:
            return None
        return {k: list(v) for k, v in self._policy.items()}

    def has_policy(self) -> bool:
        """Return True if a policy has been loaded."""
        return self._policy is not None

    # ------------------------------------------------------------------
    # Enforcement
    # ------------------------------------------------------------------

    def enforce(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """
        Enforce the policy for a given tool call.

        This is the primary method to call before executing any tool. It
        evaluates every policy rule for the tool in priority order and
        either returns normally (tool is allowed) or raises ``PolicyViolation``.

        If no policy has been loaded, all tool calls are allowed (with a
        debug-level log). This permissive default is intentional — enforce
        a policy explicitly to lock things down.

        Args:
            tool_name: Name of the tool being called.
            arguments: Keyword arguments the tool will receive.

        Raises:
            PolicyViolation: If the tool call is blocked by policy.
        """
        self._logger.tool_call(tool_name, arguments)

        if self._policy is None:
            self._logger.debug(
                f"No policy loaded — allowing '{tool_name}' by default."
            )
            return

        rules = self._policy.get(tool_name)

        if not rules:
            reason = f"Tool '{tool_name}' is not listed in the policy."
            self._logger.policy_decision(tool_name, allowed=False, reason=reason)
            raise PolicyViolation(tool_name=tool_name, arguments=arguments, reason=reason)

        try:
            _evaluate_rules(tool_name, arguments, rules)
            self._logger.policy_decision(tool_name, allowed=True)
        except PolicyViolation as exc:
            self._logger.policy_decision(tool_name, allowed=False, reason=exc.reason)
            raise


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _evaluate_rules(
    tool_name: str,
    arguments: dict[str, Any],
    rules: list[PolicyRule],
) -> None:
    """
    Walk through policy rules in priority order and apply the first match.

    Allow rules (effect=0): if all conditions pass → return (allow).
    Deny rules (effect=1): if all conditions match → block.
    If no rule matches → block by default.
    """
    skipped_reasons: list[str] = []

    for rule in rules:
        priority, effect, conditions, fallback = rule

        if effect == 0:  # Allow rule
            try:
                _check_conditions(arguments, conditions)
                return  # All conditions passed → allowed
            except ArgumentValidationError as exc:
                skipped_reasons.append(f"Allow rule (priority={priority}) skipped: {exc.message}")
                continue

        elif effect == 1:  # Deny rule
            try:
                _check_conditions(arguments, conditions)
                # All deny conditions matched → blocked
                _handle_block(tool_name, arguments, fallback, rule)
            except ArgumentValidationError:
                continue  # Deny rule didn't match → skip

    # No rule matched at all → default deny
    if skipped_reasons:
        reason = (
            f"Tool '{tool_name}' blocked - no matching allow rule. Details:\n  "
            + "\n  ".join(skipped_reasons)
        )
    else:
        reason = f"Tool '{tool_name}' blocked - no policy rule matched the provided arguments."

    raise PolicyViolation(tool_name=tool_name, arguments=arguments, reason=reason)


def _check_conditions(arguments: dict[str, Any], conditions: dict) -> None:
    """Validate all conditions against the provided arguments."""
    for arg_name, restriction in conditions.items():
        if arg_name in arguments:
            validate_argument(arg_name, arguments[arg_name], restriction)


def _handle_block(
    tool_name: str,
    arguments: dict[str, Any],
    fallback: int,
    rule: PolicyRule,
) -> None:
    """Execute the configured fallback action when a deny rule matches."""
    if fallback == 0:
        raise PolicyViolation(
            tool_name=tool_name,
            arguments=arguments,
            reason=f"Tool '{tool_name}' matched a deny rule.",
            policy_rule=rule,
        )
    elif fallback == 1:
        sys.exit(1)
    elif fallback == 2:
        print(
            f"[Janus] Agent wants to call '{tool_name}' with args {arguments}. Allow? [y/N]: ",
            end="",
            flush=True,
        )
        if input().strip().lower() != "y":
            raise PolicyViolation(
                tool_name=tool_name,
                arguments=arguments,
                reason="Tool call rejected by user.",
                policy_rule=rule,
            )
    else:
        raise PolicyViolation(
            tool_name=tool_name,
            arguments=arguments,
            reason=f"Tool '{tool_name}' matched a deny rule (unknown fallback={fallback}).",
            policy_rule=rule,
        )


def _sort_policy(policy: PolicyDict) -> PolicyDict:
    """Sort each tool's rules by (priority, -effect) so allow rules precede deny at the same level."""
    return {
        tool: sorted(rules, key=lambda r: (r[0], -r[1]))
        for tool, rules in policy.items()
    }
