"""
Public harness for asserting Janus policy decisions in test suites.

Consumers need to test their policies ("this call is denied, that one is
allowed") without importing private adapter internals. ``decide`` runs one
tool call through :func:`janus.policy.decision.decide_call` — the exact core
the Claude Agent SDK ``PreToolUse`` hook delegates to — and returns the full
:class:`~janus.policy.decision.Decision`, including which layer decided.
``replay`` runs a recorded sequence of calls, the idiom used by Janus's own
indirect-prompt-injection regression suite.

Example::

    from janus.testing import ALLOW, DENY, decide, replay

    d = decide(TOOL_POLICY, "fetch_page", {"url": "http://169.254.169.254/"})
    assert d.denied and "169.254" in d.reason

    replay(TOOL_POLICY, [
        ("web_search", {"query": "CVE-2023-23752"}, ALLOW),
        ("fetch_page", {"url": "http://127.0.0.1/admin"}, DENY),
    ])

The default ``resolve_name``/``passthrough_tools`` match the Claude Agent SDK
adapter's hook, so a green test reflects that adapter's deployed semantics.
Both are keyword-overridable for other integrations (``resolve_name=lambda
n: n`` disables the ``mcp__<server>__`` prefix stripping; it is a no-op on
bare names either way).
"""

from __future__ import annotations

from janus.adapters._base import PolicySource, resolve_enforcer
from janus.adapters.claude_agent_sdk import (
    DEFAULT_PASSTHROUGH_TOOLS,
    NameResolver,
    default_resolve_name,
)
from janus.policy.decision import Decision, decide_call
from janus.policy.enforcer import RequiredArgs
from janus.policy.taint import TaintTracker

__all__ = ["ALLOW", "DENY", "Decision", "decide", "replay"]

# Readable expected-outcome markers for replay() steps.
ALLOW = True
DENY = False


def decide(
    policy: PolicySource,
    tool_name: str,
    arguments: dict,
    *,
    required_args: RequiredArgs | None = None,
    taint: TaintTracker | None = None,
    passthrough_tools: frozenset[str] = DEFAULT_PASSTHROUGH_TOOLS,
    resolve_name: NameResolver = default_resolve_name,
) -> Decision:
    """Evaluate one tool call against a policy; return the full :class:`Decision`.

    ``policy`` is anything an adapter accepts: a policy dict (full or loader
    shorthand), a path to a JSON policy file, or a ``PolicyEnforcer`` instance
    (shared, so strictness settings carry over). The remaining keywords mirror
    the adapter hook's knobs; pass the same values your integration passes to
    ``janus_options()`` / ``janus_hooks()`` and the test reproduces the
    deployed decision exactly.
    """
    return decide_call(
        resolve_enforcer(policy),
        tool_name,
        arguments,
        passthrough_tools=passthrough_tools,
        resolve_name=resolve_name,
        required_args=required_args,
        taint=taint,
    )


def replay(
    policy: PolicySource,
    sequence: list[tuple[str, dict, bool]],
    *,
    required_args: RequiredArgs | None = None,
    taint: TaintTracker | None = None,
    passthrough_tools: frozenset[str] = DEFAULT_PASSTHROUGH_TOOLS,
    resolve_name: NameResolver = default_resolve_name,
) -> list[Decision]:
    """Feed ``(tool, args, expected)`` steps to the decision core in order.

    ``expected`` is :data:`ALLOW` or :data:`DENY`. Raises ``AssertionError``
    on the first step whose decision does not match, naming the step and the
    deny reason (if any) — a block is feedback to the model in a live loop,
    so the sequence deliberately continues past expected denials. Returns the
    per-step decisions for further assertions.

    Note: replay evaluates decisions only; it does not execute tools, so a
    ``TaintTracker`` passed here is *checked* but never fed — call
    ``taint.record_output(...)`` between steps to simulate completed reads.
    """
    enforcer = resolve_enforcer(policy)
    decisions: list[Decision] = []
    for index, (tool_name, arguments, expected) in enumerate(sequence):
        decision = decide_call(
            enforcer,
            tool_name,
            arguments,
            passthrough_tools=passthrough_tools,
            resolve_name=resolve_name,
            required_args=required_args,
            taint=taint,
        )
        decisions.append(decision)
        if decision.allowed != expected:
            expectation = "ALLOW" if expected else "DENY"
            outcome = "allowed" if decision.allowed else f"denied ({decision.reason})"
            raise AssertionError(
                f"replay step {index}: expected {expectation} for "
                f"{tool_name}({arguments!r}) but the call was {outcome}"
            )
    return decisions
