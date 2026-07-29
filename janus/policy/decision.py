"""
The single tool-call decision core.

``decide_call`` is the one place the layered pre-execution decision is made:
passthrough → taint gate → required-args backstop → policy rule evaluation.
The Claude Agent SDK adapter's hook path delegates here, and the public test
harness (``janus.testing``) exposes it to consumer test suites — so a test
asserting a decision exercises exactly the code the deployed hook runs, not a
parallel reimplementation.

This module is deliberately dependency-light (core install only) and holds no
state: every input, including session state like a ``TaintTracker``, is passed
in explicitly.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any

from janus.exceptions import PolicyViolation
from janus.policy.enforcer import PolicyEnforcer, RequiredArgs, check_required_args
from janus.policy.taint import TaintTracker

# Layer names, in evaluation order. A Decision records which layer decided it,
# so tests can assert *why* a call was denied, not just that it was.
LAYER_PASSTHROUGH = "passthrough"
LAYER_TAINT = "taint"
LAYER_REQUIRED_ARGS = "required_args"
LAYER_RULES = "rules"


@dataclass(frozen=True)
class Decision:
    """Outcome of evaluating one tool call against the enforcement layers.

    Attributes:
        allowed: Whether the call may proceed.
        reason: Deny reason (suitable to surface to the model), or ``None``.
        layer: Which layer produced the decision — one of the ``LAYER_*``
            constants for a deny (or a passthrough allow); ``LAYER_RULES``
            for an ordinary policy allow.
    """

    allowed: bool
    reason: str | None = None
    layer: str = LAYER_RULES

    @property
    def denied(self) -> bool:
        return not self.allowed


def decide_call(
    enforcer: PolicyEnforcer,
    runtime_name: str,
    arguments: dict,
    *,
    passthrough_tools: Collection[str] = (),
    resolve_name: Callable[[str], str] = lambda name: name,
    required_args: RequiredArgs | None = None,
    taint: TaintTracker | None = None,
    session: Any = None,
) -> Decision:
    """Evaluate one tool call through every enforcement layer, in order.

    Layers (first decisive one wins):

    1. **Passthrough** — ``runtime_name`` in ``passthrough_tools`` is allowed
       without consulting anything (SDK-internal transport tools).
    2. **Taint gate** — when session taint is wired (via ``taint`` or
       ``session.taint``), a sink gated by current taint is denied regardless
       of arguments.
    3. **Required args** — the ``check_required_args`` backstop rejects
       absent/blank arguments the policy's conditions may not cover.
    4. **Rules** — ``PolicyEnforcer.enforce`` evaluates the policy; a
       ``session`` is exposed to context-aware conditions (provenance,
       cross-argument checks) as ``ctx.session``.

    Pass either ``session`` (a :class:`janus.policy.session.Session`, whose
    ``.taint`` supplies the gate) or the bare ``taint`` tracker — not both;
    two competing taint sources would make the decision ambiguous.

    ``resolve_name`` maps the runtime tool name to its policy key before
    layers 2–4 (the Claude Agent SDK adapter uses it to strip the
    ``mcp__<server>__`` prefix).
    """
    if taint is not None and session is not None:
        raise ValueError(
            "decide_call: pass either taint= or session= (session.taint is "
            "the gate), not both"
        )

    if runtime_name in passthrough_tools:
        return Decision(True, layer=LAYER_PASSTHROUGH)

    policy_key = resolve_name(runtime_name)

    tracker = getattr(session, "taint", None) if session is not None else taint
    if tracker is not None:
        reason = tracker.check(policy_key)
        if reason is not None:
            return Decision(False, reason, LAYER_TAINT)

    try:
        check_required_args(policy_key, arguments, required_args or {})
    except PolicyViolation as exc:
        return Decision(False, exc.reason, LAYER_REQUIRED_ARGS)

    try:
        enforcer.enforce(policy_key, arguments, session=session)
    except PolicyViolation as exc:
        return Decision(False, exc.reason, LAYER_RULES)

    return Decision(True)
