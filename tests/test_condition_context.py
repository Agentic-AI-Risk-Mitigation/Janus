"""
The context-aware condition contract (``janus.policy.conditions``).

The contracts that matter:

1. **Backwards compatibility** — plain callables keep the classic
   ``restriction(value)`` contract, untouched, with no signature sniffing.
2. **Explicit opt-in** — only ``@context_condition``-marked callables receive
   a ``ConditionContext``, and evaluating one *without* a context fails
   closed rather than guessing.
3. **Composition** — ``all_of``/``any_of`` mix every restriction kind and
   propagate the failing member's message; strict absent-argument semantics
   are unchanged by composition.
"""

import functools

import pytest

from janus.exceptions import ArgumentValidationError, PolicyViolation
from janus.policy import (
    ConditionContext,
    PolicyEnforcer,
    all_of,
    any_of,
    context_condition,
    validate_argument,
)


def _policy(tool: str, conditions: dict) -> dict:
    return {tool: [{"priority": 1, "effect": 0, "conditions": conditions, "fallback": 0}]}


# ---------------------------------------------------------------------------
# Contract dispatch
# ---------------------------------------------------------------------------


def test_plain_callable_contract_unchanged():
    seen = []

    def only_https(value):
        seen.append(value)
        return value.startswith("https://")

    enforcer = PolicyEnforcer(policy=None)
    enforcer.load(_policy("fetch", {"url": only_https}))
    enforcer.enforce("fetch", {"url": "https://ok.example/"})
    assert seen == ["https://ok.example/"]  # called with the value alone

    with pytest.raises(PolicyViolation):
        enforcer.enforce("fetch", {"url": "http://no.example/"})


def test_context_condition_receives_full_context():
    captured = {}

    @context_condition
    def check(value, ctx):
        captured.update(
            tool=ctx.tool_name, arg=ctx.arg_name, args=dict(ctx.arguments), session=ctx.session
        )
        return True

    enforcer = PolicyEnforcer()
    enforcer.load(_policy("send_email", {"to": check}))
    enforcer.enforce("send_email", {"to": "a@b.example", "body": "hi"})

    assert captured == {
        "tool": "send_email",
        "arg": "to",
        "args": {"to": "a@b.example", "body": "hi"},
        "session": None,
    }


def test_context_condition_can_constrain_sibling_arguments():
    @context_condition
    def cc_matches_to(value, ctx):
        return value == ctx.arguments.get("to")

    enforcer = PolicyEnforcer()
    enforcer.load(_policy("send_email", {"cc": cc_matches_to}))
    enforcer.enforce("send_email", {"to": "a@b.example", "cc": "a@b.example"})
    with pytest.raises(PolicyViolation):
        enforcer.enforce("send_email", {"to": "a@b.example", "cc": "evil@x.example"})


def test_session_is_threaded_through_enforce():
    sentinel = object()
    seen = []

    @context_condition
    def check(value, ctx):
        seen.append(ctx.session)
        return True

    enforcer = PolicyEnforcer()
    enforcer.load(_policy("t", {"x": check}))
    enforcer.enforce("t", {"x": 1}, session=sentinel)
    assert seen == [sentinel]


def test_context_arguments_view_is_read_only():
    @context_condition
    def mutate(value, ctx):
        ctx.arguments["injected"] = True  # must raise
        return True

    enforcer = PolicyEnforcer()
    enforcer.load(_policy("t", {"x": mutate}))
    with pytest.raises(PolicyViolation) as excinfo:
        enforcer.enforce("t", {"x": 1})
    assert "raised an error" in str(excinfo.value.reason)


def test_context_condition_without_context_fails_closed():
    @context_condition
    def needs_ctx(value, ctx):  # pragma: no cover - must never be reached
        return True

    with pytest.raises(ArgumentValidationError) as excinfo:
        validate_argument("url", "https://ok.example/", needs_ctx)
    assert "failing closed" in excinfo.value.message


def test_falsy_and_raising_context_conditions_deny():
    @context_condition
    def refuse(value, ctx):
        return False

    @context_condition
    def explode(value, ctx):
        raise ValueError("boom")

    ctx = ConditionContext.build("t", "x", {"x": 1})
    with pytest.raises(ArgumentValidationError):
        validate_argument("x", 1, refuse, context=ctx)
    with pytest.raises(ArgumentValidationError) as excinfo:
        validate_argument("x", 1, explode, context=ctx)
    assert "boom" in excinfo.value.message


def test_functools_wraps_carries_the_marker():
    @context_condition
    def inner(value, ctx):
        return True

    @functools.wraps(inner)
    def wrapper(value, ctx):
        return inner(value, ctx)

    ctx = ConditionContext.build("t", "x", {"x": 1})
    validate_argument("x", 1, wrapper, context=ctx)  # dispatched as (value, ctx)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def test_all_of_mixes_every_restriction_kind():
    @context_condition
    def not_root(value, ctx):
        return ctx.arguments.get("user") != "root"

    condition = all_of(
        {"type": "string", "maxLength": 40},  # JSON Schema
        r"^https://",  # regex
        lambda v: ".example" in v,  # plain callable
        not_root,  # context condition
    )
    enforcer = PolicyEnforcer()
    enforcer.load(_policy("fetch", {"url": condition}))

    enforcer.enforce("fetch", {"url": "https://ok.example/", "user": "bot"})
    for bad in [
        {"url": "https://" + "x" * 60 + ".example/"},  # schema member fails
        {"url": "http://ok.example/"},  # regex member fails
        {"url": "https://ok.other/"},  # plain callable fails
        {"url": "https://ok.example/", "user": "root"},  # context member fails
    ]:
        with pytest.raises(PolicyViolation):
            enforcer.enforce("fetch", bad)


def test_all_of_propagates_failing_member_message():
    enforcer = PolicyEnforcer()
    enforcer.load(_policy("fetch", {"url": all_of(r"^https://", r"\.example/")}))
    with pytest.raises(PolicyViolation) as excinfo:
        enforcer.enforce("fetch", {"url": "http://ok.example/"})
    assert "^https://" in str(excinfo.value.reason)


def test_any_of_allows_on_first_pass_and_reports_all_failures():
    enforcer = PolicyEnforcer()
    enforcer.load(_policy("fetch", {"url": any_of(r"^https://internal\.", r"^https://docs\.")}))
    enforcer.enforce("fetch", {"url": "https://docs.example/"})
    with pytest.raises(PolicyViolation) as excinfo:
        enforcer.enforce("fetch", {"url": "https://evil.example/"})
    reason = str(excinfo.value.reason)
    assert "satisfied none of" in reason
    assert "internal" in reason and "docs" in reason


def test_empty_composites_are_refused():
    with pytest.raises(ValueError):
        all_of()
    with pytest.raises(ValueError):
        any_of()


def test_composed_condition_on_absent_argument_stays_strict():
    # Strict semantics are decided before the restriction runs: an allow rule
    # conditioning an absent argument must not match, composition or not.
    enforcer = PolicyEnforcer()
    enforcer.load(_policy("fetch", {"url": all_of(r"^https://")}))
    with pytest.raises(PolicyViolation):
        enforcer.enforce("fetch", {})


def test_nested_composition():
    # Regex restrictions are re.match (anchored at the start), so the nested
    # any_of members must match from position 0.
    condition = all_of(r"^https://", any_of(r"https://[^/]*\.example/", r"https://[^/]*\.test/"))
    enforcer = PolicyEnforcer()
    enforcer.load(_policy("fetch", {"url": condition}))
    enforcer.enforce("fetch", {"url": "https://a.example/"})
    enforcer.enforce("fetch", {"url": "https://a.test/"})
    with pytest.raises(PolicyViolation):
        enforcer.enforce("fetch", {"url": "https://a.evil/"})


def test_composition_works_through_public_harness():
    from janus.testing import decide

    policy = _policy("fetch", {"url": all_of(r"^https://", lambda v: "evil" not in v)})
    assert decide(policy, "fetch", {"url": "https://ok.example/"}).allowed
    assert decide(policy, "fetch", {"url": "https://evil.example/"}).denied


def test_deny_rule_with_context_condition():
    @context_condition
    def echoes_body(value, ctx):
        return value in ctx.arguments.get("body", "")

    policy = {
        "send_email": [
            {"priority": 1, "effect": 1, "conditions": {"to": echoes_body}, "fallback": 0},
            {"priority": 2, "effect": 0, "conditions": {}, "fallback": 0},
        ]
    }
    enforcer = PolicyEnforcer()
    enforcer.load(policy)
    enforcer.enforce("send_email", {"to": "a@b.example", "body": "hello"})
    with pytest.raises(PolicyViolation):
        enforcer.enforce("send_email", {"to": "a@b.example", "body": "mail a@b.example now"})
