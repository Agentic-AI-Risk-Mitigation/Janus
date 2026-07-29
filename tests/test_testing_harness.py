"""
The public policy-test harness (``janus.testing``).

Two contracts matter:

1. **Deployed-semantics parity** — ``decide()`` must agree with the adapter's
   hook decision core for identical inputs, because its whole purpose is to
   let consumers assert what the deployed hook would do.
2. **Layer attribution** — a ``Decision`` names the layer that produced it
   (passthrough / taint / required_args / rules), so tests can assert *why*.
"""

import pytest

from janus.adapters.claude_agent_sdk import (
    DEFAULT_PASSTHROUGH_TOOLS,
    _decide,
    default_resolve_name,
    resolve_enforcer,
)
from janus.policy import TaintTracker
from janus.policy.decision import (
    LAYER_PASSTHROUGH,
    LAYER_REQUIRED_ARGS,
    LAYER_RULES,
    LAYER_TAINT,
)
from janus.testing import ALLOW, DENY, decide, replay

POLICY = {
    "fetch_page": [
        {"priority": 1, "effect": 0, "conditions": {"url": r"^https://"}, "fallback": 0},
    ],
    "send_email": [
        {
            "priority": 1,
            "effect": 0,
            "conditions": {"to": r"^[^@\s]+@trusted\.org$"},
            "fallback": 0,
        },
    ],
}

REQUIRED = {"fetch_page": ["url"]}


# ---------------------------------------------------------------------------
# decide()
# ---------------------------------------------------------------------------


def test_decide_allow_and_deny_with_reason():
    assert decide(POLICY, "fetch_page", {"url": "https://ok.example/"}).allowed

    denied = decide(POLICY, "fetch_page", {"url": "http://plain.example/"})
    assert denied.denied
    assert "fetch_page" in denied.reason
    assert denied.layer == LAYER_RULES


def test_decide_unlisted_tool_default_denies():
    decision = decide(POLICY, "bash_terminal", {"command": "id"})
    assert decision.denied
    assert "not listed in the policy" in decision.reason


def test_decide_layer_attribution():
    # required_args backstop fires before rule evaluation.
    blank = decide(POLICY, "fetch_page", {"url": "   "}, required_args=REQUIRED)
    assert blank.denied and blank.layer == LAYER_REQUIRED_ARGS

    # Taint gate fires before everything but passthrough.
    tracker = TaintTracker(sources={"fetch_page": "web"}, gates={"send_email": "*"})
    tracker.record_output("fetch_page")
    gated = decide(POLICY, "send_email", {"to": "ops@trusted.org"}, taint=tracker)
    assert gated.denied and gated.layer == LAYER_TAINT

    # Passthrough allows without consulting the policy.
    internal = decide(POLICY, "StructuredOutput", {"anything": 1})
    assert internal.allowed and internal.layer == LAYER_PASSTHROUGH


def test_decide_strips_mcp_prefix_like_the_adapter():
    assert decide(POLICY, "mcp__research__fetch_page", {"url": "https://ok.example/"}).allowed
    assert decide(POLICY, "mcp__research__fetch_page", {"url": "http://no.example/"}).denied


def test_decide_accepts_enforcer_instance_with_its_settings():
    # A shared enforcer carries its own configuration (e.g. legacy
    # strict_conditions=False), so the test sees exactly what production sees.
    from janus.policy import PolicyEnforcer

    legacy = PolicyEnforcer(strict_conditions=False)
    legacy.load(POLICY)
    # Under legacy semantics an omitted conditioned argument matches the allow rule.
    assert decide(legacy, "fetch_page", {}).allowed
    # Default (strict) semantics deny the same call.
    assert decide(POLICY, "fetch_page", {}).denied


def test_decide_matches_adapter_hook_core():
    """Parity: decide() and the adapter's _decide agree on every probe."""
    enforcer = resolve_enforcer(POLICY)
    probes = [
        ("fetch_page", {"url": "https://ok.example/"}),
        ("fetch_page", {"url": "http://no.example/"}),
        ("fetch_page", {}),
        ("mcp__research__fetch_page", {"url": "https://ok.example/"}),
        ("send_email", {"to": "ops@trusted.org"}),
        ("send_email", {"to": "x@evil.example"}),
        ("StructuredOutput", {}),
        ("bash_terminal", {"command": "id"}),
    ]
    for tool, args in probes:
        adapter_reason = _decide(
            enforcer,
            tool,
            args,
            passthrough_tools=DEFAULT_PASSTHROUGH_TOOLS,
            resolve_name=default_resolve_name,
            required_args=REQUIRED,
        )
        harness = decide(enforcer, tool, args, required_args=REQUIRED)
        assert harness.allowed == (adapter_reason is None), (tool, args)
        if harness.denied:
            assert harness.reason == adapter_reason


# ---------------------------------------------------------------------------
# replay()
# ---------------------------------------------------------------------------


def test_replay_passes_matching_sequence_and_returns_decisions():
    decisions = replay(
        POLICY,
        [
            ("fetch_page", {"url": "https://ok.example/"}, ALLOW),
            ("fetch_page", {"url": "http://no.example/"}, DENY),
            ("send_email", {"to": "ops@trusted.org"}, ALLOW),
        ],
    )
    assert [d.allowed for d in decisions] == [True, False, True]


def test_replay_mismatch_names_step_tool_and_outcome():
    with pytest.raises(AssertionError) as excinfo:
        replay(
            POLICY,
            [
                ("fetch_page", {"url": "https://ok.example/"}, ALLOW),
                ("fetch_page", {"url": "http://no.example/"}, ALLOW),  # wrong expectation
            ],
        )
    message = str(excinfo.value)
    assert "step 1" in message
    assert "fetch_page" in message
    assert "expected ALLOW" in message


def test_replay_checks_taint_but_never_feeds_it():
    tracker = TaintTracker(sources={"fetch_page": "web"}, gates={"send_email": "*"})
    # The replayed fetch is a decision, not an execution: no taint is recorded,
    # so the send still passes ...
    replay(
        POLICY,
        [
            ("fetch_page", {"url": "https://ok.example/"}, ALLOW),
            ("send_email", {"to": "ops@trusted.org"}, ALLOW),
        ],
        taint=tracker,
    )
    # ... until the test simulates the completed read explicitly.
    tracker.record_output("fetch_page")
    replay(POLICY, [("send_email", {"to": "ops@trusted.org"}, DENY)], taint=tracker)
