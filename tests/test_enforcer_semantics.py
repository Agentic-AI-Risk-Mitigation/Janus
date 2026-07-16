"""
Behavioral regression tests for ``PolicyEnforcer`` rule evaluation.

The packaging contract is covered in ``test_standalone_enforcer.py``; this file
pins down the *semantics* of enforcement:

- priority ordering and deny-before-allow at equal priority (fail-closed
  tie-break; also what makes ``block_tools`` override ``allow_tools``)
- default-deny for unlisted tools, permissive default with no policy loaded
- strict condition semantics: an allow rule whose condition names an argument
  the call does not provide must NOT match (the historical "missing-argument
  bypass"), with ``strict_conditions=False`` restoring the legacy skip
- deny rules fail closed on a missing conditioned argument (vacuous match
  blocks), so under strict mode both rule types fail closed
- core ``required_args``: named arguments must be present and non-empty
  regardless of rule conditions
- policy management: update / allow_tools / block_tools / reset
- fallback actions 0 (raise) and 2 (ask user)
"""

import pytest

from janus.exceptions import PolicyViolation
from janus.policy.enforcer import PolicyEnforcer

TRUSTED_EMAIL = {"to": "operator@trusted.org"}
ATTACKER_EMAIL = {"to": "exfil@attacker.example"}

# Internal policy format: {tool: [(priority, effect, conditions, fallback)]}
# effect: 0 = allow, 1 = deny; fallback: 0 = raise.
SEND_EMAIL_POLICY = {
    "send_email": [(1, 0, {"to": r"^[^@\s]+@trusted\.org$"}, 0)],
}


# ---------------------------------------------------------------------------
# Baseline semantics
# ---------------------------------------------------------------------------


def test_no_policy_loaded_allows_everything():
    enforcer = PolicyEnforcer()
    enforcer.enforce("anything", {"arg": "value"})  # must not raise


def test_unlisted_tool_is_default_denied():
    enforcer = PolicyEnforcer(SEND_EMAIL_POLICY)
    with pytest.raises(PolicyViolation):
        enforcer.enforce("run_command", {"command": "ls"})


def test_allow_rule_matching_condition_allows():
    enforcer = PolicyEnforcer(SEND_EMAIL_POLICY)
    enforcer.enforce("send_email", dict(TRUSTED_EMAIL))


def test_allow_rule_failing_condition_denies():
    enforcer = PolicyEnforcer(SEND_EMAIL_POLICY)
    with pytest.raises(PolicyViolation):
        enforcer.enforce("send_email", dict(ATTACKER_EMAIL))


def test_priority_order_lower_evaluates_first():
    # Priority 1 allow (broad) sits above a priority 2 deny: allow wins.
    policy = {
        "read_file": [
            (2, 1, {}, 0),  # unconditional deny at priority 2
            (1, 0, {}, 0),  # unconditional allow at priority 1
        ]
    }
    enforcer = PolicyEnforcer(policy)
    enforcer.enforce("read_file", {"path": "notes.txt"})


def test_deny_precedes_allow_at_equal_priority():
    # Fail-closed tie-break: this is what lets block_tools() override a prior
    # allow_tools() grant, since both insert at priority 1.
    policy = {
        "read_file": [
            (1, 0, {}, 0),
            (1, 1, {}, 0),
        ]
    }
    enforcer = PolicyEnforcer(policy)
    with pytest.raises(PolicyViolation):
        enforcer.enforce("read_file", {"path": "notes.txt"})


def test_deny_rule_match_raises_with_rule_attached():
    policy = {
        "run_command": [
            (1, 1, {"command": r".*rm -rf.*"}, 0),
            (2, 0, {}, 0),
        ]
    }
    enforcer = PolicyEnforcer(policy)
    with pytest.raises(PolicyViolation) as excinfo:
        enforcer.enforce("run_command", {"command": "rm -rf /"})
    assert excinfo.value.policy_rule is not None
    enforcer.enforce("run_command", {"command": "ls"})


# ---------------------------------------------------------------------------
# Missing-argument semantics (the historical bypass)
# ---------------------------------------------------------------------------


def test_strict_allow_rule_denies_when_conditioned_arg_missing():
    """An allow rule constraining ``to`` must not vouch for a call without ``to``."""
    enforcer = PolicyEnforcer(SEND_EMAIL_POLICY)
    with pytest.raises(PolicyViolation) as excinfo:
        enforcer.enforce("send_email", {})
    assert "to" in excinfo.value.reason


def test_strict_is_the_default():
    enforcer = PolicyEnforcer(SEND_EMAIL_POLICY)
    with pytest.raises(PolicyViolation):
        enforcer.enforce("send_email", {"subject": "no recipient arg at all"})


def test_legacy_mode_restores_skip_on_missing():
    enforcer = PolicyEnforcer(SEND_EMAIL_POLICY, strict_conditions=False)
    enforcer.enforce("send_email", {})  # legacy bypass behavior, opt-in only


def test_strict_only_requires_args_the_matching_rule_conditions():
    # A rule with no conditions is unaffected by strict mode.
    policy = {"list_dir": [(1, 0, {}, 0)]}
    enforcer = PolicyEnforcer(policy)
    enforcer.enforce("list_dir", {})


def test_deny_rules_fail_closed_on_missing_argument():
    # A deny condition on an absent argument matches vacuously: the call is
    # blocked. Together with strict allow semantics, both rule types fail
    # closed when the model omits a conditioned argument.
    policy = {
        "run_command": [
            (1, 1, {"command": r".*rm .*"}, 0),
            (2, 0, {}, 0),  # unconditional allow underneath
        ]
    }
    enforcer = PolicyEnforcer(policy)
    with pytest.raises(PolicyViolation):
        enforcer.enforce("run_command", {})
    enforcer.enforce("run_command", {"command": "ls"})


# ---------------------------------------------------------------------------
# Core required_args backstop
# ---------------------------------------------------------------------------


def test_required_args_blocks_missing_argument():
    enforcer = PolicyEnforcer(
        {"send_email": [(1, 0, {}, 0)]},
        required_args={"send_email": ["to"]},
    )
    with pytest.raises(PolicyViolation) as excinfo:
        enforcer.enforce("send_email", {"subject": "hi"})
    assert "to" in excinfo.value.reason


def test_required_args_blocks_empty_and_none_values():
    enforcer = PolicyEnforcer(
        {"send_email": [(1, 0, {}, 0)]},
        required_args={"send_email": ["to"]},
    )
    with pytest.raises(PolicyViolation):
        enforcer.enforce("send_email", {"to": "   "})
    with pytest.raises(PolicyViolation):
        enforcer.enforce("send_email", {"to": None})


def test_required_args_passes_when_present():
    enforcer = PolicyEnforcer(
        {"send_email": [(1, 0, {}, 0)]},
        required_args={"send_email": ["to"]},
    )
    enforcer.enforce("send_email", dict(TRUSTED_EMAIL))


def test_required_args_checked_even_with_unconditional_allow():
    # allow_tools() grants a blanket allow; required_args must still gate it.
    enforcer = PolicyEnforcer(required_args={"send_email": ["to"]})
    enforcer.load({"send_email": [{"priority": 5, "effect": 0, "conditions": {}, "fallback": 0}]})
    enforcer.allow_tools(["send_email"])
    with pytest.raises(PolicyViolation):
        enforcer.enforce("send_email", {})


# ---------------------------------------------------------------------------
# Policy management
# ---------------------------------------------------------------------------


def test_update_merges_rules():
    enforcer = PolicyEnforcer(SEND_EMAIL_POLICY)
    enforcer.update({"read_file": [(1, 0, {}, 0)]})
    enforcer.enforce("read_file", {"path": "x"})
    enforcer.enforce("send_email", dict(TRUSTED_EMAIL))


def test_allow_tools_grants_unconditional_allow():
    enforcer = PolicyEnforcer(SEND_EMAIL_POLICY)
    enforcer.allow_tools(["git_diff"])
    enforcer.enforce("git_diff", {})


def test_block_tools_overrides_existing_allow():
    enforcer = PolicyEnforcer(SEND_EMAIL_POLICY)
    enforcer.block_tools(["send_email"])
    with pytest.raises(PolicyViolation):
        enforcer.enforce("send_email", dict(TRUSTED_EMAIL))


def test_reset_keep_manual_drops_generated_rules():
    enforcer = PolicyEnforcer(SEND_EMAIL_POLICY)
    enforcer.update({"llm_added": [(100, 0, {}, 0)]})
    enforcer.enforce("llm_added", {})
    enforcer.reset(keep_manual=True)
    with pytest.raises(PolicyViolation):
        enforcer.enforce("llm_added", {})
    enforcer.enforce("send_email", dict(TRUSTED_EMAIL))


def test_reset_full_clears_policy_back_to_permissive():
    enforcer = PolicyEnforcer(SEND_EMAIL_POLICY)
    enforcer.reset(keep_manual=False)
    enforcer.enforce("anything", {})


# ---------------------------------------------------------------------------
# Fallback actions
# ---------------------------------------------------------------------------


def test_fallback_ask_user_rejection_blocks(monkeypatch):
    policy = {"deploy": [(1, 1, {}, 2), (2, 0, {}, 0)]}
    enforcer = PolicyEnforcer(policy)
    monkeypatch.setattr("builtins.input", lambda: "n")
    with pytest.raises(PolicyViolation):
        enforcer.enforce("deploy", {})


def test_fallback_ask_user_approval_allows(monkeypatch):
    policy = {"deploy": [(1, 1, {}, 2), (2, 0, {}, 0)]}
    enforcer = PolicyEnforcer(policy)
    monkeypatch.setattr("builtins.input", lambda: "y")
    enforcer.enforce("deploy", {})
