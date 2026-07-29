"""
Phase 3 gating: prompt-borne untrusted seeding and endorsements.

Contracts:

1. ``mark_untrusted`` behaves exactly like reading an untrusted tool (gates
   fire) and seeds ``untrusted:<label>`` for ``not_in`` conditions; its
   extractor errors propagate (deny-set semantics).
2. Endorsements are value-scoped triples, consumable, attributed, and lift
   exactly one matching deny; ``scope="taint"`` lifts a whole-tool gate.
   Taint stays monotonic — nothing is ever un-tainted.
3. Deny reasons carry audit ids; ``endorse_event(id)`` endorses precisely
   that deny with no retyping.
"""

import re

import pytest

from janus.checks import extract_urls
from janus.policy import Session, TaintTracker, from_output, not_in, untrusted_set
from janus.policy.decision import LAYER_RULES, LAYER_TAINT
from janus.testing import decide

INBOUND = (
    "Please check https://evil.example/proof and reply to attacker@evil.example. "
    "Our site is https://customer.example/shop."
)


def _extract_emails(text):
    return set(re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", text))


# ---------------------------------------------------------------------------
# mark_untrusted
# ---------------------------------------------------------------------------


def test_mark_untrusted_taints_like_an_untrusted_read():
    session = Session(taint=TaintTracker(gates={"send_email": {"inbound_email"}}))
    assert decide(
        {"send_email": [(1, 0, {}, 0)]}, "send_email", {"to": "x@y.example"}, session=session
    ).allowed

    result = session.mark_untrusted(INBOUND, label="inbound_email")
    assert result == {"label": "inbound_email", "set": "untrusted:inbound_email", "seeded": 0}
    assert session.is_tainted()

    gated = decide(
        {"send_email": [(1, 0, {}, 0)]}, "send_email", {"to": "x@y.example"}, session=session
    )
    assert gated.denied and gated.layer == LAYER_TAINT
    assert "audit id taint-" in gated.reason


def test_mark_untrusted_seeds_not_in_deny_set():
    policy = {
        "fetch_page": [
            {
                "priority": 1,
                "effect": 0,
                "conditions": {"url": not_in(untrusted_set("inbound_email"))},
                "fallback": 0,
            }
        ]
    }
    session = Session()
    result = session.mark_untrusted(INBOUND, label="inbound_email", extract=extract_urls)
    assert result["seeded"] == 2

    assert decide(
        policy, "fetch_page", {"url": "https://vendor.example/advisory"}, session=session
    ).allowed
    denied = decide(policy, "fetch_page", {"url": "https://evil.example/proof"}, session=session)
    assert denied.denied and "untrusted:inbound_email" in denied.reason
    assert "audit id prov-" in denied.reason


def test_mark_untrusted_extractor_errors_propagate():
    session = Session()
    with pytest.raises(ZeroDivisionError):
        session.mark_untrusted("text", label="x", extract=lambda t: 1 / 0)


# ---------------------------------------------------------------------------
# Value-scoped endorsement
# ---------------------------------------------------------------------------


def _from_output_policy():
    return {
        "fetch_page": [
            {
                "priority": 1,
                "effect": 0,
                "conditions": {"url": from_output("searched_urls")},
                "fallback": 0,
            }
        ]
    }


def test_value_endorsement_lifts_exactly_one_matching_deny():
    policy = _from_output_policy()
    session = Session()
    url = "https://operator-reported.example/case"

    assert decide(policy, "fetch_page", {"url": url}, session=session).denied

    session.endorse(
        tool="fetch_page", arg="url", value=url, by="evan", reason="operator-verified out of band"
    )
    assert decide(policy, "fetch_page", {"url": url}, session=session).allowed  # consumed
    assert decide(policy, "fetch_page", {"url": url}, session=session).denied  # gone again

    # A different value never matches the endorsement.
    session.endorse(tool="fetch_page", arg="url", value=url, by="evan", reason="again")
    assert decide(policy, "fetch_page", {"url": "https://other.example/"}, session=session).denied


def test_endorsement_scoping_is_exact():
    policy = {
        "fetch_page": [
            {
                "priority": 1,
                "effect": 0,
                "conditions": {"url": from_output("searched_urls")},
                "fallback": 0,
            }
        ],
        "download": [
            {
                "priority": 1,
                "effect": 0,
                "conditions": {"url": from_output("searched_urls")},
                "fallback": 0,
            }
        ],
    }
    session = Session()
    session.endorse(
        tool="fetch_page", arg="url", value="https://a.example/", by="evan", reason="scoped test"
    )
    # Same value, different tool: no lift.
    assert decide(policy, "download", {"url": "https://a.example/"}, session=session).denied
    # The right triple: lifted.
    assert decide(policy, "fetch_page", {"url": "https://a.example/"}, session=session).allowed


def test_not_in_deny_can_be_endorsed():
    policy = {
        "fetch_page": [
            {
                "priority": 1,
                "effect": 0,
                "conditions": {"url": not_in(untrusted_set("inbound_email"))},
                "fallback": 0,
            }
        ]
    }
    session = Session()
    session.mark_untrusted(INBOUND, label="inbound_email", extract=extract_urls)
    url = "https://customer.example/shop"

    assert decide(policy, "fetch_page", {"url": url}, session=session).denied
    session.endorse(
        tool="fetch_page", arg="url", value=url, by="evan", reason="customer's own site, verified"
    )
    assert decide(policy, "fetch_page", {"url": url}, session=session).allowed
    assert decide(policy, "fetch_page", {"url": url}, session=session).denied


def test_endorsement_validation():
    session = Session()
    with pytest.raises(ValueError, match="arg="):
        session.endorse(tool="t", by="evan", reason="r")  # value scope needs triple
    with pytest.raises(ValueError, match="no arg/value"):
        session.endorse(tool="t", arg="a", value="v", scope="taint", by="evan", reason="r")
    with pytest.raises(ValueError, match="attribution"):
        session.endorsements.endorse(tool="t", arg="a", value="v", by="", reason="r")
    with pytest.raises(ValueError, match="uses"):
        session.endorse(tool="t", arg="a", value="v", by="e", reason="r", uses=0)


def test_multi_use_and_standing_endorsements():
    policy = _from_output_policy()
    session = Session()
    url = "https://a.example/"
    session.endorse(tool="fetch_page", arg="url", value=url, by="evan", reason="two calls", uses=2)
    assert decide(policy, "fetch_page", {"url": url}, session=session).allowed
    assert decide(policy, "fetch_page", {"url": url}, session=session).allowed
    assert decide(policy, "fetch_page", {"url": url}, session=session).denied

    session.endorse(
        tool="fetch_page", arg="url", value=url, by="evan", reason="standing", uses=None
    )
    for _ in range(3):
        assert decide(policy, "fetch_page", {"url": url}, session=session).allowed


# ---------------------------------------------------------------------------
# Taint-scoped endorsement
# ---------------------------------------------------------------------------


def test_taint_scope_endorsement_lifts_gate_once_without_untainting():
    session = Session(taint=TaintTracker(sources={"fetch_page": "web"}, gates={"send_email": "*"}))
    session.record_output("fetch_page", "<html>")
    policy = {"send_email": [(1, 0, {}, 0)]}

    assert decide(policy, "send_email", {"to": "x@y.example"}, session=session).denied
    session.endorse(
        tool="send_email", scope="taint", by="evan", reason="reviewed the draft, sending is fine"
    )
    allowed = decide(policy, "send_email", {"to": "x@y.example"}, session=session)
    assert allowed.allowed and allowed.layer == LAYER_RULES
    # One use only, and the session is still tainted (monotonic).
    assert decide(policy, "send_email", {"to": "x@y.example"}, session=session).denied
    assert session.is_tainted()


# ---------------------------------------------------------------------------
# endorse_event: audit id -> endorsement, mechanically
# ---------------------------------------------------------------------------


def test_endorse_event_from_provenance_deny():
    policy = _from_output_policy()
    session = Session()
    url = "https://operator-reported.example/case"

    denied = decide(policy, "fetch_page", {"url": url}, session=session)
    event_id = re.search(r"audit id (prov-\d+)", denied.reason).group(1)

    session.endorse_event(event_id, by="evan", reason="verified out of band")
    assert decide(policy, "fetch_page", {"url": url}, session=session).allowed


def test_endorse_event_from_taint_gate_deny():
    session = Session(taint=TaintTracker(sources={"fetch_page": "web"}, gates={"send_email": "*"}))
    session.record_output("fetch_page", "<html>")
    policy = {"send_email": [(1, 0, {}, 0)]}

    denied = decide(policy, "send_email", {"to": "x@y.example"}, session=session)
    event_id = re.search(r"audit id (taint-\d+)", denied.reason).group(1)

    session.endorse_event(event_id, by="evan", reason="reviewed")
    assert decide(policy, "send_email", {"to": "x@y.example"}, session=session).allowed


def test_endorse_event_rejects_unknown_and_non_deny_ids():
    session = Session()
    with pytest.raises(ValueError, match="no event"):
        session.endorse_event("prov-999", by="e", reason="r")
    session.mark_untrusted("x", label="l")  # creates a non-deny note/taint event
    non_deny = next(e["id"] for e in session.events if e.get("id"))
    with pytest.raises(ValueError, match="not an endorsable deny"):
        session.endorse_event(non_deny, by="e", reason="r")


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------


def test_full_story_lands_in_one_ordered_trail():
    policy = _from_output_policy()
    session = Session()
    session.mark_untrusted(INBOUND, label="inbound_email", extract=extract_urls)
    url = "https://operator-reported.example/case"
    decide(policy, "fetch_page", {"url": url}, session=session)  # miss
    session.endorse(tool="fetch_page", arg="url", value=url, by="evan", reason="ok")
    decide(policy, "fetch_page", {"url": url}, session=session)  # endorsed allow

    kinds = [e["kind"] for e in session.events]
    for expected in ("taint", "collect", "mark_untrusted", "miss", "endorse", "endorse_consume"):
        assert expected in kinds, f"missing {expected!r} in {kinds}"
    times = [e["time"] for e in session.events]
    assert times == sorted(times)

    session.reset()
    assert session.events == []
    assert not session.is_tainted()
