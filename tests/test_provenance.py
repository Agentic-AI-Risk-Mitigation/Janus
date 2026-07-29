"""
Provenance core: ledger, condition factories, Session — and the acceptance
test from plans/ipi-expansion-design.md §4, the ``allowed_urls`` pattern
expressed as policy (fetch only URLs a prior search returned).
"""

import pytest

from janus.exceptions import ArgumentValidationError, PolicyViolation
from janus.policy import (
    PolicyEnforcer,
    ProvenanceLedger,
    Session,
    TaintTracker,
    all_of,
    from_output,
    normalize_url,
    not_in,
)
from janus.policy.decision import LAYER_RULES, LAYER_TAINT
from janus.testing import decide

# ---------------------------------------------------------------------------
# normalize_url
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("HTTPS://Docs.Example/Advisory?id=7", "https://docs.example/Advisory?id=7"),
        ("https://docs.example:443/x", "https://docs.example/x"),
        ("http://docs.example:80/x", "http://docs.example/x"),
        ("https://docs.example:8443/x", "https://docs.example:8443/x"),
        ("https://docs.example", "https://docs.example/"),
        ("https://user:pw@docs.example/x#frag", "https://docs.example/x"),
        ("  https://docs.example/x  ", "https://docs.example/x"),
    ],
)
def test_normalize_url(raw, expected):
    assert normalize_url(raw) == expected


def test_normalize_url_preserves_distinguishing_parts():
    # Path case and query distinguish real resources — must NOT be collapsed.
    assert normalize_url("https://a.example/A") != normalize_url("https://a.example/a")
    assert normalize_url("https://a.example/?x=1") != normalize_url("https://a.example/?x=2")


# ---------------------------------------------------------------------------
# ProvenanceLedger
# ---------------------------------------------------------------------------


def _search_urls(out):
    return [r.get("url") for r in (out.get("results") or []) if r.get("url")]


def test_ledger_collects_and_checks_with_normalization():
    ledger = ProvenanceLedger()
    ledger.collect("web_search", label="urls", extract=_search_urls, normalize=normalize_url)

    grown = ledger.record("web_search", {"results": [{"url": "HTTPS://A.Example/x"}]})
    assert grown == ["urls"]
    # Both sides normalized: differently-cased fetch of the same resource matches.
    assert ledger.contains("urls", "https://a.example/x")
    assert not ledger.contains("urls", "https://a.example/other")
    assert ledger.values("urls") == frozenset({"https://a.example/x"})


def test_ledger_unlisted_tool_and_unknown_label():
    ledger = ProvenanceLedger()
    assert ledger.record("some_tool", {"x": 1}) == []
    assert not ledger.contains("nope", "anything")  # missing set is empty: fails closed


def test_ledger_skips_none_and_unhashable_values():
    ledger = ProvenanceLedger()
    ledger.collect("t", label="vals", extract=lambda out: [None, ["unhashable"], "ok"])
    ledger.record("t", {})
    assert ledger.values("vals") == frozenset({"ok"})


def test_ledger_conflicting_normalizer_is_refused():
    ledger = ProvenanceLedger()
    ledger.collect("a", label="urls", extract=_search_urls, normalize=normalize_url)
    with pytest.raises(ValueError, match="normalize"):
        ledger.collect("b", label="urls", extract=_search_urls, normalize=str.lower)
    # Same normalizer object is fine (two source tools, one set).
    ledger.collect("b", label="urls", extract=_search_urls, normalize=normalize_url)


def test_ledger_collector_error_skips_by_default_and_can_raise():
    ledger = ProvenanceLedger()
    ledger.collect("t", label="quiet", extract=lambda out: 1 / 0)
    assert ledger.record("t", {}) == []  # logged, collected nothing
    assert any(e["kind"] == "collector_error" for e in ledger.events)

    ledger.collect("loud", label="deny_set", extract=lambda out: 1 / 0, on_error="raise")
    with pytest.raises(ZeroDivisionError):
        ledger.record("loud", {})


def test_ledger_reset_clears_state_but_keeps_collectors():
    ledger = ProvenanceLedger()
    ledger.collect("t", label="vals", extract=lambda out: [out])
    ledger.record("t", "v1")
    ledger.reset()
    assert ledger.values("vals") == frozenset()
    assert ledger.events == []
    ledger.record("t", "v2")  # collector survived the reset
    assert ledger.values("vals") == frozenset({"v2"})


# ---------------------------------------------------------------------------
# Condition factories
# ---------------------------------------------------------------------------


def _session_with(urls):
    session = Session()
    session.provenance.collect(
        "web_search", label="searched_urls", extract=_search_urls, normalize=normalize_url
    )
    if urls:
        session.record_output("web_search", {"results": [{"url": u} for u in urls]})
    return session


def _policy(conditions):
    return {"fetch_page": [{"priority": 1, "effect": 0, "conditions": conditions, "fallback": 0}]}


def test_from_output_allows_recorded_and_denies_everything_else():
    policy = _policy({"url": from_output("searched_urls")})
    session = _session_with(["https://a.example/x"])

    assert decide(policy, "fetch_page", {"url": "https://a.example/x"}, session=session).allowed

    denied = decide(policy, "fetch_page", {"url": "https://evil.example/"}, session=session)
    assert denied.denied and "searched_urls" in denied.reason
    # The deny landed in the audit trail.
    assert any(e["kind"] == "miss" for e in session.provenance.events)


def test_from_output_empty_set_and_missing_session_fail_closed():
    policy = _policy({"url": from_output("searched_urls")})
    assert decide(
        policy, "fetch_page", {"url": "https://a.example/"}, session=_session_with([])
    ).denied
    no_session = decide(policy, "fetch_page", {"url": "https://a.example/"})
    assert no_session.denied
    assert "failing closed" in no_session.reason


def test_from_output_normalizes_the_checked_argument():
    policy = _policy({"url": from_output("searched_urls")})
    session = _session_with(["HTTPS://A.Example:443/advisory"])
    assert decide(
        policy, "fetch_page", {"url": "https://a.example/advisory"}, session=session
    ).allowed


def test_not_in_denies_members_and_fails_closed_without_session():
    policy = {
        "send_email": [
            {
                "priority": 1,
                "effect": 0,
                "conditions": {"to": not_in("untrusted_addresses")},
                "fallback": 0,
            }
        ]
    }
    session = Session()
    session.provenance.add("untrusted_addresses", ["attacker@evil.example"], source="inbound")

    assert decide(policy, "send_email", {"to": "ops@good.example"}, session=session).allowed
    denied = decide(policy, "send_email", {"to": "attacker@evil.example"}, session=session)
    assert denied.denied and "untrusted" in denied.reason
    assert any(e["kind"] == "deny_match" for e in session.provenance.events)

    assert decide(policy, "send_email", {"to": "ops@good.example"}).denied  # no session


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def test_session_record_output_feeds_taint_and_provenance():
    session = Session(taint=TaintTracker(sources={"web_search": "web"}, gates={"send_email": "*"}))
    session.provenance.collect("web_search", label="urls", extract=_search_urls)

    recorded = session.record_output("web_search", {"results": [{"url": "https://a.example/"}]})
    assert recorded == {"taint": ["web"], "provenance": ["urls"]}
    assert session.is_tainted()
    sources = {e["source"] for e in session.events}
    assert sources == {"taint", "provenance"}
    times = [e["time"] for e in session.events]
    assert times == sorted(times)

    session.reset()
    assert not session.is_tainted()
    assert session.events == []


def test_decide_refuses_both_taint_and_session():
    with pytest.raises(ValueError, match="not both"):
        decide({"t": [(1, 0, {}, 0)]}, "t", {}, taint=TaintTracker(), session=Session())


def test_session_taint_gate_reports_taint_layer():
    session = Session(taint=TaintTracker(sources={"fetch_page": "web"}, gates={"send_email": "*"}))
    policy = {"send_email": [{"priority": 1, "effect": 0, "conditions": {}, "fallback": 0}]}
    session.record_output("fetch_page", "<html>")
    denied = decide(policy, "send_email", {"to": "x@y.example"}, session=session)
    assert denied.denied and denied.layer == LAYER_TAINT


# ---------------------------------------------------------------------------
# Acceptance test — plans/ipi-expansion-design.md §4 (the allowed_urls pattern)
# ---------------------------------------------------------------------------


def _ssrf_ok(url: str) -> bool:
    if not url.startswith("https://") or "169.254." in url or "localhost" in url:
        raise ValueError(f"blocked URL (SSRF guard): {url!r}")
    return True


ACCEPTANCE_POLICY = {
    "web_search": [(1, 0, {"query": {"type": "string", "maxLength": 400}}, 0)],
    "fetch_page": [(1, 0, {"url": all_of(_ssrf_ok, from_output("searched_urls"))}, 0)],
}


def test_acceptance_fetch_only_searched_urls():
    session = _session_with(None)

    # Search is allowed and its results are recorded (simulating PostToolUse).
    assert decide(
        ACCEPTANCE_POLICY, "web_search", {"query": "CVE-2023-23752"}, session=session
    ).allowed
    session.record_output(
        "web_search",
        {"ok": True, "results": [{"url": "https://Vendor.Example/advisory"}]},
    )

    # A result URL is fetchable — including with harmless renormalization.
    assert decide(
        ACCEPTANCE_POLICY, "fetch_page", {"url": "https://vendor.example/advisory"}, session=session
    ).allowed

    # An injected/inbound URL is not, even though it passes the SSRF check.
    injected = decide(
        ACCEPTANCE_POLICY,
        "fetch_page",
        {"url": "https://attacker.example/payload"},
        session=session,
    )
    assert injected.denied and injected.layer == LAYER_RULES
    assert "searched_urls" in injected.reason

    # SSRF member still fires first for non-public shapes.
    assert decide(
        ACCEPTANCE_POLICY, "fetch_page", {"url": "http://169.254.169.254/latest/"}, session=session
    ).denied

    # A fresh session has no provenance: nothing is fetchable (fail closed).
    assert decide(
        ACCEPTANCE_POLICY,
        "fetch_page",
        {"url": "https://vendor.example/advisory"},
        session=_session_with(None),
    ).denied


def test_acceptance_enforcer_direct_path_matches_harness():
    session = _session_with(["https://vendor.example/advisory"])
    enforcer = PolicyEnforcer()
    enforcer.load(ACCEPTANCE_POLICY)
    enforcer.enforce("fetch_page", {"url": "https://vendor.example/advisory"}, session=session)
    with pytest.raises(PolicyViolation):
        enforcer.enforce("fetch_page", {"url": "https://evil.example/"}, session=session)


def test_factory_names_surface_in_composite_reprs():
    condition = all_of(_ssrf_ok, from_output("searched_urls"))
    assert "from_output('searched_urls')" in repr(condition)


def test_from_output_without_provenance_attribute_fails_closed():
    class NotASession:
        pass

    with pytest.raises(ArgumentValidationError, match="failing closed"):
        from janus.policy import ConditionContext

        ctx = ConditionContext.build("fetch_page", "url", {"url": "x"}, NotASession())
        from_output("searched_urls")("x", ctx)
