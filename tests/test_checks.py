"""
Output-side checks (``janus.checks``): the text-to-text gap and the tool-free
``structured()`` shape.

Contracts:

1. ``echoed_untrusted_values`` flags exactly the values that appeared in the
   marked untrusted input; ``values_grounded_in`` flags values in no allowed
   provenance set. Both respect the sets' bound normalizers.
2. Non-blocking by default (draft-for-review shape); ``enforce=True`` raises
   ``OutputViolation`` for pipelines that act mechanically.
3. A check that raises becomes a finding, never a silent pass; findings land
   in the session audit trail.
"""

import pytest

from janus.checks import (
    Finding,
    check_output,
    echoed_untrusted_values,
    extract_urls,
    values_grounded_in,
)
from janus.exceptions import OutputViolation
from janus.policy import Session, normalize_url

INBOUND = "See https://evil.example/proof — regards, https://customer.example/shop"


def _session():
    session = Session()
    session.mark_untrusted(
        INBOUND, label="inbound_email", extract=extract_urls, normalize=normalize_url
    )
    session.provenance.add(
        "searched_urls",
        ["https://vendor.example/advisory"],
        normalize=normalize_url,
        source="web_search",
    )
    return session


# ---------------------------------------------------------------------------
# extract_urls
# ---------------------------------------------------------------------------


def test_extract_urls_trims_trailing_punctuation():
    text = "Read (https://a.example/x), then https://b.example/y."
    assert extract_urls(text) == {"https://a.example/x", "https://b.example/y"}
    assert extract_urls("") == set()
    assert extract_urls(None) == set()


# ---------------------------------------------------------------------------
# echoed_untrusted_values
# ---------------------------------------------------------------------------


def test_echoed_untrusted_values_flags_only_echoes():
    session = _session()
    check = echoed_untrusted_values("inbound_email")

    clean = "Ground truth: https://vendor.example/advisory covers the fix."
    assert check_output(clean, session, checks=[check]) == []

    # Matching goes through the set's normalizer (case differences still hit),
    # but the finding reports the value AS WRITTEN in the output, so a
    # reviewer can locate and remove it from the draft.
    dirty = "As you noted at HTTPS://Evil.Example/proof, the site is fine."
    findings = check_output(dirty, session, checks=[check])
    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].values == ("HTTPS://Evil.Example/proof",)
    assert "inbound_email" in findings[0].message


def test_echoed_check_works_on_structured_output():
    session = _session()
    draft = {"reply_en": "see https://evil.example/proof", "confidence": "high"}
    findings = check_output(draft, session, checks=[echoed_untrusted_values("inbound_email")])
    assert findings and findings[0].values == ("https://evil.example/proof",)


# ---------------------------------------------------------------------------
# values_grounded_in
# ---------------------------------------------------------------------------


def test_values_grounded_in_flags_stray_values():
    session = _session()
    check = values_grounded_in(allowed=["searched_urls"])

    grounded = "Per https://vendor.example/advisory, upgrade to 2.9.5."
    assert check_output(grounded, session, checks=[check]) == []

    stray = "Per https://vendor.example/advisory and https://mystery.example/blog."
    findings = check_output(stray, session, checks=[check])
    assert findings[0].values == ("https://mystery.example/blog",)
    assert findings[0].severity == "warning"

    assert check_output("No links here.", session, checks=[check]) == []


# ---------------------------------------------------------------------------
# check_output semantics
# ---------------------------------------------------------------------------


def test_enforce_raises_output_violation_with_findings():
    session = _session()
    with pytest.raises(OutputViolation) as excinfo:
        check_output(
            "echoing https://evil.example/proof",
            session,
            checks=[echoed_untrusted_values("inbound_email")],
            enforce=True,
        )
    assert excinfo.value.findings[0].values == ("https://evil.example/proof",)
    assert "echoed_untrusted_values" in str(excinfo.value)


def test_broken_check_becomes_a_finding_not_a_pass():
    session = _session()

    def broken(output, sess):
        raise RuntimeError("boom")

    findings = check_output("anything", session, checks=[broken])
    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert "boom" in findings[0].message


def test_findings_land_in_session_events():
    session = _session()
    check_output(
        "echo https://evil.example/proof",
        session,
        checks=[echoed_untrusted_values("inbound_email")],
    )
    finding_events = [e for e in session.events if e.get("kind") == "output_finding"]
    assert len(finding_events) == 1
    assert finding_events[0]["values"] == ["https://evil.example/proof"]


def test_custom_checks_compose_with_builtins():
    session = _session()

    def no_dashes(output, sess):
        text = output if isinstance(output, str) else str(output)
        if "—" in text:
            return Finding(
                check="no_dashes", message="em dash present", values=("—",), severity="warning"
            )
        return None

    findings = check_output(
        "clean draft — with a dash and https://evil.example/proof",
        session,
        checks=[echoed_untrusted_values("inbound_email"), no_dashes],
    )
    assert {f.check for f in findings} == {"echoed_untrusted_values('inbound_email')", "no_dashes"}
