"""
Per-agent-run enforcement state.

``Session`` is the explicit home for everything cross-call: the taint tracker
(which untrusted sources have been read) and the provenance ledger (which
values prior tool outputs contained). One Session per agent run, created by
the integrator and passed explicitly — into the adapter
(``janus_options(session=...)``), the enforcer
(``enforce(..., session=...)``), or the test harness
(``janus.testing.decide(..., session=...)``). Context conditions see it as
``ctx.session``.

This is how the "stateless enforcer, no global state" commitment survives
statefulness: there is no module-level state anywhere; the enforcer only
*reads* a Session handed to it per call; all mutation happens at the
post-execution seam (:meth:`record_output`) or via explicit integrator calls;
and :meth:`reset` marks session boundaries, nothing else does.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from janus.policy.endorsement import SCOPE_TAINT, EndorsementLog
from janus.policy.provenance import Normalizer, ProvenanceLedger
from janus.policy.taint import TaintTracker

# Provenance sets seeded from prompt-borne untrusted input are namespaced so a
# collector-fed allow-set can never collide with an untrusted deny-set.
UNTRUSTED_PREFIX = "untrusted:"


def untrusted_set(label: str) -> str:
    """The provenance set name ``mark_untrusted(label=...)`` seeds."""
    return f"{UNTRUSTED_PREFIX}{label}"


class Session:
    """Taint + provenance state for one agent run.

    Args:
        taint: A configured :class:`TaintTracker` (sources/gates/classify).
            Defaults to an empty tracker — no sources, no gates — so a
            provenance-only Session works out of the box.
        provenance: A :class:`ProvenanceLedger`. Defaults to an empty ledger;
            register collectors on ``session.provenance`` after construction.

    Use one Session per run. For long-running services, construct a new one
    per session (preferred) or call :meth:`reset` at boundaries — otherwise
    one request's taint and provenance gate the next one's.
    """

    def __init__(
        self,
        *,
        taint: TaintTracker | None = None,
        provenance: ProvenanceLedger | None = None,
    ) -> None:
        self.taint = taint if taint is not None else TaintTracker()
        self.provenance = provenance if provenance is not None else ProvenanceLedger()
        self.endorsements = EndorsementLog()
        self._notes: list[dict[str, Any]] = []
        self._notes_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Post-execution seam
    # ------------------------------------------------------------------

    def record_output(self, tool_name: str, output: Any = None) -> dict[str, list[str]]:
        """Feed one completed tool call to both trackers.

        Returns ``{"taint": [labels...], "provenance": [labels...]}`` — the
        taint labels introduced and the provenance sets that grew. Wired
        automatically by the Claude Agent SDK adapter's ``PostToolUse`` hook
        when a Session is passed; call manually from frameworks without a
        post-execution seam.
        """
        return {
            "taint": self.taint.record_output(tool_name, output),
            "provenance": self.provenance.record(tool_name, output),
        }

    # ------------------------------------------------------------------
    # Prompt-borne untrusted input
    # ------------------------------------------------------------------

    def mark_untrusted(
        self,
        text: str,
        *,
        label: str,
        extract: Any = None,
        normalize: Normalizer | None = None,
    ) -> dict[str, Any]:
        """Declare prompt-borne content untrusted, at the call site that
        already knows it (the code that pastes an inbound email or a scraped
        page into the prompt).

        Two effects: the session is **tainted** with ``label`` exactly as if
        an untrusted tool had been read (gates on that label or ``"*"``
        start denying), and — when ``extract`` is given — the extracted
        values seed the provenance set ``untrusted:<label>`` for
        ``not_in(...)`` conditions and for output-side checks
        (:mod:`janus.checks`).

        Extractor errors **propagate** (no skip-and-continue): these sets
        deny, and a silently empty deny-set fails open.

        Janus cannot detect that pasted content is untrusted; this is the
        one-line, audited way to declare it.
        """
        self.taint.taint(label, reason="prompt")
        set_name = untrusted_set(label)
        seeded = 0
        if extract is not None:
            values = list(extract(text))
            seeded = self.provenance.add(
                set_name, values, source=f"prompt:{label}", normalize=normalize
            )
        self.note(kind="mark_untrusted", label=label, set=set_name, seeded=seeded)
        return {"label": label, "set": set_name, "seeded": seeded}

    # ------------------------------------------------------------------
    # Endorsement (declassification)
    # ------------------------------------------------------------------

    def endorse(
        self,
        *,
        tool: str,
        arg: str | None = None,
        value: Any = None,
        scope: str = "value",
        by: str,
        reason: str,
        uses: int | None = 1,
    ) -> str:
        """Record an audited, consumable declassification; returns its id.

        See :class:`janus.policy.endorsement.EndorsementLog` for the exact
        semantics (value-scoped triples by default, explicit ``scope="taint"``
        for whole-tool gate lifts, ``uses=1`` default).
        """
        return self.endorsements.endorse(
            tool=tool, arg=arg, value=value, scope=scope, by=by, reason=reason, uses=uses
        )

    def endorse_event(self, event_id: str, *, by: str, reason: str, uses: int | None = 1) -> str:
        """Endorse exactly the deny identified by ``event_id``.

        Deny reasons from taint gates and provenance conditions carry an
        ``(audit id ...)`` suffix; a human reviewer passes that id here and
        the matching triple (or taint scope) is endorsed mechanically — no
        retyping the value, no scope guessing.
        """
        for event in self.events:
            if event.get("id") != event_id:
                continue
            kind = event.get("kind")
            if kind == "gate_deny":
                return self.endorse(
                    tool=event["tool"], scope=SCOPE_TAINT, by=by, reason=reason, uses=uses
                )
            if kind in ("miss", "deny_match"):
                return self.endorse(
                    tool=event["tool"],
                    arg=event["arg"],
                    value=event["value"],
                    by=by,
                    reason=reason,
                    uses=uses,
                )
            raise ValueError(f"event {event_id!r} ({kind!r}) is not an endorsable deny")
        raise ValueError(f"no event with id {event_id!r} in this session")

    def gate_check(self, tool_name: str) -> str | None:
        """Taint-gate check with endorsement consultation.

        Used by the decision core in place of ``taint.check`` when a Session
        is wired: a taint-scoped endorsement for ``tool_name`` lifts one
        deny (consumed), otherwise the deny reason is returned with its
        audit id appended so it can be endorsed via :meth:`endorse_event`.
        """
        reason = self.taint.check(tool_name)
        if reason is None:
            return None
        if self.endorsements.consume(tool=tool_name, scope=SCOPE_TAINT) is not None:
            self.note(kind="gate_endorsed", tool=tool_name)
            return None
        event_id = None
        for event in reversed(self.taint.events):
            if event.get("kind") == "gate_deny" and event.get("tool") == tool_name:
                event_id = f"taint-{event['seq']}"
                break
        return f"{reason} (audit id {event_id})" if event_id else reason

    # ------------------------------------------------------------------
    # Introspection / lifecycle
    # ------------------------------------------------------------------

    def note(self, **event: Any) -> None:
        """Append a session-level audit event (used by mark_untrusted, gate
        endorsements, and output checks; available to integrators too)."""
        with self._notes_lock:
            self._notes.append({**event, "time": time.time()})

    @property
    def events(self) -> list[dict[str, Any]]:
        """Merged audit trail (taint + provenance + endorsements + notes),
        ordered by wall-clock time.

        Each entry keeps its originating shape, tagged with ``source``.
        Taint events gain an ``id`` (``taint-<seq>``) so gate denials are
        endorsable via :meth:`endorse_event`.
        """
        merged = [
            dict(e, source="taint", id=f"taint-{e['seq']}")
            if "seq" in e
            else dict(e, source="taint")
            for e in self.taint.events
        ]
        merged += [dict(e, source="provenance") for e in self.provenance.events]
        merged += [dict(e, source="endorsement") for e in self.endorsements.events]
        with self._notes_lock:
            merged += [dict(e, source="session") for e in self._notes]
        merged.sort(key=lambda e: e.get("time", 0.0))
        return merged

    def is_tainted(self) -> bool:
        return self.taint.is_tainted()

    def reset(self) -> None:
        """Clear taint, provenance, endorsement, and note state (session
        boundary only).

        Ledger collectors and taint source/gate configuration survive —
        they are configuration, not session state.
        """
        self.taint.reset()
        self.provenance.reset()
        self.endorsements.reset()
        with self._notes_lock:
            self._notes.clear()
