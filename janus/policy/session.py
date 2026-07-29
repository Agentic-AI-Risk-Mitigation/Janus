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

from typing import Any

from janus.policy.provenance import ProvenanceLedger
from janus.policy.taint import TaintTracker


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
    # Introspection / lifecycle
    # ------------------------------------------------------------------

    @property
    def events(self) -> list[dict[str, Any]]:
        """Merged audit trail (taint + provenance), ordered by wall-clock time.

        Each entry keeps its originating shape; the merge is by the ``time``
        field both trackers stamp.
        """
        merged = [dict(e, source="taint") for e in self.taint.events]
        merged += [dict(e, source="provenance") for e in self.provenance.events]
        merged.sort(key=lambda e: e.get("time", 0.0))
        return merged

    def is_tainted(self) -> bool:
        return self.taint.is_tainted()

    def reset(self) -> None:
        """Clear taint and provenance state (session boundary only).

        Ledger collectors and taint source/gate configuration survive —
        they are configuration, not session state.
        """
        self.taint.reset()
        self.provenance.reset()
