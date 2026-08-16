"""
Session taint tracking with per-source integrity labels.

This is the framework-agnostic core of Janus's indirect-prompt-injection
defense: as an agent reads from untrusted sources (web pages, inbound email,
honeypot logs, …), the session accumulates *per-source* taint labels, and
tools that send data outward or change state can be gated on which sources
have been read — Meta's "Rule of Two" enforced mechanically.

Unlike the PDE engine's monotonic session-wide scalar, taint here is a set of
source labels, each remembering which tool call introduced it. Derivation is
automatic: wire :meth:`TaintTracker.record_output` to a post-execution seam
(e.g. the Claude Agent SDK's ``PostToolUse`` hook via
``janus.adapters.claude_agent_sdk.janus_posttooluse_hook``) and
:meth:`TaintTracker.check` to the pre-execution seam, and no manual
``update_taint()`` calls are needed.

Design notes:

- **Source-granular, not per-datum.** When the agent loop runs inside a vendor
  subprocess (Claude Code), Janus sees pre/post tool hooks only — it cannot
  prove which bytes of a tool output influenced which argument. Labeling at
  the source level and gating sinks conservatively is the sound retrofit.
- **Monotonic within a session.** Labels are never removed by ordinary
  operation (only :meth:`reset`, e.g. at a session boundary). Once untrusted
  content has entered the context window it cannot be un-read.
- **Auditable.** Every taint event and gate denial is appended to
  :attr:`events`, so each allow/deny decision can be traced to the tool call
  that introduced the taint.

Example::

    tracker = TaintTracker(
        sources={"fetch_page": "web", "read_email": "email"},
        gates={"send_email": {"web", "email"}, "run_scan": "*"},
    )
    tracker.record_output("fetch_page")        # session now tainted by "web"
    tracker.check("send_email")                # -> deny reason (str)
    tracker.check("git_diff")                  # -> None (not gated)
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

# {tool_name: source_label} — reading from this tool taints the session with the label.
TaintSources = dict[str, str]
# {tool_name: labels} — the tool is blocked once any listed label (or "*": any label)
# has tainted the session.
TaintGates = dict[str, "set[str] | frozenset[str] | str"]
# Optional content-aware classifier: (tool_name, tool_output) -> extra label | None.
TaintClassifier = Callable[[str, Any], "str | None"]

ANY_SOURCE = "*"


class TaintTracker:
    """
    Tracks which untrusted sources a session has read and gates sinks on them.

    Instances are independent (one per agent session) and safe to share
    between concurrently running hook callbacks.

    Args:
        sources: ``{tool_name: source_label}`` — when ``record_output`` sees a
            tool listed here, the session is tainted with that label. Tools
            not listed are treated as taint-neutral.
        gates: ``{tool_name: labels}`` — ``check`` denies the tool once any of
            the listed labels has tainted the session. The string ``"*"``
            gates on *any* taint (strict Rule of Two: after reading anything
            untrusted, the tool needs out-of-band approval).
        classify: Optional callback ``(tool_name, tool_output) -> label | None``
            run on every recorded output, so integrators can add content-aware
            labels (e.g. "this fetched page contains an email address") on top
            of the static source map.
    """

    def __init__(
        self,
        *,
        sources: TaintSources | None = None,
        gates: TaintGates | None = None,
        classify: TaintClassifier | None = None,
    ):
        self._sources = dict(sources or {})
        self._gates = dict(gates or {})
        self._classify = classify
        self._lock = threading.Lock()
        # label -> the tool call that introduced it (first cause, for audit)
        self._tainted: dict[str, dict[str, Any]] = {}
        self._events: list[dict[str, Any]] = []
        self._seq = 0

    # ------------------------------------------------------------------
    # Recording (post-execution seam)
    # ------------------------------------------------------------------

    def record_output(self, tool_name: str, output: Any = None) -> list[str]:
        """
        Derive taint from a completed tool call. Returns the labels recorded
        (empty if the tool is taint-neutral and the classifier abstains).

        Call this after the tool has returned — its output is now in the
        model's context, so the session is tainted from this point on.
        """
        labels: list[str] = []
        static = self._sources.get(tool_name)
        if static is not None:
            labels.append(static)
        if self._classify is not None:
            extra = self._classify(tool_name, output)
            if extra is not None and extra not in labels:
                labels.append(extra)

        if not labels:
            return []

        with self._lock:
            for label in labels:
                if label not in self._tainted:
                    self._seq += 1
                    cause = {"seq": self._seq, "tool": tool_name, "time": time.time()}
                    self._tainted[label] = cause
                    self._events.append({"kind": "taint", "label": label, **cause})
        return labels

    def taint(self, label: str, *, reason: str = "manual") -> None:
        """Manually add a taint label (escape hatch; prefer ``record_output``)."""
        with self._lock:
            if label not in self._tainted:
                self._seq += 1
                cause = {"seq": self._seq, "tool": reason, "time": time.time()}
                self._tainted[label] = cause
                self._events.append({"kind": "taint", "label": label, **cause})

    # ------------------------------------------------------------------
    # Gating (pre-execution seam)
    # ------------------------------------------------------------------

    def check(self, tool_name: str) -> str | None:
        """
        Return a deny reason if ``tool_name`` is gated by current taint,
        else ``None`` (allowed as far as taint is concerned).
        """
        gate = self._gates.get(tool_name)
        if gate is None:
            return None

        with self._lock:
            if gate == ANY_SOURCE:
                triggered = dict(self._tainted)
            else:
                triggered = {lb: c for lb, c in self._tainted.items() if lb in gate}

            if not triggered:
                return None

            causes = "; ".join(
                f"'{label}' (introduced by {cause['tool']})"
                for label, cause in sorted(triggered.items(), key=lambda kv: kv[1]["seq"])
            )
            reason = (
                f"Tool '{tool_name}' is gated after reading untrusted sources: "
                f"{causes}. The session is tainted; this action requires "
                "verified provenance or out-of-band approval."
            )
            self._seq += 1
            self._events.append(
                {
                    "kind": "gate_deny",
                    "seq": self._seq,
                    "tool": tool_name,
                    "labels": sorted(triggered),
                    "time": time.time(),
                }
            )
            return reason

    # ------------------------------------------------------------------
    # Introspection / lifecycle
    # ------------------------------------------------------------------

    @property
    def source_tools(self) -> frozenset[str]:
        """Tool names whose output taints the session (configuration, not state)."""
        return frozenset(self._sources)

    @property
    def gated_tools(self) -> frozenset[str]:
        """Tool names that can be denied by taint (configuration, not state)."""
        return frozenset(self._gates)

    @property
    def tainted_by(self) -> frozenset[str]:
        """The set of source labels that have tainted this session."""
        with self._lock:
            return frozenset(self._tainted)

    @property
    def events(self) -> list[dict[str, Any]]:
        """Audit trail: taint introductions and gate denials, in order."""
        with self._lock:
            return list(self._events)

    def is_tainted(self) -> bool:
        with self._lock:
            return bool(self._tainted)

    def reset(self) -> None:
        """Clear all taint and the audit trail (session boundary only)."""
        with self._lock:
            self._tainted.clear()
            self._events.clear()
            self._seq = 0
