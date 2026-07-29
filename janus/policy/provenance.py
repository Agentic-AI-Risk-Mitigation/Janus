"""
Argument-value provenance: named value-sets derived from tool outputs.

At the hook seams Janus cannot prove which *bytes* of a tool output influenced
which argument — that limit is documented in :mod:`janus.policy.taint`. What it
can prove is exact-match provenance: an argument value literally appeared in a
prior output of a named tool. :class:`ProvenanceLedger` collects those values
at the post-execution seam, and the condition factories gate arguments on
membership at the pre-execution seam:

- :func:`from_output` — **positive provenance**: the argument must be a value
  a listed tool actually returned ("fetch only URLs a prior search surfaced").
- :func:`not_in` — **negative provenance**: the argument must not be a value
  from an untrusted set (e.g. URLs extracted from an inbound email).

Both fail closed: no session wired, or an empty/missing set, denies.

Extraction is the integrator's: Janus does not guess output shapes. A
collector registered with :meth:`ProvenanceLedger.collect` names the source
tool, the set label, an ``extract`` callable over the tool's output, and an
optional ``normalize`` applied to both recorded values and checked arguments
(exact string match is a strength against confusion attacks but a utility
risk — :func:`normalize_url` covers the common URL case).

Example (the ``allowed_urls`` pattern, expressed as policy)::

    session = Session()
    session.provenance.collect(
        "web_search", label="searched_urls",
        extract=lambda out: [r["url"] for r in out.get("results") or []],
        normalize=normalize_url,
    )
    POLICY = {
        "web_search": [(1, 0, {"query": {"type": "string", "maxLength": 400}}, 0)],
        "fetch_page": [(1, 0, {"url": all_of(ssrf_ok, from_output("searched_urls"))}, 0)],
    }
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from janus.exceptions import ArgumentValidationError
from janus.logger import get_logger
from janus.policy.conditions import ConditionContext, context_condition

# extract: (tool_output) -> iterable of values to record. Written by the
# integrator against the dict their tool body returns.
Extractor = Callable[[Any], "Iterable[Any]"]
# normalize: applied to string values on BOTH sides of a membership check.
Normalizer = Callable[[str], str]


def normalize_url(url: str) -> str:
    """Conservative URL normalization for provenance membership checks.

    Lowercases scheme and host, strips a default port (80/443), drops the
    fragment and any userinfo (credentials in URLs are a confusion-attack
    vector, never an identity), and normalizes an empty path to ``/``. Path
    case, query, and non-default ports are preserved — over-normalizing would
    make *distinct* resources compare equal, which fails open.
    """
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    try:
        port = parts.port
    except ValueError:
        port = None
    default_port = {"http": 80, "https": 443}.get(scheme)
    netloc = host if port is None or port == default_port else f"{host}:{port}"
    return urlunsplit((scheme, netloc, parts.path or "/", parts.query, ""))


@dataclass(frozen=True)
class _Collector:
    label: str
    extract: Extractor
    on_error: str  # "skip" (log, collect nothing) or "raise"


class ProvenanceLedger:
    """Named value-sets recorded from tool outputs, queried by conditions.

    One per session (see :class:`janus.policy.session.Session`); lock-protected
    so concurrent hook callbacks can share an instance. Like ``TaintTracker``,
    sets only grow during a session — values are never removed by ordinary
    operation; only :meth:`reset` clears them, at a session boundary.

    Error handling is asymmetric by design: a collector that raises defaults
    to "collected nothing" (``on_error="skip"``, logged and recorded in
    :attr:`events`). For sets consumed by :func:`from_output` that fails
    **closed** — a missing value denies. For deny-sets consumed by
    :func:`not_in` it would fail **open**, so register those collectors with
    ``on_error="raise"``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._collectors: dict[str, list[_Collector]] = {}
        self._normalizers: dict[str, Normalizer] = {}
        self._sets: dict[str, set[Any]] = {}
        self._events: list[dict[str, Any]] = []
        self._seq = 0
        self._logger = get_logger()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def collect(
        self,
        tool_name: str,
        *,
        label: str,
        extract: Extractor,
        normalize: Normalizer | None = None,
        on_error: str = "skip",
    ) -> None:
        """Register a collector: when ``tool_name`` output is recorded, add
        ``extract(output)``'s values to the set ``label``.

        ``normalize`` is bound to the *label* (it must apply identically at
        record time and check time); registering the same label twice with
        different normalizers is a conflict and raises ``ValueError``.
        """
        if on_error not in ("skip", "raise"):
            raise ValueError(f"on_error must be 'skip' or 'raise', got {on_error!r}")
        with self._lock:
            self._bind_normalizer(label, normalize)
            self._collectors.setdefault(tool_name, []).append(
                _Collector(label=label, extract=extract, on_error=on_error)
            )

    def _bind_normalizer(self, label: str, normalize: Normalizer | None) -> None:
        # Caller holds the lock.
        if normalize is None:
            return
        bound = self._normalizers.get(label)
        if bound is not None and bound is not normalize:
            raise ValueError(
                f"provenance label {label!r} is already bound to a different "
                "normalizer; one label must normalize one way"
            )
        self._normalizers[label] = normalize

    # ------------------------------------------------------------------
    # Recording (post-execution seam)
    # ------------------------------------------------------------------

    def record(self, tool_name: str, output: Any = None) -> list[str]:
        """Run ``tool_name``'s collectors over ``output``; return labels that
        gained at least one new value."""
        grown: list[str] = []
        for collector in self._collectors.get(tool_name, ()):
            try:
                values = list(collector.extract(output))
            except Exception as exc:
                if collector.on_error == "raise":
                    raise
                self._logger.warning(
                    f"PROVENANCE collector for '{collector.label}' failed on "
                    f"'{tool_name}' output ({type(exc).__name__}: {exc}); "
                    "collected nothing"
                )
                self._append_event(
                    kind="collector_error",
                    label=collector.label,
                    tool=tool_name,
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue
            added = self.add(collector.label, values, source=tool_name)
            if added:
                grown.append(collector.label)
        return grown

    def add(
        self,
        label: str,
        values: Iterable[Any],
        *,
        source: str = "manual",
        normalize: Normalizer | None = None,
    ) -> int:
        """Add values to a set directly (used by ``record`` and by
        untrusted-input seeding). Returns how many values were new.

        Values pass through the label's bound normalizer (strings only);
        ``None`` and unhashable values are skipped. ``normalize`` binds a
        normalizer to the label just like :meth:`collect` does (same
        conflict rule).
        """
        with self._lock:
            self._bind_normalizer(label, normalize)
            normalizer = self._normalizers.get(label)
            target = self._sets.setdefault(label, set())
            added = 0
            for value in values:
                if value is None or not isinstance(value, Hashable):
                    continue
                if normalizer is not None and isinstance(value, str):
                    value = normalizer(value)
                if value not in target:
                    target.add(value)
                    added += 1
            if added:
                self._append_event(
                    kind="collect",
                    label=label,
                    tool=source,
                    count=added,
                    locked=True,
                )
            return added

    # ------------------------------------------------------------------
    # Querying (pre-execution seam)
    # ------------------------------------------------------------------

    def contains(self, label: str, value: Any) -> bool:
        """Membership check, applying the label's normalizer to string values.

        A label with no recorded values is simply empty: ``False``. Callers
        gating on this (``from_output``) therefore fail closed by default.
        """
        with self._lock:
            normalizer = self._normalizers.get(label)
            if normalizer is not None and isinstance(value, str):
                value = normalizer(value)
            return value in self._sets.get(label, ())

    def values(self, label: str) -> frozenset:
        """The current (normalized) value-set for a label."""
        with self._lock:
            return frozenset(self._sets.get(label, ()))

    @property
    def labels(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._sets)

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def record_miss(
        self, label: str, value: Any, *, tool_name: str, arg_name: str, kind: str = "miss"
    ) -> str:
        """Record that a provenance condition denied a call; returns the event
        id (surfaced in deny reasons so a human can endorse exactly this deny
        via ``Session.endorse_event``). Called by the condition factories, not
        by integrators."""
        return self._append_event(kind=kind, label=label, tool=tool_name, arg=arg_name, value=value)

    def _append_event(self, *, locked: bool = False, **event: Any) -> str:
        if locked:  # caller already holds the lock
            self._seq += 1
            entry = {**event, "id": f"prov-{self._seq}", "time": time.time()}
            self._events.append(entry)
        else:
            with self._lock:
                self._seq += 1
                entry = {**event, "id": f"prov-{self._seq}", "time": time.time()}
                self._events.append(entry)
        return entry["id"]

    @property
    def events(self) -> list[dict[str, Any]]:
        """Audit trail: collections, denials caused by misses, collector errors."""
        with self._lock:
            return list(self._events)

    def reset(self) -> None:
        """Clear all sets and the audit trail (session boundary only).

        Registered collectors and normalizers survive a reset — they are
        configuration, not session state.
        """
        with self._lock:
            self._sets.clear()
            self._events.clear()
            self._seq = 0


# ---------------------------------------------------------------------------
# Condition factories
# ---------------------------------------------------------------------------


def _consume_endorsement(ctx: ConditionContext, value: Any) -> bool:
    """Lift this deny iff a value-scoped endorsement matches the exact triple.

    Raw-equality match on the argument value as passed — endorse exactly what
    was denied (``Session.endorse_event`` does this mechanically from the
    audit id in the deny reason).
    """
    log = getattr(ctx.session, "endorsements", None)
    if log is None:
        return False
    return log.consume(tool=ctx.tool_name, arg=ctx.arg_name, value=value) is not None


def _ledger_of(ctx: ConditionContext, label: str) -> ProvenanceLedger:
    ledger = getattr(ctx.session, "provenance", None)
    if ledger is None:
        raise ArgumentValidationError(
            argument_name=ctx.arg_name,
            value=None,
            restriction=label,
            message=(
                f"Provenance condition on '{ctx.arg_name}' (set {label!r}) was "
                "evaluated without a wired Session; failing closed. Pass "
                "session= through the adapter (janus_options / janus_hooks) "
                "or PolicyEnforcer.enforce()."
            ),
        )
    return ledger


def from_output(label: str):
    """Allow the argument only if its value is in the provenance set ``label``.

    Positive provenance: the value must literally have been recorded from a
    prior listed tool output (via :meth:`ProvenanceLedger.collect` +
    ``record``). An empty or missing set denies; a missing session denies.
    Every deny is recorded in the ledger's audit trail.
    """

    @context_condition
    def provenance_from_output(value: Any, ctx: ConditionContext) -> bool:
        ledger = _ledger_of(ctx, label)
        if ledger.contains(label, value):
            return True
        if _consume_endorsement(ctx, value):
            return True
        event_id = ledger.record_miss(label, value, tool_name=ctx.tool_name, arg_name=ctx.arg_name)
        raise ArgumentValidationError(
            argument_name=ctx.arg_name,
            value=value,
            restriction=label,
            message=(
                f"Argument '{ctx.arg_name}' value {value!r} is not a recorded "
                f"value of provenance set {label!r} — it must come verbatim "
                "from a prior listed tool output in this session. "
                f"(audit id {event_id})"
            ),
        )

    provenance_from_output.__name__ = f"from_output({label!r})"
    return provenance_from_output


def not_in(label: str):
    """Deny the argument if its value is in the provenance set ``label``.

    Negative provenance, for untrusted-seeded sets ("this recipient must not
    be an address that appeared in the inbound email"). A missing session
    denies — an unwired deny-set is indistinguishable from an unchecked one,
    and guessing would fail open. Remember the collector caveat: feed
    deny-sets with ``on_error="raise"`` so a broken extractor cannot silently
    empty the set.
    """

    @context_condition
    def provenance_not_in(value: Any, ctx: ConditionContext) -> bool:
        ledger = _ledger_of(ctx, label)
        if ledger.contains(label, value):
            if _consume_endorsement(ctx, value):
                return True
            event_id = ledger.record_miss(
                label,
                value,
                tool_name=ctx.tool_name,
                arg_name=ctx.arg_name,
                kind="deny_match",
            )
            raise ArgumentValidationError(
                argument_name=ctx.arg_name,
                value=value,
                restriction=label,
                message=(
                    f"Argument '{ctx.arg_name}' value {value!r} appears in the "
                    f"untrusted provenance set {label!r} and is refused. "
                    f"(audit id {event_id})"
                ),
            )
        return True

    provenance_not_in.__name__ = f"not_in({label!r})"
    return provenance_not_in
