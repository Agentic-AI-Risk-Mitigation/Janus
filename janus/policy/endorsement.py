"""
Endorsements: audited, consumable declassification for in-the-loop tasks.

Blocking is the wrong primitive when untrusted data legitimately must drive
the action (reply to the sender of an inbound email, fetch the URL an
operator reported). The alternative that keeps gating honest is an
**endorsement**: an explicit, attributed, narrowly-scoped statement that one
specific deny may pass.

Deliberately narrow semantics:

- **Value-scoped by default.** An endorsement names an exact
  ``(tool, arg, value)`` triple and satisfies only the deny that matches it.
  Values compare by raw equality — endorse exactly what was denied (use
  :meth:`janus.policy.Session.endorse_event` to do that mechanically).
- **Taint-scope is explicit.** ``scope="taint"`` lifts a whole-tool taint
  gate for one tool — broader, so it must be asked for by name and carries
  no arg/value.
- **Consumable.** ``uses=1`` by default: one deny lifted, then the gate is
  closed again. ``uses=None`` is a standing endorsement and is logged at
  warning level on every consumption — a standing declassification should be
  loud in the logs.
- **Never un-taints.** Taint stays monotonic; an endorsement is checked *at*
  the gate or condition and changes nothing else.
- **Audited.** Creation and every consumption land in :attr:`events` with
  who/why, so the trail answers "who approved sending to this address,
  when, and for how many calls".
"""

from __future__ import annotations

import threading
import time
from typing import Any

from janus.logger import get_logger

SCOPE_VALUE = "value"
SCOPE_TAINT = "taint"


class EndorsementLog:
    """Holds a session's endorsements; consulted by gates and conditions.

    One per :class:`janus.policy.Session`; lock-protected for concurrent hook
    callbacks. Integrators normally go through ``Session.endorse`` /
    ``Session.endorse_event`` rather than instantiating this directly.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._seq = 0
        self._logger = get_logger()

    def endorse(
        self,
        *,
        tool: str,
        arg: str | None = None,
        value: Any = None,
        scope: str = SCOPE_VALUE,
        by: str,
        reason: str,
        uses: int | None = 1,
    ) -> str:
        """Record an endorsement; returns its id.

        ``scope="value"`` (default) requires ``arg`` and ``value`` and lifts
        only a deny of that exact triple. ``scope="taint"`` requires *no*
        arg/value and lifts the whole-tool taint gate for ``tool``.
        ``by`` and ``reason`` are mandatory attribution — an unattributed
        declassification is not an endorsement, it is a hole.
        """
        if scope == SCOPE_VALUE:
            if arg is None or value is None:
                raise ValueError(
                    "value-scoped endorsement requires arg= and value= "
                    "(endorse exactly what was denied)"
                )
        elif scope == SCOPE_TAINT:
            if arg is not None or value is not None:
                raise ValueError("taint-scoped endorsement takes no arg/value")
        else:
            raise ValueError(f"unknown endorsement scope {scope!r}")
        if not by or not reason:
            raise ValueError("endorsements require by= and reason= attribution")
        if uses is not None and uses < 1:
            raise ValueError("uses must be >= 1, or None for a standing endorsement")

        with self._lock:
            self._seq += 1
            endorsement_id = f"end-{self._seq}"
            self._entries.append(
                {
                    "id": endorsement_id,
                    "scope": scope,
                    "tool": tool,
                    "arg": arg,
                    "value": value,
                    "by": by,
                    "reason": reason,
                    "uses": uses,
                }
            )
            self._events.append(
                {
                    "kind": "endorse",
                    "id": endorsement_id,
                    "scope": scope,
                    "tool": tool,
                    "arg": arg,
                    "value": value,
                    "by": by,
                    "reason": reason,
                    "uses": uses,
                    "time": time.time(),
                }
            )
        return endorsement_id

    def consume(
        self,
        *,
        tool: str,
        arg: str | None = None,
        value: Any = None,
        scope: str = SCOPE_VALUE,
    ) -> str | None:
        """Consume one use of a matching endorsement, if any.

        Returns the endorsement id when a deny may be lifted, else ``None``.
        Standing endorsements (``uses=None``) never deplete but are logged at
        warning level on every consumption.
        """
        with self._lock:
            for entry in self._entries:
                if (
                    entry["scope"] == scope
                    and entry["tool"] == tool
                    and entry["arg"] == arg
                    and entry["value"] == value
                    and (entry["uses"] is None or entry["uses"] > 0)
                ):
                    if entry["uses"] is not None:
                        entry["uses"] -= 1
                    self._events.append(
                        {
                            "kind": "endorse_consume",
                            "id": entry["id"],
                            "scope": scope,
                            "tool": tool,
                            "arg": arg,
                            "value": value,
                            "remaining": entry["uses"],
                            "time": time.time(),
                        }
                    )
                    if entry["uses"] is None:
                        self._logger.warning(
                            f"ENDORSEMENT standing endorsement {entry['id']} consumed "
                            f"(tool={tool!r}, arg={arg!r}) — standing declassifications "
                            "should be rare and deliberate"
                        )
                    return entry["id"]
        return None

    @property
    def events(self) -> list[dict[str, Any]]:
        """Audit trail: every endorsement created and every consumption."""
        with self._lock:
            return list(self._events)

    def reset(self) -> None:
        """Clear endorsements and their audit trail (session boundary only)."""
        with self._lock:
            self._entries.clear()
            self._events.clear()
            self._seq = 0
