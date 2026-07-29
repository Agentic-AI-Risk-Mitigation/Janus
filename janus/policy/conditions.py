"""
Context-aware policy conditions and condition composition.

The classic callable-condition contract is ``restriction(value) -> truthy``:
one scalar in, no way to see the tool name, sibling arguments, or session
state. That boundary is exactly where real integrations fall back to
hand-rolled checks inside tool bodies (the ``secure`` pipeline's
``allowed_urls`` guard). This module widens the contract without breaking it:

- :class:`ConditionContext` — what a condition may additionally see: the tool
  name, the argument name, a **read-only** view of the full call, and the
  per-run session state (``None`` until one is wired).
- :func:`context_condition` — an explicit opt-in marker. Marked callables are
  invoked as ``restriction(value, ctx)``; everything else keeps the classic
  single-argument contract untouched. The marker is an attribute, not
  signature inspection: ``functools.wraps`` copies it, partials and decorated
  callables don't silently change contract, and the dispatch stays a contract
  rather than a heuristic.
- :func:`all_of` / :func:`any_of` — compose restrictions of *any* supported
  kind (JSON Schema dict, regex string, plain callable, context condition),
  since a real rule usually stacks checks ("passes the SSRF check AND came
  from a prior search"). ``all_of`` fails closed on the first failing member
  and propagates that member's message.

Failure semantics are unchanged from the classic contract: truthy allows,
falsy or a raised exception denies (surfaced as ``ArgumentValidationError``
by the validator).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypeVar

from janus.exceptions import ArgumentValidationError

# Attribute marking a callable as context-aware. Checked with getattr() by the
# validator so this module never becomes an import cycle.
CONTEXT_MARKER = "__janus_context__"

_C = TypeVar("_C", bound=Callable)


@dataclass(frozen=True)
class ConditionContext:
    """What a context condition may see beyond the argument value itself.

    Attributes:
        tool_name: Policy key of the tool being called (post ``resolve_name``).
        arg_name: Name of the argument this condition constrains.
        arguments: Read-only mapping of the *full* call arguments, so a
            condition can constrain one argument in terms of another.
        session: Per-run session state (provenance ledger, taint, …) when the
            integration wires one; ``None`` otherwise. Conditions that need
            session state must fail closed on ``None``.
    """

    tool_name: str
    arg_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    session: Any = None

    @staticmethod
    def build(
        tool_name: str,
        arg_name: str,
        arguments: Mapping[str, Any],
        session: Any = None,
    ) -> ConditionContext:
        """Construct a context with a read-only copy of ``arguments``."""
        return ConditionContext(
            tool_name=tool_name,
            arg_name=arg_name,
            arguments=MappingProxyType(dict(arguments)),
            session=session,
        )


def context_condition(fn: _C) -> _C:
    """Mark a callable as context-aware: it will be called as ``fn(value, ctx)``.

    Apply closest to the function (below other decorators) so
    ``functools.wraps``-based wrappers copy the marker along with ``__dict__``.
    """
    setattr(fn, CONTEXT_MARKER, True)
    return fn


def is_context_condition(restriction: Any) -> bool:
    """True if ``restriction`` opted into the ``(value, ctx)`` contract."""
    return bool(getattr(restriction, CONTEXT_MARKER, False))


class _Composite:
    """Base for composed restrictions. Context-aware so members can be too."""

    __janus_context__ = True
    _joiner = ""

    def __init__(self, *restrictions: Any):
        if not restrictions:
            # An empty composite has no defined semantics; vacuous-allow would
            # fail open, so refuse construction outright.
            raise ValueError(f"{type(self).__name__} requires at least one restriction")
        self.restrictions = restrictions

    def __repr__(self) -> str:  # surfaces in deny messages — keep it readable
        members = ", ".join(_describe(r) for r in self.restrictions)
        return f"{self._joiner}({members})"


def _describe(restriction: Any) -> str:
    name = getattr(restriction, "__name__", None)
    return name if name else repr(restriction)


class _AllOf(_Composite):
    _joiner = "all_of"

    def __call__(self, value: Any, ctx: ConditionContext) -> bool:
        from janus.policy.validator import validate_argument

        for restriction in self.restrictions:
            # Raises ArgumentValidationError on the first failing member; the
            # outer validator re-raises it as-is, so the member's message —
            # not a generic composite message — reaches the deny reason.
            validate_argument(ctx.arg_name, value, restriction, context=ctx)
        return True


class _AnyOf(_Composite):
    _joiner = "any_of"

    def __call__(self, value: Any, ctx: ConditionContext) -> bool:
        from janus.policy.validator import validate_argument

        failures: list[str] = []
        for restriction in self.restrictions:
            try:
                validate_argument(ctx.arg_name, value, restriction, context=ctx)
                return True
            except ArgumentValidationError as exc:
                failures.append(exc.message)
        raise ArgumentValidationError(
            argument_name=ctx.arg_name,
            value=value,
            restriction=self,
            message=(
                f"Argument '{ctx.arg_name}' value {value!r} satisfied none of "
                f"{self!r}: " + " | ".join(failures)
            ),
        )


def all_of(*restrictions: Any) -> _AllOf:
    """Compose restrictions that must **all** pass (fail-closed conjunction).

    Members may be any supported restriction type — JSON Schema dict, regex
    string, plain callable, or context condition — evaluated in order; the
    first failure denies with that member's own message.
    """
    return _AllOf(*restrictions)


def any_of(*restrictions: Any) -> _AnyOf:
    """Compose restrictions where **at least one** must pass (disjunction).

    Denies only when every member fails, with the member messages joined so
    the deny reason shows what was tried.
    """
    return _AnyOf(*restrictions)
