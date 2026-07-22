"""
Janus policy module.

Provides policy enforcement, loading, validation, and LLM-based generation.
The ``pde`` subpackage contains SpiceDB-backed ReBAC enforcement with taint tracking.

``PolicyEnforcer`` and the static-policy tooling (loader, validator) import only
stdlib + ``jsonschema`` + ``pydantic``, so this package can be used standalone
without the optional ``pde`` extra (SpiceDB/authzed) installed. ``PDEEnforcer`` is
resolved lazily via :pep:`562` module-level ``__getattr__`` so that importing it —
or the package as a whole — does not eagerly pull in ``authzed``.
"""

from typing import Any

from janus.policy.enforcer import PolicyEnforcer
from janus.policy.loader import parse_policy, save_policy, validate_policy_structure
from janus.policy.taint import TaintTracker
from janus.policy.validator import validate_argument, validate_schema

__all__ = [
    "PDEEnforcer",
    "PolicyEnforcer",
    "TaintTracker",
    "parse_policy",
    "save_policy",
    "validate_policy_structure",
    "validate_argument",
    "validate_schema",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve ``PDEEnforcer`` so ``authzed`` stays an optional dependency.

    Importing this package (or ``PolicyEnforcer`` from it) must not require the
    ``pde`` extra. ``PDEEnforcer`` still imports under the same public name once
    ``janus-guard[pde]`` is installed; without it, accessing the name raises an
    actionable :class:`ImportError`.
    """
    if name == "PDEEnforcer":
        try:
            from janus.policy.pde_enforcer import PDEEnforcer
        except ImportError as exc:
            raise ImportError(
                "PDE/taint enforcement requires the 'pde' extra: pip install 'janus-guard[pde]'"
            ) from exc
        return PDEEnforcer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
