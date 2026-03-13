"""
PDE (Policy Discovery Engine) subpackage.

SpiceDB-backed ReBAC enforcement with runtime taint tracking.
"""

from janus.policy.pde.config import RISK_TO_TAINT, SCHEMA, TOOL_TAINT_LIMIT

__all__ = [
    "RISK_TO_TAINT",
    "SCHEMA",
    "TOOL_TAINT_LIMIT",
]
