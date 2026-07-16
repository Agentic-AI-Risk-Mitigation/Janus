"""
Regression tests for the standalone ``PolicyEnforcer`` import boundary.

These guard the packaging contract: ``PolicyEnforcer`` and the static-policy
path must be importable and usable with the optional ``pde`` extra (SpiceDB /
``authzed``) absent, and the PDE path must fail with a clear, actionable
``ImportError`` rather than a raw ``ModuleNotFoundError`` for ``authzed``.

The ``authzed_absent`` fixture simulates a clean core-only install by purging
all ``janus.*`` and ``authzed.*`` modules from ``sys.modules`` and installing a
meta-path finder that refuses to import ``authzed``. Tests then re-import Janus
from scratch under that block, so this exercises the real import graph even when
``authzed`` happens to be installed in the dev environment.
"""

import sys

import pytest


class _BlockAuthzedFinder:
    """Meta-path finder that makes ``authzed`` (and submodules) unimportable."""

    def find_spec(self, name, path=None, target=None):
        if name == "authzed" or name.startswith("authzed."):
            raise ModuleNotFoundError(f"No module named {name!r} (blocked by test)")
        return None


@pytest.fixture
def authzed_absent(monkeypatch):
    # Drop cached janus + authzed modules so imports re-evaluate under the block.
    for mod in list(sys.modules):
        if (
            mod == "janus"
            or mod.startswith("janus.")
            or mod == "authzed"
            or mod.startswith("authzed.")
        ):
            monkeypatch.delitem(sys.modules, mod, raising=False)

    finder = _BlockAuthzedFinder()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
    # monkeypatch.delitem restores the original sys.modules entries on teardown.


# ---------------------------------------------------------------------------
# Import boundary
# ---------------------------------------------------------------------------


def test_enforcer_importable_without_authzed(authzed_absent):
    from janus.exceptions import PolicyViolation  # noqa: F401
    from janus.policy import PolicyEnforcer as PkgEnforcer
    from janus.policy.enforcer import PolicyEnforcer as ModEnforcer

    assert PkgEnforcer is ModEnforcer
    # Importing the enforcer must not drag in the optional PDE stack.
    assert "authzed" not in sys.modules


def test_import_janus_does_not_import_pde_or_server(authzed_absent):
    # Other test files may legitimately have imported server deps already
    # (claude_agent_sdk pulls in mcp -> uvicorn), so assert on the *delta*:
    # a fresh janus import must not ADD any of these to sys.modules.
    pre = set(sys.modules)

    import janus  # noqa: F401

    added = set(sys.modules) - pre
    assert not {m for m in added if m == "authzed" or m.startswith("authzed.")}
    assert not {m for m in added if m == "fastapi" or m.startswith("fastapi.")}
    assert not {m for m in added if m == "uvicorn" or m.startswith("uvicorn.")}


def test_pde_path_raises_actionable_importerror(authzed_absent):
    from janus import policy

    with pytest.raises(ImportError) as excinfo:
        _ = policy.PDEEnforcer

    msg = str(excinfo.value)
    assert "pde" in msg
    assert "install" in msg.lower()


def test_policy_getattr_rejects_unknown_name(authzed_absent):
    from janus import policy

    with pytest.raises(AttributeError):
        _ = policy.DefinitelyNotAThing


# ---------------------------------------------------------------------------
# Enforcement behavior with authzed absent
# ---------------------------------------------------------------------------


def _fresh_enforcer_cls():
    from janus.policy.enforcer import PolicyEnforcer

    return PolicyEnforcer


def test_callable_condition_predicate(authzed_absent):
    PolicyEnforcer = _fresh_enforcer_cls()
    from janus.exceptions import PolicyViolation

    def is_localhost(url: str) -> bool:
        return url.startswith("http://localhost")

    policy = {"fetch_page": [(1, 0, {"url": is_localhost}, 0)]}
    enforcer = PolicyEnforcer(policy)

    # Predicate returns truthy -> allowed.
    enforcer.enforce("fetch_page", {"url": "http://localhost:8080/x"})

    # Predicate returns falsy -> blocked (no matching allow rule).
    with pytest.raises(PolicyViolation):
        enforcer.enforce("fetch_page", {"url": "http://evil.example/x"})


def test_callable_condition_that_raises(authzed_absent):
    PolicyEnforcer = _fresh_enforcer_cls()
    from janus.exceptions import PolicyViolation

    def reject_ssrf(url: str) -> bool:
        if "169.254.169.254" in url:
            raise ValueError("SSRF target blocked")
        return True

    enforcer = PolicyEnforcer({"fetch_page": [(1, 0, {"url": reject_ssrf}, 0)]})

    enforcer.enforce("fetch_page", {"url": "https://example.com"})
    with pytest.raises(PolicyViolation):
        enforcer.enforce("fetch_page", {"url": "http://169.254.169.254/latest/meta-data"})


def test_json_schema_dict_condition(authzed_absent):
    PolicyEnforcer = _fresh_enforcer_cls()
    from janus.exceptions import PolicyViolation

    schema = {"type": "string", "enum": ["read", "list"]}
    enforcer = PolicyEnforcer({"run_command": [(1, 0, {"action": schema}, 0)]})

    enforcer.enforce("run_command", {"action": "read"})
    with pytest.raises(PolicyViolation):
        enforcer.enforce("run_command", {"action": "delete"})


def test_default_deny_for_unlisted_tool(authzed_absent):
    PolicyEnforcer = _fresh_enforcer_cls()
    from janus.exceptions import PolicyViolation

    enforcer = PolicyEnforcer({"read_file": [(1, 0, {}, 0)]})

    enforcer.enforce("read_file", {"path": "a.txt"})  # allowed
    with pytest.raises(PolicyViolation):
        enforcer.enforce("send_email", {"to": "x@y.z"})  # not listed -> deny


def test_no_policy_allows_everything(authzed_absent):
    PolicyEnforcer = _fresh_enforcer_cls()

    enforcer = PolicyEnforcer()  # no policy loaded
    enforcer.enforce("anything", {"foo": "bar"})  # permissive default


def test_fallback_raise_policy_violation(authzed_absent):
    PolicyEnforcer = _fresh_enforcer_cls()
    from janus.exceptions import PolicyViolation

    # Deny rule, fallback=0 -> raise PolicyViolation.
    enforcer = PolicyEnforcer({"danger": [(1, 1, {}, 0)]})
    with pytest.raises(PolicyViolation):
        enforcer.enforce("danger", {})


def test_fallback_sys_exit(authzed_absent):
    PolicyEnforcer = _fresh_enforcer_cls()

    # Deny rule, fallback=1 -> sys.exit(1) -> SystemExit.
    enforcer = PolicyEnforcer({"danger": [(1, 1, {}, 1)]})
    with pytest.raises(SystemExit):
        enforcer.enforce("danger", {})


def test_fallback_ask_user_rejects(authzed_absent, monkeypatch):
    PolicyEnforcer = _fresh_enforcer_cls()
    from janus.exceptions import PolicyViolation

    # Deny rule, fallback=2 -> prompt; user answers "n" -> PolicyViolation.
    monkeypatch.setattr("builtins.input", lambda: "n")
    enforcer = PolicyEnforcer({"danger": [(1, 1, {}, 2)]})
    with pytest.raises(PolicyViolation):
        enforcer.enforce("danger", {})


def test_fallback_ask_user_approves(authzed_absent, monkeypatch):
    PolicyEnforcer = _fresh_enforcer_cls()

    # Deny rule (fallback=2) followed by an allow rule. User answers "y", so the
    # deny fallback returns without raising and the allow rule permits the call.
    monkeypatch.setattr("builtins.input", lambda: "y")
    enforcer = PolicyEnforcer({"danger": [(1, 1, {}, 2), (2, 0, {}, 0)]})
    enforcer.enforce("danger", {})  # should not raise
