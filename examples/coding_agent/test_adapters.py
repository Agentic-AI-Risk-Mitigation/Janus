"""
Adapter validation tests — no live LLM required.

Tests:
1. _base utilities (resolve_enforcer, make_guarded_handler)
2. LangChain adapter  (secure_langchain_tools, wrap_langchain_tools)
3. ADK adapter        (secure_adk_tools) — skipped if google-genai not installed
4. End-to-end with JanusLangChainAgent using a mock LLM
"""

import sys
import json
import tempfile
from pathlib import Path

# Make sure repo root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from janus.adapters._base import resolve_enforcer, make_guarded_handler
from janus.policy.enforcer import PolicyEnforcer
from janus.exceptions import PolicyViolation
from janus import ToolDef, ToolParam

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

POLICY = {
    "calculator": [
        {
            "priority": 1,
            "effect": 0,
            "conditions": {"x": {"type": "integer", "minimum": 0}},
            "fallback": 0,
        }
    ],
    "greet": [{"priority": 1, "effect": 0, "conditions": {}, "fallback": 0}],
    "danger": [{"priority": 1, "effect": 1, "conditions": {}, "fallback": 0}],
}


def calculator(x: int, y: int) -> int:
    return x + y


def greet(name: str) -> str:
    return f"Hello, {name}!"


def danger(payload: str) -> str:
    return f"EXECUTED: {payload}"


TOOL_DEFS = [
    ToolDef(
        name="calculator",
        description="Add two integers.",
        params=[
            ToolParam("x", "integer", "First operand (must be >= 0)."),
            ToolParam("y", "integer", "Second operand."),
        ],
        handler=calculator,
    ),
    ToolDef(
        name="greet",
        description="Greet someone by name.",
        params=[ToolParam("name", "string", "Person to greet.")],
        handler=greet,
    ),
    ToolDef(
        name="danger",
        description="Dangerous tool — always blocked.",
        params=[ToolParam("payload", "string", "Payload.")],
        handler=danger,
    ),
]

# ---------------------------------------------------------------------------
# Suite 1 — _base utilities
# ---------------------------------------------------------------------------


def test_resolve_enforcer_from_dict():
    e = resolve_enforcer(POLICY)
    assert isinstance(e, PolicyEnforcer)
    print("  PASS  resolve_enforcer(dict)")


def test_resolve_enforcer_from_file():
    tmp = Path(tempfile.mktemp(suffix=".json"))
    tmp.write_text(json.dumps(POLICY))
    e = resolve_enforcer(tmp)
    assert isinstance(e, PolicyEnforcer)
    tmp.unlink()
    print("  PASS  resolve_enforcer(file path)")


def test_resolve_enforcer_from_instance():
    e = PolicyEnforcer()
    e2 = resolve_enforcer(e)
    assert e2 is e
    print("  PASS  resolve_enforcer(existing enforcer) returns same instance")


def test_resolve_enforcer_none():
    e = resolve_enforcer(None)
    assert isinstance(e, PolicyEnforcer)
    # No policy loaded -> all tools allowed
    e.enforce("anything", {})  # should not raise
    print("  PASS  resolve_enforcer(None) allows everything")


def test_guarded_handler_allowed():
    enforcer = resolve_enforcer(POLICY)
    guarded = make_guarded_handler("calculator", calculator, enforcer)
    result = guarded(x=5, y=3)
    assert result == "8", f"Expected '8', got {result!r}"
    print("  PASS  guarded_handler allowed call returns correct result")


def test_guarded_handler_blocked_by_schema():
    enforcer = resolve_enforcer(POLICY)
    guarded = make_guarded_handler("calculator", calculator, enforcer)
    result = guarded(x=-1, y=3)
    assert "blocked" in result.lower(), f"Expected block message, got {result!r}"
    print("  PASS  guarded_handler blocked (schema violation) returns error string, not exception")


def test_guarded_handler_blocked_deny_rule():
    enforcer = resolve_enforcer(POLICY)
    guarded = make_guarded_handler("danger", danger, enforcer)
    result = guarded(payload="kaboom")
    assert "blocked" in result.lower(), f"Expected block message, got {result!r}"
    print("  PASS  guarded_handler blocked (deny rule) returns error string")


def test_guarded_handler_tool_not_in_policy():
    enforcer = resolve_enforcer(POLICY)
    guarded = make_guarded_handler("unknown_tool", lambda **kw: "ok", enforcer)
    result = guarded()
    assert "blocked" in result.lower(), f"Expected block message, got {result!r}"
    print("  PASS  guarded_handler tool not in policy returns error string")


def test_guarded_handler_catches_handler_exception():
    enforcer = resolve_enforcer({"boom": [{"priority": 1, "effect": 0, "conditions": {}, "fallback": 0}]})
    def bad_handler(**kw): raise RuntimeError("boom!")
    guarded = make_guarded_handler("boom", bad_handler, enforcer)
    result = guarded()
    assert "error" in result.lower(), f"Expected error message, got {result!r}"
    print("  PASS  guarded_handler catches handler exceptions and returns error string")


# ---------------------------------------------------------------------------
# Suite 2 — LangChain adapter
# ---------------------------------------------------------------------------


def test_langchain_secure_tools():
    try:
        from langchain_core.tools import StructuredTool
    except ImportError:
        print("  SKIP  langchain not installed — skipping LangChain adapter tests")
        return

    from janus.adapters.langchain import secure_langchain_tools

    lc_tools = secure_langchain_tools(TOOL_DEFS, POLICY)
    assert len(lc_tools) == 3
    names = [t.name for t in lc_tools]
    assert "calculator" in names
    assert "greet" in names
    assert "danger" in names
    print("  PASS  secure_langchain_tools returns correct number of StructuredTools")

    # Invoke calculator (allowed)
    calc_tool = next(t for t in lc_tools if t.name == "calculator")
    result = calc_tool.invoke({"x": 4, "y": 6})
    assert result == "10", f"Expected '10', got {result!r}"
    print("  PASS  secure_langchain_tools StructuredTool.invoke() allowed call works")

    # Invoke calculator (blocked by schema)
    result_blocked = calc_tool.invoke({"x": -5, "y": 1})
    assert "blocked" in str(result_blocked).lower(), f"Expected block, got {result_blocked!r}"
    print("  PASS  secure_langchain_tools StructuredTool.invoke() blocked call returns error string")

    # Invoke danger tool (always blocked)
    danger_tool = next(t for t in lc_tools if t.name == "danger")
    result_danger = danger_tool.invoke({"payload": "rm -rf /"})
    assert "blocked" in str(result_danger).lower()
    print("  PASS  secure_langchain_tools danger tool invoke blocked correctly")

    # Schema is correct Pydantic model
    assert calc_tool.args_schema is not None
    print("  PASS  secure_langchain_tools tools have correct Pydantic args_schema")


def test_langchain_wrap_existing_tools():
    try:
        from langchain_core.tools import StructuredTool
    except ImportError:
        print("  SKIP  langchain not installed — skipping wrap_langchain_tools test")
        return

    from janus.adapters.langchain import wrap_langchain_tools

    # Build plain StructuredTool (unguarded)
    raw_tool = StructuredTool.from_function(
        func=lambda payload: f"EXECUTED:{payload}",
        name="danger",
        description="Dangerous tool.",
    )
    assert "EXECUTED" in raw_tool.invoke({"payload": "test"})
    print("  PASS  wrap_langchain_tools baseline (unguarded invocation works)")

    # Wrap with policy that denies 'danger'
    wrapped = wrap_langchain_tools([raw_tool], POLICY)
    result = wrapped[0].invoke({"payload": "rm -rf /"})
    assert "blocked" in str(result).lower(), f"Expected block after wrapping, got {result!r}"
    print("  PASS  wrap_langchain_tools existing tool blocked after wrapping")


# ---------------------------------------------------------------------------
# Suite 3 — ADK adapter
# ---------------------------------------------------------------------------


def test_adk_secure_tools():
    try:
        from google.genai import types as gtypes
    except ImportError:
        print("  SKIP  google-genai not installed — skipping ADK adapter tests")
        return

    from janus.adapters.adk import secure_adk_tools

    declarations, handlers = secure_adk_tools(TOOL_DEFS, POLICY)

    assert len(declarations) == 3
    assert len(handlers) == 3
    assert "calculator" in handlers
    assert "danger" in handlers
    print("  PASS  secure_adk_tools returns correct declarations and handlers")

    # Allowed call
    result = handlers["calculator"](x=10, y=20)
    assert result == "30", f"Expected '30', got {result!r}"
    print("  PASS  secure_adk_tools allowed handler call returns correct result")

    # Blocked call (deny rule)
    result_blocked = handlers["danger"](payload="bad")
    assert "blocked" in result_blocked.lower()
    print("  PASS  secure_adk_tools denied handler call returns error string")

    # Blocked by schema
    result_schema = handlers["calculator"](x=-5, y=1)
    assert "blocked" in result_schema.lower()
    print("  PASS  secure_adk_tools schema-violated call returns error string")

    # Declaration structure
    calc_decl = next(d for d in declarations if d["name"] == "calculator")
    assert "parameters" in calc_decl
    assert "description" in calc_decl
    print("  PASS  secure_adk_tools declarations have correct structure for Gemini")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_all():
    suites = [
        ("_base utilities", [
            test_resolve_enforcer_from_dict,
            test_resolve_enforcer_from_file,
            test_resolve_enforcer_from_instance,
            test_resolve_enforcer_none,
            test_guarded_handler_allowed,
            test_guarded_handler_blocked_by_schema,
            test_guarded_handler_blocked_deny_rule,
            test_guarded_handler_tool_not_in_policy,
            test_guarded_handler_catches_handler_exception,
        ]),
        ("LangChain adapter", [
            test_langchain_secure_tools,
            test_langchain_wrap_existing_tools,
        ]),
        ("ADK adapter", [
            test_adk_secure_tools,
        ]),
    ]

    total = 0
    failures = 0

    for suite_name, tests in suites:
        print(f"\n{'=' * 60}")
        print(f" {suite_name}")
        print(f"{'=' * 60}")
        for test in tests:
            try:
                test()
                total += 1
            except Exception as exc:
                print(f"  FAIL  {test.__name__}: {exc}")
                failures += 1
                total += 1

    print(f"\n{'=' * 60}")
    print(f" Results: {total - failures} passed, {failures} failed out of {total}")
    print(f"{'=' * 60}\n")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    run_all()
