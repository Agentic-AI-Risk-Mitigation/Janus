"""Janus × Claude Agent SDK adapter.

Covers the PreToolUse hook (the primary seam), the required-args backstop, the
StructuredOutput passthrough, name resolution, the belt-and-braces body wrapper,
and the can_use_tool callback (SDK-gated, importorskip). The hook tests run fully
offline — no ``claude`` CLI, no subprocess — by calling the callback directly with
the SDK's PreToolUse input shape.
"""
from __future__ import annotations

import asyncio

import pytest

from janus.adapters.claude_agent_sdk import (
    default_resolve_name,
    guard_tool_body,
    janus_pretooluse_hook,
    make_can_use_tool,
)


def _url_ok(url: str) -> bool:
    if not url.startswith("https://") or "169.254." in url or "localhost" in url:
        raise ValueError(f"blocked URL (SSRF guard): {url!r}")
    return True


# Policy keyed on bare names (the common case); the SDK invokes them prefixed.
POLICY = {
    "web_search": [(1, 0, {"query": {"type": "string", "maxLength": 100}}, 0)],
    "fetch_page": [(1, 0, {"url": _url_ok}, 0)],
    # read_secret deliberately unlisted -> default-deny
}
REQUIRED = {"fetch_page": ["url"]}


def _pretool_input(tool_name: str, tool_input: dict) -> dict:
    return {"hook_event_name": "PreToolUse", "tool_name": tool_name, "tool_input": tool_input}


def _run_hook(hook, tool_name: str, tool_input: dict) -> dict:
    return asyncio.run(hook(_pretool_input(tool_name, tool_input), "tool-use-1", None))


def _denied(out: dict) -> bool:
    hso = out.get("hookSpecificOutput") or {}
    return hso.get("permissionDecision") == "deny"


# --- name resolution ------------------------------------------------------------


@pytest.mark.parametrize("runtime,expected", [
    ("mcp__research__fetch_page", "fetch_page"),
    ("mcp__research__web_search", "web_search"),
    ("mcp__my_server__do_thing", "do_thing"),  # server name with a single underscore
    ("StructuredOutput", "StructuredOutput"),  # unprefixed -> unchanged
    ("Bash", "Bash"),
])
def test_default_resolve_name(runtime, expected):
    assert default_resolve_name(runtime) == expected


# --- the hook: allow / condition-deny / default-deny ----------------------------


def test_hook_allows_valid_call():
    hook = janus_pretooluse_hook(POLICY, required_args=REQUIRED)
    out = _run_hook(hook, "mcp__research__web_search", {"query": "CVE-2026-48908"})
    assert out == {}  # empty -> no decision -> allowed


def test_hook_allows_prefixed_fetch():
    hook = janus_pretooluse_hook(POLICY, required_args=REQUIRED)
    out = _run_hook(hook, "mcp__research__fetch_page",
                    {"url": "https://www.cisa.gov/known-exploited-vulnerabilities"})
    assert out == {}


@pytest.mark.parametrize("bad_url", [
    "http://169.254.169.254/latest/meta-data/",
    "http://localhost:8080/search",
    "http://example.com/x",  # not https
])
def test_hook_denies_ssrf_url(bad_url):
    hook = janus_pretooluse_hook(POLICY, required_args=REQUIRED)
    out = _run_hook(hook, "mcp__research__fetch_page", {"url": bad_url})
    assert _denied(out)
    assert "Janus" in out["hookSpecificOutput"]["permissionDecisionReason"]


def test_hook_default_denies_unlisted_tool():
    hook = janus_pretooluse_hook(POLICY, required_args=REQUIRED)
    out = _run_hook(hook, "mcp__research__read_secret", {})
    assert _denied(out)


def test_hook_denies_overlong_query():
    hook = janus_pretooluse_hook(POLICY, required_args=REQUIRED)
    out = _run_hook(hook, "mcp__research__web_search", {"query": "x" * 500})
    assert _denied(out)


# --- required-args backstop (the absent-argument bypass) ------------------------


def test_hook_backstops_missing_url():
    hook = janus_pretooluse_hook(POLICY, required_args=REQUIRED)
    # Without the backstop Janus skips the absent-url condition and would allow.
    assert _denied(_run_hook(hook, "mcp__research__fetch_page", {}))
    assert _denied(_run_hook(hook, "mcp__research__fetch_page", {"url": "   "}))


def test_hook_denies_missing_arg_even_without_required_args():
    # The core enforcer's strict condition semantics (strict_conditions=True,
    # the default) close the historical absent-argument bypass: an allow rule
    # conditioned on ``url`` no longer matches a call that omits ``url``.
    # ``required_args`` remains as a belt-and-braces guard (and catches blank
    # strings a permissive condition schema might let through).
    hook = janus_pretooluse_hook(POLICY)  # no required_args
    assert _denied(_run_hook(hook, "mcp__research__fetch_page", {}))


# --- StructuredOutput passthrough ----------------------------------------------


def test_hook_passes_through_structured_output():
    hook = janus_pretooluse_hook(POLICY, required_args=REQUIRED)
    # Not in the policy, so default-deny WOULD block it and null structured_output.
    out = _run_hook(hook, "StructuredOutput", {"anything": 1})
    assert out == {}


def test_hook_custom_passthrough_set():
    hook = janus_pretooluse_hook(POLICY, passthrough_tools=frozenset({"KeepMe"}))
    assert _run_hook(hook, "KeepMe", {}) == {}
    # StructuredOutput no longer passed through -> default-denied
    assert _denied(_run_hook(hook, "StructuredOutput", {}))


# --- full-prefix policy via resolve_name override -------------------------------


def test_hook_full_prefix_policy():
    prefixed = {"mcp__research__web_search": [(1, 0, {"query": {"type": "string"}}, 0)]}
    hook = janus_pretooluse_hook(prefixed, resolve_name=lambda n: n)
    assert _run_hook(hook, "mcp__research__web_search", {"query": "hi"}) == {}
    assert _denied(_run_hook(hook, "mcp__research__other", {}))


# --- belt-and-braces body wrapper -----------------------------------------------


def test_guard_tool_body_blocks_before_execution():
    ran = {"count": 0}

    async def body(args):
        ran["count"] += 1
        return {"content": [{"type": "text", "text": "did it"}]}

    guarded = guard_tool_body("fetch_page", body, POLICY, required_args=REQUIRED)

    blocked = asyncio.run(guarded({"url": "http://169.254.169.254/"}))
    assert blocked.get("isError") is True
    assert "Janus" in blocked["content"][0]["text"]
    assert ran["count"] == 0  # body never ran

    ok = asyncio.run(guarded({"url": "https://www.cisa.gov/"}))
    assert ok["content"][0]["text"] == "did it"
    assert ran["count"] == 1


def test_guard_tool_body_missing_arg_backstop():
    async def body(args):
        return {"content": [{"type": "text", "text": "ran"}]}

    guarded = guard_tool_body("fetch_page", body, POLICY, required_args=REQUIRED)
    out = asyncio.run(guarded({}))  # no url
    assert out.get("isError") is True


# --- can_use_tool callback (SDK types required) ---------------------------------


def test_can_use_tool_allow_and_deny():
    pytest.importorskip("claude_agent_sdk")
    from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

    cb = make_can_use_tool(POLICY, required_args=REQUIRED)

    allow = asyncio.run(cb("mcp__research__web_search", {"query": "hi"}, None))
    assert isinstance(allow, PermissionResultAllow)

    deny = asyncio.run(cb("mcp__research__fetch_page", {"url": "http://localhost/"}, None))
    assert isinstance(deny, PermissionResultDeny)
    assert "Janus" in deny.message

    passthrough = asyncio.run(cb("StructuredOutput", {}, None))
    assert isinstance(passthrough, PermissionResultAllow)
