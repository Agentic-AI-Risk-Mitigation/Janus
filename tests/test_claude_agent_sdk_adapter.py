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
    DEFAULT_DISALLOWED_TOOLS,
    default_resolve_name,
    guard_tool_body,
    janus_options,
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


# --- fail closed on Janus's own defects ------------------------------------------


def test_hook_denies_on_unexpected_exception():
    # A raising resolve_name stands in for any internal enforcement bug
    # (malformed tool_input, enforcer defect, taint defect). The hook must
    # deny, not error-and-proceed (which the CLI treats as fail-open).
    def broken(name: str) -> str:
        raise RuntimeError("resolver bug")

    hook = janus_pretooluse_hook(POLICY, resolve_name=broken)
    out = _run_hook(hook, "mcp__research__web_search", {"query": "hi"})
    assert _denied(out)
    assert "failing closed" in out["hookSpecificOutput"]["permissionDecisionReason"]


def test_can_use_tool_denies_on_unexpected_exception():
    pytest.importorskip("claude_agent_sdk")
    from claude_agent_sdk import PermissionResultDeny

    def broken(name: str) -> str:
        raise RuntimeError("resolver bug")

    cb = make_can_use_tool(POLICY, resolve_name=broken)
    out = asyncio.run(cb("mcp__research__web_search", {"query": "hi"}, None))
    assert isinstance(out, PermissionResultDeny)
    assert "failing closed" in out.message


# --- hook-approved sinks: explicit allow decision --------------------------------


def test_hook_approved_tool_gets_explicit_allow():
    hook = janus_pretooluse_hook(POLICY, hook_approved_tools={"web_search"})
    out = _run_hook(hook, "mcp__research__web_search", {"query": "hi"})
    hso = out.get("hookSpecificOutput") or {}
    assert hso.get("permissionDecision") == "allow"
    # Non-approved tools keep the neutral {} (permission layer decides)
    hook2 = janus_pretooluse_hook(POLICY, hook_approved_tools={"web_search"})
    assert _run_hook(hook2, "mcp__research__fetch_page", {"url": "https://a.example/"}) == {}
    # Deny still denies
    assert _denied(_run_hook(hook, "mcp__research__web_search", {"query": "x" * 500}))


# --- janus_options(): the locked-down options builder (SDK required) -------------


def _make_server():
    pytest.importorskip("claude_agent_sdk")
    from claude_agent_sdk import create_sdk_mcp_server, tool

    @tool("web_search", "Search the web", {"query": str})
    async def web_search(args):
        return {"content": []}

    @tool("fetch_page", "Fetch a page", {"url": str})
    async def fetch_page(args):
        return {"content": []}

    @tool("read_secret", "Read a secret", {})
    async def read_secret(args):
        return {"content": []}

    return create_sdk_mcp_server("research", tools=[web_search, fetch_page, read_secret])


def test_janus_options_lockdown_fields():
    server = _make_server()
    opts = janus_options(POLICY, mcp_servers={"research": server})

    assert opts.tools == []
    assert opts.strict_mcp_config is True
    assert opts.permission_mode == "dontAsk"
    assert opts.setting_sources is None                 # SDK default: no filesystem settings
    assert "Task" in opts.disallowed_tools
    assert "Agent" in opts.disallowed_tools  # Task's runtime name on CLI >= 2.1.x
    assert set(DEFAULT_DISALLOWED_TOOLS) <= set(opts.disallowed_tools)
    assert opts.hooks and "PreToolUse" in opts.hooks
    assert "PostToolUse" not in opts.hooks              # no taint tracker passed


def test_janus_options_wires_posttooluse_with_taint():
    from janus.policy.taint import TaintTracker

    server = _make_server()
    tracker = TaintTracker(sources={"fetch_page": "web"}, gates={"web_search": "*"})
    opts = janus_options(POLICY, mcp_servers={"research": server}, taint=tracker)
    assert "PostToolUse" in opts.hooks


def test_janus_options_allowed_is_policy_intersect_mounted():
    server = _make_server()
    opts = janus_options(POLICY, mcp_servers={"research": server})
    # read_secret is mounted but unknown to the policy -> unreachable
    assert sorted(opts.allowed_tools) == [
        "mcp__research__fetch_page",
        "mcp__research__web_search",
    ]


def test_janus_options_hook_approved_kept_off_allowed_tools():
    server = _make_server()
    opts = janus_options(
        POLICY, mcp_servers={"research": server}, hook_approved_tools={"fetch_page"}
    )
    assert opts.allowed_tools == ["mcp__research__web_search"]
    # Unknown-to-policy hook_approved entries could never be approved -> loud error
    with pytest.raises(ValueError, match="hook_approved_tools"):
        janus_options(
            POLICY, mcp_servers={"research": server}, hook_approved_tools={"nope"}
        )


@pytest.mark.parametrize("override", [
    {"tools": ["Bash"]},
    {"strict_mcp_config": False},
    {"permission_mode": "bypassPermissions"},
    {"permission_mode": "acceptEdits"},
    {"can_use_tool": lambda *a: None},
    {"setting_sources": ["project"]},
    {"hooks": {}},
])
def test_janus_options_guarded_overrides_raise(override):
    server = _make_server()
    with pytest.raises(ValueError, match="unsafe_overrides"):
        janus_options(POLICY, mcp_servers={"research": server}, **override)
    # ... and pass through with unsafe_overrides=True
    janus_options(
        POLICY, mcp_servers={"research": server}, unsafe_overrides=True, **override
    )


def test_janus_options_merges_shrink_only():
    server = _make_server()
    opts = janus_options(
        POLICY,
        mcp_servers={"research": server},
        disallowed_tools=["mcp__research__fetch_page"],
        # allowed_tools additions still pass the policy filter: read_secret and
        # a made-up name are unknown to the policy and must NOT become reachable
        allowed_tools=["mcp__research__read_secret", "mcp__research__made_up"],
    )
    assert "mcp__research__fetch_page" in opts.disallowed_tools
    assert set(DEFAULT_DISALLOWED_TOOLS) <= set(opts.disallowed_tools)
    assert "mcp__research__read_secret" not in opts.allowed_tools
    assert "mcp__research__made_up" not in opts.allowed_tools


def test_janus_options_non_enumerable_server_errors():
    _make_server()  # skip when SDK absent
    external = {"type": "stdio", "command": "some-mcp-server"}
    with pytest.raises(ValueError, match="cannot enumerate"):
        janus_options(POLICY, mcp_servers={"ext": external})
    # An explicit allowed_tools merge acknowledges the server; entries are
    # still policy-filtered.
    opts = janus_options(
        POLICY,
        mcp_servers={"ext": external},
        allowed_tools=["mcp__ext__fetch_page", "mcp__ext__evil_tool"],
    )
    assert opts.allowed_tools == ["mcp__ext__fetch_page"]


def test_janus_options_empty_policy_errors():
    server = _make_server()
    with pytest.raises(ValueError, match="at least one rule"):
        janus_options({}, mcp_servers={"research": server})


def test_janus_options_extra_hooks_merge_alongside():
    pytest.importorskip("claude_agent_sdk")
    from claude_agent_sdk import HookMatcher

    async def user_hook(input_data, tool_use_id, context):
        return {}

    server = _make_server()
    extra = {"PreToolUse": [HookMatcher(hooks=[user_hook])], "Stop": [HookMatcher(hooks=[user_hook])]}
    opts = janus_options(POLICY, mcp_servers={"research": server}, extra_hooks=extra)
    assert len(opts.hooks["PreToolUse"]) == 2       # Janus's first, user's appended
    assert "Stop" in opts.hooks


def test_janus_options_import_error_without_sdk(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    with pytest.raises(ImportError, match="claude-agent-sdk is required"):
        janus_options(POLICY, mcp_servers={})


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
