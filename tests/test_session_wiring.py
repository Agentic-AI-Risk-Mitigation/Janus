"""
Session wiring through the Claude Agent SDK adapter.

Offline, like the other adapter seam tests: the hooks are called directly
with the SDK's PreToolUse/PostToolUse input shapes. What must hold:

- ``janus_hooks(session=...)`` wires both seams; the PostToolUse hook unwraps
  MCP content blocks so provenance extractors see the tool body's dict.
- The full loop closes: a search response recorded at PostToolUse makes
  exactly that URL fetchable at PreToolUse, hook-level deny for the rest.
- ``taint=`` keeps working (deprecation-warned); ``taint=`` + ``session=``
  together are refused.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from janus.adapters.claude_agent_sdk import (
    janus_hooks,
    janus_options,
    janus_posttooluse_hook,
    janus_pretooluse_hook,
    unwrap_tool_response,
)
from janus.policy import Session, TaintTracker, from_output, normalize_url

POLICY = {
    "web_search": [(1, 0, {"query": {"type": "string", "maxLength": 400}}, 0)],
    "fetch_page": [(1, 0, {"url": from_output("searched_urls")}, 0)],
    "send_email": [(1, 0, {}, 0)],
}


def _search_urls(out):
    return [r.get("url") for r in (out.get("results") or []) if r.get("url")]


def make_session() -> Session:
    session = Session(taint=TaintTracker(sources={"web_search": "web"}, gates={"send_email": "*"}))
    session.provenance.collect(
        "web_search", label="searched_urls", extract=_search_urls, normalize=normalize_url
    )
    return session


def _mcp_response(payload: dict) -> dict:
    return {"content": [{"type": "text", "text": json.dumps(payload)}]}


def _run_pre(hook, tool: str, args: dict) -> dict:
    data = {"hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": args}
    return asyncio.run(hook(data, "tu-1", None))


def _run_post(hook, tool: str, response) -> dict:
    data = {"hook_event_name": "PostToolUse", "tool_name": tool, "tool_response": response}
    return asyncio.run(hook(data, "tu-1", None))


def _denied(out: dict) -> bool:
    return (out.get("hookSpecificOutput") or {}).get("permissionDecision") == "deny"


# --- unwrap_tool_response --------------------------------------------------------


def test_unwrap_content_blocks_to_dict():
    payload = {"ok": True, "results": [{"url": "https://a.example/"}]}
    assert unwrap_tool_response(_mcp_response(payload)) == payload
    assert unwrap_tool_response([{"type": "text", "text": json.dumps(payload)}]) == payload


def test_unwrap_non_json_text_and_foreign_shapes():
    assert unwrap_tool_response({"content": [{"type": "text", "text": "plain"}]}) == "plain"
    assert unwrap_tool_response({"already": "a dict"}) == {"already": "a dict"}
    assert unwrap_tool_response("raw string") == "raw string"
    assert unwrap_tool_response({"content": [{"type": "image"}]}) == {
        "content": [{"type": "image"}]
    }


# --- the full loop through both seams --------------------------------------------


def test_session_loop_search_then_fetch():
    session = make_session()
    hooks = janus_hooks(POLICY, session=session)
    pre = hooks["PreToolUse"][0].hooks[0]
    post = hooks["PostToolUse"][0].hooks[0]

    # Nothing searched yet: no fetch is possible (fail closed).
    assert _denied(_run_pre(pre, "mcp__research__fetch_page", {"url": "https://a.example/x"}))

    # Search runs; its (MCP-wrapped, differently-cased) results are recorded.
    assert _run_pre(pre, "mcp__research__web_search", {"query": "CVE-2023-23752"}) == {}
    _run_post(
        post,
        "mcp__research__web_search",
        _mcp_response({"ok": True, "results": [{"url": "HTTPS://A.Example/x"}]}),
    )

    # Exactly the searched URL is now fetchable; everything else stays denied.
    assert _run_pre(pre, "mcp__research__fetch_page", {"url": "https://a.example/x"}) == {}
    out = _run_pre(pre, "mcp__research__fetch_page", {"url": "https://evil.example/"})
    assert _denied(out)
    assert "searched_urls" in out["hookSpecificOutput"]["permissionDecisionReason"]

    # And the search tainted the session, so the gated sink is closed.
    gated = _run_pre(pre, "mcp__research__send_email", {"to": "x@y.example"})
    assert _denied(gated)
    assert "tainted" in gated["hookSpecificOutput"]["permissionDecisionReason"]


def test_post_hook_ignores_missing_response_and_survives_unwrap_errors():
    session = make_session()
    post = janus_posttooluse_hook(session, unwrap=unwrap_tool_response)
    _run_post(post, "mcp__research__web_search", None)  # denied/blocked call: no record
    assert session.provenance.values("searched_urls") == frozenset()

    def broken_unwrap(response):
        raise RuntimeError("boom")

    post_broken = janus_posttooluse_hook(session, unwrap=broken_unwrap)
    # Falls back to the raw response; the extractor collects nothing from it.
    _run_post(post_broken, "mcp__research__web_search", _mcp_response({"results": []}))
    assert session.provenance.values("searched_urls") == frozenset()


# --- parameter contract ----------------------------------------------------------


def test_taint_alone_still_works_with_deprecation_warning():
    tracker = TaintTracker(sources={"web_search": "web"}, gates={"send_email": "*"})
    with pytest.warns(DeprecationWarning, match="session="):
        hook = janus_pretooluse_hook(POLICY, taint=tracker)
    tracker.record_output("web_search")
    assert _denied(_run_pre(hook, "mcp__research__send_email", {"to": "x@y.example"}))


def test_taint_and_session_together_are_refused():
    with pytest.raises(ValueError, match="not both"):
        janus_pretooluse_hook(POLICY, taint=TaintTracker(), session=Session())


def test_janus_options_wires_posttooluse_with_session():
    pytest.importorskip("claude_agent_sdk")
    from claude_agent_sdk import create_sdk_mcp_server, tool

    @tool("web_search", "Search the web", {"query": str})
    async def web_search(args):
        return {"content": []}

    server = create_sdk_mcp_server("research", tools=[web_search])
    session = make_session()
    opts = janus_options(POLICY, mcp_servers={"research": server}, session=session)
    assert "PostToolUse" in opts.hooks
    with pytest.raises(ValueError, match="not both"):
        janus_options(
            POLICY,
            mcp_servers={"research": server},
            taint=TaintTracker(),
            session=session,
        )
