"""End-to-end demo: Janus policy enforcement on a Claude Agent SDK tool loop.

Unlike the scripted web-demo scenarios (which use a mock LLM), this is a *live* example: it
drives the real `claude` CLI via the Claude Agent SDK, with a Janus policy enforced at the
`PreToolUse` hook seam (`janus.adapters.claude_agent_sdk.janus_hooks`).

Three toy tools are exposed to the model:
  * ``echo``        — allowed, with a length-bounded argument.
  * ``fetch``       — allowed only for https non-internal URLs (a callable SSRF condition).
  * ``read_secret`` — deliberately NOT in the policy, so Janus default-denies it.

The model is asked to call all three. You should see ``echo`` succeed, and both ``fetch``
(internal URL) and ``read_secret`` (unlisted) get blocked before their bodies run, with the
denial reason handed back to the model.

Requirements:
    uv add "janus-guard[claude]"     # the `claude` CLI must be on PATH
    export CLAUDE_CODE_OAUTH_TOKEN=... # or ANTHROPIC_API_KEY

Run:
    uv run python -m examples.claude_agent_sdk_demo
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile

try:
    from claude_agent_sdk import (
        ClaudeAgentOptions,
        ResultMessage,
        create_sdk_mcp_server,
        query,
        tool,
    )
except ImportError:
    raise SystemExit(
        'This demo needs the Claude Agent SDK. Install it with: uv add "janus-guard[claude]"'
    )

from janus.adapters.claude_agent_sdk import janus_hooks

# --- toy tools (bodies never run for a blocked call) --------------------------------

@tool("echo", "Echo a short piece of text back.", {"text": str})
async def echo_tool(args):
    return {"content": [{"type": "text", "text": f"ECHO: {args['text']}"}]}


@tool("fetch", "Fetch a URL and return its text.", {"url": str})
async def fetch_tool(args):
    # Real code would fetch here; the point is that Janus blocks the internal URL first.
    return {"content": [{"type": "text", "text": f"FETCHED: {args['url']}"}]}


@tool("read_secret", "Read the operator's secret credential file.", {})
async def read_secret_tool(args):
    return {"content": [{"type": "text", "text": "SECRET: hunter2"}]}


SERVER = create_sdk_mcp_server("demo", tools=[echo_tool, fetch_tool, read_secret_tool])


# --- Janus policy (keyed on bare names; the adapter strips the mcp__demo__ prefix) --

def _url_ok(url: str) -> bool:
    """Callable condition: allow only https, non-internal URLs; raise to block."""
    if not url.startswith("https://") or "169.254." in url or "localhost" in url:
        raise ValueError(f"blocked URL (SSRF guard): {url!r}")
    return True


TOOL_POLICY = {
    "echo": [(1, 0, {"text": {"type": "string", "maxLength": 100}}, 0)],
    "fetch": [(1, 0, {"url": _url_ok}, 0)],
    # read_secret intentionally omitted -> Janus default-deny.
}


PROMPT = (
    "Do these steps in order, each as a separate tool call, even if one fails:\n"
    "1. Call echo with text='hello janus'.\n"
    "2. Call fetch with url='http://169.254.169.254/latest/meta-data'.\n"
    "3. Call read_secret.\n"
    "Then report, for each call, whether it succeeded or was denied and quote any denial reason."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "echo_ok": {"type": "boolean"},
        "fetch_ok": {"type": "boolean"},
        "read_secret_ok": {"type": "boolean"},
        "denial_reasons": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["echo_ok", "fetch_ok", "read_secret_ok", "denial_reasons"],
}


async def _streaming_prompt():
    yield {"type": "user", "message": {"role": "user", "content": PROMPT}}


async def main() -> None:
    if not (os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")):
        raise SystemExit("Set CLAUDE_CODE_OAUTH_TOKEN (or ANTHROPIC_API_KEY) to run this demo.")

    options = ClaudeAgentOptions(
        system_prompt="You are a tool-driving test harness. Follow instructions exactly.",
        model=os.environ.get("DEMO_MODEL", "sonnet"),
        tools=[],                       # built-ins off; only the demo server is callable
        setting_sources=[],             # hermetic
        permission_mode="dontAsk",
        mcp_servers={"demo": SERVER},
        allowed_tools=["mcp__demo__echo", "mcp__demo__fetch", "mcp__demo__read_secret"],
        output_format={"type": "json_schema", "schema": SCHEMA},
        # The Janus gate. required_args backstops the absent-argument bypass for `fetch`.
        hooks=janus_hooks(TOOL_POLICY, required_args={"fetch": ["url"]}),
        cli_path=shutil.which("claude"),
        cwd=tempfile.gettempdir(),
        max_turns=12,
    )

    result: ResultMessage | None = None
    async for msg in query(prompt=_streaming_prompt(), options=options):
        if isinstance(msg, ResultMessage):
            result = msg

    if result is None:
        print("No result returned.")
        return
    print("structured_output:", json.dumps(result.structured_output, indent=2))
    print(f"\n(cost: ${result.total_cost_usd or 0:.4f}, turns: {result.num_turns})")
    print("Expected: echo_ok=true, fetch_ok=false, read_secret_ok=false.")


if __name__ == "__main__":
    asyncio.run(main())
