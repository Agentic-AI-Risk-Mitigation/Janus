"""
Claude Agent SDK based agent implementation.

This is a THIN ADAPTER that:
1. Uses the claude-agent-sdk to spawn Claude Code as a subprocess
2. Claude Code handles tool execution internally (Read, Write, Bash, etc.)
3. Progent policy enforcement is applied via the can_use_tool callback

Authentication is via CLAUDE_CODE_OAUTH_TOKEN environment variable.
Requires the Claude Code CLI: npm install -g @anthropic-ai/claude-code
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

if TYPE_CHECKING:
    from claude_agent_sdk import PermissionResult
    from claude_agent_sdk import ToolPermissionContext as _ToolPermissionContext

try:
    import anyio
    from claude_agent_sdk import (
        ClaudeAgentOptions,
        PermissionResultAllow,
        PermissionResultDeny,
        ToolPermissionContext,
        query,
    )
    from claude_agent_sdk.types import HookMatcher

    CLAUDE_SDK_AVAILABLE = True
except ImportError:
    CLAUDE_SDK_AVAILABLE = False

from implementations.core.progent_enforcer import enforce_policy, init_progent
from implementations.core.tool_definitions import TOOL_DEFINITIONS
from implementations.frameworks.base_agent import BaseAgent

# Default tools Claude Code can use
DEFAULT_ALLOWED_TOOLS = [
    "Bash",
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
]

# Mapping from Claude Code tool names to (progent_tool_name, arg_mapper_fn).
# The arg mapper translates Claude Code's input dict to Progent's expected kwargs.
CLAUDE_TO_PROGENT_MAP: dict[str, tuple[str, Any]] = {
    "Bash": ("run_command", lambda inp: {"command": inp.get("command", "")}),
    "Read": ("read_file", lambda inp: {"file_path": inp.get("file_path", "")}),
    "Write": (
        "write_file",
        lambda inp: {"file_path": inp.get("file_path", ""), "content": inp.get("content", "")},
    ),
    "Edit": (
        "edit_file",
        lambda inp: {
            "file_path": inp.get("file_path", ""),
            "old_string": inp.get("old_string", ""),
            "new_string": inp.get("new_string", ""),
        },
    ),
    "Glob": ("list_directory", lambda inp: {"path": inp.get("pattern", ".")}),
}


@dataclass
class SDKConfig:
    """Configuration for ClaudeSDKRunner.

    Attributes:
        cwd: Working directory for Claude Code.
        allowed_tools: List of tool names Claude is allowed to use.
        mcp_servers: MCP server configurations keyed by server name.
        can_use_tool: Optional callback for policy enforcement on tool calls.
            Signature: async (tool_name: str, input_data: dict, context: ToolPermissionContext)
                       -> PermissionResultAllow | PermissionResultDeny
            Returns PermissionResult (Allow or Deny).
    """

    cwd: str = "."
    allowed_tools: list[str] = field(default_factory=list)
    mcp_servers: dict[str, dict] = field(default_factory=dict)
    can_use_tool: (
        Callable[[str, dict[str, Any], "_ToolPermissionContext"], Awaitable["PermissionResult"]]
        | None
    ) = None


class ClaudeSDKRunner:
    """Runs Claude Code queries via the claude-agent-sdk.

    Wraps the async query() interface into a synchronous run_query() method
    with a per-message callback.
    """

    def __init__(self, config: SDKConfig):
        self.config = config

    def _build_options(self) -> ClaudeAgentOptions:
        """Build ClaudeAgentOptions from our SDKConfig."""
        # When can_use_tool is set, don't use acceptEdits — that auto-approves
        # without consulting the callback. Use default mode so the CLI routes
        # permission requests through the control protocol to our callback.
        if self.config.can_use_tool:
            permission_mode = "default"
            print("[PROGENT] can_use_tool is SET — using permission_mode='default'")
        else:
            permission_mode = "acceptEdits"

        opts = ClaudeAgentOptions(
            allowed_tools=self.config.allowed_tools,
            cwd=self.config.cwd,
            permission_mode=permission_mode,
            stderr=lambda line: print(f"[CLI STDERR] {line}"),
        )
        if self.config.mcp_servers:
            opts.mcp_servers = self.config.mcp_servers
        if self.config.can_use_tool:
            opts.can_use_tool = self.config.can_use_tool

            # CRITICAL: Python SDK requires a PreToolUse hook to keep stream open
            # Without this, the stream closes before can_use_tool can be invoked
            async def _dummy_hook(input_data, tool_use_id, context):
                """Dummy hook that returns continue=True to keep stream alive."""
                return {"continue_": True}

            opts.hooks = {"PreToolUse": [HookMatcher(matcher=None, hooks=[_dummy_hook])]}
            print("[PROGENT] Added dummy PreToolUse hook to keep stream open")
        return opts

    async def _as_async_iter(self, prompt: str, done_event: anyio.Event):
        """Wrap a prompt string as an AsyncIterable.

        Required when can_use_tool is set. Keeps the iterator alive until
        done_event is set, so the SDK's stream_input doesn't close stdin
        before control protocol messages (can_use_tool responses) are sent.
        """
        yield {"type": "user", "message": {"role": "user", "content": prompt}}
        print("[PROGENT] Async iter: waiting to keep stdin open...")
        await done_event.wait()
        print("[PROGENT] Async iter: released, stdin will close")

    async def _run_async(
        self,
        prompt: str,
        on_message: Callable[[Any], None] | None = None,
    ) -> tuple[bool, dict]:
        """Run a query asynchronously and collect results."""
        options = self._build_options()
        result_info: dict[str, Any] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
        }
        success = False
        final_text = ""

        # can_use_tool requires streaming mode (AsyncIterable prompt).
        # Use anyio.Event (not asyncio.Event) since the SDK uses anyio internally.
        done_event = anyio.Event()
        prompt_input = (
            self._as_async_iter(prompt, done_event) if self.config.can_use_tool else prompt
        )

        async for message in query(prompt=prompt_input, options=options):
            if on_message:
                on_message(message)

            # Collect token usage from result messages
            if hasattr(message, "num_turns"):
                success = True
                done_event.set()  # Signal async iter to finish
                if hasattr(message, "total_cost_usd") and message.total_cost_usd:
                    result_info["cost_usd"] = message.total_cost_usd
            if hasattr(message, "content"):
                for block in message.content:
                    if hasattr(block, "text"):
                        final_text += block.text

        result_info["text"] = final_text
        return success, result_info

    def run_query(
        self,
        prompt: str,
        on_message: Callable[[Any], None] | None = None,
    ) -> tuple[bool, dict]:
        """Run a Claude Code query synchronously.

        Args:
            prompt: The prompt to send to Claude.
            on_message: Optional callback invoked for each streamed message/event.

        Returns:
            Tuple of (success: bool, result_info: dict) where result_info contains
            keys: text, input_tokens, output_tokens, cost_usd.
        """
        return asyncio.run(self._run_async(prompt, on_message))


class ClaudeSDKAgent(BaseAgent):
    """
    Agent implementation using the Claude Agent SDK.

    Spawns Claude Code as a subprocess. Claude Code handles its own
    tool execution loop — this adapter sends prompts, collects output,
    and enforces Progent policies via the can_use_tool callback.
    """

    def __init__(
        self,
        config: dict[str, Any],
        workspace: str | Path,
        policies_path: Optional[str | Path] = None,
    ):
        if not CLAUDE_SDK_AVAILABLE:
            raise ImportError(
                "Claude Agent SDK not installed. Install with: uv add claude-agent-sdk"
            )

        # DEBUG: Print immediately to verify this code is running
        print("=" * 70, file=sys.stderr)
        print("[DEBUG] ClaudeSDKAgent.__init__() called", file=sys.stderr)
        print(f"[DEBUG]   policies_path = {policies_path!r}", file=sys.stderr)
        print(f"[DEBUG]   policies_path type = {type(policies_path)}", file=sys.stderr)
        print(f"[DEBUG]   policies_path bool = {bool(policies_path)}", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        sys.stderr.flush()

        # Override model display — Claude Code manages its own model
        config.setdefault("llm", {})["model"] = "claude (via Claude Code)"

        super().__init__(config, workspace)

        # Check for auth token
        oauth_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
        if oauth_token:
            self.logger.info(f"CLAUDE_CODE_OAUTH_TOKEN is set (prefix: {oauth_token[:20]}...)")
        else:
            self.logger.info("CLAUDE_CODE_OAUTH_TOKEN is NOT set — using API key auth")

        # Initialize Progent policies
        if policies_path:
            try:
                print(f"[PROGENT] Loading policies from: {policies_path}", file=sys.stderr)
                sys.stderr.flush()
                tool_defs = [
                    {"name": t.name, "description": t.description, "args": {}}
                    for t in TOOL_DEFINITIONS
                ]
                init_progent(policies_path, tool_defs)
                print("[PROGENT] Policies loaded successfully", file=sys.stderr)
                sys.stderr.flush()
            except Exception as e:
                print(f"[PROGENT] ERROR loading policies: {e}", file=sys.stderr)
                import traceback

                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                raise
        else:
            print(f"[PROGENT] WARNING: policies_path is falsy: {policies_path!r}", file=sys.stderr)
            sys.stderr.flush()

        # Build allowed tools list from config or defaults
        self.allowed_tools = config.get("agent", {}).get("allowed_tools", DEFAULT_ALLOWED_TOOLS)

        # Build SDK config with policy enforcement callback
        sdk_config = SDKConfig(
            cwd=str(self.workspace),
            allowed_tools=self.allowed_tools,
            mcp_servers=config.get("agent", {}).get("mcp_servers", {}),
            can_use_tool=self._enforce_tool_policy if policies_path else None,
        )

        self.runner = ClaudeSDKRunner(sdk_config)

        # Verify callback is set
        if policies_path:
            print("[PROGENT] Policy enforcement ENABLED")
            print(f"[PROGENT]   Callback: {self._enforce_tool_policy}")
            print(f"[PROGENT]   Is async: {asyncio.iscoroutinefunction(self._enforce_tool_policy)}")
            print(f"[PROGENT]   Will intercept: {list(CLAUDE_TO_PROGENT_MAP.keys())}")
        else:
            print("[PROGENT] Policy enforcement DISABLED (no policies_path)")

        # System prompt (passed as part of the prompt since SDK manages its own system prompt)
        self.system_prompt = config.get("agent", {}).get("system_prompt", "")

    async def _enforce_tool_policy(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        context: ToolPermissionContext,
    ) -> PermissionResultAllow | PermissionResultDeny:
        """Enforce Progent policy on a Claude Code tool call.

        Translates Claude Code tool names/args to Progent equivalents,
        then calls enforce_policy(). Returns SDK-compatible PermissionResult.
        """
        print(f"\n{'=' * 60}")
        print("[PROGENT] can_use_tool INVOKED!")
        print(f"[PROGENT]   tool: {tool_name}")
        print(f"[PROGENT]   input: {input_data}")
        print(f"{'=' * 60}\n")

        mapping = CLAUDE_TO_PROGENT_MAP.get(tool_name)
        if mapping is None:
            print(f"[PROGENT] WARNING: No mapping for tool '{tool_name}'")
            print(f"[PROGENT]   Available mappings: {list(CLAUDE_TO_PROGENT_MAP.keys())}")
            print("[PROGENT]   Allowing by default (no policy enforcement)")
            return PermissionResultAllow()

        progent_name, arg_mapper = mapping
        progent_kwargs = arg_mapper(input_data)
        print(f"[PROGENT] Mapped to progent tool='{progent_name}', kwargs={progent_kwargs}")

        allowed, reason = enforce_policy(progent_name, progent_kwargs)
        print(f"[PROGENT] Policy decision: allowed={allowed}, reason='{reason}'")

        if allowed:
            return PermissionResultAllow()
        return PermissionResultDeny(message=reason)

    def _handle_message(self, event: Any) -> None:
        """Log each streamed message/event from Claude Code."""
        if hasattr(event, "content"):
            for block in event.content:
                if hasattr(block, "text") and block.text:
                    self.logger.debug(f"Claude: {block.text[:200]}")
                elif hasattr(block, "name"):
                    self.logger.debug(f"Tool use: {block.name}")

    def run(self, user_input: str) -> str:
        """Process user input and return response."""
        try:
            # Prepend system prompt context if set
            prompt = user_input
            if self.system_prompt:
                prompt = f"{self.system_prompt}\n\nUser request: {user_input}"

            success, result_info = self.runner.run_query(
                prompt=prompt,
                on_message=self._handle_message,
            )

            if result_info.get("cost_usd"):
                self.logger.info(f"Cost: ${result_info['cost_usd']:.4f}")

            if success:
                return result_info.get("text", "No response generated.")
            else:
                return "Claude SDK query failed."

        except Exception as e:
            self.logger.error("Claude SDK agent execution error", e)
            return f"Error: {e}"

    def get_tools(self) -> list:
        """Get the list of allowed tools (built-in Claude Code tool names)."""
        return self.allowed_tools

    def clear_history(self) -> None:
        """Clear conversation history (no-op — each query is a fresh session)."""
        pass
