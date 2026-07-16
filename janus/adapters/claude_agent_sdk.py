"""
Janus × Claude Agent SDK (``claude-agent-sdk``) integration adapter.

The Claude Agent SDK runs the tool loop inside the ``claude`` CLI subprocess, so
Janus cannot sit in the call path the way it does when it owns the loop
(``JanusAgent`` / ``ToolRegistry``). Instead this adapter plugs a
``PolicyEnforcer`` into the two pre-execution seams the SDK exposes, so a policy
decision still happens *before* a tool runs:

┌──────────────────────────────────────────────────────────────────────────┐
│  Primary — janus_pretooluse_hook() / janus_hooks()                        │
│  A ``PreToolUse`` hook. Fires for EVERY tool call, even ones listed in     │
│  ``allowed_tools`` and under ``permission_mode="dontAsk"``. This is the    │
│  robust seam — use it.                                                     │
├──────────────────────────────────────────────────────────────────────────┤
│  Alternative — make_can_use_tool()                                         │
│  A ``can_use_tool`` callback. Cleaner return type, but the SDK lets a      │
│  whole-tool ``allowed_tools`` entry (and ``permission_mode`` values like   │
│  ``bypassPermissions``) auto-approve a call BEFORE the callback is         │
│  consulted — so it is silently bypassed for allow-listed tools. Only safe  │
│  when nothing shadows it. See the docstring warning.                       │
├──────────────────────────────────────────────────────────────────────────┤
│  Belt-and-braces — guard_tool_body()                                       │
│  Wrap an in-process ``@tool`` body so enforcement also runs at execution,  │
│  independent of any SDK permission semantics.                              │
└──────────────────────────────────────────────────────────────────────────┘

Two SDK-specific facts this adapter handles for you, both learned from live
runs (see ``docs`` / the ``claude-agent-sdk-integration`` note):

  * **Tool names are prefixed.** An in-process ``@tool("fetch", …)`` mounted on
    an MCP server named ``"research"`` is invoked as ``mcp__research__fetch``.
    Your policy is normally keyed on the bare name (``fetch``), so this adapter
    maps the runtime name back to the policy key before enforcing. Override via
    ``resolve_name`` if your policy uses the full prefixed names.

  * **``StructuredOutput`` must pass through.** When you set
    ``ClaudeAgentOptions.output_format``, the SDK returns the final structured
    result via an internal ``StructuredOutput`` tool call that the PreToolUse
    hook also sees. A default-deny policy would block it and the run's
    ``structured_output`` would come back ``None``. Such SDK-internal tools are
    passed through untouched (configurable via ``passthrough_tools``).

Installation requirement::

    pip install claude-agent-sdk        # or: uv add claude-agent-sdk

Note: the primary hook path (``janus_pretooluse_hook`` / ``janus_hooks``) returns
plain dicts and does **not** import the SDK, so it works even where the SDK is
not installed. Only ``make_can_use_tool`` and ``janus_hooks`` (which builds
``HookMatcher`` objects) touch the SDK.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any

from janus.adapters._base import PolicySource, resolve_enforcer
from janus.exceptions import PolicyViolation
from janus.logger import get_logger
from janus.policy.enforcer import RequiredArgs, check_required_args

# SDK-internal tools that are part of the transport, not the agent's toolset, and
# must never be policy-gated. ``StructuredOutput`` delivers an ``output_format``
# result; blocking it silently nulls ``ResultMessage.structured_output``.
DEFAULT_PASSTHROUGH_TOOLS = frozenset({"StructuredOutput"})

# Strips the SDK's ``mcp__<server>__`` prefix from an in-process tool name so the
# bare name can be matched against a policy. Non-greedy, so a server name that
# itself contains single underscores (``mcp__my_server__fetch``) still resolves.
_MCP_PREFIX_RE = re.compile(r"^mcp__.+?__")

NameResolver = Callable[[str], str]

# The required-args backstop now lives in the core engine
# (janus.policy.enforcer.check_required_args); this adapter keeps its per-call
# ``required_args`` parameter and delegates. With the core enforcer's strict
# condition semantics (strict_conditions=True default), an allow rule already
# refuses calls that omit a conditioned argument — required_args remains useful
# for arguments no condition covers and for rejecting blank strings.
_check_required_args = check_required_args


def default_resolve_name(tool_name: str) -> str:
    """Map an SDK runtime tool name to its policy key (strip ``mcp__server__``)."""
    return _MCP_PREFIX_RE.sub("", tool_name)


def _decide(
    enforcer,
    runtime_name: str,
    arguments: dict,
    *,
    passthrough_tools: frozenset[str],
    resolve_name: NameResolver,
    required_args: RequiredArgs,
) -> str | None:
    """Shared decision core. Returns ``None`` to allow, or a deny reason string.

    Passthrough tools return ``None`` (allow) without consulting the policy.
    """
    if runtime_name in passthrough_tools:
        return None

    policy_key = resolve_name(runtime_name)
    try:
        _check_required_args(policy_key, arguments, required_args)
        enforcer.enforce(policy_key, arguments)
        return None
    except PolicyViolation as exc:
        return exc.reason


# =============================================================================
# Primary seam — PreToolUse hook
# =============================================================================


def janus_pretooluse_hook(
    policy: PolicySource,
    *,
    required_args: RequiredArgs | None = None,
    passthrough_tools: frozenset[str] = DEFAULT_PASSTHROUGH_TOOLS,
    resolve_name: NameResolver = default_resolve_name,
) -> Callable[[dict, str | None, Any], Awaitable[dict]]:
    """Build a ``PreToolUse`` hook callback that enforces a Janus policy.

    This is the seam to use. A ``PreToolUse`` hook fires for every tool call
    regardless of ``allowed_tools`` or ``permission_mode``, so the policy is not
    bypassable the way ``can_use_tool`` is.

    Parameters
    ----------
    policy : str | Path | dict | PolicyEnforcer | None
        Any source ``resolve_enforcer`` accepts.
    required_args : dict[str, list[str]] | None
        ``{policy_key: [arg, …]}`` — arguments that must be present and non-empty
        for that tool. The core enforcer's strict condition semantics already
        deny calls that omit a conditioned argument; this additionally covers
        arguments no condition names, and blank strings. Keyed on the
        resolved (bare) policy name, e.g. ``{"fetch_page": ["url"]}``.
    passthrough_tools : frozenset[str]
        Tool names allowed without consulting the policy. Defaults to the
        SDK-internal ``{"StructuredOutput"}``; extend it, don't drop it, if you
        use ``output_format``.
    resolve_name : Callable[[str], str]
        Maps the SDK runtime tool name to the policy key. Defaults to stripping
        the ``mcp__server__`` prefix. Pass ``lambda n: n`` if your policy is
        keyed on the full prefixed names.

    Returns
    -------
    An async hook callback suitable for a ``HookMatcher``. On a policy block it
    returns a ``PreToolUse`` ``permissionDecision: "deny"`` whose reason is fed
    back to the model, so the agent can adjust rather than crash.

    Example::

        from claude_agent_sdk import ClaudeAgentOptions, HookMatcher
        from janus.adapters.claude_agent_sdk import janus_pretooluse_hook

        hook = janus_pretooluse_hook(TOOL_POLICY, required_args={"fetch_page": ["url"]})
        options = ClaudeAgentOptions(
            mcp_servers={"research": server},
            allowed_tools=["mcp__research__web_search", "mcp__research__fetch_page"],
            permission_mode="dontAsk",
            output_format={"type": "json_schema", "schema": SCHEMA},
            hooks={"PreToolUse": [HookMatcher(hooks=[hook])]},
        )
    """
    enforcer = resolve_enforcer(policy)
    required = required_args or {}
    logger = get_logger()

    async def hook(input_data: dict, tool_use_id: str | None, context: Any) -> dict:
        runtime_name = input_data.get("tool_name", "")
        arguments = dict(input_data.get("tool_input") or {})
        logger.tool_call(runtime_name, arguments)

        reason = _decide(
            enforcer, runtime_name, arguments,
            passthrough_tools=passthrough_tools,
            resolve_name=resolve_name,
            required_args=required,
        )
        if reason is None:
            logger.policy_decision(runtime_name, allowed=True)
            return {}

        logger.policy_decision(runtime_name, allowed=False, reason=reason)
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"[Janus] blocked by policy: {reason}",
            }
        }

    return hook


def janus_hooks(
    policy: PolicySource,
    *,
    required_args: RequiredArgs | None = None,
    passthrough_tools: frozenset[str] = DEFAULT_PASSTHROUGH_TOOLS,
    resolve_name: NameResolver = default_resolve_name,
) -> dict:
    """Convenience wrapper: return a ready ``hooks=`` dict for ``ClaudeAgentOptions``.

    Equivalent to building ``{"PreToolUse": [HookMatcher(hooks=[hook])]}`` yourself
    from :func:`janus_pretooluse_hook`. Imports the SDK (for ``HookMatcher``); use
    :func:`janus_pretooluse_hook` directly if you want to avoid the import or set a
    matcher pattern.
    """
    try:
        from claude_agent_sdk import HookMatcher
    except ImportError as exc:  # pragma: no cover - exercised via _require message
        raise ImportError(
            "claude-agent-sdk is required for janus_hooks().\n"
            "Install with: pip install claude-agent-sdk"
        ) from exc

    hook = janus_pretooluse_hook(
        policy, required_args=required_args,
        passthrough_tools=passthrough_tools, resolve_name=resolve_name,
    )
    # The hook is intentionally typed with a broad dict signature so this module
    # imports without the SDK; it is shape-correct for the SDK's HookCallback.
    return {"PreToolUse": [HookMatcher(hooks=[hook])]}  # type: ignore[list-item]


# =============================================================================
# Alternative seam — can_use_tool callback
# =============================================================================


def make_can_use_tool(
    policy: PolicySource,
    *,
    required_args: RequiredArgs | None = None,
    passthrough_tools: frozenset[str] = DEFAULT_PASSTHROUGH_TOOLS,
    resolve_name: NameResolver = default_resolve_name,
) -> Callable[[str, dict, Any], Awaitable[Any]]:
    """Build a ``can_use_tool`` callback that enforces a Janus policy.

    .. warning::
       ``can_use_tool`` is **bypassable**. The SDK auto-approves any tool listed
       as a whole-tool entry in ``allowed_tools`` (and every tool under
       ``permission_mode="bypassPermissions"``) *before* the callback runs — the
       callback simply never fires for those calls, and Janus never sees them.
       The SDK emits a ``CanUseToolShadowedWarning`` when it detects this. Prefer
       :func:`janus_pretooluse_hook`, which cannot be shadowed. Use this callback
       only when no ``allowed_tools`` entry and no bypassing permission mode is in
       play. Also note ``can_use_tool`` requires the streaming prompt form
       (an ``AsyncIterable``, not a plain string).

    Returns an async ``(tool_name, arguments, context)`` callback returning
    ``PermissionResultAllow`` / ``PermissionResultDeny(message=…)``.
    """
    try:
        from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny
    except ImportError as exc:
        raise ImportError(
            "claude-agent-sdk is required for make_can_use_tool().\n"
            "Install with: pip install claude-agent-sdk"
        ) from exc

    enforcer = resolve_enforcer(policy)
    required = required_args or {}
    logger = get_logger()

    async def can_use_tool(tool_name: str, arguments: dict, context: Any):
        args = dict(arguments or {})
        logger.tool_call(tool_name, args)
        reason = _decide(
            enforcer, tool_name, args,
            passthrough_tools=passthrough_tools,
            resolve_name=resolve_name,
            required_args=required,
        )
        if reason is None:
            logger.policy_decision(tool_name, allowed=True)
            return PermissionResultAllow()
        logger.policy_decision(tool_name, allowed=False, reason=reason)
        return PermissionResultDeny(message=f"[Janus] blocked by policy: {reason}")

    return can_use_tool


# =============================================================================
# Belt-and-braces — wrap an in-process @tool body
# =============================================================================


def guard_tool_body(
    tool_name: str,
    body: Callable[[dict], Awaitable[dict]],
    policy: PolicySource,
    *,
    required_args: RequiredArgs | None = None,
    resolve_name: NameResolver = default_resolve_name,
) -> Callable[[dict], Awaitable[dict]]:
    """Wrap an async ``@tool`` handler so Janus also enforces at execution time.

    Independent of any SDK permission seam: even if a hook/callback is
    misconfigured or shadowed, a blocked call never runs its body. The wrapped
    body returns an SDK tool-result dict describing the block (so the model sees
    it as a tool result) instead of raising.

    ``body`` and the returned callable take the SDK ``@tool`` shape:
    ``async def(args: dict) -> {"content": [...]}``.
    """
    enforcer = resolve_enforcer(policy)
    required = required_args or {}
    logger = get_logger()

    async def guarded(args: dict) -> dict:
        arguments = dict(args or {})
        policy_key = resolve_name(tool_name)
        try:
            _check_required_args(policy_key, arguments, required)
            enforcer.enforce(policy_key, arguments)
        except PolicyViolation as exc:
            logger.policy_decision(tool_name, allowed=False, reason=exc.reason)
            return {
                "content": [{"type": "text",
                             "text": f"[Janus] blocked by policy: {exc.reason}"}],
                "isError": True,
            }
        logger.policy_decision(tool_name, allowed=True)
        return await body(arguments)

    guarded.__name__ = getattr(body, "__name__", tool_name)
    guarded.__doc__ = body.__doc__ or ""
    return guarded
