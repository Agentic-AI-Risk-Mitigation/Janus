"""
Janus × Claude Agent SDK (``claude-agent-sdk``) integration adapter.

The Claude Agent SDK runs the tool loop inside the ``claude`` CLI subprocess, so
Janus cannot sit in the call path the way it does when it owns the loop
(``JanusAgent`` / ``ToolRegistry``). Instead this adapter plugs a
``PolicyEnforcer`` into the two pre-execution seams the SDK exposes, so a policy
decision still happens *before* a tool runs:

┌──────────────────────────────────────────────────────────────────────────┐
│  Recommended — janus_options()                                             │
│  Builds a locked-down ``ClaudeAgentOptions``: no built-in tools            │
│  (``tools=[]``), no MCP config leakage (``strict_mcp_config=True``),       │
│  ``allowed_tools`` = policy ∩ mounted, ``permission_mode="dontAsk"``, and  │
│  the Janus hooks wired. Tool-level reachability then no longer depends on  │
│  the hook firing at all.                                                   │
├──────────────────────────────────────────────────────────────────────────┤
│  Primary seam — janus_pretooluse_hook() / janus_hooks()                    │
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

import asyncio
import json
import re
import warnings
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only; runtime import stays lazy
    from claude_agent_sdk import ClaudeAgentOptions

from janus.adapters._base import PolicySource, resolve_enforcer
from janus.exceptions import PolicyViolation
from janus.logger import get_logger
from janus.policy.decision import decide_call
from janus.policy.enforcer import RequiredArgs, check_required_args
from janus.policy.session import Session
from janus.policy.taint import TaintTracker

# SDK-internal tools that are part of the transport, not the agent's toolset, and
# must never be policy-gated. ``StructuredOutput`` delivers an ``output_format``
# result; blocking it silently nulls ``ResultMessage.structured_output``.
DEFAULT_PASSTHROUGH_TOOLS = frozenset({"StructuredOutput"})

# Built-in Claude Code tools that janus_options() denies explicitly. ``tools=[]``
# already removes them at session start; listing them in ``disallowed_tools`` is
# defense in depth should the ``tools`` field's semantics ever shift upstream.
# ``Task`` (invoked as ``Agent`` on current CLIs — both names are denied) is
# excluded deliberately: even though the live smoke suite verified (SDK 0.2.120
# + CLI 2.1.218) that PreToolUse fires for tool calls inside a subagent,
# enabling it also exposes every filesystem-defined agent (``~/.claude``
# agents load regardless of ``setting_sources``), so subagents stay opt-in via
# an explicit ``unsafe_overrides=True`` tools override.
DEFAULT_DISALLOWED_TOOLS: tuple[str, ...] = (
    "Task",
    "Agent",
    "Bash",
    "BashOutput",
    "KillShell",
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "NotebookEdit",
    "Glob",
    "Grep",
    "WebFetch",
    "WebSearch",
    "TodoWrite",
    "ExitPlanMode",
    "Skill",
    "SlashCommand",
    "ListMcpResources",
    "ReadMcpResource",
)

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


def unwrap_tool_response(response: Any) -> Any:
    """Best-effort unwrap of a ``PostToolUse`` ``tool_response`` to the dict
    the tool body returned.

    MCP tool results arrive as content blocks
    (``{"content": [{"type": "text", "text": "<json>"}], ...}`` or a bare
    block list). Provenance extractors are written against the tool's own
    return value, so this pulls the text blocks and attempts ``json.loads``.
    Anything unrecognized is returned unchanged — extractors must tolerate
    the raw shape anyway (and collect nothing on mismatch, which fails
    closed for allow-sets).
    """
    blocks: Any = None
    if isinstance(response, dict) and isinstance(response.get("content"), list):
        blocks = response["content"]
    elif isinstance(response, list):
        blocks = response
    if blocks is None:
        return response

    texts: list[str] = [
        b["text"]
        for b in blocks
        if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
    ]
    if not texts:
        return response
    joined = "\n".join(texts)
    try:
        return json.loads(joined)
    except ValueError:
        return joined


def _resolve_state(
    taint: TaintTracker | None, session: Session | None
) -> tuple[TaintTracker | None, Session | None]:
    """Shared validation for the ``taint=``/``session=`` pair.

    ``session`` supersedes ``taint`` (its ``.taint`` is the gate); passing
    both is ambiguous and refused. ``taint=`` alone still works but warns —
    a Session adds provenance and the merged audit trail for free.
    """
    if taint is not None and session is not None:
        raise ValueError(
            "pass either taint= or session= (session.taint is the gate), not both"
        )
    if taint is not None:
        warnings.warn(
            "taint= is superseded by session= (janus.policy.Session wraps a "
            "TaintTracker and adds provenance); it keeps working but new "
            "integrations should pass session=Session(taint=tracker).",
            DeprecationWarning,
            stacklevel=3,
        )
    return taint, session


def _decide(
    enforcer,
    runtime_name: str,
    arguments: dict,
    *,
    passthrough_tools: frozenset[str],
    resolve_name: NameResolver,
    required_args: RequiredArgs,
    taint: TaintTracker | None = None,
    session: Session | None = None,
) -> str | None:
    """Hook-shaped view of the decision core: ``None`` to allow, else a reason.

    Delegates to :func:`janus.policy.decision.decide_call` — the same core the
    public test harness (``janus.testing``) exposes, so consumer tests exercise
    exactly what the deployed hook runs. Passthrough tools are allowed without
    consulting the policy; the taint gate (``taint`` or ``session.taint``)
    runs before the static policy, and ``session`` is exposed to context-aware
    conditions.
    """
    decision = decide_call(
        enforcer,
        runtime_name,
        arguments,
        passthrough_tools=passthrough_tools,
        resolve_name=resolve_name,
        required_args=required_args,
        taint=taint,
        session=session,
    )
    return None if decision.allowed else decision.reason


# =============================================================================
# Primary seam — PreToolUse hook
# =============================================================================


def janus_pretooluse_hook(
    policy: PolicySource,
    *,
    required_args: RequiredArgs | None = None,
    passthrough_tools: frozenset[str] = DEFAULT_PASSTHROUGH_TOOLS,
    resolve_name: NameResolver = default_resolve_name,
    taint: TaintTracker | None = None,
    session: Session | None = None,
    hook_approved_tools: set[str] | frozenset[str] | None = None,
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
    taint : TaintTracker | None
        Session taint tracker. When supplied, taint gates run before the static
        policy — a sink tool is denied once the session has read a gated
        untrusted source, regardless of arguments. Pair with
        :func:`janus_posttooluse_hook` (same tracker instance) so taint is
        derived automatically from tool outputs; :func:`janus_hooks` wires both
        seams for you. Superseded by ``session=`` (deprecation warning).
    session : Session | None
        Per-run :class:`janus.policy.Session`. Its ``.taint`` supplies the
        gate above, and the whole Session is exposed to context-aware policy
        conditions (``ctx.session``) — this is what provenance conditions
        (``from_output`` / ``not_in``) require. Pass either ``session`` or
        ``taint``, not both. Pair with the ``PostToolUse`` seam (same Session)
        so taint and provenance are recorded automatically.
    hook_approved_tools : set[str] | None
        Bare policy keys (post-``resolve_name``) whose *allow* decision is
        returned as an explicit ``permissionDecision: "allow"`` instead of the
        neutral ``{}``. Used by :func:`janus_options` for high-risk sinks kept
        off ``allowed_tools``: under ``permission_mode="dontAsk"`` such a tool
        runs only when this hook affirmatively approves it — if the hook is
        skipped (upstream hook regressions), the permission layer denies it.

    Returns
    -------
    An async hook callback suitable for a ``HookMatcher``. On a policy block it
    returns a ``PreToolUse`` ``permissionDecision: "deny"`` whose reason is fed
    back to the model, so the agent can adjust rather than crash. An unexpected
    exception inside the hook body (enforcer bug, malformed ``tool_input``,
    taint bug) also returns a deny — Janus's own defects fail closed, never
    open.

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
    taint, session = _resolve_state(taint, session)
    enforcer = resolve_enforcer(policy)
    required = required_args or {}
    explicit_allow = frozenset(hook_approved_tools or ())
    logger = get_logger()

    async def hook(input_data: dict, tool_use_id: str | None, context: Any) -> dict:
        runtime_name = ""
        try:
            runtime_name = input_data.get("tool_name", "")
            arguments = dict(input_data.get("tool_input") or {})
            logger.tool_call(runtime_name, arguments)

            reason = _decide(
                enforcer, runtime_name, arguments,
                passthrough_tools=passthrough_tools,
                resolve_name=resolve_name,
                required_args=required,
                taint=taint,
                session=session,
            )
        except Exception as exc:  # fail closed on Janus's own defects
            reason = (
                f"internal enforcement error ({type(exc).__name__}: {exc}); "
                "failing closed"
            )
        if reason is None:
            logger.policy_decision(runtime_name, allowed=True)
            try:
                needs_explicit = resolve_name(runtime_name) in explicit_allow
            except Exception:  # neutral, never an unintended explicit allow
                needs_explicit = False
            if needs_explicit:
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "allow",
                        "permissionDecisionReason": "[Janus] allowed by policy",
                    }
                }
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


def janus_posttooluse_hook(
    taint: TaintTracker | Session,
    *,
    resolve_name: NameResolver = default_resolve_name,
    unwrap: Callable[[Any], Any] | None = None,
) -> Callable[[dict, str | None, Any], Awaitable[dict]]:
    """Build a ``PostToolUse`` hook that records tool outputs into session state.

    Fires after a tool has executed — its output is now in the model's context.
    The first argument is anything with ``record_output(tool, output)``: a bare
    :class:`TaintTracker` (taint only) or a :class:`janus.policy.Session`
    (taint **and** provenance). Pass the SAME instance to
    :func:`janus_pretooluse_hook` (or use :func:`janus_hooks`, which wires
    both seams).

    ``unwrap`` is applied to the raw ``tool_response`` before recording; when
    :func:`janus_hooks` wires a Session it passes :func:`unwrap_tool_response`
    so provenance extractors (and taint classifiers) see the dict the tool
    body returned rather than MCP content blocks. The taint-only path keeps
    the raw response for backwards compatibility.

    Denied calls never reach this hook's derivation in a harmful way: the SDK
    still reports blocked tools here, but recording their *attempt* as a read
    would be wrong — so only calls with a response are recorded.
    """
    logger = get_logger()

    async def hook(input_data: dict, tool_use_id: str | None, context: Any) -> dict:
        runtime_name = input_data.get("tool_name", "")
        policy_key = resolve_name(runtime_name)
        response = input_data.get("tool_response")
        if response is None:
            return {}
        if unwrap is not None:
            try:
                response = unwrap(response)
            except Exception as exc:  # record the raw shape rather than nothing
                logger.warning(
                    f"PROVENANCE unwrap failed for '{policy_key}' "
                    f"({type(exc).__name__}: {exc}); recording raw response"
                )
                response = input_data.get("tool_response")
        recorded = taint.record_output(policy_key, response)
        labels = recorded.get("taint", []) if isinstance(recorded, dict) else recorded
        if labels:
            logger.warning(
                f"TAINT session tainted by {labels} (read via '{policy_key}')"
            )
        return {}

    return hook


def janus_hooks(
    policy: PolicySource,
    *,
    required_args: RequiredArgs | None = None,
    passthrough_tools: frozenset[str] = DEFAULT_PASSTHROUGH_TOOLS,
    resolve_name: NameResolver = default_resolve_name,
    taint: TaintTracker | None = None,
    session: Session | None = None,
    hook_approved_tools: set[str] | frozenset[str] | None = None,
) -> dict:
    """Convenience wrapper: return a ready ``hooks=`` dict for ``ClaudeAgentOptions``.

    Equivalent to building ``{"PreToolUse": [HookMatcher(hooks=[hook])]}`` yourself
    from :func:`janus_pretooluse_hook`. Imports the SDK (for ``HookMatcher``); use
    :func:`janus_pretooluse_hook` directly if you want to avoid the import or set a
    matcher pattern.

    With ``session=`` a :class:`janus.policy.Session`, both seams are wired:
    a ``PostToolUse`` hook records every tool output into the session (taint
    labels *and* provenance value-sets, with MCP content blocks unwrapped via
    :func:`unwrap_tool_response`), and the ``PreToolUse`` hook gates sinks on
    session taint and exposes the session to context-aware policy conditions
    (``from_output`` / ``not_in`` / custom ``@context_condition``)::

        session = Session(taint=TaintTracker(
            sources={"fetch_page": "web"},
            gates={"send_email": "*"},          # Rule of Two: no send after any read
        ))
        session.provenance.collect(
            "web_search", label="searched_urls",
            extract=lambda out: [r["url"] for r in out.get("results") or []],
            normalize=normalize_url,
        )
        options = ClaudeAgentOptions(..., hooks=janus_hooks(POLICY, session=session))

    ``taint=`` (a bare tracker, taint gating only, raw responses) keeps
    working but is superseded by ``session=``; passing both raises. Use one
    Session/tracker per agent session; ``reset()`` only at session boundaries.
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
        taint=taint, session=session, hook_approved_tools=hook_approved_tools,
    )
    # The hooks are intentionally typed with broad dict signatures so this module
    # imports without the SDK; they are shape-correct for the SDK's HookCallback.
    hooks: dict = {"PreToolUse": [HookMatcher(hooks=[hook])]}  # type: ignore[list-item]
    if session is not None:
        post = janus_posttooluse_hook(
            session, resolve_name=resolve_name, unwrap=unwrap_tool_response,
        )
        hooks["PostToolUse"] = [HookMatcher(hooks=[post])]  # type: ignore[list-item]
    elif taint is not None:
        post = janus_posttooluse_hook(taint, resolve_name=resolve_name)
        hooks["PostToolUse"] = [HookMatcher(hooks=[post])]  # type: ignore[list-item]
    return hooks


# =============================================================================
# Recommended entry point — locked-down ClaudeAgentOptions builder
# =============================================================================

# Overrides that weaken the lockdown; forwarding them requires unsafe_overrides=True.
_GUARDED_PERMISSION_MODES = frozenset({"bypassPermissions", "acceptEdits"})


def _run_coro_sync(coro: Awaitable[Any]) -> Any:
    """Run a coroutine to completion from sync code, even under a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]
    # Called from inside an event loop (e.g. an async integration test): drive
    # the coroutine on a private loop in a worker thread.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()  # type: ignore[arg-type]


def _enumerate_sdk_tools(instance: Any) -> list[str]:
    """List the tool names an in-process (``create_sdk_mcp_server``) server exposes.

    The SDK registers a ``tools/list`` handler on the underlying ``mcp`` Server
    only when tools were passed, so a missing handler means an empty server.
    """
    import mcp.types as mcp_types

    handler = getattr(instance, "request_handlers", {}).get(mcp_types.ListToolsRequest)
    if handler is None:
        return []
    result = _run_coro_sync(handler(mcp_types.ListToolsRequest(method="tools/list")))
    return [t.name for t in result.root.tools]


def janus_options(
    policy: PolicySource,
    *,
    mcp_servers: dict[str, Any],
    required_args: RequiredArgs | None = None,
    taint: TaintTracker | None = None,
    session: Session | None = None,
    resolve_name: NameResolver = default_resolve_name,
    passthrough_tools: frozenset[str] = DEFAULT_PASSTHROUGH_TOOLS,
    hook_approved_tools: set[str] | frozenset[str] | None = None,
    extra_hooks: dict | None = None,
    unsafe_overrides: bool = False,
    **overrides: Any,
) -> ClaudeAgentOptions:
    """Build a locked-down ``ClaudeAgentOptions`` from a Janus policy + MCP servers.

    The hook seam alone leaves tool-level reachability hostage to the hook
    firing: a silently skipped ``PreToolUse`` hook executes ``Bash(<anything>)``.
    This builder layers CLI-enforced controls in front of the hook so a skipped
    hook's worst case shrinks to *a policy-listed Janus tool running with
    arguments the policy would have refused*:

    - ``tools=[]`` — built-ins (``Bash``/``Write``/…) don't exist in the session.
    - ``strict_mcp_config=True`` — only ``mcp_servers`` passed here exist; no
      ``~/.claude.json`` / ``.mcp.json`` / project-config leakage.
    - ``allowed_tools`` = policy ∩ mounted — tool names are enumerated from the
      in-process servers, prefixed ``mcp__<server>__<name>``, and kept only if
      their ``resolve_name``-bare name has a rule in the loaded policy. Mounted
      tools the policy doesn't know are unreachable, not silently allowed.
    - ``permission_mode="dontAsk"`` — anything not allow-listed is denied, not
      prompted.
    - ``disallowed_tools`` = built-ins + ``Task`` (see
      ``DEFAULT_DISALLOWED_TOOLS``) — defense in depth behind ``tools=[]``, and
      subagents stay off until PreToolUse coverage inside them is verified.
    - ``setting_sources`` untouched (SDK default: none) — filesystem settings
      cannot re-enable anything.
    - ``hooks=janus_hooks(...)`` — argument-level enforcement, same knobs.

    Parameters beyond the hook-seam ones:

    session : Session | None
        Per-run :class:`janus.policy.Session` — wires the ``PostToolUse``
        recording seam (taint + provenance, responses unwrapped) and exposes
        the session to context-aware conditions at ``PreToolUse``. Supersedes
        ``taint=`` (which keeps working, taint-only); passing both raises.
    hook_approved_tools : set[str] | None
        Bare policy keys of high-risk sinks (e.g. ``{"send_email"}``) to keep
        **off** ``allowed_tools`` even though mounted and policy-listed. The
        Janus hook then approves them explicitly on allow, so under ``dontAsk``
        the permission layer and the hook must *both* agree before a sink runs
        — a skipped hook means the sink is denied, not silently allowed.
    extra_hooks : dict | None
        Additional ``hooks=`` entries (same shape the SDK takes) merged
        *alongside* the Janus wiring — your matchers are appended after Janus's
        for each event. Use this instead of a ``hooks=`` override, which would
        replace the Janus PreToolUse wiring and therefore requires
        ``unsafe_overrides=True``.
    unsafe_overrides : bool
        Weakening the lockdown must be loud. Forwarding any of ``tools``,
        ``strict_mcp_config=False``, ``permission_mode`` in
        ``{"bypassPermissions", "acceptEdits"}``, ``can_use_tool``,
        ``setting_sources``, or ``hooks`` raises ``ValueError`` unless this is
        True.
    **overrides
        Forwarded to ``ClaudeAgentOptions`` (e.g. ``output_format=…``,
        ``max_turns=…``). ``allowed_tools`` / ``disallowed_tools`` overrides
        are merged, not replaced: user ``allowed_tools`` entries still pass the
        policy filter (needed for non-enumerable external servers), and user
        ``disallowed_tools`` entries are unioned — additions can only shrink
        the reachable surface, never grow it.

    Raises
    ------
    ValueError
        If the policy has no rules (the lockdown is meaningless without one);
        if a server config can't be enumerated (external stdio/SSE/HTTP
        servers) and no explicit ``allowed_tools`` entries cover it; or if a
        guarded override is passed without ``unsafe_overrides=True``.
    """
    try:
        from claude_agent_sdk import ClaudeAgentOptions
    except ImportError as exc:
        raise ImportError(
            "claude-agent-sdk is required for janus_options().\n"
            "Install with: pip install claude-agent-sdk"
        ) from exc

    enforcer = resolve_enforcer(policy)
    policy_dict = enforcer.policy or {}
    if not policy_dict:
        raise ValueError(
            "janus_options() requires a policy with at least one rule: the "
            "generated allowed_tools is the intersection of the policy and the "
            "mounted tools, so an empty policy makes every tool unreachable. "
            "Load a policy, or use janus_hooks() directly if you really want a "
            "hook-only integration."
        )

    # ---- guardrails: weakening the lockdown must be loud -----------------
    if not unsafe_overrides:
        problems = []
        if "tools" in overrides:
            problems.append("tools (built-ins would exist again)")
        if overrides.get("strict_mcp_config") is False:
            problems.append("strict_mcp_config=False (filesystem MCP config would load)")
        if overrides.get("permission_mode") in _GUARDED_PERMISSION_MODES:
            problems.append(
                f"permission_mode={overrides['permission_mode']!r} "
                "(auto-approves calls before any permission check)"
            )
        if "can_use_tool" in overrides:
            problems.append(
                "can_use_tool (documented-bypassable seam; it must not "
                "masquerade as an enforcement layer)"
            )
        if "setting_sources" in overrides:
            problems.append("setting_sources (re-introduces filesystem settings)")
        if "hooks" in overrides:
            problems.append(
                "hooks (would replace the Janus PreToolUse wiring; pass "
                "extra_hooks= to merge alongside it)"
            )
        if problems:
            raise ValueError(
                "janus_options(): override(s) weaken the lockdown: "
                + "; ".join(problems)
                + ". Pass unsafe_overrides=True if you really mean it."
            )

    user_allowed: list[str] = list(overrides.pop("allowed_tools", None) or [])
    user_disallowed: list[str] = list(overrides.pop("disallowed_tools", None) or [])

    # ---- allowed_tools = policy ∩ mounted, minus hook-approved sinks -----
    mounted: list[str] = []
    for server_name, server_cfg in mcp_servers.items():
        instance = server_cfg.get("instance") if isinstance(server_cfg, dict) else None
        if isinstance(server_cfg, dict) and server_cfg.get("type") == "sdk" and instance:
            names = _enumerate_sdk_tools(instance)
            mounted.extend(f"mcp__{server_name}__{n}" for n in names)
            continue
        # External (stdio/SSE/HTTP) server: tool names live in another process,
        # so they can't be enumerated here. Require an explicit allowed_tools
        # merge rather than silently allowing mcp__<server>__*.
        prefix = f"mcp__{server_name}__"
        if not any(t.startswith(prefix) for t in user_allowed):
            raise ValueError(
                f"janus_options(): cannot enumerate tools for MCP server "
                f"{server_name!r} (not an in-process create_sdk_mcp_server "
                "instance). Pass its tool names explicitly, e.g. "
                f"allowed_tools=['{prefix}<tool>', …] — they are still "
                "filtered against the policy."
            )

    approved = frozenset(hook_approved_tools or ())
    unknown_approved = approved - set(policy_dict)
    if unknown_approved:
        raise ValueError(
            "janus_options(): hook_approved_tools entries not in the policy "
            f"(they could never be approved): {sorted(unknown_approved)}"
        )

    allowed: list[str] = []
    for runtime_name in dict.fromkeys(mounted + user_allowed):  # ordered de-dup
        bare = resolve_name(runtime_name)
        if bare in policy_dict and bare not in approved:
            allowed.append(runtime_name)

    disallowed = list(dict.fromkeys([*DEFAULT_DISALLOWED_TOOLS, *user_disallowed]))

    # ---- hooks: Janus wiring first, user extras merged alongside ---------
    if unsafe_overrides and "hooks" in overrides:
        hooks = overrides.pop("hooks")
    else:
        hooks = janus_hooks(
            enforcer,
            required_args=required_args,
            passthrough_tools=passthrough_tools,
            resolve_name=resolve_name,
            taint=taint,
            session=session,
            hook_approved_tools=approved or None,
        )
        for event, matchers in (extra_hooks or {}).items():
            hooks.setdefault(event, []).extend(matchers)

    return ClaudeAgentOptions(
        tools=overrides.pop("tools", []),
        strict_mcp_config=overrides.pop("strict_mcp_config", True),
        mcp_servers=mcp_servers,
        allowed_tools=allowed,
        disallowed_tools=disallowed,
        permission_mode=overrides.pop("permission_mode", "dontAsk"),
        hooks=hooks,
        **overrides,
    )


# =============================================================================
# Alternative seam — can_use_tool callback
# =============================================================================


def make_can_use_tool(
    policy: PolicySource,
    *,
    required_args: RequiredArgs | None = None,
    passthrough_tools: frozenset[str] = DEFAULT_PASSTHROUGH_TOOLS,
    resolve_name: NameResolver = default_resolve_name,
    taint: TaintTracker | None = None,
    session: Session | None = None,
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

    taint, session = _resolve_state(taint, session)
    enforcer = resolve_enforcer(policy)
    required = required_args or {}
    logger = get_logger()

    async def can_use_tool(tool_name: str, arguments: dict, context: Any):
        try:
            args = dict(arguments or {})
            logger.tool_call(tool_name, args)
            reason = _decide(
                enforcer, tool_name, args,
                passthrough_tools=passthrough_tools,
                resolve_name=resolve_name,
                required_args=required,
                taint=taint,
                session=session,
            )
        except Exception as exc:  # fail closed on Janus's own defects
            reason = (
                f"internal enforcement error ({type(exc).__name__}: {exc}); "
                "failing closed"
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
    session: Session | None = None,
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
            enforcer.enforce(policy_key, arguments, session=session)
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
