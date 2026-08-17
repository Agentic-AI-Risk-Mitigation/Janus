"""
Janus × Claude Code CLI (interactive `claude`) hook adapter.

This is a *different target* from :mod:`janus.adapters.claude_agent_sdk`, and the
difference is the whole security story. On the SDK path, ``janus_options()``
constructs the agent's world — ``tools=[]``, ``strict_mcp_config=True``,
``allowed_tools`` = policy ∩ mounted — so a skipped ``PreToolUse`` hook cannot
escalate past a tool surface Janus itself defined. On the interactive CLI, Janus
does not construct the session: the human does. The built-in tools exist, the
user's MCP servers exist, and the only seams available are hooks and
``permissions.deny`` rules.

**State the consequence plainly, because the docs must carry it verbatim:** on
the CLI, Janus is a *policy monitor over a session it does not own*, backstopped
by operator-supplied ``permissions.deny`` rules. It is not a reachability
lockdown.

+-----------------------------+----------------------------+---------------------------------+
| Layer                       | SDK path                   | CLI path                        |
+=============================+============================+=================================+
| does the tool exist?        | ``tools=[]`` + strict mcp  | **gone** — the session is the   |
|                             |                            | user's                          |
+-----------------------------+----------------------------+---------------------------------+
| may it run unprompted?      | ``allowed_tools`` ∩ policy | ``permissions.deny`` (see       |
|                             | + ``dontAsk``              | :data:`DEFAULT_CLI_SINK_DENY`)  |
+-----------------------------+----------------------------+---------------------------------+
| may it run with these args? | Janus PreToolUse hook      | Janus PreToolUse hook (the CLI  |
|                             | (fails closed on timeout)  | fails **open** on hook timeout) |
+-----------------------------+----------------------------+---------------------------------+
| runs even if all above lied | ``guard_tool_body``        | **gone** — tool bodies are the  |
|                             |                            | CLI's                           |
+-----------------------------+----------------------------+---------------------------------+

Two modes, and the reason there are two
---------------------------------------

``mode="gate"`` (default): the policy names the tools Janus has an opinion about
(sources, sinks, argument-conditioned tools) and every other tool gets ``{}`` —
"no opinion", deferring to the CLI's normal permission flow and the human behind
it. This bends CLAUDE.md's default-deny invariant *only on this seam*, and only
because the seam has a downstream authority the SDK seam lacks. Default-deny
over an interactive session's surface (every built-in plus every mounted MCP
server) means either shipping a curated allowlist of Anthropic's built-ins that
one CLI release can invalidate, or a session that fights its user until the
plugin is uninstalled — both worse for security than a monitor that stays
installed.

``mode="policy"``: strict default-deny, exactly as the library and SDK paths
behave. Recommended for headless and managed deployments.

**Abstention is not certification.** ``{}`` means "Janus has no opinion; ask the
permission system". Where there *is* no permission system — ``permission_mode``
of ``bypassPermissions``, i.e. nothing downstream will ever ask a human —
abstention would degrade to a silent allow, and the injected agent's cheapest
move is a sink Janus has no opinion on. So gate mode **auto-promotes to policy
mode** under those permission modes (disable with
``strict_when_unsupervised=False``, and own the consequence).

Payload dialects are pinned, not assumed
----------------------------------------

Everything this module parses is pinned by verbatim payload captures in
``tests/fixtures/claude_code_payloads/`` (CLI 2.1.233). Where the fixtures
contradict the published docs, the fixtures win, and the normalizer is written
so that a future upstream rename degrades to "the other key still works" rather
than to silent zero-taint:

* ``PostToolUse`` sends ``tool_response``; the docs say ``tool_output``.
  :func:`normalize_cli_event` reads either.
* ``PostToolBatch`` has **no** ``tool_name`` — it carries a ``tool_calls``
  array, so :func:`normalize_cli_events` fans it out.
* The same call has *different* output shapes in ``PostToolUse`` and
  ``PostToolBatch`` (``Read`` is a dict in the former, a plain string in the
  latter), which is why :func:`unwrap_cli_response` handles three dialects.
* ``agent_id``/``agent_type`` appear only inside a subagent, but the subagent
  shares its parent's ``session_id`` — so one Session per ``session_id``
  covers both, with ``agent_id`` retained for audit.
* ``prompt_id`` identifies the model turn, which is what makes "these calls
  were issued before the model saw that output" checkable rather than assumed.

The CLI's decision vocabulary was pinned the same way, by emitting candidate
values from a real hook and observing whether the tool ran (CLI 2.1.233):
``deny`` and ``ask`` block; ``escalate`` runs the tool, indistinguishable from a
misspelled string. See :data:`ASK`.

Phase 1 scope
-------------

This module is the stateless decision core plus the payload contract: it is
importable on a core install (stdlib + existing core deps) and holds no state.
Cross-call taint requires a Session that outlives the hook process, which is the
daemon's job (phase 2); pass ``session=`` here and taint gating works within
whatever process owns that Session.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from janus.adapters._base import PolicySource, resolve_enforcer
from janus.adapters.claude_agent_sdk import unwrap_tool_response
from janus.logger import get_logger
from janus.policy.decision import (
    LAYER_PASSTHROUGH,
    LAYER_RULES,
    LAYER_TAINT,
    decide_call,
)
from janus.policy.enforcer import PolicyEnforcer, RequiredArgs
from janus.policy.session import Session
from janus.policy.taint import TaintTracker

__all__ = [
    "ABSTAIN",
    "ALLOW",
    "CliDecision",
    "CliHookEvent",
    "DEFAULT_CLI_PASSTHROUGH_TOOLS",
    "DEFAULT_CLI_SINK_DENY",
    "DENY",
    "ASK",
    "UNKNOWN_MCP_SERVER",
    "UNSUPERVISED_PERMISSION_MODES",
    "claude_code_resolve_name",
    "cli_name_resolver",
    "decide_cli_event",
    "evaluate_cli_event",
    "handle_cli_payload",
    "interesting_tools",
    "normalize_cli_event",
    "normalize_cli_events",
    "record_cli_event",
    "unwrap_cli_response",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOW = "allow"
DENY = "deny"
#: Escalate to the human — the CLI's ``permissionDecision`` for "I have an
#: opinion but a person should resolve it".
#:
#: **The spelling is load-bearing, and this was verified on the wire rather than
#: read off a doc.** An earlier draft used ``"escalate"``. On CLI 2.1.233 that
#: value behaves *identically to a garbage string*: the hook output is ignored
#: and the tool runs. A taint gate emitting it would have silently allowed every
#: single hit — the worst possible failure for the mechanism whose entire job is
#: to stop consequential actions after untrusted input. ``"ask"`` blocks the
#: call and surfaces the Janus reason to the model, in headless and
#: ``bypassPermissions`` sessions alike. Do not "modernize" this constant
#: without re-running that experiment.
ASK = "ask"
#: Janus expresses "no opinion" as an empty hook output; this is its name in
#: :class:`CliDecision`, never a value the CLI sees.
ABSTAIN = "abstain"

Mode = Literal["gate", "policy"]
GateAction = Literal["deny", "ask"]

#: CLI-internal tools that are transport, not agent capability, and must never
#: be policy-gated — blocking them breaks the session without denying anything
#: consequential. ``ToolSearch`` loads deferred tool *schemas* (observed on the
#: wire in ``posttoolbatch.top-level.json``); it executes nothing. Extend, don't
#: drop, if a future CLI adds more.
DEFAULT_CLI_PASSTHROUGH_TOOLS = frozenset({"ToolSearch"})

#: Permission modes under which nothing downstream will ask a human. Verified on
#: CLI 2.1.233: a hook ``deny`` and a hook ``ask`` are both still honored here —
#: hooks win over ``bypassPermissions`` — but an *abstention* is not a decision
#: at all, so it degrades to a silent allow. That is what gate mode suppresses;
#: see the module docstring.
UNSUPERVISED_PERMISSION_MODES = frozenset({"bypassPermissions"})

#: Policy key that an ``mcp__`` tool from an unrecognized server resolves to
#: when ``known_servers`` is supplied. Reserved: it contains characters no real
#: policy key should, so it can never accidentally match an allow rule, and in
#: ``mode="policy"`` it default-denies. This is the CLI's stand-in for the SDK's
#: ``strict_mcp_config``, which has no upstream equivalent here.
UNKNOWN_MCP_SERVER = "<janus:unknown-mcp-server>"

#: The ``permissions.deny`` backstop, as data. These rules are enforced by the
#: CLI itself with zero hooks running, which is the only mitigation that
#: survives an upstream hook-dispatch regression. Operators paste this into
#: settings (or managed settings); it is deliberately about *sinks*, since a
#: blocked sink is what bounds an injection's blast radius.
DEFAULT_CLI_SINK_DENY: dict[str, list[str]] = {
    "deny": [
        "Bash(curl:*)",
        "Bash(wget:*)",
        "Bash(ssh:*)",
        "Bash(scp:*)",
        "Bash(nc:*)",
        "Bash(git push:*)",
        "WebFetch",
    ]
}

# Strips ``mcp__<server>__`` (non-greedy, so single-underscore server names
# survive) — same grammar as the SDK adapter's resolver.
_MCP_PREFIX_RE = re.compile(r"^mcp__(?P<server>.+?)__(?P<tool>.+)$")
# Plugin-mounted MCP servers are namespaced ``plugin_<plugin>_<server>``.
_PLUGIN_SERVER_PREFIX = "plugin_"

NameResolver = Callable[[str], str]
OnDecision = Callable[["CliHookEvent", "CliDecision"], None]


# ---------------------------------------------------------------------------
# Payload contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CliHookEvent:
    """One normalized Claude Code hook event.

    Constructed only by :func:`normalize_cli_event` /
    :func:`normalize_cli_events`. ``raw`` keeps the untouched payload so audit
    records the bytes the CLI actually sent, not our reading of them.
    """

    event: str
    session_id: str | None = None
    tool_name: str | None = None
    tool_input: dict = field(default_factory=dict)
    tool_output: Any | None = None
    tool_use_id: str | None = None
    agent_id: str | None = None
    agent_type: str | None = None
    permission_mode: str | None = None
    #: Identifies the model turn. Calls sharing a ``prompt_id`` were emitted
    #: before the model saw any of their outputs — the ordering fact the
    #: concurrency story rests on.
    prompt_id: str | None = None
    cwd: str | None = None
    #: Failure text from a ``PostToolUseFailure``. That event *replaces*
    #: ``PostToolUse`` for a failed call and carries no ``tool_response`` at
    #: all, so this is the only thing the seam reports about what happened.
    error: str | None = None
    #: True when this event was fanned out of a ``PostToolBatch`` envelope.
    in_batch: bool = False
    raw: dict = field(default_factory=dict)

    @property
    def is_subagent(self) -> bool:
        """True when the call was made from inside a subagent."""
        return self.agent_id is not None

    @property
    def unsupervised(self) -> bool:
        """True when no permission prompt can reach a human this session."""
        return self.permission_mode in UNSUPERVISED_PERMISSION_MODES


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def normalize_cli_event(payload: Mapping[str, Any]) -> CliHookEvent:
    """Normalize one hook payload into a :class:`CliHookEvent`.

    Never raises on shape drift: unknown or missing keys become ``None``/``{}``.
    That is deliberate — a payload the CLI changed under us must surface as a
    *detected* anomaly (an event with no ``tool_name``, caught by the
    payload-pin tests and, from phase 2, the PostToolUse cross-check), not as a
    ``KeyError`` that the caller's blanket fail-closed turns into "Janus denies
    every tool call in your editor".

    ``tool_output`` is read from ``tool_response`` **or** ``tool_output``,
    whichever is present: CLI 2.1.233 sends the former, the docs describe the
    latter, and whichever way upstream settles the failure mode is "the other
    key still works" rather than silent zero-taint.

    A ``PostToolBatch`` envelope has no per-call fields; use
    :func:`normalize_cli_events` to fan it out.
    """
    output: Any = None
    for key in ("tool_response", "tool_output"):
        if key in payload:
            output = payload[key]
            break

    tool_input = payload.get("tool_input")
    return CliHookEvent(
        event=str(payload.get("hook_event_name") or ""),
        session_id=_str_or_none(payload.get("session_id")),
        tool_name=_str_or_none(payload.get("tool_name")),
        tool_input=dict(tool_input) if isinstance(tool_input, Mapping) else {},
        tool_output=output,
        tool_use_id=_str_or_none(payload.get("tool_use_id")),
        agent_id=_str_or_none(payload.get("agent_id")),
        agent_type=_str_or_none(payload.get("agent_type")),
        permission_mode=_str_or_none(payload.get("permission_mode")),
        prompt_id=_str_or_none(payload.get("prompt_id")),
        cwd=_str_or_none(payload.get("cwd")),
        error=_str_or_none(payload.get("error")),
        raw=dict(payload),
    )


def normalize_cli_events(payload: Mapping[str, Any]) -> list[CliHookEvent]:
    """Normalize a payload into one event per tool call.

    ``PostToolBatch`` carries a ``tool_calls`` array and no ``tool_name``, so it
    fans out to one :class:`CliHookEvent` per entry (each inheriting the
    envelope's session/agent/turn fields, each flagged ``in_batch``). Every
    other payload yields a single-element list. An envelope with an empty or
    malformed ``tool_calls`` yields ``[]`` — there is nothing to decide about.
    """
    calls = payload.get("tool_calls")
    if not isinstance(calls, list):
        return [normalize_cli_event(payload)]

    envelope = normalize_cli_event(payload)
    fanned: list[CliHookEvent] = []
    for call in calls:
        if not isinstance(call, Mapping):
            continue
        merged = {**payload, **call}
        merged.pop("tool_calls", None)
        event = normalize_cli_event(merged)
        # Audit keeps the envelope: the batch is what the CLI actually sent.
        fanned.append(replace(event, in_batch=True, raw=envelope.raw))
    return fanned


def unwrap_cli_response(response: Any) -> Any:
    """Best-effort unwrap of a CLI tool response to the value the tool returned.

    Three dialects are live on CLI 2.1.233, all pinned by fixtures:

    1. **MCP tools** return a raw JSON *string* (``'{"result": "..."}'``) — the
       SDK's unwrapper passes those through untouched, since it only knows
       content blocks.
    2. **Built-ins in ``PostToolUse``** return dicts (``Bash``:
       ``stdout``/``stderr``/…; ``Read``: ``type``/``file``).
    3. **The same built-ins in ``PostToolBatch``** return plain strings
       (``Read`` → the numbered file text), and ``ToolSearch`` returns a block
       list.

    Strings are JSON-parsed only when they *look* like JSON (leading ``{``/
    ``[``), so ``"hello-janus"`` stays a string and ``"123"`` does not silently
    become an int under a taint classifier. Anything unrecognized is returned
    unchanged — extractors must tolerate the raw shape anyway, and collecting
    nothing fails closed for allow-sets.
    """
    if isinstance(response, str):
        text = response.strip()
        if text[:1] in ("{", "["):
            try:
                return json.loads(text)
            except ValueError:
                return response
        return response
    return unwrap_tool_response(response)


# ---------------------------------------------------------------------------
# Tool-name resolution
# ---------------------------------------------------------------------------


def claude_code_resolve_name(
    tool_name: str,
    *,
    known_servers: Collection[str] | None = None,
) -> str:
    """Map a CLI runtime tool name to its policy key.

    Built-ins (``Bash``, ``Read``, …) pass through verbatim. MCP tools arrive as
    ``mcp__<server>__<tool>`` or, for plugin-mounted servers,
    ``mcp__plugin_<plugin>_<server>__<tool>``; both resolve to the bare
    ``<tool>``.

    With ``known_servers``, a name whose server segment is not recognized
    resolves to :data:`UNKNOWN_MCP_SERVER` instead of its bare tool name, so a
    server the operator never sanctioned cannot inherit an allow rule written
    for a same-named tool elsewhere. The CLI has no ``strict_mcp_config``, so
    this is the only place that leak can be closed. Plugin namespacing is
    matched by suffix (``plugin_<anything>_<server>``), since plugin and server
    names may both contain underscores and the grammar is genuinely ambiguous.
    """
    match = _MCP_PREFIX_RE.match(tool_name)
    if match is None:
        return tool_name

    server = match.group("server")
    bare = match.group("tool")
    if known_servers is None:
        return bare

    if server in known_servers:
        return bare
    if server.startswith(_PLUGIN_SERVER_PREFIX) and any(
        server.endswith(f"_{known}") for known in known_servers
    ):
        return bare
    return UNKNOWN_MCP_SERVER


def cli_name_resolver(known_servers: Collection[str] | None = None) -> NameResolver:
    """Bind ``known_servers`` into a one-argument resolver for ``decide_call``."""
    if known_servers is None:
        return claude_code_resolve_name
    sanctioned = frozenset(known_servers)

    def resolve(tool_name: str) -> str:
        return claude_code_resolve_name(tool_name, known_servers=sanctioned)

    return resolve


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CliDecision:
    """Structured outcome of evaluating one CLI tool call.

    The CLI has three expressible outcomes plus abstention, so this replaces the
    SDK adapter's ``(allowed: bool, reason)`` audit shape — a boolean cannot
    distinguish "denied" from "asked the human", and an audit trail that
    conflates them is useless for exactly the events worth reviewing.
    """

    decision: str
    policy_key: str
    mode: str
    reason: str | None = None
    layer: str | None = None
    #: Set when gate mode was promoted to policy mode because the session is
    #: unsupervised, or when an escalation was downgraded to a deny.
    override: str | None = None

    @property
    def blocked(self) -> bool:
        return self.decision in (DENY, ASK)

    def to_hook_output(self) -> dict:
        """Render the ``PreToolUse`` hook JSON the CLI expects.

        Abstention and allow are both the empty object: Janus only speaks when
        it has something to say, so an allow does not override a
        ``permissions.deny`` rule the operator wrote.
        """
        if self.decision in (ABSTAIN, ALLOW):
            return {}
        prefix = "requires approval" if self.decision == ASK else "blocked by policy"
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": self.decision,
                "permissionDecisionReason": f"[Janus] {prefix}: {self.reason}",
            }
        }


def interesting_tools(
    enforcer: PolicyEnforcer,
    *,
    taint: TaintTracker | None = None,
    required_args: RequiredArgs | None = None,
) -> frozenset[str]:
    """Policy keys Janus has any opinion about — rules, gates, sources, or
    required arguments.

    This is the manifest a degraded/offline shim needs: without it, "fail closed
    when the daemon is unreachable" means denying ``Read`` and ``TodoWrite`` in
    an interactive session, which is the uninstall pressure gate mode exists to
    avoid. With it, an unreachable daemon can deny exactly the tools Janus would
    have judged and abstain on the rest.

    Taint *sources* are included even though they never gate a call: the offline
    shim must know that missing their output loses taint.
    """
    names = set(enforcer.tool_names)
    if taint is not None:
        names |= set(taint.source_tools) | set(taint.gated_tools)
    names |= set(required_args or {})
    return frozenset(names)


def evaluate_cli_event(
    event: CliHookEvent,
    enforcer: PolicyEnforcer,
    *,
    session: Session | None = None,
    taint: TaintTracker | None = None,
    mode: Mode = "gate",
    on_gate: GateAction = "ask",
    gate_overrides: Mapping[str, str] | None = None,
    required_args: RequiredArgs | None = None,
    passthrough_tools: Collection[str] = DEFAULT_CLI_PASSTHROUGH_TOOLS,
    resolve_name: NameResolver = claude_code_resolve_name,
    headless: bool = False,
    strict_when_unsupervised: bool = True,
) -> CliDecision:
    """Evaluate one ``PreToolUse`` event and return the structured decision.

    Delegates the actual layering to :func:`janus.policy.decision.decide_call` —
    the same core the SDK hook and ``janus.testing`` use, so a green consumer
    test reflects deployed semantics here too. Everything this function adds is
    CLI-seam specific:

    * **Gate-mode abstention.** A deny produced by the *rules* layer for a tool
      the policy never mentions is default-deny, and on this seam default-deny
      is not what we want: it becomes ``abstain``. Note this correctly covers
      the subtle case of a taint-gated sink that is *not* in the policy — the
      gate itself still fires (that is an opinion), but an untainted session
      does not then trip over default-deny on the way out.
    * **Unsupervised promotion.** Under a ``permission_mode`` where nothing can
      ask a human, gate mode promotes to policy mode: abstaining would be a
      silent allow, not a deferral.
    * **Escalation.** A taint-gate hit means "consequential action after
      untrusted input" — precisely the case where a human's out-of-band
      approval is the right resolution, and the deny reason (carrying its
      ``(audit id …)`` suffix) is what makes that approval informed. Static
      policy denies stay denies: the operator already decided those calls are
      wrong, and prompting would just train click-through. The wire value is
      ``"ask"``; see :data:`ASK` for why that spelling is not cosmetic.
    * **Escalation downgrade.** ``ask`` was verified to block in both headless
      and ``bypassPermissions`` sessions, so this is defense in depth against
      upstream drift rather than a live necessity: when the session is declared
      headless or reports an unsupervised ``permission_mode``, the escalation
      becomes a plain deny, which needs no downstream authority at all.
    """
    policy_key = resolve_name(event.tool_name or "")
    effective_mode: str = mode
    override: str | None = None

    unsupervised = event.unsupervised
    if mode == "gate" and strict_when_unsupervised and unsupervised:
        effective_mode = "policy"
        override = f"promoted to policy mode: permission_mode={event.permission_mode!r}"

    decision = decide_call(
        enforcer,
        event.tool_name or "",
        dict(event.tool_input),
        passthrough_tools=tuple(passthrough_tools),
        resolve_name=resolve_name,
        required_args=dict(required_args or {}),
        taint=taint,
        session=session,
    )

    if decision.allowed:
        # Report the *reason* the call may proceed, not just that it may. An
        # empty or unloaded policy allows everything, and recording that as
        # "Janus approved this" would make the audit trail claim a judgement
        # that never happened. Only a tool something actually has an opinion
        # about — a rule, a gate, a required-arg entry — or an explicit
        # passthrough gets ALLOW; the rest is abstention wearing the same
        # bytes on the wire.
        if effective_mode == "gate" and decision.layer != LAYER_PASSTHROUGH:
            tracker = getattr(session, "taint", None) if session is not None else taint
            opinionated = (
                policy_key in enforcer.tool_names
                or policy_key in (required_args or {})
                or (tracker is not None and policy_key in tracker.gated_tools)
            )
            if not opinionated:
                return CliDecision(
                    ABSTAIN,
                    policy_key,
                    effective_mode,
                    reason="no policy opinion; deferred to the CLI permission flow",
                    layer=decision.layer,
                    override=override,
                )
        return CliDecision(ALLOW, policy_key, effective_mode, layer=decision.layer)

    if (
        effective_mode == "gate"
        and decision.layer == LAYER_RULES
        and policy_key not in enforcer.tool_names
    ):
        return CliDecision(
            ABSTAIN,
            policy_key,
            effective_mode,
            reason="no policy opinion; deferred to the CLI permission flow",
            layer=decision.layer,
            override=override,
        )

    action: str = DENY
    if decision.layer == LAYER_TAINT:
        action = (gate_overrides or {}).get(policy_key, on_gate)
        if action not in (DENY, ASK):
            action = DENY
        if action == ASK and (headless or unsupervised):
            action = DENY
            override = (
                "escalation downgraded to deny: no human can answer a "
                f"permission prompt (headless={headless}, "
                f"permission_mode={event.permission_mode!r})"
            )

    return CliDecision(
        action, policy_key, effective_mode, reason=decision.reason, layer=decision.layer,
        override=override,
    )


def decide_cli_event(
    event: CliHookEvent,
    enforcer: PolicyEnforcer,
    *,
    on_decision: OnDecision | None = None,
    **kwargs: Any,
) -> dict:
    """Evaluate a ``PreToolUse`` event and return ready-to-print hook JSON.

    Fails closed on Janus's own defects: any unexpected exception inside the
    decision path becomes a deny, never a pass-through. ``on_decision`` is
    strictly observational — it receives the :class:`CliDecision`, and its
    exceptions are logged and swallowed so an audit defect can never flip an
    enforcement outcome.

    Keyword arguments are forwarded to :func:`evaluate_cli_event`.
    """
    logger = get_logger()
    try:
        decision = evaluate_cli_event(event, enforcer, **kwargs)
    except Exception as exc:  # fail closed on Janus's own defects
        decision = CliDecision(
            DENY,
            event.tool_name or "",
            str(kwargs.get("mode", "gate")),
            reason=(
                f"internal enforcement error ({type(exc).__name__}: {exc}); failing closed"
            ),
        )

    if on_decision is not None:
        try:
            on_decision(event, decision)
        except Exception as exc:
            logger.warning(
                f"on_decision callback error for '{event.tool_name}' "
                f"({type(exc).__name__}: {exc}); ignoring"
            )

    # Audit: every blocked call, and every call whose outcome this seam
    # *changed*, has to be reconstructable from the events trail. The taint
    # tracker already records gate denials, but not that a gate hit became an
    # escalation rather than a deny, nor that an escalation was downgraded or
    # gate mode promoted — and those overrides are precisely the decisions a
    # reviewer needs to see, since they are where CLI-seam semantics diverge
    # from the policy as written. ``policy_deny`` keeps its SDK-adapter shape so
    # consumers parsing the trail do not need a second code path.
    session = kwargs.get("session")
    if session is not None and (decision.blocked or decision.override):
        rules_deny = decision.decision == DENY and decision.layer == LAYER_RULES
        try:
            if rules_deny:
                session.note(
                    kind="policy_deny", tool=decision.policy_key, reason=decision.reason
                )
            else:
                session.note(
                    kind="cli_decision",
                    tool=decision.policy_key,
                    decision=decision.decision,
                    layer=decision.layer,
                    mode=decision.mode,
                    reason=decision.reason,
                    override=decision.override,
                )
        except Exception as exc:
            logger.warning(
                f"session note failed for '{event.tool_name}' "
                f"({type(exc).__name__}: {exc}); ignoring"
            )

    logger.policy_decision(
        event.tool_name or "",
        allowed=not decision.blocked,
        reason=decision.reason or "",
    )
    return decision.to_hook_output()


# ---------------------------------------------------------------------------
# Post-execution seam
# ---------------------------------------------------------------------------


def record_cli_event(
    event: CliHookEvent,
    session: Session | TaintTracker,
    *,
    resolve_name: NameResolver = claude_code_resolve_name,
    unwrap: Callable[[Any], Any] | None = unwrap_cli_response,
) -> dict[str, list[str]] | list[str] | None:
    """Record one completed tool call into session state.

    Returns whatever ``record_output`` returned, or ``None`` when there was
    nothing to record. A call with no output is *not* recorded: the tool was
    denied or failed, so nothing entered the model's context and treating the
    attempt as a read would taint the session for an action that never
    happened.

    That covers failures for free, but the reason is worth stating because it
    is a judgement call rather than an accident. A failed call arrives as
    ``PostToolUseFailure`` — a *different event* that replaces ``PostToolUse``
    and carries an ``error`` string instead of a ``tool_response``. So a
    `WebFetch` that failed contributes an error message, not fetched content,
    and tainting the session on it would gate every downstream sink over a
    404. The tradeoff: an error string can carry some attacker-influenced text
    (a server-chosen message), so a deployment that treats *any* contact with a
    source as tainting should record failures explicitly rather than rely on
    this seam.

    Only ``PostToolUse`` should drive this. ``PostToolBatch`` reports the same
    calls a second time, in a *different* output dialect, so recording both
    would double-count events and hand content-aware classifiers different
    bytes for the same call; the batch event is for cross-checking, not
    derivation.
    """
    if event.tool_output is None:
        return None
    output = event.tool_output
    if unwrap is not None:
        try:
            output = unwrap(output)
        except Exception as exc:  # record the raw shape rather than nothing
            get_logger().warning(
                f"unwrap failed for '{event.tool_name}' "
                f"({type(exc).__name__}: {exc}); recording raw response"
            )
    return session.record_output(resolve_name(event.tool_name or ""), output)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def handle_cli_payload(
    payload: Mapping[str, Any],
    policy: PolicySource = None,
    *,
    session: Session | None = None,
    **kwargs: Any,
) -> dict:
    """Dispatch a raw hook payload by ``hook_event_name`` and return hook JSON.

    ``PreToolUse`` decides; ``PostToolUse`` records into ``session`` when one is
    supplied; everything else (lifecycle events, ``PostToolBatch``,
    ``PostToolUseFailure``) returns ``{}``. This is the entry point the
    ``janus-hook`` shim uses, so the shim stays a transport concern and the
    semantics live here where they are testable offline.
    """
    enforcer = resolve_enforcer(policy)
    event = normalize_cli_event(payload)
    if event.event == "PreToolUse":
        return decide_cli_event(event, enforcer, session=session, **kwargs)
    if event.event == "PostToolUse" and session is not None:
        resolve_name = kwargs.get("resolve_name", claude_code_resolve_name)
        try:
            record_cli_event(event, session, resolve_name=resolve_name)
        except Exception as exc:
            # A failed recording is a fail-OPEN in the taint mechanism, and a
            # silent one is the worst kind: the session simply stays untainted,
            # so every downstream sink this output should have gated is allowed
            # instead, with nothing in the trail to say why. There is no deny to
            # emit on this seam — the tool has already run — so the only honest
            # response is to make it loud in both the log and the session's own
            # events, where the audit will show a hole rather than a clean run.
            get_logger().error(
                f"TAINT NOT RECORDED for '{event.tool_name}' "
                f"({type(exc).__name__}: {exc}); session state is now incomplete "
                "and sinks that should be gated may be allowed"
            )
            try:
                session.note(
                    kind="record_failed",
                    tool=event.tool_name,
                    tool_use_id=event.tool_use_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
            except Exception:  # the trail is best-effort; the log already fired
                pass
    return {}
