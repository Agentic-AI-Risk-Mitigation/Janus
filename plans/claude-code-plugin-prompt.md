# Handoff prompt: design Janus's Claude Code CLI integration

Write a design doc at `plans/claude-code-plugin-design.md` for shipping Janus as a
**Claude Code CLI** guard — hooks configured through a settings file or a plugin. Do not
implement anything. The output is a design doc that a later agent implements from.

Janus (`janus-guard`) is at `~/projects/archive/aisc/Janus`. Read `CLAUDE.md` first, then
`janus/adapters/claude_agent_sdk.py`, `janus/policy/taint.py`, `janus/policy/session.py`,
`docs/adapters.md`, and `plans/claude-agent-sdk-hardening.md`. Note the distinction in
`CLAUDE.md` between PDE taint (manual scalar, SpiceDB) and `TaintTracker` (per-source
labels, automatic) — only the latter is in scope.

## The core distinction to hold onto

Janus already integrates with the **Claude Agent SDK** (`janus_options()`,
`janus_hooks()`), where the SDK runs in-process and Janus builds a locked-down
`ClaudeAgentOptions`. This is a *different* target: the **Claude Code CLI** that a human
drives interactively. There is no `ClaudeAgentOptions` to lock down, no `allowed_tools` to
shadow, no `disallowed_tools`, no `strict_mcp_config`. Every layer `janus_options()` adds
in front of the hook is unavailable. What exists instead is `permissions.deny` rules and
managed settings. The doc must be explicit about what security property is lost and what
replaces it.

## Verified facts — treat as given, don't re-research

From the Claude Code docs (`code.claude.com/docs/en/hooks`, `/plugins`,
`/plugins-reference`, `/plugin-marketplaces`, `/settings`) as of 2026-08-15:

**Hook seam**
- PreToolUse stdin JSON: `session_id`, `prompt_id`, `transcript_path`, `cwd`,
  `permission_mode`, `hook_event_name`, `tool_name`, `tool_input`, `tool_use_id`,
  `agent_id`, `agent_type`.
- PreToolUse output: `{"hookSpecificOutput": {"hookEventName": "PreToolUse",
  "permissionDecision": "allow"|"deny"|"escalate", "permissionDecisionReason": "..."}}`
  plus optional top-level `systemMessage`, `additionalContext`, `continue`. This is
  **byte-identical** to what `janus_pretooluse_hook()` already returns, except `escalate`.
- **PostToolUse input uses `tool_output`, not `tool_response`.** `janus_posttooluse_hook()`
  reads `tool_response` and returns `{}` when it is `None`, so an unadapted shim silently
  records zero taint forever. This is the single highest-risk detail in the whole design.
- Exit codes: 0 + JSON → decision honored; 0 without JSON → normal permission flow; 2 →
  blocks regardless of JSON. **On hook timeout the tool call proceeds to the normal
  permission flow** — i.e. the seam fails *open*. Default command-hook timeout is 600s.
- Hook handler types: `command`, **`http`** (POST event JSON to a URL), `mcp_tool`,
  `prompt`, `agent`. All matching hooks for an event run in parallel.
- Relevant events beyond Pre/PostToolUse: `PostToolUseFailure`, `PostToolBatch` (parallel
  batch resolved), `PermissionRequest`, `PermissionDenied`, `SessionStart`, `SessionEnd`,
  `SubagentStart`, `SubagentStop`, `PreCompact`, `UserPromptSubmit`.
- **Hooks from settings files are not snapshotted** — re-read from disk on each settings
  load. **Plugin hooks and managed-policy hooks are snapshotted per session.**

**Plugin packaging**
- Layout: only `plugin.json` inside `.claude-plugin/`; `hooks/hooks.json`, `skills/`,
  `agents/`, `bin/`, `.mcp.json` at plugin root. `hooks/hooks.json` takes the same `hooks`
  object as settings.json.
- `${CLAUDE_PLUGIN_ROOT}` (install dir) and `${CLAUDE_PLUGIN_DATA}`
  (`~/.claude/plugins/data/{id}/`, survives updates, removed on uninstall) are exported to
  hook processes. `bin/` is added to Bash's PATH while the plugin is enabled.
- `userConfig` declares typed values prompted at enable time (`string`/`number`/`boolean`/
  `file`/`directory`; `required`, `default`, `min`/`max`, `sensitive` → Keychain). Exposed
  as `CLAUDE_PLUGIN_OPTION_<KEY>` env vars and `${user_config.KEY}` substitution —
  **shell-form hook commands reject `${user_config.*}`; exec form with `args` is required.**
- Dependency auto-install covers Node lockfiles only (`--ignore-scripts`, 60s timeout).
  Python is explicitly a do-it-yourself case: PEP 723 `uv run --script`, or a
  `SessionStart` hook installing into `${CLAUDE_PLUGIN_DATA}`.
- Plugin MCP tools are named `mcp__plugin_<plugin>_<server>__<tool>`, which
  `default_resolve_name`'s current regex does not handle.
- Project-scope plugins load hooks/MCP/monitors only after the workspace-trust dialog.
- Plugin-shipped agents cannot declare `hooks`, `mcpServers`, or `permissionMode`.
- Sources: `github` (`ref`/`sha`), `archive` (HTTPS zip + verified `sha256`), `npm`,
  `command`, relative path. A declared `version` in `plugin.json` that isn't bumped means
  users keep the cached copy. `claude plugin validate --strict` is the CI check.

**Managed settings (the enforcement story)**
- `allowManagedHooksOnly: true` blocks user, project, and plugin hooks **except** plugins
  force-enabled via managed `enabledPlugins` (since v2.1.101). `--plugin-dir` cannot
  override a managed force-enabled/disabled plugin.
- Also available: `allowedHttpHookUrls` (allowlist for `http` hooks),
  `strictKnownMarketplaces`, `extraKnownMarketplaces`, `blockedMarketplaces`,
  `disableSideloadFlags`, `disableCommandPluginSources`,
  `allowManagedPermissionRulesOnly`, `disableAllHooks`.
- Managed settings live at `/etc/claude-code/managed-settings.json` +
  `managed-settings.d/*.json` on Linux, and are read-only to Claude Code.
- Open upstream issues to check before betting on this: anthropics/claude-code #33824
  (managed-settings hooks not loaded into the hook registry) and #46387
  (`allowManagedHooksOnly` docs vs. behavior for plugin hooks). **Verify both are still
  open and note the risk; do not assume they are fixed.**

**Landscape** — Anthropic's own security plugins (security-guidance, Claude Security beta)
are shift-left SAST on code and diffs, not runtime tool-call enforcement. Community
Claude Code guardrails are bash + jq + grep, stateless, and mostly warn rather than block.
Nothing ships runtime taint/IFC at the tool-call boundary. Frame Janus against
FIDES (information-flow labels), CaMeL (capability gating, 77% vs 84% utility), and Meta's
Rule of Two, not against the grep plugins.

## Decisions the doc must make and justify

1. **Delivery shape.** `command` hook (per-call Python), `http` hook against a warm
   `janusd`, or both with one as the documented default. My prior: HTTP daemon primary,
   `command` shim as the no-daemon fallback — the daemon is the only shape that keeps a
   live `Session`/`TaintTracker`, avoids cold-start latency against a fail-open timeout,
   and puts enforcement code outside the workspace the agent can write to. Argue it or
   overturn it, but decide.
2. **Session state.** Keyed on `session_id`; what `agent_id`/`agent_type` mean for
   subagent taint (does a subagent's taint propagate to the parent? both directions?);
   lifecycle via `SessionStart`/`SessionEnd`; concurrency under `PostToolBatch` and
   parallel hook execution; what survives a `command`-hook process boundary and how
   (`TaintTracker` has no serializer — `taint()` replay loses `events` first-cause audit).
3. **Fail-closed under a fail-open seam.** The hook times out → tool proceeds. Latency
   budget, watchdog behavior, and which `permissions.deny` rules the plugin must ship or
   instruct the operator to add as a backstop. State the residual risk plainly.
4. **Default-deny vs. Claude Code's built-ins.** A loaded policy denies unlisted tools, and
   the CLI has `TodoWrite`, `Glob`, `Grep`, `Task`, `Skill`, `NotebookEdit`, etc. Decide:
   ship a curated default policy over built-in tools, extend `passthrough_tools`, or invert
   to a gate-only mode where the policy names only sinks. Whatever you choose must not
   quietly break `CLAUDE.md`'s default-deny invariant — if it bends it, say so and scope it.
5. **`escalate`.** Should a taint gate deny or escalate to the human? This is Rule of Two's
   "human supervises consequential actions" — argue for the default and the API shape
   (`on_gate="deny"|"escalate"`, per-tool override?).
6. **Public API surface.** Proposal: `janus/adapters/claude_code.py` with payload
   normalization (`tool_output`, plugin MCP name resolution), an `escalate` path, and a
   session registry; `janusd` behind the existing `server` extra; a `janus-hook` console
   script (`[project.scripts]` — the repo has none today). Name things, list signatures,
   say what is reused from `claude_agent_sdk.py` vs. what is genuinely new. Do not
   duplicate `_decide`.
7. **Python bootstrap** for the plugin (PEP 723 uv script vs. `SessionStart` install into
   `${CLAUDE_PLUGIN_DATA}`), and what happens on a machine without `uv`.
8. **Distribution and trust.** Marketplace source type, version/tag discipline, whether to
   submit to `claude-plugins-community`, and the managed-settings deployment block an
   enterprise operator pastes in. A *security* plugin distributed through an ecosystem whose
   provenance story is immature needs its own trust posture — address it.
9. **Testing.** These are new offline tests in `tests/` plus, critically, a pinned-payload
   test: we got burned assuming the CLI payload matched the SDK's. Decide what belongs in
   `tests/smoke/` behind `JANUS_LIVE_SMOKE=1` against a real `claude` CLI, and what the
   smoke suite must assert to catch a payload-shape regression.

## Non-goals

PDE/SpiceDB. Changing `janus_options()` or SDK-path semantics. LangChain/ADK adapters.
Anything requiring a Claude Code feature that does not exist today.

## Doc requirements

Follow the house style in `plans/ipi-expansion-design.md` and
`plans/claude-agent-sdk-hardening.md`: dense, decision-first, no filler. Include a threat
model section stating explicitly what an attacker who lands an indirect prompt injection
can and cannot do under each delivery shape — including the case where the agent itself has
`Write` access to `~/.claude/settings.json` and to the repo. Include a phased
implementation plan with a first phase small enough to land in one commit. Flag every place
you are uncertain or where a doc claim needs live verification against the installed
`claude` CLI rather than asserting it.
