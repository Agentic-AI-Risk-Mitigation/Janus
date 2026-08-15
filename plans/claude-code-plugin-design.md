# Janus × Claude Code CLI — hook/plugin integration design

Status: proposal, 2026-08-15. Responds to `plans/claude-code-plugin-prompt.md`. Design
only — no code here is final API, but every signature is concrete enough to critique.
CLI facts below are from the Claude Code docs as of 2026-08-15 plus the verified-facts
block in the handoff prompt; every claim that still needs live verification against an
installed `claude` CLI is marked **[verify-live]**.

Two corrections to the prompt's inputs, checked 2026-08-15: anthropics/claude-code
**#33824 is closed as stale/not-planned** (never confirmed fixed) and **#46387 is closed
as completed** (the `allowManagedHooksOnly` docs were corrected) — consequences in §7.
And a live capture against CLI **2.1.233** (fixtures in
`tests/fixtures/claude_code_payloads/`, findings in its README) shows **`PostToolUse`
sends `tool_response`, not `tool_output`** — the docs' claimed rename is not (or not
yet) real on the installed CLI. The design's answer is unchanged and now
evidence-backed: the normalizer reads both keys, so whichever way upstream settles, the
failure mode is "the other key still works", never silent zero-taint.

---

## 1. What this target is, and what is lost

The SDK adapter's whole security story is `janus_options()`: Janus *constructs* the
agent's world (`tools=[]`, `strict_mcp_config=True`, `allowed_tools` = policy ∩ mounted,
`dontAsk`, hooks) so that a skipped `PreToolUse` hook cannot escalate past the project's
own tool surface. On the interactive **Claude Code CLI** none of that exists. Janus does
not construct the session; the human does. The CLI's built-in tools exist, the user's MCP
servers exist, and the only seams Janus gets are:

- **hooks** (`PreToolUse` / `PostToolUse` / lifecycle events), configured via settings
  files or a plugin — argument-level, but dependent on the hook firing, and the CLI's
  hook dispatch **fails open on timeout** (tool proceeds to the normal permission flow);
- **`permissions.deny` rules** and **managed settings** — whole-tool/pattern granularity,
  CLI-enforced, nothing for a hook to miss.

So the layered table from `plans/claude-agent-sdk-hardening.md` degrades to:

| Layer | SDK path | CLI path |
|---|---|---|
| does the tool exist? | `tools=[]` + `strict_mcp_config` | **gone** — session is the user's |
| may it run unprompted? | `allowed_tools` ∩ policy + `dontAsk` | `permissions.deny` (+ managed `allowManagedPermissionRulesOnly`) |
| may it run with these args? | Janus PreToolUse hook (fails closed on timeout, verified) | Janus PreToolUse hook (**fails open on timeout**, per docs) |
| runs even if all above lied | `guard_tool_body` | **gone** — tool bodies are the CLI's |

The honest statement, which the docs for this adapter must carry verbatim: **on the CLI,
Janus is a policy monitor over a session it does not own, backstopped by `permissions.deny`;
it is not a reachability lockdown.** The replacement for `janus_options()`'s "skipped hook
⇒ bounded blast radius" property is (a) operator-supplied `permissions.deny` backstop
rules for the worst sinks, and (b) a PostToolUse cross-check that converts a silently
skipped PreToolUse into a detected incident (§5.3). Tamper-resistance against the agent
itself exists only under managed settings (§7, §10).

## 2. Decision 1 — delivery shape: daemon brain, command-shim transport

**Decision: a warm daemon (`janusd`) is the only place enforcement state lives. The
default transport to it is a stdlib-only `command` hook shim (`janus-hook`) that fails
closed when the daemon is unreachable. Direct `http` hooks to `janusd` are the supported
enterprise variant, not the default.**

The prompt's prior was "HTTP primary, command shim fallback". Half-overturned, on one
argument: **the `command` shim is the only shape that can convert daemon-down into a
deny.** An `http` hook whose endpoint is unreachable errors, and a hook error that isn't
exit-code-2 proceeds to the normal permission flow — fail open **[verify-live: exact CLI
behavior on http-hook connection refused]**. A `command` shim owns its exit code: daemon
unreachable → print a deny decision (or exit 2) → fail closed. The daemon remains
primary in every sense that matters — it is where the `Session`/`TaintTracker`/policy
live — but the hop through the shim buys fail-closed at ~a process fork.

Why a daemon at all (unchanged from the prior):

- **State.** `TaintTracker` has no serializer and taint is cross-call by nature. A
  per-call Python process would need snapshot/restore through a file with locking under
  the CLI's *parallel* hook execution; the daemon holds live `Session` objects instead.
- **Latency against a fail-open timeout.** Per-call `python -c "import janus"` is
  ~150–400 ms cold (interpreter + jsonschema + pydantic); a `uv run --script` resolve on
  first call is seconds. The shim as designed imports **stdlib only** (socket + json), so
  the full path is fork + unix-socket round trip — target budget **p95 < 50 ms**, decision
  itself < 5 ms in the daemon. The default 600 s hook timeout is then never the binding
  constraint; the shim carries its own internal deadline (§5.2).
- **Tamper surface.** Enforcement code and policy live outside the workspace the agent
  edits, in `${CLAUDE_PLUGIN_DATA}` (or a system service for enterprise), not in
  repo-relative paths a `Write` call can reach through the project.

A pure per-call mode with no daemon exists only as **phase-1 scaffolding and a degraded
mode**: stateless policy evaluation (no taint, no provenance, no cross-check), loudly
documented as such. It is not the recommended deployment.

Shim ↔ daemon transport: **unix domain socket** at `${CLAUDE_PLUGIN_DATA}/janusd.sock`
(0600; no port squatting, no accidental network exposure). `janusd` can additionally
bind localhost TCP for `http`-hook deployments; that listener is what
`allowedHttpHookUrls` allowlists.

## 3. Decision 4 first, because everything hangs on it — gate mode vs. default-deny

**Decision: the CLI adapter runs in an explicit, named `mode="gate"` by default: the
policy names the tools Janus has an opinion about (sources, sinks, argument-conditioned
tools), and every other tool gets `{}` — "no opinion", deferring to the CLI's normal
permission flow. Strict default-deny remains available as `mode="policy"` and is the
recommended setting for headless/managed deployments.**

Rationale, stated against the invariant it bends:

- CLAUDE.md's default-deny invariant ("a loaded policy denies unlisted tools") is the
  right semantic when Janus defines the tool surface — the library path and the SDK path.
  On an interactive CLI the tool surface is `TodoWrite`, `Glob`, `Grep`, `Task`, `Skill`,
  `Read`, plus every MCP server the user mounted. Default-deny over that surface means
  either a shipped curated allowlist of Anthropic's built-ins — a list *we* then maintain
  against CLI releases, where one missed new tool bricks the session — or a session that
  fights its user until the plugin is uninstalled. Both outcomes are worse for security
  than a monitor that stays installed.
- Critically, `{}` on the CLI is **not** "silently allowed". It is "no opinion, fall
  through to the permission system and the human" — the CLI seam has a downstream
  authority the SDK seam lacks. Gate mode delegates the long tail to that authority
  instead of impersonating it.
- **Scoping the bend explicitly:** default-deny is not weakened anywhere existing code
  runs. `decide_call` and the SDK adapter are untouched. The CLI adapter introduces
  `mode` as a *required-to-be-explicit-in-config* knob (the plugin's `userConfig`
  defaults it to `gate` and surfaces it at enable time), and `mode="gate"` is documented
  as: *Janus enforces its opinions and abstains elsewhere; it does not certify abstained
  calls.* The enforcement-review checklist gains a line item: any change to gate-mode
  abstention semantics is a default-deny-adjacent change.

Mechanically, gate mode is the existing `decide_call` with one wrapper rule: if
`resolve_name(tool)` has no policy rule, no taint gate, and no required-args entry →
abstain (`{}`) instead of deny. Taint gates fire in gate mode exactly as in policy mode —
a gated sink is denied/escalated even though its neighbors abstain. `mode="policy"`
additionally needs a `passthrough_tools` extension for CLI-internal tool names
(the CLI analog of `StructuredOutput`; enumerate from a live session, **[verify-live]**).

## 4. Decision 2 — session state

**Keying.** One `Session` per `session_id`, held in a daemon-side `SessionRegistry`.

**Subagents.** `agent_id`/`agent_type` arrive in every hook payload. **Decision: subagent
tool calls share the parent `session_id`'s Session — taint propagates both directions.**
Justification: a subagent's output returns into the parent's context (so child taint must
flow up), and a subagent is spawned from a possibly-tainted parent context (so parent
taint must flow down — the injection can instruct the parent to launder an action through
a `Task`). Per-source labels make bidirectional sharing cheap: `record_output` events
carry `agent_id` in their cause dict for audit, so the merged trail still answers *which
agent* introduced each label. A future refinement (per-agent label namespaces with
endorsed declassification at `SubagentStop`) is explicitly out of scope — conservative
first.

**Lifecycle.** `SessionStart` → `registry.get_or_create(session_id)` (also ensures the
daemon is up, §8). `SessionEnd` → `registry.end(session_id)` after flushing the audit
trail to `${CLAUDE_PLUGIN_DATA}/audit/<session_id>.jsonl`. `SessionEnd` is not guaranteed
(crash, kill −9), so the registry also runs TTL eviction (default 24 h idle,
flush-on-evict). Missing `SessionStart` (hook added mid-session) → `get_or_create` at
first `PreToolUse`; never fail a decision because lifecycle events were missed.

**Concurrency.** The CLI runs all matching hooks for an event in parallel, and parallel
tool batches mean interleaved Pre/Post events across calls. `TaintTracker` and
`Session._notes` are already lock-guarded; the registry adds one lock around
create/evict. The real issue is **ordering**: a `PreToolUse` for call B can be decided
before the `PostToolUse` of concurrent call A is recorded, so B is judged against
slightly stale taint. Because taint is monotonic and calls in one parallel batch were
issued from the *same* model turn (the model had not yet seen A's output when it emitted
B), this is not a laundering channel for outputs-influence-arguments — but it is a real
race for "no send after any read" gates across a batch boundary. Mitigations, in order:
(a) document it; (b) subscribe to `PostToolBatch` and re-check gated sinks at batch
resolution, downgrading to a logged incident (can't un-run the tool); (c) optional
strict mode: a `PreToolUse` for a gated sink while any source-listed call is in flight
(Pre seen, Post not yet) → deny/escalate. Ship (a)+(b) in phase 2, (c) as a knob.

**Process-boundary survival.** The shim carries no state — every event is forwarded to
the daemon, so nothing must survive a shim process. What must survive a *daemon* restart
is session taint (restart ⇒ empty registry ⇒ gates silently lifted — fail-open by
amnesia). **Decision: add `snapshot()/restore()` to `TaintTracker` and `Session`**
(full-fidelity: `_tainted` causes, `_events`, `_seq`, provenance sets, endorsements —
not a lossy `taint()` replay, which drops first-cause audit). The daemon writes a
snapshot per session on mutation (atomic rename) and restores on start. This serializer
is core-library work with its own offline tests, and it is what makes the daemon
restartable during live sessions.

## 5. Decision 3 — fail-closed posture under a fail-open seam

Three distinct failure classes, three answers:

**5.1 Daemon down / unreachable.** The shim's job. Connect timeout 250 ms, one retry,
then emit `permissionDecision: "deny"` with reason "Janus daemon unreachable — failing
closed; run `janus-hook doctor`". `PostToolUse` events on daemon-down are spooled to
`${CLAUDE_PLUGIN_DATA}/spool/` and replayed by the daemon on reconnect, so taint is not
lost while denies are happening. Configurable to `escalate` instead of `deny` for
interactive comfort; never configurable to allow.

**5.2 Slow decision vs. CLI hook timeout.** The CLI kills the hook at its timeout and
proceeds — fail open. Therefore the *shim* enforces an internal deadline (default 5 s,
≪ the hook timeout) and emits a deny on expiry; the CLI-level timeout becomes
unreachable in practice. We also set an explicit generous `timeout` on our hook entries
rather than inheriting 600 s, purely to bound pathological cases. Decision latency
itself is not a risk (§2 budget); this machinery exists for the daemon-wedged case.

**5.3 Hook never fires (upstream dispatch regression — the #6305/#10814 class).**
Nothing hook-side can prevent this; two compensations:
- **Detection:** the daemon asserts every `tool_use_id` seen at `PostToolUse` was
  decided at `PreToolUse` (the SDK plan's follow-up 2, but implemented here first since
  the daemon makes it trivial). On a miss: error-level audit event, `systemMessage` to
  the user on the next decision, and optional deny-all-for-session.
- **Backstop `permissions.deny`:** the plugin cannot install permission rules
  (plugins ship hooks/skills/agents/MCP — not permissions **[verify-live: confirm no
  permissions surface in plugin.json]**), so the docs ship a copy-paste block the
  operator adds to settings (or the enterprise pastes into managed settings, §10):
  deny rules for the deployment's true sinks — e.g. `Bash(curl:*)`, `Bash(ssh:*)`,
  `WebFetch`, the outbound MCP tools. These hold with zero hooks running.

**Residual risk, stated plainly:** between a dispatch regression and its detection at
the next `PostToolUse`, calls not covered by `permissions.deny` run unenforced. On the
CLI seam this window cannot be closed, only shrunk and alarmed. Deployments that cannot
accept it should use the SDK path, which is why `janus_options()` remains the flagship.

## 6. Decision 5 — `escalate`

**Decision: taint-gate hits default to `escalate`; static policy denies stay `deny`.
API: `on_gate="escalate" | "deny"` with a per-tool override map, and an automatic
downgrade escalate→deny when the session cannot ask a human.**

Reasoning: a taint gate firing means "consequential action after untrusted input" —
exactly Rule of Two's *human supervises*; the human approving at the CLI prompt **is**
the out-of-band approval the gate's deny message asks for, with the
`permissionDecisionReason` (including the `(audit id …)` suffix from
`Session.gate_check`) rendered in the prompt so the approval is informed. Static policy
denies are different in kind — the operator already decided those calls are wrong; a
prompt would just train click-through. Downgrade rule: payload `permission_mode` of
`bypassPermissions`, or headless invocation, means no meaningful prompt — emit `deny`
**[verify-live: what `escalate` does in `claude -p` and under each permission_mode]**.
An escalate that the human approves is recorded via the existing endorsement path
(`session.endorse_event(audit_id, by="cli-user", …)`) driven from the
`PermissionRequest`/`PermissionDenied` events **[verify-live: payload contents suffice
to correlate]** — the audit trail then shows deny → human endorsement → allow.

## 7. Decision 8 groundwork — the tamper story (threat model)

Attacker model: an indirect prompt injection has landed in the model's context (via a
fetched page, a poisoned README, a tool output). The *agent* is now adversarial; the
*human and host* are not. Same-user malicious code execution is out of scope — if the
attacker already runs arbitrary Bash outside policy, no hook seam helps; gating Bash is
the policy's job, not the threat model's.

What the injected agent can and cannot do, per deployment shape:

| Capability of injected agent | settings.json hooks | plugin (user-installed) | plugin + managed settings |
|---|---|---|---|
| Call a policy-denied / taint-gated tool this session | no (hook denies) | no | no |
| Disarm the guard **this session** by editing settings | **yes** — settings hooks are re-read from disk, not snapshotted; one `Edit` of `~/.claude/settings.json` removes the hook mid-session | no — plugin hooks are snapshotted per session | no |
| Disarm for **future** sessions (edit settings, disable plugin, `--plugin-dir` shadow) | yes | yes | no — `allowManagedHooksOnly` + managed force-enabled `enabledPlugins`; `--plugin-dir` cannot override |
| Tamper with daemon state (reset taint via socket) | possible if it can run code touching the socket — which is itself a policy-gated action (`Bash`); mutation endpoints additionally require the per-boot token in the 0600 socket dir | same | same, and enterprise runs `janusd` as a separate service user, closing the same-user hole |
| Exfiltrate via a tool Janus has no opinion on (gate mode) | falls through to the CLI permission flow / human | same | same, plus managed `permissions.deny` backstop |

Two hard conclusions the docs must state:
1. **Settings-file hook delivery is not a security boundary against the agent it
   guards** (row 2). It is fine for evaluation and self-protection against accidents;
   the plugin is the minimum for "the session I started stays guarded"; managed settings
   are the minimum for "the machine stays guarded".
2. The managed story must use the **force-enabled-plugin** path, not hooks declared
   directly in managed settings: #33824 ("managed/policy settings hooks not loaded into
   the hook registry") was closed *stale, not fixed* — so hooks defined inline in
   managed settings may silently not run, which for a guard is the worst failure mode
   available. The plugin-exception path is the one #46387's completed docs fix
   describes. **[verify-live: both paths, on the pinned CLI version, before publishing
   enterprise guidance — a managed-settings deployment that silently loads no hooks
   must be caught by our own smoke test, not a customer.]**

## 8. Decision 6 + 7 — public API surface and bootstrap

New module **`janus/adapters/claude_code.py`** — core-install only, stdlib + existing
core deps, importable without any extra:

```python
@dataclass(frozen=True)
class CliHookEvent:
    event: str                      # hook_event_name
    session_id: str
    tool_name: str
    tool_input: dict
    tool_output: Any | None         # PostToolUse only
    tool_use_id: str | None
    agent_id: str | None
    agent_type: str | None
    permission_mode: str | None
    cwd: str | None
    raw: dict                       # untouched payload, for audit

def normalize_cli_event(payload: dict) -> CliHookEvent
    # THE load-bearing function. Reads `tool_response` OR `tool_output`,
    # whichever is present (live CLI 2.1.233 sends `tool_response`; the docs
    # say `tool_output` — see the fixtures README), so one normalizer serves
    # both dialects and a rename degrades to the other key, not to silence.
    # Unknown/missing keys -> None, never KeyError (the decision path fails
    # closed on exceptions, but a payload-shape drift must surface in the
    # cross-check and payload-pin tests, not as a blanket deny of everything).

def claude_code_resolve_name(name: str, *, known_servers: Collection[str] | None = None) -> str
    # Handles both `mcp__<server>__<tool>` and `mcp__plugin_<plugin>_<server>__<tool>`;
    # built-in names (Bash, Read, ...) pass through verbatim. With known_servers,
    # an mcp__ name whose server segment is unknown resolves to a reserved
    # never-allowed sentinel (the SDK plan's follow-up 4, defaulted on here since
    # there is no strict_mcp_config upstream to close the leak at the source).

DEFAULT_CLI_SINK_DENY: dict  # the documented permissions.deny backstop block (§5.3), as data

def decide_cli_event(
    event: CliHookEvent,
    enforcer: PolicyEnforcer,
    *,
    session: Session | None = None,
    mode: Literal["gate", "policy"] = "gate",
    on_gate: Literal["escalate", "deny"] = "escalate",
    gate_overrides: dict[str, str] | None = None,     # {tool: "deny"|"escalate"}
    required_args: RequiredArgs | None = None,
    passthrough_tools: frozenset[str] = ...,
    resolve_name: NameResolver = claude_code_resolve_name,
    on_decision: OnDecision | None = None,            # same shape as the SDK adapter's
) -> dict   # ready-to-print CLI hook JSON, or {} for abstain / PostToolUse record
```

`decide_cli_event` delegates to **`decide_call` — `_decide` is not duplicated**; the new
logic is only: gate-mode abstention (§3), `Decision.layer == LAYER_TAINT` →
escalate-vs-deny mapping (§6), and `hookSpecificOutput` serialization (identical bytes
to `janus_pretooluse_hook`'s deny, plus the `escalate` variant). `PostToolUse` events
route to `session.record_output(policy_key, tool_output)`. Output shapes are now
pinned (fixtures, CLI 2.1.233): built-ins return dicts (`Bash`:
`stdout`/`stderr`/…; `Read`: `type`/`file`), but an MCP tool's `tool_response` is a
**raw JSON string** — which the SDK's `unwrap_tool_response` passes through unparsed
(it only unwraps content blocks). The CLI adapter therefore gets its own
`unwrap_cli_response`: try `json.loads` on a bare string, delegate block shapes to the
SDK unwrapper, else return unchanged — with the fixture files as its test inputs.

**`janus/registry.py`** (name bikesheddable): `SessionRegistry` —
`get_or_create(session_id) -> Session`, `end(session_id)`, TTL sweep, snapshot/restore
wiring (§4). Framework-agnostic on purpose: the LangChain/ADK adapters' missing
post-execution seam work can reuse it later.

**`janusd`** — `janus/hookd.py` behind the existing `server` extra (FastAPI+uvicorn are
already there). Endpoints: `POST /hook` (dispatch on `hook_event_name`), `GET /healthz`,
`POST /admin/...` (endorse, snapshot, reload-policy) — admin routes require the per-boot
token; the hook route deliberately does not (it is decision-only and must never be the
thing that breaks). Config from a TOML/JSON file naming: policy path, mode, taint
sources/gates, on_gate, audit dir.

**Console scripts** (`[project.scripts]`, first in the repo):
- `janus-hook` — the shim. Subcommands: `pre`, `post`, `session-start`, `session-end`
  (wired in `hooks.json`), `doctor` (connectivity, versions, payload self-test). Import
  posture is **per mode, deliberately different**: in phase 2's proxy mode (the
  recommended deployment) the hot path is stdlib only — socket + json, zero janus
  imports, so the ~50 ms budget holds under any Python ≥ 3.10. In phase 1's stateless
  mode there is no daemon, so the shim *is* the enforcement and imports janus per call
  (~150–400 ms cold — acceptable for the degraded mode, and part of why it is degraded).
  Do not contort the phase-1 shim to avoid the import; do not let a janus import creep
  into the proxy hot path.
  Config plumbing, fixed now so phase 3 doesn't have to break it: explicit argv flags on
  the hook command — `janus-hook pre --policy <path> --mode gate --on-gate escalate`
  (plus `--socket <path>` in proxy mode). Phase 3's `userConfig` values slot into the
  same flags via exec-form `args`; no env vars, no fixed-path config file on the shim
  side (the daemon keeps its own config file, §8 above).
- `janusd` — run the daemon.

**Bootstrap (decision 7).** The shim being stdlib-only splits the problem: hooks need
only *a* Python; the daemon needs janus-guard installed once. **Decision: the plugin's
`SessionStart` hook bootstraps `${CLAUDE_PLUGIN_DATA}/venv` (via `uv venv` when uv
exists, else `python3 -m venv` + pip) pinned to the plugin's own version, starts
`janusd` from it if `/healthz` fails, then execs the shim.** PEP 723 `uv run --script`
was rejected as the primary: it makes every cold start depend on uv *and* the network,
and the machine-without-uv answer would be "no enforcement". No `python3` at all →
the hook entry is a `sh` wrapper that exits 2 with an instructive message — **inert
means fail closed and loud, never silently open** (a plain missing interpreter would
exit non-zero-non-2, which the CLI treats as non-blocking). `userConfig`: `policy_file`
(type `file`, required), `mode`, `on_gate` — passed exec-form with `args` (shell-form
rejects `${user_config.*}`).

## 9. Decision 8 — distribution and trust

A security plugin that can be spoofed or silently downgraded is negative-value, so:

- **Canonical source: the Janus GitHub repo itself as a marketplace** (`.claude-plugin/`
  in-repo, installed via `github` source pinned to a release **tag, with `sha` in the
  enterprise block**). `archive` (HTTPS zip + `sha256`) documented for air-gapped/vendored
  installs. No `npm`, no `command` source.
- **Version discipline:** `plugin.json` `version` bumps in the same commit as any
  behavior change (unbumped = users keep the cached copy — for a guard, that is a
  stale-policy-engine bug); release skill gains this check; `claude plugin validate
  --strict` in CI.
- **Community marketplace (`claude-plugins-community`): submit, but only after the
  smoke suite (§11) is green in CI**, and the README states that the provenance-critical
  install path is the pinned-sha one — a marketplace listing is discovery, not a trust
  anchor, in an ecosystem whose provenance story is immature.
- **Enterprise block** (documented verbatim in `docs/`): managed
  `enabledPlugins: {"janus@<marketplace>": true}` (force-enable) +
  `allowManagedHooksOnly: true` + `extraKnownMarketplaces` +
  `strictKnownMarketplaces: true` + `disableSideloadFlags: true` +
  `allowedHttpHookUrls: ["http://127.0.0.1:<port>/hook"]` (only if using http hooks) +
  `allowManagedPermissionRulesOnly` + the §5.3 `permissions.deny` backstop. Plus, outside
  Claude Code: `janusd` as a systemd service under its own user, socket group-readable
  by the agent user, config root-owned.

## 10. Decision 9 — testing

**Offline (`tests/test_claude_code_adapter.py` etc., default suite):**
- **Pinned-payload fixtures** — the burn we already took (`tool_output` vs
  `tool_response`) becomes the test design: JSON payloads captured verbatim from a real
  CLI session, asserting `normalize_cli_event` extracts every field from the *bytes the
  CLI actually sent*, not from shapes we invented. **Captured 2026-08-15 against CLI
  2.1.233: `tests/fixtures/claude_code_payloads/`** (Pre/PostToolUse for built-in,
  MCP, and in-subagent calls; `Agent` spawn/result; PostToolBatch; lifecycle events) —
  its README records provenance, the doc-contradicting findings, and the gaps still to
  capture (plugin-MCP names, PostToolUseFailure, PermissionRequest/Denied, PreCompact,
  non-default permission modes).
- Normalizer: `tool_output` read, `tool_response` fallback, both-absent → recorded as
  no-output (and the cross-check still marks the id seen).
- Gate-mode semantics: unlisted tool → `{}`; listed tool → enforced; taint-gated sink →
  escalate/deny per `on_gate` + overrides + downgrade rule; `mode="policy"` →
  default-deny preserved (this is the enforcement-review line item).
- Resolver: both mcp name grammars; unknown-server sentinel never matches a policy key.
- Escalate/deny JSON byte-shapes; exception in decision path → deny (fail closed).
- `SessionRegistry`: lifecycle, TTL eviction with flush, concurrent get_or_create,
  snapshot/restore round-trip preserving events/first-cause/seq.
- `janusd` via FastAPI TestClient: dispatch, cross-check miss detection, admin auth.
- Shim: daemon-unreachable → deny JSON within deadline; spool-and-replay of PostToolUse.
- Packaging: `claude plugin validate --strict` on the checked-in plugin dir; module
  imports on core install (no `server` extra).

**Live smoke (`tests/smoke/test_live_cli_semantics.py`, `JANUS_LIVE_SMOKE=1`), asserting
the CLI-side contract on a pinned CLI version, results logged in this doc's table:**
1. **Payload shape**: drive `claude -p` with a temp settings file whose hooks dump raw
   stdin to files; diff key-sets against the pinned fixtures → this is the regression
   tripwire for the next `tool_response`-style rename.
2. PreToolUse deny JSON is honored (denied tool did not run); reason reaches the model.
3. PostToolUse fires with `tool_output` for an executed call; taint recorded end-to-end
   (fetch-then-gated-sink scenario denies).
4. `escalate` behavior headless and interactive-simulated **[verify-live gap closes here]**.
5. `JANUS_SMOKE_SLOW=1`: hook exceeding its timeout → observe whether the tool ran
   (documented fail-open confirmed or refuted on the pinned version — the SDK-path
   experiment found docs wrong once; do not assume either way).
6. Managed-settings experiment (root required, opt-in env guard): force-enabled plugin
   hooks fire under `allowManagedHooksOnly`; inline managed hooks — do they load (#33824)?

| Date | CLI | Result |
|---|---|---|
| — | — | no verified runs yet |

## 11. Phased implementation plan

**Phase 1 — adapter core (one commit).** `janus/adapters/claude_code.py`
(`CliHookEvent`, `normalize_cli_event`, `claude_code_resolve_name`, `decide_cli_event`
in gate + policy modes, escalate mapping), `janus-hook` in stateless mode (policy file
read per call via the argv-flag contract in §8 — no daemon, no taint; imports janus,
unlike the phase-2 proxy hot path; documented as degraded), `[project.scripts]`
entry, offline tests including hand-captured pinned-payload fixtures, `docs/adapters.md`
section with the §1 honesty table. Independently useful: settings-file hook enforcement
of a static policy, today.

**Phase 2 — the daemon.** `SessionRegistry`, `TaintTracker/Session.snapshot()/restore()`
(core), `janus/hookd.py` + `janusd`, shim proxy mode with fail-closed + spool,
PostToolUse cross-check, PostToolBatch re-check. Live smoke suite lands here (payload
pinning automated).

**Phase 3 — the plugin.** `.claude-plugin/` + `hooks/hooks.json` + `userConfig` +
SessionStart bootstrap + `doctor`, marketplace-in-repo, `claude plugin validate
--strict` in CI, release-skill version-bump check.

**Phase 4 — enterprise + escalate polish.** Managed-settings verification (smoke #6),
documented enterprise block, endorsement-on-approval wiring from
`PermissionRequest`/`PermissionDenied`, `permissions.deny` backstop doc block finalized
against real deployments.

## 12. Non-goals (restated from the prompt)

PDE/SpiceDB; any change to `janus_options()`/SDK-path semantics (the CLI adapter reuses
`decide_call` and `Session` but touches neither's behavior); LangChain/ADK; anything
requiring an upstream CLI feature that does not exist today (notably: no wishing for a
CLI `strict_mcp_config` or fail-closed hook-timeout mode — designed around, not assumed).

## 13. Open questions / uncertainty register

Resolved by the 2026-08-15 fixture capture (CLI 2.1.233): output shapes for built-in
and MCP tools (was item 3 — plugin-MCP shapes still open), the `tool_response`-not-
`tool_output` key question, `Agent`-not-`Task` spawn naming, subagent-only
`agent_id`/`agent_type`, and `PostToolBatch` firing with a `tool_calls` array.

Still open **[verify-live]**, in priority order: (1) http-hook behavior on
connection-refused; (2) `escalate` semantics headless and per permission_mode; (3)
plugin-MCP tool-name grammar and output shapes on the wire; (4) hook-timeout
fail-open confirmation on the pinned CLI; (5) managed-settings inline-hooks loading
(#33824 stale-closed) and force-enabled-plugin exception; (6) whether plugin.json truly
has no permissions surface; (7) CLI-internal tool names needing passthrough in
`mode="policy"`; (8) `PermissionRequest`/`PermissionDenied` payloads sufficing to
correlate an approval back to a specific escalated `tool_use_id`.
