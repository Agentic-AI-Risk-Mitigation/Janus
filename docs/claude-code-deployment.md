# Claude Code CLI: Deployment & Threat Model

This page is for the operator choosing *how* to deliver the Janus hook to an interactive
Claude Code deployment. The mechanics of wiring the hook are in
[Adapters → Claude Code CLI](adapters.md#claude-code-cli-interactive-claude); this page
answers the prior question — which delivery vehicle, and what each one actually defends
against.

## The attacker model

The attacker is **the agent itself**. The scenario Janus exists for is an indirect prompt
injection that has already landed — a poisoned README, a fetched page, a tool output
carrying instructions. From that point the model driving the session is adversarial; the
human and the host are not. Same-user malicious *code execution* is out of scope: if the
attacker already runs arbitrary Bash outside policy, no hook seam helps — gating Bash is
the policy's job, not the threat model's.

That framing changes the question that picks the delivery vehicle. It is not "can a hook
deny a bad call?" — every vehicle below can. It is: **can the guarded agent remove its own
guard?**

## The deployment ladder

| Capability of the injected agent | settings.json hooks | plugin (user-installed) | plugin + managed settings |
|---|---|---|---|
| Call a policy-denied / taint-gated tool this session | no (hook denies) | no | no |
| Disarm the guard **this session** by editing settings | **yes** — settings hooks are re-read from disk, not snapshotted; one `Edit` of `~/.claude/settings.json` removes the hook mid-session | no — plugin hooks are snapshotted per session | no |
| Disarm for **future** sessions (edit settings, disable plugin, `--plugin-dir` shadow) | yes | yes | no — `allowManagedHooksOnly` + managed force-enabled `enabledPlugins`; `--plugin-dir` cannot override |
| Exfiltrate via a tool Janus has no opinion on (gate mode) | falls through to the CLI permission flow / the human — **except** under `bypassPermissions`, where gate mode promotes to default-deny because there is no human to fall through to | same | same, plus a managed `permissions.deny` backstop |

Three tiers, three guarantees:

1. **Settings-file hooks** (phase 1, what `janus-hook` wires today) catch accidents and
   provide argument-level enforcement of a static policy — genuinely useful, and **not a
   security boundary against the agent it guards**. Hooks in a settings file are re-read
   from disk, so the agent can edit them away mid-session; and even where `Write`/`Edit`
   deny rules protect the settings and policy files, `Bash` can rewrite them anyway. Fine
   for evaluation; be honest about what it is.
2. **A plugin** is the minimum for *"the session I started stays guarded"* — plugin hooks
   are snapshotted when the session starts, so nothing the agent writes to disk disarms
   the running session.
3. **Plugin + managed settings** is the minimum for *"the machine stays guarded"* —
   `allowManagedHooksOnly: true` with the plugin force-enabled via managed
   `enabledPlugins` means the agent's user cannot disarm future sessions either.

The plugin and managed tiers ship in later phases; the design and verified probe results
live in `plans/claude-code-plugin-design.md` in the repository.

## Why managed settings must use the force-enabled-plugin path

The "obvious" enterprise design — declare the hooks inline in managed settings, skip the
plugin — is exactly the one to avoid. Upstream issue **#33824** (managed/policy settings
hooks not loaded into the hook registry) was closed **stale, never confirmed fixed**: hooks
declared inline in managed settings may silently not run, which for a guard is the worst
failure mode available. Route enterprise deployments through the force-enabled-plugin
exception instead, and treat any managed deployment as unverified until a live smoke test
shows its hooks actually firing.

## The backstop is load-bearing at every tier

`janus-hook backstop` prints a `permissions.deny` block (network sinks: `curl`, `wget`,
`ssh`, `scp`, `nc`, `git push`, `WebFetch`) to merge into settings — managed settings, for
the enterprise tier. It is not decoration: the CLI's hook dispatch has shipped regressions
where `PreToolUse` hooks silently did not run, and permission rules are the only layer that
holds with zero hooks running. Extend it with the true sinks of your deployment (outbound
MCP tools included).

## Residual risk, stated plainly

Between a hook-dispatch regression and its detection, calls not covered by
`permissions.deny` run unenforced. On the CLI seam this window cannot be closed — only
shrunk and alarmed. Deployments that cannot accept that should use the
[Claude Agent SDK path](adapters.md#claude-agent-sdk-claude-code): `janus_options()` remains
the flagship precisely because it constructs the session, so tool *existence* is enforced
by the CLI at session start rather than depending on a hook firing.
