# Starter policy for the Claude Code CLI

`policy.starter.json` is a ready-to-copy gate-mode policy for guarding an interactive
`claude` session with `janus-hook`. To have it written and wired for you — customized by a
few questions and verified afterwards — run `janus init` instead; it builds this same
policy from `janus.cli.starter_policy.build_starter_policy()`, and a test pins the two
together so the file you copy and the file the wizard writes cannot drift. Wiring
instructions:
[Getting Started → Guard Your Interactive Claude Code](../../docs/getting-started.md);
security model and flag reference: `docs/adapters.md`; choosing a delivery vehicle:
`docs/claude-code-deployment.md`.

What it does:

- **Denies secret reads** — `Read` of `.env*` files (`.env.example` excepted), anything
  under `~/.ssh/`, AWS credentials, `*.pem`, and `~/.claude/.credentials.json`.
- **Denies pipe-to-shell downloads** — `Bash` commands matching `curl … | sh` / `wget … | sh`,
  and commands touching SSH keys, AWS credentials, or Claude credentials.
- **Denies guard tampering** — `Write`/`Edit`/`MultiEdit` of `.claude/settings*.json` or the
  Janus policy directory. (An agent can still route around this via `Bash` — phase 1 is a
  policy monitor, not a lockdown; see `docs/claude-code-deployment.md`.)
- **Explicitly allows the other built-in tools** so sessions running under
  `bypassPermissions` — where gate mode promotes to strict default-deny — keep working.

## Patterns worth copying (and their gotchas)

**Deny rules first, then an unconditional allow.** For any tool the policy lists, "no rule
matched" is a default-deny. A guarded-but-usable tool is therefore two rules: the deny
conditions at a low priority number (evaluated first), then `{"priority": 10, "effect": 0,
"conditions": {}, "fallback": 0}` so everything the denies don't catch falls through.
Omit the trailing allow and the tool is deny-by-default.

**Bypass sessions need the tool enumerated.** Under `bypassPermissions` (including
`--dangerously-skip-permissions`) gate mode promotes to strict policy mode, so an unlisted
tool is denied, not deferred to a prompt — the symptom is
`Tool 'X' is not listed in the policy`. That includes MCP tools: add each one (by its bare
name — the `mcp__<server>__` prefix is stripped) with an allow rule, and set
`known_servers` in the `--config` sidecar so a rogue server can't inherit the rule.

**Regex conditions are searches, not full matches.** JSON Schema `pattern` matches anywhere
in the string (Python `re.search`), so anchor deliberately: `(^|[/\\])\.env` rather than a
bare `\.env`, `\.pem$` rather than `\.pem`. Negative lookahead works —
`\.env(?!\.example)` is how the starter exempts `.env.example`. Anchoring bounds where a
match may *start*; it does not make the match exact, so the starter's `.env` rule also
covers `.environment` — the safe direction to be wrong in.

**Match both path separators.** Claude Code reports `file_path` with the *host's* separator
— verified on Windows (CLI 2.1.246), which sends `C:\Users\...\.env`. A pattern anchored on
`/` alone matches nothing there, so use a class: `[/\\]` for a separator and `[^/\\]` for a
"rest of the segment" tail. This is not hypothetical: an earlier version of this starter was
`/`-only, and on Windows it allowed every `.env`, `~/.ssh`, `~/.aws/credentials` and
`~/.claude/.credentials.json` read, plus writes to `.claude/settings.json`.

**Deny conditions fail closed on absent arguments; allow conditions fail strict.** A deny
rule conditioning an argument the call omits *matches vacuously*; an allow rule
conditioning an absent argument does *not* match. Add `required_args` in the `--config`
sidecar for arguments that must never be absent or blank.
