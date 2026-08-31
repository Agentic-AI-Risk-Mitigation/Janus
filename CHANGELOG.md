# Changelog

All notable changes to Janus will be documented in this file.

This project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- **Path policies did not match on Windows — every secret-read deny was silently allowed
  there.** Claude Code reports `tool_input.file_path` with the *host's* separator
  (`C:\Users\...\.env`, verified against a live CLI 2.1.246 session), while the starter
  policy anchored on `/`. Against the previous starter, reads of `.env`, `~/.ssh/id_rsa`,
  `~/.aws/credentials` and `~/.claude/.credentials.json` were all **allowed** on Windows, as
  were writes to `.claude/settings.json` — the rule meant to stop an agent disarming the
  guard. Only `\.pem$` held, being the one pattern needing no separator. Path patterns now
  use a separator class (`janus.cli.starter_policy.SEP`, `[/\\]`), user-supplied paths are
  normalized the same way, and `examples/claude_code/policy.starter.json` is regenerated to
  match. A Windows payload fixture captured from a live session
  (`tests/fixtures/claude_code_payloads/pretooluse.windows-read.json`) pins it. The bug was
  invisible because every prior fixture, and the whole CI matrix, was Linux.
- **`janus init` verification reported PASS against paths the CLI never sends.** Its probes
  built paths with `as_posix()`, so on Windows they exercised forward slashes while the
  deployment received backslashes — seven green checks on a policy that was allowing `.env`
  reads. Probes now use the host's native separator.
- **The `janus-hook` deadline was inert on Windows.** `_deadline` needs `SIGALRM`, so on
  Windows it degraded to no deadline at all, and a wedged decision ran until the CLI's own
  hook timeout — which fails **open**. A worker-thread fallback restores the property: the
  shim reaches its own limit and emits a deny while it still can. This also fixes the one
  test that had been failing on Windows.
- CI now runs the suite on `windows-latest` as well as `ubuntu-latest`. All three bugs above
  were platform-specific and a Linux-only matrix could not see any of them.

### Changed

- **BREAKING — `openai` and `jinja2` moved out of core dependencies** into the new `generate`
  extra (`pip install 'janus-guard[generate]'`). They were only ever used by the LLM policy
  generator, yet every consumer of the enforcer or an adapter paid for them — and `secure`
  documents the fallout (a defensively pinned `openai` in its own requirements). The core
  install is now what the docs always claimed: stdlib + `jsonschema` + `pydantic`. If you call
  `generate_policy()`/`refine_policy()`, add the extra; nothing else changes.
- **BREAKING — the policy generator no longer calls `load_dotenv()` at import time**, and
  `python-dotenv` is no longer a dependency. Importing any part of Janus previously executed an
  upward `.env` search from the process's *current working directory* and injected whatever it
  found into `os.environ` — a real-world footgun that silently supplied a stale
  `OPENAI_API_KEY` to a downstream project and made its auth behavior cwd-dependent. Callers
  now manage their own environment; the generator reads keys from `os.environ` or its
  `api_key` argument only.
- `generate_policy`/`refine_policy` are now resolved lazily (PEP 562) from `janus`, so
  `import janus` — and therefore every adapter import — no longer imports the generator module
  or its dependencies. The public names are unchanged.
- **BREAKING — strict condition semantics by default, closing the missing-argument bypass.**
  Previously the enforcer only checked a condition when its argument was *present*, so a rule
  gating `url` was silently skipped if the model omitted `url` entirely — an allow rule could
  vouch for a call it never inspected. Now (`strict_conditions=True`, the default) an allow rule
  whose condition names an absent argument does **not** match, and the call falls through to
  default-deny. Deny rules already failed closed (a condition on an absent argument matches
  vacuously) and are unchanged. Policies that relied on "constrain only if provided" can opt
  back into the legacy behavior with `PolicyEnforcer(strict_conditions=False)`.

### Added

- **`janus init` — an onboarding wizard, behind a new `janus` console script.** Setting Janus
  up on the Claude Code CLI previously meant hand-writing a policy, pasting a hooks block into
  a settings file, and merging the backstop by hand; a guard nobody finishes installing
  protects nothing. `janus init` asks a handful of questions (scope, what to protect, network
  posture, git posture, MCP servers, strictness), shows the exact settings diff, and on
  confirmation writes the policy, the `PreToolUse` entry — with the explicit `timeout` the docs
  always asked for and no example ever showed — and the `permissions.deny` backstop. It then
  verifies by feeding synthetic payloads through `handle_cli_payload` with the flags it just
  wrote, so a `PASS` reflects the deployed decision path rather than the wizard's intent.
  Re-running updates the existing hook in place; foreign hooks, foreign deny entries, and
  unrelated settings are never touched, and the previous file is backed up. `--dry-run`,
  `--yes` (CI; a non-TTY without it is refused rather than defaulted), `--scope`, `--force`.
  Optional: with the `generate` extra and an API key, it can draft argument-level rules for
  review — accepting *replaces* a tool's blanket allow, since generated priority-100 rules
  would otherwise sit unreachable behind it.
  The `janus` script is deliberately separate from `janus-hook`, which stays a pure
  decision process with no interactive surface. `janus doctor` delegates to the same
  `janus.cli.hook.run_doctor` (renamed from `_doctor`) that `janus-hook doctor` uses.
- **Claude Code CLI adapter** (`janus.adapters.claude_code` + the `janus-hook` console script,
  core install — no extra): enforce a Janus policy on the *interactive* `claude` CLI via its
  `PreToolUse`/`PostToolUse` hooks. Unlike the SDK path, Janus does not construct the session
  here, so this is a policy monitor backstopped by `permissions.deny` (`janus-hook backstop`
  prints the block), not a reachability lockdown — `docs/adapters.md` spells out the weaker
  security model. `mode="gate"` (default) enforces only the tools the policy has an opinion
  about and abstains to the CLI permission flow elsewhere; `mode="policy"` is strict
  default-deny; gate mode auto-promotes to policy mode under `bypassPermissions`, where
  abstention would degrade to a silent allow. The shim owns its exit path so enforcement fails
  *closed* (unreadable policy, internal error, or its own `--deadline` all deny) even though
  the CLI's hook dispatch fails *open* — a hook that overran the CLI's `timeout` had its deny
  discarded on 2.1.233. Taint-gate escalation emits the CLI's `ask` decision (verified to
  block and surface the reason; `escalate` is unrecognized and would silently allow).
  Phase 1 is deliberately stateless — static policy evaluation per call, no taint, no
  provenance, no cross-call state; the daemon that restores those is phase 2
  (`plans/claude-code-plugin-design.md`).
- `tests/test_claude_code_adapter.py` + `tests/test_claude_code_shim.py` (81 offline tests)
  covering payload normalization, gate/policy semantics, unsupervised promotion, escalation
  downgrade, and the shim's fail-closed paths.
- **`on_decision` audit callback in the Claude Agent SDK adapter** — `janus_options()`,
  `janus_hooks()`, and `janus_pretooluse_hook()` accept an optional
  `on_decision(runtime_tool_name, arguments, allowed, reason)` callable, invoked once per
  PreToolUse evaluation (passthrough tools and the fail-closed internal-error path included;
  `reason` is `None` on allow). Gives downstream consumers a programmatic seam to audit
  hook-level policy denies, which previously surfaced only in Python logging. Strictly
  observational: callback exceptions are logged and swallowed and can never change an
  enforcement outcome. When a `Session` is wired, denies are additionally recorded as
  `{"kind": "policy_deny", ...}` session notes, giving `session.events` symmetry with the
  taint `gate_deny` events. Version bumped to 0.1.1 so consumers can feature-detect
  `on_decision` from `janus.__version__`.
- **Prompt-borne untrusted input** — `Session.mark_untrusted(text, label=, extract=,
  normalize=)`: the one-line, audited way to declare pasted content (an inbound email, a
  scraped page) untrusted at the call site that already knows it. Taints the session exactly
  like reading an untrusted tool (gates fire), and optionally seeds the provenance set
  `untrusted:<label>` for `not_in(...)` argument conditions and output checks. Closes the gap
  where a fully-wired tracker reported a clean session while the hottest input in the system
  rode in via the prompt. Extractor errors propagate — a silently empty deny-set fails open.
- **Endorsements** (`janus.policy.endorsement`, via `Session.endorse` /
  `Session.endorse_event`): audited, consumable declassification for in-the-loop tasks where
  untrusted data legitimately drives the action. Value-scoped by default — an exact
  `(tool, arg, value)` triple lifts only the deny that matches it, `uses=1` then the gate is
  closed again; `scope="taint"` (explicit) lifts a whole-tool taint gate once; `uses=None`
  standing endorsements warn on every consumption. Mandatory `by=`/`reason=` attribution.
  Taint stays monotonic — nothing is ever un-tainted. Deny reasons from taint gates and
  provenance conditions now carry an `(audit id ...)` suffix, and `endorse_event(id, by=,
  reason=)` endorses exactly that deny mechanically — no retyping values, no scope guessing.
- **Output-side checks** (`janus.checks`): deterministic, provenance-aware assertions over
  model output for the text-to-text gap and the tool-free `structured()` shape — not an
  injection classifier. `check_output(output, session, checks=[...], enforce=False)` runs
  checks over final text or structured-output dicts, appends findings to the session audit
  trail, and with `enforce=True` raises the new `OutputViolation` before code acts on the
  output. Built-ins: `echoed_untrusted_values(label)` (flags values from the marked untrusted
  input that rode into the output — the generalized "inbound URL echoed into the draft"
  backstop; reports values as written in the output, matches through the set's normalizer)
  and `values_grounded_in(allowed=[...])` (every extracted value must come from an allowed
  provenance set). A check that raises becomes a finding, never a silent pass. `extract_urls`
  ships as the common extractor for both `mark_untrusted` and the check factories.
- **Argument-value provenance** (`janus.policy.provenance`): `ProvenanceLedger` records named
  value-sets from tool outputs at the post-execution seam (`collect(tool, label=, extract=,
  normalize=)` + `record`), and two condition factories gate arguments on membership at the
  pre-execution seam — `from_output(label)` (**positive** provenance: the argument must be a
  value a listed tool actually returned; empty set or missing session denies) and
  `not_in(label)` (**negative** provenance: deny values from an untrusted set). This makes the
  hand-rolled "fetch only URLs a prior search returned" pattern a one-line policy condition:
  `"url": all_of(ssrf_ok, from_output("searched_urls"))`. Exact-match membership with opt-in
  normalization (`normalize_url` ships for the URL case); every collection and every
  provenance-caused deny lands in an audit trail. Collector errors default to "collected
  nothing" (fails closed for allow-sets); register deny-set collectors with `on_error="raise"`.
- **`Session`** (`janus.policy.Session`): the explicit per-run home for cross-call state —
  wraps a `TaintTracker` and a `ProvenanceLedger`, feeds both from one
  `record_output(tool, output)`, and merges their audit trails in `events`. Threaded
  explicitly everywhere: `janus_options(session=...)` / `janus_hooks(session=...)` wire the
  `PostToolUse` recording seam (with MCP content blocks unwrapped via the new
  `unwrap_tool_response()` so extractors see the dict the tool body returned) and expose the
  session to context conditions at `PreToolUse`; `PolicyEnforcer.enforce(session=)`,
  `decide_call(session=)`, and `janus.testing.decide/replay(session=)` take the same object.
  The adapter's `taint=` keeps working but is deprecation-warned (`session=Session(taint=...)`
  supersedes it); passing both raises.
- **Context-aware conditions** (`janus.policy.conditions`): callable conditions can now opt
  into a richer contract with the explicit `@context_condition` marker — they are invoked as
  `restriction(value, ctx)` where `ConditionContext` carries the tool name, the argument name,
  a **read-only** view of the full call's arguments, and per-run `session` state (threaded via
  the new keyword-only `PolicyEnforcer.enforce(..., session=)`; `None` until an integration
  wires one). Unmarked callables keep the classic single-argument contract untouched — the
  dispatch is an attribute check, never signature inspection, so wrappers and partials can't
  silently change a condition's contract. A marked condition evaluated without a context fails
  closed. This dissolves the expressiveness boundary that forced consumers to hand-roll
  cross-argument and session-dependent checks inside tool bodies.
- **Condition composition**: `all_of(*restrictions)` / `any_of(*restrictions)` compose
  restrictions of any supported kind (JSON Schema dict, regex string, plain callable, context
  condition), nestably. `all_of` denies on the first failing member and propagates that
  member's own message; `any_of` denies only if every member fails, reporting all attempts.
  Strict absent-argument semantics are unchanged: an allow rule conditioning an absent argument
  still does not match, composed or not.
- **`janus.testing` — a public policy-test harness**, replacing the private-API imports
  consumers were forced into (both `secure` policy test files imported the adapter's
  `_decide`). `decide(policy, tool, args, ...)` evaluates one call through the exact decision
  core the Claude Agent SDK `PreToolUse` hook runs (now factored into
  `janus.policy.decision.decide_call`) and returns a `Decision` with `allowed`/`denied`, the
  deny `reason`, and which **layer** decided (passthrough / taint / required_args / rules).
  `replay(policy, sequence)` feeds recorded `(tool, args, ALLOW|DENY)` sequences through the
  same core — `tests/test_ipi_scenarios.py` is its first consumer. The adapter's `_decide`
  remains as a thin delegate, so existing imports keep working.
- **`TaintTracker`** (`janus.policy.TaintTracker`, core install — no SpiceDB): framework-agnostic
  session taint with **per-source integrity labels** instead of the PDE engine's monotonic
  scalar. `sources={tool: label}` classifies untrusted reads, `gates={tool: labels}` blocks sinks
  once a listed label has tainted the session (`"*"` gates on any taint — the strict Rule of Two),
  and `classify=` adds content-aware labels. Derivation is automatic via the two seams
  (`record_output` post-execution, `check` pre-execution), so no manual `update_taint()` calls are
  needed. Every taint introduction and gate denial is recorded in `events` for audit; `reset()`
  clears a session. New `docs/taint.md`.
- **`janus_options()`** (Claude Agent SDK adapter): the recommended entry point, building a
  locked-down `ClaudeAgentOptions` so a silently skipped `PreToolUse` hook — a regression that has
  shipped upstream — can no longer escalate to arbitrary `Bash`. Generates `tools=[]`,
  `strict_mcp_config=True`, `allowed_tools` = policy ∩ mounted tools, `permission_mode="dontAsk"`,
  and built-ins + `Task` in `disallowed_tools`. Overrides that weaken the lockdown raise
  `ValueError` unless `unsafe_overrides=True`; `allowed_tools`/`disallowed_tools` overrides are
  merged, not replaced, so they can only shrink the reachable surface. `hook_approved_tools=`
  keeps high-risk sinks off `allowed_tools` so the hook and the permission layer must both agree;
  `extra_hooks=` merges your own matchers alongside Janus's.
- **`janus_posttooluse_hook()` / `taint=` on `janus_hooks()`**: automatic taint derivation for the
  Claude Agent SDK — `PostToolUse` records untrusted reads, `PreToolUse` gates sinks on them before
  the static policy runs. Only calls that produced a response are recorded, so a blocked read never
  taints the session.
- **Live SDK smoke suite** (`tests/smoke/`, skipped unless `JANUS_LIVE_SMOKE=1`): exercises the real
  `claude` CLI to verify the SDK-side semantics `janus_options()` depends on — hook firing,
  `allowed_tools` shadowing, `tools=[]` availability, `StructuredOutput` delivery. Verified against
  `claude-agent-sdk` 0.2.120 / CLI 2.1.218; runs recorded in `plans/claude-agent-sdk-hardening.md`.
  `JANUS_SMOKE_SLOW=1` adds the hook-timeout experiment.
- **Core `required_args`**: `PolicyEnforcer(required_args={"tool": ["arg", …]})` requires named
  arguments to be present and non-empty (rejecting `None` and blank strings) before rule
  evaluation, complementing strict conditions for arguments no condition covers. Promoted from
  the Claude Agent SDK adapter, which now delegates to the shared
  `janus.policy.enforcer.check_required_args` (its per-call `required_args` parameter is
  unchanged).
- **Enforcer semantics regression suite** (`tests/test_enforcer_semantics.py`): priority
  ordering, deny-before-allow tie-break at equal priority, default-deny, strict/legacy
  missing-argument behavior, `required_args`, policy management, and fallback actions.
- **Indirect-prompt-injection scenario suite** (`tests/test_ipi_scenarios.py`): replay-style
  attack/legitimate tool-call sequences under an outreach-pipeline-shaped policy — poisoned
  page → email exfiltration, malicious reply → missing-argument bypass, honeypot log → SSRF /
  attacker-directed scans — asserting attacks are blocked and legitimate twins still flow.
- **CI test workflow** (`.github/workflows/test.yml`): pytest + ruff + mypy on Python 3.10–3.13.

## [0.0.5] — 2026-07-16 (Alpha)

### Added

- **Claude Agent SDK adapter** (`janus.adapters.claude_agent_sdk`, behind the new `janus-guard[claude]` extra): enforce a Janus policy on a [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) / Claude Code tool loop, whose loop runs inside the `claude` CLI subprocess. `janus_pretooluse_hook()` / `janus_hooks()` gate every tool call via a `PreToolUse` hook (the robust seam); `make_can_use_tool()` is a `can_use_tool` alternative (bypassable by `allowed_tools`/`bypassPermissions` shadowing, documented as such); `guard_tool_body()` wraps an in-process `@tool` body as belt-and-braces. The adapter strips the `mcp__<server>__` tool-name prefix, passes the SDK-internal `StructuredOutput` tool through, and backstops the enforcer's absent-argument bypass via `required_args`.
- `tests/test_claude_agent_sdk_adapter.py` (20 offline tests) and `examples/claude_agent_sdk_demo.py` (live end-to-end demo). New `docs/adapters.md` reference page.
- Regression tests (`tests/test_standalone_enforcer.py`) covering standalone `PolicyEnforcer` import/enforcement (callable + JSON-Schema conditions, default-deny, all three fallbacks) with `authzed` unimportable, and the actionable PDE `ImportError`.

### Changed

- **Lightweight core install**: the core package no longer depends on `authzed`, `fastapi`, or `uvicorn`. `PolicyEnforcer` and the static-policy path now import only `jsonschema` + `pydantic` (+ `jinja2`/`python-dotenv`/`openai` for generation and the default provider), so `from janus.policy import PolicyEnforcer` works as a standalone tool-call gate without the SpiceDB/PDE stack.
- **New optional extras**: `janus-guard[pde]` (SpiceDB-backed ReBAC + taint tracking / `PDEEnforcer`) and `janus-guard[server]` (FastAPI/uvicorn demo webapp). `authzed`, `fastapi`, and `uvicorn[standard]` moved out of core into these extras; they are still bundled in `[all]` and `[dev]`.
- **`PDEEnforcer` is now imported lazily** (PEP 562 `__getattr__` in `janus.policy`). Using the PDE engine (`policy_engine="pde"` or `from janus.policy import PDEEnforcer`) without the `pde` extra installed raises a clear, actionable `ImportError` instead of a raw `ModuleNotFoundError` for `authzed`.
- **Python 3.10 supported**: lowered `requires-python` to `>=3.10` (core modules use no 3.11-only features; ruff/mypy targets updated accordingly).

## [0.0.4] — 2026-03-13 (Alpha)

### Changed

- **PDE integration**: Policy-Discovery-Engine has been merged into the main repo. SpiceDB-backed enforcement now lives under `janus/policy/pde/` (config, interceptor, discovery, bootstrap). `PDEEnforcer` imports from `janus.policy.pde.interceptor`; no separate `Policy-Discovery-Engine/` directory. Demos and docs updated accordingly.
- **Demo workflow**: added `scripts/run_demo_webapp.sh` as a repo-root entrypoint for the FastAPI demo UI, with optional local SpiceDB startup for Demo 5.
- **Project docs**: updated AGENTS/CLAUDE/README guidance to match the current `examples/`-based demo layout and the absence of a checked-in `tests/` tree on `main`.
- **Pytest config**: removed the stale `testpaths = ["tests"]` setting so local `pytest` no longer warns about a missing `tests/` directory.

## [0.0.3] — 2026-03-12 (Alpha)

### Added

- **Core engine**: `PolicyEnforcer` with JSON Schema–based argument validation and priority-ordered rule evaluation
- **JanusAgent**: single entry point wrapping LLM conversation loop with policy enforcement
- **10+ LLM providers**: OpenAI, Anthropic, Google Gemini, Azure OpenAI, AWS Bedrock, Ollama, vLLM, Together AI, OpenRouter
- **Framework adapters**: LangChain (3 integration depths) and Google ADK
- **LLM-generated policies**: auto-infer minimum-privilege policies from a user query via `policy="generate"`
- **Policy refinement**: incrementally tighten policies as an agent discovers new information
- **Built-in tools**: `read_file`, `write_file`, `edit_file`, `list_directory`, `run_command`, `fetch_url` — all workspace-scoped with path-traversal rejection
- **Custom tools**: `ToolDef` / `ToolParam` dataclasses for registering and guarding arbitrary tools
- **Three fallback actions**: raise `PolicyViolation`, call `sys.exit()`, or prompt user interactively
- **PDE (janus/policy/pde/)**: SpiceDB-backed graph enforcement with Zanzibar-style ReBAC and runtime taint tracking for IPI defence
- **Demo scenarios**: scripted demos for poisoned-README (IPI) and taint-cascade attacks
- **Web demo app**: FastAPI split-panel UI for live demo playback
- **Documentation site**: MkDocs Material with architecture, policy reference, getting-started, and demo guides
- **Test suite**: E2E PDE/SpiceDB integration tests and unit tests for enforcement, scripted LLM, mock tools, and scenario runner
