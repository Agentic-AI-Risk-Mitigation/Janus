---
name: enforcement-review
description: Pre-commit review checklist for changes touching janus/policy/, janus/adapters/, janus/checks.py, or janus/testing.py. Use before committing any change to enforcement semantics, adapters, or the testing harness — it operationalizes CLAUDE.md's "must preserve default-deny" rule.
---

# Enforcement-semantics review

Adopt the reviewer role from CLAUDE.md: seasoned senior programmer with deep Python/security/agent-systems experience. Priorities in order: (1) code quality, (2) security, (3) best practices. Question the intent of the code against its functionality, as an architect.

Review the current diff (`git diff` + `git diff --staged`; include untracked files under the touched paths).

## 1. Run the semantics-critical test subset, then the full suite

```bash
uv run pytest tests/test_enforcer_semantics.py tests/test_standalone_enforcer.py \
  tests/test_ipi_scenarios.py tests/test_condition_context.py \
  tests/test_provenance.py tests/test_untrusted_and_endorsement.py \
  tests/test_checks.py tests/test_testing_harness.py \
  tests/test_claude_agent_sdk_adapter.py tests/test_session_wiring.py -q
uv run pytest
```

## 2. Walk the invariant list against the diff

For EACH invariant below, state either where the diff touches it (file:line) and why it still holds, or "not touched":

- **Default-deny**: a tool absent from a loaded policy is denied; no rule matching ⇒ denied.
- **Strict conditions**: an allow rule conditioning an absent argument does not match (`strict_conditions=True` default); deny rules match vacuously on absent args.
- **Tie-break**: at equal priority, deny sorts before allow (`_sort_policy`).
- **Fail closed on Janus's own defects**: the PreToolUse hook wraps everything — an unexpected exception returns `permissionDecision: "deny"`, never an error the CLI treats as pass-through. Same for `make_can_use_tool`.
- **Context conditions fail closed**: a `@context_condition` evaluated without a `ConditionContext` denies; provenance conditions deny on missing session, missing set, or empty set (`from_output`) — and `not_in` denies on missing session too (an unwired deny-set must not fail open).
- **Collector asymmetry**: ledger collectors default to skip-on-error (fail-closed for allow-sets); deny-set collectors and `mark_untrusted` extractors must propagate errors.
- **Monotonic taint**: nothing un-taints within a session; endorsements are consumed at the gate/condition and never clear state; `reset()` only at session boundaries.
- **No global state**: no new module-level mutable state anywhere; all cross-call state lives in explicitly passed objects (`Session`, `TaintTracker`, `ProvenanceLedger`, `EndorsementLog`), each lock-protected.
- **Audit completeness**: every NEW deny path or state mutation the diff adds must land in an events trail (taint events, ledger events with `prov-N` ids, endorsement events, session notes); deny reasons for endorsable denials must carry the `(audit id ...)` suffix.
- **Back-compat surface**: `_decide`'s signature/return in `janus/adapters/claude_agent_sdk.py` (the `secure` repo's tests import it), policy dict/tuple formats, `janus_options()` parameters, and the deprecated-but-working `taint=` must not break without a CHANGELOG entry and migration note.

## 3. Consumer check (only if the public API changed)

```bash
grep -rn "from janus\|import janus" ~/projects/secure --include='*.py' | head -50
```

Confirm every touched symbol is either unchanged for those consumers, or the change is deliberate and noted in the CHANGELOG.

## 4. Report

Output:

1. A short findings list — each entry: `file:line — invariant — why it holds or breaks`.
2. An explicit verdict line: **safe to commit** or **needs changes** (with the blocking items).
