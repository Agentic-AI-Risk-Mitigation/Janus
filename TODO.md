# Janus — TODO / Improvement Backlog

Things to improve from the repo

---

## 🔴 Security — Fix Before Production

### 1. `_check_conditions` silently skips missing arguments
**File:** `janus/policy/enforcer.py:253-255`

The allow-rule evaluator only validates arguments that are *present*. If an LLM omits an argument that a policy condition restricts, the condition is bypassed and the tool call is allowed. This is a genuine security hole.

```python
# Current (broken):
for arg_name, restriction in conditions.items():
    if arg_name in arguments:        # ← skips missing args
        validate_argument(...)
```

**Fix:** Raise `ArgumentValidationError` when a restricted argument is missing.

---

### 2. `run()` swallows all exceptions, including unexpected ones
**File:** `janus/llm/runner.py:88-93`

`LLMRunner.run()` catches `Exception` broadly and returns the error as a plain string. The caller has no way to distinguish a successful response from a fatal error.

Note: `_execute()` (line 178) deliberately catches `PolicyViolation` and feeds it back to the LLM as text — *this is correct by design* so the LLM can adjust. The problem is the outer `run()` catch, which should re-raise non-recoverable errors.

**Fix:** Remove the broad `except Exception` in `run()`, or at minimum re-raise `PolicyViolation` and other `JanusError` subclasses.

---

### 3. Hardcoded SpiceDB token `"somerandomkey"` in three places
**Files:** `enforcement.py:26`, `pde_enforcer.py:22`, `docker-compose.yml:8`

No environment variable fallback. Fine for local dev/testing, but must not ship to production.

**Fix:** `os.environ.get("SPICEDB_TOKEN", "somerandomkey")` as default; document the env var.

---

## 🟡 Important — Quality & Robustness

### 4. No unit tests for the native `PolicyEnforcer`
**Status:** The `tests/` directory does not exist. All current tests are E2E only (on `coding-agents-tests` branch).

The rule evaluation logic (`_evaluate_rules`, `_check_conditions`, priority ordering, deny/allow precedence) is untested. This is the most security-critical code in the repo.

**Fix:** Create `tests/test_enforcer.py` with parametrised cases covering: allow rules, deny rules, priority ordering, missing-arg behaviour (issue #1), default-deny on no match, and all three fallback modes.

---

### 5. No graceful degradation when SpiceDB is unreachable
**File:** `Policy-Discovery-Engine/policy_engine/enforcement.py`

If SpiceDB is down, `check_tool_access` raises a raw gRPC exception. No timeout, no retry, no fail-open/fail-closed toggle.

**Fix:** Add a configurable `fail_closed` mode (default: deny all) and a timeout on `CheckPermission`.

---

### 6. Taint level has no session reset
**File:** `Policy-Discovery-Engine/policy_engine/enforcement.py:31-33`

`update_taint` uses `max()` so taint only ever increases. There is no way to start a fresh session without creating a new agent object. In long-running services this means taint leaks across tasks.

**Fix:** Add `reset_session()` to `GraphInterceptor` and `PDEEnforcer`.

---

### 7. `main.py` mixes schema, bootstrapping, constants, and demo code
**File:** `Policy-Discovery-Engine/policy_engine/main.py` (305 lines)

`SCHEMA`, `TOOL_TAINT_LIMIT`, `RISK_TO_TAINT`, `bootstrap()`, `Session`, `allow_tool()`, and the `__main__` demo script are all in one file. Other modules import constants from it, creating tight coupling.

**Fix:** Split into `schema.py`, `bootstrap.py`, and `__main__.py`.

---

### 8. `discovery.py` is partially broken, `caveats.py` is empty
- `caveats.py` — 0 bytes.
- `discovery.py` — uses `ObjectRef` (should be `ObjectReference` in current authzed-py), and `define_risk()` is a stub. `learn_edge` uses the old generic `tool` object type, not the per-tool types from `main.py`.

Both are listed in the README project structure as functional components.

**Fix:** Mark as `# TODO: not yet implemented` and add a note in README, or implement properly.

---

## 🟢 Minor — Polish

### 9. `_drive_loop` off-by-one on max iterations
**File:** `janus/llm/runner.py:109-147`

The first `generate()` call happens outside the while loop. The iteration counter starts at 0 and increments inside the loop. This means the first tool-call round doesn't count toward `max_tool_iterations` — the actual max is `max_tool_iterations + 1`.

**Fix:** Move the initial `generate()` into the loop body.

---

### 10. PDE enforce message is always "blocked by Graph" even for taint blocks
**File:** `janus/policy/pde_enforcer.py:60`

When taint blocks a call, SpiceDB was never queried, but the message says "blocked by Policy-Discovery-Engine Graph". Confusing for debugging.

**Fix:** Have `check_tool_access` return a result enum/named-tuple indicating the denial reason.

---

### 11. Docker healthcheck missing
**File:** `Policy-Discovery-Engine/docker-compose.yml`

SpiceDB takes 3-5 seconds to become ready. No Docker healthcheck means `docker compose up --wait` won't work.

**Fix:** Add a healthcheck using `curl` or `wget` against the HTTP health endpoint (note: `grpc_health_probe` is not bundled in the SpiceDB image).

```yaml
healthcheck:
  test: ["CMD", "wget", "--spider", "-q", "http://localhost:8443/healthz"]
  interval: 2s
  timeout: 5s
  retries: 10
```

---

### 12. CI pipeline not configured
**Status:** `pyproject.toml` already has `ruff`, `mypy`, and `pytest` in dev dependencies with config sections. But there is no GitHub Actions workflow or CI config to run them automatically.

**Fix:** Add `.github/workflows/ci.yml` that runs `ruff check`, `mypy janus/`, and `pytest`.

---

### 13. Connection pooling for SpiceDB client
Every `PDEEnforcer.__init__` creates a new `GraphInterceptor` → new gRPC `Client`. Not an issue in the current single-agent-per-process usage, but would be a problem at scale.

**Fix:** (When needed) Share a single `Client` instance via module-level singleton or dependency injection.
