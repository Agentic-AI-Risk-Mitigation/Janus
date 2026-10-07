I re-read the code as it stands today and ran probes against it. The short version: the Claude adapters are hardened well beyond the rest of the project. The biggest risks are now in the policy loader, the built-in tools, and the release pipeline.

Everything in the "confirmed" tables below I ran on this machine. Nothing in the repo was changed.

1. Fix first: confirmed fail-open behaviours

Policy loading accepts mistakes silently

The policy file is the security-critical artifact, but janus/policy/loader.py:157 never validates it. A typo turns a restriction into an open door:

┌─────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     Policy mistake                      │                                                  Result                                                   │
├─────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ "condition" instead of "conditions" on a URL-restricted │ fetch(http://evil.com) allowed — unknown key ignored, rule becomes unconditional                          │
│  allow rule                                             │                                                                                                           │
├─────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Deny rule with "effect": 2 followed by a catch-all      │ Read(.env) allowed — rule silently skipped                                                                │
│ allow                                                   │                                                                                                           │
├─────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Rule with no effect at all                              │ allowed — defaults to allow (loader.py:170)                                                               │
├─────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ "effect": "deny" (string)                               │ crashes with a raw TypeError, not a PolicyLoadError                                                       │
├─────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ {"type": "string", "format": "email"}                   │ to="not an email; rm -rf /" allowed — jsonschema ignores format unless a checker is passed                │
│                                                         │ (validator.py:56)                                                                                         │
├─────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ {"pattern": "^data/"} without type                      │ path=["/etc/passwd"] allowed — pattern only applies to strings                                            │
├─────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Typed deny rule plus catch-all allow (the               │ file_path=[".env"] allowed — deny doesn't match a non-string                                              │
│ starter-policy shape)                                   │                                                                                                           │
├─────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Bare-string regex "https://trusted\.com"                │ https://trusted.com.evil.io allowed — bare strings use re.match (prefix), schema pattern uses search; two │
│                                                         │  semantics                                                                   │
└─────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────┘

The fix is a strict loader: reject unknown keys, require effect to be g, and warn on untyped patterns and unanchored regexes.validate_policy_structure() already exists but nothing calls it on load.

Other confirmed holes

- fallback: 1 breaks janus-hook. A matching deny rule calls sys.exit(1 prints nothing and exits 1. By the CLI's hook convention a non-2 exitis a non-blocking error, so the tool should proceed — I did not run this against a live CLI. The same rule with fallback: 0 denies correctly.
- Built-in fetch_url reads local files. fetch_url("file:///…") returned my probe file's contents (command_tools.py:63), bypassing the workspace sandbox. It also follows redirects, so a URL allowlist checks only the first hop, and it has no size cap.
- No policy means allow-all, and policy="generate" can land there. If the generator returns an empty policy, agent.py:292 logs a warning and the agent runs with no policy
  at all.
- Regex denial of service. A policy regex ^(a+)+$ took 0.99s on a 24-character argument and doubles per extra character. Arguments are attacker-controlled and there is no timeout.
- The workspace sandbox is a module global (file_tools.py:17). A seconhe first one's sandbox, contradicting the "no global state" designrule.

The published package is far behind

- PyPI's latest is 0.0.5 (July). main says 0.1.1 and has never been released.
- The missing-argument bypass fix landed after 0.0.5, so anyone who installs today gets the bypass.
- janus_options(), Session, provenance, endorsements and janus-hook exist only in git.
- README says v0.0.5, docs reference a "0.0.6" that never existed.

Cutting a release is the highest-value single action.

Windows is unprotected and untested

- The shim's deadline uses SIGALRM, which Windows lacks (hook.py:97), so a stalled hook is killed by the CLI and the tool runs.
- TestDeadline::test_slow_decision_denies_rather_than_overrunning fails here (254 passed, 1 failed), and mypy reports 4 errors in hook.py.
- CI runs only on Ubuntu, so neither shows up.
- janus-hook took 1.3–1.8s per tool call on this machine; the docs say 150–400ms.
- About 500ms of the 690ms import is janus.agent and its tree, which the hook never uses. Lazy-loading JanusAgent in janus/__init__.py (as already done for generate_policy) is a cheap win before the daemon.

2. Design gaps (from reading the code)

- The loop Janus owns has no taint tracking. ToolRegistry.execute calls enforce() with no session and never records output (registry.py:166). JanusAgent, LangChain and ADK all bypass decide_call, so taint, provenance and endorsements only work on the Claude SDK path. Routing every path through decide_call would fix this.
- The SpiceDB engine is a liability as shipped.
  - load, update, allow_tools, block_tools and reset are silent no-ops; a block_tools() that does nothing is dangerous.
  - It ignores arguments and any JSON policy entirely.
  - Unknown tools get a default taint limit of 50.
  - Its enforce() lacks the session parameter the decision core passes.
  - It logs with print() and defaults to the token "somerandomkey".
  - Either deprecate it or rebuild it as an optional layer inside decide_call.
- Policy refinement mixes untrusted data into the rule-writer. refine_policy feeds raw tool output to the LLM that writes policy (generator.py:156), and nothing checks
  the result is narrower than before. An injected page can ask for a w
- Deny reasons help the attacker. A failed custom validator returns its own source code to the model (validator.py:107), and regex denials echo the pattern. Keep the detail in the audit log and send the model a terse reason.
- Library hygiene.
  - configure_logging installs a stdout handler by default (logger.py:120), the same behaviour the shim has to work around.
  - sys.exit and input() live inside the enforcer.
  - The generator keeps global token counters.
  - Audit output is unstructured f-strings.

3. Production-readiness gaps

- Tests: none for agent.py, runner.py, registry.py, the built-in tools (including the path-traversal sandbox), the generator, LangChain, ADK, providers, or PDE. No fuzz or property tests on the enforcer.
- CI:
  - No Windows or macOS runners.
  - No format check; 24 of 51 files fail ruff format --check.
  - No coverage, dependency audit, or CodeQL.
  - Actions are pinned by tag rather than SHA.
  - The live smoke suite is not scheduled, so upstream CLI regressions are caught only when someone runs it by hand.
- Evaluation: there are no attack-success or utility numbers. The plan names AgentDojo and there is an unmerged agentdyn-benchmark branch I did not inspect. Without numbers the project can't make an efficacy claim.
- Open bug: issue #7, JanusLangChainAgent cannot be constructed under LangChain 1.x; the dependency has no upper bound.
- Packaging and process: no py.typed, and SECURITY.md routes reports to one personal address with no supported-versions statement.
- Still-open follow-ups from your own plans: the PostToolUse cross-cherver-aware name resolution on the SDK path.

4. Features worth adding

1. janus policy lint / janus policy test: a strict schema check plus the footgun warnings above, runnable in CI.
2. Shadow mode: log what would be denied without blocking, so teams ca
3. Structured audit log: one JSON record per decision, with decision ID, layer, rule matched and redacted arguments, and pluggable sinks.
4. Policy format versioning: a version field, and serialisable named conditions so SSRF checks don't require Python callables.
5. Session snapshot/restore: needed by the daemon and by any multi-process server; also bound the event lists.
6. The phase-2 daemon and the plugin: brings taint to the CLI path and removes the per-call import.
7. Hardened built-ins: scheme allowlist, redirect re-checking and sizebased command tool as an alternative to shell=True.

5. Suggested order

┌───────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ When  │                                                                                                                                 │
├───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Now   │ Strict loader; fix fallback: 1 in hook paths; fix fetch_url; fail closed on empty generated policy; Windows deadline (watchdog thread); lazy JanusAgent      │
│       │ import; add Windows to CI; release                                                                                                                           │
├───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Next  │ Route registry, LangChain and ADK through decide_call with sessions; structured audit log; lint CLI and shadow mode; tests for built-ins and loader;         │
│       │ scheduled smoke suite; fix issue #7; decide PDE's fate                                                                          │
├───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Later │ Daemon and plugin; AgentDojo numbers with an adaptive attacker; managed-settings verification; signed policies                                               │
└───────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

I'd fix the strict loader first: it is small, and it closes the most s