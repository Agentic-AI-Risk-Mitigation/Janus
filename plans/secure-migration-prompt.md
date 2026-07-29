# Handoff prompt: migrate `secure` onto Janus 0.1.0 (Session/provenance API)

Janus (`janus-guard`, repo at `~/projects/archive/aisc/Janus`) has shipped four phases of
new capability specifically designed around this repo's needs. Your job is to migrate
`secure`'s outreach pipeline onto it. Read `plans/ipi-expansion-design.md` in the Janus
repo first (especially §4, the acceptance test — it IS this migration), plus the
`[Unreleased]` section of its CHANGELOG. The relevant Janus commits are `d1a8dab` through
`a5c2938` on `main`. All 165 Janus tests pass; `janus.testing`, `Session`, provenance
conditions, `mark_untrusted`, and `janus.checks` are the new public API.

**Dependency note first:** Janus 0.1.0 is only in the local repo — it is NOT on PyPI yet.
Point `secure`'s environment at the local path (editable install or a path requirement)
before anything else, and verify `python -c "import janus; print(janus.__version__)"`
prints `0.1.0`. Also: 0.1.0 no longer depends on `openai`, `jinja2`, or `python-dotenv`,
and no longer runs `load_dotenv()` at import time.

## Migration steps, smallest-diff first

### 1. Tests off the private API (no behavior change)

`tests/test_reply_agent_policy.py` and `tests/test_registrar_agent_policy.py` both import
`_decide` from the adapter. Replace with the public harness:

```python
from janus.testing import decide
d = decide(TOOL_POLICY, "fetch_page", {"url": ...}, required_args=_REQUIRED_ARGS)
assert d.denied and "blocked URL" in d.reason
```

`decide()` runs the exact decision core the PreToolUse hook runs (same defaults:
mcp-prefix stripping, StructuredOutput passthrough) and returns a `Decision` with
`allowed/denied/reason/layer` (layer ∈ passthrough/taint/required_args/rules — use it to
assert *why*). `janus.testing.replay(policy, [(tool, args, ALLOW|DENY), ...])` covers
sequence-style tests. `_decide` still works (thin delegate) so this can land as its own
commit.

### 2. `outreach/llm.py` — one parameter

Add `session=None` to `agent_structured()` (and `_aagent`) and forward it to
`janus_options(..., session=session)`. That is the whole change; `llm.py` stays the single
SDK seam. With a session, `janus_options` wires a `PostToolUse` hook that records every
tool output into the session (MCP content blocks are unwrapped via
`unwrap_tool_response()`, so provenance extractors see the dict the tool body returned)
and exposes the session to policy conditions at `PreToolUse`.

### 3. `outreach/reply_agent.py` — the real payoff

This replaces the hand-rolled `_ResearchSession.allowed_urls` control with an enforced,
audited, hook-level policy condition.

```python
from janus.checks import check_output, echoed_untrusted_values, extract_urls
from janus.policy import Session, all_of, from_output, normalize_url, not_in, untrusted_set

def _search_urls(out: dict) -> list[str]:
    return [r.get("url") for r in (out.get("results") or []) if r.get("url")]

def _make_session() -> Session:
    s = Session()
    s.provenance.collect("web_search", label="searched_urls",
                         extract=_search_urls, normalize=normalize_url)
    return s

TOOL_POLICY = {
    "web_search": [(1, 0, {"query": {"type": "string", "maxLength": 400}}, 0)],
    "fetch_page": [(1, 0, {"url": all_of(policy_url_ok,
                                         from_output("searched_urls"),
                                         not_in(untrusted_set("inbound_email")))}, 0)],
}
```

Concretely:

- In `draft_reply()` (web path): `session = _make_session()`, then
  `session.mark_untrusted(inbound, label="inbound_email", extract=extract_urls,
  normalize=normalize_url)`, then pass `session=session` through `agent_structured`.
  Build ONE session per draft — never share or reuse across drafts, or one inbound's
  taint/URL-sets gate the next draft.
- **Delete `_ResearchSession` entirely.** The tool bodies become the stateless
  module-level `web_search`/`fetch_page`; the allowlist check and its soft
  `{"ok": False, "reason": ...}` return go away. A disallowed fetch is now a hook-level
  `permissionDecision: "deny"` — pre-execution, audited in `session.events`, not a tool
  result. Keep the SSRF/redirect/byte-cap hardening inside `fetch_page` — that is
  belt-and-braces at execution time and stays.
- The `not_in(untrusted_set("inbound_email"))` member closes a gap the old allowlist had:
  an inbound-planted URL can no longer be laundered by searching for it and fetching it
  as a "search result".
- In `run()`, replace `echoed_inbound_links` with:

  ```python
  findings = check_output(draft["reply_final"], session,
                          checks=[echoed_untrusted_values("inbound_email")])
  for f in findings:
      draft.setdefault("safety_flags", []).append(f.message)
  ```

  Same semantics (flag, don't block — draft-for-human-review), but matching goes through
  `normalize_url` on both sides, and findings report values as written in the draft.
  Keep `_extract_urls`/`echoed_inbound_links` deleted, not aliased — `extract_urls` from
  `janus.checks` replaces them (same regex/trim behavior; verify against the existing
  tests before deleting).
- The `--no-web` path (`llm.structured`, tool-free): still create the session, still
  `mark_untrusted`, still run `check_output` on the result. That path previously had zero
  Janus coverage; the output check is its first.
- `_REQUIRED_ARGS` stays as-is (still belt-and-braces for blank strings).

### 4. `outreach/registrar_agent.py` — optional, judge the fit

Same pattern is available for `fetch_url` (RDAP-seeded URL sets via
`session.provenance.collect("rdap_domain", ...)` on the abuse-policy links), but the flow
seeds RDAP deterministically into the prompt and fetches few URLs — decide whether the
added coupling pays. If you skip it, say so in the PR; don't half-wire it.

### 5. Dependency cleanup

- Bump/point the `janus-guard` requirement (local path until published).
- Remove the defensive `openai` pin from `requirements.txt` IF its only reason was
  Janus's transitive dependency (check `openai_llm.py` still gets openai from somewhere
  legitimate if that module is still used — it has its own real openai usage, so likely
  the pin stays but the *comment* about Janus changes).
- The `_ensure_repo_key` docstring in `openai_llm.py` and `_ensure_env_loaded` in
  `llm.py` document Janus's import-time `load_dotenv()` footgun. That footgun is gone in
  0.1.0. Do NOT delete the override mechanisms (they're still correct hygiene and the
  test suite depends on the `OUTREACH_*_ENV_FILE=0` switches) — but update the
  docstrings so they describe history, not a live hazard.

## Things to know / not do

- **Do not gate the read tools on taint.** The drafting session is tainted from turn
  zero by `mark_untrusted` — by design. Taint gates are for sinks; this agent has none
  yet. The active defenses here are provenance conditions and output checks. If a send
  tool ever enters the loop, gate it `"*"` and use `session.endorse_event(audit_id,
  by=..., reason=...)` as the human-approval step — deny reasons carry `(audit id ...)`
  suffixes for exactly this.
- `janus_options(taint=...)` now emits a DeprecationWarning (`secure` never used it —
  just don't introduce it).
- Provenance membership is exact-match after `normalize_url` (lowercased scheme/host,
  default ports stripped, fragments/userinfo dropped, path case PRESERVED). If live runs
  show the model retyping URLs in ways that miss (e.g. dropped query strings), report it
  — don't loosen the normalizer unilaterally.
- Known residual, unchanged from the old design: a URL the attacker gets into search
  results (SEO/planted) is fetchable. The migration changes enforceability and audit,
  not that trust decision.
- New policy tests to add: search-then-fetch allowed; unsearched URL denied (assert
  `"searched_urls"` in reason); inbound-planted URL denied even after it appears in
  search results (the `not_in` member); fresh session fully closed; simulate prior
  outputs with `session.record_output("web_search", {...})` in tests.

## Validation

Run `secure`'s full test suite; run one real draft end-to-end (use a saved inbound, e.g.
the werunrome case) and eyeball `session.events` — it should read as one ordered story:
`mark_untrusted` → search `collect` → any `miss` denials → output findings. Confirm the
model, when denied a fetch, gets the deny reason as feedback and routes to search instead
of stalling. List validation performed in the PR per the repo's conventions.
