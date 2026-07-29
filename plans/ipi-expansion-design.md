# Janus IPI expansion — design proposal

Status: proposal, 2026-07-29. Responds to the handoff prompt (audit findings verified
against both repos; deviations flagged in §8). No code in this document is final API —
it is concrete enough to critique, which is the point.

---

## 1. Thesis

Janus should not compete with CaMeL or FIDES, and should stop implicitly framing itself
as a general policy engine. Its defensible position is the **practical retrofit rung**:
a deterministic, fail-closed reference monitor for agent loops the integrator does *not*
own — today, concretely, the Claude Agent SDK's vendor subprocess — where enforcement
must live entirely at pre/post tool-call hook seams. Nothing in the current field
occupies that rung well: research systems require owning the loop (CaMeL, FIDES) or a
custom interpreter/Datalog runtime (FORGE); Progent is the closest architectural
neighbor but is a research artifact, not a maintained library with a lockdown story for
its own seams being skipped; Invariant-style trace policies have flow operators but — per
their own docs — no way to require that an argument *value* originated from a specific
earlier tool output. That last capability is exactly what Janus's one real consumer
hand-rolled (`_ResearchSession.allowed_urls`) because Janus could not express it. The
expansion, therefore, is not "more policy features"; it is **session-scoped value
provenance at the hook seams**, plus the four coverage gaps the consumer audit exposed
(prompt-borne taint, tool-free structured calls, output-side checks, a public test
harness), shipped behind a core that stays stdlib + jsonschema + pydantic. Janus wins by
being the thing you can actually deploy on a loop you don't control, with an audit trail,
tests you can write against the deployed semantics, and honest documentation of what it
cannot see.

## 2. Where the state lives (the design commitment, restated)

"Stateless enforcer, no global state" conflates two claims worth separating:

- **No global state** — no module-level mutable state; every object independent and
  concurrency-safe. This commitment is kept absolutely.
- **Stateless enforcer** — `PolicyEnforcer.enforce()` is a pure function of (policy,
  tool, args). This is relaxed *explicitly*: cross-call provenance is inherently
  stateful, and the state lives in a new **`Session`** object — created per agent run by
  the integrator, passed explicitly into the adapter (`janus_options(session=...)`),
  threaded into conditions via a read-only context parameter, resettable only at session
  boundaries. This is the same pattern `TaintTracker` already established; `Session`
  generalizes it. The enforcer itself remains stateless: it *reads* session state handed
  to it per call and never owns or mutates it (all mutation happens at the PostToolUse
  seam or via explicit integrator calls).

```python
class Session:
    """Per-agent-run enforcement state. One per run; reset() at boundaries only."""
    taint: TaintTracker              # existing per-source labels, unchanged
    provenance: ProvenanceLedger     # new: named value-sets derived from outputs
    endorsements: EndorsementLog     # new: audited declassifications (§3.4)

    def mark_untrusted(self, text, *, label, extract=None): ...   # §3.3
    @property
    def events(self) -> list[dict]: ...   # merged, ordered audit trail
    def reset(self) -> None: ...
```

## 3. Phased plan

Each phase is independently shippable and independently useful. Ordering is by
dependency (1 is a prerequisite of 2) and by what unblocks `secure` (0 removes the two
documented adoption footguns immediately; 2 is the acceptance test).

### Phase 0 — hygiene + public test harness (no new enforcement capability)

Unblocks adoption and costs days, not weeks. Ship as **0.1.0** with migration notes.

1. **Kill the import-time `load_dotenv()`** in `janus/policy/generator.py:21-22`. No
   library may mutate its consumer's process environment as an import side effect —
   `secure` documents two separate debugging passes this cost (`openai_llm.py`
   `_ensure_repo_key`, `llm.py` `_ensure_env_loaded`). Callers of `generate_policy()`
   manage their own environment.
2. **Make the generator lazy and its dependency optional.** `janus/__init__.py:96`
   imports the generator eagerly, which is why *every* Janus import drags it (and
   `openai`) in. Move `generate_policy`/`refine_policy` behind a module-level
   `__getattr__`, and move `openai>=1.0,<2.0` from core deps to a `generate` extra.
   Core install becomes what CLAUDE.md already claims it is.
3. **Public policy-test harness — `janus.testing`.** Both `secure` policy test files
   import `_decide` because there is no public way to assert "this call is denied."
   Promote the exact decision core the hook runs:

   ```python
   from janus.testing import decide, replay

   d = decide(TOOL_POLICY, "fetch_page", {"url": "http://169.254.169.254/"},
              required_args=REQUIRED, session=None)
   assert d.denied and "non-public" in d.reason

   replay(TOOL_POLICY, [                       # the test_ipi_scenarios idiom, public
       ("web_search", {"query": "CVE-2023-23752"}, ALLOW),
       ("fetch_page", {"url": "https://evil.example/"}, DENY),
   ], session=session)
   ```

   `decide()` is a thin public wrapper over the same `_decide` path the PreToolUse hook
   uses (passthrough tools, name resolution, required_args, taint, provenance), so a
   green test proves the *deployed* semantics, not a parallel reimplementation.
   `Decision` carries `allowed/denied/reason` plus which layer denied (taint gate,
   required_args, rule evaluation) for precise assertions.

### Phase 1 — richer condition contract (core only)

The expressiveness boundary the audit traced: `validate_argument`
(`janus/policy/validator.py:74`) calls `restriction(value)` with one scalar. Fix the
contract, not the call sites:

```python
from janus.policy import ConditionContext, context_condition, all_of, any_of

@dataclass(frozen=True)
class ConditionContext:
    tool_name: str
    arg_name: str
    arguments: Mapping[str, Any]     # read-only view of the full call
    session: Session | None          # None when no session is wired

@context_condition                    # explicit opt-in marker, no signature sniffing
def url_from_search(value: str, ctx: ConditionContext) -> bool:
    return ctx.session is not None and ctx.session.provenance.contains("searched_urls", value)
```

- **Detection is explicit.** `@context_condition` sets `__janus_context__ = True`;
  the validator calls marked restrictions as `restriction(value, ctx)` and everything
  else as `restriction(value)`. Zero behavior change for every existing callable —
  `policy_url_ok` in `secure` keeps working untouched. Signature inspection was
  considered and rejected: it breaks under `functools.wraps`/partials and turns a
  contract into a heuristic.
- **Composition.** Conditions today are one restriction per argument; provenance almost
  always composes with an existing check (SSRF *and* came-from-search). `all_of(*rs)` /
  `any_of(*rs)` return context-aware callables that dispatch each member by its own
  contract. Fail-closed: `all_of` short-circuits on first failure and propagates the
  failing member's message.
- **Enforcer seam.** `enforce(tool, args)` grows `enforce(tool, args, *, session=None)`.
  Backwards compatible; when `session is None`, context conditions still run (with
  `ctx.session=None`) and must decide accordingly — the provenance factories below
  deny in that case (fail closed), by design.
- **Failure semantics unchanged**: truthy = allow, falsy/raise = deny; strict-condition
  handling of absent arguments is untouched.

### Phase 2 — Session, ProvenanceLedger, adapter wiring (the flagship)

**ProvenanceLedger** — named value-sets derived from tool outputs at the post-execution
seam, queried by conditions at the pre-execution seam:

```python
session = Session()
session.provenance.collect(
    "web_search",                       # source tool (policy key, post-resolve_name)
    label="searched_urls",
    extract=lambda out: [r.get("url") for r in (out.get("results") or [])],
    normalize=normalize_url,            # optional; default: identity
)
# At PostToolUse the adapter calls session.record_output(tool, output):
#   -> TaintTracker.record_output(...)  (existing behavior)
#   -> ProvenanceLedger.record(...)     (runs matching collectors, adds values)
# At PreToolUse, conditions query:
session.provenance.contains("searched_urls", value)   # membership, normalized
```

Condition factories in `janus.policy.provenance` (both context conditions):

- `from_output(label)` — allow iff the value is a member of the named set. An empty or
  missing set denies; a missing session denies. **Positive provenance**: "this argument
  must be a value a prior `web_search` actually returned."
- `not_in(label)` — deny iff the value is a member. **Negative provenance**, used with
  untrusted-seeded sets (§3.3): "this argument must not be a value that appeared in the
  inbound email."

Design points, argued:

- **Value-set membership, not dataflow.** At hook granularity Janus cannot prove which
  bytes influenced which argument (the documented `TaintTracker` rationale). What it
  *can* prove is exact-match provenance: the argument value literally appeared in a
  prior output of a named tool. That is precisely the invariant `allowed_urls` enforces
  today, it is sound (no over-claim), and it is the retrofittable version of
  capability-tagged outputs. Fuzzy/derived-value matching is explicitly out (§7).
- **Extractors are the consumer's.** Janus does not guess output shapes. For the Claude
  Agent SDK, `tool_response` arrives as MCP content blocks; the adapter ships
  `unwrap_tool_response()` (pull text blocks, attempt `json.loads`) so extractors see
  the dict the tool body returned, and collectors that raise are logged and treated as
  "collected nothing" — a broken extractor must not grow an allow-set (fail closed on
  the allow side; for `not_in` deny-sets a broken extractor is logged loudly since
  there it fails *open*, and `mark_untrusted` extractors therefore re-raise by default).
- **Normalization is opt-in and visible.** Exact string match is a strength against
  confusion attacks but a utility risk (trailing slash, percent-encoding differences
  between what search returned and what the model retypes). A provided `normalize_url`
  helper (scheme/host lowercase, default port strip, trailing-slash policy) is applied
  to both sides when supplied. The audit trail records the normalized form.

**Adapter wiring.** `janus_options(policy, mcp_servers=..., session=session)` (also on
`janus_hooks`/`janus_pretooluse_hook`). `taint=` remains as a deprecated alias that
constructs a `Session` wrapping the tracker — no `secure` code breaks. The PostToolUse
hook already exists and already receives full outputs; it gains one call
(`session.record_output`). Denied calls still record nothing.

**Audit trail.** Every collection, membership miss that caused a deny, taint event, and
endorsement lands in `session.events`, ordered, with the causing `tool_use_id` where the
SDK provides one. This is what the hand-rolled version lacks entirely.

### Phase 3 — prompt-borne taint, endorsement, output-side checks

#### 3.3 Seeding from prompt-borne untrusted input

The audit's finding 4: the most hostile input in `secure` (the inbound email) enters via
the *prompt*, which `record_output` never sees — a wired tracker would report a clean
session. First-class seam:

```python
session.mark_untrusted(inbound_text, label="inbound_email", extract=extract_urls)
```

Effects: (a) `taint.taint("inbound_email", reason="prompt")` — gates fire exactly as if
an untrusted tool had been read; (b) if `extract` is given, the extracted values seed a
provenance set named `untrusted:inbound_email` for `not_in(...)` conditions and for the
output checks below. This also answers the literature's "provenance assignment
under-specification" complaint in the only honest way available: Janus cannot *detect*
that pasted content is untrusted; it gives the integrator a one-line, audited way to
*declare* it at the point they already know it (in `secure`, `_build_user_prompt` — the
call site that writes the `<<<UNTRUSTED_INBOUND_BEGIN>>>` fence).

#### 3.4 Endorsement / declassification

Blocking is the wrong primitive for in-the-loop tasks where untrusted data legitimately
drives the action. The primitive that keeps gating usable:

```python
session.endorse(tool="send_email", arg="to", value="ops@example.com",
                by="evan", reason="operator-verified out of band", uses=1)
```

Semantics, deliberately narrow:

- An endorsement is an exact **(tool, arg, value)** triple (value optional only for
  whole-tool taint-gate lifts, which additionally require `scope="taint"` to be loud).
  Never "un-taint the session" — taint stays monotonic; an endorsement is checked *at
  the gate/condition*, satisfies exactly the deny that names its triple, and nothing
  else.
- **Consumable**: `uses=1` default; each consumption is an audit event. A standing
  endorsement (`uses=None`) is allowed but logged at warning level on every use.
- **Human-in-the-loop shape**: gate/provenance deny reasons carry a stable `event_id`;
  `session.endorse_event(event_id, by=..., reason=...)` endorses precisely the denied
  triple. An integrator surfaces the denial (CLI prompt, dashboard) between runs or
  turns; Janus does not build the UI.

This is endorsement in the Biba sense (integrity declassification), value-scoped so the
audit trail can answer "who approved sending to this address, when, and why."

#### 3.5 Output-side checks — `janus.checks` (the text-to-text gap, and `structured()` coverage)

Action mediation gives nothing when the harm is output text (audit finding 6; the
literature's "expanding gap"), and `secure`'s largest LLM surface is tool-free
`structured()` calls Janus never sees. Janus should *not* grow an injection classifier
or wrap provider calls. What it can honestly offer: **deterministic, provenance-aware
output assertions** sharing the session's value-sets:

```python
from janus.checks import check_output, echoed_untrusted_values, values_grounded_in

findings = check_output(
    draft["reply_final"], session,
    checks=[
        echoed_untrusted_values("inbound_email", extract=extract_urls),
        values_grounded_in(extract=extract_urls, allowed=["searched_urls"]),
    ],
)
```

- `echoed_untrusted_values(label, extract)` — generalizes `echoed_inbound_links`
  (`reply_agent.py:137`): any extracted value from the output that appears in the named
  untrusted set. `values_grounded_in` is its positive twin: every extracted value must
  be a member of an allowed provenance set (every URL in the draft must be one research
  actually surfaced).
- `check_output` returns `Finding` records (check, severity, message, offending
  values); non-blocking by default because the consumer's shape is draft-for-human-
  review. `enforce=True` raises `OutputViolation` for pipelines that act on the output
  mechanically — which is exactly the `structured()`-then-act shape (the recipient gate,
  contact extraction): seed the session from the untrusted input, run the tool-free
  call as today, `check_output(result, session, enforce=True)` before acting.
- Findings append to `session.events` — one audit trail across input marking, tool
  provenance, gates, and output checks.

This is a *library of backstops*, not a policy engine for prose, and the docs must say
so. It covers the tool-free shape without Janus wrapping any provider call.

### Phase 4 — evaluation (§6)

Ordered last as a phase but started early: the AgentDojo harness can be built against
Phase-2 APIs while Phase 3 lands.

## 4. `allowed_urls` as the acceptance test

The policy under this design, in `secure/outreach/reply_agent.py`:

```python
from janus.policy import Session, all_of
from janus.policy.provenance import from_output, normalize_url
from .url_guard import policy_url_ok

def _search_urls(out: dict) -> list[str]:
    return [r.get("url") for r in (out.get("results") or []) if r.get("url")]

def make_session() -> Session:
    s = Session()
    s.provenance.collect("web_search", label="searched_urls",
                         extract=_search_urls, normalize=normalize_url)
    return s

TOOL_POLICY = {
    "web_search": [(1, 0, {"query": {"type": "string", "maxLength": 400}}, 0)],
    "fetch_page": [(1, 0, {"url": all_of(policy_url_ok,
                                         from_output("searched_urls"))}, 0)],
}
```

Migration in `secure`:

1. `draft_reply()` builds `session = make_session()`; optionally
   `session.mark_untrusted(inbound, label="inbound_email", extract=_extract_urls)`.
2. `llm.agent_structured(...)` grows a `session=` parameter forwarded to
   `janus_options()` — one line in `llm.py`, preserving it as the single SDK seam.
3. `_ResearchSession` is **deleted**. The tool bodies become the stateless module-level
   `web_search`/`fetch_page`; the allowlist check and its soft
   `{"ok": False, "reason": ...}` return are gone. A disallowed fetch is now a
   hook-level `permissionDecision: "deny"` — enforced before execution, logged in
   `session.events`, not a tool result the model can reinterpret.
4. `run()` replaces `echoed_inbound_links` with
   `check_output(..., checks=[echoed_untrusted_values("inbound_email", ...)])`, keeping
   the same safety-flag behavior.
5. `tests/test_reply_agent_policy.py` drops the `_decide` import for `janus.testing`:

   ```python
   session = make_session()
   session.record_output("web_search", {"ok": True, "results": [{"url": GOOD}]})
   assert decide(TOOL_POLICY, "fetch_page", {"url": GOOD}, session=session).allowed
   assert decide(TOOL_POLICY, "fetch_page", {"url": EVIL}, session=session).denied
   assert decide(TOOL_POLICY, "fetch_page", {"url": GOOD}, session=Session()).denied  # empty set
   ```

What the migration does **not** change, stated honestly: if an attacker gets their URL
into search results (SEO poisoning, a planted result), it becomes fetchable — exactly as
with the hand-rolled version. The trust decision ("search results are an acceptable
provenance root") is unchanged; what changes is that it is now enforced pre-execution,
auditable, testable, and expressible in one declarative line.

## 5. Backwards compatibility

- Policy dicts, tuple rules, loader shorthand: unchanged. All new condition power is
  additive (callables/factories).
- `janus_options()`/`janus_hooks()` signatures: `session=` added; `taint=` deprecated
  alias (warns, wraps into a Session); everything else untouched.
- `enforce()` gains keyword-only `session=None`.
- 0.1.0 breaking changes are limited to Phase 0's packaging (openai → `generate` extra,
  lazy generator import) with CHANGELOG migration notes; `secure` is unaffected (it
  never uses the generator).

## 6. Evaluation plan

**What `tests/test_ipi_scenarios.py` establishes today, honestly:** seven offline replay
tests over hand-authored tool-call sequences. They prove the *monitor's semantics* —
that these rules deny these recorded calls and allow their benign twins, including the
missing-argument bypass regressions. They prove nothing about defense efficacy: no model
is in the loop, the attack sequences were written by the defense's authors, and there is
no utility measurement beyond the benign twins. They are regression tests, and we keep
them (extended with provenance/endorsement scenarios), but they justify no external
claim.

**Benchmark:** [AgentDojo](https://arxiv.org/abs/2406.13352) — the standard IPI suite,
the one Progent reports on, with pluggable defenses. Build a Janus defense adapter
(enforcer + Session in the AgentDojo pipeline; per-suite policies hand-written, since
automatic policy generation is explicitly not in scope). Report, per suite:
attack success rate (ASR) under injection, and benign-task utility, defended vs
undefended, same model.

**Adaptive attacker:** the field's methodological error was static-only evaluation. Run
a white-box adaptive pass — the attacker knows the exact policy, provenance rules, and
gate configuration — across three attack classes, reported separately:

1. **Within-policy argument abuse** — malicious calls that satisfy every condition.
2. **Provenance laundering** — getting attacker-controlled values into an allowed
   provenance set (planted search results; a compromised RDAP mirror). This is the known
   residual of `from_output` and must be measured, not hidden.
3. **Sink-free / text-to-text harms** — payloads whose effect is output text, measured
   against `janus.checks` coverage (and expected to show the gap: checks catch value
   echo, not persuasion).

**Numbers that would justify the claims:** on gated-sink suites, static ASR reduced to
low single digits (Progent's verified figure is 39.9% → 1.0% on AgentDojo; matching its
order of magnitude is the bar), adaptive ASR under ~5% for classes 1–2, with utility
degradation ≤ 5 points. For class 3 we publish the measured gap rather than a defense
claim. Additionally: replay `secure`'s real scenarios (the werunrome/JCE reply flow)
through the migrated policy as a live smoke, and re-run the pinned-version SDK smoke
suite — the fail-closed layering is an empirical per-version fact, not a theorem.

## 7. Not building

- **An injection detector/classifier.** Probabilistic, adversarially brittle, and it
  would dilute the deterministic-monitor identity. The `classify=` hook and
  `janus.checks` let integrators compose one in.
- **A policy DSL / trace language** (Invariant-style). Python callables + dicts already
  express everything the consumer needed; a DSL is a parser, a spec, and a second
  security surface to maintain. Revisit only if a second real consumer demands
  serializable cross-call rules.
- **Byte-level / model-internal dataflow.** Unobservable at hook seams; claiming it
  would be dishonest. Value-set membership is the sound approximation.
- **Formal non-interference claims** (FIDES). Cannot hold at source-granular seams;
  we do not pretend.
- **Owning the agent loop** for the SDK adapter (CaMeL-style interpreter). The entire
  point of Janus's rung is that it doesn't get to.
- **Confidentiality labels.** Integrity first; secrecy tracking (what may flow *out*
  per-datum) is a different lattice and a different project until integrity provenance
  has proven itself.
- **PDE/SpiceDB unification.** Already documented direction: `TaintTracker`/`Session`
  supersedes for new work; PDE stays as-is. Merging them now buys nothing and risks the
  core-stays-light constraint.
- **LangChain/ADK provenance parity in this arc.** Both lack a post-execution seam in
  the adapter today; that is its own follow-up, and pretending Session works there
  without `record_output` wiring would ship a silent no-op.
- **A human-approval UI.** Endorsement gives the primitive and the audit trail;
  surfaces belong to integrators.

## 8. Where I disagree with (or would sharpen) the audit

1. **"Fails by returning `{"ok": False}` that the model can reinterpret rather than a
   hook-level deny it cannot route around" (finding 3) — overstated.** The tool body's
   refusal is not actually routable-around: the fetch never happens, and no
   reinterpretation changes that. The real defects are different and sufficient:
   the control is unaudited, untested by the policy tests, invisible to Janus's audit
   trail, burns model turns on soft failures, and — the audit's strongest point — is
   *inexpressible* in the policy layer, so it can never be reviewed as policy. The
   migration's value is enforceability-at-the-seam, auditability, and testability; it
   closes no live bypass, and the proposal should not claim otherwise.
2. **Findings 4 and 5 compose into a sharper statement than either makes alone:**
   wiring `taint=` into `secure` today would be a no-op *and* seeding prompt-borne taint
   (finding 4's fix) would **also** be a no-op, because there is no sink to gate
   (finding 5). The near-term value in `secure` is provenance conditions and output
   checks, not taint gates. Gates become load-bearing only if/when a send/write tool
   enters an agent loop — which the registrar agent's docstring explicitly anticipates
   as "a separate, deliberate decision." The design still ships prompt seeding now,
   because it is the prerequisite for that day and for the output checks today.
3. **"ChainCaps" is unverifiable.** Web search finds no system by that name; either it
   is very recent and unindexed, misnamed, or synthesized. Its described design point
   (positive capability assertions on outputs, monotonic attenuation) is directionally
   what §3.2's ledger does, but the proposal cites no such system. Progent's headline
   numbers I did verify ([arXiv 2504.11703](https://arxiv.org/abs/2504.11703): 39.9% →
   1.0% AgentDojo; adaptive-resilience is claimed in the paper, though I could not
   confirm the audit's specific 4.2%/2.6% figures). CaMeL, FIDES, RTBAS-style screeners,
   and Invariant Guardrails check out as described to the best of available sources.
4. **"Hooks must fail closed — existing, verified behavior" needs a version
   qualifier.** It is an empirical fact verified on the pinned pair (SDK 0.2.120 + CLI
   2.1.218), with upstream history of the opposite (hook-timeout fail-open on older
   versions, per claude-agent-sdk-python #304). The smoke suite exists precisely
   because this is a per-version property. The design constraint stands; the word
   "existing" is doing load-bearing work it should share with "re-verified on every
   bump."
5. **Minor confirmations, for the record:** five allow rules total across both agents ✓
   (2 reply + 3 registrar); `_decide` imported by both policy test files ✓
   (`tests/test_reply_agent_policy.py:21`, `test_registrar_agent_policy.py:21`);
   import-time `load_dotenv()` ✓ (`generator.py:21-22`, pulled in via
   `janus/__init__.py:96`); `openai>=1.0,<2.0` in core deps ✓ (`pyproject.toml:32`);
   both footguns documented in `secure` ✓ (`openai_llm.py` `_ensure_repo_key`,
   `llm.py` `_ensure_env_loaded` — the latter also documents the cwd-dependence).

## 9. Immediate next steps (if this proposal is accepted)

1. Phase 0 PR: generator lazy-import + dotenv removal + `generate` extra + CHANGELOG
   migration note (small, self-contained, mostly deletion).
2. Phase 0 PR: `janus.testing` with `decide`/`replay`/`Decision`; migrate
   `tests/test_ipi_scenarios.py` to it as the first consumer; PR to `secure` dropping
   the `_decide` imports.
3. Phase 1 PR: `ConditionContext` + `@context_condition` + `all_of`/`any_of` in
   `janus/policy/`, `enforce(session=)` threading, validator dispatch.
4. Phase 2 in two PRs: core (`Session`, `ProvenanceLedger`, provenance factories,
   `normalize_url`) then adapter (`session=` wiring, `unwrap_tool_response`, docs).
   Acceptance: the §4 migration lands in `secure` and its policy tests go green against
   the public harness.
