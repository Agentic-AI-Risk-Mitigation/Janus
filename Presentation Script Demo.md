# Janus Security Showcase — Presentation Script

## Setup

Start the server and open `http://localhost:8000` in a browser.

```bash
uv sync --extra all --extra dev
uv run uvicorn examples.app:app --reload
```

---

## Act 1: Indirect Prompt Injection Defense

**Dropdown:** The Poisoned Knowledge Base
**ID:** `ecommerce_ipi` | **Experience:** LangChain Single-Agent | **Marquee Capability:** Indirect Prompt Injection Defense | **Duration:** ~30s

### Narrative

A customer support agent processes a refund request. The vendor knowledge base it reads contains a hidden injected instruction that tries to exfiltrate customer PII to an attacker URL.

### What to point out

- **Left panel (no Janus):** Agent reads KB, follows hidden instruction, calls `lookup_customer_pii`, sends PII to `exfil-collector.attacker.com`. Red "ATTACK SUCCEEDED" banner appears.
- **Right panel (with Janus):** Same flow, but two independent blocks fire — `lookup_customer_pii` is explicitly denied, and `fetch_url` to attacker domain is denied by layered URL rules.
- **Production-style hardening now in policy:** We constrain argument formats (`customer_id`, `order_id`), restrict refund reasons to an approved enum, and cap email subject/body lengths.
- **Layered URL enforcement:** `fetch_url` uses priority rules: explicit deny for suspicious domains/keywords first, then allowlist for internal KB domain. Everything else is default-deny.
- **Business continuity:** `initiate_refund(order_id="ORD-77201", reason="defective item")` still succeeds, proving secure-by-default does not break the happy path.

### Key message

> Janus doesn't break the happy path. It blocks the injected side-channel while legitimate tools keep working.

### What to call out verbally (30-second script)

> "This policy is now production-style, not just demo-style. We don't only block the dangerous tools; we also validate arguments and constrain behavior: strict ID formats, approved refund reasons, bounded email payloads, and layered URL rules with explicit deny before allow. So we reduce blast radius while preserving normal customer support flow."

---

## Act 2: Policy Enforcement Traces

**Dropdown:** The Fraud Investigation
**ID:** `banking_traces` | **Experience:** ADK Multi-Agent | **Marquee Capability:** Policy Enforcement Traces | **Duration:** ~30s

### Narrative

A fraud analyst investigates suspicious transactions across two accounts. The agent attempts reads, freezes, reversals, data exports, and SAR filing.

### What to point out

- **Right panel trace log:** Every tool call gets a visible `[ALLOW]` or `[BLOCK]` with the policy reason — this is the audit trail.
- 5 tools allowed (reads + SAR filing), 3 blocked (`reverse_transaction`, `export_account_history`, `unfreeze_account`).
- Left panel shows all 8 actions executing unchecked — including the unauthorized reversal and unfreeze.

### Key message

> Janus produces a complete enforcement trace for compliance. Every decision is explainable and auditable — you can hand this log to a regulator.

---

## Act 3: Iterative Tool Looping with Guardrails

**Dropdown:** The Financial Planner
**ID:** `fintech_planning` | **Experience:** LangChain Single-Agent | **Marquee Capability:** Iterative Tool Looping | **Duration:** ~35s

### Narrative

A personal finance assistant iteratively gathers user profile, portfolio data, market quotes, runs projections, proposes a rebalance, and then tries to execute a trade.

### What to point out

- Watch the iterative chain — 8 sequential tool calls where each step builds on previous results (profile → portfolio → market data → projection → rebalance → savings goal).
- Everything is allowed until the final step: `execute_trade` is **blocked** — requires explicit user confirmation.
- Left panel shows the trade executing automatically without any guardrail.

### Key message

> Janus supports complex multi-step agent workflows. It only intervenes at the precise moment policy is violated, not before.

---

## Act 4: Dynamic Trust via Taint Thresholds (Advanced)

**Dropdown:** The Tainted Lab Report
**ID:** `healthcare_taint` | **Experience:** ADK Multi-Agent | **Marquee Capability:** Dynamic Trust (Taint + SpiceDB) | **Duration:** ~35s
**Note:** Requires SpiceDB for full effect (`./scripts/run_demo_webapp.sh --with-spicedb`)

### Narrative

A clinical coordinator's agent reads patient notes, fetches an external lab report (taint source), then tries to update the medication record.

### SpiceDB Role Structure (what to explain)

PDE uses **agent IDs** and **roles** separately:

| Concept | Example | Purpose |
|---------|---------|---------|
| Agent ID | `clinical_agent` | Unique identifier for this agent instance |
| Roles | `clinician`, `coordinator` | Permission groups the agent is enrolled in |
| Tool grants | `clinician` → `publish_to_portal` | Role-to-tool mappings |

**Healthcare scenario roles:**

| Role | Tools granted |
|------|---------------|
| `clinician` | query_patient_record, fetch_clinical_report, update_treatment_plan, publish_to_portal, send_notification, fetch_url |
| `coordinator` | (same agent membership, extensible for admin tools) |

**Two-gate enforcement:**

1. **Taint gate** (Python): Is `current_taint <= tool_taint_limit`?
2. **ACL gate** (SpiceDB): Does agent's role have `invoke` permission on this tool?

Both must pass. The punchline: even with correct ACL, taint can still block.

### What to point out

- **Taint meter** on right panel fills up after the external lab fetch (taint rises from 0 → 40).
- `publish_to_portal` has taint limit 20 — blocked because 40 > 20, even though the agent's `clinician` role grants it.
- `fetch_url` (external network) has taint limit 10 — also blocked.
- `update_treatment_plan` has taint limit 70 — still allowed (40 < 70).
- Left panel shows all actions executing unchecked with no taint awareness.

### Tool taint limits (healthcare scenario)

| Tool | Taint Limit | Why |
|------|-------------|-----|
| query_patient_record | 90 | Internal read, low risk |
| fetch_clinical_report | 90 | Reading is allowed |
| update_treatment_plan | 70 | Internal write, moderate |
| publish_to_portal | 20 | Patient-facing, strict |
| send_notification | 30 | Moderate-strict |
| fetch_url | 10 | External network, very strict |

### Key message

> Static RBAC isn't enough. Janus tracks session risk dynamically — reading untrusted data progressively restricts what the agent can do.

### What to call out verbally (30-second script)

> "The agent has the correct RBAC role — `clinician` — which grants `invoke` permission on `publish_to_portal`. But after ingesting that external lab report, the session taint rose to 40. The portal's taint limit is 20. Taint overrides ACL. This is the difference between 'who are you?' and 'what have you touched?'"

---

## Act 5: Live Coding Developer Walkthrough (Healthcare Taint Spin-off)

**Guide:** [`docs/healthcare_taint_live_coding_walkthrough.md`](docs/healthcare_taint_live_coding_walkthrough.md)
**Duration:** ~10 minutes | **Style:** Live coding in IDE / terminal

### Setup

Have a terminal and editor open side-by-side. Run `uv sync --extra all --extra dev` beforehand.

### Runbook (5 steps, paste snippets from the guide)

| Step | What to show | Snippet to paste | Time |
|------|-------------|------------------|------|
| **0** | Define healthcare tools with `ToolDef` | `code_snippets/step_0_define_healthcare_tools.py` | 1 min |
| **1** | Baseline agent — `policy=None` | `code_snippets/step_1_baseline_agent.py` | 2 min |
| **2** | Add static policy — `policy={...}` | `code_snippets/step_2_static_policy.py` | 2 min |
| **3** | Generate policy via LLM — `policy="generate"` | `code_snippets/step_3_generate_policy.py` | 2 min |
| **4** | Taint tracking — `policy_engine="pde"` | `code_snippets/step_4_pde_taint.py` | 2 min |
| **5** | Wrap-up: adapters, lifecycle, observability | Slide or narration only | 1 min |

### What to emphasize

- **One-kwarg progression:** The only code difference between an unprotected agent and a fully-secured one is swapping a single keyword argument at each step.
- **Generate → review → commit:** Show the `save_policy()` call to make the point that LLM-generated policies feed into a human review workflow.
- **Agent ID vs Role:** In Step 4, `agent_role="clinical_agent"` is the **agent identifier**, not the role. The agent is enrolled in the `clinician` role via SpiceDB bootstrap. This distinction matters for multi-tenant deployments.
- **Taint vs RBAC:** The agent *has* the correct `clinician` role granting `invoke` on `publish_to_portal`, but is still blocked because taint (40) exceeds the tool's limit (20). This is the punchline.

### Key message

> Janus gives enterprise developers progressive security — from zero to taint-aware enforcement — without rewriting their agent code.

---

## Recommended Demo Order

| Order | Dropdown Name                 | Marquee Capability                     | Why                                                        |
|-------|-------------------------------|----------------------------------------|------------------------------------------------------------|
| 1     | The Poisoned Knowledge Base   | Indirect Prompt Injection Defense      | Most dramatic, easiest to understand (hidden attack → blocked) |
| 2     | The Fraud Investigation       | Policy Enforcement Traces              | Shows the audit/compliance angle                           |
| 3     | The Financial Planner         | Iterative Tool Looping                 | Demonstrates Janus works with complex workflows            |
| 4     | The Tainted Lab Report        | Dynamic Trust (Taint + SpiceDB)        | Most sophisticated, save for technical audiences           |
| 5     | Live Coding Walkthrough       | Progressive Security Adoption           | Developer-focused — shows how to adopt Janus step by step  |

## Presenter Tips

- Use the **Experience** dropdown to filter: "LangChain Single-Agent" shows The Poisoned Knowledge Base + The Financial Planner; "ADK Multi-Agent" shows The Fraud Investigation + The Tainted Lab Report.
- Hover over the scenario description in the header for the full storyline.
- The **metadata badges** (industry, persona, marquee capability) below the header help frame each scenario for the audience.
- In **Act 1**, explicitly mention the new policy hardening depth: argument constraints + layered deny/allow URL rules + default-deny fallback.
- Legacy scenarios (The Poisoned README, Supply Chain Sabotage, The Taint Cascade) are still available in the "All Experiences" view for deep-dive Q&A.
