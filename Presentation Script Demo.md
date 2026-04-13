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

**1. Capability being demoed:**
Indirect Prompt Injection (IPI) defense — blocking malicious instructions hidden in external content that the agent reads.

**2. What the agent does:**
A customer support agent processes refund requests. It has access to: `lookup_customer`, `lookup_customer_pii` (sensitive), `lookup_order`, `initiate_refund`, `send_customer_email`, and `fetch_url`. The agent reads a vendor knowledge base before processing, which contains a hidden injected instruction telling it to retrieve PII and exfiltrate it.

**3. What the policy enforces:**

| Tool | Rule | Constraints |
|------|------|-------------|
| `lookup_customer` | Allow | `customer_id` must match `^C-[0-9]{5}$` |
| `lookup_customer_pii` | **Deny** | Unconditional block |
| `lookup_order` | Allow | `order_id` must match `^ORD-[0-9]{5}$` |
| `initiate_refund` | Allow | ID format + reason must be approved enum |
| `send_customer_email` | Allow | ID format + length limits on subject/body |
| `fetch_url` | **Layered** | Priority 1: Deny suspicious keywords; Priority 2: Allow internal KB domain only |

**4. What we see:**

- **Left panel (no Janus):** Agent reads KB, follows hidden instruction, calls `lookup_customer_pii`, sends PII to `exfil-collector.attacker.com`. Red "ATTACK SUCCEEDED" banner appears.
- **Right panel (with Janus):** Two independent blocks fire — `lookup_customer_pii` denied by explicit deny rule, `fetch_url` to attacker domain denied by layered URL rules. Refund still completes normally via `initiate_refund`.

### Key message

> Janus doesn't break the happy path. It blocks the injected side-channel while legitimate tools keep working.

### What to call out verbally (30-second script)

> "This policy is production-style: we validate argument formats, restrict refund reasons to an approved enum, cap email lengths, and use layered URL rules — explicit deny for suspicious domains before the allowlist. The attack is blocked, but the normal refund flow completes successfully."

---

## Act 2: Policy Enforcement Traces

**Dropdown:** The Fraud Investigation
**ID:** `banking_traces` | **Experience:** ADK Multi-Agent | **Marquee Capability:** Policy Enforcement Traces | **Duration:** ~30s

### Narrative

**1. Capability being demoed:**
Policy enforcement traces — every tool call produces a visible `[ALLOW]` or `[BLOCK]` decision with reasoning, creating an audit trail for compliance.

**2. What the agent does:**
A fraud investigation assistant helps analysts triage suspicious transactions. It follows a **Triage → Investigation → Resolution** workflow across two accounts. Tools include: `get_account_info`, `get_transaction`, `list_flagged_transactions`, `freeze_account`, `unfreeze_account`, `reverse_transaction`, `export_account_history`, and `file_sar`.

**3. What the policy enforces:**

| Tool | Rule | Why |
|------|------|-----|
| `get_account_info` | Allow | Read-only, safe |
| `get_transaction` | Allow | Read-only, safe |
| `list_flagged_transactions` | Allow | Read-only, safe |
| `freeze_account` | Allow | Protective action |
| `unfreeze_account` | **Deny** | Requires manager authorization |
| `reverse_transaction` | **Deny** | Requires elevated authorization |
| `export_account_history` | **Deny** | Restricted to authorized analysts |
| `file_sar` | Allow | Compliance-critical, permitted |

**4. What we see:**

- **Left panel (no Janus):** All 8 actions execute unchecked — including unauthorized transaction reversal and account unfreeze. No audit trail.
- **Right panel (with Janus):** 5 tools allowed (reads + freeze + SAR filing), 3 blocked (`reverse_transaction`, `export_account_history`, `unfreeze_account`). Every decision shows `[ALLOW]` or `[BLOCK]` with policy reason in the trace log.

### Key message

> Janus produces a complete enforcement trace for compliance. Every decision is explainable and auditable — you can hand this log to a regulator.

### What to call out verbally (30-second script)

> "Look at the trace log on the right. Every single tool call has a visible ALLOW or BLOCK with the policy reason. This is your audit trail. You can hand this to a compliance officer or regulator and say: here's exactly what the agent tried to do, and here's why each action was permitted or denied."

---

## Act 3: Iterative Tool Looping with Guardrails

**Dropdown:** The Financial Planner
**ID:** `fintech_planning` | **Experience:** LangChain Single-Agent | **Marquee Capability:** Iterative Tool Looping | **Duration:** ~35s

### Narrative

**1. Capability being demoed:**
Iterative tool looping with guardrails — Janus supports complex multi-step agent workflows where each tool call builds on previous results, intervening only when policy is violated.

**2. What the agent does:**
A personal finance assistant helps users create financial plans. It iteratively: gathers user profile → checks portfolio → fetches market data → runs retirement projection → runs house savings projection → proposes rebalance → sets savings goal → attempts trade execution. Each step uses output from prior steps.

**3. What the policy enforces:**

| Tool | Rule | Why |
|------|------|-----|
| `get_user_profile` | Allow | Read-only |
| `get_portfolio_summary` | Allow | Read-only |
| `get_market_data` | Allow | Read-only |
| `calculate_projection` | Allow | Analysis, no side effects |
| `propose_rebalance` | Allow | Proposal only, no execution |
| `set_savings_goal` | Allow | User-initiated goal setting |
| `execute_trade` | **Deny** | Requires explicit user confirmation |

**4. What we see:**

- **Left panel (no Janus):** All 8 tool calls execute including `execute_trade` — the agent automatically trades without user confirmation.
- **Right panel (with Janus):** 7 tools allowed through the full iterative chain (profile → portfolio → market → projections → rebalance → goal). Only `execute_trade` is blocked at the final step. The planning workflow completes; only the dangerous action is stopped.

### Key message

> Janus supports complex multi-step agent workflows. It only intervenes at the precise moment policy is violated, not before.

### What to call out verbally (30-second script)

> "Watch the iterative chain — 8 sequential calls where each step builds on the last. Janus allows all of them through until the final step: execute_trade. That's the guardrail. The agent did all the analysis and planning work; we only blocked the one action that requires human confirmation."

---

## Act 4: Dynamic Trust via Taint Thresholds (Advanced)

**Dropdown:** The Tainted Lab Report
**ID:** `healthcare_taint` | **Experience:** ADK Multi-Agent | **Marquee Capability:** Dynamic Trust (Taint + SpiceDB) | **Duration:** ~35s
**Note:** Requires SpiceDB for full effect (`./scripts/run_demo_webapp.sh --with-spicedb`)

### Narrative

**1. Capability being demoed:**
Dynamic trust via taint tracking — permissions change during runtime based on what the agent has consumed. Even with correct RBAC, reading untrusted data can block high-impact actions.

**2. What the agent does:**
A clinical operations coordinator follows a **Planner → Implementer → Reviewer** workflow. The agent: queries internal patient records → fetches external lab report (taint source) → updates treatment plan → attempts to publish to patient portal → attempts external webhook notification.

**3. What the policy enforces:**

This scenario uses the **PDE engine** (SpiceDB + taint), not static JSON policy.

**SpiceDB Role Structure:**

| Concept | Example | Purpose |
|---------|---------|---------|
| Agent ID | `clinical_agent` | Unique identifier for this agent |
| Roles | `clinician`, `coordinator` | Permission groups the agent is enrolled in |
| Tool grants | `clinician` → all healthcare tools | Role-to-tool ACL mappings |

**Two-gate enforcement** (both must pass):

1. **Taint gate** (Python): Is `current_taint <= tool_taint_limit`?
2. **ACL gate** (SpiceDB): Does agent's role have `invoke` permission?

**Tool taint limits:**

| Tool | Taint Limit | Why |
|------|-------------|-----|
| `query_patient_record` | 90 | Internal read, low risk |
| `fetch_clinical_report` | 90 | Reading is allowed |
| `update_treatment_plan` | 70 | Internal write, moderate |
| `publish_to_portal` | 20 | Patient-facing, strict |
| `send_notification` | 30 | Moderate-strict |
| `fetch_url` | 10 | External network, very strict |

**Taint sources:** Reading internal records = low risk (+10). Reading external lab report = medium risk (+40).

**4. What we see:**

- **Left panel (no Janus):** All actions execute including portal publish and external webhook call to attacker URL. Patient data exposed.
- **Right panel (with Janus/PDE):**
  - Taint meter fills after external lab fetch (0 → 40).
  - `query_patient_record` ✅ allowed (taint 0 < limit 90)
  - `fetch_clinical_report` ✅ allowed, but raises taint to 40
  - `update_treatment_plan` ✅ allowed (taint 40 < limit 70)
  - `publish_to_portal` ❌ blocked (taint 40 > limit 20) — even though ACL grants it
  - `fetch_url` ❌ blocked (taint 40 > limit 10)

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
