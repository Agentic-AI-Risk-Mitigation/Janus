# Demos

Janus ships with a demo web app and CLI runner for scenario-based walkthroughs that compare the same agent flow **without Janus** and **with Janus**. The examples use scripted prompts and mocked tools for reproducible playback, but the enforcement is real: Janus evaluates actual tool calls, arguments, and, for PDE scenarios, live taint and SpiceDB-backed authorization.

The scenarios are ordered to match the presentation script's recommended demo flow: start with the most intuitive product stories, then move into the coding-agent deep dives.

## Demo surfaces

### Web app

The web app presents a split-panel comparison:

- Left panel: same scenario without Janus, where unsafe tool calls succeed.
- Right panel: the protected run, where Janus emits explicit ALLOW/BLOCK decisions.
- Scenario metadata: experience, industry, persona, and marquee capability.
- Taint meter: shown for PDE scenarios that use dynamic trust.

Run it from the repo root:

```bash
uv sync --extra all --extra dev
./scripts/run_demo_webapp.sh
```

If you want the PDE scenarios to run with real SpiceDB enforcement, start the demo with:

```bash
./scripts/run_demo_webapp.sh --with-spicedb
```

Then open [http://localhost:8000](http://localhost:8000).

### CLI

Use the CLI when you want to run a single scenario directly:

```bash
uv run python -m examples.run --list
uv run python -m examples.run ecommerce_ipi --unprotected
uv run python -m examples.run ecommerce_ipi --protected
```

For PDE-backed scenarios, make sure SpiceDB is available before running the protected flow.

## Scenario catalog

| Order | Scenario | Capability / Focus | Engine | What the protected run demonstrates |
|---|---|---|---|---|
| 1 | `ecommerce_ipi` | Indirect Prompt Injection Defense | JSON policy | A poisoned vendor knowledge base tries to make the agent retrieve customer PII and exfiltrate it; Janus blocks the PII lookup tool and non-allowlisted outbound URL. |
| 2 | `banking_traces` | Policy Enforcement Traces | JSON policy | A multi-step fraud workflow mixes allowed and denied actions so the operator can see clear audit-grade allow/block reasoning for every tool call. |
| 3 | `fintech_planning` | Iterative Tool Looping | JSON policy | The agent can gather data, run projections, and propose a plan, but the final trade execution step is blocked until explicit approval exists in policy. |
| 4 | `healthcare_taint` | Dynamic Trust (Taint + SpiceDB) | PDE + SpiceDB | A tainted external lab report is allowed to inform internal treatment updates, but patient-facing and external actions are blocked once session taint exceeds configured thresholds. |
| 5 | `coding_agent_poisoned_readme` | Poisoned README / secret exfiltration | JSON policy | A hidden instruction in `README.md` attempts to read `.env` and exfiltrate it; Janus blocks the sensitive file read and the outbound request. |
| 6 | `coding_agent_supply_chain` | Typosquat package README / malicious file writes | JSON policy | A fake package README tries to coerce the agent into writing a backdoor and poisoning dependencies; Janus blocks `write_file` and `edit_file` because the task is read-only. |
| 7 | `coding_agent_taint_cascade` | External issue taints a coding workflow | PDE + SpiceDB | After ingesting a GitHub issue, the session taint rises and the later `git_push` is blocked because it exceeds the allowed taint threshold for code publication. |

## Choosing a scenario

- `ecommerce_ipi` for the cleanest indirect prompt injection story.
- `banking_traces` for auditability and explainable enforcement.
- `fintech_planning` for guarded multi-step planning.
- `healthcare_taint` for dynamic trust and taint thresholds.
- `coding_agent_poisoned_readme` for secret theft from local files.
- `coding_agent_supply_chain` for supply-chain style prompt injection and malicious writes.
- `coding_agent_taint_cascade` for taint propagation into a risky downstream action.

## Repo layout

- `examples/app.py` powers the FastAPI + WebSocket demo UI.
- `examples/run.py` is the CLI entrypoint.
- `examples/shared/` contains the runner, event model, mocked tools, and scripted LLM support.
- `examples/scenarios/<name>/` contains each scenario's `scenario.py`, `scripts.py`, workspace files, and policy or PDE bootstrap data.
- `examples/docker-compose.yml` starts the local SpiceDB dependency used by PDE demos.

For the underlying scenario framework and setup details, see [examples/README.md](https://github.com/Agentic-AI-Risk-Mitigation/Janus/blob/main/examples/README.md).
