# Architecture

## Overview

Janus sits between the LLM agent and its tools. Every tool call is intercepted, evaluated against the policy, and either allowed or blocked before execution.

## Design Decisions

**Tool-level enforcement**: We enforce at the tool-call boundary, not at prompt or output. Prompt-level filters cannot stop a shell command; they operate on text. This is the same insight as Progent; we implement it with JSON Schema conditions and optional SpiceDB/taint.

**JSON Schema for conditions**: Policies use JSON Schema to restrict arguments. We chose this over a custom DSL because (a) developers already know JSON Schema, (b) it integrates with existing tool definitions (OpenAI, etc.), and (c) validation is deterministic and well-tested. The tradeoff: complex logic (e.g. cross-argument dependencies) requires multiple rules or the SpiceDB engine.

**Two engines**: The default JSON enforcer is stateless and policy-file driven. The SpiceDB engine adds role-based ACLs and taint tracking. We keep them separate because taint requires session state and SpiceDB; many deployments need only the JSON enforcer. The SpiceDB engine is optional and can be omitted at install time.

**Deny by default posture**: Tools not in the policy are blocked. 

## Competitive Landscape

| Tool | Focus | How Janus Differs |
|------|-------|--------------------|
| **Lakera Guard, NeMo Guardrails** | Input/output content filtering | They filter text. Janus enforces at tool execution. A blocked prompt does not stop a tool call; a blocked tool call does. |
| **IronCurtain** | Natural-language policies via MCP | Janus uses machine-readable JSON Schema; deterministic evaluation; no reliance on NL interpretation. |
| **Protect AI, Robust Intelligence** | Pre-deployment scanning, red-teaming | Janus is runtime enforcement. It does not scan; it intercepts every call. |
| **Progent** | Academic prototype; privilege control | Janus is a production implementation with LangChain/ADK adapters, LLM policy generation, and optional SpiceDB/taint. |

## Enforcement Flow

1. Agent produces a tool call (name + arguments).
2. `ToolRegistry` receives the call and invokes `enforcer.enforce(tool_name, arguments)`.
3. Enforcer evaluates rules in priority order. First matching rule wins.
4. If allowed: tool executes; result returns to the agent.
5. If blocked: `PolicyViolation` is raised (or fallback: exit / ask user). The exception can be fed back to the LLM as text so it can adjust.

## Components

| Component | Role |
|-----------|------|
| `JanusAgent` | Entry point. Wires enforcer, tool registry, LLM provider, and runner. |
| `PolicyEnforcer` | Evaluates tool calls against loaded policy. Used by `ToolRegistry`. |
| `ToolRegistry` | Registers tools, dispatches calls, and invokes enforcer before execution. |
| `LLMRunner` | Conversation loop: messages → LLM → tool calls → results → LLM until done. |
| Providers | `janus/llm/providers/` — OpenAI, Anthropic, Google, Azure, Bedrock, Ollama, vLLM, Together, OpenRouter. |

## Threat Model and Guarantees

### In Scope

- **Over-privileged tool calls**: Agent attempts tools or arguments beyond what the task requires.
- **Indirect prompt injection (IPI)**: Malicious instructions in files, web pages, or retrieved data that cause dangerous tool calls.
- **Memory/knowledge poisoning**: Poisoned RAG or memory leading to harmful tool invocations.
- **Malicious tools**: Attacker-introduced tools; policies restrict to an allowlist.
- **Path traversal**: Built-in file tools sandbox to `workspace`; out-of-workspace paths are rejected.
- **Runaway commands**: `run_command` can be constrained via policy (enum, pattern) to an allowlist.

### Out of Scope

- **Preference manipulation**: Attacker tricks the agent to favor a malicious option among valid choices without exceeding least-privilege tool access. Janus cannot distinguish "valid but attacker-preferred" from "valid and user-intended."
- **Text-output attacks**: Exfiltration or harm via the model's text response only (no tool calls). Janus enforces at the tool boundary; it does not filter or constrain text output.
- **Attacks within least privilege**: If the user task legitimately requires a dangerous tool and the policy allows it, Janus will not block it. Policy correctness is the user's responsibility.

### Guarantees

- **Deterministic enforcement**: Policy evaluation is symbolic and reproducible.
- **Default-deny**: Tools not covered by any matching rule are blocked. Unlisted tools are blocked when a policy is loaded.
- **No reliance on LLM trust**: Enforcement happens at the tool-call boundary; it does not depend on the model being resistant to prompt injection.

### Caveats

- **Policy correctness**: Security guarantees hold only if policies are correct. Incorrect or incomplete policies can allow harmful calls.
- **SpiceDB engine**: Taint tracking depends on `update_taint()` being called when the agent reads from risky sources. If integration omits this, taint stays low and high-risk tools may remain accessible.

## Known Limitations

- **Missing arguments**: The JSON enforcer validates only arguments that are present. If the LLM omits an argument that a policy condition restricts, the condition is skipped and the call may be allowed. Fix planned: raise `ArgumentValidationError` when a restricted argument is missing.
- **Taint session reset**: Taint only increases during a session. Long-running services need a way to reset (e.g. per-request sessions). No `reset_session()` yet.
- **SpiceDB unreachable**: If SpiceDB is down, the engine raises a gRPC exception. No timeout, retry, or fail-closed toggle.
- **Schema divergence**: The SpiceDB schema in `main.py` and `schema.zed` have diverged; `schema.zed` is not used at runtime.
- **LLM-generated policies**: Effective but not provably complete. Manual policies can be crafted for provable coverage; LLM-generated ones reduce attack surface but may miss edge cases.

## Project Structure (Post-Merge)

```
janus/
├── agent.py              # JanusAgent
├── exceptions.py
├── llm/
│   ├── base.py
│   ├── runner.py
│   └── providers/
├── policy/
│   ├── enforcer.py       # PolicyEnforcer (JSON Schema)
│   ├── pde_enforcer.py   # SpiceDB engine adapter
│   ├── generator.py      # LLM policy generation
│   ├── loader.py
│   └── validator.py
├── tools/
│   ├── base.py
│   ├── registry.py
│   └── builtin/
├── adapters/
│   ├── langchain.py
│   └── adk.py
└── pde/                  # SpiceDB-backed enforcement (optional)
    ├── enforcement.py
    ├── main.py           # Schema, bootstrap
    └── ...
```
