# Policy Discovery & Enforcement

---

## Motivation

Modern AI agents (coding assistants, planners, autonomous tools) are *over-privileged* by default. They can:

- Read from **untrusted sources** (web pages, public repos, user uploads).
- Call **high-impact tools** (shell, HTTP, SQL, email, file system).
- Be **indirectly hacked** via hidden instructions embedded in data (indirect prompt injection, or IPI).

You cannot realistically prevent an agent from ever **seeing** malicious or adversarial input. Web pages, READMEs, PDFs, and user uploads can all carry hostile instructions that the model will process.

Instead of trying to perfectly sanitize all inputs, this project focuses on:

> **Curbing the impact surface, not the attack surface.**
> We assume the agent will eventually see malevolent prompts. Our goal is to ensure those prompts cannot *translate into dangerous actions*.

We do this by:

- Enforcing **Access Control Lists (ACLs)** *before* capabilities are exposed to the agent.
- Re-checking those ACLs *before* any sensitive tool invocation is executed.
- Using **graph-based authorization** plus **taint tracking** so that permissions degrade as the agent touches untrusted data.

---

## High-Level Architecture

We combine three main ideas:

1. **Relationship-Based Access Control (ReBAC) via SpiceDB (Zanzibar Model)**

   - Agents, roles, tools, and resources are nodes in a **permission graph**.
   - Permissions are modeled as relationships (edges) and derived permissions.
   - Example: `tool:bash_terminal # can_execute @ role:coding_bot`.
2. **Runtime Taint Tracking for Indirect Prompt Injection**

   - As the agent interacts with data sources, we maintain a **session taint score**:
     - 0 = only trusted internal data.
     - 100 = heavily contaminated by untrusted content (web, unknown repos, user uploads).
   - The taint score only **increases** during a session.
   - Policies tie tool access to the current taint level: high-risk tools are automatically disabled in highly tainted contexts.
3. **Policy Discovery from Agent Traces**

   - In a controlled “staging” phase, we observe which tools and resources the agent legitimately uses to complete tasks.
   - We translate those observations into **SpiceDB relationship tuples** (edges) representing the minimal ACL graph required.
   - In production, we **enforce** this discovered graph and block anything outside it.

---

## Curbing Impact Surface with Pre-Tool ACLs

The core security stance is:

> “We constrain what the agent can *do* before it is ever exposed to potentially malicious prompts.”

There are two enforcement layers:

1. **Pre-Exposure ACL (Capability Surfacing)**

   - Before the agent runs, we query SpiceDB to compute the set of tools and resources the agent’s role is allowed to access.
   - We only:
     - Register these tools with the agent framework (LangChain / Progent / ADK), and/or
     - Include them in the system prompt/tool description context.
   - Tools that are not authorized are **structurally invisible** to the LLM—prompt injection cannot ask the agent to use a tool that does not exist in its world.
2. **Pre-Execution ACL (Runtime Check)**

   - If the agent somehow emits a call to a tool (or target) outside the pre-authorized set (e.g., by fabricating a name, or because the framework auto-resolves something), we still gate execution:
     - The interceptor calls SpiceDB with:
       - `resource = tool:<name>`
       - `subject = role:<agent_role>`
       - `context = {current_taint: ..., max_allowed_taint: ...}`
     - If the check fails, the call is blocked and never hits the real system.

This dual-layer approach **curbs the impact surface**:

- Attackers can manipulate text and model reasoning.
- Attackers cannot increase the agent’s pre-authorized capability set or bypass taint-aware safety thresho
