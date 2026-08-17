# Janus

Janus is a system-level security layer for LLM agents. It intercepts every tool call an agent makes and enforces a security policy before execution. Policies restrict agents to only the tools and arguments needed for a task, following the principle of least privilege.

**In one sentence:** Janus enforces at the tool-call boundary so prompt injection and overprivileged agents cannot execute dangerous actions even when the model is compromised.

**Use cases**: Pentesters assessing AI-integrated systems; red teamers testing agent attack surface; AppSec teams deploying coding agents in CI/CD; developers building LangChain/ADK agents who need a policy enforcer for code and tool execution.

## Why Janus

Coding agents (Claude Code, Cursor, LangChain, Google ADK) read untrusted files and execute tools. They have no innate ability to distinguish legitimate instructions from malicious ones embedded in data. Janus enforces at the **tool execution boundary**: regardless of what the model thinks it should do, only policy-allowed calls run. Filtering text does not stop a shell command; blocking the tool call does.

## Key Concepts

- **Privilege control at tool level**: Enforcement happens at the tool-call boundary. The LLM decides what to call; Janus decides whether the call is allowed.
- **Deterministic enforcement**: Policy evaluation is symbolic and reproducible. It does not rely on the model being trustworthy.
- **Default-deny**: Tools not covered by any matching rule are blocked. Unlisted tools are blocked when a policy is loaded.

## Enforcement Engines

Janus supports two engines:

1. **JSON Schema enforcer** (default): Policy rules expressed in JSON. Conditions use JSON Schema to restrict tool arguments (pattern, enum, range, etc.). Composes with [`TaintTracker`](taint.md) for session-level IPI defense — both in the core install.
2. **SpiceDB-backed enforcement** (optional): ReBAC via SpiceDB plus a session-scalar taint gate.

## Features

- Fine-grained policies with argument-level conditions
- Automatic per-source taint tracking to gate sinks after untrusted reads (IPI defense)
- LLM-generated policies from user query
- Policy refinement as the agent gathers information
- Three fallback actions: raise exception, exit, or prompt user
- Framework adapters for LangChain, Google ADK, and the Claude Agent SDK (Claude Code), plus a `PreToolUse` hook shim (`janus-hook`) for the interactive Claude Code CLI
- Standalone `PolicyEnforcer` for custom integrations
- Built-in file and command tools with workspace sandboxing

## Quick Demo

Run the web app or CLI scenarios in under 5 minutes: [Demo →](demo.md)

[Get started →](getting-started.md)
