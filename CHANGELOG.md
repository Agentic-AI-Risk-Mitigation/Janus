# Changelog

All notable changes to Janus will be documented in this file.

This project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- **PDE integration**: Policy-Discovery-Engine has been merged into the main repo. SpiceDB-backed enforcement now lives under `janus/policy/pde/` (config, interceptor, discovery, bootstrap). `PDEEnforcer` imports from `janus.policy.pde.interceptor`; no separate `Policy-Discovery-Engine/` directory. Demos and docs updated accordingly.

## [0.1.0] — 2026-03-12 (Alpha)

### Added

- **Core engine**: `PolicyEnforcer` with JSON Schema–based argument validation and priority-ordered rule evaluation
- **JanusAgent**: single entry point wrapping LLM conversation loop with policy enforcement
- **10+ LLM providers**: OpenAI, Anthropic, Google Gemini, Azure OpenAI, AWS Bedrock, Ollama, vLLM, Together AI, OpenRouter
- **Framework adapters**: LangChain (3 integration depths) and Google ADK
- **LLM-generated policies**: auto-infer minimum-privilege policies from a user query via `policy="generate"`
- **Policy refinement**: incrementally tighten policies as an agent discovers new information
- **Built-in tools**: `read_file`, `write_file`, `edit_file`, `list_directory`, `run_command`, `fetch_url` — all workspace-scoped with path-traversal rejection
- **Custom tools**: `ToolDef` / `ToolParam` dataclasses for registering and guarding arbitrary tools
- **Three fallback actions**: raise `PolicyViolation`, call `sys.exit()`, or prompt user interactively
- **PDE (janus/policy/pde/)**: SpiceDB-backed graph enforcement with Zanzibar-style ReBAC and runtime taint tracking for IPI defence
- **Demo scenarios**: scripted demos for poisoned-README (IPI) and taint-cascade attacks
- **Web demo app**: FastAPI split-panel UI for live demo playback
- **Documentation site**: MkDocs Material with architecture, policy reference, getting-started, and demo guides
- **Test suite**: E2E PDE/SpiceDB integration tests and unit tests for enforcement, scripted LLM, mock tools, and scenario runner
