# Claude Code CLI hook payloads — pinned fixtures

Verbatim stdin payloads received by `command` hooks in a live `claude -p` session.
**Do not hand-edit these files** — their value is being the bytes the CLI actually
sent, not shapes derived from documentation. To refresh, re-run the capture (below)
against a newer CLI and update this provenance block.

## Provenance

- Captured: 2026-08-15
- CLI: **2.1.233** (Claude Code), Linux
- Method: `--settings` file wiring a dump script (`cat > <event>.<ns>.json`) as a
  `command` hook for every event; one `claude -p` run exercising built-in `Read` +
  `Bash` + a stdio MCP tool (`mcp__janusfix__echo`), one run spawning a
  general-purpose subagent via the `Agent` tool. Design context:
  `plans/claude-code-plugin-design.md` §10.

## Findings the fixtures pin (where they contradict the docs, the fixtures win)

- **`PostToolUse` carries `tool_response`, NOT `tool_output`, on CLI 2.1.233** —
  the hooks docs (as read 2026-08-15) say `tool_output`. The normalizer must read
  both keys and take whichever is present; a payload with neither is the
  regression signal.
- MCP tool `tool_response` is a **raw JSON string** (`"{\"result\":...}"`), not
  MCP content blocks; built-in tools return dicts (`Bash`: `stdout`/`stderr`/
  `interrupted`/…; `Read`: `type`/`file`). Unwrapping must handle both.
- `agent_id` / `agent_type` are present **only** on payloads from inside a
  subagent (absent, not null, at top level). The subagent spawn tool is named
  **`Agent`** (not `Task`) — matching the SDK-path smoke finding.
- `PostToolBatch` fires (payload has a `tool_calls` array, no `tool_name`), even
  for single-call "batches".
- Extra keys beyond the documented set: `effort`, `prompt_id` (most events),
  `duration_ms` (PostToolUse); `SubagentStop` is rich (`agent_transcript_path`,
  `last_assistant_message`, `stop_hook_active`, …). `SessionStart`/`SessionEnd`
  omit `permission_mode`.

## Not yet captured (known gaps)

- Plugin-MCP tool names (`mcp__plugin_<plugin>_<server>__<tool>`) — needs an
  installed plugin.
- `PostToolUseFailure`, `PermissionRequest`, `PermissionDenied`, `PreCompact`.
- Payloads under non-`default` `permission_mode` and in interactive (non `-p`)
  sessions.
