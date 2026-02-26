"""
Coding agent tool implementations.

Each tool is defined in two parts:
1. A plain Python function that does the actual work.
2. A ToolDef that registers it with Janus — describing its name,
   description (shown to the LLM), parameters, and handler.

These tools intentionally do NOT use janus.tools.builtin so that this
example demonstrates how any user-defined tools are wired into Janus.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional

from janus import ToolDef, ToolParam

# ---------------------------------------------------------------------------
# Workspace management
# ---------------------------------------------------------------------------

_workspace: Optional[Path] = None


def set_workspace(path: str | Path) -> None:
    """Set the root workspace directory for all file operations."""
    global _workspace
    _workspace = Path(path).resolve()
    _workspace.mkdir(parents=True, exist_ok=True)


def _workspace_path() -> Path:
    if _workspace is None:
        raise RuntimeError("Workspace not set. Call set_workspace() before running the agent.")
    return _workspace


def _safe_resolve(file_path: str) -> Path:
    """
    Resolve a path relative to the workspace and reject traversal attempts.
    """
    workspace = _workspace_path()
    if os.path.isabs(file_path):
        file_path = os.path.relpath(file_path, workspace)
    resolved = (workspace / file_path).resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError:
        raise ValueError(f"Access denied: '{file_path}' is outside the workspace.")
    return resolved


def _fix_escapes(content: str) -> str:
    """Convert literal \\n / \\t sequences that LLMs sometimes emit."""
    if "\\n" not in content and "\\t" not in content:
        return content
    placeholder = "\x00BS\x00"
    result = content.replace("\\\\", placeholder)
    result = result.replace("\\n", "\n").replace("\\t", "\t")
    return result.replace(placeholder, "\\")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def read_file(file_path: str) -> str:
    """Read the full contents of a file inside the workspace."""
    resolved = _safe_resolve(file_path)
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not resolved.is_file():
        raise ValueError(f"'{file_path}' is not a file.")
    try:
        return resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return resolved.read_bytes().decode("utf-8", errors="replace")


def write_file(file_path: str, content: str) -> str:
    """Create or overwrite a file with the given content."""
    resolved = _safe_resolve(file_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    existed = resolved.exists()
    processed = _fix_escapes(content)
    resolved.write_text(processed, encoding="utf-8")
    action = "Updated" if existed else "Created"
    lines = processed.count("\n") + 1
    size = len(processed.encode())
    return f"{action} '{file_path}' ({size} bytes, {lines} lines)."


def edit_file(file_path: str, old_string: str, new_string: str) -> str:
    """Replace a unique occurrence of old_string with new_string in a file."""
    resolved = _safe_resolve(file_path)
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    content = resolved.read_text(encoding="utf-8")
    old = _fix_escapes(old_string)
    new = _fix_escapes(new_string)
    count = content.count(old)
    if count == 0:
        count = content.count(old_string)
        if count:
            old, new = old_string, new_string
        else:
            raise ValueError(f"String not found in '{file_path}': '{old_string[:60]}'")
    if count > 1:
        raise ValueError(f"String found {count} times in '{file_path}'. Provide a more specific match.")
    resolved.write_text(content.replace(old, new), encoding="utf-8")
    return f"Edited '{file_path}': replaced {old.count(chr(10)) + 1} line(s) with {new.count(chr(10)) + 1} line(s)."


def list_directory(path: str = ".") -> str:
    """List the contents of a directory in the workspace."""
    resolved = _safe_resolve(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    if not resolved.is_dir():
        raise ValueError(f"'{path}' is not a directory.")
    entries = []
    for entry in sorted(resolved.iterdir()):
        if entry.is_dir():
            entries.append(f"[DIR]  {entry.name}/")
        else:
            entries.append(f"[FILE] {entry.name} ({entry.stat().st_size} bytes)")
    if not entries:
        return f"Directory '{path}' is empty."
    return f"Contents of '{path}':\n" + "\n".join(entries)


def run_command(command: str, timeout: int = 60) -> str:
    """Execute a shell command in the workspace directory."""
    workspace = _workspace_path()
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        parts = []
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append(f"\n[STDERR]:\n{result.stderr}")
        output = "".join(parts).strip()
        if result.returncode != 0:
            output += f"\n\n[Exit code: {result.returncode}]"
        return output or f"Command completed (exit code: {result.returncode})."
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s."
    except Exception as exc:
        return f"Command failed: {type(exc).__name__}: {exc}"


def send_email(to: str, subject: str, body: str) -> str:
    """Send an email notification (simulated — logs instead of sending)."""
    print(f"\n{'=' * 50}")
    print("EMAIL SENT (SIMULATED)")
    print(f"To:      {to}")
    print(f"Subject: {subject}")
    print(f"Body:\n{body}")
    print(f"{'=' * 50}\n")
    return f"Email sent to '{to}' with subject '{subject}'."


# ---------------------------------------------------------------------------
# ToolDef registrations — these are what Janus uses
# ---------------------------------------------------------------------------

CODING_AGENT_TOOLS: list[ToolDef] = [
    ToolDef(
        name="read_file",
        description=(
            "Read and return the full contents of a file. "
            "The path is relative to the workspace."
        ),
        params=[
            ToolParam("file_path", "string", "Path to the file to read (relative to workspace)."),
        ],
        handler=read_file,
    ),
    ToolDef(
        name="write_file",
        description=(
            "Create or overwrite a file with the given content. "
            "Use real newline characters in content — not literal \\n."
        ),
        params=[
            ToolParam("file_path", "string", "Destination path (relative to workspace)."),
            ToolParam("content",   "string", "Full text content to write."),
        ],
        handler=write_file,
    ),
    ToolDef(
        name="edit_file",
        description=(
            "Edit a file by replacing an exact string with a new one. "
            "old_string must appear exactly once in the file."
        ),
        params=[
            ToolParam("file_path",  "string", "Path to the file to edit."),
            ToolParam("old_string", "string", "Exact text to find and replace."),
            ToolParam("new_string", "string", "Replacement text."),
        ],
        handler=edit_file,
    ),
    ToolDef(
        name="list_directory",
        description="List the contents (files and subdirectories) of a workspace directory.",
        params=[
            ToolParam("path", "string", "Directory path (relative to workspace, default '.').",
                      required=False, default="."),
        ],
        handler=list_directory,
    ),
    ToolDef(
        name="run_command",
        description=(
            "Execute a shell command in the workspace directory and return its output. "
            "stdout and stderr are combined in the response."
        ),
        params=[
            ToolParam("command", "string", "Shell command to run."),
        ],
        handler=run_command,
    ),
    ToolDef(
        name="send_email",
        description="Send an email notification (simulated — prints to console).",
        params=[
            ToolParam("to",      "string", "Recipient email address."),
            ToolParam("subject", "string", "Email subject line."),
            ToolParam("body",    "string", "Email body content."),
        ],
        handler=send_email,
    ),
]
