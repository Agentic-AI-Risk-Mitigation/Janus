"""
Built-in file system tools.

All file operations are scoped to a configurable workspace directory.
Attempts to access paths outside the workspace are rejected with a
ValueError to prevent path-traversal attacks.

Set the workspace before using any of these functions::

    from janus.tools.builtin.file_tools import set_workspace
    set_workspace("/path/to/workspace")
"""

import os
from pathlib import Path
from typing import Optional


_workspace: Optional[Path] = None


def set_workspace(path: "str | Path") -> None:
    """Set the workspace directory for all file operations."""
    global _workspace
    _workspace = Path(path).resolve()
    _workspace.mkdir(parents=True, exist_ok=True)


def get_workspace() -> Path:
    """Return the current workspace directory."""
    if _workspace is None:
        raise RuntimeError("Workspace not set. Call set_workspace() first.")
    return _workspace


def _resolve(file_path: str) -> Path:
    """
    Resolve a path relative to the workspace, rejecting traversal attempts.

    Raises:
        ValueError: If the path escapes the workspace.
    """
    workspace = get_workspace()

    if os.path.isabs(file_path):
        file_path = os.path.relpath(file_path, workspace)

    full_path = (workspace / file_path).resolve()

    try:
        full_path.relative_to(workspace)
    except ValueError:
        raise ValueError(
            f"Access denied: '{file_path}' is outside the workspace boundary."
        )

    return full_path


def _process_escapes(content: str) -> str:
    """
    Convert literal escape sequences to actual characters.

    LLMs sometimes emit ``\\n`` (two characters) instead of a real newline.
    This function converts the most common sequences safely.
    """
    if "\\n" not in content and "\\t" not in content and "\\r" not in content:
        return content

    placeholder = "\x00BKSL\x00"
    result = content.replace("\\\\", placeholder)
    result = result.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
    return result.replace(placeholder, "\\")


# ------------------------------------------------------------------
# Tool functions
# ------------------------------------------------------------------


def read_file(file_path: str) -> str:
    """
    Read and return the full contents of a file.

    :param file_path: Path to the file (relative to workspace).
    :return: The file's text content.
    """
    resolved = _resolve(file_path)

    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not resolved.is_file():
        raise ValueError(f"'{file_path}' is a directory, not a file.")

    try:
        return resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return resolved.read_bytes().decode("utf-8", errors="replace")


def write_file(file_path: str, content: str) -> str:
    """
    Create or overwrite a file with the given content.

    :param file_path: Path to the file (relative to workspace).
    :param content: Text to write.
    :return: Confirmation message.
    """
    resolved = _resolve(file_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    existed = resolved.exists()
    processed = _process_escapes(content)
    resolved.write_text(processed, encoding="utf-8")

    lines = processed.count("\n") + 1
    size = len(processed.encode("utf-8"))
    action = "Updated" if existed else "Created"
    return f"{action} '{file_path}' ({size} bytes, {lines} lines)."


def edit_file(file_path: str, old_string: str, new_string: str) -> str:
    """
    Replace a unique occurrence of old_string with new_string in a file.

    The old_string must appear exactly once in the file to guarantee an
    unambiguous edit.

    :param file_path: Path to the file (relative to workspace).
    :param old_string: Exact text to find.
    :param new_string: Replacement text.
    :return: Confirmation message.
    """
    resolved = _resolve(file_path)

    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not resolved.is_file():
        raise ValueError(f"'{file_path}' is not a file.")

    content = resolved.read_text(encoding="utf-8")

    processed_old = _process_escapes(old_string)
    processed_new = _process_escapes(new_string)

    count = content.count(processed_old)
    if count == 0:
        # Try without escape processing
        count = content.count(old_string)
        if count > 0:
            processed_old, processed_new = old_string, new_string
        else:
            snippet = old_string[:50] + ("..." if len(old_string) > 50 else "")
            raise ValueError(f"String not found in '{file_path}': '{snippet}'")

    if count > 1:
        raise ValueError(
            f"String appears {count} times in '{file_path}'. "
            "Provide a more specific string for an unambiguous match."
        )

    new_content = content.replace(processed_old, processed_new)
    resolved.write_text(new_content, encoding="utf-8")

    old_lines = processed_old.count("\n") + 1
    new_lines = processed_new.count("\n") + 1
    return f"Edited '{file_path}': replaced {old_lines} line(s) with {new_lines} line(s)."


def list_directory(path: str = ".") -> str:
    """
    List the contents of a directory.

    :param path: Directory path (relative to workspace, defaults to workspace root).
    :return: Formatted directory listing.
    """
    resolved = _resolve(path)

    if not resolved.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    if not resolved.is_dir():
        raise ValueError(f"'{path}' is not a directory.")

    entries = []
    for entry in sorted(resolved.iterdir()):
        if entry.is_dir():
            entries.append(f"[DIR]  {entry.name}/")
        else:
            size = entry.stat().st_size
            entries.append(f"[FILE] {entry.name} ({size} bytes)")

    if not entries:
        return f"Directory '{path}' is empty."

    return f"Contents of '{path}':\n" + "\n".join(entries)
