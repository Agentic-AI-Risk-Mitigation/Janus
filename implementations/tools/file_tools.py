"""File manipulation tools for the coding agent."""

import os
from pathlib import Path
from typing import Optional

# Workspace path will be set by the agent
_workspace: Optional[Path] = None


def _process_escape_sequences(content: str) -> str:
    """
    Process escape sequences in content.

    LLMs sometimes send literal escape sequences (e.g., '\\n' as two characters)
    instead of actual escape characters. This function converts common ones.

    We use a targeted approach rather than unicode_escape to avoid issues
    with unicode characters and unexpected decoding.
    """
    # Check if content contains likely escape sequences
    if "\\n" not in content and "\\t" not in content and "\\r" not in content:
        return content

    # Replace common escape sequences manually
    # Order matters: handle double backslash last to avoid double-processing
    result = content

    # First, temporarily replace actual escaped backslashes to preserve them
    placeholder = "\x00ESCAPED_BACKSLASH\x00"
    result = result.replace("\\\\", placeholder)

    # Now replace common escape sequences
    result = result.replace("\\n", "\n")
    result = result.replace("\\t", "\t")
    result = result.replace("\\r", "\r")

    # Restore escaped backslashes
    result = result.replace(placeholder, "\\")

    return result


def set_workspace(path: str | Path) -> None:
    """Set the workspace directory for file operations."""
    global _workspace
    _workspace = Path(path).resolve()
    _workspace.mkdir(parents=True, exist_ok=True)


def get_workspace() -> Path:
    """Get the current workspace directory."""
    if _workspace is None:
        raise RuntimeError("Workspace not set. Call set_workspace() first.")
    return _workspace


def _resolve_path(file_path: str) -> Path:
    """
    Resolve a file path relative to workspace, with security checks.

    Raises:
        ValueError: If path attempts to escape workspace
    """
    workspace = get_workspace()

    # Handle absolute paths by making them relative
    if os.path.isabs(file_path):
        file_path = os.path.relpath(file_path, workspace)

    # Resolve the full path
    full_path = (workspace / file_path).resolve()

    # Security check: ensure path is within workspace
    try:
        full_path.relative_to(workspace)
    except ValueError:
        raise ValueError(f"Path '{file_path}' is outside the workspace. Access denied.")

    return full_path


def read_file(file_path: str) -> str:
    """
    Read the contents of a file.

    :param file_path: Path to the file to read (relative to workspace)
    :return: The contents of the file
    """
    resolved = _resolve_path(file_path)

    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not resolved.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    try:
        return resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Try reading as binary and decode with error handling
        return resolved.read_bytes().decode("utf-8", errors="replace")


def write_file(file_path: str, content: str) -> str:
    """
    Create or overwrite a file with the given content.

    :param file_path: Path to the file to write (relative to workspace)
    :param content: The content to write to the file
    :return: Confirmation message with file details
    """
    resolved = _resolve_path(file_path)

    # Create parent directories if needed
    resolved.parent.mkdir(parents=True, exist_ok=True)

    # Check if file existed
    existed = resolved.exists()

    # Process escape sequences (LLMs often send literal \n instead of newlines)
    processed_content = _process_escape_sequences(content)

    # Write the content
    resolved.write_text(processed_content, encoding="utf-8")

    # Return confirmation with line count for clarity
    line_count = processed_content.count("\n") + 1
    size = len(processed_content.encode("utf-8"))
    action = "Updated" if existed else "Created"
    return f"{action} file: {file_path} ({size} bytes, {line_count} lines)"


def edit_file(file_path: str, old_string: str, new_string: str) -> str:
    """
    Edit a file by replacing a specific string with a new string.

    The old_string must exist in the file exactly once for the replacement to succeed.
    This ensures precise, unambiguous edits.

    :param file_path: Path to the file to edit (relative to workspace)
    :param old_string: The exact string to find and replace
    :param new_string: The string to replace it with
    :return: Confirmation message with edit details
    """
    resolved = _resolve_path(file_path)

    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not resolved.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    # Read current content
    content = resolved.read_text(encoding="utf-8")

    # Process escape sequences in both old and new strings
    processed_old = _process_escape_sequences(old_string)
    processed_new = _process_escape_sequences(new_string)

    # Count occurrences
    count = content.count(processed_old)

    if count == 0:
        # Try the original strings in case escape processing changed them incorrectly
        count_original = content.count(old_string)
        if count_original > 0:
            processed_old = old_string
            processed_new = new_string
            count = count_original
        else:
            raise ValueError(
                f"String not found in file: '{old_string[:50]}{'...' if len(old_string) > 50 else ''}'"
            )

    if count > 1:
        raise ValueError(
            f"String found {count} times in file. Please provide a more specific string to ensure a unique match."
        )

    # Perform the replacement
    new_content = content.replace(processed_old, processed_new)

    # Write back
    resolved.write_text(new_content, encoding="utf-8")

    # Calculate change stats
    old_lines = processed_old.count("\n") + 1
    new_lines = processed_new.count("\n") + 1

    return f"Edited file: {file_path} (replaced {old_lines} line(s) with {new_lines} line(s))"


def list_directory(path: str = ".") -> str:
    """
    List the contents of a directory.

    :param path: Path to the directory to list (relative to workspace, defaults to workspace root)
    :return: A formatted list of directory contents
    """
    resolved = _resolve_path(path)

    if not resolved.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    if not resolved.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    entries = []
    for entry in sorted(resolved.iterdir()):
        if entry.is_dir():
            entries.append(f"[DIR]  {entry.name}/")
        else:
            size = entry.stat().st_size
            entries.append(f"[FILE] {entry.name} ({size} bytes)")

    if not entries:
        return f"Directory '{path}' is empty"

    header = f"Contents of '{path}':\n"
    return header + "\n".join(entries)
