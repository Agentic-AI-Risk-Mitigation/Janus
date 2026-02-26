"""
Janus built-in tools.

Provides ready-to-use ToolDef instances for common file and command operations.
Import ``BUILTIN_TOOLS`` to register all of them at once::

    from janus.tools.builtin import BUILTIN_TOOLS
    registry.register_many(BUILTIN_TOOLS)
"""

from janus.tools.base import ToolDef, ToolParam
from janus.tools.builtin.command_tools import fetch_url, run_command
from janus.tools.builtin.file_tools import (
    edit_file,
    list_directory,
    read_file,
    write_file,
)

BUILTIN_TOOLS: list[ToolDef] = [
    ToolDef(
        name="read_file",
        description=(
            "Read and return the full contents of a file. "
            "Paths are relative to the workspace."
        ),
        params=[
            ToolParam(name="file_path", type="string", description="Path to the file to read."),
        ],
        handler=read_file,
    ),
    ToolDef(
        name="write_file",
        description=(
            "Create or overwrite a file with the provided content. "
            "Use actual newline characters, not literal \\\\n."
        ),
        params=[
            ToolParam(name="file_path", type="string", description="Path to write (relative to workspace)."),
            ToolParam(name="content", type="string", description="Text content to write."),
        ],
        handler=write_file,
    ),
    ToolDef(
        name="edit_file",
        description=(
            "Replace an exact string in a file with a new string. "
            "The old_string must appear exactly once."
        ),
        params=[
            ToolParam(name="file_path", type="string", description="Path to the file to edit."),
            ToolParam(name="old_string", type="string", description="Exact text to find and replace."),
            ToolParam(name="new_string", type="string", description="Replacement text."),
        ],
        handler=edit_file,
    ),
    ToolDef(
        name="list_directory",
        description="List the contents of a directory within the workspace.",
        params=[
            ToolParam(
                name="path",
                type="string",
                description="Directory path (relative to workspace, default '.').",
                required=False,
                default=".",
            ),
        ],
        handler=list_directory,
    ),
    ToolDef(
        name="run_command",
        description=(
            "Execute a shell command in the workspace directory and return its output. "
            "stdout and stderr are combined."
        ),
        params=[
            ToolParam(name="command", type="string", description="Shell command to execute."),
        ],
        handler=run_command,
    ),
    ToolDef(
        name="fetch_url",
        description="Fetch the content of a URL via HTTP GET.",
        params=[
            ToolParam(name="url", type="string", description="URL to fetch."),
        ],
        handler=fetch_url,
    ),
]

__all__ = ["BUILTIN_TOOLS", "read_file", "write_file", "edit_file", "list_directory", "run_command", "fetch_url"]
