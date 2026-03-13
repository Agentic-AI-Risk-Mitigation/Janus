"""
ToolDef factories for demo scenarios.

Provides factory functions that return lists of ToolDef objects using
the mock tool handlers. File tools bind to a specific workspace path
via closures (no global state -- safe for concurrent use).
"""

from __future__ import annotations

from pathlib import Path

from examples.shared import mock_tools
from janus.tools.base import ToolDef, ToolParam


def get_file_tools(workspace: Path) -> list[ToolDef]:
    """File system tools using real I/O scoped to the given workspace via closures."""
    read_file = mock_tools.make_read_file(workspace)
    write_file = mock_tools.make_write_file(workspace)
    edit_file = mock_tools.make_edit_file(workspace)
    list_directory = mock_tools.make_list_directory(workspace)

    return [
        ToolDef(
            name="read_file",
            description="Read the full contents of a file in the workspace.",
            params=[
                ToolParam(name="file_path", type="string",
                          description="Path to the file (relative to workspace)."),
            ],
            handler=read_file,
        ),
        ToolDef(
            name="write_file",
            description="Create or overwrite a file with the given content.",
            params=[
                ToolParam(name="file_path", type="string",
                          description="Path to the file (relative to workspace)."),
                ToolParam(name="content", type="string",
                          description="Text content to write."),
            ],
            handler=write_file,
        ),
        ToolDef(
            name="edit_file",
            description="Replace a unique string in a file with new content.",
            params=[
                ToolParam(name="file_path", type="string",
                          description="Path to the file (relative to workspace)."),
                ToolParam(name="old_string", type="string",
                          description="Exact text to find (must appear exactly once)."),
                ToolParam(name="new_string", type="string",
                          description="Replacement text."),
            ],
            handler=edit_file,
        ),
        ToolDef(
            name="list_directory",
            description="List the contents of a directory in the workspace.",
            params=[
                ToolParam(name="path", type="string",
                          description="Directory path (relative to workspace, defaults to root).",
                          required=False, default="."),
            ],
            handler=list_directory,
        ),
    ]


def get_network_tools(url_responses: dict[str, str] | None = None) -> list[ToolDef]:
    """Network tools (fully mocked -- no real HTTP requests).

    Args:
        url_responses: Optional mapping of URL regex patterns to fake responses.
            If provided, creates an isolated fetch_url handler (safe for concurrent use).
    """
    handler = mock_tools.make_fetch_url(url_responses) if url_responses else mock_tools.fetch_url
    return [
        ToolDef(
            name="fetch_url",
            description="Fetch content from a URL via HTTP GET.",
            params=[
                ToolParam(name="url", type="string",
                          description="The URL to fetch."),
            ],
            handler=handler,
        ),
    ]


def get_git_tools() -> list[ToolDef]:
    """Git tools (fully mocked -- no real git operations)."""
    return [
        ToolDef(
            name="git_commit",
            description="Commit staged changes with a message.",
            params=[
                ToolParam(name="message", type="string",
                          description="Commit message."),
            ],
            handler=mock_tools.git_commit,
        ),
        ToolDef(
            name="git_push",
            description="Push commits to a remote repository.",
            params=[
                ToolParam(name="remote", type="string",
                          description="Remote name.", required=False, default="origin"),
                ToolParam(name="branch", type="string",
                          description="Branch name.", required=False, default="main"),
            ],
            handler=mock_tools.git_push,
        ),
    ]


def get_all_tools(workspace: Path) -> list[ToolDef]:
    """All available tools for a scenario."""
    return get_file_tools(workspace) + get_network_tools() + get_git_tools()
