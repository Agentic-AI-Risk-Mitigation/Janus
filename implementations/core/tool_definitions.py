"""
Unified tool definitions - SINGLE SOURCE OF TRUTH.

All tools are defined here ONCE. Framework adapters convert these
to their specific formats (LangChain, ADK, OpenAI, etc.).

DO NOT define tools in individual agent files.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Type

from pydantic import BaseModel, Field, create_model

from implementations.tools.command_tools import fetch_url, git_clone, pip_install, run_command
from implementations.tools.communication_tools import send_email
from implementations.tools.file_tools import (
    edit_file,
    list_directory,
    read_file,
    write_file,
)

# Type mapping from JSON Schema types to Python types
_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


@dataclass
class ToolParameter:
    """Definition of a single tool parameter."""

    name: str
    type: str  # "string", "integer", "boolean", etc.
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolDefinition:
    """
    Complete definition of a tool.

    This is the single source of truth for:
    - Tool name
    - Description (shown to LLM)
    - Parameters (schema)
    - Handler function (actual implementation)
    """

    name: str
    description: str
    parameters: list[ToolParameter]
    handler: Callable

    def to_json_schema(self) -> dict:
        """Convert parameters to JSON Schema format (used by most frameworks)."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description,
            }
            if param.required:
                required.append(param.name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def to_pydantic_model(self) -> Type[BaseModel]:
        """
        Create a Pydantic model for this tool's parameters.

        Used by LangChain's StructuredTool for proper schema inference.
        """
        fields = {}

        for param in self.parameters:
            python_type = _TYPE_MAP.get(param.type, str)

            if param.required:
                fields[param.name] = (python_type, Field(description=param.description))
            else:
                fields[param.name] = (
                    Optional[python_type],
                    Field(default=param.default, description=param.description),
                )

        # Create model with tool name
        model_name = f"{self.name.title().replace('_', '')}Args"
        return create_model(model_name, **fields)


# =============================================================================
# TOOL DEFINITIONS - Add new tools here
# =============================================================================

TOOL_DEFINITIONS: list[ToolDefinition] = [
    ToolDefinition(
        name="read_file",
        description=(
            "Read and return the full contents of a file. "
            "You will receive the actual file contents."
        ),
        parameters=[
            ToolParameter(
                name="file_path",
                type="string",
                description="Path to the file to read (relative to workspace)",
            ),
        ],
        handler=read_file,
    ),
    ToolDefinition(
        name="write_file",
        description=(
            "Create or overwrite a file with the given content. "
            "Use actual newlines in content, not literal backslash-n."
        ),
        parameters=[
            ToolParameter(
                name="file_path",
                type="string",
                description="Path to the file to write (relative to workspace)",
            ),
            ToolParameter(
                name="content",
                type="string",
                description="The content to write to the file",
            ),
        ],
        handler=write_file,
    ),
    ToolDefinition(
        name="edit_file",
        description=(
            "Edit a file by replacing a specific string with a new string. "
            "The old_string must exist exactly once in the file."
        ),
        parameters=[
            ToolParameter(
                name="file_path",
                type="string",
                description="Path to the file to edit (relative to workspace)",
            ),
            ToolParameter(
                name="old_string",
                type="string",
                description="The exact string to find and replace",
            ),
            ToolParameter(
                name="new_string",
                type="string",
                description="The string to replace it with",
            ),
        ],
        handler=edit_file,
    ),
    ToolDefinition(
        name="list_directory",
        description="List the contents of a directory.",
        parameters=[
            ToolParameter(
                name="path",
                type="string",
                description="Path to the directory (relative to workspace, defaults to root)",
                required=False,
                default=".",
            ),
        ],
        handler=list_directory,
    ),
    ToolDefinition(
        name="run_command",
        description=(
            "Execute a shell command and return its output. "
            "You will receive the actual stdout/stderr. "
            "Read the output carefully and report it accurately."
        ),
        parameters=[
            ToolParameter(
                name="command",
                type="string",
                description="The shell command to execute",
            ),
        ],
        handler=run_command,
    ),
    ToolDefinition(
        name="send_email",
        description="Send an email notification (simulated).",
        parameters=[
            ToolParameter(
                name="to",
                type="string",
                description="The recipient email address",
            ),
            ToolParameter(
                name="subject",
                type="string",
                description="The email subject line",
            ),
            ToolParameter(
                name="body",
                type="string",
                description="The email body content",
            ),
        ],
        handler=send_email,
    ),
    ToolDefinition(
        name="pip_install",
        description="Install a Python package using pip.",
        parameters=[
            ToolParameter(
                name="package_name",
                type="string",
                description="Name of the package to install",
            ),
            ToolParameter(
                name="upgrade",
                type="boolean",
                description="Whether to upgrade the package if it exists",
                required=False,
                default=False,
            ),
        ],
        handler=pip_install,
    ),
    ToolDefinition(
        name="fetch_url",
        description="Fetch content from a URL via HTTP GET.",
        parameters=[
            ToolParameter(
                name="url",
                type="string",
                description="The URL to fetch",
            ),
        ],
        handler=fetch_url,
    ),
    ToolDefinition(
        name="git_clone",
        description="Clone a git repository.",
        parameters=[
            ToolParameter(
                name="repo_url",
                type="string",
                description="URL of the repository to clone",
            ),
            ToolParameter(
                name="target_dir",
                type="string",
                description="Directory to clone into (optional)",
                required=False,
                default=None,
            ),
        ],
        handler=git_clone,
    ),
]


def get_tool_by_name(name: str) -> Optional[ToolDefinition]:
    """Get a tool definition by name."""
    for tool in TOOL_DEFINITIONS:
        if tool.name == name:
            return tool
    return None


def get_all_tool_names() -> list[str]:
    """Get list of all tool names."""
    return [tool.name for tool in TOOL_DEFINITIONS]
