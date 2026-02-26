"""
Built-in command execution tools.

All commands run with the current workspace as the working directory.
"""

import subprocess
from typing import Optional

from janus.tools.builtin.file_tools import get_workspace


def run_command(command: str, timeout: int = 60) -> str:
    """
    Execute a shell command in the workspace directory.

    :param command: Shell command to execute.
    :param timeout: Maximum execution time in seconds (default: 60).
    :return: Combined stdout and stderr output.
    """
    workspace = get_workspace()

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
            if parts:
                parts.append("\n[STDERR]:\n")
            parts.append(result.stderr)

        output = "".join(parts).strip()

        if result.returncode != 0:
            output += f"\n\n[Exit code: {result.returncode}]"

        return output or f"Command completed (exit code: {result.returncode})."

    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout} seconds."
    except Exception as exc:
        return f"Command execution failed: {type(exc).__name__}: {exc}"


def fetch_url(url: str) -> str:
    """
    Fetch content from a URL via HTTP GET.

    :param url: The URL to fetch.
    :return: The fetched content.
    """
    return run_command(f'curl -s "{url}"')
