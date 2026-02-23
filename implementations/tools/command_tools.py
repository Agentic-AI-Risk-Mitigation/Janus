"""Command execution tools for the coding agent."""

import subprocess

# Import workspace from file_tools to ensure consistency
from .file_tools import get_workspace


def run_command(command: str, timeout: int = 60) -> str:
    """
    Execute a shell command in the workspace directory.

    The command runs with the workspace as the current directory.
    Standard output and error are captured and returned.

    :param command: The shell command to execute
    :param timeout: Maximum execution time in seconds (default: 60)
    :return: The command output (stdout and stderr combined)
    """
    workspace = get_workspace()

    try:
        # Run the command
        result = subprocess.run(
            command,
            shell=True,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # Combine stdout and stderr
        output_parts = []

        if result.stdout:
            output_parts.append(result.stdout)

        if result.stderr:
            if output_parts:
                output_parts.append("\n[STDERR]:\n")
            output_parts.append(result.stderr)

        output = "".join(output_parts).strip()

        # Add exit code info if non-zero
        if result.returncode != 0:
            output += f"\n\n[Exit code: {result.returncode}]"

        if not output:
            output = f"Command completed successfully (exit code: {result.returncode})"

        return output

    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout} seconds"
    except Exception as e:
        return f"Command execution failed: {type(e).__name__}: {e}"


def pip_install(package_name: str, upgrade: bool = False) -> str:
    """
    Install a Python package using pip.

    :param package_name: Name of the package to install
    :param upgrade: Whether to upgrade the package if it exists
    :return: The command output
    """
    cmd = f"pip install {package_name}"
    if upgrade:
        cmd += " --upgrade"
    return run_command(cmd)


def fetch_url(url: str) -> str:
    """
    Fetch content from a URL via HTTP GET.

    :param url: The URL to fetch
    :return: The fetched content or error message
    """
    # Mock data for testing with http://internal
    if url.startswith("http://internal/"):
        filename = url.split("/")[-1]
        if filename == "data.csv":
            # Return mock employee data for testing
            return """Name,Department,Salary,StartDate
John Smith,Sales,75000,2020-01-15
Jane Doe,Engineering,95000,2019-03-10
Bob Johnson,Marketing,65000,2021-07-22
Alice Williams,Engineering,105000,2018-11-05
Mike Brown,Sales,70000,2022-02-14
Sarah Davis,Marketing,72000,2020-09-30
Tom Wilson,Engineering,110000,2017-06-18
Lisa Taylor,HR,60000,2021-04-12"""
        else:
            return f"Mock data not available for {filename}"

    # Use curl on Unix, curl is also available on Windows 10+
    return run_command(f'curl -s "{url}"')


def git_clone(repo_url: str, target_dir: str = None) -> str:
    """
    Clone a git repository.

    :param repo_url: URL of the repository to clone
    :param target_dir: Directory to clone into (optional)
    :return: The command output
    """
    cmd = f'git clone "{repo_url}"'
    if target_dir:
        cmd += f' "{target_dir}"'
    return run_command(cmd)
