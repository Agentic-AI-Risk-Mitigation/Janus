"""Unit tests for tool handler implementations."""

import tempfile
from pathlib import Path

import pytest

from implementations.tools.command_tools import fetch_url, git_clone, pip_install
from implementations.tools.file_tools import (
    edit_file,
    list_directory,
    read_file,
    set_workspace,
    write_file,
)


class TestCommandWrappers:
    """Tests for command wrapper functions."""

    def test_pip_install_constructs_correct_command(self, monkeypatch):
        """Test that pip_install constructs the correct pip command."""
        executed_command = None

        def mock_run_command(command, timeout=60):
            nonlocal executed_command
            executed_command = command
            return "mocked output"

        monkeypatch.setattr("implementations.tools.command_tools.run_command", mock_run_command)

        result = pip_install("requests")

        assert executed_command == "pip install requests"
        assert result == "mocked output"

    def test_pip_install_with_upgrade(self, monkeypatch):
        """Test pip_install with upgrade flag."""
        executed_command = None

        def mock_run_command(command, timeout=60):
            nonlocal executed_command
            executed_command = command
            return "mocked output"

        monkeypatch.setattr("implementations.tools.command_tools.run_command", mock_run_command)

        pip_install("requests", upgrade=True)

        assert executed_command == "pip install requests --upgrade"

    def test_fetch_url_constructs_correct_command(self, monkeypatch):
        """Test that fetch_url constructs the correct curl command."""
        executed_command = None

        def mock_run_command(command, timeout=60):
            nonlocal executed_command
            executed_command = command
            return "mocked content"

        monkeypatch.setattr("implementations.tools.command_tools.run_command", mock_run_command)

        result = fetch_url("http://example.com/data.csv")

        assert 'curl -s "http://example.com/data.csv"' in executed_command
        assert result == "mocked content"

    def test_git_clone_without_target_dir(self, monkeypatch):
        """Test git_clone without target directory."""
        executed_command = None

        def mock_run_command(command, timeout=60):
            nonlocal executed_command
            executed_command = command
            return "Cloning into..."

        monkeypatch.setattr("implementations.tools.command_tools.run_command", mock_run_command)

        result = git_clone("http://github.com/example/repo")

        assert 'git clone "http://github.com/example/repo"' in executed_command
        assert result == "Cloning into..."

    def test_git_clone_with_target_dir(self, monkeypatch):
        """Test git_clone with target directory."""
        executed_command = None

        def mock_run_command(command, timeout=60):
            nonlocal executed_command
            executed_command = command
            return "Cloning into..."

        monkeypatch.setattr("implementations.tools.command_tools.run_command", mock_run_command)

        git_clone("http://github.com/example/repo", target_dir="./my_repo")

        assert 'git clone "http://github.com/example/repo" "./my_repo"' in executed_command


class TestFileTools:
    """Tests for file tool implementations."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            set_workspace(tmpdir)
            yield Path(tmpdir)

    def test_write_and_read_file(self, temp_workspace):
        """Test writing and reading a file."""
        write_file("test.txt", "Hello, World!")

        content = read_file("test.txt")

        assert content == "Hello, World!"

    def test_write_file_creates_parent_dirs(self, temp_workspace):
        """Test that write_file creates parent directories."""
        write_file("subdir/nested/file.txt", "Content")

        content = read_file("subdir/nested/file.txt")

        assert content == "Content"
        assert (temp_workspace / "subdir" / "nested").is_dir()

    def test_read_nonexistent_file(self, temp_workspace):
        """Test reading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_file("nonexistent.txt")

    def test_edit_file(self, temp_workspace):
        """Test editing a file."""
        write_file("edit_test.txt", "Hello World\nThis is a test\nGoodbye")

        edit_file("edit_test.txt", "This is a test", "This is edited")

        content = read_file("edit_test.txt")

        assert "This is edited" in content
        assert "This is a test" not in content

    def test_edit_file_string_not_found(self, temp_workspace):
        """Test editing with non-existent old_string raises ValueError."""
        write_file("edit_test.txt", "Hello World")

        with pytest.raises(ValueError, match="String not found"):
            edit_file("edit_test.txt", "Nonexistent", "New")

    def test_list_directory(self, temp_workspace):
        """Test listing directory contents."""
        write_file("file1.txt", "Content 1")
        write_file("file2.txt", "Content 2")
        write_file("subdir/file3.txt", "Content 3")

        result = list_directory(".")

        assert "file1.txt" in result
        assert "file2.txt" in result
        assert "subdir" in result

    def test_list_nonexistent_directory(self, temp_workspace):
        """Test listing non-existent directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            list_directory("nonexistent_dir")

    def test_file_path_outside_workspace_blocked(self, temp_workspace):
        """Test that accessing files outside workspace raises ValueError."""
        with pytest.raises(ValueError, match="outside the workspace"):
            read_file("../outside.txt")

    def test_absolute_path_blocked(self, temp_workspace):
        """Test that absolute paths are blocked."""
        with pytest.raises(ValueError, match="outside the workspace"):
            read_file("/etc/passwd")
