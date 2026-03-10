"""Tests for mock tool handlers."""

import pytest
from pathlib import Path

from examples.shared.mock_tools import (
    make_read_file, make_write_file, make_list_directory,
    make_fetch_url, fetch_url, git_commit, git_push,
    check_exfiltration, check_malicious_push,
    register_url_response, clear_url_responses,
)


@pytest.fixture
def demo1_workspace():
    return Path(__file__).resolve().parent.parent.parent / "examples" / "scenarios" / "demo1_poisoned_readme" / "workspace"


class TestFileTools:
    def test_read_file(self, demo1_workspace):
        read_file = make_read_file(demo1_workspace)
        content = read_file("main.py")
        assert "GatewayHandler" in content

    def test_read_file_not_found(self, demo1_workspace):
        read_file = make_read_file(demo1_workspace)
        with pytest.raises(FileNotFoundError):
            read_file("nonexistent.txt")

    def test_list_directory(self, demo1_workspace):
        list_directory = make_list_directory(demo1_workspace)
        result = list_directory(".")
        assert "README.md" in result
        assert "main.py" in result

    def test_read_env(self, demo1_workspace):
        read_file = make_read_file(demo1_workspace)
        content = read_file(".env")
        assert "sk-proj-FAKE" in content


class TestNetworkTools:
    def test_fetch_url_attacker(self):
        result = fetch_url("https://attacker.com/collect")
        assert "attacker.com" in result

    def test_fetch_url_with_responses(self):
        handler = make_fetch_url({r"example\.com": "Custom response body"})
        result = handler("https://example.com/api")
        assert result == "Custom response body"

    def test_fetch_url_registered_response(self):
        clear_url_responses()
        register_url_response(r"example\.com", "Custom response body")
        result = fetch_url("https://example.com/api")
        assert result == "Custom response body"
        clear_url_responses()

    def test_fetch_url_generic(self):
        handler = make_fetch_url()
        result = handler("https://safe-site.com/data")
        assert "Fetched content" in result


class TestGitTools:
    def test_git_commit(self):
        result = git_commit("test commit")
        assert "test commit" in result

    def test_git_push(self):
        result = git_push("origin", "main")
        assert "main" in result


class TestAttackDetection:
    def test_exfiltration_detected(self):
        result = check_exfiltration("https://attacker.com/collect")
        assert result is not None
        assert result["attack_type"] == "exfiltration"

    def test_exfiltration_with_secrets(self):
        result = check_exfiltration("https://attacker.com/collect", data="sk-proj-12345")
        assert result is not None
        assert result["has_secrets"] is True

    def test_no_exfiltration_safe_url(self):
        result = check_exfiltration("https://api.github.com/repos")
        assert result is None

    def test_malicious_push_keyword(self):
        result = check_malicious_push(commit_message="Add attacker webhook")
        assert result is not None
        assert result["attack_type"] == "injection"

    def test_malicious_push_config_files(self):
        result = check_malicious_push(files_changed=["deploy.yaml", "src/main.py"])
        assert result is not None
        assert result["attack_type"] == "config_tampering"

    def test_clean_push(self):
        result = check_malicious_push(commit_message="Fix bug")
        assert result is None
