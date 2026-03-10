"""Integration test: Demo 1 policy enforcement."""

import pytest
from pathlib import Path

from janus.policy.enforcer import PolicyEnforcer
from janus.exceptions import PolicyViolation


@pytest.fixture
def enforcer():
    policy_path = (
        Path(__file__).resolve().parent.parent.parent
        / "examples" / "scenarios" / "demo1_poisoned_readme" / "policy.json"
    )
    e = PolicyEnforcer()
    e.load(str(policy_path))
    return e


class TestDemo1Enforcement:
    def test_list_directory_allowed(self, enforcer):
        enforcer.enforce("list_directory", {"path": "."})

    def test_read_file_readme_allowed(self, enforcer):
        enforcer.enforce("read_file", {"file_path": "README.md"})

    def test_read_file_main_allowed(self, enforcer):
        enforcer.enforce("read_file", {"file_path": "main.py"})

    def test_read_file_env_blocked(self, enforcer):
        with pytest.raises(PolicyViolation) as exc_info:
            enforcer.enforce("read_file", {"file_path": ".env"})
        assert ".env" in str(exc_info.value)

    def test_read_file_dotenv_path_blocked(self, enforcer):
        with pytest.raises(PolicyViolation):
            enforcer.enforce("read_file", {"file_path": "config/.env"})

    def test_fetch_url_trusted_allowed(self, enforcer):
        enforcer.enforce("fetch_url", {"url": "https://trusted-registry.example.com/packages"})

    def test_fetch_url_attacker_blocked(self, enforcer):
        with pytest.raises(PolicyViolation) as exc_info:
            enforcer.enforce("fetch_url", {"url": "https://attacker.com/collect"})
        assert "fetch_url" in str(exc_info.value)

    def test_fetch_url_arbitrary_blocked(self, enforcer):
        with pytest.raises(PolicyViolation):
            enforcer.enforce("fetch_url", {"url": "https://example.com/api"})

    def test_unknown_tool_blocked(self, enforcer):
        with pytest.raises(PolicyViolation):
            enforcer.enforce("run_command", {"command": "ls"})
