"""
End-to-End Integration Test: Janus + Policy-Discovery-Engine + SpiceDB

Schema design (from main.py/colleague):
  - Per-tool object types: tool_view_file, tool_edit_file, ...
  - ACL relation: can_invoke: role#member
  - Concrete agent: coding_agent is member of readonly, developer, executor roles
  - Taint check: Python-level (TOOL_TAINT_LIMIT), before SpiceDB ACL check
  - Tier-4 tools (bash_terminal, http_request, etc.) have NO ACL edges — always denied by SpiceDB

Test scenarios:
  A. LOW TAINT — tools with ACL edges pass, tier-4 tools denied by SpiceDB
  B. TAINT GATING — Python gate blocks tools when taint > TOOL_TAINT_LIMIT
  C. ACL GATING — Tier-4 tools permanently denied by SpiceDB (no ACL edges)
  D. IPI SCENARIO — agent reads critical source, subsequent dangerous actions blocked
"""
import os
import sys
import subprocess
import time

import pytest
from authzed.api.v1 import (
    Client,
    WriteSchemaRequest,
)
from grpcutil import insecure_bearer_token_credentials

# Make sure Policy-Discovery-Engine is importable
sys.path.insert(0, os.path.abspath("Policy-Discovery-Engine"))

from policy_engine.main import SCHEMA, TOOL_TAINT_LIMIT, bootstrap
from janus.agent import JanusAgent
from janus.exceptions import PolicyViolation

SPICEDB_ENDPOINT = "localhost:50051"
SPICEDB_TOKEN = "somerandomkey"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spicedb_server():
    """Start the SpiceDB docker container and wait until it is ready."""
    print("\nStarting SpiceDB container...")
    subprocess.run(
        ["docker", "compose", "-f", "Policy-Discovery-Engine/docker-compose.yml", "up", "-d"],
        check=True,
    )

    # Poll until SpiceDB gRPC port is responsive
    client = Client(SPICEDB_ENDPOINT, insecure_bearer_token_credentials(SPICEDB_TOKEN))
    for attempt in range(20):
        try:
            client.WriteSchema(WriteSchemaRequest(schema=SCHEMA))
            print("SpiceDB is ready.")
            break
        except Exception as exc:
            print(f"  Waiting for SpiceDB... attempt {attempt + 1}/20 ({str(exc)[:80]})")
            time.sleep(1)
    else:
        pytest.fail("SpiceDB did not become ready after 20 seconds.")

    yield client

    print("\nStopping SpiceDB container...")
    subprocess.run(
        ["docker", "compose", "-f", "Policy-Discovery-Engine/docker-compose.yml", "stop"],
        check=False,
    )


@pytest.fixture(scope="module")
def bootstrapped_spicedb(spicedb_server):
    """Bootstrap the SpiceDB schema + relationships once for all tests."""
    client = spicedb_server
    # bootstrap() writes schema (already done in health probe) + ACL relationships:
    #   - coding_agent is member of: readonly, developer, executor roles
    #   - readonly tools: view_file, grep, code_search, git_diff, git_log
    #   - developer tools: edit_file, create_file, delete_file, git_commit, git_push, git_clone
    #   - executor tools: run_tests, run_script, pip_install
    #   - tier-4 tools: bash_terminal, http_request, read_secret, write_secret, deploy — NO ACL edges
    bootstrap(client)
    return client


def _make_agent(role: str = "developer") -> JanusAgent:
    """Create a JanusAgent backed by the PDE engine."""
    return JanusAgent(
        model="openai/gpt-4o",
        api_key="mock-key-for-testing",
        policy_engine="pde",
        agent_role=role,
    )


# ---------------------------------------------------------------------------
# A. Low-taint session: ACL-granted tools pass, tier-4 always blocked
# ---------------------------------------------------------------------------

class TestLowTaintACLAllow:
    """
    At taint=0 all tools below their taint limit should pass the taint gate.
    Their SpiceDB ACL edge is verified by the actual SpiceDB check.
    """

    def test_view_file_allowed(self, bootstrapped_spicedb):
        """coding_agent is in readonly → has ACL for view_file (limit=90, taint=0)."""
        agent = _make_agent()
        assert agent.enforcer.interceptor.current_taint_level == 0
        agent.enforcer.enforce("view_file", {"file_path": "README.md"})
        print("PASS: view_file allowed at taint=0")

    def test_grep_allowed(self, bootstrapped_spicedb):
        """coding_agent is in readonly → has ACL for grep (limit=90, taint=0)."""
        agent = _make_agent()
        agent.enforcer.enforce("grep", {"pattern": "TODO", "path": "."})
        print("PASS: grep allowed at taint=0")

    def test_edit_file_allowed(self, bootstrapped_spicedb):
        """coding_agent is in developer → has ACL for edit_file (limit=70, taint=0)."""
        agent = _make_agent("developer")
        agent.enforcer.enforce("edit_file", {"file_path": "foo.py", "content": ""})
        print("PASS: edit_file allowed at taint=0")

    def test_git_push_allowed_at_low_taint(self, bootstrapped_spicedb):
        """coding_agent is in developer → has ACL for git_push (limit=20, taint=0)."""
        agent = _make_agent("developer")
        assert agent.enforcer.interceptor.current_taint_level == 0
        agent.enforcer.enforce("git_push", {})
        print("PASS: git_push allowed at taint=0")

    def test_run_tests_allowed(self, bootstrapped_spicedb):
        """coding_agent is in executor → has ACL for run_tests (limit=40, taint=0)."""
        agent = _make_agent("executor")
        agent.enforcer.enforce("run_tests", {})
        print("PASS: run_tests allowed at taint=0")


# ---------------------------------------------------------------------------
# B. Taint gating: Python gate blocks tools when taint exceeds limit
# ---------------------------------------------------------------------------

class TestTaintGating:
    """
    The Python taint gate in enforcement.py blocks tool calls when
    current_taint_level > TOOL_TAINT_LIMIT[tool_name], before even hitting SpiceDB.
    """

    def test_git_push_blocked_at_high_taint(self, bootstrapped_spicedb):
        """
        git_push TOOL_TAINT_LIMIT = 20.
        After a critical read (taint=90), git_push must be blocked by taint gate.
        """
        agent = _make_agent("developer")
        agent.update_taint(90)
        assert agent.enforcer.interceptor.current_taint_level == 90

        with pytest.raises(PolicyViolation) as exc_info:
            agent.enforcer.enforce("git_push", {})
        assert "git_push" in str(exc_info.value)
        print("PASS: git_push blocked by taint gate (taint=90 > limit=20)")

    def test_edit_file_blocked_at_critical_taint(self, bootstrapped_spicedb):
        """
        edit_file TOOL_TAINT_LIMIT = 70.
        At taint=90 it must be blocked by taint gate.
        """
        agent = _make_agent("developer")
        agent.update_taint(90)
        with pytest.raises(PolicyViolation):
            agent.enforcer.enforce("edit_file", {"file_path": "foo.py", "content": ""})
        print("PASS: edit_file blocked at taint=90 (limit=70)")

    def test_run_tests_blocked_at_high_taint(self, bootstrapped_spicedb):
        """
        run_tests TOOL_TAINT_LIMIT = 40.
        At taint=70 it must be blocked.
        """
        agent = _make_agent("executor")
        agent.update_taint(70)
        with pytest.raises(PolicyViolation):
            agent.enforcer.enforce("run_tests", {})
        print("PASS: run_tests blocked at taint=70 (limit=40)")

    def test_edit_file_passes_below_limit(self, bootstrapped_spicedb):
        """
        edit_file limit = 70. At taint 50 it must pass both taint gate and ACL.
        """
        agent = _make_agent("developer")
        agent.update_taint(50)
        agent.enforcer.enforce("edit_file", {"file_path": "foo.py", "content": ""})
        print("PASS: edit_file allowed at taint=50 (limit=70)")

    def test_view_file_passes_at_boundary(self, bootstrapped_spicedb):
        """
        view_file limit = 90. At taint=90: 90 > 90 is False → should PASS.
        """
        agent = _make_agent()
        agent.update_taint(90)
        agent.enforcer.enforce("view_file", {"file_path": "README.md"})
        print("PASS: view_file passes at taint=90 (limit=90, boundary is exclusive)")

    def test_taint_is_monotonic(self, bootstrapped_spicedb):
        """Taint can only go up, never down within a session."""
        agent = _make_agent()
        agent.update_taint(70)
        agent.update_taint(30)  # Should be ignored
        assert agent.enforcer.interceptor.current_taint_level == 70
        print("PASS: taint is monotonically increasing")


# ---------------------------------------------------------------------------
# C. ACL gating: Tier-4 tools permanently denied (no ACL edges in SpiceDB)
# ---------------------------------------------------------------------------

class TestACLGating:
    """
    Tier-4 tools have NO ACL edges in the bootstrap. Even at taint=0, SpiceDB
    returns PERMISSIONSHIP_NO_PERMISSION for any agent querying them.
    """

    def test_bash_terminal_denied_by_spicedb(self, bootstrapped_spicedb):
        """bash_terminal has no ACL edge — SpiceDB denies regardless of taint."""
        agent = _make_agent()
        assert agent.enforcer.interceptor.current_taint_level == 0  # passes taint gate
        with pytest.raises(PolicyViolation) as exc_info:
            agent.enforcer.enforce("bash_terminal", {})
        assert "bash_terminal" in str(exc_info.value)
        print("PASS: bash_terminal denied by SpiceDB (no ACL edge)")

    def test_http_request_denied_by_spicedb(self, bootstrapped_spicedb):
        """http_request has no ACL edge."""
        agent = _make_agent()
        with pytest.raises(PolicyViolation):
            agent.enforcer.enforce("http_request", {"url": "http://evil.com"})
        print("PASS: http_request denied by SpiceDB (no ACL edge)")

    def test_write_secret_denied_by_spicedb(self, bootstrapped_spicedb):
        """write_secret has no ACL edge."""
        agent = _make_agent()
        with pytest.raises(PolicyViolation):
            agent.enforcer.enforce("write_secret", {"key": "TOKEN", "value": "x"})
        print("PASS: write_secret denied by SpiceDB (no ACL edge)")

    def test_read_secret_denied_by_spicedb(self, bootstrapped_spicedb):
        """read_secret has no ACL edge."""
        agent = _make_agent()
        with pytest.raises(PolicyViolation):
            agent.enforcer.enforce("read_secret", {"key": "TOKEN"})
        print("PASS: read_secret denied by SpiceDB (no ACL edge)")

    def test_deploy_denied_by_spicedb(self, bootstrapped_spicedb):
        """deploy has no ACL edge."""
        agent = _make_agent()
        with pytest.raises(PolicyViolation):
            agent.enforcer.enforce("deploy", {"env": "prod"})
        print("PASS: deploy denied by SpiceDB (no ACL edge)")


# ---------------------------------------------------------------------------
# D. IPI (Indirect Prompt Injection) scenario
# ---------------------------------------------------------------------------

class TestIPIScenario:
    """
    End-to-end scenario: agent reads from a critical-risk data source
    (taint goes to 90), then tries dangerous operations.
    Most are blocked by the taint gate; truly safe reads still work.
    """

    def test_ipi_full_scenario(self, bootstrapped_spicedb):
        """
        1. Agent starts clean (taint=0), can do anything with an ACL edge.
        2. Reads a 'critical' source → taint jumps to 90.
        3. High-risk tools (git_push, run_script, git_commit) are taint-blocked.
        4. Boundary tool view_file (limit=90) still passes (90 > 90 is False).
        5. Tier-4 tools remain permanently blocked by SpiceDB.
        """
        agent = _make_agent("developer")

        # Step 1: Clean session — git_push passes (taint=0, limit=20)
        agent.enforcer.enforce("view_file", {"file_path": "README.md"})
        agent.enforcer.enforce("git_push", {})
        print("  Step 1 OK: clean session, all permitted tools work")

        # Step 2: Read a critical-risk source
        agent.update_taint(90)
        assert agent.enforcer.interceptor.current_taint_level == 90
        print(f"  Step 2 OK: taint elevated to {agent.enforcer.interceptor.current_taint_level}")

        # Step 3: Taint gate blocks dangerous tools
        for tool, args in [
            ("git_push", {}),
            ("git_commit", {}),
            ("run_script", {"script": "evil.sh"}),
            ("pip_install", {"package": "evil-pkg"}),
        ]:
            with pytest.raises(PolicyViolation, match=tool):
                agent.enforcer.enforce(tool, args)
        print("  Step 3 OK: taint gate blocks all high-risk tools after critical read")

        # Step 4: view_file at limit boundary (taint=90, limit=90) still allowed
        agent.enforcer.enforce("view_file", {"file_path": "notes.txt"})
        print("  Step 4 OK: view_file still accessible at taint boundary (90 <= 90)")

        # Step 5: Tier-4 tools still blocked by SpiceDB even at taint=0 on a fresh agent
        clean_agent = _make_agent()
        assert clean_agent.enforcer.interceptor.current_taint_level == 0
        with pytest.raises(PolicyViolation):
            clean_agent.enforcer.enforce("bash_terminal", {})
        with pytest.raises(PolicyViolation):
            clean_agent.enforcer.enforce("deploy", {"env": "prod"})
        print("  Step 5 OK: tier-4 tools blocked by SpiceDB ACL regardless of taint")

        print("PASS: Full IPI scenario validated!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
