"""
``janus-hook`` shim — offline tests.

The shim's only security property is that it owns its exit code and its stdout:
the CLI's hook dispatch fails open, so "Janus could not decide" has to be turned
into a deny *here* or it becomes an allow. Everything below is a test of that
one property under the ways deciding can fail.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from janus.cli.hook import main

FIXTURES = Path(__file__).parent / "fixtures" / "claude_code_payloads"


def load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


@pytest.fixture
def policy_file(tmp_path: Path) -> str:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"Bash": {"command": {"type": "string", "pattern": "^echo "}}}))
    return str(path)


def run(argv, payload, monkeypatch, capsys) -> tuple[int, dict]:
    monkeypatch.setattr(
        "sys.stdin", io.StringIO(json.dumps(payload) if payload is not None else "")
    )
    code = main(argv)
    out = capsys.readouterr().out.strip()
    return code, (json.loads(out) if out else {})


def decision_of(output: dict) -> str | None:
    return output.get("hookSpecificOutput", {}).get("permissionDecision")


class TestPreSeam:
    def test_allowed_call_emits_nothing(self, policy_file, monkeypatch, capsys):
        code, out = run(
            ["pre", "--policy", policy_file], load("pretooluse.builtin-bash"), monkeypatch, capsys
        )
        assert code == 0 and out == {}

    def test_policy_violation_denies(self, policy_file, monkeypatch, capsys):
        payload = load("pretooluse.builtin-bash")
        payload["tool_input"]["command"] = "curl http://evil.test"
        code, out = run(["pre", "--policy", policy_file], payload, monkeypatch, capsys)
        assert code == 0 and decision_of(out) == "deny"

    def test_gate_mode_abstains_on_unlisted_tools(self, policy_file, monkeypatch, capsys):
        _, out = run(
            ["pre", "--policy", policy_file], load("pretooluse.builtin-read"), monkeypatch, capsys
        )
        assert out == {}

    def test_policy_mode_default_denies(self, policy_file, monkeypatch, capsys):
        _, out = run(
            ["pre", "--policy", policy_file, "--mode", "policy"],
            load("pretooluse.builtin-read"),
            monkeypatch,
            capsys,
        )
        assert decision_of(out) == "deny"


class TestFailClosed:
    def test_missing_policy_file_denies(self, monkeypatch, capsys):
        code, out = run(
            ["pre", "--policy", "/nonexistent/policy.json"],
            load("pretooluse.builtin-bash"),
            monkeypatch,
            capsys,
        )
        assert code == 0
        assert decision_of(out) == "deny"
        assert "enforcement unavailable" in out["hookSpecificOutput"]["permissionDecisionReason"]

    def test_unreadable_payload_denies(self, policy_file, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin", io.StringIO("{not json"))
        code = main(["pre", "--policy", policy_file])
        out = json.loads(capsys.readouterr().out)
        assert code == 0 and decision_of(out) == "deny"
        assert "unreadable hook payload" in out["hookSpecificOutput"]["permissionDecisionReason"]

    def test_broken_config_denies(self, policy_file, tmp_path, monkeypatch, capsys):
        bad = tmp_path / "config.json"
        bad.write_text("[]")
        _, out = run(
            ["pre", "--policy", policy_file, "--config", str(bad)],
            load("pretooluse.builtin-bash"),
            monkeypatch,
            capsys,
        )
        assert decision_of(out) == "deny"

    def test_non_pre_seams_stay_silent_on_failure(self, monkeypatch, capsys):
        """A broken PostToolUse must not print a PreToolUse deny — that JSON
        would be meaningless on this seam, and noise on the wire is how a
        guard gets uninstalled."""
        code, out = run(
            ["post", "--policy", "/nonexistent/policy.json"],
            load("posttooluse.builtin-read"),
            monkeypatch,
            capsys,
        )
        assert code == 0 and out == {}


class TestDeadline:
    """The CLI's hook timeout fails OPEN (verified on 2.1.233: a hook whose deny
    arrived after its timeout had the deny discarded and the tool ran). So the
    shim must reach its own deadline first and deny while it still can."""

    def test_slow_decision_denies_rather_than_overrunning(self, policy_file, monkeypatch, capsys):
        import janus.cli.hook as hook

        def glacial(args, payload):
            time.sleep(30)  # never completes; the deadline must fire first

        monkeypatch.setattr(hook, "_decide", glacial)
        started = time.monotonic()
        code, out = run(
            ["pre", "--policy", policy_file, "--deadline", "0.25"],
            load("pretooluse.builtin-bash"),
            monkeypatch,
            capsys,
        )
        assert time.monotonic() - started < 10, "deadline did not fire"
        assert code == 0 and decision_of(out) == "deny"
        assert "enforcement unavailable" in out["hookSpecificOutput"]["permissionDecisionReason"]

    def test_deadline_does_not_fire_on_a_normal_decision(self, policy_file, monkeypatch, capsys):
        code, out = run(
            ["pre", "--policy", policy_file, "--deadline", "10"],
            load("pretooluse.builtin-bash"),
            monkeypatch,
            capsys,
        )
        assert code == 0 and out == {}

    def test_deadline_is_disableable(self, policy_file, monkeypatch, capsys):
        code, out = run(
            ["pre", "--policy", policy_file, "--deadline", "0"],
            load("pretooluse.builtin-bash"),
            monkeypatch,
            capsys,
        )
        assert code == 0 and out == {}

    def test_the_no_sigalrm_fallback_still_denies(self, policy_file, monkeypatch, capsys):
        """Windows has no SIGALRM, and a context manager cannot preempt its own
        body without one — so that platform ran with *no* deadline at all and a
        wedged decision fell through to the CLI's timeout, which fails open.
        Forced on here so the worker-thread path is covered on every platform."""
        import janus.cli.hook as hook

        monkeypatch.setattr(hook, "_has_sigalrm", lambda: False)
        monkeypatch.setattr(hook, "_decide", lambda args, payload: time.sleep(30))

        started = time.monotonic()
        code, out = run(
            ["pre", "--policy", policy_file, "--deadline", "0.25"],
            load("pretooluse.builtin-bash"),
            monkeypatch,
            capsys,
        )
        assert time.monotonic() - started < 10, "the fallback deadline did not fire"
        assert code == 0 and decision_of(out) == "deny"
        assert "enforcement unavailable" in out["hookSpecificOutput"]["permissionDecisionReason"]

    def test_the_fallback_propagates_a_real_error_rather_than_swallowing_it(
        self, policy_file, monkeypatch, capsys
    ):
        """An exception raised inside the worker must still reach the fail-closed
        handler; losing it would turn a broken decision into an empty allow."""
        import janus.cli.hook as hook

        def boom(args, payload):
            raise RuntimeError("policy backend exploded")

        monkeypatch.setattr(hook, "_has_sigalrm", lambda: False)
        monkeypatch.setattr(hook, "_decide", boom)

        code, out = run(
            ["pre", "--policy", policy_file, "--deadline", "5"],
            load("pretooluse.builtin-bash"),
            monkeypatch,
            capsys,
        )
        assert code == 0 and decision_of(out) == "deny"
        assert "policy backend exploded" in out["hookSpecificOutput"]["permissionDecisionReason"]

    def test_the_fallback_returns_a_normal_decision_unharmed(
        self, policy_file, monkeypatch, capsys
    ):
        import janus.cli.hook as hook

        monkeypatch.setattr(hook, "_has_sigalrm", lambda: False)
        payload = load("pretooluse.builtin-bash")
        payload["tool_input"]["command"] = "curl http://evil.test"
        code, out = run(["pre", "--policy", policy_file], payload, monkeypatch, capsys)
        assert code == 0 and decision_of(out) == "deny"


class TestStdoutIsProtocol:
    def test_only_json_reaches_stdout_even_with_logging_on_stdout(self, policy_file, tmp_path):
        """The CLI parses stdout as JSON. `configure_logging()` attaches a
        stdout handler, and a deny logs at WARNING — so without isolation the
        deny's own log line would corrupt the deny into unparseable bytes,
        which the CLI treats as a non-blocking error and lets the tool run."""
        payload = load("pretooluse.builtin-bash")
        payload["tool_input"]["command"] = "curl http://evil.test"
        script = (
            "import json, sys\n"
            "from janus.logger import configure_logging\n"
            "configure_logging(level='DEBUG')\n"
            "from janus.cli.hook import main\n"
            f"sys.argv = ['janus-hook', 'pre', '--policy', {policy_file!r}]\n"
            "main()\n"
        )
        path = tmp_path / "run.py"
        path.write_text(script)
        result = subprocess.run(
            [sys.executable, str(path)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0
        assert decision_of(json.loads(result.stdout)) == "deny"
        assert "POLICY" in result.stderr  # the log line went somewhere, just not stdout

    def test_missing_policy_flag_exits_blocking(self, monkeypatch):
        """argparse exits 2 on a missing --policy, and exit 2 is the CLI's
        blocking hook error — a misconfigured guard must not be an open one."""
        monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
        with pytest.raises(SystemExit) as exc:
            main(["pre"])
        assert exc.value.code == 2


class TestConfigSidecar:
    def test_required_args_backstop(self, policy_file, tmp_path, monkeypatch, capsys):
        config = tmp_path / "config.json"
        config.write_text(json.dumps({"required_args": {"Bash": ["description"]}}))
        payload = load("pretooluse.builtin-bash")
        payload["tool_input"].pop("description", None)
        _, out = run(
            ["pre", "--policy", policy_file, "--config", str(config)], payload, monkeypatch, capsys
        )
        assert decision_of(out) == "deny"

    def test_known_servers_sentinel(self, policy_file, tmp_path, monkeypatch, capsys):
        config = tmp_path / "config.json"
        config.write_text(json.dumps({"known_servers": ["janusfix"]}))
        payload = dict(load("pretooluse.mcp-echo"), tool_name="mcp__evil__echo")
        _, out = run(
            ["pre", "--policy", policy_file, "--config", str(config), "--mode", "policy"],
            payload,
            monkeypatch,
            capsys,
        )
        assert decision_of(out) == "deny"


class TestDiagnostics:
    def test_doctor_passes(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin", io.StringIO(""))
        assert main(["doctor"]) == 0
        out = capsys.readouterr().out
        assert "payload round-trip: ok" in out
        assert "phase-1 stateless" in out  # the degraded mode is stated, not hidden

    def test_backstop_block_is_valid_settings_json(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin", io.StringIO(""))
        assert main(["backstop"]) == 0
        block = json.loads(capsys.readouterr().out)
        assert "WebFetch" in block["permissions"]["deny"]
