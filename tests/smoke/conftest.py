"""Shared fixtures for the live SDK smoke suite.

These tests talk to the real ``claude`` CLI subprocess and spend real tokens.
They are excluded from the offline default run: every test is skipped unless
``JANUS_LIVE_SMOKE=1`` is set. See ``test_live_sdk_semantics.py`` for what is
verified and ``plans/claude-agent-sdk-hardening.md`` (follow-ups 5 and 6) for
why this suite exists.

Run with::

    JANUS_LIVE_SMOKE=1 uv run python -m pytest tests/smoke/ -v -s

Requirements: the ``claude`` CLI on PATH with working credentials (stored
subscription OAuth or ANTHROPIC_API_KEY). Cost is a few cents (haiku, short
prompts). The suite prints a versions/findings summary at the end — record the
verified SDK+CLI pair in ``plans/claude-agent-sdk-hardening.md`` and the
integration memory note after every run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

LIVE = os.environ.get("JANUS_LIVE_SMOKE") == "1"

# Cheap + fast; the suite tests SDK/CLI plumbing semantics, not model quality.
SMOKE_MODEL = os.environ.get("JANUS_SMOKE_MODEL", "haiku")

REPORT_PATH = Path(__file__).parent / "last_run.json"


def pytest_collection_modifyitems(config, items):
    if LIVE:
        return
    skip = pytest.mark.skip(reason="live smoke suite: set JANUS_LIVE_SMOKE=1 to run")
    for item in items:
        if item.path and Path(item.path).parent == Path(__file__).parent:
            item.add_marker(skip)


def _versions() -> dict:
    import claude_agent_sdk

    cli = shutil.which("claude")
    cli_version = "unavailable"
    if cli:
        out = subprocess.run([cli, "--version"], capture_output=True, text=True)
        cli_version = out.stdout.strip() or out.stderr.strip()
    return {
        "claude_agent_sdk": claude_agent_sdk.__version__,
        "claude_cli": cli_version,
        "run_started": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }


@pytest.fixture(scope="session")
def smoke_report():
    """Session-wide findings record, written to last_run.json and printed."""
    report: dict = {"versions": _versions() if LIVE else {}, "findings": {}}
    yield report
    if not LIVE:
        return
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print("\n" + "=" * 72)
    print("JANUS LIVE SMOKE REPORT (record in plans/claude-agent-sdk-hardening.md)")
    print(json.dumps(report, indent=2, default=str))
    print("=" * 72)
