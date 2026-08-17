"""
Import hygiene: importing Janus must not drag in optional dependencies or
mutate the host process's environment.

Regression tests for two documented consumer footguns: the policy generator's
import-time ``load_dotenv()`` (which injected Janus's own ``.env`` — including
an ``OPENAI_API_KEY`` — into any process that merely imported an adapter), and
the eager generator import that forced ``openai``/``jinja2`` onto every
consumer. Checks run in subprocesses so a polluted ``sys.modules`` from other
tests can't mask a regression.
"""

import subprocess
import sys

import pytest


def _run(code: str, cwd=None) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_importing_janus_does_not_import_generator_or_its_deps():
    out = _run(
        "import janus, sys; "
        "print(sorted(m for m in ('janus.policy.generator', 'openai', 'jinja2', 'dotenv') "
        "if m in sys.modules))"
    )
    assert out == "[]"


def test_importing_claude_adapter_does_not_import_generator():
    out = _run(
        "import janus.adapters.claude_agent_sdk, sys; "
        "print('janus.policy.generator' in sys.modules)"
    )
    assert out == "False"


def test_importing_janus_does_not_load_dotenv_from_cwd(tmp_path):
    (tmp_path / ".env").write_text("JANUS_HYGIENE_SENTINEL=leaked\n")
    out = _run(
        "import janus, os; print(os.environ.get('JANUS_HYGIENE_SENTINEL'))",
        cwd=tmp_path,
    )
    assert out == "None"


def test_generate_policy_still_resolves_lazily():
    import janus

    assert "generate_policy" in dir(janus)
    assert callable(janus.generate_policy)
    assert callable(janus.refine_policy)


def test_unknown_attribute_still_raises():
    import janus

    with pytest.raises(AttributeError):
        janus.no_such_symbol


def test_claude_code_adapter_imports_on_a_core_install():
    """The CLI hook shim must run wherever `claude` runs — no `claude` extra,
    no SDK, no server deps. A missing import here means a deployed hook fails
    closed on every call."""
    out = _run(
        "import janus.adapters.claude_code, sys; "
        "print(sorted(m for m in ('claude_agent_sdk', 'fastapi', 'openai') if m in sys.modules))"
    )
    assert out == "[]"


def test_janus_hook_shim_imports_on_a_core_install():
    out = _run("from janus.cli.hook import main; print(callable(main))")
    assert out == "True"
