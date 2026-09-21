"""
``janus`` — the operator-facing CLI.

Deliberately separate from ``janus-hook``. That shim is a hot path with one
job: read a payload, decide, own its exit code. Its module docstring pins a
contract (argv flags only, stdout is protocol, fail closed on anything
unexpected) that an interactive wizard would sit awkwardly inside — and mixing
a prompt loop into the process Claude executes on every tool call is a good way
to eventually print a question where a decision belongs.

So operator commands live here and enforcement lives there. ``janus doctor`` is
the one overlap, and it delegates to the same :func:`janus.cli.hook.run_doctor`
the shim exposes rather than reimplementing the check.

Subcommands import their implementation lazily so ``janus --help`` stays fast
and this module imports cleanly on a core install.
"""

from __future__ import annotations

import argparse

from janus.cli.init import SCOPE_PROJECT, SCOPE_PROJECT_LOCAL, SCOPE_USER


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="janus",
        description="Janus — policy enforcement for LLM agent tool calls.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser(
        "init",
        help="Set up Janus for the Claude Code CLI (interactive).",
        description=(
            "Ask a few questions, then write a policy, wire the PreToolUse hook, "
            "and add the permissions.deny backstop. Shows every change and asks "
            "before writing."
        ),
    )
    init.add_argument(
        "--yes",
        action="store_true",
        help="Accept the recommended default for every question (non-interactive).",
    )
    init.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the policy, hook command, and settings diff, then exit without writing.",
    )
    init.add_argument(
        "--scope",
        choices=(SCOPE_PROJECT, SCOPE_PROJECT_LOCAL, SCOPE_USER),
        help="Which settings file to edit; skips the first question.",
    )
    init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing policy file without asking.",
    )
    init.add_argument(
        "--project-dir",
        help="Project root to configure. Defaults to the current directory.",
    )

    sub.add_parser(
        "doctor",
        help="Self-test the install and the hook payload round-trip.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "init":
        from janus.cli.init import run_init

        return run_init(args)

    if args.command == "doctor":
        from janus.cli.hook import run_doctor

        return run_doctor()

    return 2  # pragma: no cover - argparse rejects unknown subcommands first


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
