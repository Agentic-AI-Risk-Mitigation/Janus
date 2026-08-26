"""
``janus-hook`` — the Claude Code CLI hook shim.

Wired into a settings file (or, later, a plugin's ``hooks.json``) as a
``command`` hook. It reads one hook payload on stdin and writes the CLI's hook
JSON to stdout.

**Why a shim owns the exit code.** The CLI's hook dispatch fails *open*: a hook
that errors or times out lets the tool proceed to the normal permission flow. An
``http`` hook to an unreachable endpoint therefore fails open. A ``command``
shim owns its own output, so "enforcement is unavailable" can be turned into a
deny. That is the entire reason this process exists.

**Phase 1 is the degraded mode, deliberately.** There is no daemon yet, so this
process *is* the enforcement: it imports Janus per call (~150–400 ms cold) and
holds no cross-call state, which means **no taint, no provenance, no
PreToolUse/PostToolUse cross-check** — static policy evaluation only. That is
useful on its own (argument-level enforcement of a static policy, today) but it
is not the recommended deployment, and the ``permissions.deny`` backstop
(``janus-hook backstop``) matters more here than anywhere.

Phase 2 adds proxy mode, where the hot path is stdlib-only (socket + json, zero
``janus`` imports) and forwards to a warm daemon holding the Session. The flag
contract below is fixed now so that transition does not have to break it:
configuration arrives as **explicit argv flags**, never env vars or a
fixed-path config file, because a plugin's ``userConfig`` values slot into
exec-form ``args`` and nothing else.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Any

_DENY_TEMPLATE = {
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": "",
    }
}


def _fail_closed(reason: str) -> dict:
    """The deny emitted when Janus cannot decide. Never an empty allow."""
    output = json.loads(json.dumps(_DENY_TEMPLATE))
    output["hookSpecificOutput"]["permissionDecisionReason"] = f"[Janus] {reason}"
    return output


def _isolate_stdout() -> None:
    """Make stdout carry the hook protocol and nothing else.

    The CLI parses this process's stdout as JSON; a stray line corrupts the
    decision into unparseable bytes, which the CLI treats as a non-blocking hook
    error — so a *deny* that logs itself to stdout becomes an allow. Two writers
    have to be redirected, and each needs its own treatment:

    * ``print()`` and friends resolve ``sys.stdout`` at call time, so
      reassigning it is enough;
    * a ``logging.StreamHandler`` captured the stream object when it was
      installed (``janus.logger.configure_logging`` installs one on stdout), so
      it keeps writing to the real stdout no matter what ``sys.stdout`` says
      afterwards, and has to be repointed explicitly.
    """
    for logger in (logging.getLogger(), logging.getLogger("janus")):
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stdout:
                handler.setStream(sys.stderr)
    sys.stdout = sys.stderr


class _DeadlineExceeded(Exception):
    """The shim ran out of its own time budget."""


@contextlib.contextmanager
def _deadline(seconds: float):
    """Bound the decision by our own clock, not the CLI's.

    The CLI kills an overrunning hook and *proceeds* — verified on 2.1.233, a
    hook whose deny arrived after its timeout had the deny discarded and the
    tool ran. So the one timeout we must never hit is the CLI's: whatever goes
    wrong (a wedged import, a policy file on a stalled mount, a pathological
    regex in a condition), the shim has to reach its own limit first and emit a
    deny while it still can.

    ``SIGALRM`` is POSIX-only; where it is unavailable this degrades to no
    deadline, which is the pre-existing behaviour rather than a regression.
    """
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def on_alarm(signum, frame):
        raise _DeadlineExceeded(f"decision exceeded {seconds:g}s")

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _load_config(path: str | None) -> dict[str, Any]:
    """Load the optional sidecar config (``required_args``, ``known_servers``).

    Kept separate from ``--policy`` because these are adapter wiring, not policy
    rules, and because a plugin's ``userConfig`` can only pass scalars.
    """
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"config {path!r} must contain a JSON object")
    return data


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="janus-hook",
        description="Janus policy enforcement for Claude Code CLI hooks.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        # Required, deliberately. An enforcer with no policy loaded allows
        # everything, so a shim wired without --policy is a guard that reports
        # for duty and watches nothing — the exact failure a security tool must
        # never do quietly. argparse exits 2 on a missing flag, which the CLI
        # treats as a blocking hook error, so even the misconfiguration is
        # fail-closed.
        p.add_argument("--policy", required=True, help="Path to a Janus JSON policy file.")
        p.add_argument("--config", help="Path to a JSON sidecar (required_args, known_servers).")
        p.add_argument(
            "--mode",
            choices=("gate", "policy"),
            default="gate",
            help=(
                "gate (default): enforce Janus's opinions, abstain elsewhere and defer "
                "to the CLI permission flow. policy: strict default-deny."
            ),
        )
        p.add_argument(
            "--on-gate",
            choices=("ask", "deny"),
            default="ask",
            help=(
                "What a taint-gate hit does. 'ask' is the CLI's own decision value "
                "(verified: it blocks and surfaces the reason; 'escalate' is NOT "
                "recognized and lets the tool run). Ignored in phase 1, which has "
                "no cross-call taint."
            ),
        )
        p.add_argument(
            "--headless",
            action="store_true",
            help=(
                "Declare that no human can answer a permission prompt. The payload "
                "cannot tell us this: a `claude -p` run reports permission_mode "
                "'default' exactly like an interactive one, so escalation would "
                "silently become an allow unless the deployment says otherwise."
            ),
        )
        p.add_argument(
            "--deadline",
            type=float,
            default=5.0,
            help=(
                "Seconds before the shim gives up and denies (0 disables). Must stay "
                "well under the hook's own timeout, because the CLI's timeout fails "
                "OPEN — verified on 2.1.233: a hook that overran its timeout had its "
                "deny discarded and the tool ran. Owning the deadline ourselves is "
                "what keeps a wedged enforcement path from becoming an allow."
            ),
        )

    for name in ("pre", "post", "session-start", "session-end"):
        p = sub.add_parser(name)
        add_common(p)

    sub.add_parser("doctor", help="Self-test: imports, policy load, payload round-trip.")
    backstop = sub.add_parser(
        "backstop", help="Print the permissions.deny backstop block (see DEFAULT_CLI_SINK_DENY)."
    )
    backstop.add_argument("--indent", type=int, default=2)
    return parser


def _run_hook(args: argparse.Namespace, payload: dict) -> dict:
    # The deadline wraps the janus import too: in phase 1's stateless mode that
    # import is the slowest thing the shim does, so leaving it outside the
    # budget would leave the one path most likely to stall unguarded.
    with _deadline(args.deadline):
        return _decide(args, payload)


def _decide(args: argparse.Namespace, payload: dict) -> dict:
    from janus.adapters.claude_code import cli_name_resolver, handle_cli_payload

    config = _load_config(args.config)
    return handle_cli_payload(
        payload,
        args.policy,
        mode=args.mode,
        on_gate=args.on_gate,
        headless=args.headless,
        required_args=config.get("required_args"),
        resolve_name=cli_name_resolver(config.get("known_servers")),
    )


def run_doctor() -> int:
    """Self-test: imports, policy round-trip, and the degraded-mode banner.

    Public because ``janus init`` ends by running it — a wizard that reports
    success without exercising the path it just wired is reporting on its own
    intentions rather than on the deployment.
    """
    ok = True
    print(f"python: {sys.version.split()[0]} ({sys.executable})")
    try:
        import janus
        from janus.adapters.claude_code import handle_cli_payload, normalize_cli_event

        print(f"janus: {getattr(janus, '__version__', 'unknown')}")
        sample = {
            "hook_event_name": "PreToolUse",
            "session_id": "doctor",
            "tool_name": "Bash",
            "tool_input": {"command": "echo ok"},
            "permission_mode": "default",
        }
        event = normalize_cli_event(sample)
        assert event.tool_name == "Bash", event
        # No policy loaded -> nothing is listed -> gate mode abstains.
        assert handle_cli_payload(sample, None) == {}, "gate-mode abstain broken"
        # Strict mode must default-deny the same call.
        strict = handle_cli_payload(sample, {}, mode="policy")
        assert strict.get("hookSpecificOutput", {}).get("permissionDecision") == "deny", strict
        print("payload round-trip: ok (gate abstains, policy denies)")
    except Exception as exc:  # pragma: no cover - diagnostic path
        ok = False
        print(f"payload round-trip: FAILED ({type(exc).__name__}: {exc})")
    print(
        "mode: phase-1 stateless (no daemon) — static policy only; "
        "no taint, no provenance, no PreToolUse/PostToolUse cross-check"
    )
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "doctor":
        return run_doctor()
    if args.command == "backstop":
        from janus.adapters.claude_code import DEFAULT_CLI_SINK_DENY

        print(json.dumps({"permissions": DEFAULT_CLI_SINK_DENY}, indent=args.indent))
        return 0

    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
        if not isinstance(payload, dict):
            raise ValueError("hook payload must be a JSON object")
    except Exception as exc:
        # An unreadable payload on the pre seam is not a reason to let the call
        # through; on every other seam there is nothing to deny, so stay quiet.
        if args.command == "pre":
            print(json.dumps(_fail_closed(f"unreadable hook payload ({exc}); failing closed")))
        return 0

    real_stdout = sys.stdout
    _isolate_stdout()
    try:
        output = _run_hook(args, payload)
    except Exception as exc:
        output = (
            _fail_closed(
                f"enforcement unavailable ({type(exc).__name__}: {exc}); "
                "failing closed — run `janus-hook doctor`"
            )
            if args.command == "pre"
            else {}
        )
    finally:
        sys.stdout = real_stdout

    if output:
        print(json.dumps(output))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
