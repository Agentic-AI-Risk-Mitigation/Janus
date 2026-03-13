"""
CLI runner for demo scenarios.

Usage:
    python -m examples.run demo1_poisoned_readme --protected
    python -m examples.run demo1_poisoned_readme --unprotected
    python -m examples.run --list
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from examples.scenarios import get_scenario, list_scenarios
from examples.shared.events import BaseEvent
from examples.shared.scenario_runner import ScenarioRunner


def _format_event(event: BaseEvent) -> str:
    """Format an event for CLI output."""
    d = event.to_dict()
    etype = d.pop("event_type")
    panel = d.pop("panel")
    d.pop("timestamp", None)
    d.pop("delay_ms", None)

    prefix = f"[{panel.upper():>5}]"

    if etype == "chat_message":
        role = d["role"].upper()
        content = d["content"]
        if len(content) > 120:
            content = content[:120] + "..."
        return f"{prefix} {role}: {content}"

    if etype == "tool_call":
        args_str = ", ".join(f"{k}={v!r}" for k, v in d["args"].items())
        return f"{prefix} >> {d['tool']}({args_str})"

    if etype == "tool_result":
        result = d["result"]
        if len(result) > 100:
            result = result[:100] + "..."
        status = "OK" if d["success"] else "FAIL"
        return f"{prefix}    <- [{status}] {result}"

    if etype == "janus_decision":
        icon = "ALLOW" if d["allowed"] else "BLOCK"
        reason = f" | {d['reason']}" if d.get("reason") else ""
        args_str = ", ".join(f"{k}={v!r}" for k, v in d["args"].items())
        return f"{prefix} [{icon}] {d['tool']}({args_str}){reason}"

    if etype == "attack_event":
        return f"{prefix} !! {d['attack_type'].upper()}: {d['detail']}"

    if etype == "taint_update":
        return f"{prefix} [TAINT] Level: {d['level']} (source={d['source']}, risk={d['risk']})"

    if etype == "system_event":
        return f"{prefix} --- {d['message']} ---"

    return f"{prefix} {etype}: {d}"


async def _print_event(event: BaseEvent) -> None:
    print(_format_event(event))


async def run_scenario(name: str, protected: bool) -> None:
    scenario = get_scenario(name)
    runner = ScenarioRunner()
    await runner.run(scenario, protected=protected, event_callback=_print_event)


def main():
    parser = argparse.ArgumentParser(description="Run a Janus demo scenario")
    parser.add_argument("scenario", nargs="?", help="Scenario name (e.g. demo1_poisoned_readme)")
    parser.add_argument("--protected", action="store_true", help="Run with Janus enforcement")
    parser.add_argument("--unprotected", action="store_true", help="Run without Janus enforcement")
    parser.add_argument("--list", action="store_true", help="List available scenarios")
    args = parser.parse_args()

    if args.list:
        scenarios = list_scenarios()
        if not scenarios:
            print("No scenarios found.")
            return
        print("Available scenarios:")
        for s in scenarios:
            print(f"  {s['name']:30s} {s['title']} — {s['description']}")
        return

    if not args.scenario:
        parser.error("Specify a scenario name or use --list")

    if not args.protected and not args.unprotected:
        parser.error("Specify --protected or --unprotected")

    asyncio.run(run_scenario(args.scenario, protected=args.protected))


if __name__ == "__main__":
    main()
