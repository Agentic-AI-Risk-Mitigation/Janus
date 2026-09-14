"""
Demo 2 — a LangChain agent that reads a GitHub issue and explains it.

Tools are secured with ``secure_langchain_tools`` (depth 1 of the LangChain
adapter), so Janus evaluates the policy on every tool call before the handler
runs. LangChain drives the reasoning loop.

Two things are worth watching in this demo:

1. **Least privilege by task scope.** The task is "explain issue #N in repo X".
   That needs read access to one issue in one repository — nothing more. With
   ``--pin-repo`` the policy is narrowed to exactly that, so a hijacked agent
   cannot pivot to reading a different repository.

2. **Untrusted input reaching a privileged loop.** An issue body is text a
   stranger wrote. ``--poisoned`` loads a fixture whose body carries a hidden
   instruction telling the agent to post the developer's environment variables
   as a comment. The system prompt tells the model to ignore it; the policy is
   what makes that refusal enforceable rather than aspirational.

Usage::

    # Explain a real issue
    python -m demos.demo_2.agent --repo Agentic-AI-Risk-Mitigation/Janus --issue 4

    # Narrow the policy to just that issue
    python -m demos.demo_2.agent --repo python/cpython --issue 100000 --pin-repo

    # Injection scenario, both sides, no network or API key needed for --check
    python -m demos.demo_2.agent --poisoned --mode both
    python -m demos.demo_2.agent --check

Requires ``OPENAI_API_KEY`` (or ``--model`` pointing at another provider) for
every mode except ``--check``, which exercises the enforcer directly.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from demos.demo_2.github_api import load_fixture
from demos.demo_2.tools import GITHUB_TOOLS
from janus import PolicyEnforcer
from janus.exceptions import PolicyViolation

HERE = Path(__file__).parent
POLICY_PATH = HERE / "policies" / "issue_reader_policy.json"
PROMPT_PATH = HERE / "prompts" / "system_prompt.md"
POISONED_FIXTURE = HERE / "fixtures" / "poisoned_issue.json"

DEFAULT_REPO = "Agentic-AI-Risk-Mitigation/Janus"
DEFAULT_ISSUE = 4

# Arguments the enforcer rejects when absent or blank, before any rule runs.
# Without this, omitting `owner` would sidestep the owner pattern entirely.
REQUIRED_ARGS = {
    "fetch_github_issue": ["owner", "repo", "issue_number"],
    "fetch_issue_comments": ["owner", "repo", "issue_number"],
}


# ---------------------------------------------------------------------------
# Policy construction
# ---------------------------------------------------------------------------


def build_pinned_policy(owner: str, repo: str, issue_number: int) -> dict[str, Any]:
    """
    Build a policy scoped to exactly one issue in one repository.

    This is the least-privilege reading of the task: "explain issue #N in repo
    X" needs read access to issue N of repo X, so that is all the policy grants.
    ``enum`` is an exact-match set, so a hijacked agent cannot widen the scope
    by asking for a different owner, repo, or issue number.

    ``post_issue_comment`` is absent from the result on purpose — a tool with no
    rule is denied, so the write sink needs no deny rule of its own.
    """
    conditions = {
        "owner": {"type": "string", "enum": [owner]},
        "repo": {"type": "string", "enum": [repo]},
        "issue_number": {"type": "integer", "enum": [issue_number]},
    }
    rule = [{"priority": 1, "effect": 0, "conditions": conditions, "fallback": 0}]
    return {"fetch_github_issue": rule, "fetch_issue_comments": rule}


def load_policy(owner: str, repo: str, issue_number: int, *, pin_repo: bool) -> PolicyEnforcer:
    """Return the enforcer for a protected run."""
    enforcer = PolicyEnforcer(required_args=REQUIRED_ARGS)
    if pin_repo:
        enforcer.load(build_pinned_policy(owner, repo, issue_number))
    else:
        enforcer.load(POLICY_PATH)
    return enforcer


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------


class IssueExplainer:
    """
    A LangChain agent whose tools are wrapped in Janus enforcement.

    Tools come from ``secure_langchain_tools`` (depth 1 of the LangChain
    adapter), which converts each ``ToolDef`` into a ``StructuredTool`` whose
    handler calls ``enforcer.enforce()`` before the real function runs. A
    blocked call returns a refusal string to the model instead of raising, so
    the agent can report the denial rather than crashing.

    The reasoning loop is built against whichever LangChain generation is
    installed: ``create_agent`` on 1.x, ``AgentExecutor`` on 0.3. Depth 3's
    ``JanusLangChainAgent`` is not used here because it imports
    ``AgentExecutor`` unconditionally and so cannot load under LangChain 1.x.
    """

    def __init__(self, *, policy: PolicyEnforcer | None, model: str, verbose: bool):
        from janus.adapters.langchain import secure_langchain_tools

        self.lc_tools = secure_langchain_tools(GITHUB_TOOLS, policy)
        self.verbose = verbose
        self.tool_calls: list[str] = []

        system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
        llm = _init_chat_model(model)

        if _has_create_agent():
            from langchain.agents import create_agent

            self._flavor = "create_agent"
            self._executor = create_agent(
                model=llm,
                tools=self.lc_tools,
                system_prompt=system_prompt,
            )
        else:
            # Only reachable on LangChain 0.3, where these still exist. mypy
            # resolves against whichever generation is installed, so one of the
            # two branches always looks wrong to it.
            from langchain.agents import (  # type: ignore[attr-defined]
                AgentExecutor,
                create_tool_calling_agent,
            )
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )
            self._flavor = "agent_executor"
            self._executor = AgentExecutor(
                agent=create_tool_calling_agent(llm, self.lc_tools, prompt),
                tools=self.lc_tools,
                verbose=verbose,
                handle_parsing_errors=True,
                max_iterations=8,
                return_intermediate_steps=True,
            )

    def list_tools(self) -> list[str]:
        """Names of the tools the agent can see (whether or not policy allows them)."""
        return [tool.name for tool in self.lc_tools]

    def run(self, task: str) -> str:
        """Run one task and return the agent's final text answer."""
        self.tool_calls = []

        if self._flavor == "create_agent":
            result = self._executor.invoke({"messages": [{"role": "user", "content": task}]})
            messages = result.get("messages", [])
            self._record_tool_calls(messages)
            final = messages[-1].content if messages else ""
            return final if isinstance(final, str) else str(final)

        result = self._executor.invoke({"input": task})  # type: ignore[arg-type]
        for action, _observation in result.get("intermediate_steps", []):
            self.tool_calls.append(f"{action.tool}({action.tool_input})")
        return result.get("output", "No response generated.")

    def _record_tool_calls(self, messages: list[Any]) -> None:
        """Pull the tool calls out of a LangGraph message list for the trace."""
        for message in messages:
            for call in getattr(message, "tool_calls", None) or []:
                name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "?")
                args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
                self.tool_calls.append(f"{name}({args})")


def _has_create_agent() -> bool:
    """True on LangChain >= 1.0, where ``create_agent`` replaced ``AgentExecutor``."""
    try:
        from langchain.agents import create_agent  # noqa: F401
    except ImportError:
        return False
    return True


def _init_chat_model(model: str) -> Any:
    """
    Build a chat model from a ``"<provider>/<name>"`` string.

    Uses LangChain's own ``init_chat_model``, which takes ``provider:name``, so
    only the separator needs translating.
    """
    from langchain.chat_models import init_chat_model

    if "/" not in model:
        raise ValueError(f"--model must be '<provider>/<name>', got '{model}'")
    provider, name = model.split("/", 1)
    return init_chat_model(f"{provider}:{name}", temperature=0)


def build_agent(
    *,
    policy: PolicyEnforcer | None,
    model: str,
    verbose: bool,
) -> IssueExplainer:
    """
    Build the agent, with or without Janus enforcement.

    ``policy=None`` produces the unprotected comparison agent: the same model
    and the same tools, with no enforcement layer in front of them.
    """
    return IssueExplainer(policy=policy, model=model, verbose=verbose)


def explain_issue(agent: IssueExplainer, owner: str, repo: str, issue_number: int) -> str:
    """Ask the agent to explain one issue, printing the tool trace when verbose."""
    answer = agent.run(
        f"Explain GitHub issue #{issue_number} in the repository {owner}/{repo}. "
        "Read the issue first, then explain what it is about."
    )
    if agent.verbose and agent.tool_calls:
        print("\nTool calls attempted:")
        for call in agent.tool_calls:
            print(f"  - {call}")
        print()
    return answer


# ---------------------------------------------------------------------------
# Offline policy check (no LLM, no network)
# ---------------------------------------------------------------------------

# (tool, arguments, expected_allowed, what this case demonstrates)
CHECK_CASES: list[tuple[str, dict[str, Any], bool, str]] = [
    (
        "fetch_github_issue",
        {"owner": "python", "repo": "cpython", "issue_number": 12345},
        True,
        "reading a well-formed issue reference",
    ),
    (
        "fetch_issue_comments",
        {"owner": "python", "repo": "cpython", "issue_number": 12345},
        True,
        "reading the comment thread",
    ),
    (
        "post_issue_comment",
        {"owner": "python", "repo": "cpython", "issue_number": 12345, "body": "env dump"},
        False,
        "write sink absent from the policy -> default-deny",
    ),
    (
        "fetch_github_issue",
        {"owner": "python", "repo": "cpython"},
        False,
        "issue_number omitted -> required_args rejects it",
    ),
    (
        "fetch_github_issue",
        {"owner": "python", "repo": "../../etc/passwd", "issue_number": 1},
        False,
        "path traversal in repo name -> fails the pattern",
    ),
    (
        "fetch_github_issue",
        {"owner": "python", "repo": "cpython", "issue_number": 0},
        False,
        "issue_number below minimum",
    ),
]


def run_check() -> int:
    """
    Exercise the policy directly against the enforcer and report each outcome.

    No LLM and no network, so this verifies the policy is doing what the README
    claims on any machine, with no API key.
    """
    enforcer = PolicyEnforcer(required_args=REQUIRED_ARGS)
    enforcer.load(POLICY_PATH)

    print(f"Policy: {POLICY_PATH.name}")
    print(f"Tools in policy: {', '.join(sorted(enforcer.policy or {}))}\n")

    failures = 0
    for tool, arguments, expected_allowed, description in CHECK_CASES:
        try:
            enforcer.enforce(tool, arguments)
            allowed, detail = True, "allowed"
        except PolicyViolation as exc:
            allowed, detail = False, f"blocked — {exc.reason}"

        ok = allowed == expected_allowed
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {tool:<22} {description}")
        print(f"      -> {detail}\n")

    print(f"{len(CHECK_CASES) - failures}/{len(CHECK_CASES)} cases behaved as expected.")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _split_repo(value: str) -> tuple[str, str]:
    """Split ``owner/name``, raising ``ValueError`` so the caller can report it."""
    owner, _, name = value.partition("/")
    if not owner or not name or "/" in name:
        raise ValueError(f"--repo must be 'owner/name', got '{value}'")
    return owner, name


def _banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="demos.demo_2.agent",
        description="A Janus-guarded LangChain agent that explains a GitHub issue.",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"Repository as 'owner/name' (default: {DEFAULT_REPO}).",
    )
    parser.add_argument(
        "--issue",
        type=int,
        default=DEFAULT_ISSUE,
        help=f"Issue number to explain (default: {DEFAULT_ISSUE}).",
    )
    parser.add_argument(
        "--mode",
        choices=("protected", "unprotected", "both"),
        default="protected",
        help="Run with Janus, without it, or both for comparison.",
    )
    parser.add_argument(
        "--pin-repo",
        action="store_true",
        help="Narrow the policy to exactly the requested owner/repo/issue.",
    )
    parser.add_argument(
        "--poisoned",
        action="store_true",
        help="Read the issue from the poisoned fixture instead of GitHub.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exercise the policy against the enforcer and exit. No LLM, no network.",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4o",
        help="Model as '<provider>/<name>' (default: openai/gpt-4o).",
    )
    parser.add_argument("--verbose", action="store_true", help="Show LangChain's trace.")
    args = parser.parse_args(argv)

    if args.check:
        return run_check()

    try:
        owner, repo = _split_repo(args.repo)
    except ValueError as exc:
        parser.error(str(exc))
    issue_number = args.issue

    provider = args.model.split("/", 1)[0]
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Set it, or pass --model for another "
            "provider, or run --check to verify the policy without an LLM.",
            file=sys.stderr,
        )
        return 1

    if args.poisoned:
        if not POISONED_FIXTURE.exists():
            print(f"Missing fixture: {POISONED_FIXTURE}", file=sys.stderr)
            return 1
        load_fixture(POISONED_FIXTURE)
        owner, repo, issue_number = "acme-corp", "widget-sdk", 42
        print(f"Reading from fixture {POISONED_FIXTURE.name} -> {owner}/{repo}#{issue_number}")

    if args.mode in ("unprotected", "both"):
        _banner("UNPROTECTED — no Janus enforcement")
        agent = build_agent(policy=None, model=args.model, verbose=args.verbose)
        print(explain_issue(agent, owner, repo, issue_number))

    if args.mode in ("protected", "both"):
        scope = "pinned to this issue" if args.pin_repo else POLICY_PATH.name
        _banner(f"PROTECTED — Janus enforcing ({scope})")
        enforcer = load_policy(owner, repo, issue_number, pin_repo=args.pin_repo)
        agent = build_agent(policy=enforcer, model=args.model, verbose=args.verbose)
        print(explain_issue(agent, owner, repo, issue_number))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
