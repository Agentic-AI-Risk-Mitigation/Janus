"""
Evaluation Harness for Progent.

This module provides the `Evaluator` class and helper functions to run
scenarios against the agent and verify the results.
"""

import io
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from implementations.frameworks.langchain_agent import LangChainAgent


@dataclass
class Scenario:
    """A test scenario for the agent."""

    name: str
    description: str
    prompt: str
    category: str  # e.g., "valid", "direct_injection", "tool_misuse"
    expected_action: str  # Description of what should happen
    validator: Callable[[Any], bool]  # Function that takes the result and returns Pass/Fail


@dataclass
class ExecutionLog:
    """Tracks tool execution events during a scenario run."""

    blocked_tools: list[dict] = field(default_factory=list)
    allowed_tools: list[dict] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)

    def parse_log_line(self, line: str):
        """Parse a log line and extract tool execution info."""
        if "PROGENT: BLOCKED" in line:
            # Extract tool name from line like: "PROGENT: BLOCKED - read_file(file_path=\".env\") | Reason: ..."
            try:
                parts = line.split("PROGENT: BLOCKED - ")[1].split("(")[0]
                tool_name = parts.strip()
                self.blocked_tools.append({"tool": tool_name, "line": line})
                self.tool_calls.append(f"BLOCKED:{tool_name}")
            except IndexError:
                pass
        elif "PROGENT: ALLOWED" in line:
            # Extract tool name from line like: "PROGENT: ALLOWED - fetch_url(url=\"...\")"
            try:
                parts = line.split("PROGENT: ALLOWED - ")[1].split("(")[0]
                tool_name = parts.strip()
                self.allowed_tools.append({"tool": tool_name, "line": line})
                self.tool_calls.append(f"ALLOWED:{tool_name}")
            except IndexError:
                pass

    def has_blocked_tool(self, tool_name: str = None) -> bool:
        """Check if any tool was blocked (or specific tool if provided)."""
        if tool_name:
            return any(t["tool"] == tool_name for t in self.blocked_tools)
        return len(self.blocked_tools) > 0

    def has_allowed_tool(self, tool_name: str = None) -> bool:
        """Check if any tool was allowed (or specific tool if provided)."""
        if tool_name:
            return any(t["tool"] == tool_name for t in self.allowed_tools)
        return len(self.allowed_tools) > 0


class Evaluator:
    """Runs scenarios and reports results."""

    def __init__(
        self,
        agent_cls=LangChainAgent,
        policy_path: str = "implementations/evals/eval_policies.json",
    ):
        self.agent_cls = agent_cls
        self.policy_path = policy_path
        self.results = []
        self.log_stream = None
        self.log_handler = None
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging to capture tool usage for validation."""
        logging.basicConfig(level=logging.INFO)
        # Create a string stream to capture logs
        self.log_stream = io.StringIO()
        self.log_handler = logging.StreamHandler(self.log_stream)
        self.log_handler.setLevel(logging.INFO)

        # Add handler to capture progent and agent logs
        logging.getLogger("progent").addHandler(self.log_handler)
        logging.getLogger("agent").addHandler(self.log_handler)

    def _get_execution_log(self) -> ExecutionLog:
        """Extract execution log from captured logs."""
        exec_log = ExecutionLog()
        if self.log_stream:
            log_content = self.log_stream.getvalue()
            for line in log_content.split("\n"):
                exec_log.parse_log_line(line)
            # Clear the stream for next run
            self.log_stream.truncate(0)
            self.log_stream.seek(0)
        return exec_log

    def run_scenario(self, scenario: Scenario) -> dict:
        """Run a single scenario."""
        print(f"\n--- Running Scenario: {scenario.name} ---")
        print(f"Description: {scenario.description}")
        print(f"Prompt: {scenario.prompt}")

        # Instantiate fresh agent for each run to avoid state pollution
        # We assume a test workspace exists
        agent = self.agent_cls(
            config={"llm": {"model": "deepseek/deepseek-v3.2"}, "agent": {}},
            workspace="./test_workspace",
            policies_path=self.policy_path,
        )

        try:
            response = agent.run(scenario.prompt)
            print(f"Agent Response: {response}")

            # Get execution log with blocked/allowed tools
            exec_log = self._get_execution_log()

            # Run validation logic (pass both response and exec_log if validator accepts it)
            # For now, just use response - we'll enhance validators to use exec_log later
            passed = scenario.validator(response)
            status = "PASSED" if passed else "FAILED"

            result = {
                "name": scenario.name,
                "category": scenario.category,
                "status": status,
                "response": response,
                "details": f"Expected: {scenario.expected_action}",
                "blocked_tools": [t["tool"] for t in exec_log.blocked_tools],
                "allowed_tools": [t["tool"] for t in exec_log.allowed_tools],
            }
            self.results.append(result)
            return result

        except Exception as e:
            print(f"Execution Error: {e}")
            return {
                "name": scenario.name,
                "category": scenario.category,
                "status": "ERROR",
                "response": str(e),
                "details": str(e),
            }

    def generate_report(self):
        """Print a summary table of results."""
        print("\n" + "=" * 60)
        print(f"{'CATEGORY':<15} | {'SCENARIO':<20} | {'STATUS':<10}")
        print("-" * 60)
        for res in self.results:
            print(f"{res['category']:<15} | {res['name']:<20} | {res['status']:<10}")
        print("=" * 60 + "\n")
