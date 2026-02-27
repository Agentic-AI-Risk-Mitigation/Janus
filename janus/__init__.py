"""
Janus — System-level security for LLM agents.

Janus enforces fine-grained security policies on every tool call an LLM agent
makes, following the principle of least privilege. Policies are defined in JSON
(or generated automatically by an LLM) and validated against JSON Schema
restrictions at runtime.

Quick start
-----------
::

    from janus import JanusAgent, ToolDef, ToolParam

    agent = JanusAgent(
        model="openai/gpt-4o",
        api_key="sk-...",
        policy="policies.json",
        system_prompt="You are a helpful coding assistant.",
    )

    response = agent.run("List the Python files in the project.")

Using built-in file/command tools
----------------------------------
::

    from janus import JanusAgent
    from janus.tools.builtin import BUILTIN_TOOLS

    agent = JanusAgent(
        model="openai/gpt-4o",
        use_builtin_tools=True,   # default
        policy={"read_file": [{"priority": 1, "effect": 0, "conditions": {}, "fallback": 0}]},
    )

Custom tools
------------
::

    from janus import JanusAgent, ToolDef, ToolParam

    def my_tool(query: str) -> str:
        return f"Results for: {query}"

    agent = JanusAgent(
        model="anthropic/claude-3-5-sonnet-20241022",
        tools=[
            ToolDef(
                name="search",
                description="Search for information",
                params=[ToolParam("query", "string", "Search query")],
                handler=my_tool,
            )
        ],
        policy={"search": [{"priority": 1, "effect": 0, "conditions": {}, "fallback": 0}]},
    )

LLM-generated policies
-----------------------
::

    agent = JanusAgent(
        model="openai/gpt-4o",
        policy="generate",           # generate on first run()
        policy_model="openai/gpt-4o",
    )

Standalone policy enforcement (for framework integrations)
-----------------------------------------------------------
::

    from janus.policy import PolicyEnforcer

    enforcer = PolicyEnforcer()
    enforcer.load("policies.json")
    enforcer.enforce("run_command", {"command": "ls"})  # OK or raises PolicyViolation
"""

from janus.agent import JanusAgent

# Adapters are imported lazily (optional dependencies) — access via submodule:
#   from janus.adapters.langchain import secure_langchain_tools, JanusLangChainAgent
#   from janus.adapters.adk import secure_adk_tools, JanusADKAgent

from janus.exceptions import (
    ArgumentValidationError,
    JanusError,
    PolicyGenerationError,
    PolicyLoadError,
    PolicyViolation,
    ProviderError,
    ToolNotFoundError,
)
from janus.logger import configure_logging, get_logger
from janus.policy.enforcer import PolicyEnforcer
from janus.policy.generator import generate_policy, refine_policy
from janus.policy.loader import parse_policy, save_policy
from janus.tools.base import ToolDef, ToolParam
from janus.tools.builtin import BUILTIN_TOOLS
from janus.tools.registry import ToolRegistry

__version__ = "0.1.0"

__all__ = [
    # Main entry point
    "JanusAgent",
    # Tool types
    "ToolDef",
    "ToolParam",
    "ToolRegistry",
    "BUILTIN_TOOLS",
    # Policy
    "PolicyEnforcer",
    "generate_policy",
    "refine_policy",
    "parse_policy",
    "save_policy",
    # Logging
    "configure_logging",
    "get_logger",
    # Exceptions
    "JanusError",
    "PolicyViolation",
    "ArgumentValidationError",
    "PolicyLoadError",
    "PolicyGenerationError",
    "ToolNotFoundError",
    "ProviderError",
    # Version
    "__version__",
]
