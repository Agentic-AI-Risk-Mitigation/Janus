"""
Janus × LangChain — Integration Example
========================================

Demonstrates all three integration depths:

  Depth 1  secure_langchain_tools()  — plug Janus tools into your own agent
  Depth 2  wrap_langchain_tools()    — retrofit existing LangChain tools
  Depth 3  JanusLangChainAgent       — turnkey agent

Run:
    uv run python examples/langchain_example/example_langchain.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from janus import ToolDef, ToolParam

# ---------------------------------------------------------------------------
# 1. Define your tools in Janus format (once — used by all frameworks)
# ---------------------------------------------------------------------------

WORKSPACE = Path(__file__).parent / "sandbox"
WORKSPACE.mkdir(exist_ok=True)


def read_file(path: str) -> str:
    full = WORKSPACE / path
    if not full.exists():
        return f"File not found: {path}"
    return full.read_text()


def write_file(path: str, content: str) -> str:
    full = WORKSPACE / path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content)
    return f"Written {len(content)} bytes to {path}"


def run_command(command: str) -> str:
    import subprocess
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout or result.stderr or "(no output)"
    except Exception as exc:
        return f"Error: {exc}"


TOOLS = [
    ToolDef(
        name="read_file",
        description="Read the contents of a file in the sandbox workspace.",
        params=[ToolParam("path", "string", "Relative file path inside the sandbox.")],
        handler=read_file,
    ),
    ToolDef(
        name="write_file",
        description="Write content to a file in the sandbox.",
        params=[
            ToolParam("path", "string", "Relative file path inside the sandbox."),
            ToolParam("content", "string", "Content to write."),
        ],
        handler=write_file,
    ),
    ToolDef(
        name="run_command",
        description="Execute a shell command.",
        params=[ToolParam("command", "string", "The shell command to run.")],
        handler=run_command,
    ),
]

# ---------------------------------------------------------------------------
# 2. Define policy — block dangerous commands, allow safe file ops
# ---------------------------------------------------------------------------

POLICY = {
    "read_file": [{"priority": 1, "effect": 0, "conditions": {}, "fallback": 0}],
    "write_file": [
        {
            "priority": 1,
            "effect": 0,
            "conditions": {
                "path": {"type": "string", "pattern": r"^[a-zA-Z0-9_./-]+$"}
            },
            "fallback": 0,
        }
    ],
    "run_command": [
        {
            "priority": 1,
            "effect": 0,
            "conditions": {
                "command": {"type": "string", "enum": ["ls", "pwd", "echo hello"]}
            },
            "fallback": 0,
        }
    ],
}


# ===========================================================================
# Depth 1 — secure_langchain_tools()
# ===========================================================================
# Use when you build and control your own LangChain agent but want Janus
# to guard every tool call.
# ---------------------------------------------------------------------------

def demo_depth1():
    print("\n" + "=" * 60)
    print(" Depth 1 — secure_langchain_tools()")
    print("=" * 60)

    try:
        from langchain_core.tools import StructuredTool
    except ImportError:
        print("  [skip] langchain-core not installed")
        return

    from janus.adapters.langchain import secure_langchain_tools

    # Convert to LangChain StructuredTools with enforcement baked in
    lc_tools = secure_langchain_tools(TOOLS, POLICY)
    print(f"  Created {len(lc_tools)} secured StructuredTool objects:")
    for t in lc_tools:
        print(f"    - {t.name}")

    # You could now pass lc_tools directly to create_tool_calling_agent:
    #   from langchain.agents import create_tool_calling_agent, AgentExecutor
    #   agent = create_tool_calling_agent(my_llm, lc_tools, my_prompt)
    #   executor = AgentExecutor(agent=agent, tools=lc_tools)

    # Demo: call tools directly to show enforcement
    read_tool = next(t for t in lc_tools if t.name == "read_file")
    write_tool = next(t for t in lc_tools if t.name == "write_file")
    cmd_tool = next(t for t in lc_tools if t.name == "run_command")

    # Allowed write
    r = write_tool.invoke({"path": "hello.txt", "content": "Hello from LangChain example"})
    print(f"\n  write_file (allowed): {r}")

    # Allowed read
    r = read_tool.invoke({"path": "hello.txt"})
    print(f"  read_file  (allowed): {r}")

    # Blocked command (not in enum)
    r = cmd_tool.invoke({"command": "rm -rf /"})
    print(f"  run_command (blocked): {r}")

    # Allowed command
    r = cmd_tool.invoke({"command": "echo hello"})
    print(f"  run_command (allowed): {r.strip()}")


# ===========================================================================
# Depth 2 — wrap_langchain_tools()
# ===========================================================================
# Use when you have existing LangChain tools and want to retrofit Janus
# enforcement without rewriting them as ToolDef objects.
# ---------------------------------------------------------------------------

def demo_depth2():
    print("\n" + "=" * 60)
    print(" Depth 2 — wrap_langchain_tools()")
    print("=" * 60)

    try:
        from langchain_core.tools import StructuredTool
    except ImportError:
        print("  [skip] langchain-core not installed")
        return

    from janus.adapters.langchain import wrap_langchain_tools

    # Pretend these are existing LangChain tools from your codebase
    existing_tool = StructuredTool.from_function(
        func=lambda command: f"EXECUTED: {command}",
        name="run_command",
        description="Runs a command.",
    )

    print(f"  Before wrapping — invoke 'rm -rf /': {existing_tool.invoke({'command': 'rm -rf /'})}")

    # Wrap in-place with Janus enforcement
    wrap_langchain_tools([existing_tool], POLICY)

    result = existing_tool.invoke({"command": "rm -rf /"})
    print(f"  After wrapping  — invoke 'rm -rf /': {result}")

    result = existing_tool.invoke({"command": "echo hello"})
    print(f"  After wrapping  — invoke 'echo hello': {result}")


# ===========================================================================
# Depth 3 — JanusLangChainAgent (turnkey)
# ===========================================================================
# Use for quick setup.  Provide model + tools + policy, call .run().
# ---------------------------------------------------------------------------

def demo_depth3():
    print("\n" + "=" * 60)
    print(" Depth 3 — JanusLangChainAgent (turnkey)")
    print("=" * 60)
    print("  (Requires OPENAI_API_KEY to actually call the LLM)")
    print("  Showing initialization and policy management only.\n")

    try:
        import langchain  # noqa: F401
    except ImportError:
        print("  [skip] langchain not installed")
        return

    from janus.adapters.langchain import JanusLangChainAgent

    # Full agent construction — omit run() so we don't need an API key
    try:
        agent = JanusLangChainAgent(
            model="openai/gpt-4o-mini",
            tools=TOOLS,
            policy=POLICY,
            system_prompt="You are a helpful coding assistant.",
        )
        print(f"  Agent created with tools: {agent.list_tools()}")
        print(f"  Current policy keys: {list(agent.get_policy().keys())}")

        # Runtime policy update
        agent.block_tools(["run_command"])
        print("  Blocked run_command at runtime.")
        print("  agent.enforcer.policy (run_command):", agent.enforcer.policy.get("run_command"))

        # To run interactively:
        #   response = agent.run("Write a hello world script in Python")
        #   print(response)

    except Exception as exc:
        print(f"  Note: {exc}")
        print("  (This is expected without a valid API key)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_depth1()
    demo_depth2()
    demo_depth3()
    print("\nDone.")
