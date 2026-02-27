"""
Janus × Google ADK (Gemini) — Integration Example
===================================================

Demonstrates both integration depths:

  Depth 1  secure_adk_tools()  — plug Janus into your own Gemini loop
  Depth 2  JanusADKAgent       — turnkey agent

Run:
    uv run python examples/adk_example/example_adk.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from janus import ToolDef, ToolParam

# ---------------------------------------------------------------------------
# Tool definitions (same as any Janus project)
# ---------------------------------------------------------------------------

WORKSPACE = Path(__file__).parent / "sandbox"
WORKSPACE.mkdir(exist_ok=True)


def read_file(path: str) -> str:
    full = WORKSPACE / path
    return full.read_text() if full.exists() else f"File not found: {path}"


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

POLICY = {
    "read_file": [{"priority": 1, "effect": 0, "conditions": {}, "fallback": 0}],
    "write_file": [{"priority": 1, "effect": 0, "conditions": {}, "fallback": 0}],
    "run_command": [
        {
            "priority": 1,
            "effect": 0,
            "conditions": {"command": {"type": "string", "enum": ["ls", "pwd", "echo hello"]}},
            "fallback": 0,
        }
    ],
}


# ===========================================================================
# Depth 1 — secure_adk_tools()
# ===========================================================================
# Use when you build your own Gemini chat session and function-calling loop.
# Janus provides secured handlers; you drive the conversation.
# ---------------------------------------------------------------------------

def demo_depth1():
    print("\n" + "=" * 60)
    print(" Depth 1 — secure_adk_tools()")
    print("=" * 60)

    try:
        from google.genai import types as gtypes  # noqa: F401
    except ImportError:
        print("  [skip] google-genai not installed")
        return

    from janus.adapters.adk import secure_adk_tools

    declarations, handlers = secure_adk_tools(TOOLS, POLICY)
    print(f"  Declarations ready for Gemini ({len(declarations)} tools):")
    for d in declarations:
        print(f"    - {d['name']}: {d['description'][:60]}")

    print("\n  Testing handlers directly (no LLM call needed):")

    # Allowed write
    r = handlers["write_file"](path="test.txt", content="Hello from ADK example")
    print(f"  write_file (allowed): {r}")

    # Allowed read
    r = handlers["read_file"](path="test.txt")
    print(f"  read_file  (allowed): {r}")

    # Blocked command
    r = handlers["run_command"](command="rm -rf /")
    print(f"  run_command (blocked): {r}")

    # Allowed command
    r = handlers["run_command"](command="echo hello")
    print(f"  run_command (allowed): {r.strip()}")

    # How to use with a real Gemini client:
    print("""
  Usage with your own Gemini chat loop:

    from google import genai
    from google.genai import types

    client = genai.Client(api_key="GOOGLE_API_KEY")
    config = types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=declarations)],
        system_instruction="You are helpful.",
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    chat = client.chats.create(model="gemini-2.0-flash", config=config)

    response = chat.send_message("List files in the sandbox")
    while response.function_calls:
        fc = response.function_calls[0]
        result = handlers[fc.name](**dict(fc.args))   # <-- Janus guards this
        response = chat.send_message(
            types.Part.from_function_response(fc.name, {"result": result})
        )
    print(response.text)
    """)


# ===========================================================================
# Depth 2 — JanusADKAgent (turnkey)
# ===========================================================================
# Use for quick setup.  Provide model + tools + policy, call .run().
# ---------------------------------------------------------------------------

def demo_depth2():
    print("\n" + "=" * 60)
    print(" Depth 2 — JanusADKAgent (turnkey)")
    print("=" * 60)
    print("  (Requires GOOGLE_API_KEY or GEMINI_API_KEY to actually call Gemini)")
    print("  Showing initialization and handler verification only.\n")

    try:
        from google.genai import types as gtypes  # noqa: F401
    except ImportError:
        print("  [skip] google-genai not installed")
        return

    import os
    from janus.adapters.adk import JanusADKAgent

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("  No API key found. Showing JanusADKAgent API only (no live call).\n")
        print("""
  agent = JanusADKAgent(
      model="gemini-2.0-flash",
      tools=TOOLS,
      policy=POLICY,
      system_prompt="You are a helpful coding assistant.",
      api_key="YOUR_GOOGLE_API_KEY",
  )

  # Run a task — Janus guards every tool call
  response = agent.run("Write a Python hello-world script and show me its content")
  print(response)

  # Multi-turn conversation
  response = agent.run("Now add error handling to the script")
  print(response)

  # Runtime policy updates
  agent.block_tools(["run_command"])
  agent.allow_tools(["read_file", "write_file"])

  # Clear history (new conversation)
  agent.clear_history()
        """)
        return

    agent = JanusADKAgent(
        model="gemini-2.0-flash",
        tools=TOOLS,
        policy=POLICY,
        system_prompt="You are a helpful coding assistant. Use tools only within the sandbox.",
        api_key=api_key,
    )
    print(f"  Agent created with tools: {agent.list_tools()}")

    response = agent.run("Write 'Hello from ADK' to a file called greeting.txt, then read it back.")
    print(f"\n  Agent response:\n{response}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_depth1()
    demo_depth2()
    print("\nDone.")
