"""
Janus × LangChain integration adapter.

Provides three integration depths so users can choose how deeply Janus
plugs into their LangChain setup:

┌─────────────────────────────────────────────────────────────────────┐
│  Depth 1 — secure_langchain_tools()                                  │
│  Convert Janus ToolDef list → LangChain StructuredTool list with    │
│  policy enforcement baked into each tool's handler.                  │
│  Use when: you build your own agent but want Janus-guarded tools.   │
├─────────────────────────────────────────────────────────────────────┤
│  Depth 2 — wrap_langchain_tools()                                    │
│  Add Janus enforcement to tools you already defined as LangChain    │
│  BaseTool / StructuredTool objects (migration / retrofit path).      │
│  Use when: you have an existing LangChain codebase.                  │
├─────────────────────────────────────────────────────────────────────┤
│  Depth 3 — JanusLangChainAgent                                       │
│  Turnkey agent: provide model + tools + policy, call .run().         │
│  Internally uses Depth 1. You get the LangChain AgentExecutor and   │
│  full conversation history management, wrapped in the Janus API.     │
└─────────────────────────────────────────────────────────────────────┘

Installation requirement:
    uv add langchain langchain-core langchain-openai

For Anthropic / Google / other providers, install the matching
langchain-<provider> package and pass a pre-built ``chat_model``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from janus.adapters._base import PolicySource, make_guarded_handler, resolve_enforcer
from janus.exceptions import PolicyViolation
from janus.logger import get_logger
from janus.policy.enforcer import PolicyEnforcer
from janus.tools.base import ToolDef

try:
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import BaseTool, StructuredTool

    _LC_AVAILABLE = True
except ImportError:
    _LC_AVAILABLE = False

_logger = get_logger()


def _require_langchain() -> None:
    if not _LC_AVAILABLE:
        raise ImportError(
            "LangChain is required for this adapter.\n"
            "Install with: uv add langchain langchain-core langchain-openai"
        )


# =============================================================================
# Depth 1 — Convert ToolDef list → Secured StructuredTool list
# =============================================================================


def secure_langchain_tools(
    tools: list[ToolDef],
    policy: PolicySource,
) -> list["StructuredTool"]:
    """
    Convert a list of Janus ``ToolDef`` objects into LangChain
    ``StructuredTool`` instances with Janus policy enforcement built in.

    Every tool call is intercepted by the ``PolicyEnforcer`` before the
    underlying handler runs.  Blocked calls return a descriptive error
    string so the LLM can reason about the refusal instead of crashing.

    Args:
        tools:  List of ``ToolDef`` objects describing the tools.
        policy: Policy source — a JSON file path, a policy dict, an existing
                ``PolicyEnforcer`` instance, or ``None`` (no enforcement).

    Returns:
        List of LangChain ``StructuredTool`` objects ready to pass to
        ``create_tool_calling_agent`` or any ``AgentExecutor``.

    Example::

        from janus.adapters.langchain import secure_langchain_tools
        from langchain.agents import AgentExecutor, create_tool_calling_agent

        lc_tools = secure_langchain_tools(my_janus_tools, "policies.json")

        agent = create_tool_calling_agent(llm, lc_tools, prompt)
        executor = AgentExecutor(agent=agent, tools=lc_tools)
    """
    _require_langchain()
    enforcer = resolve_enforcer(policy)

    lc_tools: list[StructuredTool] = []
    for tool_def in tools:
        guarded = make_guarded_handler(tool_def.name, tool_def.handler, enforcer)
        args_schema = tool_def.to_pydantic_model()

        lc_tool = StructuredTool(
            name=tool_def.name,
            description=tool_def.description,
            func=guarded,
            args_schema=args_schema,
        )
        lc_tools.append(lc_tool)

    _logger.info(f"[langchain adapter] Secured {len(lc_tools)} tools.")
    return lc_tools


# =============================================================================
# Depth 2 — Wrap existing LangChain BaseTool instances
# =============================================================================


def wrap_langchain_tools(
    lc_tools: list["BaseTool"],
    policy: PolicySource,
) -> list["BaseTool"]:
    """
    Add Janus policy enforcement to tools already defined as LangChain
    ``BaseTool`` / ``StructuredTool`` objects.

    This is the *retrofit* path — useful when you have an existing LangChain
    codebase and want to add Janus security without rewriting tool definitions.
    Each tool's underlying callable is replaced with a guarded version;
    the tool's name, description, and schema are preserved.

    Args:
        lc_tools: List of existing LangChain ``BaseTool`` instances.
        policy:   Policy source (same as ``secure_langchain_tools``).

    Returns:
        The same list, mutated in-place, with enforcement added.

    Example::

        from janus.adapters.langchain import wrap_langchain_tools

        # tools is your existing list of LangChain tools
        tools = wrap_langchain_tools(tools, "policies.json")
        # Pass tools to your existing AgentExecutor as usual
    """
    _require_langchain()
    enforcer = resolve_enforcer(policy)

    for tool in lc_tools:
        if not isinstance(tool, BaseTool):
            _logger.warning(f"Skipping non-BaseTool object: {type(tool).__name__}")
            continue
        _wrap_existing_tool(tool, enforcer)

    _logger.info(f"[langchain adapter] Wrapped {len(lc_tools)} existing tools with Janus enforcement.")
    return lc_tools


def _wrap_existing_tool(tool: "BaseTool", enforcer: PolicyEnforcer) -> None:
    """
    Retrofit a single LangChain BaseTool with Janus enforcement.

    Strategy: extract the underlying Python callable from the tool object,
    wrap it with the guarded handler, and replace the tool's ``func``
    attribute.  For custom ``BaseTool`` subclasses that override ``_run``
    directly (without a ``func`` attribute), we patch ``_run`` instead.
    """
    tool_name = tool.name

    # StructuredTool and most factory-created tools expose .func
    if hasattr(tool, "func") and callable(tool.func):
        original_func = tool.func
        tool.func = make_guarded_handler(tool_name, original_func, enforcer)
        return

    # Fallback: patch _run (used by custom BaseTool subclasses)
    # We create a bound-method-style wrapper via a closure.
    original_run = tool.__class__._run

    def patched_run(self, *args, **kwargs):
        # Keyword args are the tool's input params
        try:
            enforcer.enforce(self.name, kwargs)
        except PolicyViolation as exc:
            return f"[Janus] Tool '{self.name}' was blocked by policy: {exc.reason}"
        return original_run(self, *args, **kwargs)

    # Bind to the instance so it shadows the class method
    import types as _types
    tool._run = _types.MethodType(patched_run, tool)


# =============================================================================
# Depth 3 — JanusLangChainAgent (turnkey)
# =============================================================================


class JanusLangChainAgent:
    """
    A policy-enforced LangChain agent.

    Janus handles the security layer; LangChain's ``AgentExecutor`` drives
    the reasoning and tool-calling loop.

    Parameters
    ----------
    model : str
        Model string in ``<provider>/<model-name>`` format.
        Examples:
          - ``"openai/gpt-4o"``
          - ``"openrouter/anthropic/claude-3.5-sonnet"``
          - ``"ollama/llama3.2"``
    tools : list[ToolDef]
        Janus tool definitions (converted to ``StructuredTool`` internally).
    policy : str | Path | dict | PolicyEnforcer | None
        Policy source.  Pass a JSON file path, dict, an existing enforcer,
        or ``None`` to run without enforcement.
    system_prompt : str
        System message for the agent.
    api_key : str | None
        API key.  Falls back to the provider's default env var.
    chat_model : BaseChatModel | None
        If provided, this pre-configured LangChain chat model is used
        directly and ``model`` / ``api_key`` are ignored.  Useful when you
        want to configure the LLM yourself (e.g. with custom parameters).
    max_iterations : int
        Maximum tool-call cycles per ``run()`` call.
    verbose : bool
        Whether to enable LangChain's verbose logging.
    **provider_kwargs
        Forwarded to the LangChain chat model constructor (e.g. ``temperature``,
        ``base_url``).

    Attributes
    ----------
    enforcer : PolicyEnforcer
        The active policy enforcer.  Inspect or update it at runtime.
    lc_tools : list[StructuredTool]
        The secured LangChain tools.
    agent_executor : AgentExecutor
        The underlying LangChain executor.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        tools: list[ToolDef] | None = None,
        policy: PolicySource = None,
        system_prompt: str = "You are a helpful assistant.",
        api_key: str | None = None,
        chat_model: Any = None,
        max_iterations: int = 10,
        verbose: bool = False,
        **provider_kwargs: Any,
    ):
        _require_langchain()
        # AgentExecutor moved between LangChain versions; try both locations
        try:
            from langchain.agents import AgentExecutor, create_tool_calling_agent
        except ImportError:
            from langchain_core.agents import AgentExecutor  # type: ignore[no-redef]
            from langchain_core.agents import create_tool_calling_agent  # type: ignore[no-redef]
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        self.enforcer = resolve_enforcer(policy)
        self._logger = get_logger()

        # Build secured LangChain tools
        self.lc_tools = secure_langchain_tools(tools or [], self.enforcer)

        # LLM — use provided chat_model or build one from model string
        if chat_model is not None:
            llm = chat_model
        else:
            llm = _build_chat_model(model, api_key, **provider_kwargs)

        # Prompt with conversation history slot
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(llm, self.lc_tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.lc_tools,
            verbose=verbose,
            handle_parsing_errors=True,
            max_iterations=max_iterations,
            return_intermediate_steps=True,
        )

        self._chat_history: list = []
        self._logger.agent_event(
            "INIT",
            f"framework=langchain model={model} tools={[t.name for t in (tools or [])]}",
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> str:
        """
        Send a user message and return the agent's response.

        Args:
            user_input: The user's message or task.

        Returns:
            The agent's final text response.
        """
        self._logger.agent_event("USER_INPUT", user_input[:120])
        try:
            result = self.agent_executor.invoke({
                "input": user_input,
                "chat_history": self._chat_history,
            })

            response = result.get("output", "No response generated.")

            for i, (action, _) in enumerate(result.get("intermediate_steps", []), 1):
                self._logger.debug(f"Step {i}: {action.tool}({action.tool_input})")

            self._chat_history.append(HumanMessage(content=user_input))
            self._chat_history.append(AIMessage(content=response))

            self._logger.agent_event("RESPONSE", response[:120])
            return response

        except Exception as exc:
            error = f"Agent error: {type(exc).__name__}: {exc}"
            self._logger.error(error)
            return error

    def clear_history(self) -> None:
        """Reset conversation history."""
        self._chat_history = []
        self._logger.agent_event("HISTORY_CLEARED")

    def list_tools(self) -> list[str]:
        """Return the names of all registered tools."""
        return [t.name for t in self.lc_tools]

    def get_policy(self) -> dict | None:
        """Return the current policy dict."""
        return self.enforcer.policy

    def allow_tools(self, tools: list[str]) -> None:
        """Unconditionally allow the given tools."""
        self.enforcer.allow_tools(tools)

    def block_tools(self, tools: list[str]) -> None:
        """Unconditionally block the given tools."""
        self.enforcer.block_tools(tools)


# =============================================================================
# Internal helpers
# =============================================================================


def _build_chat_model(model: str, api_key: str | None, **kwargs: Any) -> Any:
    """
    Build a LangChain chat model from a ``"provider/model-name"`` string.

    Supported providers:
    - ``openai``      → ChatOpenAI
    - ``anthropic``   → ChatAnthropic   (requires langchain-anthropic)
    - ``google`` / ``gemini`` → ChatGoogleGenerativeAI (requires langchain-google-genai)
    - ``azure``       → AzureChatOpenAI (requires langchain-openai)
    - ``openrouter``  → ChatOpenAI with OpenRouter base_url
    - ``ollama``      → ChatOpenAI with Ollama base_url
    - ``together``    → ChatOpenAI with Together AI base_url
    - ``vllm``        → ChatOpenAI with vLLM base_url
    """
    import os

    if "/" not in model:
        raise ValueError(
            f"Invalid model string '{model}'. Expected '<provider>/<model-name>'."
        )

    provider, _, model_name = model.partition("/")
    provider = provider.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            **kwargs,
        )

    if provider in ("anthropic",):
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError("Install langchain-anthropic: uv add langchain-anthropic")
        return ChatAnthropic(
            model=model_name,
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            **kwargs,
        )

    if provider in ("google", "gemini"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("Install langchain-google-genai: uv add langchain-google-genai")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"),
            **kwargs,
        )

    if provider == "azure":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=model_name,
            api_key=api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
            azure_endpoint=kwargs.pop("base_url", os.environ.get("AZURE_OPENAI_ENDPOINT", "")),
            api_version=kwargs.pop("api_version", os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")),
            **kwargs,
        )

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        # model_name may itself contain "/" (e.g. "anthropic/claude-3.5-sonnet")
        return ChatOpenAI(
            model=model_name,
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            base_url=kwargs.pop("base_url", "https://openrouter.ai/api/v1"),
            **kwargs,
        )

    if provider == "ollama":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key="ollama",
            base_url=kwargs.pop("base_url", os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")),
            **kwargs,
        )

    if provider == "together":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=api_key or os.environ.get("TOGETHER_API_KEY"),
            base_url=kwargs.pop("base_url", "https://api.together.xyz/v1"),
            **kwargs,
        )

    if provider == "vllm":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=api_key or os.environ.get("VLLM_API_KEY", "vllm"),
            base_url=kwargs.pop("base_url", os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")),
            **kwargs,
        )

    # Unknown provider — try ChatOpenAI as a fallback (OpenAI-compatible)
    _logger.warning(
        f"Unknown LangChain provider '{provider}'. "
        "Falling back to ChatOpenAI — set base_url in provider_kwargs if needed."
    )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model_name, api_key=api_key, **kwargs)
