"""
JanusAgent — the primary entry point for the Janus library.

JanusAgent wires together the policy enforcer, tool registry, LLM provider,
and conversation runner into a single object that callers can set up with
minimal boilerplate.

Quick start::

    from janus import JanusAgent, ToolDef, ToolParam

    agent = JanusAgent(
        model="openai/gpt-4o",
        api_key="sk-...",
        tools=[
            ToolDef(
                name="read_file",
                description="Read a file",
                params=[ToolParam("file_path", "string", "Path to read")],
                handler=my_read_file,
            )
        ],
        policy="path/to/policy.json",  # or a dict, or "generate"
        system_prompt="You are a helpful coding assistant.",
    )

    response = agent.run("Summarize the file report.txt")

The ``policy`` argument accepts:
    - A file path (str or Path) to a JSON policy file.
    - A dict in Janus/Progent policy format.
    - The string ``"generate"`` to auto-generate policies from the user's
      first query using an LLM (requires ``policy_model``).
    - ``None`` to run without a policy (all tools allowed — not recommended
      for production).

For framework integrations (LangChain, ADK, etc.), the components that
JanusAgent orchestrates are all independently accessible:
    - ``agent.enforcer``   → PolicyEnforcer
    - ``agent.registry``   → ToolRegistry
    - ``agent.runner``     → LLMRunner
"""

from pathlib import Path
from typing import Any

from janus.exceptions import PolicyGenerationError
from janus.llm.providers import get_provider
from janus.llm.runner import LLMRunner
from janus.logger import configure_logging, get_logger
from janus.policy.enforcer import PolicyEnforcer
from janus.policy.loader import parse_policy, save_policy
from janus.tools.builtin import BUILTIN_TOOLS
from janus.tools.builtin.file_tools import set_workspace
from janus.tools.registry import ToolRegistry
from janus.tools.base import ToolDef


_GENERATE_SENTINEL = "generate"


class JanusAgent:
    """
    A policy-enforcing LLM agent.

    Parameters
    ----------
    model : str
        Model string in ``<provider>/<model-name>`` format.
        Examples: ``"openai/gpt-4o"``, ``"anthropic/claude-3-5-sonnet-20241022"``,
        ``"google/gemini-2.0-flash"``, ``"ollama/llama3.2"``.
    system_prompt : str
        System message shown to the LLM at the start of every conversation.
    tools : list[ToolDef] | None
        Tool definitions to register. If None and ``use_builtin_tools=True``,
        the built-in file/command tools are used.
    use_builtin_tools : bool
        If True (default), the built-in file and command tools are registered
        in addition to any ``tools`` provided.
    policy : str | Path | dict | None
        Security policy source. Pass a file path, a dict, the string
        ``"generate"`` for LLM-based generation, or None for no policy.
    policy_model : str | None
        Model string used for LLM-based policy generation (when
        ``policy="generate"``). Defaults to ``"openai/gpt-4o-2024-08-06"``.
    api_key : str | None
        API key for the main model's provider. If omitted, the provider reads
        from the appropriate environment variable (e.g. ``OPENAI_API_KEY``).
    workspace : str | Path | None
        Root directory for file-system tools. Defaults to the current
        working directory.
    max_tool_iterations : int
        Maximum tool-call → response cycles per ``run()`` call.
    temperature : float
        Sampling temperature for the main model.
    log_level : str | None
        Configure Janus logging (``"DEBUG"``, ``"INFO"``, ``"WARNING"``).
        If None, logging is left unconfigured (use ``configure_logging()``
        explicitly if needed).
    **provider_kwargs
        Additional keyword arguments forwarded to the provider constructor
        (e.g. ``base_url``, ``api_version``, ``region``).
    """

    def __init__(
        self,
        model: str,
        system_prompt: str = "You are a helpful assistant.",
        tools: list[ToolDef] | None = None,
        use_builtin_tools: bool = True,
        policy: "str | Path | dict | None" = None,
        policy_model: str | None = None,
        api_key: str | None = None,
        workspace: "str | Path | None" = None,
        max_tool_iterations: int = 10,
        temperature: float = 0.1,
        log_level: str | None = "INFO",
        **provider_kwargs: Any,
    ):
        if log_level:
            configure_logging(level=log_level)

        self._logger = get_logger()
        self._policy_source = policy
        self._policy_model = policy_model
        self._generate_policy_on_first_run = (policy == _GENERATE_SENTINEL)

        # Workspace (used by file tools)
        ws = Path(workspace) if workspace else Path.cwd()
        set_workspace(ws)

        # Policy enforcer
        self.enforcer = PolicyEnforcer()
        if policy and policy != _GENERATE_SENTINEL:
            self.enforcer.load(policy)

        # Tool registry
        self.registry = ToolRegistry(enforcer=self.enforcer)

        if use_builtin_tools:
            self.registry.register_many(BUILTIN_TOOLS)

        if tools:
            self.registry.register_many(tools)

        # LLM provider + runner
        provider_name, model_name = _parse_model_string(model)
        provider_instance = get_provider(provider_name, api_key=api_key, **provider_kwargs)

        self.runner = LLMRunner(
            provider=provider_instance,
            model_name=model_name,
            registry=self.registry,
            system_prompt=system_prompt,
            max_tool_iterations=max_tool_iterations,
            temperature=temperature,
        )

        self._logger.agent_event(
            "INIT",
            f"model={model} tools={self.registry.names()} policy={'none' if not self.enforcer.has_policy() and not self._generate_policy_on_first_run else 'loaded'}",
        )

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> str:
        """
        Run the agent with a user message and return the response.

        If the agent was initialized with ``policy="generate"``, the policy
        is generated on the first call using the user's query and the
        registered tools.

        Args:
            user_input: The user's message or task description.

        Returns:
            The agent's final text response.
        """
        if self._generate_policy_on_first_run:
            self._generate_and_apply_policy(user_input)
            self._generate_policy_on_first_run = False

        return self.runner.run(user_input)

    def clear_history(self) -> None:
        """Reset the conversation history (keeps policy and tools intact)."""
        self.runner.clear_history()

    # ------------------------------------------------------------------
    # Policy management
    # ------------------------------------------------------------------

    def set_policy(self, policy: "str | Path | dict") -> None:
        """
        Load or replace the current security policy.

        Args:
            policy: File path, dict, or ``"generate"`` sentinel.
        """
        if policy == _GENERATE_SENTINEL:
            self._generate_policy_on_first_run = True
            return

        self.enforcer.load(policy)
        self._logger.agent_event("POLICY_UPDATED")

    def get_policy(self) -> dict | None:
        """Return the current enforced policy dict, or None if unset."""
        return self.enforcer.policy

    def save_policy(self, path: "str | Path") -> None:
        """
        Persist the current policy to a JSON file.

        Args:
            path: Destination file path.
        """
        policy = self.enforcer.policy
        if policy is None:
            raise ValueError("No policy is currently loaded.")
        save_policy(policy, path)
        self._logger.agent_event("POLICY_SAVED", str(path))

    def allow_tools(self, tools: list[str]) -> None:
        """Grant unconditional allow (highest priority) to the specified tools."""
        self.enforcer.allow_tools(tools)

    def block_tools(self, tools: list[str]) -> None:
        """Apply an unconditional deny (highest priority) to the specified tools."""
        self.enforcer.block_tools(tools)

    # ------------------------------------------------------------------
    # Tool management
    # ------------------------------------------------------------------

    def add_tool(self, tool: ToolDef) -> None:
        """Register an additional tool at runtime."""
        self.registry.register(tool)

    def remove_tool(self, name: str) -> None:
        """Unregister a tool by name."""
        self.registry.unregister(name)

    def list_tools(self) -> list[str]:
        """Return the names of all currently registered tools."""
        return self.registry.names()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_and_apply_policy(self, query: str) -> None:
        """Call the policy generator and apply the result to the enforcer."""
        from janus.policy.generator import generate_policy

        self._logger.agent_event("POLICY_GENERATING", f"query={query[:80]}")
        tool_specs = self.registry.to_janus_specs()

        try:
            generated = generate_policy(
                query=query,
                tools=tool_specs,
                model=self._policy_model,
            )
            if generated:
                self.enforcer.update(generated)
                self._logger.agent_event(
                    "POLICY_GENERATED",
                    f"tools={list(generated.keys())}",
                )
            else:
                self._logger.warning("Policy generation returned an empty policy.")
        except PolicyGenerationError as exc:
            self._logger.error(f"Policy generation failed: {exc}")


def _parse_model_string(model: str) -> tuple[str, str]:
    """
    Split a ``"provider/model-name"`` string.

    For multi-part model names like ``"vllm/meta-llama/Llama-3.3-70B-Instruct"``
    the provider is the first segment and the model name is the rest.

    Args:
        model: Full model string.

    Returns:
        ``(provider_name, model_name)`` tuple.

    Raises:
        ValueError: If the string does not contain a ``/``.
    """
    if "/" not in model:
        raise ValueError(
            f"Invalid model string: '{model}'. "
            "Expected format: '<provider>/<model-name>', e.g. 'openai/gpt-4o'."
        )
    provider, _, model_name = model.partition("/")
    return provider, model_name
