"""
Scenario runner -- orchestrates a demo scenario for one panel.

Drives the ScriptedChatModel through the JanusLangChainAgent pipeline,
captures tool calls, enforcement decisions, and attack events, and
emits them as structured events via a callback.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Awaitable, Callable

import yaml

from examples.shared.events import (
    BaseEvent, ChatMessage, ToolCall, ToolResult,
    JanusDecision, AttackEvent, TaintUpdate, SystemEvent,
)
from examples.shared.mock_tools import check_exfiltration, check_malicious_push
from examples.shared.scenario_base import BaseScenario
from examples.shared.scripted_llm import ScriptedChatModel

from janus.adapters.langchain import secure_langchain_tools
from janus.policy.enforcer import PolicyEnforcer
from janus.adapters._base import resolve_enforcer, make_guarded_handler


EventCallback = Callable[[BaseEvent], Awaitable[None]]

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {}


class ScenarioRunner:
    """
    Runs a scenario for a single panel (protected or unprotected).

    Events are emitted asynchronously via the callback, with configurable
    delays between them for playback pacing.
    """

    def __init__(self):
        self._config = _load_config()
        playback = self._config.get("playback", {})
        self._msg_delay = playback.get("delay_between_messages_ms", 1500)
        self._tool_delay = playback.get("delay_between_tool_calls_ms", 800)
        self._block_delay = playback.get("delay_after_block_ms", 1200)

    async def run(
        self,
        scenario: BaseScenario,
        protected: bool,
        event_callback: EventCallback,
    ) -> None:
        """
        Execute a scenario and stream events via the callback.

        Args:
            scenario: The scenario to run.
            protected: If True, run with Janus enforcement. If False, run unprotected.
            event_callback: Async callable receiving BaseEvent instances.
        """
        panel = "right" if protected else "left"
        label = "WITH JANUS" if protected else "WITHOUT JANUS"

        await event_callback(SystemEvent(panel=panel, message=f"Starting: {scenario.title} ({label})"))

        # Use a temp copy of the workspace so real files aren't modified
        # Each panel gets its own isolated copy -- no shared global state
        tmp_dir = tempfile.mkdtemp(prefix=f"janus_demo_{scenario.name}_")
        tmp_workspace = Path(tmp_dir) / "workspace"
        shutil.copytree(scenario.workspace_dir, tmp_workspace)

        # Build enforcer
        enforcer = None
        if protected:
            if scenario.enforcer_type == "pde":
                try:
                    enforcer = self._setup_pde_enforcer(scenario)
                except Exception as exc:
                    await event_callback(SystemEvent(
                        panel=panel,
                        message=(
                            f"⚠️  SpiceDB/PDE not available: {exc}\n"
                            "Start SpiceDB with: docker compose up -d\n"
                            "Running unprotected instead."
                        ),
                    ))
                    enforcer = None
            else:
                policy = scenario.get_policy()
                enforcer = resolve_enforcer(policy)

        # Get tools bound to the temp workspace (closures, no global state)
        tools = scenario.get_tools(workspace_override=tmp_workspace)

        if enforcer is not None and scenario.enforcer_type == "pde":
            # PDE enforcer isn't a PolicyEnforcer -- build LangChain tools manually
            lc_tools = self._build_pde_lc_tools(tools, enforcer)
        elif enforcer is not None:
            lc_tools = secure_langchain_tools(tools, enforcer)
        else:
            lc_tools = secure_langchain_tools(tools, None)

        # Get the script
        script = scenario.get_protected_script() if protected else scenario.get_unprotected_script()

        # Emit user prompt
        await asyncio.sleep(self._msg_delay / 1000)
        await event_callback(ChatMessage(
            panel=panel, role="user", content=scenario.user_prompt,
            delay_ms=self._msg_delay,
        ))

        # Build the scripted model
        model = ScriptedChatModel(responses=script)

        # Drive the agent loop manually to capture intermediate events
        taint_sources = scenario.get_taint_sources()
        await self._drive_agent_loop(
            model=model,
            lc_tools=lc_tools,
            tools_raw=tools,
            enforcer=enforcer,
            scenario=scenario,
            panel=panel,
            protected=protected,
            taint_sources=taint_sources,
            event_callback=event_callback,
        )

        # Clean up temp workspace
        shutil.rmtree(tmp_dir, ignore_errors=True)

        await event_callback(SystemEvent(panel=panel, message="Demo complete"))

    async def _drive_agent_loop(
        self,
        model: ScriptedChatModel,
        lc_tools: list,
        tools_raw: list,
        enforcer: PolicyEnforcer | None,
        scenario: BaseScenario,
        panel: str,
        protected: bool,
        taint_sources: dict[str, str],
        event_callback: EventCallback,
    ) -> None:
        """
        Manually drive the scripted LLM through the tool-calling loop,
        emitting events at each step.

        This avoids using AgentExecutor so we have full control over
        event emission and timing.
        """
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        tool_map = {t.name: t for t in lc_tools}
        messages = [HumanMessage(content=scenario.user_prompt)]
        last_commit_message = ""

        while True:
            await asyncio.sleep(self._tool_delay / 1000)

            result = model._generate(messages)
            ai_msg = result.generations[0].message

            if not ai_msg.tool_calls:
                # Final text response
                await event_callback(ChatMessage(
                    panel=panel, role="assistant", content=ai_msg.content,
                    delay_ms=self._msg_delay,
                ))
                break

            messages.append(ai_msg)

            for tc in ai_msg.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                call_id = tc["id"]

                # Emit tool call event
                await event_callback(ToolCall(
                    panel=panel, tool=tool_name, args=tool_args, call_id=call_id,
                    delay_ms=self._tool_delay,
                ))
                await asyncio.sleep(self._tool_delay / 1000)

                # Execute via the secured (or unsecured) LangChain tool
                lc_tool = tool_map.get(tool_name)
                if lc_tool is None:
                    result_str = f"Tool '{tool_name}' not found."
                    success = False
                else:
                    try:
                        result_str = lc_tool.invoke(tool_args)
                        success = True
                    except Exception as exc:
                        result_str = f"Error: {type(exc).__name__}: {exc}"
                        success = False

                blocked = isinstance(result_str, str) and result_str.startswith("[Janus]")

                # Emit Janus decision for protected panel
                if protected:
                    if blocked:
                        reason = result_str.replace("[Janus] ", "")
                        await event_callback(JanusDecision(
                            panel=panel, tool=tool_name, args=tool_args,
                            allowed=False, reason=reason,
                            delay_ms=self._block_delay,
                        ))
                    else:
                        await event_callback(JanusDecision(
                            panel=panel, tool=tool_name, args=tool_args,
                            allowed=True, reason="",
                            delay_ms=self._tool_delay,
                        ))

                # Emit tool result
                display_result = result_str
                if len(result_str) > 300:
                    display_result = result_str[:300] + "..."
                await event_callback(ToolResult(
                    panel=panel, tool=tool_name, call_id=call_id,
                    result=display_result, success=success and not blocked,
                    delay_ms=self._tool_delay,
                ))

                # Track last commit message for git_push detection
                if tool_name == "git_commit":
                    last_commit_message = tool_args.get("message", "")

                # Check for attacks on unprotected panel
                if not protected and not blocked:
                    if tool_name == "fetch_url":
                        exfil = check_exfiltration(tool_args.get("url", ""))
                        if exfil:
                            await event_callback(AttackEvent(
                                panel=panel,
                                attack_type=exfil["attack_type"],
                                detail=exfil["detail"],
                                delay_ms=self._block_delay,
                            ))
                    elif tool_name == "git_push":
                        attack = check_malicious_push(commit_message=last_commit_message)
                        if attack:
                            await event_callback(AttackEvent(
                                panel=panel,
                                attack_type=attack["attack_type"],
                                detail=attack["detail"],
                                delay_ms=self._block_delay,
                            ))
                    else:
                        # Per-scenario attack detection (e.g. write_file backdoor)
                        attack = scenario.check_attack(tool_name, tool_args)
                        if attack:
                            await event_callback(AttackEvent(
                                panel=panel,
                                attack_type=attack["attack_type"],
                                detail=attack["detail"],
                                delay_ms=self._block_delay,
                            ))

                # Update taint for PDE scenarios
                if protected and call_id in taint_sources and not blocked:
                    risk = taint_sources[call_id]
                    if hasattr(enforcer, "update_taint"):
                        from policy_engine.main import RISK_TO_TAINT
                        taint_value = RISK_TO_TAINT.get(risk, 50)
                        enforcer.update_taint(taint_value)
                        current_taint = enforcer.interceptor.current_taint_level
                        await event_callback(TaintUpdate(
                            panel=panel, level=current_taint,
                            source=tool_name, risk=risk,
                            delay_ms=self._tool_delay,
                        ))

                # Feed tool result back to the model
                messages.append(ToolMessage(
                    content=result_str,
                    tool_call_id=call_id,
                ))

    @staticmethod
    def _build_pde_lc_tools(tools: list, enforcer) -> list:
        """Build LangChain StructuredTool list with PDE enforcement."""
        from langchain_core.tools import StructuredTool
        from janus.exceptions import PolicyViolation

        lc_tools = []
        for tool_def in tools:
            def _make_guarded(td, enf):
                def guarded(**kwargs):
                    try:
                        enf.enforce(td.name, kwargs)
                    except PolicyViolation as exc:
                        return f"[Janus] Tool '{td.name}' was blocked by policy: {exc.reason}"
                    except Exception as exc:
                        return f"[Janus] Tool '{td.name}' enforcement error: {type(exc).__name__}: {exc}"
                    try:
                        result = td.handler(**kwargs)
                        return str(result) if not isinstance(result, str) else result
                    except Exception as exc:
                        return f"Tool '{td.name}' raised an error: {type(exc).__name__}: {exc}"
                return guarded

            lc_tool = StructuredTool(
                name=tool_def.name,
                description=tool_def.description,
                func=_make_guarded(tool_def, enforcer),
                args_schema=tool_def.to_pydantic_model(),
            )
            lc_tools.append(lc_tool)
        return lc_tools

    def _setup_pde_enforcer(self, scenario: BaseScenario):
        """Set up the PDE enforcer for SpiceDB-backed scenarios."""
        from janus.policy.pde_enforcer import PDEEnforcer
        enforcer = PDEEnforcer(agent_role="coding_agent")

        if hasattr(scenario, "bootstrap_pde"):
            scenario.bootstrap_pde(enforcer)

        return enforcer
