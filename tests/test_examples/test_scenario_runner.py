"""Integration test: ScenarioRunner with Demo 1."""

import asyncio

import pytest

from examples.scenarios import get_scenario
from examples.shared.events import (
    AttackEvent,
    BaseEvent,
    JanusDecision,
    SystemEvent,
)
from examples.shared.scenario_runner import ScenarioRunner


@pytest.fixture
def demo1():
    return get_scenario("demo1_poisoned_readme")


class TestScenarioRunnerDemo1:
    def _run_and_collect(self, scenario, protected: bool) -> list[BaseEvent]:
        events = []

        async def collect(event: BaseEvent):
            events.append(event)

        runner = ScenarioRunner()
        # Override delays for fast testing
        runner._msg_delay = 0
        runner._tool_delay = 0
        runner._block_delay = 0

        asyncio.run(runner.run(scenario, protected=protected, event_callback=collect))
        return events

    def test_unprotected_emits_exfiltration_event(self, demo1):
        events = self._run_and_collect(demo1, protected=False)
        event_types = [e.to_dict()["event_type"] for e in events]

        assert "system_event" in event_types
        assert "chat_message" in event_types
        assert "tool_call" in event_types
        assert "tool_result" in event_types
        assert "attack_event" in event_types

        attack = [e for e in events if isinstance(e, AttackEvent)]
        assert len(attack) >= 1
        assert "exfiltration" in attack[0].attack_type

    def test_protected_blocks_env_and_fetch(self, demo1):
        events = self._run_and_collect(demo1, protected=True)
        event_types = [e.to_dict()["event_type"] for e in events]

        assert "janus_decision" in event_types

        decisions = [e for e in events if isinstance(e, JanusDecision)]
        allows = [d for d in decisions if d.allowed]
        blocks = [d for d in decisions if not d.allowed]

        assert len(allows) >= 3  # list_directory, read README, read main.py
        assert len(blocks) >= 2  # read .env, fetch_url

        blocked_tools = {d.tool for d in blocks}
        assert "read_file" in blocked_tools
        assert "fetch_url" in blocked_tools

    def test_unprotected_no_janus_decisions(self, demo1):
        events = self._run_and_collect(demo1, protected=False)
        decisions = [e for e in events if isinstance(e, JanusDecision)]
        assert len(decisions) == 0

    def test_protected_no_attack_events(self, demo1):
        events = self._run_and_collect(demo1, protected=True)
        attacks = [e for e in events if isinstance(e, AttackEvent)]
        assert len(attacks) == 0

    def test_both_emit_system_events(self, demo1):
        for protected in [True, False]:
            events = self._run_and_collect(demo1, protected=protected)
            system = [e for e in events if isinstance(e, SystemEvent)]
            assert len(system) >= 2  # start + complete
