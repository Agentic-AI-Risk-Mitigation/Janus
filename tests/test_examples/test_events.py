"""Tests for the event system."""

import pytest
from examples.shared.events import (
    ChatMessage, ToolCall, ToolResult, JanusDecision,
    AttackEvent, TaintUpdate, SystemEvent,
)


class TestEventSerialization:
    def test_chat_message_to_dict(self):
        evt = ChatMessage(panel="left", role="user", content="Hello")
        d = evt.to_dict()
        assert d["event_type"] == "chat_message"
        assert d["panel"] == "left"
        assert d["role"] == "user"
        assert d["content"] == "Hello"
        assert "timestamp" in d

    def test_tool_call_to_dict(self):
        evt = ToolCall(panel="right", tool="read_file", args={"file_path": "test.txt"}, call_id="c1")
        d = evt.to_dict()
        assert d["event_type"] == "tool_call"
        assert d["tool"] == "read_file"
        assert d["args"] == {"file_path": "test.txt"}
        assert d["call_id"] == "c1"

    def test_tool_result_to_dict(self):
        evt = ToolResult(panel="left", tool="read_file", call_id="c1", result="content", success=True)
        d = evt.to_dict()
        assert d["event_type"] == "tool_result"
        assert d["success"] is True

    def test_janus_decision_to_dict(self):
        evt = JanusDecision(panel="right", tool="fetch_url", args={"url": "http://x"}, allowed=False, reason="blocked")
        d = evt.to_dict()
        assert d["event_type"] == "janus_decision"
        assert d["allowed"] is False
        assert d["reason"] == "blocked"

    def test_attack_event_to_dict(self):
        evt = AttackEvent(panel="left", attack_type="exfiltration", detail="data sent")
        d = evt.to_dict()
        assert d["event_type"] == "attack_event"
        assert d["attack_type"] == "exfiltration"

    def test_taint_update_to_dict(self):
        evt = TaintUpdate(panel="right", level=40, source="fetch_url", risk="medium")
        d = evt.to_dict()
        assert d["event_type"] == "taint_update"
        assert d["level"] == 40

    def test_system_event_to_dict(self):
        evt = SystemEvent(panel="left", message="Demo started")
        d = evt.to_dict()
        assert d["event_type"] == "system_event"
        assert d["message"] == "Demo started"

    def test_delay_ms_default(self):
        evt = ChatMessage(panel="left", role="user", content="x")
        assert evt.delay_ms == 0
        assert evt.to_dict()["delay_ms"] == 0

    def test_delay_ms_custom(self):
        evt = ChatMessage(panel="left", role="user", content="x", delay_ms=1500)
        assert evt.delay_ms == 1500
