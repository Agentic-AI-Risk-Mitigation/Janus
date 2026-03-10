"""Tests for the ScriptedChatModel."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from examples.shared.scripted_llm import ScriptedChatModel


class TestScriptedChatModel:
    def test_returns_responses_in_order(self):
        script = [
            AIMessage(content="First"),
            AIMessage(content="Second"),
            AIMessage(content="Third"),
        ]
        model = ScriptedChatModel(responses=script)

        r1 = model._generate([HumanMessage(content="hi")])
        assert r1.generations[0].message.content == "First"

        r2 = model._generate([HumanMessage(content="hi")])
        assert r2.generations[0].message.content == "Second"

        r3 = model._generate([HumanMessage(content="hi")])
        assert r3.generations[0].message.content == "Third"

    def test_tool_calls_preserved(self):
        script = [
            AIMessage(content="", tool_calls=[
                {"name": "read_file", "args": {"file_path": "test.txt"}, "id": "c1"}
            ]),
        ]
        model = ScriptedChatModel(responses=script)
        result = model._generate([HumanMessage(content="read")])
        msg = result.generations[0].message
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "read_file"

    def test_exhausted_script(self):
        script = [AIMessage(content="Only one")]
        model = ScriptedChatModel(responses=script)
        model._generate([HumanMessage(content="1")])
        result = model._generate([HumanMessage(content="2")])
        assert "No more scripted responses" in result.generations[0].message.content

    def test_reset(self):
        script = [AIMessage(content="First"), AIMessage(content="Second")]
        model = ScriptedChatModel(responses=script)
        model._generate([HumanMessage(content="1")])
        model.reset()
        result = model._generate([HumanMessage(content="1")])
        assert result.generations[0].message.content == "First"

    def test_llm_type(self):
        model = ScriptedChatModel(responses=[])
        assert model._llm_type == "scripted"

    def test_bind_tools_noop(self):
        model = ScriptedChatModel(responses=[])
        result = model.bind_tools([{"name": "foo"}])
        assert result is model
