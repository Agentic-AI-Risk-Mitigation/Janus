"""
Scripted LLM for demo scenarios.

ScriptedChatModel is a LangChain-compatible BaseChatModel that returns
pre-defined AIMessage responses in sequence. This lets us drive the full
JanusLangChainAgent pipeline (including real policy enforcement) without
making any LLM API calls.

To swap to a real LLM later, just replace chat_model=ScriptedChatModel(...)
with model="openrouter/openai/gpt-4o" in the JanusLangChainAgent constructor.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedChatModel(BaseChatModel):
    """
    A fake ChatModel that returns scripted AIMessage responses in order.

    Usage::

        from langchain_core.messages import AIMessage

        script = [
            AIMessage(content="", tool_calls=[
                {"name": "read_file", "args": {"file_path": "README.md"}, "id": "call_1"}
            ]),
            AIMessage(content="Here is the summary of the repo..."),
        ]
        model = ScriptedChatModel(responses=script)
    """

    responses: list[Any]  # list[AIMessage], using Any to avoid Pydantic issues
    _turn: int = 0

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._turn >= len(self.responses):
            msg = AIMessage(content="[ScriptedChatModel] No more scripted responses.")
            return ChatResult(generations=[ChatGeneration(message=msg)])

        response = self.responses[self._turn]
        self._turn += 1
        return ChatResult(generations=[ChatGeneration(message=response)])

    def reset(self) -> None:
        """Reset the turn counter to replay the script."""
        self._turn = 0

    def bind_tools(self, tools: list, **kwargs: Any) -> ScriptedChatModel:
        """No-op: scripted model doesn't need tool schemas bound."""
        return self
