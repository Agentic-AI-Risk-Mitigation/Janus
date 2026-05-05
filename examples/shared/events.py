"""
Event types streamed from the scenario runner to the frontend via WebSocket.

Each event carries a `panel` tag ("left" or "right") so the frontend knows
where to render it, and a `delay_ms` field that controls playback pacing.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BaseEvent:
    panel: str  # "left" or "right"
    delay_ms: int = 0
    timestamp: float = field(default_factory=time.time)

    @property
    def event_type(self) -> str:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["event_type"] = self.event_type
        return d


@dataclass
class ChatMessage(BaseEvent):
    role: str = ""       # "user" or "assistant"
    content: str = ""
    agent_role: str = ""

    @property
    def event_type(self) -> str:
        return "chat_message"


@dataclass
class ToolCall(BaseEvent):
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""
    agent_role: str = ""

    @property
    def event_type(self) -> str:
        return "tool_call"


@dataclass
class ToolResult(BaseEvent):
    tool: str = ""
    call_id: str = ""
    result: str = ""
    success: bool = True
    agent_role: str = ""

    @property
    def event_type(self) -> str:
        return "tool_result"


@dataclass
class JanusDecision(BaseEvent):
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    allowed: bool = True
    reason: str = ""
    agent_role: str = ""

    @property
    def event_type(self) -> str:
        return "janus_decision"


@dataclass
class AttackEvent(BaseEvent):
    attack_type: str = ""   # "exfiltration", "destruction", "injection", etc.
    detail: str = ""

    @property
    def event_type(self) -> str:
        return "attack_event"


@dataclass
class TaintUpdate(BaseEvent):
    level: int = 0
    source: str = ""
    risk: str = ""

    @property
    def event_type(self) -> str:
        return "taint_update"


@dataclass
class SystemEvent(BaseEvent):
    message: str = ""

    @property
    def event_type(self) -> str:
        return "system_event"
