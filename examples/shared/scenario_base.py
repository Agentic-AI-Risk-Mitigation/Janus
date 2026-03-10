"""
Abstract base class for all demo scenarios.

Each scenario defines its workspace files, policy, tools, system prompt,
and scripted LLM responses for both protected and unprotected runs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseScenario(ABC):
    """
    Subclass this for each demo scenario.

    The scenario runner uses these methods to configure the agent,
    load the policy, and drive the scripted conversation.
    """

    name: str = ""
    title: str = ""
    description: str = ""
    user_prompt: str = ""
    enforcer_type: str = "janus"  # "janus" or "pde"

    @property
    @abstractmethod
    def workspace_dir(self) -> Path:
        """Absolute path to the workspace/ directory for this scenario."""
        ...

    @abstractmethod
    def get_tools(self, workspace_override: Path | None = None) -> list:
        """Return list of ToolDef objects for this scenario.

        Args:
            workspace_override: If provided, use this path instead of the
                default workspace_dir. The scenario runner passes a temp
                copy here so concurrent panels don't share state.
        """
        ...

    @abstractmethod
    def get_policy(self) -> dict | str:
        """Return a policy dict or path to a policy JSON file."""
        ...

    @abstractmethod
    def get_unprotected_script(self) -> list:
        """Return list of AIMessage objects for the unprotected (no Janus) run."""
        ...

    @abstractmethod
    def get_protected_script(self) -> list:
        """Return list of AIMessage objects for the Janus-protected run."""
        ...

    def get_system_prompt(self) -> str:
        """System prompt for the agent. Override per scenario."""
        return "You are a helpful coding assistant with access to file and command tools."

    def get_url_responses(self) -> dict[str, str]:
        """Map URL regex patterns to fake responses for fetch_url.
        Override per scenario to provide custom mocked URL responses.
        """
        return {}

    def get_taint_sources(self) -> dict[str, str]:
        """
        Map of tool_call_id -> risk level for PDE taint tracking.
        Only relevant for PDE scenarios.
        Returns e.g. {"call_2": "medium"} meaning after call_2 completes,
        taint should be updated with medium risk.
        """
        return {}

    def to_metadata(self) -> dict[str, Any]:
        """Serializable metadata for the frontend scenario selector."""
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "enforcer_type": self.enforcer_type,
        }
