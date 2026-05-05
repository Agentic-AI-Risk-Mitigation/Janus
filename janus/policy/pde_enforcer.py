"""
PDE enforcer adapter.

Wraps the SpiceDB-based GraphInterceptor behind the same interface
as PolicyEnforcer so the rest of Janus can swap engines transparently.
"""

from typing import Any

from janus.exceptions import PolicyViolation
from janus.policy.pde.interceptor import GraphInterceptor


class PDEEnforcer:
    """
    Adapter that allows Janus to use the SpiceDB-based Policy-Discovery-Engine
    for policy enforcement, adhering loosely to the semantic interface of PolicyEnforcer.
    """

    def __init__(
        self,
        agent_id: str = "coding_agent",
        token: str = "somerandomkey",
        endpoint: str = "localhost:50051",
    ):
        self.agent_id = agent_id
        self.interceptor = GraphInterceptor(token=token, endpoint=endpoint)
        self._policy = None

    def load(self, source: Any) -> None:
        pass

    def update(self, additional: Any) -> None:
        pass

    def allow_tools(self, tools: list[str], *, allow_no_arg_tools: bool = False) -> None:
        pass

    def block_tools(self, tools: list[str]) -> None:
        pass

    def reset(self, *, keep_manual: bool = True) -> None:
        pass

    @property
    def policy(self) -> dict | None:
        return self._policy

    def has_policy(self) -> bool:
        return True

    def enforce(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Query PDE Graph to see if tool execution is allowed given the current taint level."""
        allowed = self.interceptor.check_tool_access(tool_name, self.agent_id)
        if not allowed:
            raise PolicyViolation(
                tool_name=tool_name,
                arguments=arguments,
                reason=(
                    f"Tool '{tool_name}' blocked by PDE for agent '{self.agent_id}' "
                    f"with taint {self.interceptor.current_taint_level}."
                ),
            )

    def update_taint(self, source_risk: int) -> None:
        """Increase the taint level of the current session based on reading a data source."""
        self.interceptor.update_taint(source_risk)
