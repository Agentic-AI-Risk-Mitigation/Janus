import sys
from pathlib import Path
from typing import Any

from janus.exceptions import PolicyViolation

# Add Policy-Discovery-Engine to sys.path so we can import from it without copying code yet
project_root = Path(__file__).resolve().parent.parent.parent
pde_path = project_root / "Policy-Discovery-Engine"
if str(pde_path) not in sys.path:
    sys.path.append(str(pde_path))

from policy_engine.enforcement import GraphInterceptor


class PDEEnforcer:
    """
    Adapter that allows Janus to use the SpiceDB-based Policy-Discovery-Engine 
    for policy enforcement, adhering loosely to the semantic interface of PolicyEnforcer.
    """

    def __init__(self, agent_role: str = "coding_agent", token: str = "somerandomkey", endpoint: str = "localhost:50051"):
        self.agent_role = agent_role
        self.interceptor = GraphInterceptor(token=token, endpoint=endpoint)
        self._policy = None

    def load(self, source: Any) -> None:
        """PDEEnforcer does not load static JSON policies in the same way, but we implement the interface to avoid errors."""
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
        """
        Query PDE Graph to see if tool execution is allowed given the current taint level.
        """
        allowed = self.interceptor.check_tool_access(tool_name, self.agent_role)
        if not allowed:
            # Raise standard PolicyViolation similar to Janus
            raise PolicyViolation(
                tool_name=tool_name,
                arguments=arguments,
                reason=f"Tool '{tool_name}' blocked by Policy-Discovery-Engine Graph for role '{self.agent_role}' with taint {self.interceptor.current_taint_level}.",
            )

    def update_taint(self, source_risk: int) -> None:
        """
        Increase the taint level of the current session based on reading a data source.
        """
        self.interceptor.update_taint(source_risk)
