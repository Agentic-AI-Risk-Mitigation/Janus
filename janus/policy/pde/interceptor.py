"""
SpiceDB-based enforcement interceptor.

Combines Python-level taint gating with SpiceDB ReBAC permission checks
to decide whether a tool call is allowed at the current taint level.
"""

import grpc
from authzed.api.v1 import (
    CheckPermissionRequest,
    CheckPermissionResponse,
    Client,
    Consistency,
    ObjectReference,
    SubjectReference,
)
from grpcutil import insecure_bearer_token_credentials

from janus.policy.pde.config import TOOL_TAINT_LIMIT


class _SyncClient(Client):
    """Forces a synchronous gRPC channel regardless of asyncio event loop presence."""

    def create_channel(self, target, credentials, options=None, compression=None):
        return grpc.secure_channel(target, credentials, options, compression)


class GraphInterceptor:
    """
    Enforces tool access via SpiceDB (ReBAC) combined with Python-level taint checking.

    Two-step check per tool call:
    1. Python taint gate: is current_taint_level <= TOOL_TAINT_LIMIT[tool_name]?
    2. SpiceDB ACL gate: does the agent's role have 'invoke' permission on tool_<tool_name>?
    """

    def __init__(
        self,
        token: str = "somerandomkey",
        endpoint: str = "localhost:50051",
        agent_id: str = "coding_agent",
    ):
        self.client = _SyncClient(endpoint, insecure_bearer_token_credentials(token))
        self.agent_id = agent_id
        self.current_taint_level = 0

    def update_taint(self, source_risk: int) -> None:
        """Called when agent reads from a data source. Taint only ever goes up."""
        self.current_taint_level = max(self.current_taint_level, source_risk)
        print(f"[Runtime] Taint Level updated to: {self.current_taint_level}")

    def check_tool_access(self, tool_name: str, agent_role: str) -> bool:
        """
        Returns True if the tool is allowed for this role at the current taint level.
        """
        print(
            f"[Enforcement] Checking '{tool_name}' for role '{agent_role}' "
            f"with taint {self.current_taint_level}..."
        )

        limit = TOOL_TAINT_LIMIT.get(tool_name, 50)
        if self.current_taint_level > limit:
            print(f"  TAINT DENY [{tool_name}] taint {self.current_taint_level} > limit {limit}")
            return False

        resp = self.client.CheckPermission(
            CheckPermissionRequest(
                resource=ObjectReference(
                    object_type=f"tool_{tool_name}",
                    object_id=tool_name,
                ),
                permission="invoke",
                subject=SubjectReference(
                    object=ObjectReference(object_type="agent", object_id=self.agent_id),
                ),
                consistency=Consistency(fully_consistent=True),
            )
        )

        allowed = resp.permissionship == CheckPermissionResponse.PERMISSIONSHIP_HAS_PERMISSION
        if allowed:
            print(f"  ALLOW [{tool_name}] taint={self.current_taint_level}/{limit}")
        else:
            print(f"  ACL DENY [{tool_name}] no ACL edge in SpiceDB")
        return allowed
