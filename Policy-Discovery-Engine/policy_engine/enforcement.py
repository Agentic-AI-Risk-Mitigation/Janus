from authzed.api.v1 import (
    Client,
    CheckPermissionRequest,
    CheckPermissionResponse,
    Consistency,
    ObjectReference,
    SubjectReference,
)
from grpcutil import insecure_bearer_token_credentials

from policy_engine.main import TOOL_TAINT_LIMIT


class GraphInterceptor:
    """
    Enforces tool access via SpiceDB (ReBAC) combined with Python-level taint checking.

    Architecture (matching main.py design):
    - Each tool has its OWN object type: tool_view_file, tool_run_tests, etc.
    - ACL edges are: tool_<name>:<name>#can_invoke @ role:<role>#member
    - Taint checking is done in Python (not SpiceDB caveats), comparing
      self.current_taint_level against TOOL_TAINT_LIMIT[tool_name].
    - SpiceDB is checked for ACL membership: does this role have invoke permission?
    """

    def __init__(self, token="somerandomkey", endpoint="localhost:50051", agent_id="coding_agent"):
        self.client = Client(endpoint, insecure_bearer_token_credentials(token))
        self.agent_id = agent_id  # Concrete agent identity in SpiceDB graph
        self.current_taint_level = 0  # Monotonically increases during a session

    def update_taint(self, source_risk: int):
        """Called when agent reads from a data source. Taint only ever goes up."""
        self.current_taint_level = max(self.current_taint_level, source_risk)
        print(f"[Runtime] Taint Level updated to: {self.current_taint_level}")

    def check_tool_access(self, tool_name: str, agent_role: str) -> bool:
        """
        Returns True if the tool is allowed for this role at the current taint level.

        Two-step check:
        1. Python taint gate: is current_taint_level <= TOOL_TAINT_LIMIT[tool_name]?
        2. SpiceDB ACL gate: does role have 'invoke' permission on tool_<tool_name>?
        """
        print(
            f"[Enforcement] Checking '{tool_name}' for role '{agent_role}' "
            f"with taint {self.current_taint_level}..."
        )

        # Step 1: Taint gate (Python-level, matching main.py's allow_tool logic)
        limit = TOOL_TAINT_LIMIT.get(tool_name, 50)
        if self.current_taint_level > limit:
            print(f"  🛑 TAINT DENY [{tool_name}] taint {self.current_taint_level} > limit {limit}")
            return False

        # Step 2: SpiceDB ACL gate
        # Object type is "tool_<tool_name>", object id is "<tool_name>"
        # Subject is agent:<agent_id> — SpiceDB traverses role#member automatically.
        # fully_consistent=True ensures we read the latest state (post-bootstrap).
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
            print(f"  ✅ ALLOW [{tool_name}] taint={self.current_taint_level}/{limit}")
        else:
            print(f"  🛑 ACL DENY [{tool_name}] no ACL edge in SpiceDB")
        return allowed
