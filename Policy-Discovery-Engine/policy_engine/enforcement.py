from authzed.api.v1 import Client, CheckPermissionRequest, CheckPermissionResponse, ObjectReference, SubjectReference

class GraphInterceptor:
    def __init__(self, token="somerandomkey", endpoint="localhost:50051"):
        self.client = Client(endpoint, token, insecure=True)
        self.current_taint_level = 0 # Dynamic state

    def update_taint(self, source_risk: int):
        """Called when agent reads from a data source"""
        # Monotonic increase: Taint only goes up, never down in a session
        self.current_taint_level = max(self.current_taint_level, source_risk)
        print(f"[Runtime] Taint Level Increased to: {self.current_taint_level}")

    def check_tool_access(self, tool_name: str, agent_role: str) -> bool:
        """
        Queries the Graph to see if tool execution is allowed
        given the CURRENT taint level.
        """
        print(f"[Enforcement] Checking '{tool_name}' for role '{agent_role}' with taint {self.current_taint_level}...")
        
        resp = self.client.CheckPermission(CheckPermissionRequest(
            resource=ObjectReference(object_type="tool", object_id=tool_name),
            permission="call",
            subject=SubjectReference(object=ObjectReference(object_type="role", object_id=agent_role)),
            # THIS IS THE KEY: Passing runtime context to the Graph
            context={
                "current_taint": self.current_taint_level,
                "max_allowed_taint": 50 # Example threshold (0-100)
            }
        ))
        
        allowed = (resp.permissionship == CheckPermissionResponse.PERMISSIONSHIP_HAS_PERMISSION)
        if not allowed:
            print(f"🛑 BLOCKED: Policy forbids {tool_name} due to high taint.")
        else:
            print(f"✅ ALLOWED: {tool_name} is safe.")
            
        return allowed
