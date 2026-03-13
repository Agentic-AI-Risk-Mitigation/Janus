"""
SpiceDB bootstrap utilities.

Writes the default schema and role/tool relationships into SpiceDB
so the enforcement interceptor can check permissions.
"""

from authzed.api.v1 import (
    Client,
    ObjectReference,
    Relationship,
    RelationshipUpdate,
    SubjectReference,
    WriteRelationshipsRequest,
    WriteSchemaRequest,
)
from grpcutil import insecure_bearer_token_credentials

from janus.policy.pde.config import RISK_TO_TAINT, SCHEMA, TOOL_TAINT_LIMIT


def make_client(
    endpoint: str = "localhost:50051",
    token: str = "somerandomkey",
) -> Client:
    return Client(endpoint, insecure_bearer_token_credentials(token))


def _rel(
    res_type: str,
    res_id: str,
    relation: str,
    sub_type: str,
    sub_id: str,
    sub_rel: str = "",
) -> Relationship:
    return Relationship(
        resource=ObjectReference(object_type=res_type, object_id=res_id),
        relation=relation,
        subject=SubjectReference(
            object=ObjectReference(object_type=sub_type, object_id=sub_id),
            optional_relation=sub_rel,
        ),
    )


def write_rels(client: Client, rels: list[Relationship]) -> None:
    client.WriteRelationships(
        WriteRelationshipsRequest(
            updates=[
                RelationshipUpdate(
                    operation=RelationshipUpdate.Operation.OPERATION_TOUCH,
                    relationship=r,
                )
                for r in rels
            ]
        )
    )


def bootstrap(client: Client) -> None:
    """Write the default schema and ACL relationships to SpiceDB."""
    client.WriteSchema(WriteSchemaRequest(schema=SCHEMA))

    write_rels(client, [
        _rel("role", "readonly",  "member", "agent", "coding_agent"),
        _rel("role", "developer", "member", "agent", "coding_agent"),
        _rel("role", "executor",  "member", "agent", "coding_agent"),

        _rel("tool_view_file",    "view_file",    "can_invoke", "role", "readonly",  "member"),
        _rel("tool_grep",         "grep",         "can_invoke", "role", "readonly",  "member"),
        _rel("tool_code_search",  "code_search",  "can_invoke", "role", "readonly",  "member"),
        _rel("tool_git_diff",     "git_diff",     "can_invoke", "role", "readonly",  "member"),
        _rel("tool_git_log",      "git_log",      "can_invoke", "role", "readonly",  "member"),

        _rel("tool_edit_file",    "edit_file",    "can_invoke", "role", "developer", "member"),
        _rel("tool_create_file",  "create_file",  "can_invoke", "role", "developer", "member"),
        _rel("tool_delete_file",  "delete_file",  "can_invoke", "role", "developer", "member"),

        _rel("tool_run_tests",    "run_tests",    "can_invoke", "role", "executor",  "member"),
        _rel("tool_run_script",   "run_script",   "can_invoke", "role", "executor",  "member"),
        _rel("tool_pip_install",  "pip_install",  "can_invoke", "role", "executor",  "member"),

        _rel("tool_git_commit",   "git_commit",   "can_invoke", "role", "developer", "member"),
        _rel("tool_git_push",     "git_push",     "can_invoke", "role", "developer", "member"),
        _rel("tool_git_clone",    "git_clone",    "can_invoke", "role", "developer", "member"),
    ])
    print("[Bootstrap] Done.")


class Session:
    """Tracks taint accumulation during an agent session."""

    def __init__(self) -> None:
        self.taint: int = 0

    def read_source(self, risk: str) -> None:
        self.taint = max(self.taint, RISK_TO_TAINT.get(risk, 50))
        print(f"  [Taint] -> {self.taint}  (risk={risk})")


def allow_tool(client: Client, session: Session, agent_id: str, tool_name: str) -> bool:
    """Check if a tool is allowed given the current session taint and SpiceDB ACLs."""
    from authzed.api.v1 import (
        CheckPermissionRequest,
        CheckPermissionResponse,
    )

    limit = TOOL_TAINT_LIMIT.get(tool_name, 50)

    if session.taint > limit:
        print(f"  DENY  [{tool_name}]  taint {session.taint} > limit {limit}")
        return False

    resp = client.CheckPermission(
        CheckPermissionRequest(
            resource=ObjectReference(object_type=f"tool_{tool_name}", object_id=tool_name),
            permission="invoke",
            subject=SubjectReference(
                object=ObjectReference(object_type="agent", object_id=agent_id)
            ),
        )
    )

    ok = resp.permissionship == CheckPermissionResponse.PERMISSIONSHIP_HAS_PERMISSION
    tag = "ALLOW" if ok else "DENY (no ACL)"
    print(f"  {tag}  [{tool_name}]  taint={session.taint}/{limit}")
    return ok
