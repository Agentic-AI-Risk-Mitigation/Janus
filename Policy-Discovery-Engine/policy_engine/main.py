SCHEMA = """
// ─── Identity ─────────────────────────────────────────────────────────────
definition user {}

definition agent {
  relation owner:    user
  relation delegate: user
}

definition role {
  relation member: agent | user
}

// ─── Code & Data Resources ────────────────────────────────────────────────
definition repository {
  relation reader: user | agent | role#member
  relation writer: user | agent | role#member
  relation admin:  user | agent | role#member

  permission read   = reader + writer + admin
  permission write  = writer + admin
  permission manage = admin
}

definition codefile {
  relation repository: repository
  relation viewer: user | agent | role#member
  relation editor: user | agent | role#member

  permission read  = viewer + editor + repository->read
  permission write = editor + repository->write
}

definition environment {
  relation operator: user | agent | role#member
  relation deployer: user | agent | role#member

  permission use    = operator + deployer
  permission deploy = deployer
}

definition secret {
  relation reader: user | agent | role#member
  relation writer: user | agent | role#member

  permission read  = reader + writer
  permission write = writer
}

// ─── Tier 0: read only ────────────────────────────────────────────────────
definition tool_view_file {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_grep {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_code_search {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_git_diff {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_git_log {
  relation can_invoke: role#member
  permission invoke = can_invoke
}

// ─── Tier 1: disk writes ──────────────────────────────────────────────────
definition tool_edit_file {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_create_file {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_delete_file {
  relation can_invoke: role#member
  permission invoke = can_invoke
}

// ─── Tier 2: local execution ──────────────────────────────────────────────
definition tool_run_tests {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_run_script {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_pip_install {
  relation can_invoke: role#member
  permission invoke = can_invoke
}

// ─── Tier 3: remote git ───────────────────────────────────────────────────
definition tool_git_commit {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_git_push {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_git_clone {
  relation can_invoke: role#member
  permission invoke = can_invoke
}

// ─── Tier 4: dangerous — no ACL edges ever assigned ───────────────────────
definition tool_bash_terminal {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_http_request {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_read_secret {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_write_secret {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_deploy {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
"""


TOOL_TAINT_LIMIT = {
    "view_file":     90,
    "grep":          90,
    "code_search":   90,
    "git_diff":      90,
    "git_log":       90,
    "edit_file":     70,
    "create_file":   70,
    "delete_file":   50,
    "run_tests":     40,
    "run_script":    30,
    "pip_install":   30,
    "git_commit":    40,
    "git_push":      20,
    "git_clone":     60,
    "bash_terminal": 10,
    "http_request":  10,
    "read_secret":   10,
    "write_secret":   5,
    "deploy":         5,
}

RISK_TO_TAINT = {
    "low":      10,
    "medium":   40,
    "high":     70,
    "critical": 90,
}

from authzed.api.v1 import (
    Client,
    WriteSchemaRequest,
    WriteRelationshipsRequest,
    RelationshipUpdate,
    CheckPermissionRequest,
    CheckPermissionResponse,
    Relationship,
    ObjectReference,
    SubjectReference,
)
from grpcutil import insecure_bearer_token_credentials
from typing import Optional


def make_client() -> Client:
    return Client("localhost:50051", insecure_bearer_token_credentials("somerandomkey"))


def _rel(res_type, res_id, relation, sub_type, sub_id, sub_rel="") -> Relationship:
    return Relationship(
        resource=ObjectReference(object_type=res_type, object_id=res_id),
        relation=relation,
        subject=SubjectReference(
            object=ObjectReference(object_type=sub_type, object_id=sub_id),
            optional_relation=sub_rel,
        ),
    )


def write_rels(client, rels):
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
    def __init__(self):
        self.taint = 0

    def read_source(self, risk: str) -> None:
        self.taint = max(self.taint, RISK_TO_TAINT.get(risk, 50))
        print(f"  [Taint] → {self.taint}  (risk={risk})")


def allow_tool(client: Client, session: Session, agent_id: str, tool_name: str) -> bool:
    limit = TOOL_TAINT_LIMIT.get(tool_name, 50)

    if session.taint > limit:
        print(f"  ✗ DENY  [{tool_name}]  taint {session.taint} > limit {limit}")
        return False

    resp = client.CheckPermission(
        CheckPermissionRequest(
            resource   = ObjectReference(object_type=f"tool_{tool_name}", object_id=tool_name),
            permission = "invoke",
            subject    = SubjectReference(
                object=ObjectReference(object_type="agent", object_id=agent_id)
            ),
        )
    )

    ok = resp.permissionship == CheckPermissionResponse.PERMISSIONSHIP_HAS_PERMISSION
    print(f"  {'✓ ALLOW' if ok else '✗ DENY (no ACL)'}  [{tool_name}]  taint={session.taint}/{limit}")
    return ok


if __name__ == "__main__":
    client = make_client()
    bootstrap(client)

    print("\n── Safe read session (low risk) ────────────────────────────────")
    s = Session()
    s.read_source("low")
    allow_tool(client, s, "coding_agent", "view_file")
    allow_tool(client, s, "coding_agent", "grep")
    allow_tool(client, s, "coding_agent", "git_diff")

    print("\n── Developer session (low risk) ────────────────────────────────")
    s2 = Session()
    s2.read_source("low")
    allow_tool(client, s2, "coding_agent", "edit_file")
    allow_tool(client, s2, "coding_agent", "run_tests")
    allow_tool(client, s2, "coding_agent", "git_commit")
    allow_tool(client, s2, "coding_agent", "git_push")

    print("\n── IPI: agent read scraped web before acting ───────────────────")
    s3 = Session()
    s3.read_source("critical")
    allow_tool(client, s3, "coding_agent", "view_file")
    allow_tool(client, s3, "coding_agent", "edit_file")
    allow_tool(client, s3, "coding_agent", "git_push")
    allow_tool(client, s3, "coding_agent", "run_tests")

    print("\n── Dangerous tools (Tier 4) — always DENY ──────────────────────")
    s4 = Session()
    s4.read_source("low")
    allow_tool(client, s4, "coding_agent", "bash_terminal")
    allow_tool(client, s4, "coding_agent", "http_request")
    allow_tool(client, s4, "coding_agent", "read_secret")
    allow_tool(client, s4, "coding_agent", "write_secret")
    allow_tool(client, s4, "coding_agent", "deploy")