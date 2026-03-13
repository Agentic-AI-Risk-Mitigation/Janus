"""
PDE bootstrap for Demo 5: The Taint Cascade.

Extends the existing PDE schema with demo-specific tools (read_file, fetch_url,
write_file) and configures taint limits and ACL relationships for the demo.
"""

from __future__ import annotations

DEMO_SCHEMA = """
// --- Identity ---
definition user {}

definition agent {
  relation owner:    user
  relation delegate: user
}

definition role {
  relation member: agent | user
}

// --- Tier 0: read-only ---
definition tool_read_file {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_list_directory {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_view_file {
  relation can_invoke: role#member
  permission invoke = can_invoke
}

// --- Tier 1: disk writes ---
definition tool_edit_file {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_write_file {
  relation can_invoke: role#member
  permission invoke = can_invoke
}

// --- Tier 2: network ---
definition tool_fetch_url {
  relation can_invoke: role#member
  permission invoke = can_invoke
}

// --- Tier 3: git ---
definition tool_git_commit {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_git_push {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
"""

DEMO_TOOL_TAINT_LIMIT = {
    "read_file": 90,
    "list_directory": 90,
    "view_file": 90,
    "edit_file": 70,
    "write_file": 70,
    "fetch_url": 10,
    "git_commit": 40,
    "git_push": 20,
}

DEMO_RISK_TO_TAINT = {
    "low": 10,
    "medium": 40,
    "high": 70,
    "critical": 90,
}


def bootstrap_spicedb(client) -> None:
    """Write the demo schema and ACL relationships to SpiceDB."""
    from authzed.api.v1 import (
        ObjectReference,
        Relationship,
        RelationshipUpdate,
        SubjectReference,
        WriteRelationshipsRequest,
        WriteSchemaRequest,
    )

    client.WriteSchema(WriteSchemaRequest(schema=DEMO_SCHEMA))

    def _rel(res_type, res_id, relation, sub_type, sub_id, sub_rel=""):
        return Relationship(
            resource=ObjectReference(object_type=res_type, object_id=res_id),
            relation=relation,
            subject=SubjectReference(
                object=ObjectReference(object_type=sub_type, object_id=sub_id),
                optional_relation=sub_rel,
            ),
        )

    rels = [
        # Agent membership in roles
        _rel("role", "readonly", "member", "agent", "coding_agent"),
        _rel("role", "developer", "member", "agent", "coding_agent"),

        # Tier 0: read-only tools -> readonly role
        _rel("tool_read_file", "read_file", "can_invoke", "role", "readonly", "member"),
        _rel("tool_list_directory", "list_directory", "can_invoke", "role", "readonly", "member"),
        _rel("tool_view_file", "view_file", "can_invoke", "role", "readonly", "member"),

        # Tier 1: write tools -> developer role
        _rel("tool_edit_file", "edit_file", "can_invoke", "role", "developer", "member"),
        _rel("tool_write_file", "write_file", "can_invoke", "role", "developer", "member"),

        # Tier 2: network -> developer role
        _rel("tool_fetch_url", "fetch_url", "can_invoke", "role", "developer", "member"),

        # Tier 3: git -> developer role
        _rel("tool_git_commit", "git_commit", "can_invoke", "role", "developer", "member"),
        _rel("tool_git_push", "git_push", "can_invoke", "role", "developer", "member"),
    ]

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
    print("[Demo 5 Bootstrap] Schema and relationships written to SpiceDB.")
