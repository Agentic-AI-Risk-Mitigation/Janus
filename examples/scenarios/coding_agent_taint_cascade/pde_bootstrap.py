"""
PDE bootstrap for the coding agent taint cascade scenario.

Writes the shared PDE schema and scenario-specific ACL relationships.
"""

from __future__ import annotations

from janus.policy.pde.bootstrap import _rel, write_rels
from janus.policy.pde.config import SCHEMA


def bootstrap_spicedb(client) -> None:
    """Write the demo schema and ACL relationships to SpiceDB."""
    from authzed.api.v1 import WriteSchemaRequest

    client.WriteSchema(WriteSchemaRequest(schema=SCHEMA))

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

    write_rels(client, rels)
    print("[Coding Agent Taint Cascade Bootstrap] Schema and relationships written to SpiceDB.")
