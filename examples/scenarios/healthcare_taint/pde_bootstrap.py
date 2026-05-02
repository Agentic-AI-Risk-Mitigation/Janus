"""
PDE bootstrap for the Healthcare Taint scenario.

Writes the shared PDE schema and healthcare-specific ACL relationships.
"""

from __future__ import annotations

from janus.policy.pde.bootstrap import _rel, write_rels
from janus.policy.pde.config import SCHEMA


def bootstrap_spicedb(client) -> None:
    """Write the healthcare schema and relationships to SpiceDB."""
    from authzed.api.v1 import WriteSchemaRequest

    client.WriteSchema(WriteSchemaRequest(schema=SCHEMA))

    write_rels(client, [
        # Agent enrollment: clinical_agent is a member of clinician and coordinator roles
        _rel("role", "clinician", "member", "agent", "clinical_agent"),
        _rel("role", "coordinator", "member", "agent", "clinical_agent"),

        # Tool access for clinician role
        _rel("tool_query_patient_record", "query_patient_record", "can_invoke", "role", "clinician", "member"),
        _rel("tool_fetch_clinical_report", "fetch_clinical_report", "can_invoke", "role", "clinician", "member"),
        _rel("tool_update_treatment_plan", "update_treatment_plan", "can_invoke", "role", "clinician", "member"),
        _rel("tool_publish_to_portal", "publish_to_portal", "can_invoke", "role", "clinician", "member"),
        _rel("tool_send_notification", "send_notification", "can_invoke", "role", "clinician", "member"),

        # fetch_url via existing tool definition
        _rel("tool_fetch_url", "fetch_url", "can_invoke", "role", "clinician", "member"),
    ])
    print("[Bootstrap] Healthcare PDE setup complete.")
