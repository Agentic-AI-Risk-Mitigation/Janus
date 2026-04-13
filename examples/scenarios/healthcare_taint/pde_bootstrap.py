"""
PDE bootstrap for the Healthcare Taint scenario.

Sets up SpiceDB schema and relationships for healthcare tools
with appropriate taint limits.
"""

from __future__ import annotations

from janus.policy.pde.bootstrap import _rel, write_rels
from janus.policy.pde.config import SCHEMA

# Tool taint limits for healthcare scenario
DEMO_TOOL_TAINT_LIMIT: dict[str, int] = {
    "query_patient_record": 90,    # read internal records: almost always OK
    "fetch_clinical_report": 90,   # reading reports is allowed
    "update_treatment_plan": 70,   # internal write: moderate threshold
    "publish_to_portal": 20,       # patient-facing: strict threshold
    "send_notification": 30,       # notifications: moderate-strict
    "fetch_url": 10,               # external network: very strict
}

# Additional SpiceDB schema for healthcare tools
HEALTHCARE_SCHEMA = SCHEMA + """
definition tool_query_patient_record {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_fetch_clinical_report {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_update_treatment_plan {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_publish_to_portal {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_send_notification {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
definition tool_fetch_url {
  relation can_invoke: role#member
  permission invoke = can_invoke
}
"""


def bootstrap_spicedb(client) -> None:
    """Write the healthcare schema and relationships to SpiceDB."""
    from authzed.api.v1 import WriteSchemaRequest

    client.WriteSchema(WriteSchemaRequest(schema=HEALTHCARE_SCHEMA))

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
