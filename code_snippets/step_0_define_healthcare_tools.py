from janus import ToolDef, ToolParam


def query_patient_record(patient_id: str) -> str:
    """Retrieve a patient record from the internal EHR."""
    return '{"name": "Jane Doe", "condition": "Type 2 Diabetes", "risk_level": "moderate"}'


def fetch_clinical_report(report_id: str) -> str:
    """Fetch a clinical report, potentially from an external lab."""
    return (
        "Source: third-party-lab.example.com\n"
        "Patient P-2001: HbA1c 8.2%, fasting glucose 185 mg/dL.\n"
        "Recommendation: Adjust medication dosage."
    )


def update_treatment_plan(patient_id: str, plan: str) -> str:
    """Update a patient's treatment plan (internal write)."""
    return f"Treatment plan updated for {patient_id}: {plan[:80]}"


def publish_to_portal(patient_id: str, content: str) -> str:
    """Publish to the patient-facing portal (high-impact action)."""
    return f"Published to portal for {patient_id}: {content[:80]}"


def fetch_url(url: str) -> str:
    """Generic HTTP GET, used for external integrations."""
    return f"HTTP 200 OK - Fetched {url}"


healthcare_tools = [
    ToolDef(
        name="query_patient_record",
        description="Retrieve a patient record by patient ID.",
        params=[ToolParam("patient_id", "string", "Patient ID (e.g. P-2001).")],
        handler=query_patient_record,
    ),
    ToolDef(
        name="fetch_clinical_report",
        description="Fetch a clinical report by ID. May come from external labs (higher risk).",
        params=[ToolParam("report_id", "string", "Report ID (e.g. CR-401).")],
        handler=fetch_clinical_report,
    ),
    ToolDef(
        name="update_treatment_plan",
        description="Update a patient's treatment plan in the internal EHR.",
        params=[
            ToolParam("patient_id", "string", "Patient ID."),
            ToolParam("plan", "string", "Updated treatment plan text."),
        ],
        handler=update_treatment_plan,
    ),
    ToolDef(
        name="publish_to_portal",
        description="Publish information to the patient-facing portal. High-impact action.",
        params=[
            ToolParam("patient_id", "string", "Patient ID."),
            ToolParam("content", "string", "Content to publish."),
        ],
        handler=publish_to_portal,
    ),
    ToolDef(
        name="fetch_url",
        description="Fetch content from a URL via HTTP GET.",
        params=[ToolParam("url", "string", "The URL to fetch.")],
        handler=fetch_url,
    ),
]
