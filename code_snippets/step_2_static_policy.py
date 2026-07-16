from code_snippets.step_0_define_healthcare_tools import healthcare_tools
from janus import JanusAgent

healthcare_policy = {
    "query_patient_record": [
        {"priority": 1, "effect": 0, "conditions": {}, "fallback": 0}
    ],
    "fetch_clinical_report": [
        {"priority": 1, "effect": 0, "conditions": {}, "fallback": 0}
    ],
    "update_treatment_plan": [
        {"priority": 1, "effect": 0, "conditions": {}, "fallback": 0}
    ],
    "publish_to_portal": [
        {"priority": 1, "effect": 1, "conditions": {}, "fallback": 1}
    ],
    "fetch_url": [
        {"priority": 1, "effect": 1, "conditions": {}, "fallback": 1}
    ],
}

agent = JanusAgent(
    model="openai/gpt-4o",
    system_prompt=(
        "You are a clinical operations coordinator with access to patient records, "
        "lab reports, treatment planning tools, and a patient portal."
    ),
    tools=healthcare_tools,
    use_builtin_tools=False,
    policy=healthcare_policy,
)

response = agent.run(
    "Review the latest lab results for patient P-2001 and update their treatment plan."
)
print(response)
