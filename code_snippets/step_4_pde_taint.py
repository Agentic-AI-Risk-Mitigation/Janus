from janus import JanusAgent

from code_snippets.step_0_define_healthcare_tools import healthcare_tools


agent = JanusAgent(
    model="openai/gpt-4o",
    system_prompt=(
        "You are a clinical operations coordinator with access to patient records, "
        "lab reports, treatment planning tools, and a patient portal."
    ),
    tools=healthcare_tools,
    use_builtin_tools=False,
    policy_engine="pde",          # switch from static enforcer to PDE
    agent_role="clinical_agent",  # agent ID enrolled in clinician role
)

response = agent.run(
    "Review the latest lab results for patient P-2001 and update their treatment plan."
)
print(response)
