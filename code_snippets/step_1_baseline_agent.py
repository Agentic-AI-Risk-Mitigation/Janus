from code_snippets.step_0_define_healthcare_tools import healthcare_tools
from janus import JanusAgent

agent = JanusAgent(
    model="openai/gpt-4o",
    system_prompt=(
        "You are a clinical operations coordinator with access to patient records, "
        "lab reports, treatment planning tools, and a patient portal."
    ),
    tools=healthcare_tools,
    use_builtin_tools=False,
    policy=None,  # no policy: all tools can execute
)

response = agent.run(
    "Review the latest lab results for patient P-2001 and update their treatment plan."
)
print(response)
