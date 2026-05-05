import json

from janus import JanusAgent, save_policy

from code_snippets.step_0_define_healthcare_tools import healthcare_tools


agent = JanusAgent(
    model="openai/gpt-4o",
    system_prompt=(
        "You are a clinical operations coordinator with access to patient records, "
        "lab reports, treatment planning tools, and a patient portal."
    ),
    tools=healthcare_tools,
    use_builtin_tools=False,
    policy="generate",  # LLM generates least-privilege policy on first run()
    policy_model="openai/gpt-4o",
)

response = agent.run(
    "Review the latest lab results for patient P-2001 and update their treatment plan."
)
print(response)

# Inspect generated policy and save for review.
print(json.dumps(agent.enforcer._policy, indent=2))
save_policy(agent.enforcer._policy, "generated_healthcare_policy.json")
