"""
Entry point for running Progent evaluations.
"""

from pathlib import Path

# Add project root to path
from implementations.evals.harness import Evaluator
from implementations.evals.scenarios import SCENARIOS


def main():
    print("Starting Progent Evaluations...", flush=True)

    # Initialize Evaluator
    policy_path = str(Path(__file__).parent / "eval_policies.json")
    print(f"Loading policy from: {policy_path}", flush=True)

    evaluator = Evaluator(policy_path=policy_path)

    # Run all scenarios
    for scenario in SCENARIOS:
        print(f"Running scenario: {scenario.name}", flush=True)
        evaluator.run_scenario(scenario)

    # Generate Report
    evaluator.generate_report()


if __name__ == "__main__":
    main()
