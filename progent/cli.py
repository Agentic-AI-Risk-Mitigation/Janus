import argparse
import json
import sys
from pathlib import Path

from progent.core import check_tool_call, update_security_policy
from progent.exceptions import ProgentBlockedError
from progent.logger import configure_logging, get_logger

# Try to load .env file if available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = get_logger()


def main():
    # Common arguments parser
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--policy", "-p", help="Path to policy JSON file (not needed for generate command)"
    )
    parent_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parent_parser.add_argument(
        "--log-level", "-l", help="Set logging level (DEBUG, INFO, WARNING, ERROR)", default=None
    )

    # Main parser
    parser = argparse.ArgumentParser(description="Progent Policy Debugger")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Check command
    check_parser = subparsers.add_parser(
        "check", help="Check if a tool call is allowed", parents=[parent_parser]
    )
    check_parser.add_argument("--tool", "-t", required=True, help="Name of the tool to check")
    check_parser.add_argument("--args", "-a", required=True, help="JSON string of arguments")

    # Analyze command
    subparsers.add_parser("analyze", help="Analyze policy for conflicts", parents=[parent_parser])

    # Generate command
    generate_parser = subparsers.add_parser(
        "generate", help="Generate policy from user query", parents=[parent_parser]
    )
    generate_parser.add_argument("--query", "-q", required=True, help="User query/task description")
    generate_parser.add_argument(
        "--manual-check", "-m", action="store_true", help="Manually approve generated policy"
    )
    generate_parser.add_argument(
        "--model", help="LLM model to use (overrides PROGENT_POLICY_MODEL env var)"
    )

    args = parser.parse_args()

    # Configure Logging
    level = "DEBUG" if args.verbose else args.log_level
    configure_logging(level=level)

    # Load Policy (not needed for generate command)
    if args.command != "generate":
        policy_path = Path(args.policy)
        if not policy_path.exists():
            logger.error(f"Policy file not found: {policy_path}")
            sys.exit(1)

        try:
            with open(policy_path, "r") as f:
                policy = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in policy file: {e}")
            sys.exit(1)

        # Initialize Progent
        update_security_policy(policy)
    else:
        # For generate command, we don't need a policy file
        policy_path = Path(args.policy) if args.policy else None

    if args.command == "analyze":
        try:
            from progent import analysis
        except ImportError:
            logger.error(
                "Analysis module requires 'progent[analysis]'. pip install progent[analysis]"
            )
            sys.exit(1)

        logger.info(f"Analyzing policy: {policy_path}")

        warnings = analysis.analyze_policies(policy)
        type_warnings = analysis.check_policy_type_errors(policy)
        all_warnings = warnings + type_warnings

        if not all_warnings:
            print("\n✅ Policy looks good! No conflicts or errors found.")
            sys.exit(0)
        else:
            print(f"\n⚠️ Found {len(all_warnings)} issues:")
            for w in all_warnings:
                print(f" - {w}")
            sys.exit(1)

    elif args.command == "generate":
        try:
            from progent.generation import generate_policies, set_policy_model
        except ImportError:
            logger.error(
                "Generation module requires 'progent[generation]'. pip install progent[generation]"
            )
            sys.exit(1)

        if args.model:
            set_policy_model(args.model)

        logger.info(f"Generating policy for query: {args.query}")

        try:
            generated = generate_policies(
                query=args.query,
                manual_confirm=args.manual_check,
            )

            if generated:
                print(f"\n✅ Policy generated for {len(generated)} tools:")
                for tool_name in generated.keys():
                    print(f"   - {tool_name}")
                print("\nPolicy has been applied to the current session.")
                print(f"To persist, save to file: {policy_path}")
            else:
                print("\n⚠️  No policy generated (user cancelled or no tools matched).")

            sys.exit(0)
        except Exception as e:
            logger.error(f"Policy generation failed: {e}")
            sys.exit(1)

    # Tool Execution Check
    if args.command == "check":
        if not args.tool or not args.args:
            logger.error("check command requires --tool and --args")
            sys.exit(1)

        try:
            tool_args = json.loads(args.args)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in arguments: {e}")
            sys.exit(1)

        logger.info(
            f"Checking access for tool '{args.tool}' with args: {json.dumps(tool_args, indent=2)}"
        )

        try:
            check_tool_call(args.tool, tool_args)
            print("\n✅ ALLOWED")
            sys.exit(0)

        except ProgentBlockedError as e:
            print("\n❌ BLOCKED")
            print(f"Reason: {e.reason}")
            if hasattr(e, "policy_rule") and e.policy_rule:
                print(f"Rule: {e.policy_rule}")
            if hasattr(e, "failed_condition") and e.failed_condition:
                print(f"Failed Condition: {e.failed_condition}")
            sys.exit(1)

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
