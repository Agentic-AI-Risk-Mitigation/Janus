#!/usr/bin/env python3
"""
Progent Coding Agent - CLI Entry Point

Usage:
    python run_agent.py                           # Use defaults (sandbox workspace, langchain)
    python run_agent.py --workspace ./my_project  # Specify workspace
    python run_agent.py --framework adk           # Use Google ADK
    python run_agent.py --model anthropic/claude-3.5-sonnet  # Override model
"""

import argparse
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Directory setup
EXAMPLE_DIR = Path(__file__).parent.resolve()  # coding_agent/
IMPL_DIR = EXAMPLE_DIR.parent.parent.resolve()  # implementations/
ROOT_DIR = IMPL_DIR.parent.resolve()  # progent/

# Add paths for imports
sys.path.insert(0, str(IMPL_DIR))  # For core/, frameworks/, tools/
sys.path.insert(0, str(ROOT_DIR))  # For progent/

# Load environment variables from .env file
load_dotenv(EXAMPLE_DIR / ".env")


def load_config(config_path: str | Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Progent Coding Agent - A secure coding assistant with policy enforcement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--workspace",
        "-w",
        type=str,
        default=None,
        help="Path to the workspace directory (default: ./sandbox)",
    )

    parser.add_argument(
        "--framework",
        "-f",
        type=str,
        choices=["langchain", "adk", "raw_sdk", "claude_sdk"],
        default=None,
        help="Agent framework to use (default: from config)",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="LLM model to use (default: from config)",
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=str(EXAMPLE_DIR / "config.yaml"),
        help="Path to config file (default: ./config.yaml)",
    )

    parser.add_argument(
        "--policies",
        "-p",
        type=str,
        default=str(EXAMPLE_DIR / "policies.json"),
        help="Path to policies file (default: ./policies.json)",
    )

    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="API base URL (default: from config)",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for log files (default: from config)",
    )

    args = parser.parse_args()

    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid config file: {e}")
        sys.exit(1)

    # Apply command-line overrides
    if args.framework:
        config.setdefault("agent", {})["framework"] = args.framework

    if args.model:
        config.setdefault("llm", {})["model"] = args.model

    if args.api_base:
        config.setdefault("llm", {})["api_base"] = args.api_base

    # Determine workspace
    if args.workspace:
        workspace = Path(args.workspace).resolve()
    else:
        default_workspace = config.get("workspace", {}).get("default_path", "./sandbox")
        workspace = (EXAMPLE_DIR / default_workspace).resolve()

    # Ensure workspace exists
    workspace.mkdir(parents=True, exist_ok=True)

    # Initialize logging
    log_dir = args.log_dir or config.get("logging", {}).get("log_dir", "./logs")
    log_dir = (EXAMPLE_DIR / log_dir).resolve()
    log_level = config.get("logging", {}).get("level", "INFO")

    from implementations.core.logging_utils import init_logger

    init_logger(log_dir=str(log_dir), level=log_level)

    # Get framework
    framework = config.get("agent", {}).get("framework", "langchain")

    # Create the agent
    print(f"Initializing {framework} agent...")
    print(f"Workspace: {workspace}")
    print(f"Policies: {args.policies}")

    try:
        if framework == "langchain":
            from implementations.frameworks.langchain_agent import LangChainAgent

            agent = LangChainAgent(
                config=config,
                workspace=workspace,
                policies_path=args.policies,
            )
        elif framework == "adk":
            from implementations.frameworks.adk_agent import ADKAgent

            agent = ADKAgent(
                config=config,
                workspace=workspace,
                policies_path=args.policies,
            )
        elif framework == "raw_sdk":
            from implementations.frameworks.raw_sdk_agent import RawSDKAgent

            agent = RawSDKAgent(
                config=config,
                workspace=workspace,
                policies_path=args.policies,
            )
        elif framework == "claude_sdk":
            from implementations.frameworks.claude_sdk_agent import ClaudeSDKAgent

            agent = ClaudeSDKAgent(
                config=config,
                workspace=workspace,
                policies_path=args.policies,
            )
        else:
            print(f"Error: Unknown framework: {framework}")
            sys.exit(1)

    except ImportError as e:
        print(f"Error: Could not import {framework} agent. Is the framework installed?")
        print(f"Details: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error initializing agent: {e}")
        sys.exit(1)

    # Start the REPL
    agent.start_repl()


if __name__ == "__main__":
    main()
