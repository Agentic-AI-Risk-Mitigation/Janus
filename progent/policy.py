"""
Progent policy loading utilities.

Handles loading policies from various sources (JSON files, dicts, etc.).
"""

import json
from pathlib import Path

from progent.core import update_security_policy
from progent.exceptions import PolicyLoadError


def load_policies(source: str | Path | dict) -> dict:
    """
    Load security policies from a file or dict and apply them.

    Supports two formats:

    1. Progent format (recommended):
       {
           "tool_name": [
               {
                   "priority": 1,
                   "effect": 0,  # 0=allow, 1=deny
                   "conditions": {...},  # JSON Schema
                   "fallback": 0  # 0=error, 1=terminate, 2=ask user
               }
           ]
       }

    2. Simple format:
       {
           "tool_name": {
               "arg_name": {"type": "string", "pattern": "..."},
               ...
           }
       }

    Args:
        source: Path to JSON file, or dict of policies

    Returns:
        The loaded policies in internal format

    Raises:
        PolicyLoadError: If policies cannot be loaded or parsed
    """
    # Load raw policies
    if isinstance(source, dict):
        raw = source
    elif isinstance(source, (str, Path)):
        raw = _load_from_file(source)
    else:
        raise PolicyLoadError(f"Unsupported policy source type: {type(source)}")

    # Convert to internal format
    converted = _convert_policies(raw)

    # Apply to core
    update_security_policy(converted)

    return converted


def _load_from_file(path: str | Path) -> dict:
    """Load policies from a JSON file."""
    path = Path(path)

    if not path.exists():
        raise PolicyLoadError(f"Policy file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise PolicyLoadError(f"Invalid JSON in policy file: {e}")
    except Exception as e:
        raise PolicyLoadError(f"Error reading policy file: {e}")


def _convert_policies(raw: dict) -> dict:
    """
    Convert policies to internal format.

    Internal format: {tool_name: [(priority, effect, conditions, fallback), ...]}
    """
    converted = {}

    for tool_name, rules in raw.items():
        converted[tool_name] = []

        if isinstance(rules, list):
            # Already in list format (Progent format)
            for rule in rules:
                if isinstance(rule, dict):
                    # Dict with keys: priority, effect, conditions, fallback
                    converted[tool_name].append(
                        (
                            rule.get("priority", 1),
                            rule.get("effect", 0),
                            rule.get("conditions", {}),
                            rule.get("fallback", 0),
                        )
                    )
                elif isinstance(rule, (list, tuple)):
                    # Already a tuple/list
                    converted[tool_name].append(tuple(rule))

        elif isinstance(rules, dict):
            # Simple format: conditions only
            # Check if it has policy structure or is just conditions
            if _is_policy_structure(rules):
                converted[tool_name].append(
                    (
                        rules.get("priority", 1),
                        rules.get("effect", 0),
                        rules.get("conditions", {}),
                        rules.get("fallback", 0),
                    )
                )
            else:
                # Treat as conditions dict
                converted[tool_name].append((1, 0, rules, 0))

    return converted


def _is_policy_structure(d: dict) -> bool:
    """Check if a dict looks like a policy structure vs. conditions."""
    policy_keys = {"priority", "effect", "conditions", "fallback"}
    return bool(policy_keys & set(d.keys()))


def save_policies(policies: dict, path: str | Path) -> None:
    """
    Save policies to a JSON file.

    Args:
        policies: Policies in internal format
        path: Path to save to
    """
    path = Path(path)

    # Convert to serializable format
    output = {}
    for tool_name, rules in policies.items():
        output[tool_name] = []
        for rule in rules:
            if isinstance(rule, (list, tuple)):
                output[tool_name].append(
                    {
                        "priority": rule[0],
                        "effect": rule[1],
                        "conditions": rule[2],
                        "fallback": rule[3] if len(rule) > 3 else 0,
                    }
                )
            else:
                output[tool_name].append(rule)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


def validate_policies(policies: dict) -> list[str]:
    """
    Validate policies for common issues.

    Returns list of warning messages.
    """
    from progent.validation import validate_schema

    warnings = []

    for tool_name, rules in policies.items():
        for i, rule in enumerate(rules):
            if isinstance(rule, (list, tuple)):
                conditions = rule[2] if len(rule) > 2 else {}
            elif isinstance(rule, dict):
                conditions = rule.get("conditions", {})
            else:
                warnings.append(f"Invalid rule format for {tool_name}[{i}]")
                continue

            # Validate each condition schema
            for arg_name, schema in conditions.items():
                if isinstance(schema, dict):
                    schema_warnings = validate_schema(schema)
                    for w in schema_warnings:
                        warnings.append(f"{tool_name}.{arg_name}: {w}")

    return warnings
