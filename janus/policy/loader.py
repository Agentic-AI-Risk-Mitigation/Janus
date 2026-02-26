"""
Policy loading and persistence.

Handles reading policies from JSON files or raw dicts, converting between
the user-facing JSON format and the internal tuple format, and saving
policies back to disk.

Supported JSON formats:

1. Full format (recommended):
   {
       "tool_name": [
           {
               "priority": 1,
               "effect": 0,
               "conditions": {"arg": {"type": "string", "pattern": "..."}},
               "fallback": 0
           }
       ]
   }

2. Shorthand format (conditions only — implies priority=1, effect=allow, fallback=error):
   {
       "tool_name": {
           "arg_name": {"type": "string", "pattern": "..."}
       }
   }
"""

import json
from pathlib import Path
from typing import Any

from janus.exceptions import PolicyLoadError


# Internal type: list of (priority, effect, conditions, fallback) tuples
PolicyRule = tuple[int, int, dict, int]
PolicyDict = dict[str, list[PolicyRule]]


def parse_policy(source: "str | Path | dict") -> PolicyDict:
    """
    Parse a policy from a file path or dict into the internal tuple format.

    Args:
        source: A path to a JSON file, or a dict in one of the supported formats.

    Returns:
        Parsed policy in internal format: ``{tool_name: [(priority, effect, conditions, fallback), ...]}``

    Raises:
        PolicyLoadError: If the source cannot be read or parsed.
    """
    if isinstance(source, dict):
        raw = source
    elif isinstance(source, (str, Path)):
        raw = _read_json_file(source)
    else:
        raise PolicyLoadError(
            f"Unsupported policy source type: {type(source).__name__}. "
            "Expected a file path (str/Path) or dict."
        )

    return _convert(raw)


def save_policy(policy: PolicyDict, path: "str | Path") -> None:
    """
    Serialize a policy dict to a JSON file.

    The saved format uses human-readable dicts (not raw tuples) so that
    the file can be edited manually and re-loaded later.

    Args:
        policy: Internal policy dict.
        path: Destination file path.
    """
    path = Path(path)
    serialized: dict[str, list[dict]] = {}

    for tool_name, rules in policy.items():
        serialized[tool_name] = [
            {
                "priority": r[0],
                "effect": r[1],
                "conditions": r[2],
                "fallback": r[3] if len(r) > 3 else 0,
            }
            for r in rules
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2)


def validate_policy_structure(policy: PolicyDict, tools: list[dict[str, Any]]) -> list[str]:
    """
    Cross-validate a parsed policy against a list of tool definitions.

    Checks that:
    - Every tool named in the policy exists in the tool list.
    - Every condition argument exists in the corresponding tool's schema.
    - Every condition schema is valid JSON Schema.

    Returns a list of warning strings. Empty list = no issues found.
    """
    from janus.policy.validator import validate_schema

    warnings: list[str] = []
    tool_index = {t["name"]: t.get("args", {}) for t in tools}

    for tool_name, rules in policy.items():
        if tool_name not in tool_index:
            warnings.append(f"Policy references unknown tool '{tool_name}'.")
            continue

        tool_args = tool_index[tool_name]

        for rule in rules:
            if len(rule) < 3:
                warnings.append(f"Malformed rule for '{tool_name}': {rule}")
                continue

            conditions = rule[2]
            for arg_name, restriction in conditions.items():
                if arg_name not in tool_args:
                    warnings.append(
                        f"Policy condition for '{tool_name}.{arg_name}' "
                        "references an unknown argument."
                    )
                if isinstance(restriction, dict):
                    for w in validate_schema(restriction):
                        warnings.append(f"'{tool_name}.{arg_name}': {w}")

    return warnings


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _read_json_file(path: "str | Path") -> dict:
    path = Path(path)
    if not path.exists():
        raise PolicyLoadError(f"Policy file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise PolicyLoadError(f"Invalid JSON in policy file '{path}': {exc}")
    except OSError as exc:
        raise PolicyLoadError(f"Cannot read policy file '{path}': {exc}")


def _convert(raw: dict) -> PolicyDict:
    """Convert raw JSON policy dict → internal tuple format."""
    converted: PolicyDict = {}

    for tool_name, value in raw.items():
        converted[tool_name] = []

        if isinstance(value, list):
            # Full format: list of rule dicts or tuples
            for item in value:
                if isinstance(item, dict):
                    converted[tool_name].append((
                        item.get("priority", 1),
                        item.get("effect", 0),
                        item.get("conditions", {}),
                        item.get("fallback", 0),
                    ))
                elif isinstance(item, (list, tuple)) and len(item) >= 4:
                    converted[tool_name].append(tuple(item[:4]))

        elif isinstance(value, dict):
            if _looks_like_rule(value):
                # Single rule dict
                converted[tool_name].append((
                    value.get("priority", 1),
                    value.get("effect", 0),
                    value.get("conditions", {}),
                    value.get("fallback", 0),
                ))
            else:
                # Shorthand: treat the whole dict as conditions
                converted[tool_name].append((1, 0, value, 0))

    return converted


def _looks_like_rule(d: dict) -> bool:
    """Heuristic: does this dict look like a policy rule vs. a conditions dict?"""
    return bool({"priority", "effect", "conditions", "fallback"} & set(d.keys()))
