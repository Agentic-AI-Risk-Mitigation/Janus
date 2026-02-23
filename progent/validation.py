"""
Progent validation utilities.

Handles JSON Schema validation and other restriction types.
"""

import inspect
import re
from typing import Any, Callable

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate

from progent.exceptions import PolicyValidationError


def check_argument(
    arg_name: str,
    value: Any,
    restriction: Any,
) -> None:
    """
    Validate a single argument against a restriction.

    Supports multiple restriction types:
    - dict: JSON Schema validation
    - str: Regex pattern matching
    - callable: Custom validation function

    Args:
        arg_name: Name of the argument being checked
        value: The value to validate
        restriction: The restriction to validate against

    Raises:
        PolicyValidationError: If validation fails
    """
    if isinstance(restriction, dict):
        # JSON Schema validation
        try:
            validate(instance=value, schema=restriction)
        except JsonSchemaValidationError as e:
            raise PolicyValidationError(
                argument_name=arg_name,
                value=value,
                restriction=restriction,
                message=f"Invalid value for argument '{arg_name}': {e.message}",
            )

    elif isinstance(restriction, str):
        # Regex pattern matching
        if not isinstance(value, str):
            raise PolicyValidationError(
                argument_name=arg_name,
                value=value,
                restriction=restriction,
                message=f"Cannot match regex against non-string value for '{arg_name}'",
            )
        if not re.match(restriction, value):
            raise PolicyValidationError(
                argument_name=arg_name,
                value=value,
                restriction=restriction,
                message=(
                    f"Invalid value for argument '{arg_name}': "
                    f"value '{value}' does not match pattern '{restriction}'"
                ),
            )

    elif isinstance(restriction, Callable):
        # Custom validation function
        try:
            result = restriction(value)
            if not result:
                source = inspect.getsource(restriction)
                raise PolicyValidationError(
                    argument_name=arg_name,
                    value=value,
                    restriction=restriction,
                    message=(
                        f"Invalid value for argument '{arg_name}': "
                        f"value '{value}' rejected by validator: {source}"
                    ),
                )
        except PolicyValidationError:
            raise
        except Exception as e:
            raise PolicyValidationError(
                argument_name=arg_name,
                value=value,
                restriction=restriction,
                message=f"Validation error for '{arg_name}': {e}",
            )

    else:
        raise PolicyValidationError(
            argument_name=arg_name,
            value=value,
            restriction=restriction,
            message=f"Unsupported restriction type: {type(restriction).__name__}",
        )


def validate_schema(schema: dict) -> list[str]:
    """
    Validate a JSON Schema for correctness.

    Returns a list of warning messages if there are issues.
    """
    from jsonschema.validators import validator_for

    warnings = []

    try:
        validator = validator_for(schema)
        validator.check_schema(schema)
    except Exception as e:
        warnings.append(f"Invalid JSON Schema: {e}")

    # Check for type-specific keyword misuse
    warnings.extend(_check_type_specific_keywords(schema))

    return warnings


# Keywords allowed for each type
_ALLOWED_KEYWORDS_BY_TYPE = {
    "string": {"pattern", "maxLength", "minLength", "format", "enum"},
    "number": {"multipleOf", "maximum", "exclusiveMaximum", "minimum", "exclusiveMinimum", "enum"},
    "integer": {"multipleOf", "maximum", "exclusiveMaximum", "minimum", "exclusiveMinimum", "enum"},
    "array": {"items", "maxItems", "minItems", "uniqueItems"},
    "object": {"properties", "required", "additionalProperties"},
    "boolean": {"enum"},
    "null": set(),  # Empty set for null type
}

_ALL_TYPE_SPECIFIC_KEYWORDS = set()
for kws in _ALLOWED_KEYWORDS_BY_TYPE.values():
    _ALL_TYPE_SPECIFIC_KEYWORDS |= kws


def _check_type_specific_keywords(schema: dict, path: str = "") -> list[str]:
    """
    Recursively check a JSON Schema for invalid type-specific keyword usage.
    """
    warnings = []

    if not isinstance(schema, dict):
        return warnings

    if "type" in schema:
        types = schema["type"]

        # Determine allowed keywords
        if isinstance(types, list):
            allowed = set()
            for t in types:
                allowed |= _ALLOWED_KEYWORDS_BY_TYPE.get(t, set())
        elif isinstance(types, str):
            allowed = _ALLOWED_KEYWORDS_BY_TYPE.get(types, set())
        else:
            allowed = set()

        # Check each key
        for key in schema:
            if key in _ALL_TYPE_SPECIFIC_KEYWORDS and key not in allowed:
                warnings.append(f"Warning at {path}: '{key}' is not valid for type '{types}'")

    # Recursively check nested schemas
    for k, v in schema.items():
        if k in ["anyOf", "allOf", "oneOf"]:
            if isinstance(v, list):
                for i, item in enumerate(v):
                    warnings.extend(_check_type_specific_keywords(item, f"{path}.{k}[{i}]"))
        elif k in ["not", "if", "then", "else"]:
            if isinstance(v, dict):
                warnings.extend(_check_type_specific_keywords(v, f"{path}.{k}"))
        elif k == "items":
            if isinstance(v, dict):
                warnings.extend(_check_type_specific_keywords(v, f"{path}.items"))
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    warnings.extend(_check_type_specific_keywords(item, f"{path}.items[{i}]"))
        elif k == "properties":
            if isinstance(v, dict):
                for prop_name, prop_schema in v.items():
                    if isinstance(prop_schema, dict):
                        warnings.extend(
                            _check_type_specific_keywords(
                                prop_schema, f"{path}.properties.{prop_name}"
                            )
                        )

    return warnings


def validate_policy_against_tools(
    policy: dict[str, list],
    tools: list[dict[str, Any]],
) -> list[str]:
    """
    Validate a security policy against available tool definitions.

    Checks:
    - Tool existence
    - Argument existence
    - Type-specific keyword usage in schemas

    Args:
        policy: The security policy dict
        tools: List of tool definitions

    Returns:
        List of warning messages
    """
    warnings = []
    if not policy:
        return warnings

    # Create a lookup dict for tools: tool_name -> args_dict
    tools_lookup = {t["name"]: t.get("args", {}) for t in tools}

    for tool_name, rules in policy.items():
        if tool_name not in tools_lookup:
            warnings.append(f"Policy Warning: Tool '{tool_name}' is not available.")
            continue

        tool_args = tools_lookup[tool_name]

        for rule in rules:
            # Rule format: (priority, effect, conditions, fallback)
            if len(rule) < 3:
                continue

            conditions = rule[2]

            for arg_name, restriction in conditions.items():
                if arg_name not in tool_args:
                    warnings.append(
                        f"Policy Warning: Argument '{arg_name}' is not available for tool '{tool_name}'."
                    )

                if isinstance(restriction, dict):  # JSON Schema
                    # Validate schema structure
                    warnings.extend(validate_schema(restriction))

                    # Check for type-specific keyword misuse
                    from jsonschema.validators import validator_for

                    try:
                        validator = validator_for(restriction)
                        validator.check_schema(restriction)
                    except Exception as e:
                        warnings.append(
                            f"Policy Warning: Invalid restriction for argument '{arg_name}' "
                            f"in tool '{tool_name}': {e}"
                        )

    return warnings
