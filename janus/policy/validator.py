"""
Policy argument validation.

Validates tool call arguments against JSON Schema restrictions, regex
patterns, or custom callable validators. This is the lowest-level module
in the enforcement stack — it only validates single values and raises
ArgumentValidationError; it knows nothing about policies or tools.
"""

import inspect
import re
from typing import Any, Callable

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate

from janus.exceptions import ArgumentValidationError


def validate_argument(arg_name: str, value: Any, restriction: Any) -> None:
    """
    Validate a single argument against a restriction.

    Three restriction types are supported:

    - ``dict``: Treated as a JSON Schema. The value must pass JSON Schema
      validation (powered by the jsonschema library).
    - ``str``: Treated as a regex pattern. The value must be a string and
      must match the pattern via ``re.match``.
    - ``callable``: A custom validator. It must return a truthy value to
      indicate acceptance, or raise an exception to indicate rejection.

    Args:
        arg_name: Name of the argument (used in error messages).
        value: The value to validate.
        restriction: The restriction to validate against.

    Raises:
        ArgumentValidationError: If validation fails.
    """
    if isinstance(restriction, dict):
        try:
            validate(instance=value, schema=restriction)
        except JsonSchemaValidationError as exc:
            raise ArgumentValidationError(
                argument_name=arg_name,
                value=value,
                restriction=restriction,
                message=f"Argument '{arg_name}' failed schema validation: {exc.message}",
            )

    elif isinstance(restriction, str):
        if not isinstance(value, str):
            raise ArgumentValidationError(
                argument_name=arg_name,
                value=value,
                restriction=restriction,
                message=(
                    f"Argument '{arg_name}' must be a string to match regex pattern "
                    f"'{restriction}', got {type(value).__name__}."
                ),
            )
        if not re.match(restriction, value):
            raise ArgumentValidationError(
                argument_name=arg_name,
                value=value,
                restriction=restriction,
                message=(
                    f"Argument '{arg_name}' value {value!r} does not match "
                    f"pattern '{restriction}'."
                ),
            )

    elif callable(restriction):
        try:
            result = restriction(value)
            if not result:
                try:
                    source = inspect.getsource(restriction)
                except OSError:
                    source = repr(restriction)
                raise ArgumentValidationError(
                    argument_name=arg_name,
                    value=value,
                    restriction=restriction,
                    message=(
                        f"Argument '{arg_name}' value {value!r} was rejected by "
                        f"custom validator: {source}"
                    ),
                )
        except ArgumentValidationError:
            raise
        except Exception as exc:
            raise ArgumentValidationError(
                argument_name=arg_name,
                value=value,
                restriction=restriction,
                message=f"Custom validator raised an error for '{arg_name}': {exc}",
            )

    else:
        raise ArgumentValidationError(
            argument_name=arg_name,
            value=value,
            restriction=restriction,
            message=(
                f"Unsupported restriction type for '{arg_name}': "
                f"{type(restriction).__name__}. Expected dict (JSON Schema), "
                "str (regex), or callable."
            ),
        )


def validate_schema(schema: dict) -> list[str]:
    """
    Check a JSON Schema dict for correctness.

    Returns a list of warning strings. An empty list means the schema looks
    valid. Does not raise — designed to be used for advisory checks.
    """
    from jsonschema.validators import validator_for

    warnings: list[str] = []

    try:
        cls = validator_for(schema)
        cls.check_schema(schema)
    except Exception as exc:
        warnings.append(f"Invalid JSON Schema: {exc}")

    warnings.extend(_check_type_keyword_misuse(schema))
    return warnings


# Allowed type-specific keywords by JSON Schema type
_KEYWORDS_BY_TYPE: dict[str, set[str]] = {
    "string": {"pattern", "minLength", "maxLength", "format", "enum"},
    "number": {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "enum"},
    "integer": {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "enum"},
    "array": {"items", "minItems", "maxItems", "uniqueItems"},
    "object": {"properties", "required", "additionalProperties"},
    "boolean": {"enum"},
    "null": set(),
}

_ALL_TYPE_KEYWORDS: set[str] = set().union(*_KEYWORDS_BY_TYPE.values())


def _check_type_keyword_misuse(schema: dict, path: str = "") -> list[str]:
    """Recursively detect type-specific keyword misuse in a JSON Schema."""
    warnings: list[str] = []

    if not isinstance(schema, dict):
        return warnings

    schema_type = schema.get("type")
    if schema_type:
        if isinstance(schema_type, str):
            allowed = _KEYWORDS_BY_TYPE.get(schema_type, set())
        elif isinstance(schema_type, list):
            allowed = set().union(*(_KEYWORDS_BY_TYPE.get(t, set()) for t in schema_type))
        else:
            allowed = set()

        for key in schema:
            if key in _ALL_TYPE_KEYWORDS and key not in allowed:
                warnings.append(
                    f"Schema keyword '{key}' is not valid for type '{schema_type}' at '{path or 'root'}'."
                )

    # Recurse into nested schemas
    for key in ("not", "if", "then", "else"):
        if key in schema and isinstance(schema[key], dict):
            warnings.extend(_check_type_keyword_misuse(schema[key], f"{path}.{key}"))

    for compound in ("anyOf", "allOf", "oneOf"):
        if compound in schema and isinstance(schema[compound], list):
            for i, sub in enumerate(schema[compound]):
                warnings.extend(_check_type_keyword_misuse(sub, f"{path}.{compound}[{i}]"))

    if "items" in schema:
        items = schema["items"]
        if isinstance(items, dict):
            warnings.extend(_check_type_keyword_misuse(items, f"{path}.items"))
        elif isinstance(items, list):
            for i, item in enumerate(items):
                warnings.extend(_check_type_keyword_misuse(item, f"{path}.items[{i}]"))

    if "properties" in schema and isinstance(schema["properties"], dict):
        for prop, sub_schema in schema["properties"].items():
            if isinstance(sub_schema, dict):
                warnings.extend(
                    _check_type_keyword_misuse(sub_schema, f"{path}.properties.{prop}")
                )

    return warnings
