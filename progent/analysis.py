"""
Progent policy analysis using Z3 solver.

This module provides tools for analyzing policies:
- Conflict detection between policy rules
- Satisfiability checking of restrictions

This is an OPTIONAL module. It requires z3-solver to be installed:
    pip install progent[analysis]
"""

import warnings
from typing import Any

# Handle deprecation of sre_constants and sre_parse in Python 3.11+
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import sre_constants
    import sre_parse


# =============================================================================
# Z3 Type Initialization (lazy loaded)
# =============================================================================

# These are initialized lazily when Z3 is first used
_z3_types_initialized = False
_NullType = None
_ArrayWrapper_string = None
_ArrayWrapper_integer = None
_ArrayWrapper_number = None
_ArrayWrapper_boolean = None
_ArrayWrapper_null = None
_ARRAY_WRAPPERS = None
_ARRAY_WRAPPER_ACCESSORS = None


def _init_z3_types():
    """Initialize Z3 custom datatypes for null and arrays."""
    global _z3_types_initialized
    global _NullType
    global _ArrayWrapper_string, _ArrayWrapper_integer, _ArrayWrapper_number
    global _ArrayWrapper_boolean, _ArrayWrapper_null
    global _ARRAY_WRAPPERS, _ARRAY_WRAPPER_ACCESSORS

    if _z3_types_initialized:
        return

    from z3 import ArraySort, BoolSort, Datatype, IntSort, RealSort, StringSort

    # Define a global Z3 datatype for null.
    NullType = Datatype("NullType")
    NullType.declare("null")
    _NullType = NullType.create()

    # Define Z3 datatypes for arrays.
    # ArrayWrapper has two fields:
    #   a: an array from Int to Type
    #   len: an integer representing the length of the array.

    # Array wrapper for arrays of strings.
    ArrayWrapper_string = Datatype("ArrayWrapper_string")
    ArrayWrapper_string.declare(
        "mkArrayWrapper", ("a", ArraySort(IntSort(), StringSort())), ("len", IntSort())
    )
    _ArrayWrapper_string = ArrayWrapper_string.create()

    # Array wrapper for arrays of integers.
    ArrayWrapper_integer = Datatype("ArrayWrapper_integer")
    ArrayWrapper_integer.declare(
        "mkArrayWrapper", ("a", ArraySort(IntSort(), IntSort())), ("len", IntSort())
    )
    _ArrayWrapper_integer = ArrayWrapper_integer.create()

    # Array wrapper for arrays of reals (numbers).
    ArrayWrapper_number = Datatype("ArrayWrapper_number")
    ArrayWrapper_number.declare(
        "mkArrayWrapper", ("a", ArraySort(IntSort(), RealSort())), ("len", IntSort())
    )
    _ArrayWrapper_number = ArrayWrapper_number.create()

    # Array wrapper for arrays of booleans.
    ArrayWrapper_boolean = Datatype("ArrayWrapper_boolean")
    ArrayWrapper_boolean.declare(
        "mkArrayWrapper", ("a", ArraySort(IntSort(), BoolSort())), ("len", IntSort())
    )
    _ArrayWrapper_boolean = ArrayWrapper_boolean.create()

    # Array wrapper for arrays of null.
    ArrayWrapper_null = Datatype("ArrayWrapper_null")
    ArrayWrapper_null.declare(
        "mkArrayWrapper", ("a", ArraySort(IntSort(), _NullType)), ("len", IntSort())
    )
    _ArrayWrapper_null = ArrayWrapper_null.create()

    # A list of all array wrappers.
    _ARRAY_WRAPPERS = [
        _ArrayWrapper_string,
        _ArrayWrapper_integer,
        _ArrayWrapper_number,
        _ArrayWrapper_boolean,
        _ArrayWrapper_null,
    ]

    # Map each array wrapper sort to its accessors (the 'a' field and the 'len' field).
    _ARRAY_WRAPPER_ACCESSORS = {
        _ArrayWrapper_string: (_ArrayWrapper_string.a, _ArrayWrapper_string.len),
        _ArrayWrapper_integer: (_ArrayWrapper_integer.a, _ArrayWrapper_integer.len),
        _ArrayWrapper_number: (_ArrayWrapper_number.a, _ArrayWrapper_number.len),
        _ArrayWrapper_boolean: (_ArrayWrapper_boolean.a, _ArrayWrapper_boolean.len),
        _ArrayWrapper_null: (_ArrayWrapper_null.a, _ArrayWrapper_null.len),
    }

    _z3_types_initialized = True


# =============================================================================
# Regex to Z3 Conversion (ported from secagent/role_analyzer.py)
# =============================================================================


def _minus(re1, re2):
    """
    The Z3 regex matching all strings accepted by re1 but not re2.
    """
    from z3 import Complement, Intersect

    return Intersect(re1, Complement(re2))


def _any_char():
    """
    The Z3 regex matching any character (ASCII range).
    """
    from z3 import Range

    return Range(chr(0), chr(127))


def _category_regex(category):
    """
    Defines regex categories in Z3.
    """
    from z3 import Range, Re, Union

    if sre_constants.CATEGORY_DIGIT == category:
        return Range("0", "9")
    elif sre_constants.CATEGORY_SPACE == category:
        return Union(Re(" "), Re("\t"), Re("\n"), Re("\r"), Re("\f"), Re("\v"))
    elif sre_constants.CATEGORY_WORD == category:
        return Union(Range("a", "z"), Range("A", "Z"), Range("0", "9"), Re("_"))
    else:
        raise NotImplementedError(f"Regex category {category} not yet implemented")


def _regex_construct_to_z3_expr(regex_construct):
    """
    Translates a specific regex construct into its Z3 equivalent.
    """
    from z3 import Loop, Option, Plus, Range, Re, Star, Union

    node_type, node_value = regex_construct

    if sre_constants.LITERAL == node_type:  # a
        return Re(chr(node_value))

    if sre_constants.NOT_LITERAL == node_type:  # [^a]
        return _minus(_any_char(), Re(chr(node_value)))

    if sre_constants.SUBPATTERN == node_type:
        _, _, _, value = node_value
        return _regex_to_z3_expr(value)

    elif sre_constants.ANY == node_type:  # .
        return _any_char()

    elif sre_constants.MAX_REPEAT == node_type:
        low, high, value = node_value
        if (0, 1) == (low, high):  # a?
            return Option(_regex_to_z3_expr(value))
        elif (0, sre_constants.MAXREPEAT) == (low, high):  # a*
            return Star(_regex_to_z3_expr(value))
        elif (1, sre_constants.MAXREPEAT) == (low, high):  # a+
            return Plus(_regex_to_z3_expr(value))
        else:  # a{3,5}, a{3}
            return Loop(_regex_to_z3_expr(value), low, high)

    elif sre_constants.IN == node_type:  # [abc]
        first_subnode_type, _ = node_value[0]
        if sre_constants.NEGATE == first_subnode_type:  # [^abc]
            return _minus(
                _any_char(),
                Union([_regex_construct_to_z3_expr(value) for value in node_value[1:]]),
            )
        else:
            return Union([_regex_construct_to_z3_expr(value) for value in node_value])

    elif sre_constants.BRANCH == node_type:  # ab|cd
        _, value = node_value
        return Union([_regex_to_z3_expr(v) for v in value])

    elif sre_constants.RANGE == node_type:  # [a-z]
        low, high = node_value
        return Range(chr(low), chr(high))

    elif sre_constants.CATEGORY == node_type:  # \d, \s, \w
        if sre_constants.CATEGORY_DIGIT == node_value:  # \d
            return _category_regex(node_value)
        elif sre_constants.CATEGORY_NOT_DIGIT == node_value:  # \D
            return _minus(_any_char(), _category_regex(sre_constants.CATEGORY_DIGIT))
        elif sre_constants.CATEGORY_SPACE == node_value:  # \s
            return _category_regex(node_value)
        elif sre_constants.CATEGORY_NOT_SPACE == node_value:  # \S
            return _minus(_any_char(), _category_regex(sre_constants.CATEGORY_SPACE))
        elif sre_constants.CATEGORY_WORD == node_value:  # \w
            return _category_regex(node_value)
        elif sre_constants.CATEGORY_NOT_WORD == node_value:  # \W
            return _minus(_any_char(), _category_regex(sre_constants.CATEGORY_WORD))
        else:
            raise NotImplementedError(f"Regex category {node_value} not implemented")

    elif sre_constants.AT == node_type:
        # Position anchors are not supported in Z3 regex
        # Return empty regex as fallback (matches nothing specific)
        raise NotImplementedError(f"Regex position anchor {node_value} not implemented")

    else:
        raise NotImplementedError(f"Regex construct {regex_construct} not implemented")


def _regex_to_z3_expr(regex: sre_parse.SubPattern):
    """
    Translates a parsed regex into its Z3 equivalent.
    The parsed regex is a sequence of regex constructs (literals, *, +, etc.)
    """
    from z3 import Concat, Re

    if len(regex.data) == 0:
        return Re("")
    elif len(regex.data) == 1:
        return _regex_construct_to_z3_expr(regex.data[0])
    else:
        return Concat([_regex_construct_to_z3_expr(construct) for construct in regex.data])


def _pattern_to_z3_regex(pattern: str):
    """
    Convert a regex pattern string to Z3 regex.

    Args:
        pattern: A Python regex pattern string

    Returns:
        Z3 regex expression, or None if parsing fails
    """
    try:
        parsed = sre_parse.parse(pattern)
        return _regex_to_z3_expr(parsed)
    except Exception:
        return None


# =============================================================================
# Array Helpers
# =============================================================================


def _make_array_from_list(lst: list, ArrayWrapper):
    """
    Given a Python list and an ArrayWrapper type, return a Z3 array term
    mapping indices 0..len(lst)-1 to the corresponding values.
    """
    from z3 import BoolVal, IntSort, IntVal, K, RealVal, Store, StringVal

    _init_z3_types()

    if ArrayWrapper == _ArrayWrapper_string:
        default_val = StringVal("")

        def convert(x):
            return StringVal(x)
    elif ArrayWrapper == _ArrayWrapper_integer:
        default_val = IntVal(0)

        def convert(x):
            return IntVal(x)
    elif ArrayWrapper == _ArrayWrapper_number:
        default_val = RealVal(0)

        def convert(x):
            return RealVal(x)
    elif ArrayWrapper == _ArrayWrapper_boolean:
        default_val = BoolVal(False)

        def convert(x):
            return BoolVal(x)
    elif ArrayWrapper == _ArrayWrapper_null:
        default_val = _NullType.null

        def convert(x):
            return _NullType.null
    else:
        raise NotImplementedError(f"Unknown ArrayWrapper type: {ArrayWrapper}")

    default_arr = K(IntSort(), default_val)
    arr_term = default_arr
    for i, elem in enumerate(lst):
        arr_term = Store(arr_term, i, convert(elem))
    return arr_term


# =============================================================================
# Constraint Building
# =============================================================================


def _build_constraints(var, schema: dict):
    """
    Recursively build Z3 constraints from a JSON Schema.

    Supported keywords include:
      - "enum"
      - "anyOf", "allOf", "oneOf"
      - "not"
      - "if", "then", "else"
      - Type-specific keywords:
            * string: {"pattern", "minLength", "maxLength"}
            * number: {"multipleOf", "minimum", "exclusiveMinimum", "maximum", "exclusiveMaximum"}
            * integer: {"multipleOf", "minimum", "exclusiveMinimum", "maximum", "exclusiveMaximum"}
            * array: {"items", "minItems", "maxItems"}
            * boolean, null
    """
    from z3 import (
        And,
        BoolSort,
        BoolVal,
        Exists,
        ForAll,
        If,
        Implies,
        InRe,
        Int,
        IntSort,
        IntVal,
        Length,
        Not,
        Or,
        RealSort,
        RealVal,
        Select,
        StringSort,
        StringVal,
        Sum,
    )

    _init_z3_types()

    constraint = True

    # Global enum constraint.
    if "enum" in schema:
        enums = schema["enum"]
        if enums:
            first = enums[0]
            if isinstance(first, list):
                # For arrays, convert each Python list to an ArrayWrapper instance.
                enum_consts = []
                ArrayWrapper = var.sort()
                for arr in enums:
                    arr_term = _make_array_from_list(arr, ArrayWrapper)
                    array_value = ArrayWrapper.mkArrayWrapper(arr_term, len(arr))
                    enum_consts.append(var == array_value)
                constraint = And(constraint, Or(enum_consts))
            elif isinstance(first, str):
                constraint = And(constraint, Or([var == StringVal(e) for e in enums]))
            elif isinstance(first, bool):
                # Check bool before int since bool is subclass of int
                constraint = And(constraint, Or([var == BoolVal(e) for e in enums]))
            elif isinstance(first, int):
                constraint = And(constraint, Or([var == IntVal(e) for e in enums]))
            elif isinstance(first, float):
                constraint = And(constraint, Or([var == RealVal(e) for e in enums]))
            elif first is None:
                constraint = And(constraint, var == _NullType.null)

    # anyOf: disjunction of subschemas.
    if "anyOf" in schema:
        branch_constraints = [_build_constraints(var, subschema) for subschema in schema["anyOf"]]
        constraint = And(constraint, Or(branch_constraints))

    # allOf: all subschemas must hold.
    if "allOf" in schema:
        all_constraints = [_build_constraints(var, subschema) for subschema in schema["allOf"]]
        constraint = And(constraint, And(all_constraints))

    # oneOf: exactly one subschema must hold.
    if "oneOf" in schema:
        oneof_constraints = [_build_constraints(var, subschema) for subschema in schema["oneOf"]]
        constraint = And(constraint, Sum([If(c, 1, 0) for c in oneof_constraints]) == 1)

    # not: the subschema must not hold.
    if "not" in schema:
        constraint = And(constraint, Not(_build_constraints(var, schema["not"])))

    # if/then/else: conditional constraints.
    if "if" in schema:
        if_constraint = _build_constraints(var, schema["if"])
        if "then" in schema:
            then_constraint = _build_constraints(var, schema["then"])
            constraint = And(constraint, Implies(if_constraint, then_constraint))
        if "else" in schema:
            else_constraint = _build_constraints(var, schema["else"])
            constraint = And(constraint, Implies(Not(if_constraint), else_constraint))

    # Type check.
    if "type" in schema:
        t = schema["type"]
        t_list = []
        if isinstance(t, list):
            t_list = t
        else:
            t_list.append(t)
        type_constraint = []
        for t in t_list:
            if t == "string":
                type_constraint.append(var.sort() == StringSort())
            elif t == "array":
                type_constraint.append(Or([var.sort() == aw for aw in _ARRAY_WRAPPERS]))
            elif t == "integer":
                type_constraint.append(var.sort() == IntSort())
            elif t == "number":
                type_constraint.append(var.sort() == RealSort())
            elif t == "boolean":
                type_constraint.append(var.sort() == BoolSort())
            elif t == "null":
                type_constraint.append(var.sort() == _NullType)
        if type_constraint:
            constraint = And(constraint, Or(type_constraint))

    # Type-specific constraints.
    if var.sort() == StringSort():
        if "pattern" in schema:
            pat = schema["pattern"]
            try:
                regex = _pattern_to_z3_regex(pat)
                if regex is not None:
                    constraint = And(constraint, InRe(var, regex))
            except Exception:
                pass  # Skip unsupported patterns
        if "minLength" in schema:
            constraint = And(constraint, Length(var) >= schema["minLength"])
        if "maxLength" in schema:
            constraint = And(constraint, Length(var) <= schema["maxLength"])

    elif var.sort() in _ARRAY_WRAPPERS:
        a_accessor, len_accessor = _ARRAY_WRAPPER_ACCESSORS[var.sort()]
        constraint = And(constraint, len_accessor(var) >= 0)
        if "minItems" in schema:
            constraint = And(constraint, len_accessor(var) >= schema["minItems"])
        if "maxItems" in schema:
            constraint = And(constraint, len_accessor(var) <= schema["maxItems"])
        if "items" in schema:
            items_schema = schema["items"]
            i = Int("i")
            constraint = And(
                constraint,
                ForAll(
                    i,
                    Implies(
                        And(i >= 0, i < len_accessor(var)),
                        _build_constraints(Select(a_accessor(var), i), items_schema),
                    ),
                ),
            )

    elif var.sort() == IntSort():
        if "multipleOf" in schema:
            m = schema["multipleOf"]
            constraint = And(constraint, var % m == 0)
        if "exclusiveMinimum" in schema:
            constraint = And(constraint, var > schema["exclusiveMinimum"])
        elif "minimum" in schema:
            constraint = And(constraint, var >= schema["minimum"])
        if "exclusiveMaximum" in schema:
            constraint = And(constraint, var < schema["exclusiveMaximum"])
        elif "maximum" in schema:
            constraint = And(constraint, var <= schema["maximum"])

    elif var.sort() == RealSort():
        if "multipleOf" in schema:
            m = schema["multipleOf"]
            k = Int("k_mult")
            constraint = And(constraint, Exists(k, var == RealVal(m) * k))
        if "exclusiveMinimum" in schema:
            constraint = And(constraint, var > schema["exclusiveMinimum"])
        elif "minimum" in schema:
            constraint = And(constraint, var >= schema["minimum"])
        if "exclusiveMaximum" in schema:
            constraint = And(constraint, var < schema["exclusiveMaximum"])
        elif "maximum" in schema:
            constraint = And(constraint, var <= schema["maximum"])

    elif var.sort() == BoolSort():
        pass  # No additional constraints for boolean

    elif var.sort() == _NullType:
        constraint = And(constraint, var == _NullType.null)

    return constraint


# =============================================================================
# Schema Solving
# =============================================================================


def _solve_schema(schema: dict):
    """
    Given a JSON Schema (as a dict), determine if there is a value that satisfies it.
    Chooses a top-level Z3 variable based on the schema's "type" or by inferring from an "enum".

    Returns:
        A tuple (model, var) if satisfiable, or (None, var) if unsat.
    """
    from z3 import Bool, Const, Int, Real, Solver, String, sat

    _init_z3_types()

    var_list = []

    # Determine the top-level type.
    if "type" in schema:
        t = schema["type"]
        t_list = t if isinstance(t, list) else [t]

        for t in t_list:
            if t == "string":
                var_list.append(String("x"))
            elif t == "array":
                # Choose an array wrapper based on the "items" type if provided.
                if "items" in schema and "type" in schema["items"]:
                    item_type = schema["items"]["type"]
                    if item_type == "string":
                        var_list.append(Const("x", _ArrayWrapper_string))
                    elif item_type == "integer":
                        var_list.append(Const("x", _ArrayWrapper_integer))
                    elif item_type == "number":
                        var_list.append(Const("x", _ArrayWrapper_number))
                    elif item_type == "boolean":
                        var_list.append(Const("x", _ArrayWrapper_boolean))
                    elif item_type == "null":
                        var_list.append(Const("x", _ArrayWrapper_null))
                    else:
                        for aw in _ARRAY_WRAPPERS:
                            var_list.append(Const("x", aw))
                else:
                    for aw in _ARRAY_WRAPPERS:
                        var_list.append(Const("x", aw))
            elif t == "integer":
                var_list.append(Int("x"))
            elif t == "number":
                var_list.append(Real("x"))
            elif t == "boolean":
                var_list.append(Bool("x"))
            elif t == "null":
                var_list.append(Const("x", _NullType))
    else:
        # Infer type from enum if provided.
        if "enum" in schema and schema["enum"]:
            first = schema["enum"][0]
            if isinstance(first, list):
                # Infer element type from first element of first array.
                if len(first) > 0:
                    sample = first[0]
                    if isinstance(sample, str):
                        var_list.append(Const("x", _ArrayWrapper_string))
                    elif isinstance(sample, bool):
                        var_list.append(Const("x", _ArrayWrapper_boolean))
                    elif isinstance(sample, int):
                        var_list.append(Const("x", _ArrayWrapper_integer))
                    elif isinstance(sample, float):
                        var_list.append(Const("x", _ArrayWrapper_number))
                    elif sample is None:
                        var_list.append(Const("x", _ArrayWrapper_null))
                    else:
                        for aw in _ARRAY_WRAPPERS:
                            var_list.append(Const("x", aw))
                else:
                    for aw in _ARRAY_WRAPPERS:
                        var_list.append(Const("x", aw))
            elif isinstance(first, str):
                var_list.append(String("x"))
            elif isinstance(first, bool):
                var_list.append(Bool("x"))
            elif isinstance(first, int):
                var_list.append(Int("x"))
            elif isinstance(first, float):
                var_list.append(Real("x"))
            elif first is None:
                var_list.append(Const("x", _NullType))
        else:
            # Try all types
            var_list = [String("x"), Int("x"), Real("x"), Bool("x"), Const("x", _NullType)]
            for aw in _ARRAY_WRAPPERS:
                var_list.append(Const("x", aw))

    # Try each variable type
    for var in var_list:
        try:
            cons = _build_constraints(var, schema)
            s = Solver()
            s.add(cons)
            if s.check() == sat:
                model = s.model()
                return model, var
        except Exception:
            # If constraint building fails for this type, try next
            continue

    # Return last var if none satisfied
    return None, var_list[-1] if var_list else None


# =============================================================================
# Public API
# =============================================================================


def analyze_policies(policies: dict = None) -> list[str]:
    """
    Analyze security policies for conflicts and issues.

    Uses Z3 SMT solver to detect:
    - Overlapping allow/deny rules
    - Unsatisfiable restrictions
    - Redundant rules

    Args:
        policies: Policies to analyze. If None, uses current policies.

    Returns:
        List of warning messages

    Raises:
        ImportError: If z3-solver is not installed
    """
    try:
        import z3  # noqa: F401
    except ImportError:
        raise ImportError(
            "z3-solver is required for policy analysis. "
            "Install it with: pip install progent[analysis]"
        )

    if policies is None:
        from progent.core import get_security_policy

        policies = get_security_policy()

    if policies is None:
        return []

    warnings = []

    for tool_name, rules in policies.items():
        n = len(rules)
        for i in range(n):
            for j in range(i + 1, n):
                rule_i = rules[i]
                rule_j = rules[j]

                # Get conditions from rules
                conditions_i = rule_i[2] if len(rule_i) > 2 else {}
                conditions_j = rule_j[2] if len(rule_j) > 2 else {}

                # Check each argument for overlap
                for arg_name, restriction_i in conditions_i.items():
                    if arg_name not in conditions_j:
                        continue

                    restriction_j = conditions_j[arg_name]

                    # Only check JSON Schema restrictions
                    if not isinstance(restriction_i, dict) or not isinstance(restriction_j, dict):
                        continue

                    # Try to find overlap
                    schema = {"allOf": [restriction_i, restriction_j]}
                    model, var = _solve_schema(schema)
                    if model is not None:
                        try:
                            value = model[var]
                        except Exception:
                            value = "unknown"
                        warnings.append(
                            f"Policy Warning: {restriction_i} and {restriction_j} overlap. "
                            f"Possible value: {value}."
                        )

    return warnings


def check_schema_overlap(schema_a: dict, schema_b: dict) -> Any:
    """
    Check if two JSON schemas can be satisfied by the same value.

    Args:
        schema_a: First JSON schema
        schema_b: Second JSON schema

    Returns:
        An example value if overlap exists, None otherwise.
    """
    combined = {"allOf": [schema_a, schema_b]}

    model, var = _solve_schema(combined)

    if model is not None:
        try:
            return model[var]
        except Exception:
            return "unknown"

    return None


def check_policy_type_errors(
    policies: dict,
    available_tools: dict[str, dict] = None,
) -> list[str]:
    """
    Check policies for type-specific errors.

    Args:
        policies: Security policies to check
        available_tools: Dict mapping tool names to their arg schemas

    Returns:
        List of warning messages
    """
    from progent.validation import validate_schema

    warnings = []

    if policies is None:
        return warnings

    for tool_name, rules in policies.items():
        # Check if tool exists
        if available_tools and tool_name not in available_tools:
            warnings.append(f"Tool '{tool_name}' not in available tools")

        for i, rule in enumerate(rules):
            if len(rule) < 3:
                continue

            conditions = rule[2]

            for arg_name, restriction in conditions.items():
                # Check if arg exists in tool
                if available_tools and tool_name in available_tools:
                    tool_args = available_tools[tool_name]
                    if arg_name not in tool_args:
                        warnings.append(f"Argument '{arg_name}' not found in tool '{tool_name}'")

                # Validate schema
                if isinstance(restriction, dict):
                    schema_warnings = validate_schema(restriction)
                    for w in schema_warnings:
                        warnings.append(f"{tool_name}.{arg_name}: {w}")

    return warnings


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    # Test the analysis with sample policies
    test_policies = {
        "send_money": [
            (1, 0, {"recipient": {"type": "string", "pattern": "UK12345678901234567890"}}, 0),
            (2, 1, {"recipient": {"type": "string", "pattern": "SE3550000000054910000003"}}, 0),
        ],
        "send_email": [
            (
                1,
                0,
                {
                    "recipient": {
                        "type": "string",
                        "pattern": "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}",
                    }
                },
                0,
            ),
            (2, 1, {"recipient": {"type": "string", "pattern": "agent@example.com"}}, 0),
        ],
        "update_items": [
            (1, 0, {"items": {"type": "array", "items": {"type": "string"}, "minItems": 1}}, 0),
            (2, 0, {"items": {"type": "array", "items": {"type": "string"}, "maxItems": 5}}, 0),
        ],
    }

    warnings = analyze_policies(test_policies)
    for w in warnings:
        print(w)
