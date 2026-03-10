# Policy Reference

Policies are JSON documents that map tool names to lists of rules. Each rule specifies whether to allow or deny a tool call and can include argument-level restrictions.

## Format

### Full Format

```json
{
    "read_file": [
        {
            "priority": 1,
            "effect": 0,
            "conditions": {
                "file_path": {
                    "type": "string",
                    "pattern": "^reports/.*\\.csv$"
                }
            },
            "fallback": 0
        }
    ],
    "run_command": [
        {
            "priority": 1,
            "effect": 1,
            "conditions": {},
            "fallback": 0
        }
    ]
}
```

### Shorthand Format

When you only need to restrict argument values, use the shorthand. It implies priority=1, effect=allow, fallback=raise:

```json
{
    "read_file": {
        "file_path": { "type": "string", "pattern": "^data/.*" }
    }
}
```

## Rule Fields

| Field | Type | Description |
|-------|------|-------------|
| `priority` | int | Evaluation order — lower value runs first |
| `effect` | int | `0` = allow, `1` = deny |
| `conditions` | dict | JSON Schema restrictions keyed by argument name |
| `fallback` | int | `0` = raise `PolicyViolation`, `1` = `sys.exit(1)`, `2` = ask user |

## Condition Schemas

Conditions follow JSON Schema syntax. Common patterns:

```json
{ "type": "string", "pattern": "^/safe/path/.*" }
{ "type": "string", "enum": ["ls", "pwd", "cat"] }
{ "type": "integer", "minimum": 0, "maximum": 100 }
{ "type": "array", "items": { "type": "string" } }
```

## Evaluation Logic

1. Rules for a tool are evaluated in ascending `priority` order.
2. **Allow rule** (`effect=0`): if all conditions pass → tool is allowed immediately.
3. **Deny rule** (`effect=1`): if all conditions match → tool is blocked using the configured `fallback`.
4. If no rule matches → the tool is **blocked by default**.
5. Tools not listed in the policy are blocked when a policy is loaded.

At equal priority, deny rules are evaluated before allow rules.

## LLM-Generated Policies

Janus can generate policies from a user query and tool set.

**Automatic on first run**: Pass `policy="generate"` to `JanusAgent`. The policy is generated from the first query and applied before any tool calls.

**Standalone generation**: Use `generate_policy(query, tools, model=...)` to produce a policy without running an agent. Call `save_policy(policy, "policies.json")` to persist.

**Policy refinement**: After an information-gathering tool call, use `refine_policy(query, tools, tool_call_params, tool_call_result, current_policy)` to tighten the policy. For example, after reading a file that contains an email address, the policy can restrict `send_email` to that recipient only.
