# Framework Adapters Architecture

## Key Principle: NO DUPLICATION

Tools are defined ONCE in `core/tool_definitions.py`. Framework adapters only convert formats.

## Structure

```
core/
├── tool_definitions.py   # SINGLE SOURCE OF TRUTH for all tools
├── secured_executor.py   # Unified policy enforcement wrapper
frameworks/
├── langchain_agent.py    # Converts tools to LangChain format
├── adk_agent.py          # Converts tools to Gemini format
├── raw_sdk_agent.py      # Converts tools to OpenAI format
```

## Adding a New Tool

1. Add `ToolDefinition` in `core/tool_definitions.py`
2. Implement handler in `tools/` directory
3. Done. All frameworks automatically get the new tool.

## Adding a New Framework

1. Create `frameworks/new_agent.py`
2. Extend `BaseAgent`
3. Implement `_create_tools()` to convert `TOOL_DEFINITIONS` to framework format
4. Use `create_secured_handler()` for policy enforcement

## DO NOT

- Define tools in agent files
- Duplicate tool descriptions
- Implement policy logic in agents
- Copy-paste between framework adapters
