# Demo TODO

## Currently Hardcoded / Mocked

- **LLM responses**: Both panels use `ScriptedChatModel` which returns pre-defined `AIMessage` lists. No real LLM API call is made. The config file (`examples/config.yaml`) has `OPENROUTER_API_KEY` and model settings ready but the live LLM path is not wired up yet.
- **Network tools**: `fetch_url` never makes real HTTP requests. Returns canned responses from `url_responses` mappings or generic strings.
- **Git tools**: `git_commit` and `git_push` return fake output strings. No real git operations.
- **Attack detection**: `check_exfiltration` and `check_malicious_push` use simple regex/keyword matching, not real traffic analysis.
- **User input**: In scripted mode the user prompt is pre-filled and auto-played. There is no text input box for interactive mode yet.

## Not Yet Implemented

- **Demos 2-4, 6-10**: Only Demo 1 (Poisoned README) and Demo 5 (Taint Cascade) are implemented. See `DEMOS.md` for the full list of 10 scenarios.
- **Live LLM mode**: Replace `ScriptedChatModel` with a real `ChatOpenAI` (via OpenRouter) using the config in `examples/config.yaml`. The `ScenarioRunner` needs a code path that creates a real LLM instead of the scripted one.
- **Interactive user input**: Add a text input to the frontend so users can type prompts in live LLM mode.
- **Streaming responses**: The current playback uses fixed delays. Live LLM mode should stream tokens.
- **Playback controls**: Pause/resume/speed controls for scripted demos.
- **Policy reason display**: Show the policy rule that caused a block (currently shows the reason string but not the specific rule match).

## Improvements

- **Frontend polish**: The terminal panel could use syntax highlighting for tool arguments. Scrolling behavior could be smoother.
- **Error handling**: The web app silently swallows some WebSocket errors. Add reconnection logic.
- **Concurrent workspace isolation**: URL responses for `fetch_url` are still partially global (the legacy interface). Each panel's tools are isolated via closures, but the global `register_url_response` API still exists for backward compat.
