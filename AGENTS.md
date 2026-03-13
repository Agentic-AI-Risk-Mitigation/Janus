# Repository Guidelines

## Project Structure & Module Organization
`janus/` contains the library code: `llm/` for provider and runner logic, `policy/` for enforcement (including the PDE subpackage `policy/pde/` for SpiceDB-backed ReBAC and taint tracking), `tools/` for built-in and custom tool plumbing, and `adapters/` for framework integrations. Keep coverage in `tests/`, with the current suite centered on `tests/test_examples/`. Use `examples/` for runnable security scenarios, `demos/` for the FastAPI/web demo surface, and `docs/` plus `mkdocs.yml` for documentation.

## Build, Test, and Development Commands
Use `uv` for local setup and repeatable execution:

```bash
uv sync --extra dev
uv sync --extra all --extra dev
uv run pytest
uv run pytest tests/test_examples/test_demo1_enforcement.py
uv run ruff check .
uv run ruff format .
uv run mypy janus
uv run mkdocs serve
uv run python -m build
```

The first two commands install core or full development dependencies. Run targeted `pytest` commands while iterating, then `ruff`, `mypy`, and a full test pass before a PR.

## Coding Style & Naming Conventions
Target Python 3.11+ with 4-space indentation, type hints on public interfaces, and concise triple-quoted docstrings where behavior is not obvious. Ruff enforces import ordering and core lint rules; formatting should be done with `uv run ruff format .`. Follow existing naming: modules and functions in `snake_case`, classes in `PascalCase`, constants in `UPPER_CASE`. Keep new policy, prompt, and scenario files descriptive, for example `examples/scenarios/<scenario_name>/scenario.py`.

## Testing Guidelines
Pytest is configured via `pyproject.toml` with `tests/` as the test root and `asyncio_mode = auto`. Name files `test_*.py` and keep fixtures close to the tests that use them. Add regression tests for new policy rules, tool behavior, and provider adapters; scenario changes should usually update `tests/test_examples/`. No coverage gate is configured, so meaningful regression coverage is expected.

## Commit & Pull Request Guidelines
Recent history uses short, imperative subjects such as `Clean up: remove dead schema.zed...` and `Add CLAUDE.md...`. Keep commits focused, use the imperative mood, and add a scope prefix when it clarifies intent. PRs should explain the behavioral change, list validation performed (`uv run pytest`, `ruff`, `mypy`), link related issues, and include screenshots only when UI/demo pages change.

## Security & Configuration Tips
Start from `.env.example`, never commit real API keys, and review `SECURITY.md` before touching provider auth or policy enforcement flows. Changes to built-in tools or `policy/` code should preserve Janus's default-deny behavior.
