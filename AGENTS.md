# Repository Guidelines

## Project Structure & Module Organization
`janus/` contains the library code: `llm/` for provider and runner logic, `policy/` for enforcement (including the PDE subpackage `policy/pde/` for SpiceDB-backed ReBAC and taint tracking), `tools/` for built-in and custom tool plumbing, and `adapters/` for framework integrations. Use `examples/` for runnable security scenarios and the FastAPI/web demo surface, and `docs/` plus `mkdocs.yml` for documentation. The current `main` branch does not include the historical `tests/` tree, so example flows are the primary checked-in validation surface.

## Build, Test, and Development Commands
Use `uv` for local setup and repeatable execution:

```bash
uv sync --extra dev
uv sync --extra all --extra dev
./scripts/run_demo_webapp.sh
./scripts/run_demo_webapp.sh --with-spicedb
uv run python -m examples.run --list
uv run python -m examples.run coding_agent_poisoned_readme --protected
uv run uvicorn examples.app:app --reload
uv run pytest
uv run ruff check .
uv run ruff format .
uv run mypy janus
uv run mkdocs serve
uv run python -m build
```

The first two commands install core or full development dependencies; use `uv sync --extra all --extra dev` when working on the example runner because it imports LangChain. `scripts/run_demo_webapp.sh` is the quickest way to launch the FastAPI demo UI from the repo root, and `--with-spicedb` also boots the local SpiceDB container for PDE-backed scenarios. On the current `main` branch, prefer focused `examples/` smoke tests while iterating. If your checkout also includes a `tests/` tree, run `pytest`, then `ruff`, `mypy`, and the relevant demo flow before a PR.

## Coding Style & Naming Conventions
Target Python 3.11+ with 4-space indentation, type hints on public interfaces, and concise triple-quoted docstrings where behavior is not obvious. Ruff enforces import ordering and core lint rules; formatting should be done with `uv run ruff format .`. Follow existing naming: modules and functions in `snake_case`, classes in `PascalCase`, constants in `UPPER_CASE`. Keep new policy, prompt, and scenario files descriptive, for example `examples/scenarios/<scenario_name>/scenario.py`.

## Testing Guidelines
Pytest is still configured via `pyproject.toml` with `asyncio_mode = auto`, but the current `main` branch does not check in the historical `tests/` tree. If you add or restore regression tests, place them under `tests/`, name files `test_*.py`, and keep fixtures close to the tests that use them. In the meantime, validate policy, adapter, and scenario changes with focused runs under `examples/`, and add regression coverage when the test suite is present in your checkout.

## Commit & Pull Request Guidelines
Recent history uses short, imperative subjects such as `Clean up: remove dead schema.zed...` and `Add CLAUDE.md...`. Keep commits focused, use the imperative mood, and add a scope prefix when it clarifies intent. PRs should explain the behavioral change, list validation performed (for example `uv run python -m examples.run coding_agent_poisoned_readme --protected`, `uv run pytest` when `tests/` is present, `ruff`, `mypy`), link related issues, and include screenshots only when UI/demo pages change.

## Security & Configuration Tips
Start from `.env.example`, never commit real API keys, and review `SECURITY.md` before touching provider auth or policy enforcement flows. Changes to built-in tools or `policy/` code should preserve Janus's default-deny behavior.
