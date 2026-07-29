---
name: release
description: Cut and publish a janus-guard release to PyPI. Use when asked to release, tag, or publish a new version of this package.
---

# Release janus-guard

Execute in this order. Do not reorder: gate → versions → CHANGELOG → commit → build → tag → confirm → publish → verify.

## 1. Gate — everything must pass before touching version numbers

1. `uv run pytest` — full suite. `tests/smoke/` skips unless `JANUS_LIVE_SMOKE=1`; do NOT require a live smoke run. Instead, read the "Verified runs" table in `plans/claude-agent-sdk-hardening.md` and report the date and pinned SDK/CLI versions of the last verified run in your release summary.
2. `uv run ruff check .`
3. `uv run mypy janus`
4. **Core-only import simulation** (belt-and-braces on top of `tests/test_import_hygiene.py`, which already ran in pytest). Run a Python subprocess that:
   - installs a `sys.meta_path` finder whose `find_spec` raises `ImportError` for top-level `openai`, `jinja2`, and `dotenv`;
   - then does `import janus`, `import janus.testing`, `import janus.adapters.claude_agent_sdk`;
   - runs one `janus.testing.decide(...)` call end to end, e.g.
     `decide({"read_file": [{"priority": 1, "effect": 0, "conditions": {"path": r"^README"}, "fallback": 0}]}, "read_file", {"path": "README.md"})` and asserts `.allowed`;
   - asserts `janus.generate_policy` raises an `ImportError` whose message mentions the `generate` extra.
   Rationale: core install must stay stdlib + `jsonschema` + `pydantic`; this catches an accidental eager import that unit tests on the dev env can miss.
5. **Extras-matrix drift check**: open `[project.optional-dependencies]` in `pyproject.toml`. Every dependency in a named extra that the full stack needs must also appear in `all` and (where the test suite needs it) `dev`. Known intentional duplication: `generate`'s `openai` + `jinja2` are deliberately in both `all` and `dev`. This has drifted before — check it, don't assume.

If anything fails: stop, report, fix or hand back. Never proceed to versioning on a red gate.

## 2. Pick the version

- Read `## [Unreleased]` in `CHANGELOG.md`. The project follows semver; breaking entries are bolded `**BREAKING — ...**`. Any breaking entry forces at least a **minor** bump pre-1.0.
- The version lives in **two** places and must be identical:
  - `pyproject.toml` → `[project] version`
  - `janus/__init__.py` → `__version__`
- Update both, then grep-verify they match:
  `grep -n 'version' pyproject.toml | head -3 && grep -n '__version__ =' janus/__init__.py`

## 3. Roll the CHANGELOG

- Retitle `## [Unreleased]` to `## [X.Y.Z] — YYYY-MM-DD` (today's date).
- Insert a fresh, empty `## [Unreleased]` section above it.

## 4. Commit

- `git add` the three files and commit: `Release vX.Y.Z`.

## 5. Build and inspect

- `uv build`
- Inspect the sdist and wheel file lists (`tar -tzf dist/*.tar.gz`, `unzip -l dist/*.whl`) for accidents: no `.env`, no `plans/`, no stray files. Check what `[tool.hatch.build]` in `pyproject.toml` actually includes rather than assuming.

## 6. Tag

- `git tag vX.Y.Z` and push the commit and the tag.

## 7. Publish — STOP FIRST

- Publishing to PyPI is the one **irreversible** step. STOP and ask the user for explicit confirmation before publishing. Never auto-publish just because the gate is green.
- Check what publish credentials/config actually exist (`~/.pypirc`, `UV_PUBLISH_TOKEN`, trusted publishing via CI) rather than assuming; use `uv publish` or `twine upload dist/*` accordingly.

## 8. Verify

- Confirm the new version is visible: `pip index versions janus-guard` (or the PyPI project page).
- Report: version published, last verified live-smoke date, anything skipped.
