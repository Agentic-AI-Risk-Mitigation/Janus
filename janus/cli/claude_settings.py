"""
Reading and merging Claude Code settings files.

``janus init`` is the first thing in this codebase that *writes* a file it did
not create — ``janus-hook backstop`` deliberately only prints, leaving the
operator to redirect. A settings file usually already holds configuration
someone else depends on, so every function here is built around one rule:
**never destroy what we did not write.**

That shapes three choices:

* **Idempotency by key, not by marker.** Re-running the wizard must update the
  Janus hook, not append a second one. Rather than stamping a marker key into
  the entry (which risks tripping an upstream schema check), a Janus entry is
  recognized by its command invoking ``janus-hook`` or ``-m janus.cli.hook``.
* **Union, never replace, on ``permissions.deny``.** Entries the operator
  added by hand outlive us. Relaxing a deny is a separate, explicit act —
  :func:`remove_permissions_deny` exists so the caller can *ask* first.
* **Backup, then atomic replace.** The previous file is copied aside before a
  write, and the new content lands via a temp file and ``os.replace`` so an
  interrupted write cannot leave a half-written settings file — which the CLI
  would reject wholesale, taking the user's unrelated configuration with it.

Settings files are strict JSON: no comments, no trailing commas. A file that
does not parse is an error the operator has to resolve, never something to
repair heuristically — guessing at the intent of a malformed config and
rewriting it is how a tool eats someone's configuration.
"""

from __future__ import annotations

import copy
import difflib
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from janus.exceptions import JanusError

#: Hook event the wizard wires. ``PostToolUse`` and the session seams exist in
#: the shim, but phase 1 holds no cross-call state for them to feed.
PRE_TOOL_USE = "PreToolUse"

#: Substrings that identify a hook command as ours. Both spellings the wizard
#: can emit — the console script and the ``python -m`` fallback used when
#: ``janus-hook`` is not on the PATH the CLI will use.
JANUS_COMMAND_MARKERS = ("janus-hook", "janus.cli.hook")

#: Hook-entry timeout, in seconds. Must sit above the shim's ``--deadline``
#: (default 5s) so the shim reaches its own limit first: the CLI's hook timeout
#: fails *open*, discarding a deny that arrives late, while the shim's deadline
#: fails closed.
DEFAULT_HOOK_TIMEOUT = 10


class SettingsError(JanusError):
    """Raised when a settings file cannot be read or is not a JSON object."""


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def load_settings(path: str | Path) -> dict[str, Any]:
    """Load a Claude settings file.

    A missing file is not an error — it is the common first-run case and means
    "no settings yet". Malformed JSON is: the caller must stop and let the
    operator fix it rather than overwrite a file we cannot read.
    """
    path = Path(path)
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SettingsError(f"Cannot read settings file '{path}': {exc}") from exc
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SettingsError(
            f"Settings file '{path}' is not valid JSON ({exc}). "
            "Claude settings files are strict JSON — no comments, no trailing "
            "commas. Fix the file and re-run; nothing has been written."
        ) from exc
    if not isinstance(data, dict):
        raise SettingsError(f"Settings file '{path}' must contain a JSON object.")
    return data


# ---------------------------------------------------------------------------
# Hook wiring
# ---------------------------------------------------------------------------


def _is_janus_hook(entry: Any) -> bool:
    if not isinstance(entry, dict) or entry.get("type") != "command":
        return False
    command = entry.get("command")
    return isinstance(command, str) and any(m in command for m in JANUS_COMMAND_MARKERS)


def find_janus_hooks(settings: dict[str, Any], event: str = PRE_TOOL_USE) -> list[tuple[int, int]]:
    """Locate existing Janus hook entries as ``(group_index, entry_index)``.

    Lets the caller report "an existing setup was found" and pre-seed its
    defaults from the command already on disk, before anything is modified.
    """
    found: list[tuple[int, int]] = []
    groups = settings.get("hooks", {})
    if not isinstance(groups, dict):
        return found
    matchers = groups.get(event)
    if not isinstance(matchers, list):
        return found
    for group_index, group in enumerate(matchers):
        if not isinstance(group, dict):
            continue
        entries = group.get("hooks")
        if not isinstance(entries, list):
            continue
        for entry_index, entry in enumerate(entries):
            if _is_janus_hook(entry):
                found.append((group_index, entry_index))
    return found


def janus_hook_commands(settings: dict[str, Any], event: str = PRE_TOOL_USE) -> list[str]:
    """The command strings of the Janus hook entries currently wired."""
    matchers = settings.get("hooks", {}).get(event, [])
    return [matchers[g]["hooks"][e]["command"] for g, e in find_janus_hooks(settings, event)]


def upsert_janus_hook(
    settings: dict[str, Any],
    *,
    command: str,
    timeout: int = DEFAULT_HOOK_TIMEOUT,
    event: str = PRE_TOOL_USE,
) -> dict[str, Any]:
    """Return a copy of ``settings`` with exactly one Janus hook entry wired.

    Idempotent: running the wizard twice updates the entry rather than stacking
    a second one. Existing non-Janus hooks keep their content and their order —
    hook order is observable behavior for whoever configured them.

    Duplicate Janus entries (from hand-editing, or an older wizard run whose
    write was interrupted) collapse to the first: two enforcement hooks on one
    seam means every call gets decided twice, and the stricter of two
    disagreeing answers wins in a way nobody configured on purpose.
    """
    updated = copy.deepcopy(settings)
    hooks = updated.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise SettingsError("'hooks' in settings must be a JSON object.")
    matchers = hooks.setdefault(event, [])
    if not isinstance(matchers, list):
        raise SettingsError(f"'hooks.{event}' in settings must be a JSON array.")

    existing = find_janus_hooks(updated, event)
    if not existing:
        matchers.append({"hooks": [{"type": "command", "command": command, "timeout": timeout}]})
        return updated

    keep_group, keep_entry = existing[0]
    matchers[keep_group]["hooks"][keep_entry] = {
        "type": "command",
        "command": command,
        "timeout": timeout,
    }

    # Drop the rest, deepest index first so earlier positions stay valid.
    for group_index, entry_index in sorted(existing[1:], reverse=True):
        del matchers[group_index]["hooks"][entry_index]

    # A matcher group whose only entry was a duplicate is now empty scaffolding.
    hooks[event] = [
        group for index, group in enumerate(matchers) if index == keep_group or group.get("hooks")
    ]
    return updated


# ---------------------------------------------------------------------------
# permissions.deny backstop
# ---------------------------------------------------------------------------


def _deny_list(settings: dict[str, Any]) -> list[Any]:
    permissions = settings.get("permissions")
    if not isinstance(permissions, dict):
        return []
    deny = permissions.get("deny")
    return deny if isinstance(deny, list) else []


def merge_permissions_deny(
    settings: dict[str, Any], entries: list[str] | tuple[str, ...]
) -> dict[str, Any]:
    """Union ``entries`` into ``permissions.deny``, preserving existing order.

    The backstop is the only layer that holds when no hook runs at all, so it
    is additive by construction: entries the operator added stay, ours are
    appended if absent, and nothing is reordered or removed.
    """
    updated = copy.deepcopy(settings)
    permissions = updated.setdefault("permissions", {})
    if not isinstance(permissions, dict):
        raise SettingsError("'permissions' in settings must be a JSON object.")
    deny = permissions.setdefault("deny", [])
    if not isinstance(deny, list):
        raise SettingsError("'permissions.deny' in settings must be a JSON array.")

    for entry in entries:
        if entry not in deny:
            deny.append(entry)
    return updated


def remove_permissions_deny(
    settings: dict[str, Any], entries: list[str] | tuple[str, ...]
) -> dict[str, Any]:
    """Remove ``entries`` from ``permissions.deny``.

    Separate from the merge on purpose: relaxing a deny is a decision, not a
    side effect of re-running the wizard with different answers. Callers ask
    before calling this.
    """
    updated = copy.deepcopy(settings)
    deny = _deny_list(updated)
    if not deny:
        return updated
    updated["permissions"]["deny"] = [e for e in deny if e not in entries]
    return updated


def missing_deny_entries(
    settings: dict[str, Any], entries: list[str] | tuple[str, ...]
) -> list[str]:
    """Which of ``entries`` are not yet in ``permissions.deny``."""
    deny = _deny_list(settings)
    return [e for e in entries if e not in deny]


# ---------------------------------------------------------------------------
# Diffing and writing
# ---------------------------------------------------------------------------


def _render(data: dict[str, Any]) -> list[str]:
    return json.dumps(data, indent=2, ensure_ascii=False).splitlines()


def settings_diff(before: dict[str, Any], after: dict[str, Any], label: str) -> str:
    """A unified diff of two settings dicts, or '' when they are identical.

    The wizard shows this before writing. Someone handing their agent's
    permissions to a tool deserves to see the exact edit first.
    """
    if before == after:
        return ""
    return "\n".join(
        difflib.unified_diff(
            _render(before),
            _render(after),
            fromfile=f"{label} (current)",
            tofile=f"{label} (after janus init)",
            lineterm="",
        )
    )


def backup_path_for(path: Path, *, now: datetime | None = None) -> Path:
    """Timestamped sibling backup path, e.g. ``settings.json.bak-20260825T142230``."""
    stamp = (now or datetime.now()).strftime("%Y%m%dT%H%M%S")
    return path.with_name(f"{path.name}.bak-{stamp}")


def write_settings(path: str | Path, data: dict[str, Any], *, backup: bool = True) -> Path | None:
    """Write a settings file atomically, backing up any previous content.

    Returns the backup path, or ``None`` when there was no existing file.

    The write goes to a temp file in the same directory and then ``os.replace``
    — atomic on Windows as well as POSIX — so a crash mid-write leaves the old
    file intact rather than a truncated one the CLI would refuse to parse.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    backup_path: Path | None = None
    if backup and path.exists():
        backup_path = backup_path_for(path)
        shutil.copy2(path, backup_path)

    payload = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    handle, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
        os.replace(temp_name, path)
    except BaseException:
        # Leave no debris behind if the replace never happened.
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return backup_path
