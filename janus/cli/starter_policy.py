"""
Starter-policy construction for ``janus init``.

The wizard needs a policy it can *vary* — extra secret paths, a network
posture, a git-push stance — so the starter lives here as a builder over rule
fragments rather than as a static file to string-edit.
``examples/claude_code/policy.starter.json`` stays the documented copy-paste
artifact; a parity test pins it to :func:`build_starter_policy` defaults so the
two cannot drift.

Three invariants every rule here preserves, each load-bearing:

* **Full form.** Every rule carries all four keys. ``parse_policy`` reads a
  bare ``{arg: schema}`` dict as shorthand *unless* it contains one of
  ``priority``/``effect``/``conditions``/``fallback``, so a tool with an
  argument named ``priority`` would otherwise parse as an unconditional allow.
  Emitting full form sidesteps that heuristic entirely.
* **Deny first, then an unconditional allow.** For a tool the policy lists,
  "no rule matched" is a deny. A guarded-but-usable tool is therefore deny@1
  plus allow@10; omit the trailing allow and the tool is dead.
* **Enumerate the harmless tools.** Gate mode promotes to strict default-deny
  under ``bypassPermissions``, where an unlisted tool is denied outright rather
  than deferred to a prompt. The bare allow entries keep those sessions working.

Two properties of the patterns themselves, both learned the hard way:

* **Conditions are ``re.search``, not full matches**, so anchor deliberately —
  ``(^|SEP)\\.env`` rather than a bare ``\\.env``, ``\\.pem$`` rather than
  ``\\.pem``. (The anchoring bounds *where* a match may start; it does not make
  the match exact. ``(^|SEP)\\.env…[^SEP]*$`` still covers ``.environment``,
  which is the safe direction to be wrong in.)
* **Separators must be a class, never a slash.** See :data:`SEP`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

# ---------------------------------------------------------------------------
# Pattern fragments
# ---------------------------------------------------------------------------

#: Path separator, either flavour.
#:
#: Not cosmetic, and not theoretical: Claude Code reports ``file_path`` using
#: the host's native separator, so on Windows a policy anchored on ``/`` alone
#: silently matches nothing. Verified against a live 2.1.246 session, which
#: sent ``C:\Users\...\README.md`` — under a ``/``-only pattern every one of the
#: secret-read denies below, and the guard-tamper deny, allowed the call. Any
#: new path pattern must use this class rather than a bare slash.
SEP = r"[/\\]"

#: Non-separator character, for "rest of the final path segment" tails.
NOT_SEP = r"[^/\\]"

#: Files a coding agent has no business reading. ``.env.example`` is exempted
#: by negative lookahead — it is checked into most repos on purpose.
SECRET_READ_PATTERN = (
    rf"(^|{SEP})\.env(?!\.example){NOT_SEP}*$"
    rf"|{SEP}\.ssh{SEP}"
    rf"|{SEP}\.aws{SEP}credentials"
    r"|\.pem$"
    rf"|{SEP}\.claude{SEP}\.credentials\.json$"
)

#: Pipe-to-shell downloads and direct reads of credential material. The
#: ``[^|;&]*`` between the fetch and the pipe keeps the alternation from
#: spanning an unrelated later command in a compound line.
BASH_EXFIL_PATTERN = (
    r"(curl|wget)[^|;&]*\|\s*(ba|z|fi)?sh\b"
    rf"|{SEP}\.ssh{SEP}id_"
    rf"|{SEP}\.aws{SEP}credentials"
    rf"|{SEP}\.claude{SEP}\.credentials"
)

#: Writes that would disable the guard itself.
GUARD_TAMPER_PATTERN = rf"{SEP}\.claude{SEP}settings(\.local)?\.json$|{SEP}\.claude{SEP}janus{SEP}"

#: Network clients, anchored at command position so ``foo --curl`` and a path
#: containing "nc" do not match. Mirrors the ``permissions.deny`` backstop's
#: ``Bash(curl:*)`` family at the policy layer.
NETWORK_COMMAND_PATTERN = r"(^|[;&|]\s*)(curl|wget|ssh|scp|nc|telnet)\b"

#: ``git push`` at command position.
GIT_PUSH_PATTERN = r"(^|[;&|]\s*)git\s+push\b"

#: Tools guarded by a deny rule plus a trailing allow.
FILE_TAMPER_TOOLS: tuple[str, ...] = ("Write", "Edit", "MultiEdit")

#: Built-ins that get a bare unconditional allow. Present so
#: ``bypassPermissions`` sessions — where gate mode promotes to strict
#: default-deny — keep working. Extend when the CLI grows a tool; an omission
#: shows up as ``Tool 'X' is not listed in the policy`` in those sessions only.
PLAIN_ALLOW_TOOLS: tuple[str, ...] = (
    "Glob",
    "Grep",
    "LS",
    "WebFetch",
    "WebSearch",
    "Task",
    "Agent",
    "Skill",
    "SlashCommand",
    "TodoWrite",
    "TodoRead",
    "NotebookEdit",
    "NotebookRead",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "BashOutput",
    "KillShell",
    "ListMcpResources",
    "ReadMcpResource",
)

DENY_PRIORITY = 1
ALLOW_PRIORITY = 10

_EFFECT_ALLOW = 0
_EFFECT_DENY = 1
_FALLBACK_RAISE = 0


# ---------------------------------------------------------------------------
# Rule constructors
# ---------------------------------------------------------------------------


def deny_rule(conditions: dict[str, Any], *, priority: int = DENY_PRIORITY) -> dict[str, Any]:
    """A deny rule in full form. Empty ``conditions`` denies unconditionally."""
    return {
        "priority": priority,
        "effect": _EFFECT_DENY,
        "conditions": conditions,
        "fallback": _FALLBACK_RAISE,
    }


def allow_all_rule(*, priority: int = ALLOW_PRIORITY) -> dict[str, Any]:
    """The trailing unconditional allow that makes a guarded tool usable."""
    return {
        "priority": priority,
        "effect": _EFFECT_ALLOW,
        "conditions": {},
        "fallback": _FALLBACK_RAISE,
    }


def _string_match(pattern: str) -> dict[str, Any]:
    return {"type": "string", "pattern": pattern}


def _alternation(*parts: str) -> str:
    """Join non-empty regex fragments into one alternation."""
    return "|".join(p for p in parts if p)


def pattern_for_entry(entry: str) -> str:
    """Turn one user-typed path or glob into a regex fragment.

    Users type literals (``secrets/``, ``config/prod.yaml``) and the occasional
    suffix glob (``*.key``). Everything is escaped — a stray ``.`` or ``(`` in a
    filename must not become a metacharacter — with ``*.ext`` translated to an
    anchored suffix match, since that is the one glob people reach for.

    Separators are normalized to :data:`SEP` so an entry typed one way matches a
    path reported the other. Someone who types ``secrets/`` on Windows means the
    directory, not the slash.
    """
    entry = entry.strip()
    if not entry:
        return ""
    if entry.startswith("*.") and len(entry) > 2:
        return re.escape(entry[1:]) + "$"
    # Normalize first so both flavours collapse to one placeholder, then splice
    # the separator class in after escaping (escaping would mangle the class).
    normalized = entry.replace("\\", "/")
    return SEP.join(re.escape(part) for part in normalized.split("/"))


def _entry_patterns(entries: Sequence[str]) -> list[str]:
    return [p for p in (pattern_for_entry(e) for e in entries) if p]


# ---------------------------------------------------------------------------
# Policy builder
# ---------------------------------------------------------------------------


def build_starter_policy(
    *,
    extra_secret_patterns: Sequence[str] = (),
    extra_bash_deny_patterns: Sequence[str] = (),
    deny_network_commands: bool = False,
    deny_git_push: bool = False,
    deny_webfetch: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Build a Claude Code starter policy in the user-facing JSON format.

    Called with no arguments this reproduces
    ``examples/claude_code/policy.starter.json`` exactly (pinned by test).

    Args:
        extra_secret_patterns: Additional paths/globs to deny on ``Read``.
        extra_bash_deny_patterns: Additional fragments for the ``Bash`` deny.
        deny_network_commands: Also deny curl/wget/ssh/scp/nc at the policy
            layer, not only via the ``permissions.deny`` backstop.
        deny_git_push: Deny ``git push``.
        deny_webfetch: Make ``WebFetch`` a deny rather than a bare allow. The
            tool stays listed so bypass sessions report a policy deny instead
            of "not listed in the policy".

    Returns:
        ``{tool_name: [rule, ...]}`` with every rule in full form, ready for
        ``parse_policy`` → ``save_policy``.
    """
    extra_read = _entry_patterns(extra_secret_patterns)
    extra_bash = _entry_patterns(extra_bash_deny_patterns)

    read_pattern = _alternation(SECRET_READ_PATTERN, *extra_read)
    bash_pattern = _alternation(
        BASH_EXFIL_PATTERN,
        NETWORK_COMMAND_PATTERN if deny_network_commands else "",
        GIT_PUSH_PATTERN if deny_git_push else "",
        *extra_bash,
    )

    policy: dict[str, list[dict[str, Any]]] = {
        "Read": [
            deny_rule({"file_path": _string_match(read_pattern)}),
            allow_all_rule(),
        ],
        "Bash": [
            deny_rule({"command": _string_match(bash_pattern)}),
            allow_all_rule(),
        ],
    }

    for tool in FILE_TAMPER_TOOLS:
        policy[tool] = [
            deny_rule({"file_path": _string_match(GUARD_TAMPER_PATTERN)}),
            allow_all_rule(),
        ]

    for tool in PLAIN_ALLOW_TOOLS:
        if tool == "WebFetch" and deny_webfetch:
            policy[tool] = [deny_rule({})]
        else:
            policy[tool] = [allow_all_rule()]

    return policy


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

#: Claude Code's built-in tools and their arguments.
#:
#: Claude Code tools are not Janus ``ToolDef``s — they live in the CLI, not in
#: a registry we can introspect — so the wizard carries a static table. It has
#: two consumers: ``validate_policy_structure`` (which flags a condition naming
#: an argument the tool does not have, the most common authoring typo) and the
#: optional LLM branch (which needs ``{name, description, args}`` to draft
#: rules). Argument lists cover what a policy would plausibly condition on, not
#: every optional field the CLI accepts.
CLAUDE_CODE_TOOL_DEFS: list[dict[str, Any]] = [
    {
        "name": "Read",
        "description": "Read a file from the local filesystem.",
        "args": {
            "file_path": {"type": "string"},
            "offset": {"type": "integer"},
            "limit": {"type": "integer"},
        },
    },
    {
        "name": "Write",
        "description": "Write a file to the local filesystem, overwriting if it exists.",
        "args": {"file_path": {"type": "string"}, "content": {"type": "string"}},
    },
    {
        "name": "Edit",
        "description": "Perform an exact string replacement in a file.",
        "args": {
            "file_path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean"},
        },
    },
    {
        "name": "MultiEdit",
        "description": "Apply several edits to a single file in one call.",
        "args": {"file_path": {"type": "string"}, "edits": {"type": "array"}},
    },
    {
        "name": "Bash",
        "description": "Execute a shell command.",
        "args": {
            "command": {"type": "string"},
            "description": {"type": "string"},
            "timeout": {"type": "integer"},
            "run_in_background": {"type": "boolean"},
        },
    },
    {
        "name": "Glob",
        "description": "Match file paths against a glob pattern.",
        "args": {"pattern": {"type": "string"}, "path": {"type": "string"}},
    },
    {
        "name": "Grep",
        "description": "Search file contents with a regular expression.",
        "args": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "glob": {"type": "string"},
            "type": {"type": "string"},
            "output_mode": {"type": "string"},
        },
    },
    {
        "name": "LS",
        "description": "List files and directories at a path.",
        "args": {"path": {"type": "string"}, "ignore": {"type": "array"}},
    },
    {
        "name": "WebFetch",
        "description": "Fetch a URL and process its content with a model.",
        "args": {"url": {"type": "string"}, "prompt": {"type": "string"}},
    },
    {
        "name": "WebSearch",
        "description": "Search the web and return results.",
        "args": {
            "query": {"type": "string"},
            "allowed_domains": {"type": "array"},
            "blocked_domains": {"type": "array"},
        },
    },
    {
        "name": "Task",
        "description": "Launch a subagent to handle a multi-step task.",
        "args": {
            "description": {"type": "string"},
            "prompt": {"type": "string"},
            "subagent_type": {"type": "string"},
        },
    },
    {
        "name": "Agent",
        "description": "Launch a subagent (alias of Task on some CLI versions).",
        "args": {
            "description": {"type": "string"},
            "prompt": {"type": "string"},
            "subagent_type": {"type": "string"},
        },
    },
    {
        "name": "Skill",
        "description": "Invoke a packaged skill by name.",
        "args": {"skill": {"type": "string"}, "args": {"type": "string"}},
    },
    {
        "name": "SlashCommand",
        "description": "Run a slash command.",
        "args": {"command": {"type": "string"}},
    },
    {
        "name": "TodoWrite",
        "description": "Write the session todo list.",
        "args": {"todos": {"type": "array"}},
    },
    {
        "name": "TodoRead",
        "description": "Read the session todo list.",
        "args": {},
    },
    {
        "name": "NotebookEdit",
        "description": "Edit a cell in a Jupyter notebook.",
        "args": {
            "notebook_path": {"type": "string"},
            "cell_id": {"type": "string"},
            "new_source": {"type": "string"},
            "edit_mode": {"type": "string"},
        },
    },
    {
        "name": "NotebookRead",
        "description": "Read a Jupyter notebook's cells and outputs.",
        "args": {"notebook_path": {"type": "string"}},
    },
    {
        "name": "AskUserQuestion",
        "description": "Ask the user a multiple-choice question.",
        "args": {"questions": {"type": "array"}},
    },
    {
        "name": "EnterPlanMode",
        "description": "Enter plan mode.",
        "args": {},
    },
    {
        "name": "ExitPlanMode",
        "description": "Exit plan mode and request approval of a plan.",
        "args": {"plan": {"type": "string"}},
    },
    {
        "name": "BashOutput",
        "description": "Read output from a background shell.",
        "args": {"bash_id": {"type": "string"}, "filter": {"type": "string"}},
    },
    {
        "name": "KillShell",
        "description": "Terminate a background shell.",
        "args": {"shell_id": {"type": "string"}},
    },
    {
        "name": "ListMcpResources",
        "description": "List resources exposed by connected MCP servers.",
        "args": {"server": {"type": "string"}},
    },
    {
        "name": "ReadMcpResource",
        "description": "Read one resource from a connected MCP server.",
        "args": {"server": {"type": "string"}, "uri": {"type": "string"}},
    },
]
