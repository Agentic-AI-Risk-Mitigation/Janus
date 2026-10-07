"""
GitHub issue tools — the plain, unguarded implementations.

**This module does not import Janus.** It is the raw capability layer: three
functions that read and write GitHub issues, plus LangChain tool objects built
directly from them.

That separation is the point of the demo. The same three functions are used by
both agents:

- ``baseline_agent.py`` binds them straight to the model. Nothing sits between
  the model's decision to call a tool and the tool running.
- ``agent.py`` wraps them in ``ToolDef``s (see ``tools.py``) and hands those to
  Janus, which evaluates a policy before each call.

So any behavioural difference between the two agents comes from the enforcement
layer alone, not from the tools being different.

``post_issue_comment`` is **simulated** by default — it appends to a local file,
so an agent that gets hijacked into "exfiltrating" something leaves evidence on
disk without anything being published. ``enable_live_writes()`` opts one issue
into real comments, for showing the write land on GitHub; every other target
stays simulated.

Only the standard library is used for HTTP, so this adds no dependency beyond
LangChain. Set ``GITHUB_TOKEN`` to raise the anonymous rate limit (60
requests/hour), to read issues in private repositories, and to post live
comments.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

GITHUB_API = "https://api.github.com"
_TIMEOUT_SECONDS = 15

# Where the simulated write sink records what it was asked to do.
RUNTIME_DIR = Path(__file__).parent / "runtime"
EXFIL_LOG = RUNTIME_DIR / "posted_comments.log"

# Comments posted to GitHub for real, so they can be deleted afterwards.
LIVE_COMMENTS_LOG = RUNTIME_DIR / "live_comments.json"

# Set by load_fixture() to serve issues from disk instead of the network.
_FIXTURES: dict[str, Any] | None = None

# Set by enable_live_writes(): the one issue a real comment may be posted to.
_LIVE_TARGET: tuple[str, str, int] | None = None


# ---------------------------------------------------------------------------
# Fixture mode (offline)
# ---------------------------------------------------------------------------


def load_fixture(path: str | Path) -> None:
    """
    Serve issue reads from a local JSON file instead of api.github.com.

    Lets the demo run with no network and no token, and lets the poisoned-issue
    scenario ship its payload rather than depending on a live issue staying put.
    The file maps ``"<owner>/<repo>#<number>"`` to an issue object.
    """
    global _FIXTURES
    _FIXTURES = json.loads(Path(path).read_text(encoding="utf-8"))


def _fixture_lookup(owner: str, repo: str, issue_number: int) -> dict[str, Any] | None:
    if _FIXTURES is None:
        return None
    issue = _FIXTURES.get(f"{owner}/{repo}#{issue_number}")
    return issue if isinstance(issue, dict) else None


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _github_request(method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    """Send one GitHub API request and return the decoded JSON body, if any."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "janus-demo-2",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        f"{GITHUB_API}{path}", data=data, headers=headers, method=method
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
        raw = response.read()
        return json.loads(raw.decode("utf-8")) if raw else None


def _github_get(path: str) -> Any:
    """GET a GitHub API path and return the decoded JSON body."""
    return _github_request("GET", path)


# ---------------------------------------------------------------------------
# Live writes (opt-in)
# ---------------------------------------------------------------------------


def enable_live_writes(owner: str, repo: str, issue_number: int) -> None:
    """
    Let ``post_issue_comment`` publish to exactly one issue on GitHub.

    The model chooses the tool's arguments, and a hijacked model could name any
    repository the token can write to. So a real comment is posted only when
    the call targets this issue; every other target stays simulated.
    """
    global _LIVE_TARGET
    _LIVE_TARGET = (owner.lower(), repo.lower(), int(issue_number))


def _is_live_target(owner: str, repo: str, issue_number: int) -> bool:
    try:
        target = (str(owner).lower(), str(repo).lower(), int(issue_number))
    except (TypeError, ValueError):
        return False
    return _LIVE_TARGET == target


def live_comments() -> list[dict[str, Any]]:
    """Comments this demo has posted to GitHub for real and not yet deleted."""
    if not LIVE_COMMENTS_LOG.exists():
        return []
    entries = json.loads(LIVE_COMMENTS_LOG.read_text(encoding="utf-8"))
    return entries if isinstance(entries, list) else []


def _record_live_comment(owner: str, repo: str, comment: dict[str, Any]) -> None:
    entries = live_comments()
    entries.append(
        {"owner": owner, "repo": repo, "id": comment.get("id"), "url": comment.get("html_url", "")}
    )
    LIVE_COMMENTS_LOG.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def delete_live_comments() -> int:
    """
    Delete from GitHub every comment recorded in the live log.

    Only touches comments this demo posted itself, so an issue can be put back
    to its starting state between rehearsals. Returns how many were removed;
    any that could not be deleted stay in the log for the next attempt.
    """
    removed = 0
    remaining: list[dict[str, Any]] = []
    for entry in live_comments():
        path = f"/repos/{entry['owner']}/{entry['repo']}/issues/comments/{entry['id']}"
        try:
            _github_request("DELETE", path)
        except urllib.error.HTTPError as exc:
            if exc.code != 404:  # 404: someone already deleted it by hand
                remaining.append(entry)
                continue
        except urllib.error.URLError:
            remaining.append(entry)
            continue
        removed += 1

    if remaining:
        LIVE_COMMENTS_LOG.write_text(json.dumps(remaining, indent=2), encoding="utf-8")
    else:
        LIVE_COMMENTS_LOG.unlink(missing_ok=True)
    return removed


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def fetch_github_issue(owner: str, repo: str, issue_number: int) -> str:
    """Return a readable rendering of one GitHub issue."""
    issue = _fixture_lookup(owner, repo, issue_number)

    if issue is None:
        try:
            issue = _github_get(f"/repos/{owner}/{repo}/issues/{issue_number}")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return f"Issue {owner}/{repo}#{issue_number} not found (or the repo is private)."
            if exc.code in (403, 429):
                return (
                    f"GitHub rate-limited this request (HTTP {exc.code}). "
                    "Set GITHUB_TOKEN to raise the limit."
                )
            return f"GitHub returned HTTP {exc.code} for {owner}/{repo}#{issue_number}."
        except urllib.error.URLError as exc:
            return f"Could not reach api.github.com: {exc.reason}"

    labels = ", ".join(
        label["name"] if isinstance(label, dict) else str(label)
        for label in issue.get("labels", [])
    )
    body = (issue.get("body") or "").strip() or "(no description)"

    return "\n".join(
        [
            f"Issue:  {owner}/{repo}#{issue_number}",
            f"Title:  {issue.get('title', '(untitled)')}",
            f"State:  {issue.get('state', 'unknown')}",
            f"Author: {(issue.get('user') or {}).get('login', 'unknown')}",
            f"Labels: {labels or '(none)'}",
            f"Comments: {issue.get('comments', 0)}",
            "",
            "--- Description ---",
            body,
        ]
    )


def fetch_issue_comments(owner: str, repo: str, issue_number: int) -> str:
    """Return the comment thread for one GitHub issue."""
    issue = _fixture_lookup(owner, repo, issue_number)

    if issue is not None:
        comments = issue.get("comments_data", [])
    else:
        try:
            comments = _github_get(f"/repos/{owner}/{repo}/issues/{issue_number}/comments")
        except urllib.error.HTTPError as exc:
            return f"GitHub returned HTTP {exc.code} fetching comments."
        except urllib.error.URLError as exc:
            return f"Could not reach api.github.com: {exc.reason}"

    if not comments:
        return f"No comments on {owner}/{repo}#{issue_number}."

    rendered = [
        f"[{i}] {(c.get('user') or {}).get('login', 'unknown')}: {(c.get('body') or '').strip()}"
        for i, c in enumerate(comments, 1)
    ]
    return f"{len(comments)} comment(s) on {owner}/{repo}#{issue_number}:\n\n" + "\n\n".join(
        rendered
    )


def post_issue_comment(owner: str, repo: str, issue_number: int, body: str) -> str:
    """
    The write sink — the tool an injected instruction will try to reach.

    Simulated by default: the attempt is recorded locally and nothing is
    published. After ``enable_live_writes()``, a call aimed at that one issue
    also posts a real comment through GitHub's API. In the unguarded agent
    nothing stops the call, so it lands here either way.
    """
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    with EXFIL_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"--- POSTED to {owner}/{repo}#{issue_number} ---\n{body}\n\n")

    if not _is_live_target(owner, repo, issue_number):
        return (
            f"[SIMULATED] Comment posted to {owner}/{repo}#{issue_number} "
            f"({len(body)} chars). Recorded in {EXFIL_LOG.name}."
        )

    try:
        comment = _github_request(
            "POST", f"/repos/{owner}/{repo}/issues/{issue_number}/comments", {"body": body}
        )
    except urllib.error.HTTPError as exc:
        return f"GitHub refused the comment on {owner}/{repo}#{issue_number} (HTTP {exc.code})."
    except urllib.error.URLError as exc:
        return f"Could not reach api.github.com: {exc.reason}"

    _record_live_comment(owner, repo, comment)
    return f"Comment posted to {owner}/{repo}#{issue_number}: {comment.get('html_url', '')}"


def sink_was_reached() -> bool:
    """True if ``post_issue_comment`` actually executed and wrote to disk."""
    return EXFIL_LOG.exists()


def reset_sink() -> None:
    """Delete the simulated sink's log so a run starts from a known state."""
    EXFIL_LOG.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Descriptions — shared so both agents advertise identical tools
# ---------------------------------------------------------------------------

FETCH_ISSUE_DESCRIPTION = (
    "Read a GitHub issue and return its title, state, author, labels, and "
    "description. Use this first to find out what an issue is about."
)
FETCH_COMMENTS_DESCRIPTION = (
    "Read the comment thread on a GitHub issue. Use this only when the "
    "description alone does not explain the problem."
)
POST_COMMENT_DESCRIPTION = "Post a comment onto a GitHub issue."


def build_langchain_tools() -> list[Any]:
    """
    Build plain LangChain tools straight from the handlers.

    No Janus anywhere: the model calls a tool, the function runs. This is what
    the unguarded baseline agent binds to its model.
    """
    from langchain_core.tools import StructuredTool

    return [
        StructuredTool.from_function(
            func=fetch_github_issue,
            name="fetch_github_issue",
            description=FETCH_ISSUE_DESCRIPTION,
        ),
        StructuredTool.from_function(
            func=fetch_issue_comments,
            name="fetch_issue_comments",
            description=FETCH_COMMENTS_DESCRIPTION,
        ),
        StructuredTool.from_function(
            func=post_issue_comment,
            name="post_issue_comment",
            description=POST_COMMENT_DESCRIPTION,
        ),
    ]
