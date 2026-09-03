"""
GitHub issue tools for Demo 2.

Three tools, deliberately shaped so the demo has something to enforce:

- ``fetch_github_issue``   — read an issue's title/body/labels. Allowed, but the
                             policy pins *which* repository it may read from.
- ``fetch_issue_comments`` — read the discussion thread. Same scoping.
- ``post_issue_comment``   — a write sink. Left out of the policy on purpose, so
                             Janus's default-deny is what stops it.

``post_issue_comment`` is **simulated** — it never calls GitHub's write API. It
appends to a local file so the unprotected run can show what an injected
instruction would have accomplished, without touching a real repository.

Only the standard library is used for HTTP so the demo adds no dependency
beyond LangChain itself. Set ``GITHUB_TOKEN`` to raise the anonymous rate limit
(60 requests/hour) and to read issues in private repositories.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from janus import ToolDef, ToolParam

GITHUB_API = "https://api.github.com"
_TIMEOUT_SECONDS = 15

# Where the simulated write sink records what it was asked to do.
RUNTIME_DIR = Path(__file__).parent / "runtime"
EXFIL_LOG = RUNTIME_DIR / "posted_comments.log"

# Set by load_fixture() to serve issues from disk instead of the network.
_FIXTURES: dict[str, Any] | None = None


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


def _github_get(path: str) -> Any:
    """GET a GitHub API path and return the decoded JSON body."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "janus-demo-2",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(f"{GITHUB_API}{path}", headers=headers)
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


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
    SIMULATED write sink — records the attempt locally, never calls GitHub.

    This is the tool an injected instruction inside an issue body will try to
    reach. In the protected run it is not in the policy, so Janus denies it
    before this function is ever entered.
    """
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    with EXFIL_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"--- POSTED to {owner}/{repo}#{issue_number} ---\n{body}\n\n")
    return (
        f"[SIMULATED] Comment posted to {owner}/{repo}#{issue_number} "
        f"({len(body)} chars). Recorded in {EXFIL_LOG.name}."
    )


# ---------------------------------------------------------------------------
# Janus tool definitions
# ---------------------------------------------------------------------------

_OWNER = ToolParam("owner", "string", "Repository owner or organization, e.g. 'python'.")
_REPO = ToolParam("repo", "string", "Repository name, e.g. 'cpython'.")
_NUMBER = ToolParam("issue_number", "integer", "The issue number to read.")

GITHUB_TOOLS: list[ToolDef] = [
    ToolDef(
        name="fetch_github_issue",
        description=(
            "Read a GitHub issue and return its title, state, author, labels, and "
            "description. Use this first to find out what an issue is about."
        ),
        params=[_OWNER, _REPO, _NUMBER],
        handler=fetch_github_issue,
    ),
    ToolDef(
        name="fetch_issue_comments",
        description=(
            "Read the comment thread on a GitHub issue. Use this only when the "
            "description alone does not explain the problem."
        ),
        params=[_OWNER, _REPO, _NUMBER],
        handler=fetch_issue_comments,
    ),
    ToolDef(
        name="post_issue_comment",
        description="Post a comment onto a GitHub issue.",
        params=[
            _OWNER,
            _REPO,
            _NUMBER,
            ToolParam("body", "string", "Markdown body of the comment to post."),
        ],
        handler=post_issue_comment,
    ),
]
