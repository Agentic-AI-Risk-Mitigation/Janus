"""
The GitHub tools expressed as Janus ``ToolDef``s.

The implementations live in :mod:`demos.demo_2.github_api`, which imports no
Janus at all. This module only adds the Janus-side description of them, so the
guarded and unguarded agents are demonstrably running the *same* functions and
differ only in whether a policy is consulted first.

Three tools, deliberately shaped so the demo has something to enforce:

- ``fetch_github_issue``   — read an issue's title/body/labels. Allowed, but the
                             policy pins *which* repository it may read from.
- ``fetch_issue_comments`` — read the discussion thread. Same scoping.
- ``post_issue_comment``   — a write sink. Left out of the policy on purpose, so
                             Janus's default-deny is what stops it.
"""

from __future__ import annotations

from demos.demo_2.github_api import (
    FETCH_COMMENTS_DESCRIPTION,
    FETCH_ISSUE_DESCRIPTION,
    POST_COMMENT_DESCRIPTION,
    fetch_github_issue,
    fetch_issue_comments,
    post_issue_comment,
)
from janus import ToolDef, ToolParam

_OWNER = ToolParam("owner", "string", "Repository owner or organization, e.g. 'python'.")
_REPO = ToolParam("repo", "string", "Repository name, e.g. 'cpython'.")
_NUMBER = ToolParam("issue_number", "integer", "The issue number to read.")

GITHUB_TOOLS: list[ToolDef] = [
    ToolDef(
        name="fetch_github_issue",
        description=FETCH_ISSUE_DESCRIPTION,
        params=[_OWNER, _REPO, _NUMBER],
        handler=fetch_github_issue,
    ),
    ToolDef(
        name="fetch_issue_comments",
        description=FETCH_COMMENTS_DESCRIPTION,
        params=[_OWNER, _REPO, _NUMBER],
        handler=fetch_issue_comments,
    ),
    ToolDef(
        name="post_issue_comment",
        description=POST_COMMENT_DESCRIPTION,
        params=[
            _OWNER,
            _REPO,
            _NUMBER,
            ToolParam("body", "string", "Markdown body of the comment to post."),
        ],
        handler=post_issue_comment,
    ),
]
