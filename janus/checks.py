"""
Output-side checks: deterministic, provenance-aware assertions over model output.

Action mediation gives no protection when the harm rides in the output *text*
rather than a tool call — and tool-free structured calls never touch a hook at
all. This module is Janus's honest answer to that gap: not an injection
classifier (probabilistic detection is explicitly out of scope), but a small
library of **deterministic value-provenance assertions** sharing the
session's value-sets:

- :func:`echoed_untrusted_values` — flags values that appeared in
  prompt-borne untrusted input (seeded via ``Session.mark_untrusted``) and
  rode into the output. Generalizes the hand-rolled
  "inbound URL echoed into the drafted reply" backstop.
- :func:`values_grounded_in` — the positive twin: every extracted value in
  the output must be a member of an allowed provenance set ("every URL in
  the draft must be one research actually surfaced").

Checks return :class:`Finding` records rather than raising by default,
because the common consumer shape is draft-for-human-review — surface the
finding, let the human decide. Pipelines that act on output mechanically
(the ``structured()``-then-act shape) pass ``enforce=True`` and catch
:class:`janus.exceptions.OutputViolation`::

    session = Session()
    session.mark_untrusted(inbound, label="inbound_email", extract=extract_urls)
    result = call_the_model(...)                      # tool-free, no hooks
    check_output(result, session, checks=[
        echoed_untrusted_values("inbound_email", extract=extract_urls),
    ], enforce=True)                                  # raises before you act

Findings are appended to the session's audit trail, so input marking, tool
provenance, gate denials, endorsements, and output findings read as one
ordered story.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from janus.exceptions import OutputViolation
from janus.policy.session import Session, untrusted_set

# A check inspects the (raw) output plus the session and reports one Finding
# or None. Write your own freely — the factories below are just the built-ins.
OutputCheck = Callable[[Any, Session], "Finding | None"]

# Extractor over output text (not the raw output object): (text) -> values.
TextExtractor = Callable[[str], Iterable[str]]

_URL_RE = re.compile(r'https?://[^\s<>"\')\]]+', re.I)


def extract_urls(text: str) -> set[str]:
    """URLs in a text blob, common trailing punctuation trimmed.

    Suitable as the ``extract=`` for both ``mark_untrusted`` and the check
    factories here — using the same extractor on both sides is what makes
    the membership comparison meaningful.
    """
    return {u.rstrip(".,;:!?)]}'\"") for u in _URL_RE.findall(text or "")}


@dataclass(frozen=True)
class Finding:
    """One output-check hit.

    Attributes:
        check: Name of the check that fired.
        message: Human-readable description, ready to surface to a reviewer.
        values: The offending values (sorted, for stable output).
        severity: ``"error"`` (act only after resolving) or ``"warning"``.
    """

    check: str
    message: str
    values: tuple[str, ...]
    severity: str = "error"


def _as_text(output: Any) -> str:
    """Project any output shape to text for extraction.

    Strings pass through; everything else is JSON-serialized (with ``str``
    fallback), so structured-output dicts are checked across all fields.
    """
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(output)


def check_output(
    output: Any,
    session: Session,
    *,
    checks: Iterable[OutputCheck],
    enforce: bool = False,
) -> list[Finding]:
    """Run output checks; return findings (empty list = clean).

    ``output`` may be the final text or a structured-output dict. Findings
    are appended to the session audit trail. With ``enforce=True`` any
    finding raises :class:`OutputViolation` — use this when code, not a
    human, acts on the output next. A check that itself raises is converted
    into an ``"error"``-severity finding rather than being swallowed: a
    broken check must not read as a clean pass.
    """
    findings: list[Finding] = []
    for check in checks:
        name = getattr(check, "__name__", repr(check))
        try:
            finding = check(output, session)
        except Exception as exc:
            finding = Finding(
                check=name,
                message=f"check raised {type(exc).__name__}: {exc} (treated as a finding, not a pass)",
                values=(),
                severity="error",
            )
        if finding is not None:
            findings.append(finding)
            session.note(
                kind="output_finding",
                check=finding.check,
                severity=finding.severity,
                message=finding.message,
                values=list(finding.values),
            )
    if enforce and findings:
        raise OutputViolation(findings)
    return findings


def echoed_untrusted_values(
    label: str, *, extract: TextExtractor = extract_urls, severity: str = "error"
) -> OutputCheck:
    """Flag output values that appeared in the untrusted input set ``label``.

    ``label`` is the same label passed to ``Session.mark_untrusted`` (the
    check reads the ``untrusted:<label>`` set). Membership goes through the
    set's bound normalizer, so pass the same ``normalize=`` at both ends.
    """
    set_name = untrusted_set(label)

    def check(output: Any, session: Session) -> Finding | None:
        hits = sorted(
            v for v in extract(_as_text(output)) if session.provenance.contains(set_name, v)
        )
        if not hits:
            return None
        return Finding(
            check=check.__name__,
            message=(
                f"output contains {len(hits)} value(s) that appeared in the "
                f"untrusted input {label!r} — remove or independently verify "
                "before acting: " + ", ".join(hits)
            ),
            values=tuple(hits),
            severity=severity,
        )

    check.__name__ = f"echoed_untrusted_values({label!r})"
    return check


def values_grounded_in(
    *,
    allowed: Iterable[str],
    extract: TextExtractor = extract_urls,
    severity: str = "warning",
) -> OutputCheck:
    """Flag output values not present in any of the ``allowed`` provenance sets.

    The positive grounding assertion: with ``allowed=["searched_urls"]``,
    every URL the output cites must be one a prior search actually returned.
    Values the extractor finds nowhere in the allowed sets are reported;
    an output with no extracted values passes.
    """
    allowed_labels = tuple(allowed)

    def check(output: Any, session: Session) -> Finding | None:
        stray = sorted(
            v
            for v in extract(_as_text(output))
            if not any(session.provenance.contains(lbl, v) for lbl in allowed_labels)
        )
        if not stray:
            return None
        return Finding(
            check=check.__name__,
            message=(
                f"output contains {len(stray)} value(s) grounded in none of "
                f"the provenance sets {list(allowed_labels)!r}: " + ", ".join(stray)
            ),
            values=tuple(stray),
            severity=severity,
        )

    check.__name__ = f"values_grounded_in({list(allowed_labels)!r})"
    return check
