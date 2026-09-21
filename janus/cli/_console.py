"""
Stdlib prompting for ``janus init``.

No prompt library. Janus's core install is two dependencies (``jsonschema``,
``pydantic``) and everything else is a lazy extra; a security tool that is
awkward to install is a security tool that does not get installed. Arrow-key
menus are not worth a dependency in the hot path of "guard my agent", so this
is numbered menus and ``[Y/n]`` on plain ASCII — which also sidesteps terminal
capability differences on Windows.

Streams resolve at call time rather than at construction, so a test can
monkeypatch ``sys.stdin`` (the idiom the shim tests already use) *or* inject
streams directly, and both work.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import IO, Any

from janus.exceptions import JanusError

#: Width used for the screen rules. Narrow enough for a split terminal.
_RULE_WIDTH = 66


class Aborted(JanusError):
    """The operator ended the wizard — Ctrl-D, Ctrl-C, or an explicit "no".

    Carried as an exception so every abort path unwinds to the same place:
    nothing is written unless the wizard reaches its final confirmation.
    """


def stdin_is_tty() -> bool:
    """Whether a human could actually answer a prompt.

    Piped or redirected stdin means the "defaults" a non-interactive run would
    silently accept were nobody's decision. The wizard refuses that unless
    ``--yes`` says the defaults are intended.
    """
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


class Console:
    """Prompt/print helpers over a pair of text streams."""

    def __init__(
        self,
        in_stream: IO[str] | None = None,
        out_stream: IO[str] | None = None,
        *,
        assume_defaults: bool = False,
    ) -> None:
        self._in = in_stream
        self._out = out_stream
        self.assume_defaults = assume_defaults

    # -- output ---------------------------------------------------------

    @property
    def _stdout(self) -> IO[str]:
        return self._out if self._out is not None else sys.stdout

    @property
    def _stdin(self) -> IO[str]:
        return self._in if self._in is not None else sys.stdin

    def say(self, text: str = "") -> None:
        print(text, file=self._stdout)

    def heading(self, text: str) -> None:
        self.say()
        self.say(text)
        self.say("-" * min(len(text), _RULE_WIDTH))

    def note(self, text: str) -> None:
        """An indented explanatory line under a prompt."""
        self.say(f"  {text}")

    def bullet(self, text: str) -> None:
        self.say(f"  - {text}")

    # -- input ----------------------------------------------------------

    def _read_line(self) -> str:
        line = self._stdin.readline()
        if line == "":
            raise Aborted("end of input")
        return line.strip()

    def _auto(self, prompt: str, shown: str) -> None:
        self.say(f"{prompt} -> [auto] {shown}")

    def ask_yn(self, prompt: str, *, default: bool) -> bool:
        suffix = "[Y/n]" if default else "[y/N]"
        question = f"{prompt} {suffix}"
        if self.assume_defaults:
            self._auto(question, "yes" if default else "no")
            return default

        while True:
            self.say(question)
            answer = self._read_line().lower()
            if not answer:
                return default
            if answer in ("y", "yes"):
                return True
            if answer in ("n", "no"):
                return False
            self.note("Please answer y or n.")

    def ask_text(self, prompt: str, *, default: str = "") -> str:
        shown = default if default else "none"
        question = f"{prompt} [{shown}]"
        if self.assume_defaults:
            self._auto(question, shown)
            return default

        self.say(question)
        answer = self._read_line()
        return answer or default

    def ask_choice(
        self,
        prompt: str,
        options: Sequence[tuple[str, str]],
        *,
        default: int = 0,
    ) -> str:
        """Numbered menu. ``options`` are ``(value, label)``; returns the value."""
        if not options:
            raise ValueError("ask_choice requires at least one option")
        if not 0 <= default < len(options):
            raise ValueError(f"default index {default} is out of range")

        if self.assume_defaults:
            self._auto(prompt, options[default][1])
            return options[default][0]

        self.say(prompt)
        for index, (_, label) in enumerate(options, start=1):
            marker = "*" if index - 1 == default else " "
            self.say(f"  {marker} {index}) {label}")

        while True:
            self.say(f"Choose 1-{len(options)} [{default + 1}]")
            answer = self._read_line()
            if not answer:
                return options[default][0]
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                return options[int(answer) - 1][0]
            self.note(f"Please enter a number between 1 and {len(options)}.")

    def ask_list(self, prompt: str, *, default: Sequence[str] = ()) -> list[str]:
        """Comma-separated free text, normalized to a list of non-empty items."""
        shown = ", ".join(default)
        raw = self.ask_text(prompt, default=shown)
        return [item.strip() for item in raw.split(",") if item.strip()]


def format_kv(pairs: Sequence[tuple[str, Any]], *, indent: str = "  ") -> str:
    """Aligned ``key: value`` block for the review screen."""
    if not pairs:
        return ""
    width = max(len(str(key)) for key, _ in pairs)
    return "\n".join(f"{indent}{str(key).ljust(width)}  {value}" for key, value in pairs)
