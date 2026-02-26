"""
Centralized logging for Janus.

Provides a structured logger with Janus-specific methods for tool calls,
policy decisions, and agent events. Configure once, use everywhere.
"""

import json
import logging
import os
import sys
from typing import Any


_DEFAULT_LEVEL = os.getenv("JANUS_LOG_LEVEL", "INFO").upper()


class JanusLogger:
    """
    Structured logger for Janus events.

    Wraps Python's standard logging with convenience methods for the
    security-relevant events that Janus tracks (tool calls, policy decisions).
    """

    def __init__(self, name: str = "janus"):
        self._logger = logging.getLogger(name)

    def debug(self, msg: str, **kwargs) -> None:
        self._logger.debug(msg, **kwargs)

    def info(self, msg: str, **kwargs) -> None:
        self._logger.info(msg, **kwargs)

    def warning(self, msg: str, **kwargs) -> None:
        self._logger.warning(msg, **kwargs)

    def error(self, msg: str, **kwargs) -> None:
        self._logger.error(msg, **kwargs)

    def tool_call(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Log an incoming tool call before enforcement."""
        if self._logger.isEnabledFor(logging.DEBUG):
            try:
                args_str = json.dumps(arguments)
            except (TypeError, ValueError):
                args_str = str(arguments)
            self._logger.debug(f"TOOL_CALL  tool={tool_name} args={args_str}")

    def tool_result(self, tool_name: str, result: str, *, success: bool) -> None:
        """Log the outcome of a tool execution."""
        status = "OK" if success else "ERROR"
        truncated = result[:200] + "..." if len(result) > 200 else result
        self._logger.debug(f"TOOL_RESULT [{status}] tool={tool_name} result={truncated!r}")

    def policy_decision(
        self,
        tool_name: str,
        *,
        allowed: bool,
        reason: str = "",
    ) -> None:
        """
        Log a policy enforcement decision.

        Allowed calls are logged at INFO; blocked calls at WARNING so that
        violations are visible even in production at WARNING log level.
        """
        verdict = "ALLOWED" if allowed else "BLOCKED"
        msg = f"POLICY [{verdict}] tool={tool_name}"
        if reason:
            msg += f" | {reason}"

        if allowed:
            self._logger.info(msg)
        else:
            self._logger.warning(msg)

    def agent_event(self, event: str, detail: str = "") -> None:
        """Log a high-level agent lifecycle event."""
        msg = f"AGENT [{event}]"
        if detail:
            msg += f" {detail}"
        self._logger.info(msg)


_logger: JanusLogger | None = None


def get_logger() -> JanusLogger:
    """Return the global Janus logger, creating it on first call."""
    global _logger
    if _logger is None:
        _logger = JanusLogger()
    return _logger


def configure_logging(
    level: str | None = None,
    log_file: str | None = None,
    fmt: str = "[%(levelname)s] [janus] %(message)s",
) -> None:
    """
    Configure the Janus logger.

    Call this once at application startup before creating any Janus objects.

    Args:
        level: Log level string ("DEBUG", "INFO", "WARNING", "ERROR").
               Defaults to the JANUS_LOG_LEVEL env var, then "INFO".
        log_file: Optional path to write structured logs to a file.
        fmt: Log format string for the console handler.
    """
    effective_level = (level or _DEFAULT_LEVEL).upper()

    logger = logging.getLogger("janus")
    logger.setLevel(effective_level)
    logger.handlers.clear()
    logger.propagate = False  # avoid duplicates when root logger is configured

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(fmt))
    logger.addHandler(console)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(fh)
