"""
Centralized logging for Progent.
"""

import json
import logging
import os
import sys
from typing import Any, Optional

# Default level from env, default to INFO
DEFAULT_LOG_LEVEL = os.getenv("PROGENT_LOG_LEVEL", "INFO").upper()


class ProgentLogger:
    """
    Wrapper around python standard logging to provide Progent-specific methods.
    """

    def __init__(self, name: str = "progent"):
        self.logger = logging.getLogger(name)

    def info(self, msg: str, **kwargs):
        self.logger.info(msg, **kwargs)

    def debug(self, msg: str, **kwargs):
        self.logger.debug(msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        self.logger.warning(msg, **kwargs)

    def error(self, msg: str, **kwargs):
        self.logger.error(msg, **kwargs)

    def tool_call(self, tool_name: str, arguments: dict[str, Any]):
        """Log a tool call."""
        if self.logger.isEnabledFor(logging.INFO):
            try:
                args_str = json.dumps(arguments)
            except (TypeError, ValueError):
                args_str = str(arguments)
            self.logger.info(f"TOOL_CALL: {tool_name} args={args_str}")

    def progent_decision(self, tool_name: str, allowed: bool, reason: str = ""):
        """
        Log a policy decision.
        Allowed = INFO
        Blocked = WARNING (to make it visible in production if level >= WARNING)
        """
        status = "ALLOWED" if allowed else "BLOCKED"
        msg = f"DECISION: {status} tool={tool_name}"
        if reason:
            msg += f" reason={reason}"

        if allowed:
            self.logger.info(msg)
        else:
            self.logger.warning(msg)


_logger: Optional[ProgentLogger] = None


def get_logger() -> ProgentLogger:
    """Get the global Progent logger."""
    global _logger
    if _logger is None:
        _logger = ProgentLogger()
    return _logger


def configure_logging(level: Optional[str] = None, log_file: Optional[str] = None):
    """
    Configure the progent logger handlers.
    Args:
        level: Logging level (e.g. "DEBUG", "INFO", "WARNING"). Defaults to PROGENT_LOG_LEVEL env var.
        log_file: Optional path to write logs to file.
    """
    if level is None:
        level = DEFAULT_LOG_LEVEL

    logger = logging.getLogger("progent")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = (
        False  # Do not propagate to root logger to avoid duplicates if app uses basicConfig
    )

    # Console Handler
    console = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(levelname)s] [progent] %(message)s")
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File Handler
    if log_file:
        fh = logging.FileHandler(log_file)
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)
