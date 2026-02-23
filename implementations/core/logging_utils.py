"""Structured logging utilities for the agent."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class AgentLogger:
    """Logger that writes structured logs to both console and file."""

    def __init__(
        self,
        log_dir: str = "./logs",
        level: str = "INFO",
        session_name: Optional[str] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create session-specific log file
        if session_name is None:
            session_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = self.log_dir / f"{session_name}.log"

        # Set up Python logger
        self.logger = logging.getLogger(f"agent_{session_name}")
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.handlers.clear()

        # File handler
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler (less verbose)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

    def user_input(self, message: str) -> None:
        """Log user input."""
        self.logger.info(f"USER: {message}")

    def assistant_response(self, message: str) -> None:
        """Log assistant response."""
        self.logger.info(f"ASSISTANT: {message}")

    def tool_call(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Log a tool call from the LLM."""
        json_str = json.dumps({"name": tool_name, "arguments": arguments}, indent=None)
        self.logger.info(f"LLM_TOOL_CALL: {json_str}")

    def progent_decision(
        self, tool_name: str, arguments: dict[str, Any], allowed: bool, reason: str = ""
    ) -> None:
        """Log Progent policy decision."""
        status = "ALLOWED" if allowed else "BLOCKED"
        args_str = ", ".join(
            f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}" for k, v in arguments.items()
        )
        msg = f"PROGENT: {status} - {tool_name}({args_str})"
        if reason and not allowed:
            msg += f" | Reason: {reason}"
        self.logger.info(msg)

    def tool_result(self, tool_name: str, result: str, success: bool = True) -> None:
        """Log tool execution result."""
        status = "SUCCESS" if success else "ERROR"
        # Truncate very long results for console
        display_result = result[:500] + "..." if len(result) > 500 else result
        self.logger.info(f"TOOL_RESULT [{status}]: {display_result}")

    def error(self, message: str, exception: Optional[Exception] = None) -> None:
        """Log an error."""
        if exception:
            self.logger.error(f"ERROR: {message} | {type(exception).__name__}: {exception}")
        else:
            self.logger.error(f"ERROR: {message}")

    def info(self, message: str) -> None:
        """Log general info."""
        self.logger.info(f"INFO: {message}")

    def debug(self, message: str) -> None:
        """Log debug info (file only)."""
        self.logger.debug(f"DEBUG: {message}")


# Global logger instance
_logger: Optional[AgentLogger] = None


def get_logger() -> AgentLogger:
    """Get the global logger instance."""
    global _logger
    if _logger is None:
        _logger = AgentLogger()
    return _logger


def init_logger(
    log_dir: str = "./logs", level: str = "INFO", session_name: Optional[str] = None
) -> AgentLogger:
    """Initialize the global logger with custom settings."""
    global _logger
    _logger = AgentLogger(log_dir=log_dir, level=level, session_name=session_name)
    return _logger
