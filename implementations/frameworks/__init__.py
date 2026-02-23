"""Framework adapters for different agent implementations."""

from .base_agent import BaseAgent

__all__ = ["BaseAgent"]

# Framework-specific imports are done dynamically to avoid import errors
# when dependencies are not installed
