"""
Janus custom exceptions.

All Janus exceptions inherit from JanusError so callers can catch either
the specific exception or the base class.
"""


class JanusError(Exception):
    """Base exception for all Janus errors."""
    pass


class PolicyViolation(JanusError):
    """
    Raised when a tool call is blocked by policy enforcement.

    Attributes:
        tool_name: Name of the blocked tool
        arguments: Arguments that were passed to the tool
        reason: Human-readable reason for the violation
        policy_rule: The specific rule that triggered the block
    """

    def __init__(
        self,
        tool_name: str,
        arguments: dict | None = None,
        reason: str | None = None,
        policy_rule: tuple | None = None,
    ):
        self.tool_name = tool_name
        self.arguments = arguments or {}
        self.reason = reason or f"Tool '{tool_name}' is not permitted by policy."
        self.policy_rule = policy_rule

        super().__init__(self.reason)

    def __repr__(self) -> str:
        return f"PolicyViolation(tool={self.tool_name!r}, reason={self.reason!r})"


class ArgumentValidationError(JanusError):
    """
    Raised when a tool argument fails JSON Schema validation during enforcement.

    Attributes:
        argument_name: Name of the argument that failed validation
        value: The value that was provided
        restriction: The schema restriction that was violated
        message: Human-readable validation message
    """

    def __init__(
        self,
        argument_name: str,
        value,
        restriction,
        message: str | None = None,
    ):
        self.argument_name = argument_name
        self.value = value
        self.restriction = restriction
        self.message = message or (
            f"Argument '{argument_name}' with value {value!r} "
            f"does not satisfy the policy restriction."
        )

        super().__init__(self.message)


class PolicyLoadError(JanusError):
    """Raised when a policy file cannot be loaded or parsed."""
    pass


class PolicyGenerationError(JanusError):
    """Raised when LLM-based policy generation fails."""
    pass


class ToolNotFoundError(JanusError):
    """Raised when a requested tool is not found in the registry."""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' is not registered.")


class ProviderError(JanusError):
    """Raised when an LLM provider encounters an error."""
    pass
