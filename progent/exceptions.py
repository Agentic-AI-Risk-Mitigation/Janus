"""
Progent custom exceptions.
"""


class ProgentError(Exception):
    """Base exception for all Progent errors."""

    pass


class ProgentBlockedError(ProgentError):
    """
    Raised when a tool call is blocked by policy.

    Attributes:
        tool_name: Name of the blocked tool
        arguments: Arguments that were passed
        reason: Human-readable reason for blocking
    """

    def __init__(
        self,
        tool_name: str,
        arguments: dict = None,
        reason: str = None,
        policy_rule: tuple | None = None,
        failed_condition: str | None = None,
    ):
        self.tool_name = tool_name
        self.arguments = arguments or {}
        self.reason = reason or f"Tool '{tool_name}' is not allowed."
        self.policy_rule = policy_rule
        self.failed_condition = failed_condition

        super().__init__(self.reason)


class PolicyValidationError(ProgentError):
    """
    Raised when a policy argument fails validation.

    Attributes:
        argument_name: Name of the argument that failed
        value: The value that was provided
        restriction: The restriction that was violated
    """

    def __init__(
        self,
        argument_name: str,
        value,
        restriction,
        message: str = None,
    ):
        self.argument_name = argument_name
        self.value = value
        self.restriction = restriction

        if message:
            self.message = message
        else:
            self.message = (
                f"Invalid value for argument '{argument_name}': "
                f"value '{value}' does not satisfy restriction"
            )

        super().__init__(self.message)


class PolicyLoadError(ProgentError):
    """Raised when there's an error loading or parsing policies."""

    pass


class PolicyConfigError(ProgentError):
    """Raised when there's an error in policy configuration."""

    pass
