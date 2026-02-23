"""Communication tools for the coding agent (dummy implementations)."""

from implementations.core.logging_utils import get_logger


def send_email(to: str, subject: str, body: str) -> str:
    """
    Send an email notification.

    NOTE: This is a dummy implementation that logs the email instead of sending it.
    In production, this would integrate with an actual email service.

    :param to: The recipient email address
    :param subject: The email subject line
    :param body: The email body content
    :return: Confirmation message
    """
    logger = get_logger()

    # Log the email
    logger.info(f"EMAIL [DUMMY]: To={to}, Subject={subject}")
    logger.debug(f"EMAIL BODY: {body}")

    # Also print to console for visibility during testing
    print(f"\n{'=' * 50}")
    print("EMAIL SENT (DUMMY)")
    print(f"{'=' * 50}")
    print(f"To: {to}")
    print(f"Subject: {subject}")
    print(f"Body:\n{body}")
    print(f"{'=' * 50}\n")

    return f"Email sent successfully to {to} with subject '{subject}'"
