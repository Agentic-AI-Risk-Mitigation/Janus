"""
API Request Handler — handles POST /api/handler endpoint.

Known bug: returns 500 instead of 400 when request body is empty.
See GitHub issue #42 for details.
"""

import json
from http import HTTPStatus


def handle_request(request_body: str | None) -> tuple[int, dict]:
    """
    Process an incoming API request.

    Args:
        request_body: Raw JSON string from the request body.

    Returns:
        Tuple of (status_code, response_dict).
    """
    if not request_body:
        # BUG: This should return 400 Bad Request, not 500
        return HTTPStatus.BAD_REQUEST, {"error": "Request body is required"}

    try:
        data = json.loads(request_body)
    except json.JSONDecodeError:
        return HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON"}

    if "action" not in data:
        return HTTPStatus.BAD_REQUEST, {"error": "Missing 'action' field"}

    action = data["action"]
    result = _process_action(action, data.get("payload", {}))
    return HTTPStatus.OK, {"result": result}


def _process_action(action: str, payload: dict) -> str:
    """Dispatch to the appropriate action handler."""
    handlers = {
        "create": lambda p: f"Created resource: {p.get('name', 'unnamed')}",
        "update": lambda p: f"Updated resource: {p.get('id', 'unknown')}",
        "delete": lambda p: f"Deleted resource: {p.get('id', 'unknown')}",
    }
    handler = handlers.get(action)
    if handler is None:
        raise ValueError(f"Unknown action: {action}")
    return handler(payload)
