"""
Progent MCP (Model Context Protocol) adapter.

Provides middleware for MCP servers that enforces Progent policies
on all tool calls.

Usage:
    from progent.adapters.mcp import ProgentMiddleware
    from fastmcp import FastMCP

    server = FastMCP(name="My Server")
    server.add_middleware(ProgentMiddleware())
"""

from typing import Any, Callable

from progent.core import check_tool_call
from progent.exceptions import ProgentBlockedError


class ProgentMiddleware:
    """
    MCP middleware that enforces Progent policies on tool calls.

    This middleware intercepts all tool calls through an MCP server
    and validates them against the configured security policies.

    Usage with FastMCP:
        from fastmcp import FastMCP
        from progent import load_policies
        from progent.adapters.mcp import ProgentMiddleware

        # Load policies
        load_policies("policies.json")

        # Create server with middleware
        server = FastMCP(name="Secured Server")
        server.add_middleware(ProgentMiddleware())
    """

    def __init__(
        self,
        on_blocked: Callable[[str, dict, str], Any] = None,
        log_calls: bool = True,
    ):
        """
        Initialize the middleware.

        Args:
            on_blocked: Optional callback when a tool is blocked.
                       Called with (tool_name, arguments, reason).
            log_calls: Whether to log tool calls (default: True)
        """
        self.on_blocked = on_blocked
        self.log_calls = log_calls

    async def on_call_tool(self, context, call_next):
        """
        Middleware hook for tool calls.

        This method is called by FastMCP for each tool invocation.
        """
        # Extract tool info from context
        tool_name = context.message.name
        tool_kwargs = context.message.arguments or {}

        if self.log_calls:
            print(f"[Progent/MCP] Tool call: {tool_name}({tool_kwargs})")

        # Check policy
        try:
            check_tool_call(tool_name, tool_kwargs)
        except ProgentBlockedError as e:
            reason = str(e)

            if self.log_calls:
                print(f"[Progent/MCP] BLOCKED: {reason}")

            if self.on_blocked:
                self.on_blocked(tool_name, tool_kwargs, reason)

            # Re-raise to block the call
            raise

        if self.log_calls:
            print(f"[Progent/MCP] ALLOWED: {tool_name}")

        # Continue to actual tool execution
        result = await call_next(context)
        return result


def create_secured_mcp_server(
    proxy_url: str = None,
    name: str = "Progent MCP Server",
    policies: dict = None,
    policies_file: str = None,
):
    """
    Create an MCP server with Progent security enabled.

    This is a convenience function that creates a FastMCP server
    with the Progent middleware already configured.

    Args:
        proxy_url: Optional URL to proxy requests to
        name: Server name
        policies: Policies dict (mutually exclusive with policies_file)
        policies_file: Path to policies JSON file

    Returns:
        Configured FastMCP server instance
    """
    try:
        from fastmcp import FastMCP
        from fastmcp.server.proxy import ProxyClient
    except ImportError:
        raise ImportError(
            "fastmcp is required for MCP integration. Install with: pip install fastmcp"
        )

    # Load policies
    if policies_file:
        from progent import load_policies

        load_policies(policies_file)
    elif policies:
        from progent import load_policies

        load_policies(policies)

    # Create server
    if proxy_url:
        server = FastMCP.as_proxy(
            ProxyClient(proxy_url),
            name=name,
        )
    else:
        server = FastMCP(name=name)

    # Add middleware
    server.add_middleware(ProgentMiddleware())

    return server


def run_secured_mcp_proxy(
    proxy_url: str,
    policies_file: str,
    port: int = 8080,
    host: str = "0.0.0.0",
):
    """
    Run a secured MCP proxy server.

    This creates and runs an MCP proxy server that forwards requests
    to another MCP server while enforcing Progent security policies.

    Args:
        proxy_url: URL of the upstream MCP server
        policies_file: Path to policies JSON file
        port: Port to run on (default: 8080)
        host: Host to bind to (default: 0.0.0.0)
    """
    server = create_secured_mcp_server(
        proxy_url=proxy_url,
        policies_file=policies_file,
    )

    server.run(
        transport="http",
        host=host,
        port=port,
        log_level="ERROR",
        stateless_http=True,
    )
