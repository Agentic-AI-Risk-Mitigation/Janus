"""
Progent framework adapters.

These modules provide integration with specific AI agent frameworks:
- langchain: LangChain tool integration
- mcp: Model Context Protocol middleware

Each adapter is optional and only requires its specific dependencies.
"""

__all__ = []


# Lazy imports to avoid requiring framework dependencies
def get_langchain_adapter():
    """Get the LangChain adapter module."""
    from progent.adapters import langchain

    return langchain


def get_mcp_adapter():
    """Get the MCP adapter module."""
    from progent.adapters import mcp

    return mcp
