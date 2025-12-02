"""
BioPipelines MCP Server Package
================================

Exposes BioPipelines tools via Model Context Protocol (MCP)
for integration with Claude Code, Cursor, and other MCP-compatible clients.

Usage:
    python -m workflow_composer.mcp.server
"""

from .server import BioPipelinesMCPServer, create_server

__all__ = ["BioPipelinesMCPServer", "create_server"]
