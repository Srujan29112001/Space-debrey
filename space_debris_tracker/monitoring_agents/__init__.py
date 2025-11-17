"""
Agentic Monitoring System
=========================

Multi-agent system using MCP protocol for autonomous satellite monitoring
"""

from .mcp.mcp_client import MCPClient
from .space_monitoring_agent import SpaceMonitoringAgent

__all__ = [
    "SpaceMonitoringAgent",
    "MCPClient",
]
