"""
Tool Registry
=============

Decorator-based tool registration system.
Auto-generates OpenAI function calling definitions.
"""

import re
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from functools import wraps

from .base import ToolResult, ToolParameter, RegisteredTool, ToolName

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Central registry for all agent tools.
    
    Usage:
        @ToolRegistry.register(
            name="scan_data",
            description="Scan directory for sequencing data",
            parameters=[
                ToolParameter("path", "string", "Directory to scan", required=True)
            ],
            patterns=[r"scan data in (.+)"]
        )
        def scan_data(path: str) -> ToolResult:
            ...
    """
    
    _tools: Dict[str, RegisteredTool] = {}
    _initialized: bool = False
    
    @classmethod
    def register(
        cls,
        name: str,
        description: str,
        parameters: List[ToolParameter] = None,
        patterns: List[str] = None,
        category: str = "general",
    ) -> Callable:
        """
        Decorator to register a tool.
        
        Args:
            name: Tool name (should match ToolName enum)
            description: Description for LLM
            parameters: List of ToolParameter definitions
            patterns: Regex patterns for fallback detection
            category: Tool category for grouping
        """
        def decorator(func: Callable) -> Callable:
            tool = RegisteredTool(
                name=name,
                description=description,
                parameters=parameters or [],
                patterns=patterns or [],
                handler=func,
                category=category,
            )
            cls._tools[name] = tool
            logger.debug(f"Registered tool: {name}")
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            return wrapper
        
        return decorator
    
    @classmethod
    def get_tool(cls, name: str) -> Optional[RegisteredTool]:
        """Get a registered tool by name."""
        return cls._tools.get(name)
    
    @classmethod
    def get_all_tools(cls) -> Dict[str, RegisteredTool]:
        """Get all registered tools."""
        return cls._tools.copy()
    
    @classmethod
    def get_openai_tools(cls) -> List[Dict[str, Any]]:
        """Generate OpenAI function calling definitions for all tools."""
        return [tool.to_openai_format() for tool in cls._tools.values()]
    
    @classmethod
    def get_patterns(cls) -> List[Tuple[str, str]]:
        """Get all regex patterns for fallback matching."""
        patterns = []
        for tool in cls._tools.values():
            for pattern in tool.patterns:
                patterns.append((pattern, tool.name))
        return patterns
    
    @classmethod
    def detect_tool(cls, message: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Detect which tool to use based on message patterns.
        
        Returns:
            Tuple of (tool_name, extracted_args) or None
        """
        message_lower = message.lower().strip()
        
        for tool in cls._tools.values():
            args = tool.matches(message_lower)
            if args is not None:
                return (tool.name, args)
        
        return None
    
    @classmethod
    def execute(cls, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        tool = cls.get_tool(name)
        if not tool:
            return ToolResult(
                success=False,
                tool_name=name,
                error=f"Unknown tool: {name}",
                message=f"❌ Tool '{name}' not found"
            )
        
        try:
            return tool.handler(**kwargs)
        except Exception as e:
            logger.exception(f"Tool {name} failed")
            return ToolResult(
                success=False,
                tool_name=name,
                error=str(e),
                message=f"❌ Tool error: {e}"
            )
    
    @classmethod
    def list_by_category(cls) -> Dict[str, List[str]]:
        """List tools grouped by category."""
        categories = {}
        for tool in cls._tools.values():
            if tool.category not in categories:
                categories[tool.category] = []
            categories[tool.category].append(tool.name)
        return categories
    
    @classmethod
    def get_help_text(cls) -> str:
        """Generate help text for all tools."""
        lines = ["# Available Tools\n"]
        
        categories = cls.list_by_category()
        for category, tool_names in sorted(categories.items()):
            lines.append(f"\n## {category.title()}\n")
            for name in tool_names:
                tool = cls._tools[name]
                lines.append(f"- **{name}**: {tool.description}")
        
        return "\n".join(lines)


# Convenience function for pattern-based tool detection
def detect_tool_from_message(message: str) -> Optional[Tuple[ToolName, Dict[str, Any]]]:
    """
    Detect tool from message using registry patterns.
    
    Returns:
        Tuple of (ToolName enum, args dict) or None
    """
    result = ToolRegistry.detect_tool(message)
    if result:
        name, args = result
        try:
            tool_name = ToolName(name)
            return (tool_name, args)
        except ValueError:
            # Tool name doesn't match enum
            return None
    return None


def get_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    return ToolRegistry
