"""
Agent Integration Module (Legacy)
==================================

NOTE: This module is maintained for backward compatibility.
For new code, prefer using UnifiedAgent directly:

    from workflow_composer.agents import UnifiedAgent
    agent = UnifiedAgent()
    response = await agent.process_query("scan /data for files")

The AgentBridge provided a simpler interface before UnifiedAgent was created.
It bridges LLM-based AgentRouter with the AgentTools system.

Legacy Usage:
    agent = AgentBridge(app_state)
    result = await agent.process_message(message, context)
"""

import asyncio
import logging
import warnings
from typing import Dict, Any, Optional

from workflow_composer.agents.tools import AgentTools, ToolResult, ToolName
from workflow_composer.agents._archived.router import (
    AgentRouter, 
    RouteResult, 
    RoutingStrategy,
    route_message
)

logger = logging.getLogger(__name__)


def _emit_deprecation_warning():
    """Emit deprecation warning for AgentBridge usage."""
    warnings.warn(
        "AgentBridge is deprecated. Use UnifiedAgent instead: "
        "from workflow_composer.agents import UnifiedAgent",
        DeprecationWarning,
        stacklevel=3
    )


class AgentBridge:
    """
    Bridges LLM routing with tool execution.
    
    DEPRECATED: Use UnifiedAgent instead for new code.
    
    The bridge:
    1. Uses AgentRouter to determine intent via LLM
    2. Maps router results to AgentTools execution
    3. Falls back gracefully if LLM not available
    """
    
    TOOL_NAME_MAP = {
        "scan_data": ToolName.SCAN_DATA,
        "search_databases": ToolName.SEARCH_DATABASES,
        "check_references": ToolName.CHECK_REFERENCES,
        "generate_workflow": None,  # Special case - handled by chat handler
        "submit_job": ToolName.SUBMIT_JOB,
        "get_job_status": ToolName.GET_JOB_STATUS,
        "get_logs": ToolName.GET_LOGS,
        "diagnose_error": ToolName.DIAGNOSE_ERROR,
        "cancel_job": ToolName.CANCEL_JOB,
        "list_workflows": ToolName.LIST_WORKFLOWS,
    }
    
    def __init__(
        self,
        app_state=None,
        use_llm_routing: bool = True,
        local_url: str = "http://localhost:8000/v1",
        cloud_provider: str = "lightning"
    ):
        """
        Initialize the agent bridge.
        
        Args:
            app_state: Application state from gradio_app
            use_llm_routing: Whether to try LLM routing first
            local_url: URL of local vLLM server
            cloud_provider: Cloud provider for fallback
        """
        self.app_state = app_state
        self.use_llm_routing = use_llm_routing
        
        # Initialize components
        self.tools = AgentTools(app_state)
        self.router = AgentRouter(
            local_url=local_url,
            cloud_provider=cloud_provider,
            use_local=use_llm_routing,
            use_cloud=use_llm_routing,
            use_regex_fallback=True
        ) if use_llm_routing else None
        
    async def process_message(
        self, 
        message: str, 
        context: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Process a user message - route and execute appropriate tool.
        
        Args:
            message: User's natural language message
            context: Conversation context (from ConversationContext)
            
        Returns:
            Dict with:
                - tool_result: ToolResult if tool was executed
                - route_result: RouteResult from routing
                - requires_generation: True if workflow generation needed
                - response: Direct response if no tool needed
        """
        # Try LLM routing first
        if self.router:
            try:
                route_result = await self.router.route(message, context)
                return await self._process_route_result(route_result)
            except Exception as e:
                logger.warning(f"LLM routing failed, falling back to regex: {e}")
        
        # Fall back to regex-based tool detection
        detection = self.tools.detect_tool(message)
        
        if detection:
            tool_name, args = detection
            tool_result = self.tools.execute(tool_name, args)
            return {
                "tool_result": tool_result,
                "route_result": None,
                "requires_generation": False,
                "response": None,
                "strategy": "regex_fallback"
            }
        
        # No tool detected
        return None
    
    async def _process_route_result(self, route: RouteResult) -> Optional[Dict[str, Any]]:
        """Convert RouteResult to tool execution."""
        
        result = {
            "tool_result": None,
            "route_result": route,
            "requires_generation": route.requires_generation,
            "response": route.response,
            "strategy": route.strategy.value,
            "confidence": route.confidence
        }
        
        # If workflow generation is needed, return early
        if route.requires_generation:
            result["generation_params"] = route.arguments
            return result
        
        # If direct response, return it
        if route.response and not route.tool:
            return result
        
        # Execute the tool
        if route.tool:
            tool_enum = self.TOOL_NAME_MAP.get(route.tool)
            
            if tool_enum is None and route.tool == "generate_workflow":
                # Special case - let the caller handle workflow generation
                return result
            
            if tool_enum:
                # Convert arguments dict to list for tool execution
                args = self._args_dict_to_list(route.tool, route.arguments)
                tool_result = self.tools.execute(tool_enum, args)
                result["tool_result"] = tool_result
        
        return result
    
    def _args_dict_to_list(self, tool_name: str, args: Dict[str, Any]) -> list:
        """Convert router's dict arguments to tools' list arguments."""
        
        if tool_name == "scan_data":
            return [args.get("path", ".")]
        
        elif tool_name == "search_databases":
            # search_databases takes a query string
            parts = []
            if args.get("organism"):
                parts.append(args["organism"])
            if args.get("assay_type"):
                parts.append(args["assay_type"])
            if args.get("query"):
                parts.append(args["query"])
            return [" ".join(parts) if parts else args.get("query", "")]
        
        elif tool_name == "check_references":
            return [args.get("organism", "human")]
        
        elif tool_name == "submit_job":
            return [args.get("workflow_name"), args.get("profile", "slurm")]
        
        elif tool_name == "get_job_status":
            return [args.get("job_id")] if args.get("job_id") else []
        
        elif tool_name == "get_logs":
            return [args.get("job_id")] if args.get("job_id") else []
        
        elif tool_name == "cancel_job":
            return [args.get("job_id")]
        
        elif tool_name == "diagnose_error":
            return [args.get("job_id")] if args.get("job_id") else []
        
        elif tool_name == "list_workflows":
            return []
        
        return []
    
    def process_message_sync(
        self, 
        message: str, 
        context: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Synchronous wrapper for process_message.
        
        For use in non-async contexts like Gradio handlers.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new task in the existing loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run, 
                        self.process_message(message, context)
                    )
                    return future.result(timeout=30)
            else:
                return loop.run_until_complete(
                    self.process_message(message, context)
                )
        except Exception as e:
            logger.error(f"Sync wrapper error: {e}")
            # Fall back to direct tool detection
            detection = self.tools.detect_tool(message)
            if detection:
                tool_name, args = detection
                tool_result = self.tools.execute(tool_name, args)
                return {
                    "tool_result": tool_result,
                    "route_result": None,
                    "requires_generation": False,
                    "response": None,
                    "strategy": "regex_sync_fallback"
                }
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent bridge status."""
        return {
            "llm_routing_enabled": self.use_llm_routing,
            "router_status": self.router.get_status() if self.router else None,
            "tools_available": [t.value for t in ToolName],
        }


# Singleton instance
_agent_bridge: Optional[AgentBridge] = None


def get_agent_bridge(app_state=None) -> AgentBridge:
    """Get or create the global agent bridge."""
    global _agent_bridge
    if _agent_bridge is None:
        _agent_bridge = AgentBridge(app_state)
    return _agent_bridge


async def process_with_agent(
    message: str, 
    context: Dict[str, Any] = None,
    app_state=None
) -> Optional[Dict[str, Any]]:
    """
    Convenience function for processing messages.
    
    Args:
        message: User message
        context: Conversation context
        app_state: App state
        
    Returns:
        Processing result or None
    """
    bridge = get_agent_bridge(app_state)
    return await bridge.process_message(message, context)


# Testing
if __name__ == "__main__":
    async def test():
        bridge = AgentBridge(use_llm_routing=False)  # Start with regex only
        
        print("🧪 Testing AgentBridge")
        print("=" * 60)
        print(f"Status: {bridge.get_status()}")
        print()
        
        test_messages = [
            "scan data in /scratch/mydata",
            "search for human RNA-seq in ENCODE",
            "what's the job status?",
            "show me the logs",
            "help",
        ]
        
        for msg in test_messages:
            print(f"📝 Message: \"{msg}\"")
            result = await bridge.process_message(msg)
            if result:
                print(f"   Strategy: {result.get('strategy')}")
                if result.get('tool_result'):
                    tr = result['tool_result']
                    print(f"   Tool: {tr.tool_name}")
                    print(f"   Success: {tr.success}")
                if result.get('response'):
                    print(f"   Response: {result['response'][:100]}...")
                if result.get('requires_generation'):
                    print(f"   → Needs workflow generation")
            else:
                print("   No tool/action detected")
            print()
    
    asyncio.run(test())
