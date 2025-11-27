"""
Autonomous Agent Panel for Gradio UI
=====================================

UI components for the autonomous agent system:
- Health status display
- Autonomy level selector
- Task queue viewer
- Recovery controls
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import gradio as gr

logger = logging.getLogger(__name__)

# Import autonomous components
try:
    from workflow_composer.agents.autonomous import (
        AutonomousAgent,
        create_agent,
        HealthChecker,
        HealthStatus,
        SystemHealth,
        JobMonitor,
        RecoveryManager,
        AutonomyLevel,
    )
    from workflow_composer.agents.executor import (
        PermissionManager,
        AutonomyLevel as ExecAutonomyLevel,
    )
    AUTONOMOUS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Autonomous agent not available: {e}")
    AUTONOMOUS_AVAILABLE = False
    AutonomousAgent = None
    create_agent = None
    HealthChecker = None


# =============================================================================
# Global Agent Instance
# =============================================================================

_agent_instance: Optional["AutonomousAgent"] = None
_health_checker: Optional["HealthChecker"] = None


def get_agent(level: str = "assisted") -> Optional["AutonomousAgent"]:
    """Get or create the global agent instance."""
    global _agent_instance
    
    if not AUTONOMOUS_AVAILABLE:
        return None
    
    if _agent_instance is None:
        _agent_instance = create_agent(level=level)
    
    return _agent_instance


def get_health_checker() -> Optional["HealthChecker"]:
    """Get or create the global health checker."""
    global _health_checker
    
    if not AUTONOMOUS_AVAILABLE:
        return None
    
    if _health_checker is None:
        _health_checker = HealthChecker()
    
    return _health_checker


# =============================================================================
# Health Status Formatting
# =============================================================================

def format_health_status(health: "SystemHealth") -> str:
    """Format health status as HTML for Gradio display."""
    status_icons = {
        "healthy": "🟢",
        "degraded": "🟡",
        "unhealthy": "🔴",
        "unknown": "⚪",
    }
    
    overall_icon = status_icons.get(health.status.value, "⚪")
    
    html = f"""
    <div style="font-size: 0.9em; padding: 8px; border-radius: 8px; background: #f5f5f5;">
        <div style="font-weight: bold; margin-bottom: 8px;">
            {overall_icon} System: {health.status.value.upper()}
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px;">
    """
    
    for comp in health.components:
        icon = status_icons.get(comp.status.value, "⚪")
        timing = f" ({comp.response_time_ms:.0f}ms)" if comp.response_time_ms else ""
        html += f"""
            <div style="font-size: 0.85em;">
                {icon} {comp.name}{timing}
            </div>
        """
    
    html += """
        </div>
        <div style="font-size: 0.75em; color: #666; margin-top: 4px;">
            Updated: {time}
        </div>
    </div>
    """.format(time=health.checked_at.strftime("%H:%M:%S"))
    
    return html


def format_health_status_minimal(health: "SystemHealth") -> str:
    """Format minimal health status for sidebar."""
    status_icons = {
        "healthy": "🟢",
        "degraded": "🟡",
        "unhealthy": "🔴",
        "unknown": "⚪",
    }
    
    icon = status_icons.get(health.status.value, "⚪")
    
    # Count component statuses
    healthy = sum(1 for c in health.components if c.status.value == "healthy")
    total = len(health.components)
    
    return f"{icon} **{health.status.value.title()}** ({healthy}/{total} ok)"


# =============================================================================
# UI Component Creators
# =============================================================================

def create_autonomy_selector() -> gr.Dropdown:
    """Create autonomy level dropdown."""
    choices = [
        ("🔒 Read Only - Observe only", "readonly"),
        ("📋 Monitored - Read + logging", "monitored"),
        ("✋ Assisted - Needs confirmation", "assisted"),
        ("👁️ Supervised - Most auto", "supervised"),
        ("🤖 Autonomous - Full auto", "autonomous"),
    ]
    
    return gr.Dropdown(
        choices=choices,
        value="assisted",
        label="Autonomy Level",
        info="Higher levels = more automation",
        interactive=True,
    )


def create_health_panel() -> Tuple[gr.HTML, gr.Button, gr.Timer]:
    """Create health status panel components."""
    health_display = gr.HTML(
        value="<div style='color:#666'><em>Checking health...</em></div>",
        label="System Health",
    )
    
    refresh_btn = gr.Button("🔄 Check Health", size="sm")
    
    # Auto-refresh timer (every 60 seconds)
    timer = gr.Timer(60, active=True)
    
    return health_display, refresh_btn, timer


def create_agent_status_panel() -> Tuple[gr.Markdown, gr.Markdown]:
    """Create agent status display."""
    status = gr.Markdown("⏸️ Agent idle")
    queue = gr.Markdown("📋 Queue: 0 tasks")
    
    return status, queue


def create_recovery_controls() -> Tuple[gr.Dropdown, gr.Button, gr.Textbox]:
    """Create recovery action controls."""
    action_dropdown = gr.Dropdown(
        choices=[
            ("Restart vLLM Server", "restart_vllm"),
            ("Clear Cache", "clear_cache"),
            ("Check All Health", "health_check"),
        ],
        label="Recovery Action",
        interactive=True,
    )
    
    execute_btn = gr.Button("⚡ Execute", variant="secondary", size="sm")
    
    result = gr.Textbox(
        label="Result",
        interactive=False,
        lines=2,
    )
    
    return action_dropdown, execute_btn, result


# =============================================================================
# Event Handlers
# =============================================================================

async def check_health_async() -> str:
    """Check system health asynchronously."""
    checker = get_health_checker()
    if not checker:
        return "<div style='color:#666'><em>Health checker not available</em></div>"
    
    try:
        health = await checker.check_all()
        return format_health_status(health)
    except Exception as e:
        return f"<div style='color:red'>Error: {e}</div>"


def check_health_sync() -> str:
    """Check system health synchronously."""
    try:
        # Try to get existing event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, we can't run sync
            # Return a pending message - the async version will update it
            return f"<div style='color:#666'><em>Checking health...</em></div>"
        except RuntimeError:
            # No running loop - we can create one
            pass
        
        # Create new event loop for this thread
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(check_health_async())
            finally:
                loop.close()
        except Exception as e:
            logger.warning(f"Health check loop error: {e}")
            return f"<div style='color:#666'><em>Health check pending...</em></div>"
    except Exception as e:
        logger.warning(f"Health check error: {e}")
        return f"<div style='color:#666'><em>Health check pending...</em></div>"


def get_agent_status() -> Tuple[str, str]:
    """Get current agent status."""
    agent = get_agent()
    if not agent:
        return "⚠️ Agent not available", "📋 N/A"
    
    status = agent.get_status()
    
    if status["running"]:
        if status["current_task"]:
            status_text = f"🔄 Running: {status['current_task']['description'][:30]}..."
        else:
            status_text = "✅ Running (idle)"
    else:
        status_text = "⏸️ Stopped"
    
    queue_text = f"📋 Queue: {status['queue_size']} tasks"
    
    return status_text, queue_text


def change_autonomy_level(level: str) -> str:
    """Change the agent's autonomy level."""
    global _agent_instance
    
    if not AUTONOMOUS_AVAILABLE:
        return "⚠️ Not available"
    
    # Recreate agent with new level
    _agent_instance = create_agent(level=level)
    
    level_emoji = {
        "readonly": "🔒",
        "monitored": "📋",
        "assisted": "✋",
        "supervised": "👁️",
        "autonomous": "🤖",
    }
    
    return f"{level_emoji.get(level, '⚙️')} Autonomy: **{level.title()}**"


async def execute_recovery_action(action: str) -> str:
    """Execute a recovery action."""
    agent = get_agent()
    if not agent:
        return "Agent not available"
    
    try:
        if action == "restart_vllm":
            result = await agent.recovery.handle_server_failure("vllm")
            return f"{'✅' if result.success else '❌'} {result.message}"
        
        elif action == "clear_cache":
            result = await agent.recovery._clear_cache()
            return f"{'✅' if result.success else '❌'} {result.message}"
        
        elif action == "health_check":
            health = await agent.check_health()
            return f"System: {health.status.value}\n" + \
                   "\n".join(f"  {c.name}: {c.status.value}" for c in health.components)
        
        else:
            return f"Unknown action: {action}"
            
    except Exception as e:
        return f"Error: {e}"


def execute_recovery_sync(action: str) -> str:
    """Execute recovery action synchronously."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, execute_recovery_action(action))
                return future.result(timeout=60)
        else:
            return loop.run_until_complete(execute_recovery_action(action))
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# Full Panel Creator
# =============================================================================

def create_autonomous_panel(visible: bool = True) -> Dict[str, Any]:
    """
    Create the full autonomous agent panel.
    
    Returns:
        Dictionary of Gradio components
    """
    components = {}
    
    with gr.Accordion("🤖 Autonomous Agent", open=False, visible=visible) as panel:
        components["panel"] = panel
        
        # Autonomy level selector
        components["autonomy_dropdown"] = create_autonomy_selector()
        components["autonomy_status"] = gr.Markdown("✋ Autonomy: **Assisted**")
        
        gr.Markdown("---")
        
        # Health status
        gr.Markdown("### 🏥 System Health")
        components["health_display"], components["health_refresh"], components["health_timer"] = \
            create_health_panel()
        
        gr.Markdown("---")
        
        # Agent status
        gr.Markdown("### 📊 Agent Status")
        components["agent_status"], components["queue_status"] = create_agent_status_panel()
        
        with gr.Row():
            components["start_btn"] = gr.Button("▶️ Start", size="sm", variant="primary")
            components["stop_btn"] = gr.Button("⏹️ Stop", size="sm", variant="secondary")
        
        gr.Markdown("---")
        
        # Recovery controls
        with gr.Accordion("🔧 Recovery Actions", open=False):
            components["recovery_dropdown"], components["recovery_btn"], components["recovery_result"] = \
                create_recovery_controls()
    
    return components


def setup_autonomous_events(components: Dict[str, Any]) -> None:
    """
    Set up event handlers for autonomous panel components.
    
    Args:
        components: Dictionary of Gradio components from create_autonomous_panel
    """
    # Autonomy level change
    components["autonomy_dropdown"].change(
        fn=change_autonomy_level,
        inputs=[components["autonomy_dropdown"]],
        outputs=[components["autonomy_status"]],
    )
    
    # Health check button
    components["health_refresh"].click(
        fn=check_health_sync,
        outputs=[components["health_display"]],
    )
    
    # Health timer
    components["health_timer"].tick(
        fn=check_health_sync,
        outputs=[components["health_display"]],
    )
    
    # Agent start/stop
    def start_agent():
        agent = get_agent()
        if agent:
            asyncio.get_event_loop().run_until_complete(agent.start_loop())
            return "✅ Agent started"
        return "⚠️ Could not start agent"
    
    def stop_agent():
        agent = get_agent()
        if agent:
            asyncio.get_event_loop().run_until_complete(agent.stop_loop())
            return "⏹️ Agent stopped"
        return "⚠️ Agent not running"
    
    components["start_btn"].click(
        fn=start_agent,
        outputs=[components["agent_status"]],
    )
    
    components["stop_btn"].click(
        fn=stop_agent,
        outputs=[components["agent_status"]],
    )
    
    # Recovery action
    components["recovery_btn"].click(
        fn=execute_recovery_sync,
        inputs=[components["recovery_dropdown"]],
        outputs=[components["recovery_result"]],
    )


# =============================================================================
# Minimal Health Widget for Sidebar
# =============================================================================

def create_health_widget() -> Tuple[gr.Markdown, gr.Timer]:
    """
    Create a minimal health widget for the sidebar.
    
    Returns:
        (health_display, timer)
    """
    def get_minimal_health() -> str:
        try:
            checker = get_health_checker()
            if not checker:
                return "⚪ Health: *checking...*"
            
            # Try to get cached health
            health = checker.get_last_health()
            if health:
                return format_health_status_minimal(health)
            
            return "⚪ Health: *pending...*"
        except Exception:
            return "⚪ Health: *unknown*"
    
    health_display = gr.Markdown(get_minimal_health())
    timer = gr.Timer(30, active=True)
    
    return health_display, timer


# =============================================================================
# Integration with Chat Handler
# =============================================================================

class AutonomousChatHandler:
    """
    Chat handler with autonomous agent capabilities.
    
    Extends the basic chat with:
    - Autonomous task execution
    - Real-time health monitoring
    - Automatic recovery
    """
    
    def __init__(
        self,
        vllm_url: str = "http://localhost:8000/v1",
        model: str = "MiniMaxAI/MiniMax-M2",
        autonomy_level: str = "assisted",
    ):
        self.vllm_url = vllm_url
        self.model = model
        self.autonomy_level = autonomy_level
        
        self._agent = None
        self._confirmation_queue: List[Tuple[str, Any]] = []
    
    @property
    def agent(self) -> Optional["AutonomousAgent"]:
        """Get or create agent."""
        if self._agent is None:
            self._agent = get_agent(self.autonomy_level)
            
            # Set up confirmation callback
            if self._agent:
                self._agent._confirmation_callback = self._handle_confirmation
                self._agent._response_callback = self._handle_response
        
        return self._agent
    
    def _handle_confirmation(self, message: str, action: Any) -> bool:
        """Handle confirmation requests from agent."""
        # In Gradio, we need to queue this and handle it in the UI
        self._confirmation_queue.append((message, action))
        # For now, default to requiring confirmation = deny
        return False
    
    def _handle_response(self, response: str) -> None:
        """Handle agent responses."""
        # This would stream to the chat
        pass
    
    def chat(
        self,
        message: str,
        history: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """
        Process a chat message with autonomous capabilities.
        
        Args:
            message: User message
            history: Chat history
            context: Additional context
        
        Yields:
            Response chunks
        """
        # Check for autonomous commands
        message_lower = message.lower().strip()
        
        # Health check command
        if message_lower in ["health", "status", "health check", "check health"]:
            yield "🏥 **System Health Check**\n\n"
            health = check_health_sync()
            yield health
            return
        
        # Recovery commands
        if message_lower.startswith("recover") or message_lower.startswith("fix"):
            yield "🔧 **Initiating Recovery**\n\n"
            
            if "vllm" in message_lower or "server" in message_lower:
                result = execute_recovery_sync("restart_vllm")
            else:
                result = execute_recovery_sync("health_check")
            
            yield result
            return
        
        # Agent mode commands
        if message_lower == "start agent":
            if self.agent:
                asyncio.get_event_loop().run_until_complete(self.agent.start_loop())
                yield "✅ Autonomous agent started. I'll now monitor jobs and apply fixes automatically."
            else:
                yield "⚠️ Agent not available."
            return
        
        if message_lower == "stop agent":
            if self.agent:
                asyncio.get_event_loop().run_until_complete(self.agent.stop_loop())
                yield "⏹️ Autonomous agent stopped."
            else:
                yield "⚠️ Agent not running."
            return
        
        # For other messages, use the standard chat flow
        # This would integrate with the existing chat handler
        yield "Processing with enhanced agent...\n\n"
        
        # Fall through to standard processing
        from workflow_composer.agents.chat_integration import get_chat_handler
        handler = get_chat_handler()
        yield from handler.chat(message, history, context)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "AUTONOMOUS_AVAILABLE",
    "get_agent",
    "get_health_checker",
    "create_autonomous_panel",
    "setup_autonomous_events",
    "create_health_widget",
    "create_autonomy_selector",
    "AutonomousChatHandler",
    "check_health_sync",
    "format_health_status",
]
