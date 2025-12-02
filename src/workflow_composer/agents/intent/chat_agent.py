"""
Professional Chat Agent Integration Layer.

Phase 8 of Professional Chat Agent implementation.

This module integrates all professional chat agent components:
- Dialog State Machine (Phase 1)
- Response Generation (Phase 2)  
- Conversation Analytics (Phase 3)
- Human Handoff (Phase 4)
- A/B Testing (Phase 5)
- Rich Responses (Phase 6)
- Out-of-Scope Detection (Phase 7)

Features:
- Unified ChatAgent interface
- Component orchestration
- Session management
- Multi-channel support
- Plugin architecture
"""

import asyncio
import logging
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

# Import all professional chat agent components
from .dialog_state_machine import (
    DialogStateMachine,
    DialogState,
    DialogEvent,
    DialogContext,
    get_dialog_state_machine,
    create_dialog_state_machine,
    reset_dialog_state_machine,
)

from .response_generator import (
    ResponseGenerator,
    Response,
    ResponseType,
    ResponseTone,
    create_response_generator,
)

from .conversation_analytics import (
    ConversationAnalytics,
    ConversationMetrics,
    ConversationOutcome,
    get_conversation_analytics,
    reset_analytics,
)

from .human_handoff import (
    HumanHandoffManager,
    HandoffRequest,
    HandoffStatus,
    EscalationReason,
    get_handoff_manager,
    reset_handoff_manager,
)

from .ab_testing import (
    ABTestingManager,
    Experiment,
    Variant,
    get_ab_testing_manager,
    reset_ab_testing_manager,
)

from .rich_responses import (
    MessageFormatter,
    MessageFormat,
    get_message_formatter,
    reset_message_formatter,
)

from .out_of_scope import (
    OutOfScopeHandler,
    ScopeResult,
    DeflectionResponse,
    get_out_of_scope_handler,
    reset_out_of_scope_handler,
)

# Import UnifiedAgent for tool execution
# Late import to avoid circular dependency - done in _initialize_unified_agent

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class ChannelType(Enum):
    """Supported communication channels."""
    WEB = "web"
    SLACK = "slack"
    TEAMS = "teams"
    API = "api"
    WEBHOOK = "webhook"
    CLI = "cli"


class MessageDirection(Enum):
    """Direction of message flow."""
    INCOMING = "incoming"
    OUTGOING = "outgoing"


class AgentCapability(Enum):
    """Agent capabilities that can be enabled/disabled."""
    DIALOG_STATE = "dialog_state"
    ANALYTICS = "analytics"
    HANDOFF = "handoff"
    AB_TESTING = "ab_testing"
    RICH_RESPONSES = "rich_responses"
    SCOPE_DETECTION = "scope_detection"
    SESSION_MEMORY = "session_memory"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Message:
    """Represents a chat message."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    direction: MessageDirection = MessageDirection.INCOMING
    channel: ChannelType = ChannelType.WEB
    session_id: str = ""
    user_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "direction": self.direction.value,
            "channel": self.channel.value,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class Session:
    """Represents a chat session."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    channel: ChannelType = ChannelType.WEB
    started_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    messages: List[Message] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    state: str = "idle"
    is_active: bool = True
    assigned_variant: Optional[str] = None
    
    def add_message(self, message: Message) -> None:
        """Add a message to the session."""
        message.session_id = self.id
        self.messages.append(message)
        self.last_activity = datetime.now()
    
    def get_conversation_history(self, limit: int = 10) -> List[Message]:
        """Get recent conversation history."""
        return self.messages[-limit:]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "channel": self.channel.value,
            "started_at": self.started_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "message_count": len(self.messages),
            "state": self.state,
            "is_active": self.is_active,
            "assigned_variant": self.assigned_variant,
        }


@dataclass
class AgentResponse:
    """Complete agent response with all metadata."""
    message: Message
    intent_detected: Optional[str] = None
    entities_extracted: Dict[str, Any] = field(default_factory=dict)
    state_transition: Optional[str] = None
    scope_result: Optional[ScopeResult] = None
    requires_handoff: bool = False
    handoff_request: Optional[HandoffRequest] = None
    suggestions: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message": self.message.to_dict(),
            "intent_detected": self.intent_detected,
            "entities_extracted": self.entities_extracted,
            "state_transition": self.state_transition,
            "scope_in_scope": self.scope_result.is_in_scope if self.scope_result else None,
            "requires_handoff": self.requires_handoff,
            "suggestions": self.suggestions,
            "processing_time_ms": self.processing_time_ms,
        }


@dataclass
class AgentConfig:
    """Configuration for the chat agent."""
    name: str = "BioPipelines Assistant"
    description: str = "AI assistant for bioinformatics workflows"
    enabled_capabilities: Set[AgentCapability] = field(
        default_factory=lambda: {cap for cap in AgentCapability}
    )
    default_channel: ChannelType = ChannelType.WEB
    default_format: MessageFormat = MessageFormat.MARKDOWN
    session_timeout_minutes: int = 30
    max_messages_per_session: int = 100
    enable_logging: bool = True
    strict_scope_mode: bool = False
    
    def is_enabled(self, capability: AgentCapability) -> bool:
        """Check if a capability is enabled."""
        return capability in self.enabled_capabilities


# =============================================================================
# Plugin Architecture
# =============================================================================

class AgentPlugin(ABC):
    """Base class for agent plugins."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass
    
    @abstractmethod
    def on_message_received(
        self,
        message: Message,
        session: Session,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Called when a message is received.
        
        Returns:
            Optional context updates
        """
        pass
    
    @abstractmethod
    def on_response_generated(
        self,
        response: AgentResponse,
        session: Session,
        context: Dict[str, Any]
    ) -> Optional[AgentResponse]:
        """
        Called after response is generated.
        
        Returns:
            Modified response or None
        """
        pass


class LoggingPlugin(AgentPlugin):
    """Plugin for logging all interactions."""
    
    @property
    def name(self) -> str:
        return "logging"
    
    def on_message_received(
        self,
        message: Message,
        session: Session,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        logger.info(f"[{session.id}] Received: {message.content[:100]}")
        return None
    
    def on_response_generated(
        self,
        response: AgentResponse,
        session: Session,
        context: Dict[str, Any]
    ) -> Optional[AgentResponse]:
        logger.info(
            f"[{session.id}] Response ({response.processing_time_ms:.1f}ms): "
            f"{response.message.content[:100]}"
        )
        return None


class MetricsPlugin(AgentPlugin):
    """Plugin for collecting metrics."""
    
    def __init__(self):
        self.message_count = 0
        self.total_processing_time = 0.0
    
    @property
    def name(self) -> str:
        return "metrics"
    
    def on_message_received(
        self,
        message: Message,
        session: Session,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        self.message_count += 1
        return {"message_number": self.message_count}
    
    def on_response_generated(
        self,
        response: AgentResponse,
        session: Session,
        context: Dict[str, Any]
    ) -> Optional[AgentResponse]:
        self.total_processing_time += response.processing_time_ms
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get plugin statistics."""
        return {
            "message_count": self.message_count,
            "total_processing_time_ms": self.total_processing_time,
            "avg_processing_time_ms": (
                self.total_processing_time / self.message_count 
                if self.message_count > 0 else 0
            ),
        }


# =============================================================================
# Session Manager
# =============================================================================

class SessionManager:
    """Manages chat sessions."""
    
    def __init__(self, timeout_minutes: int = 30):
        self._sessions: Dict[str, Session] = {}
        self._user_sessions: Dict[str, str] = {}  # user_id -> session_id
        self._timeout_minutes = timeout_minutes
        self._lock = threading.Lock()
    
    def create_session(
        self,
        user_id: str = "",
        channel: ChannelType = ChannelType.WEB
    ) -> Session:
        """Create a new session."""
        session = Session(
            user_id=user_id,
            channel=channel
        )
        
        with self._lock:
            self._sessions[session.id] = session
            if user_id:
                self._user_sessions[user_id] = session.id
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        return self._sessions.get(session_id)
    
    def get_session_for_user(self, user_id: str) -> Optional[Session]:
        """Get active session for a user."""
        session_id = self._user_sessions.get(user_id)
        if session_id:
            session = self._sessions.get(session_id)
            if session and session.is_active:
                return session
        return None
    
    def get_or_create_session(
        self,
        user_id: str = "",
        channel: ChannelType = ChannelType.WEB
    ) -> Session:
        """Get existing session or create new one."""
        if user_id:
            session = self.get_session_for_user(user_id)
            if session:
                return session
        return self.create_session(user_id, channel)
    
    def end_session(self, session_id: str) -> None:
        """End a session."""
        session = self._sessions.get(session_id)
        if session:
            session.is_active = False
            if session.user_id and session.user_id in self._user_sessions:
                del self._user_sessions[session.user_id]
    
    def cleanup_expired(self) -> int:
        """Clean up expired sessions."""
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(minutes=self._timeout_minutes)
        expired_ids = []
        
        with self._lock:
            for session_id, session in self._sessions.items():
                if session.last_activity < cutoff:
                    expired_ids.append(session_id)
            
            for session_id in expired_ids:
                self.end_session(session_id)
        
        return len(expired_ids)
    
    def get_active_count(self) -> int:
        """Get count of active sessions."""
        return sum(1 for s in self._sessions.values() if s.is_active)
    
    def get_all_sessions(self) -> List[Session]:
        """Get all sessions."""
        return list(self._sessions.values())


# =============================================================================
# Intent Handler Registry
# =============================================================================

IntentHandler = Callable[[Message, Session, Dict[str, Any]], AgentResponse]


class IntentRegistry:
    """Registry for intent handlers."""
    
    def __init__(self):
        self._handlers: Dict[str, IntentHandler] = {}
        self._default_handler: Optional[IntentHandler] = None
    
    def register(
        self,
        intent: str,
        handler: IntentHandler
    ) -> None:
        """Register a handler for an intent."""
        self._handlers[intent] = handler
    
    def set_default(self, handler: IntentHandler) -> None:
        """Set the default handler."""
        self._default_handler = handler
    
    def get_handler(self, intent: str) -> Optional[IntentHandler]:
        """Get handler for an intent."""
        return self._handlers.get(intent, self._default_handler)
    
    def has_handler(self, intent: str) -> bool:
        """Check if intent has a handler."""
        return intent in self._handlers


# =============================================================================
# Main Chat Agent
# =============================================================================

class ChatAgent:
    """
    Professional Chat Agent integrating all components.
    
    This is the main interface for the professional chat agent system.
    It orchestrates all components from Phases 1-7 and delegates
    tool execution to UnifiedAgent.
    
    Architecture:
        User Query → ChatAgent (professional features)
            → Scope Detection → Out-of-scope deflection
            → Dialog State Machine → Conversation flow
            → A/B Testing → Variant assignment
            → UnifiedAgent (tool execution) → Actual work
            → Response Generation → Template + tone
            → Rich Formatting → Cards, buttons
            → Analytics → Record metrics
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        
        # Initialize core components
        self._session_manager = SessionManager(
            timeout_minutes=self.config.session_timeout_minutes
        )
        self._intent_registry = IntentRegistry()
        self._plugins: List[AgentPlugin] = []
        
        # Initialize professional components based on config
        self._dialog_state: Optional[DialogStateMachine] = None
        self._response_generator: Optional[ResponseGenerator] = None
        self._analytics: Optional[ConversationAnalytics] = None
        self._handoff_manager: Optional[HumanHandoffManager] = None
        self._ab_testing: Optional[ABTestingManager] = None
        self._message_formatter: Optional[MessageFormatter] = None
        self._scope_handler: Optional[OutOfScopeHandler] = None
        
        # UnifiedAgent for tool execution (lazy initialized)
        self._unified_agent = None
        
        self._initialize_components()
        
        # Statistics
        self._stats = {
            "messages_processed": 0,
            "sessions_created": 0,
            "handoffs_initiated": 0,
            "out_of_scope_queries": 0,
            "tools_executed": 0,
        }
        
        self._lock = threading.Lock()
    
    def _initialize_components(self) -> None:
        """Initialize enabled components."""
        if self.config.is_enabled(AgentCapability.DIALOG_STATE):
            reset_dialog_state_machine()
            self._dialog_state = get_dialog_state_machine()
        
        if self.config.is_enabled(AgentCapability.ANALYTICS):
            reset_analytics()
            self._analytics = get_conversation_analytics()
        
        if self.config.is_enabled(AgentCapability.HANDOFF):
            reset_handoff_manager()
            self._handoff_manager = get_handoff_manager()
        
        if self.config.is_enabled(AgentCapability.AB_TESTING):
            reset_ab_testing_manager()
            self._ab_testing = get_ab_testing_manager()
        
        if self.config.is_enabled(AgentCapability.RICH_RESPONSES):
            reset_message_formatter()
            self._message_formatter = get_message_formatter()
        
        if self.config.is_enabled(AgentCapability.SCOPE_DETECTION):
            reset_out_of_scope_handler()
            self._scope_handler = get_out_of_scope_handler()
        
        # Always create response generator
        self._response_generator = create_response_generator()
    
    def _get_unified_agent(self):
        """Get or create UnifiedAgent for tool execution (lazy loading)."""
        if self._unified_agent is None:
            try:
                from ..unified_agent import UnifiedAgent, AutonomyLevel
                self._unified_agent = UnifiedAgent(
                    autonomy_level=AutonomyLevel.ASSISTED
                )
                logger.info("UnifiedAgent initialized for tool execution")
            except ImportError as e:
                logger.warning(f"Could not import UnifiedAgent: {e}")
            except Exception as e:
                logger.warning(f"Could not initialize UnifiedAgent: {e}")
        return self._unified_agent
    
    async def _execute_via_unified_agent(self, query: str) -> Optional[Dict[str, Any]]:
        """Execute a query through UnifiedAgent for tool execution."""
        agent = self._get_unified_agent()
        if agent is None:
            return None
        
        try:
            result = await agent.process_query(query)
            with self._lock:
                self._stats["tools_executed"] += len(result.tool_executions) if result.tool_executions else 0
            return {
                "success": result.success,
                "message": result.message,
                "task_type": result.task_type.value if result.task_type else None,
                "data": result.data,
                "suggestions": result.suggestions,
                "tools_executed": [te.tool_name for te in result.tool_executions] if result.tool_executions else [],
            }
        except Exception as e:
            logger.error(f"UnifiedAgent execution error: {e}")
            return None
    
    def _execute_via_unified_agent_sync(self, query: str) -> Optional[Dict[str, Any]]:
        """Synchronous wrapper for UnifiedAgent execution."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new loop in a thread for sync execution
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, 
                        self._execute_via_unified_agent(query)
                    )
                    return future.result(timeout=60)
            else:
                return loop.run_until_complete(self._execute_via_unified_agent(query))
        except Exception as e:
            logger.error(f"Sync UnifiedAgent execution error: {e}")
            return None
    
    # =========================================================================
    # Plugin Management
    # =========================================================================
    
    def add_plugin(self, plugin: AgentPlugin) -> None:
        """Add a plugin to the agent."""
        self._plugins.append(plugin)
        logger.info(f"Added plugin: {plugin.name}")
    
    def remove_plugin(self, plugin_name: str) -> bool:
        """Remove a plugin by name."""
        for i, plugin in enumerate(self._plugins):
            if plugin.name == plugin_name:
                self._plugins.pop(i)
                logger.info(f"Removed plugin: {plugin_name}")
                return True
        return False
    
    # =========================================================================
    # Intent Registration
    # =========================================================================
    
    def register_intent(
        self,
        intent: str,
        handler: IntentHandler
    ) -> None:
        """Register a handler for an intent."""
        self._intent_registry.register(intent, handler)
    
    def set_default_handler(self, handler: IntentHandler) -> None:
        """Set the default intent handler."""
        self._intent_registry.set_default(handler)
    
    # =========================================================================
    # Message Processing
    # =========================================================================
    
    def process_message(
        self,
        content: str,
        session_id: Optional[str] = None,
        user_id: str = "",
        channel: ChannelType = ChannelType.WEB,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Process an incoming message and generate a response.
        
        This is the main entry point for message processing.
        
        Args:
            content: The message content
            session_id: Optional session ID
            user_id: Optional user ID
            channel: Communication channel
            metadata: Additional metadata
        
        Returns:
            AgentResponse with the response and metadata
        """
        start_time = datetime.now()
        
        # Get or create session
        if session_id:
            session = self._session_manager.get_session(session_id)
            if not session:
                session = self._session_manager.create_session(user_id, channel)
        else:
            session = self._session_manager.get_or_create_session(user_id, channel)
        
        # Create incoming message
        message = Message(
            content=content,
            direction=MessageDirection.INCOMING,
            channel=channel,
            user_id=user_id,
            metadata=metadata or {}
        )
        session.add_message(message)
        
        # Build processing context
        context: Dict[str, Any] = {
            "session": session,
            "user_id": user_id,
            "channel": channel,
            "history": session.get_conversation_history(),
        }
        
        # Execute plugin pre-processing
        for plugin in self._plugins:
            try:
                updates = plugin.on_message_received(message, session, context)
                if updates:
                    context.update(updates)
            except Exception as e:
                logger.error(f"Plugin {plugin.name} error: {e}")
        
        # Process through components
        response = self._process_through_components(message, session, context)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        response.processing_time_ms = processing_time
        
        # Execute plugin post-processing
        for plugin in self._plugins:
            try:
                modified = plugin.on_response_generated(response, session, context)
                if modified:
                    response = modified
            except Exception as e:
                logger.error(f"Plugin {plugin.name} error: {e}")
        
        # Add response to session
        session.add_message(response.message)
        
        # Update statistics
        with self._lock:
            self._stats["messages_processed"] += 1
            if response.requires_handoff:
                self._stats["handoffs_initiated"] += 1
            if response.scope_result and not response.scope_result.is_in_scope:
                self._stats["out_of_scope_queries"] += 1
        
        # Record analytics
        if self._analytics:
            # Use record_turn instead of record_message
            self._analytics.record_turn(
                session.id,
                response_time=processing_time,
                intent=response.intent_detected
            )
        
        return response
    
    def _process_through_components(
        self,
        message: Message,
        session: Session,
        context: Dict[str, Any]
    ) -> AgentResponse:
        """Process message through all enabled components."""
        
        response_content = ""
        intent_detected = None
        scope_result = None
        requires_handoff = False
        handoff_request = None
        suggestions: List[str] = []
        state_transition = None
        
        # Step 1: Scope Detection
        if self._scope_handler:
            scope_result, deflection = self._scope_handler.handle_query(
                message.content,
                context
            )
            
            if not scope_result.is_in_scope:
                # Out of scope - return deflection
                if deflection:
                    response_content = deflection.message
                    suggestions = deflection.suggestions
                else:
                    response_content = (
                        "I'm sorry, but that question is outside my area of expertise. "
                        "I can help you with bioinformatics workflows and analyses."
                    )
                
                return self._create_response(
                    response_content,
                    message,
                    session,
                    intent_detected=None,
                    scope_result=scope_result,
                    suggestions=suggestions
                )
        
        # Step 2: A/B Testing - assign variant if enabled
        if self._ab_testing and session.assigned_variant is None:
            # Check for running experiments using list_experiments
            from .ab_testing import ExperimentStatus
            experiments = self._ab_testing.list_experiments(ExperimentStatus.RUNNING)
            if experiments:
                exp = experiments[0]  # Use first active experiment
                assignment = self._ab_testing.assign_variant(
                    exp.id,
                    session.user_id or session.id
                )
                if assignment:
                    session.assigned_variant = assignment.variant_id
                    context["variant"] = assignment.variant_id
        
        # Step 3: Dialog State Machine processing
        if self._dialog_state:
            # Determine event from message
            event = self._determine_event(message, context)
            
            # Create a proper DialogContext for the state machine
            dialog_context = DialogContext(
                query=message.content,
                session_id=session.id,
                turn_count=len(session.get_conversation_history()),
            )
            
            # Process through state machine
            transition = self._dialog_state.process_event(event, dialog_context)
            if transition:
                state_transition = transition.value
                session.state = state_transition
        
        # Step 4: Intent Detection (simplified)
        intent_detected = self._detect_intent(message.content, context)
        
        # Step 5: Check if handoff is needed
        if self._handoff_manager:
            # Check escalation conditions
            if self._should_escalate(message, session, context):
                handoff_request = self._handoff_manager.request_handoff(
                    conversation_id=session.id,
                    reason=EscalationReason.USER_REQUEST,
                    context={
                        "user_id": session.user_id,
                        "message": message.content,
                        "history": [m.content for m in session.get_conversation_history()],
                    }
                )
                requires_handoff = True
                response_content = (
                    "I'm connecting you with a human specialist. "
                    "Please wait a moment."
                )
                
                return self._create_response(
                    response_content,
                    message,
                    session,
                    intent_detected=intent_detected,
                    scope_result=scope_result,
                    requires_handoff=True,
                    handoff_request=handoff_request
                )
        
        # Step 6: Generate response based on intent
        handler = self._intent_registry.get_handler(intent_detected or "unknown")
        if handler:
            return handler(message, session, context)
        
        # Step 7: Execute through UnifiedAgent for tool-based queries
        # This delegates actual tool execution (workflow gen, data scan, etc.)
        unified_result = None
        if self._is_tool_query(message.content, intent_detected):
            unified_result = self._execute_via_unified_agent_sync(message.content)
            if unified_result and unified_result.get("success"):
                response_content = unified_result.get("message", "")
                suggestions = unified_result.get("suggestions", [])
                context["unified_result"] = unified_result
        
        # Step 8: Default response generation (if no tool result)
        if not response_content:
            if self._response_generator:
                response = self._response_generator.generate(
                    intent=intent_detected or "default",
                    context={
                        "intent": intent_detected,
                        "message": message.content,
                        "history": context.get("history", []),
                    }
                )
                response_content = response.primary_text
            else:
                response_content = self._generate_default_response(message, context)
        
        # Step 9: Format response if rich responses enabled
        if self._message_formatter:
            # Could add rich formatting here based on context
            pass
        
        return self._create_response(
            response_content,
            message,
            session,
            intent_detected=intent_detected,
            scope_result=scope_result,
            state_transition=state_transition,
            suggestions=suggestions
        )
    
    def _create_response(
        self,
        content: str,
        original_message: Message,
        session: Session,
        intent_detected: Optional[str] = None,
        scope_result: Optional[ScopeResult] = None,
        state_transition: Optional[str] = None,
        requires_handoff: bool = False,
        handoff_request: Optional[HandoffRequest] = None,
        suggestions: Optional[List[str]] = None
    ) -> AgentResponse:
        """Create an agent response."""
        response_message = Message(
            content=content,
            direction=MessageDirection.OUTGOING,
            channel=original_message.channel,
            session_id=session.id,
            user_id="agent"
        )
        
        return AgentResponse(
            message=response_message,
            intent_detected=intent_detected,
            scope_result=scope_result,
            state_transition=state_transition,
            requires_handoff=requires_handoff,
            handoff_request=handoff_request,
            suggestions=suggestions or []
        )
    
    def _determine_event(
        self,
        message: Message,
        context: Dict[str, Any]
    ) -> DialogEvent:
        """Determine dialog event from message."""
        content_lower = message.content.lower()
        
        if any(word in content_lower for word in ["yes", "confirm", "correct", "okay", "ok"]):
            return DialogEvent.CONFIRM
        elif any(word in content_lower for word in ["no", "cancel", "stop", "wrong"]):
            return DialogEvent.CANCEL
        elif any(word in content_lower for word in ["help", "support", "human", "agent"]):
            return DialogEvent.ESCALATE
        else:
            return DialogEvent.USER_INPUT
    
    def _detect_intent(
        self,
        content: str,
        context: Dict[str, Any]
    ) -> str:
        """Detect intent from message content."""
        content_lower = content.lower()
        
        # Simple rule-based intent detection
        if any(word in content_lower for word in ["create", "build", "run", "generate"]):
            if "workflow" in content_lower or "pipeline" in content_lower:
                return "create_workflow"
        
        if any(word in content_lower for word in ["help", "what can", "how do"]):
            return "get_help"
        
        if any(word in content_lower for word in ["status", "check", "progress"]):
            return "check_status"
        
        if any(word in content_lower for word in ["hello", "hi", "hey"]):
            return "greeting"
        
        if any(word in content_lower for word in ["bye", "goodbye", "thanks"]):
            return "farewell"
        
        return "unknown"
    
    def _should_escalate(
        self,
        message: Message,
        session: Session,
        context: Dict[str, Any]
    ) -> bool:
        """Determine if conversation should be escalated."""
        content_lower = message.content.lower()
        
        # Check for explicit escalation requests
        if any(word in content_lower for word in [
            "human", "agent", "representative", "speak to someone",
            "real person", "support"
        ]):
            return True
        
        # Check for repeated unknown intents
        recent_messages = session.get_conversation_history(5)
        unknown_count = sum(
            1 for m in recent_messages 
            if m.direction == MessageDirection.INCOMING
        )
        
        # Escalate if too many messages without resolution
        if unknown_count >= 5:
            return True
        
        return False
    
    def _is_tool_query(self, content: str, intent: Optional[str]) -> bool:
        """
        Determine if the query should be executed through UnifiedAgent.
        
        Tool queries include:
        - Workflow generation/creation
        - Data scanning/searching
        - Job submission/status
        - Reference downloads
        - Error diagnosis
        """
        content_lower = content.lower()
        
        # Check for tool-related keywords
        tool_keywords = [
            # Workflow operations
            "workflow", "pipeline", "generate", "create", "build",
            # Data operations  
            "scan", "find files", "search", "download", "dataset",
            # Job operations
            "submit", "job status", "running jobs", "cancel job",
            # Reference operations
            "reference", "genome", "index",
            # Analysis operations
            "analyze", "results", "visualize",
            # Diagnosis
            "error", "diagnose", "debug", "fix",
        ]
        
        if any(kw in content_lower for kw in tool_keywords):
            return True
        
        # Check intent
        tool_intents = ["create_workflow", "check_status", "scan_data", "submit_job"]
        if intent and intent in tool_intents:
            return True
        
        return False
    
    def _generate_default_response(
        self,
        message: Message,
        context: Dict[str, Any]
    ) -> str:
        """Generate a default response."""
        return (
            "I understand you're asking about bioinformatics workflows. "
            "Could you please provide more details about what you'd like to accomplish? "
            "For example, I can help you:\n"
            "- Create RNA-seq analysis pipelines\n"
            "- Set up variant calling workflows\n"
            "- Configure ChIP-seq analysis\n"
            "- Run methylation analysis"
        )
    
    # =========================================================================
    # Session Management
    # =========================================================================
    
    def create_session(
        self,
        user_id: str = "",
        channel: ChannelType = ChannelType.WEB
    ) -> Session:
        """Create a new session."""
        session = self._session_manager.create_session(user_id, channel)
        
        with self._lock:
            self._stats["sessions_created"] += 1
        
        # Start analytics tracking
        if self._analytics:
            self._analytics.start_conversation(session.id)
        
        return session
    
    def end_session(self, session_id: str) -> None:
        """End a session."""
        session = self._session_manager.get_session(session_id)
        if session:
            # End analytics tracking
            if self._analytics:
                self._analytics.end_conversation(
                    session_id,
                    ConversationOutcome.SUCCESS
                )
            
            self._session_manager.end_session(session_id)
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        return self._session_manager.get_session(session_id)
    
    # =========================================================================
    # Statistics and Health
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        with self._lock:
            stats = dict(self._stats)
        
        stats["active_sessions"] = self._session_manager.get_active_count()
        
        # Add component stats
        if self._analytics:
            try:
                dashboard = self._analytics.get_summary()
                stats["analytics"] = dashboard
            except Exception:
                pass
        
        if self._handoff_manager:
            try:
                stats["handoff"] = self._handoff_manager.get_metrics().to_dict()
            except Exception:
                pass
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check."""
        health = {
            "status": "healthy",
            "components": {},
            "timestamp": datetime.now().isoformat(),
        }
        
        # Check each component
        components = [
            ("dialog_state", self._dialog_state),
            ("response_generator", self._response_generator),
            ("analytics", self._analytics),
            ("handoff_manager", self._handoff_manager),
            ("ab_testing", self._ab_testing),
            ("message_formatter", self._message_formatter),
            ("scope_handler", self._scope_handler),
        ]
        
        for name, component in components:
            health["components"][name] = {
                "enabled": component is not None,
                "status": "ok" if component else "disabled"
            }
        
        return health
    
    def reset(self) -> None:
        """Reset the agent state."""
        # Reset all components
        self._initialize_components()
        
        # Clear sessions
        for session in self._session_manager.get_all_sessions():
            self._session_manager.end_session(session.id)
        
        # Reset stats
        with self._lock:
            self._stats = {
                "messages_processed": 0,
                "sessions_created": 0,
                "handoffs_initiated": 0,
                "out_of_scope_queries": 0,
            }
        
        logger.info("Agent reset complete")


# =============================================================================
# Singleton and Factory Functions
# =============================================================================

_agent: Optional[ChatAgent] = None
_agent_lock = threading.Lock()


def get_chat_agent(config: Optional[AgentConfig] = None) -> ChatAgent:
    """Get or create the singleton chat agent."""
    global _agent
    
    with _agent_lock:
        if _agent is None:
            _agent = ChatAgent(config)
    
    return _agent


def reset_chat_agent() -> None:
    """Reset the singleton chat agent."""
    global _agent
    
    with _agent_lock:
        if _agent:
            _agent.reset()
        _agent = None


def create_chat_agent(config: Optional[AgentConfig] = None) -> ChatAgent:
    """Create a new chat agent instance (non-singleton)."""
    return ChatAgent(config)


# Convenience function
def quick_chat(message: str, user_id: str = "") -> str:
    """Quick chat function for simple interactions."""
    agent = get_chat_agent()
    response = agent.process_message(message, user_id=user_id)
    return response.message.content


__all__ = [
    # Enums
    "ChannelType",
    "MessageDirection",
    "AgentCapability",
    
    # Data classes
    "Message",
    "Session",
    "AgentResponse",
    "AgentConfig",
    
    # Plugin system
    "AgentPlugin",
    "LoggingPlugin",
    "MetricsPlugin",
    
    # Session management
    "SessionManager",
    
    # Intent handling
    "IntentRegistry",
    "IntentHandler",
    
    # Main agent
    "ChatAgent",
    
    # Singleton functions
    "get_chat_agent",
    "reset_chat_agent",
    "create_chat_agent",
    
    # Convenience
    "quick_chat",
]
