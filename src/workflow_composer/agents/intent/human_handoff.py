"""
Human Handoff System for Professional Chat Agent.

This module provides sophisticated human escalation with:
- Automatic escalation triggers based on conversation signals
- Smooth handoff protocol with context preservation
- Queue management for human agents
- Handoff analytics and tracking
- Configurable escalation rules
- Agent availability detection

Phase 4 of the Professional Chat Agent implementation.
"""

import json
import logging
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from queue import PriorityQueue

from .dialog_state_machine import DialogState, DialogEvent
from .conversation_analytics import ConversationMetrics, ConversationOutcome


logger = logging.getLogger(__name__)


class EscalationReason(Enum):
    """Reasons for escalating to human agent."""
    USER_REQUESTED = "user_requested"
    LOW_CONFIDENCE = "low_confidence"
    REPEATED_ERRORS = "repeated_errors"
    COMPLEX_QUERY = "complex_query"
    SENTIMENT_NEGATIVE = "sentiment_negative"
    TOPIC_SENSITIVE = "topic_sensitive"
    LOOP_DETECTED = "loop_detected"
    TIMEOUT_EXCEEDED = "timeout_exceeded"
    OUT_OF_SCOPE = "out_of_scope"
    POLICY_VIOLATION = "policy_violation"
    TECHNICAL_ISSUE = "technical_issue"
    CUSTOM = "custom"


class HandoffStatus(Enum):
    """Status of a handoff request."""
    PENDING = "pending"
    QUEUED = "queued"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETURNED_TO_BOT = "returned_to_bot"


class AgentStatus(Enum):
    """Status of a human agent."""
    AVAILABLE = "available"
    BUSY = "busy"
    AWAY = "away"
    OFFLINE = "offline"


class Priority(Enum):
    """Handoff priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


@dataclass
class EscalationTrigger:
    """Configuration for an escalation trigger."""
    reason: EscalationReason
    condition: Callable[[Dict[str, Any]], bool]
    priority: Priority = Priority.NORMAL
    enabled: bool = True
    description: str = ""
    cooldown_seconds: int = 0  # Minimum time between triggers
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate if this trigger should fire."""
        if not self.enabled:
            return False
        try:
            return self.condition(context)
        except Exception as e:
            logger.warning(f"Trigger evaluation error for {self.reason}: {e}")
            return False


@dataclass
class ConversationContext:
    """Context to pass during handoff."""
    session_id: str
    user_id: Optional[str] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    detected_intents: List[str] = field(default_factory=list)
    extracted_entities: Dict[str, Any] = field(default_factory=dict)
    current_state: Optional[DialogState] = None
    error_history: List[str] = field(default_factory=list)
    user_sentiment: Optional[str] = None
    summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "conversation_history": self.conversation_history,
            "detected_intents": self.detected_intents,
            "extracted_entities": self.extracted_entities,
            "current_state": self.current_state.name if self.current_state else None,
            "error_history": self.error_history,
            "user_sentiment": self.user_sentiment,
            "summary": self.summary,
            "metadata": self.metadata,
        }


@dataclass
class HandoffRequest:
    """A request for human handoff."""
    id: str
    session_id: str
    reason: EscalationReason
    priority: Priority
    context: ConversationContext
    created_at: datetime = field(default_factory=datetime.now)
    status: HandoffStatus = HandoffStatus.PENDING
    assigned_agent_id: Optional[str] = None
    assigned_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    returned_to_bot: bool = False
    
    def __lt__(self, other: "HandoffRequest") -> bool:
        """Compare for priority queue ordering."""
        # Higher priority first, then older requests
        if self.priority != other.priority:
            return self.priority.value > other.priority.value
        return self.created_at < other.created_at
    
    @property
    def wait_time(self) -> float:
        """Get time waiting for assignment in seconds."""
        if self.assigned_at:
            return (self.assigned_at - self.created_at).total_seconds()
        return (datetime.now() - self.created_at).total_seconds()
    
    @property
    def handle_time(self) -> Optional[float]:
        """Get time spent handling in seconds."""
        if self.completed_at and self.assigned_at:
            return (self.completed_at - self.assigned_at).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "reason": self.reason.value,
            "priority": self.priority.name,
            "context": self.context.to_dict(),
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "assigned_agent_id": self.assigned_agent_id,
            "assigned_at": self.assigned_at.isoformat() if self.assigned_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "resolution_notes": self.resolution_notes,
            "wait_time": self.wait_time,
            "handle_time": self.handle_time,
        }


@dataclass
class HumanAgent:
    """Represents a human agent."""
    id: str
    name: str
    email: Optional[str] = None
    skills: Set[str] = field(default_factory=set)
    status: AgentStatus = AgentStatus.OFFLINE
    current_sessions: List[str] = field(default_factory=list)
    max_concurrent_sessions: int = 3
    handled_count: int = 0
    avg_handle_time: float = 0.0
    last_active: Optional[datetime] = None
    
    @property
    def is_available(self) -> bool:
        """Check if agent can take new sessions."""
        return (
            self.status == AgentStatus.AVAILABLE and
            len(self.current_sessions) < self.max_concurrent_sessions
        )
    
    @property
    def load(self) -> float:
        """Get agent load (0-1)."""
        return len(self.current_sessions) / self.max_concurrent_sessions


class AgentRouter(ABC):
    """Abstract base class for routing handoffs to agents."""
    
    @abstractmethod
    def find_agent(
        self,
        request: HandoffRequest,
        agents: List[HumanAgent]
    ) -> Optional[HumanAgent]:
        """Find the best agent for a handoff request."""
        pass


class RoundRobinRouter(AgentRouter):
    """Simple round-robin agent routing."""
    
    def __init__(self):
        self._last_agent_index = -1
    
    def find_agent(
        self,
        request: HandoffRequest,
        agents: List[HumanAgent]
    ) -> Optional[HumanAgent]:
        """Find next available agent in rotation."""
        available = [a for a in agents if a.is_available]
        if not available:
            return None
        
        self._last_agent_index = (self._last_agent_index + 1) % len(available)
        return available[self._last_agent_index]


class SkillBasedRouter(AgentRouter):
    """Route based on agent skills and request context."""
    
    def __init__(self, skill_mapping: Optional[Dict[EscalationReason, Set[str]]] = None):
        self.skill_mapping = skill_mapping or {}
    
    def find_agent(
        self,
        request: HandoffRequest,
        agents: List[HumanAgent]
    ) -> Optional[HumanAgent]:
        """Find agent with matching skills."""
        available = [a for a in agents if a.is_available]
        if not available:
            return None
        
        # Get required skills for this reason
        required_skills = self.skill_mapping.get(request.reason, set())
        
        # Score agents by skill match and load
        scored = []
        for agent in available:
            skill_match = len(required_skills & agent.skills) / len(required_skills) if required_skills else 1.0
            load_score = 1 - agent.load
            score = skill_match * 0.7 + load_score * 0.3
            scored.append((score, agent))
        
        scored.sort(key=lambda x: -x[0])
        return scored[0][1] if scored else None


class LoadBalancedRouter(AgentRouter):
    """Route to least loaded agent."""
    
    def find_agent(
        self,
        request: HandoffRequest,
        agents: List[HumanAgent]
    ) -> Optional[HumanAgent]:
        """Find least loaded agent."""
        available = [a for a in agents if a.is_available]
        if not available:
            return None
        
        # Sort by load (ascending)
        available.sort(key=lambda a: a.load)
        return available[0]


class HandoffQueue:
    """Queue for managing handoff requests."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._queue: List[HandoffRequest] = []
        self._by_session: Dict[str, HandoffRequest] = {}
        self._lock = threading.Lock()
    
    def add(self, request: HandoffRequest) -> bool:
        """Add a request to the queue."""
        with self._lock:
            if len(self._queue) >= self.max_size:
                return False
            
            # Check if session already has pending request
            if request.session_id in self._by_session:
                return False
            
            request.status = HandoffStatus.QUEUED
            self._queue.append(request)
            self._queue.sort()  # Maintain priority order
            self._by_session[request.session_id] = request
            return True
    
    def get_next(self) -> Optional[HandoffRequest]:
        """Get the next request from the queue."""
        with self._lock:
            if not self._queue:
                return None
            
            request = self._queue.pop(0)
            del self._by_session[request.session_id]
            return request
    
    def peek(self) -> Optional[HandoffRequest]:
        """Peek at the next request without removing it."""
        with self._lock:
            return self._queue[0] if self._queue else None
    
    def remove(self, session_id: str) -> Optional[HandoffRequest]:
        """Remove a request by session ID."""
        with self._lock:
            if session_id not in self._by_session:
                return None
            
            request = self._by_session.pop(session_id)
            self._queue.remove(request)
            return request
    
    def get_by_session(self, session_id: str) -> Optional[HandoffRequest]:
        """Get request for a session."""
        with self._lock:
            return self._by_session.get(session_id)
    
    @property
    def size(self) -> int:
        """Get queue size."""
        with self._lock:
            return len(self._queue)
    
    @property
    def pending_requests(self) -> List[HandoffRequest]:
        """Get all pending requests."""
        with self._lock:
            return self._queue.copy()


@dataclass
class HandoffMetrics:
    """Metrics for handoff system."""
    total_requests: int = 0
    successful_handoffs: int = 0
    cancelled_handoffs: int = 0
    timeout_handoffs: int = 0
    returned_to_bot: int = 0
    avg_wait_time: float = 0.0
    avg_handle_time: float = 0.0
    by_reason: Dict[str, int] = field(default_factory=dict)
    by_priority: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_handoffs": self.successful_handoffs,
            "cancelled_handoffs": self.cancelled_handoffs,
            "timeout_handoffs": self.timeout_handoffs,
            "returned_to_bot": self.returned_to_bot,
            "avg_wait_time": self.avg_wait_time,
            "avg_handle_time": self.avg_handle_time,
            "success_rate": self.successful_handoffs / self.total_requests if self.total_requests else 0,
            "by_reason": self.by_reason,
            "by_priority": self.by_priority,
        }


class HumanHandoffManager:
    """
    Main manager for human handoff functionality.
    
    Features:
    - Automatic escalation trigger detection
    - Queue management
    - Agent routing
    - Handoff tracking and metrics
    - Configurable escalation rules
    """
    
    def __init__(
        self,
        router: Optional[AgentRouter] = None,
        max_queue_size: int = 1000,
        queue_timeout_minutes: int = 30
    ):
        self.router = router or LoadBalancedRouter()
        self.queue = HandoffQueue(max_size=max_queue_size)
        self.queue_timeout = timedelta(minutes=queue_timeout_minutes)
        
        # Agents
        self._agents: Dict[str, HumanAgent] = {}
        
        # Active handoffs
        self._active_handoffs: Dict[str, HandoffRequest] = {}
        
        # Completed handoffs (for metrics)
        self._completed_handoffs: List[HandoffRequest] = []
        self._max_completed_history = 1000
        
        # Escalation triggers
        self._triggers: List[EscalationTrigger] = []
        self._trigger_cooldowns: Dict[str, datetime] = {}  # session_id:reason -> last_trigger
        
        # Callbacks
        self._on_handoff_requested: List[Callable[[HandoffRequest], None]] = []
        self._on_handoff_assigned: List[Callable[[HandoffRequest, HumanAgent], None]] = []
        self._on_handoff_completed: List[Callable[[HandoffRequest], None]] = []
        
        self._lock = threading.Lock()
        
        # Initialize default triggers
        self._initialize_default_triggers()
    
    def _initialize_default_triggers(self):
        """Initialize default escalation triggers."""
        # User requested handoff
        self.add_trigger(EscalationTrigger(
            reason=EscalationReason.USER_REQUESTED,
            condition=lambda ctx: ctx.get("user_requested_human", False),
            priority=Priority.HIGH,
            description="User explicitly requested to speak with a human"
        ))
        
        # Low confidence
        self.add_trigger(EscalationTrigger(
            reason=EscalationReason.LOW_CONFIDENCE,
            condition=lambda ctx: ctx.get("confidence", 1.0) < 0.3,
            priority=Priority.NORMAL,
            description="Intent detection confidence below threshold",
            cooldown_seconds=60
        ))
        
        # Repeated errors
        self.add_trigger(EscalationTrigger(
            reason=EscalationReason.REPEATED_ERRORS,
            condition=lambda ctx: ctx.get("consecutive_errors", 0) >= 3,
            priority=Priority.HIGH,
            description="Multiple consecutive errors in conversation"
        ))
        
        # Negative sentiment
        self.add_trigger(EscalationTrigger(
            reason=EscalationReason.SENTIMENT_NEGATIVE,
            condition=lambda ctx: ctx.get("sentiment") == "frustrated" or ctx.get("sentiment") == "angry",
            priority=Priority.HIGH,
            description="User showing signs of frustration"
        ))
        
        # Loop detection
        self.add_trigger(EscalationTrigger(
            reason=EscalationReason.LOOP_DETECTED,
            condition=lambda ctx: ctx.get("repeated_queries", 0) >= 3,
            priority=Priority.NORMAL,
            description="Conversation appears to be in a loop"
        ))
        
        # Timeout
        self.add_trigger(EscalationTrigger(
            reason=EscalationReason.TIMEOUT_EXCEEDED,
            condition=lambda ctx: ctx.get("conversation_duration", 0) > 600,  # 10 minutes
            priority=Priority.LOW,
            description="Conversation duration exceeded threshold"
        ))
    
    def add_trigger(self, trigger: EscalationTrigger) -> None:
        """Add an escalation trigger."""
        self._triggers.append(trigger)
    
    def remove_trigger(self, reason: EscalationReason) -> None:
        """Remove triggers for a specific reason."""
        self._triggers = [t for t in self._triggers if t.reason != reason]
    
    def check_triggers(
        self,
        session_id: str,
        context: Dict[str, Any]
    ) -> Optional[Tuple[EscalationReason, Priority]]:
        """Check if any escalation triggers should fire."""
        for trigger in self._triggers:
            # Check cooldown
            cooldown_key = f"{session_id}:{trigger.reason.value}"
            if trigger.cooldown_seconds > 0:
                last_trigger = self._trigger_cooldowns.get(cooldown_key)
                if last_trigger:
                    elapsed = (datetime.now() - last_trigger).total_seconds()
                    if elapsed < trigger.cooldown_seconds:
                        continue
            
            # Evaluate trigger
            if trigger.evaluate(context):
                # Update cooldown
                self._trigger_cooldowns[cooldown_key] = datetime.now()
                return trigger.reason, trigger.priority
        
        return None
    
    def register_agent(self, agent: HumanAgent) -> None:
        """Register a human agent."""
        with self._lock:
            self._agents[agent.id] = agent
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister a human agent."""
        with self._lock:
            if agent_id in self._agents:
                del self._agents[agent_id]
    
    def update_agent_status(self, agent_id: str, status: AgentStatus) -> None:
        """Update an agent's status."""
        with self._lock:
            if agent_id in self._agents:
                self._agents[agent_id].status = status
                self._agents[agent_id].last_active = datetime.now()
    
    def get_available_agents(self) -> List[HumanAgent]:
        """Get list of available agents."""
        with self._lock:
            return [a for a in self._agents.values() if a.is_available]
    
    def request_handoff(
        self,
        session_id: str,
        reason: EscalationReason,
        context: ConversationContext,
        priority: Priority = Priority.NORMAL,
        force: bool = False
    ) -> Optional[HandoffRequest]:
        """
        Request a handoff to a human agent.
        
        Args:
            session_id: The conversation session ID
            reason: Reason for escalation
            context: Conversation context to pass
            priority: Priority level
            force: Force handoff even if one is pending
        
        Returns:
            HandoffRequest if created, None otherwise
        """
        # Check if already pending
        existing = self.queue.get_by_session(session_id)
        if existing and not force:
            logger.info(f"Handoff already pending for session {session_id}")
            return existing
        
        # Remove existing if forcing
        if existing and force:
            self.queue.remove(session_id)
        
        # Create request
        request = HandoffRequest(
            id=str(uuid.uuid4()),
            session_id=session_id,
            reason=reason,
            priority=priority,
            context=context
        )
        
        # Try to find an available agent immediately
        available_agents = self.get_available_agents()
        agent = self.router.find_agent(request, available_agents)
        
        if agent:
            # Assign immediately
            self._assign_handoff(request, agent)
        else:
            # Add to queue
            if not self.queue.add(request):
                logger.error(f"Failed to queue handoff for session {session_id}")
                return None
            logger.info(f"Handoff queued for session {session_id}, queue size: {self.queue.size}")
        
        # Notify callbacks
        for callback in self._on_handoff_requested:
            try:
                callback(request)
            except Exception as e:
                logger.error(f"Handoff request callback error: {e}")
        
        return request
    
    def _assign_handoff(self, request: HandoffRequest, agent: HumanAgent) -> None:
        """Assign a handoff to an agent."""
        with self._lock:
            request.status = HandoffStatus.ASSIGNED
            request.assigned_agent_id = agent.id
            request.assigned_at = datetime.now()
            
            agent.current_sessions.append(request.session_id)
            agent.last_active = datetime.now()
            
            self._active_handoffs[request.session_id] = request
        
        logger.info(f"Handoff {request.id} assigned to agent {agent.name}")
        
        # Notify callbacks
        for callback in self._on_handoff_assigned:
            try:
                callback(request, agent)
            except Exception as e:
                logger.error(f"Handoff assigned callback error: {e}")
    
    def start_handling(self, session_id: str) -> bool:
        """Mark a handoff as in progress."""
        with self._lock:
            if session_id not in self._active_handoffs:
                return False
            self._active_handoffs[session_id].status = HandoffStatus.IN_PROGRESS
        return True
    
    def complete_handoff(
        self,
        session_id: str,
        resolution_notes: Optional[str] = None,
        return_to_bot: bool = False
    ) -> Optional[HandoffRequest]:
        """
        Complete a handoff.
        
        Args:
            session_id: The session ID
            resolution_notes: Notes about the resolution
            return_to_bot: Whether to return conversation to bot
        
        Returns:
            Completed HandoffRequest or None
        """
        with self._lock:
            if session_id not in self._active_handoffs:
                return None
            
            request = self._active_handoffs.pop(session_id)
            request.status = HandoffStatus.COMPLETED if not return_to_bot else HandoffStatus.RETURNED_TO_BOT
            request.completed_at = datetime.now()
            request.resolution_notes = resolution_notes
            request.returned_to_bot = return_to_bot
            
            # Update agent
            if request.assigned_agent_id and request.assigned_agent_id in self._agents:
                agent = self._agents[request.assigned_agent_id]
                if session_id in agent.current_sessions:
                    agent.current_sessions.remove(session_id)
                agent.handled_count += 1
                
                # Update average handle time
                if request.handle_time:
                    agent.avg_handle_time = (
                        (agent.avg_handle_time * (agent.handled_count - 1) + request.handle_time) /
                        agent.handled_count
                    )
            
            # Store in history
            self._completed_handoffs.append(request)
            if len(self._completed_handoffs) > self._max_completed_history:
                self._completed_handoffs = self._completed_handoffs[-self._max_completed_history:]
        
        logger.info(f"Handoff {request.id} completed")
        
        # Notify callbacks
        for callback in self._on_handoff_completed:
            try:
                callback(request)
            except Exception as e:
                logger.error(f"Handoff completed callback error: {e}")
        
        return request
    
    def cancel_handoff(self, session_id: str) -> Optional[HandoffRequest]:
        """Cancel a pending handoff."""
        # Try queue first
        request = self.queue.remove(session_id)
        if request:
            request.status = HandoffStatus.CANCELLED
            request.completed_at = datetime.now()
            return request
        
        # Try active handoffs
        with self._lock:
            if session_id in self._active_handoffs:
                request = self._active_handoffs.pop(session_id)
                request.status = HandoffStatus.CANCELLED
                request.completed_at = datetime.now()
                
                # Update agent
                if request.assigned_agent_id and request.assigned_agent_id in self._agents:
                    agent = self._agents[request.assigned_agent_id]
                    if session_id in agent.current_sessions:
                        agent.current_sessions.remove(session_id)
                
                return request
        
        return None
    
    def get_handoff_status(self, session_id: str) -> Optional[HandoffRequest]:
        """Get the status of a handoff request."""
        # Check queue
        request = self.queue.get_by_session(session_id)
        if request:
            return request
        
        # Check active
        with self._lock:
            return self._active_handoffs.get(session_id)
    
    def process_queue(self) -> int:
        """Process the queue and assign waiting requests to available agents."""
        assigned_count = 0
        
        while True:
            # Check for timed out requests
            self._handle_timeouts()
            
            # Get next request
            request = self.queue.peek()
            if not request:
                break
            
            # Find available agent
            available_agents = self.get_available_agents()
            agent = self.router.find_agent(request, available_agents)
            
            if not agent:
                break  # No agents available
            
            # Remove from queue and assign
            self.queue.get_next()
            self._assign_handoff(request, agent)
            assigned_count += 1
        
        return assigned_count
    
    def _handle_timeouts(self) -> None:
        """Handle timed out queue requests."""
        now = datetime.now()
        timed_out = []
        
        for request in self.queue.pending_requests:
            if now - request.created_at > self.queue_timeout:
                timed_out.append(request.session_id)
        
        for session_id in timed_out:
            request = self.queue.remove(session_id)
            if request:
                request.status = HandoffStatus.TIMEOUT
                request.completed_at = now
                self._completed_handoffs.append(request)
                logger.warning(f"Handoff {request.id} timed out in queue")
    
    def get_queue_position(self, session_id: str) -> Optional[int]:
        """Get queue position for a session (1-indexed)."""
        for i, request in enumerate(self.queue.pending_requests, 1):
            if request.session_id == session_id:
                return i
        return None
    
    def get_estimated_wait_time(self, session_id: str) -> Optional[float]:
        """Estimate wait time for a queued request in seconds."""
        position = self.get_queue_position(session_id)
        if position is None:
            return None
        
        # Simple estimation based on position and available agents
        available_count = len(self.get_available_agents())
        if available_count == 0:
            # Use average handle time if no agents available
            metrics = self.get_metrics()
            if metrics.avg_handle_time > 0:
                return position * metrics.avg_handle_time
            return position * 300  # Default 5 minutes per position
        
        # Estimate based on agents becoming available
        return (position / available_count) * 120  # 2 minutes per cycle
    
    def get_metrics(self) -> HandoffMetrics:
        """Get handoff metrics."""
        metrics = HandoffMetrics()
        
        all_requests = self._completed_handoffs.copy()
        
        if not all_requests:
            return metrics
        
        metrics.total_requests = len(all_requests)
        
        wait_times = []
        handle_times = []
        
        for request in all_requests:
            # Count by status
            if request.status == HandoffStatus.COMPLETED:
                metrics.successful_handoffs += 1
            elif request.status == HandoffStatus.CANCELLED:
                metrics.cancelled_handoffs += 1
            elif request.status == HandoffStatus.TIMEOUT:
                metrics.timeout_handoffs += 1
            elif request.status == HandoffStatus.RETURNED_TO_BOT:
                metrics.returned_to_bot += 1
            
            # Count by reason
            reason = request.reason.value
            metrics.by_reason[reason] = metrics.by_reason.get(reason, 0) + 1
            
            # Count by priority
            priority = request.priority.name
            metrics.by_priority[priority] = metrics.by_priority.get(priority, 0) + 1
            
            # Collect times
            if request.wait_time:
                wait_times.append(request.wait_time)
            if request.handle_time:
                handle_times.append(request.handle_time)
        
        if wait_times:
            metrics.avg_wait_time = sum(wait_times) / len(wait_times)
        if handle_times:
            metrics.avg_handle_time = sum(handle_times) / len(handle_times)
        
        return metrics
    
    def on_handoff_requested(self, callback: Callable[[HandoffRequest], None]) -> None:
        """Register callback for handoff requests."""
        self._on_handoff_requested.append(callback)
    
    def on_handoff_assigned(self, callback: Callable[[HandoffRequest, HumanAgent], None]) -> None:
        """Register callback for handoff assignments."""
        self._on_handoff_assigned.append(callback)
    
    def on_handoff_completed(self, callback: Callable[[HandoffRequest], None]) -> None:
        """Register callback for handoff completions."""
        self._on_handoff_completed.append(callback)
    
    @property
    def active_count(self) -> int:
        """Get number of active handoffs."""
        with self._lock:
            return len(self._active_handoffs)
    
    @property
    def queue_size(self) -> int:
        """Get number of queued requests."""
        return self.queue.size
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for a dashboard view."""
        metrics = self.get_metrics()
        
        return {
            "queue_size": self.queue_size,
            "active_handoffs": self.active_count,
            "available_agents": len(self.get_available_agents()),
            "total_agents": len(self._agents),
            "metrics": metrics.to_dict(),
            "pending_requests": [r.to_dict() for r in self.queue.pending_requests[:10]],
            "agents": [
                {
                    "id": a.id,
                    "name": a.name,
                    "status": a.status.value,
                    "current_sessions": len(a.current_sessions),
                    "handled_count": a.handled_count,
                }
                for a in self._agents.values()
            ]
        }


class HandoffProtocol:
    """
    Protocol for managing the handoff conversation flow.
    
    Provides messages and actions for smooth handoff experience.
    """
    
    def __init__(self, manager: HumanHandoffManager):
        self.manager = manager
    
    def get_handoff_message(
        self,
        status: HandoffStatus,
        queue_position: Optional[int] = None,
        estimated_wait: Optional[float] = None,
        agent_name: Optional[str] = None
    ) -> str:
        """Get an appropriate message for the handoff status."""
        messages = {
            HandoffStatus.PENDING: "I'm connecting you with a human specialist. Please hold...",
            HandoffStatus.QUEUED: f"You're in the queue. Position: {queue_position or '?'}. "
                                 f"Estimated wait: {self._format_time(estimated_wait)}.",
            HandoffStatus.ASSIGNED: f"You've been connected with {agent_name or 'a specialist'}. "
                                   "They'll be with you shortly.",
            HandoffStatus.IN_PROGRESS: f"You're now chatting with {agent_name or 'a specialist'}.",
            HandoffStatus.COMPLETED: "Thanks for chatting with our team! Is there anything else I can help with?",
            HandoffStatus.CANCELLED: "The handoff request has been cancelled. How else can I assist you?",
            HandoffStatus.TIMEOUT: "I apologize, but all our specialists are currently busy. "
                                  "Would you like to leave a message or try again later?",
            HandoffStatus.RETURNED_TO_BOT: "You're back with me now. How can I continue helping you?",
        }
        return messages.get(status, "Processing your request...")
    
    def _format_time(self, seconds: Optional[float]) -> str:
        """Format wait time nicely."""
        if seconds is None:
            return "Unknown"
        if seconds < 60:
            return "Less than a minute"
        minutes = int(seconds / 60)
        if minutes == 1:
            return "About 1 minute"
        if minutes < 60:
            return f"About {minutes} minutes"
        return "More than an hour"
    
    def generate_handoff_summary(
        self,
        context: ConversationContext
    ) -> str:
        """Generate a summary for the human agent."""
        lines = [
            f"**Session:** {context.session_id}",
            f"**User ID:** {context.user_id or 'Unknown'}",
        ]
        
        if context.detected_intents:
            lines.append(f"**Intents:** {', '.join(context.detected_intents)}")
        
        if context.extracted_entities:
            entities = ', '.join(f"{k}={v}" for k, v in context.extracted_entities.items())
            lines.append(f"**Entities:** {entities}")
        
        if context.error_history:
            lines.append(f"**Errors:** {len(context.error_history)} errors encountered")
        
        if context.user_sentiment:
            lines.append(f"**Sentiment:** {context.user_sentiment}")
        
        if context.summary:
            lines.append(f"\n**Summary:** {context.summary}")
        
        if context.conversation_history:
            lines.append("\n**Recent Conversation:**")
            for msg in context.conversation_history[-5:]:  # Last 5 messages
                role = msg.get("role", "?")
                content = msg.get("content", "")[:200]  # Truncate
                lines.append(f"  {role}: {content}")
        
        return "\n".join(lines)


# Singleton instance
_handoff_manager: Optional[HumanHandoffManager] = None


def get_handoff_manager() -> HumanHandoffManager:
    """Get or create the singleton handoff manager."""
    global _handoff_manager
    if _handoff_manager is None:
        _handoff_manager = HumanHandoffManager()
    return _handoff_manager


def reset_handoff_manager() -> None:
    """Reset the singleton (for testing)."""
    global _handoff_manager
    _handoff_manager = None
