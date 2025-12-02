"""
Tests for Human Handoff System.

Phase 4 of Professional Chat Agent implementation.
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from workflow_composer.agents.intent.human_handoff import (
    HumanHandoffManager,
    HandoffRequest,
    HandoffStatus,
    HandoffQueue,
    HandoffProtocol,
    HandoffMetrics,
    EscalationReason,
    EscalationTrigger,
    ConversationContext,
    HumanAgent,
    AgentStatus,
    Priority,
    AgentRouter,
    RoundRobinRouter,
    SkillBasedRouter,
    LoadBalancedRouter,
    get_handoff_manager,
    reset_handoff_manager,
)
from workflow_composer.agents.intent.dialog_state_machine import DialogState


class TestEscalationReason:
    """Test EscalationReason enum."""
    
    def test_all_reasons_defined(self):
        """Test all expected reasons exist."""
        reasons = [
            EscalationReason.USER_REQUESTED,
            EscalationReason.LOW_CONFIDENCE,
            EscalationReason.REPEATED_ERRORS,
            EscalationReason.COMPLEX_QUERY,
            EscalationReason.SENTIMENT_NEGATIVE,
            EscalationReason.TOPIC_SENSITIVE,
            EscalationReason.LOOP_DETECTED,
            EscalationReason.TIMEOUT_EXCEEDED,
            EscalationReason.OUT_OF_SCOPE,
            EscalationReason.POLICY_VIOLATION,
            EscalationReason.TECHNICAL_ISSUE,
            EscalationReason.CUSTOM,
        ]
        assert len(reasons) == 12


class TestHandoffStatus:
    """Test HandoffStatus enum."""
    
    def test_all_statuses_defined(self):
        """Test all expected statuses exist."""
        statuses = [
            HandoffStatus.PENDING,
            HandoffStatus.QUEUED,
            HandoffStatus.ASSIGNED,
            HandoffStatus.IN_PROGRESS,
            HandoffStatus.COMPLETED,
            HandoffStatus.CANCELLED,
            HandoffStatus.TIMEOUT,
            HandoffStatus.RETURNED_TO_BOT,
        ]
        assert len(statuses) == 8


class TestPriority:
    """Test Priority enum."""
    
    def test_priority_ordering(self):
        """Test priority values are ordered correctly."""
        assert Priority.LOW.value < Priority.NORMAL.value
        assert Priority.NORMAL.value < Priority.HIGH.value
        assert Priority.HIGH.value < Priority.URGENT.value
        assert Priority.URGENT.value < Priority.CRITICAL.value


class TestEscalationTrigger:
    """Test EscalationTrigger dataclass."""
    
    def test_basic_trigger(self):
        """Test creating a basic trigger."""
        trigger = EscalationTrigger(
            reason=EscalationReason.USER_REQUESTED,
            condition=lambda ctx: ctx.get("requested", False)
        )
        assert trigger.reason == EscalationReason.USER_REQUESTED
        assert trigger.enabled
    
    def test_trigger_evaluation(self):
        """Test trigger evaluation."""
        trigger = EscalationTrigger(
            reason=EscalationReason.LOW_CONFIDENCE,
            condition=lambda ctx: ctx.get("confidence", 1.0) < 0.5
        )
        
        assert trigger.evaluate({"confidence": 0.3})
        assert not trigger.evaluate({"confidence": 0.8})
    
    def test_disabled_trigger(self):
        """Test that disabled triggers don't fire."""
        trigger = EscalationTrigger(
            reason=EscalationReason.USER_REQUESTED,
            condition=lambda ctx: True,
            enabled=False
        )
        
        assert not trigger.evaluate({})
    
    def test_trigger_with_exception(self):
        """Test trigger gracefully handles exceptions."""
        trigger = EscalationTrigger(
            reason=EscalationReason.CUSTOM,
            condition=lambda ctx: ctx["missing_key"]  # Will raise
        )
        
        assert not trigger.evaluate({})  # Should return False, not raise


class TestConversationContext:
    """Test ConversationContext dataclass."""
    
    def test_basic_context(self):
        """Test creating basic context."""
        context = ConversationContext(session_id="test-123")
        assert context.session_id == "test-123"
        assert context.conversation_history == []
    
    def test_context_with_history(self):
        """Test context with conversation history."""
        context = ConversationContext(
            session_id="test",
            conversation_history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            detected_intents=["greeting"]
        )
        
        assert len(context.conversation_history) == 2
        assert "greeting" in context.detected_intents
    
    def test_context_to_dict(self):
        """Test context serialization."""
        context = ConversationContext(
            session_id="test",
            user_id="user123",
            current_state=DialogState.IDLE
        )
        
        data = context.to_dict()
        
        assert data["session_id"] == "test"
        assert data["user_id"] == "user123"
        assert data["current_state"] == "IDLE"


class TestHandoffRequest:
    """Test HandoffRequest dataclass."""
    
    def test_basic_request(self):
        """Test creating a basic request."""
        context = ConversationContext(session_id="test")
        request = HandoffRequest(
            id="req-123",
            session_id="test",
            reason=EscalationReason.USER_REQUESTED,
            priority=Priority.NORMAL,
            context=context
        )
        
        assert request.status == HandoffStatus.PENDING
        assert request.assigned_agent_id is None
    
    def test_wait_time_calculation(self):
        """Test wait time calculation."""
        context = ConversationContext(session_id="test")
        request = HandoffRequest(
            id="req-123",
            session_id="test",
            reason=EscalationReason.USER_REQUESTED,
            priority=Priority.NORMAL,
            context=context,
            created_at=datetime.now() - timedelta(seconds=30)
        )
        
        assert request.wait_time >= 29  # Allow small margin
    
    def test_handle_time_calculation(self):
        """Test handle time calculation."""
        context = ConversationContext(session_id="test")
        now = datetime.now()
        request = HandoffRequest(
            id="req-123",
            session_id="test",
            reason=EscalationReason.USER_REQUESTED,
            priority=Priority.NORMAL,
            context=context,
            created_at=now - timedelta(minutes=5),
            assigned_at=now - timedelta(minutes=3),
            completed_at=now
        )
        
        assert request.handle_time == pytest.approx(180, abs=1)  # 3 minutes
    
    def test_request_priority_comparison(self):
        """Test request comparison for queue ordering."""
        context = ConversationContext(session_id="test")
        
        high_priority = HandoffRequest(
            id="high",
            session_id="session1",
            reason=EscalationReason.USER_REQUESTED,
            priority=Priority.HIGH,
            context=context
        )
        
        low_priority = HandoffRequest(
            id="low",
            session_id="session2",
            reason=EscalationReason.USER_REQUESTED,
            priority=Priority.LOW,
            context=context
        )
        
        # Higher priority should come first (be "less than")
        assert high_priority < low_priority
    
    def test_request_to_dict(self):
        """Test request serialization."""
        context = ConversationContext(session_id="test")
        request = HandoffRequest(
            id="req-123",
            session_id="test",
            reason=EscalationReason.LOW_CONFIDENCE,
            priority=Priority.NORMAL,
            context=context
        )
        
        data = request.to_dict()
        
        assert data["id"] == "req-123"
        assert data["reason"] == "low_confidence"
        assert data["priority"] == "NORMAL"


class TestHumanAgent:
    """Test HumanAgent dataclass."""
    
    def test_basic_agent(self):
        """Test creating a basic agent."""
        agent = HumanAgent(id="agent-1", name="John")
        assert agent.status == AgentStatus.OFFLINE
        assert agent.is_available is False
    
    def test_agent_availability(self):
        """Test agent availability checking."""
        agent = HumanAgent(
            id="agent-1",
            name="John",
            status=AgentStatus.AVAILABLE,
            max_concurrent_sessions=2
        )
        
        assert agent.is_available
        
        # Add sessions
        agent.current_sessions = ["session1", "session2"]
        assert not agent.is_available  # At capacity
    
    def test_agent_load(self):
        """Test agent load calculation."""
        agent = HumanAgent(
            id="agent-1",
            name="John",
            max_concurrent_sessions=4,
            current_sessions=["s1", "s2"]
        )
        
        assert agent.load == 0.5


class TestAgentRouters:
    """Test agent routing strategies."""
    
    def create_agents(self) -> list:
        """Create test agents."""
        return [
            HumanAgent(id="a1", name="Agent 1", status=AgentStatus.AVAILABLE, skills={"workflow", "data"}),
            HumanAgent(id="a2", name="Agent 2", status=AgentStatus.AVAILABLE, skills={"support"}),
            HumanAgent(id="a3", name="Agent 3", status=AgentStatus.BUSY),
        ]
    
    def create_request(self) -> HandoffRequest:
        """Create test request."""
        return HandoffRequest(
            id="test",
            session_id="session",
            reason=EscalationReason.USER_REQUESTED,
            priority=Priority.NORMAL,
            context=ConversationContext(session_id="session")
        )
    
    def test_round_robin_router(self):
        """Test round-robin routing."""
        router = RoundRobinRouter()
        agents = self.create_agents()
        request = self.create_request()
        
        # Should rotate through available agents
        first = router.find_agent(request, agents)
        assert first is not None
        assert first.status == AgentStatus.AVAILABLE
    
    def test_load_balanced_router(self):
        """Test load-balanced routing."""
        router = LoadBalancedRouter()
        agents = self.create_agents()
        agents[0].current_sessions = ["s1", "s2"]  # Higher load
        
        request = self.create_request()
        
        # Should pick least loaded agent (a2)
        selected = router.find_agent(request, agents)
        assert selected.id == "a2"
    
    def test_skill_based_router(self):
        """Test skill-based routing."""
        skill_mapping = {
            EscalationReason.TECHNICAL_ISSUE: {"workflow", "data"},
        }
        router = SkillBasedRouter(skill_mapping=skill_mapping)
        agents = self.create_agents()
        
        # Create request needing technical skills
        request = HandoffRequest(
            id="test",
            session_id="session",
            reason=EscalationReason.TECHNICAL_ISSUE,
            priority=Priority.NORMAL,
            context=ConversationContext(session_id="session")
        )
        
        selected = router.find_agent(request, agents)
        assert selected.id == "a1"  # Has matching skills
    
    def test_router_with_no_available_agents(self):
        """Test routing when no agents available."""
        router = LoadBalancedRouter()
        agents = [
            HumanAgent(id="a1", name="Agent 1", status=AgentStatus.OFFLINE),
            HumanAgent(id="a2", name="Agent 2", status=AgentStatus.BUSY),
        ]
        
        request = self.create_request()
        
        selected = router.find_agent(request, agents)
        assert selected is None


class TestHandoffQueue:
    """Test HandoffQueue class."""
    
    def create_request(self, session_id: str, priority: Priority = Priority.NORMAL) -> HandoffRequest:
        """Create test request."""
        return HandoffRequest(
            id=f"req-{session_id}",
            session_id=session_id,
            reason=EscalationReason.USER_REQUESTED,
            priority=priority,
            context=ConversationContext(session_id=session_id)
        )
    
    def test_add_to_queue(self):
        """Test adding to queue."""
        queue = HandoffQueue()
        request = self.create_request("session1")
        
        assert queue.add(request)
        assert queue.size == 1
    
    def test_queue_max_size(self):
        """Test queue respects max size."""
        queue = HandoffQueue(max_size=2)
        
        assert queue.add(self.create_request("s1"))
        assert queue.add(self.create_request("s2"))
        assert not queue.add(self.create_request("s3"))  # Over limit
    
    def test_duplicate_session_rejected(self):
        """Test that duplicate sessions are rejected."""
        queue = HandoffQueue()
        
        assert queue.add(self.create_request("session1"))
        assert not queue.add(self.create_request("session1"))  # Duplicate
    
    def test_priority_ordering(self):
        """Test queue maintains priority order."""
        queue = HandoffQueue()
        
        queue.add(self.create_request("low", Priority.LOW))
        queue.add(self.create_request("high", Priority.HIGH))
        queue.add(self.create_request("normal", Priority.NORMAL))
        
        # Should get high priority first
        first = queue.get_next()
        assert first.session_id == "high"
        
        second = queue.get_next()
        assert second.session_id == "normal"
    
    def test_peek(self):
        """Test peeking at next item."""
        queue = HandoffQueue()
        queue.add(self.create_request("session1"))
        
        peeked = queue.peek()
        assert peeked.session_id == "session1"
        assert queue.size == 1  # Not removed
    
    def test_remove_by_session(self):
        """Test removing by session ID."""
        queue = HandoffQueue()
        queue.add(self.create_request("session1"))
        queue.add(self.create_request("session2"))
        
        removed = queue.remove("session1")
        
        assert removed.session_id == "session1"
        assert queue.size == 1
    
    def test_get_by_session(self):
        """Test getting request by session ID."""
        queue = HandoffQueue()
        queue.add(self.create_request("session1"))
        
        found = queue.get_by_session("session1")
        assert found.session_id == "session1"
        
        not_found = queue.get_by_session("nonexistent")
        assert not_found is None


class TestHumanHandoffManager:
    """Test HumanHandoffManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create manager instance for testing."""
        return HumanHandoffManager()
    
    @pytest.fixture
    def agent(self):
        """Create test agent."""
        return HumanAgent(
            id="test-agent",
            name="Test Agent",
            status=AgentStatus.AVAILABLE,
            max_concurrent_sessions=1
        )
    
    def test_register_agent(self, manager, agent):
        """Test registering an agent."""
        manager.register_agent(agent)
        
        available = manager.get_available_agents()
        assert len(available) == 1
        assert available[0].id == "test-agent"
    
    def test_unregister_agent(self, manager, agent):
        """Test unregistering an agent."""
        manager.register_agent(agent)
        manager.unregister_agent(agent.id)
        
        assert len(manager.get_available_agents()) == 0
    
    def test_update_agent_status(self, manager, agent):
        """Test updating agent status."""
        manager.register_agent(agent)
        manager.update_agent_status(agent.id, AgentStatus.BUSY)
        
        assert len(manager.get_available_agents()) == 0
    
    def test_check_triggers(self, manager):
        """Test checking escalation triggers."""
        # Test user requested trigger
        result = manager.check_triggers(
            "session1",
            {"user_requested_human": True}
        )
        
        assert result is not None
        reason, priority = result
        assert reason == EscalationReason.USER_REQUESTED
        assert priority == Priority.HIGH
    
    def test_check_triggers_low_confidence(self, manager):
        """Test low confidence trigger."""
        result = manager.check_triggers(
            "session1",
            {"confidence": 0.2}
        )
        
        assert result is not None
        assert result[0] == EscalationReason.LOW_CONFIDENCE
    
    def test_request_handoff_immediate_assignment(self, manager, agent):
        """Test handoff with immediate agent assignment."""
        manager.register_agent(agent)
        
        context = ConversationContext(session_id="test")
        request = manager.request_handoff(
            "test",
            EscalationReason.USER_REQUESTED,
            context
        )
        
        assert request is not None
        assert request.status == HandoffStatus.ASSIGNED
        assert request.assigned_agent_id == agent.id
    
    def test_request_handoff_queued(self, manager):
        """Test handoff is queued when no agents available."""
        context = ConversationContext(session_id="test")
        request = manager.request_handoff(
            "test",
            EscalationReason.USER_REQUESTED,
            context
        )
        
        assert request is not None
        assert request.status == HandoffStatus.QUEUED
        assert manager.queue_size == 1
    
    def test_complete_handoff(self, manager, agent):
        """Test completing a handoff."""
        manager.register_agent(agent)
        
        context = ConversationContext(session_id="test")
        manager.request_handoff("test", EscalationReason.USER_REQUESTED, context)
        
        completed = manager.complete_handoff(
            "test",
            resolution_notes="Resolved successfully"
        )
        
        assert completed is not None
        assert completed.status == HandoffStatus.COMPLETED
        assert completed.resolution_notes == "Resolved successfully"
        assert manager.active_count == 0
    
    def test_complete_with_return_to_bot(self, manager, agent):
        """Test completing handoff with return to bot."""
        manager.register_agent(agent)
        
        context = ConversationContext(session_id="test")
        manager.request_handoff("test", EscalationReason.USER_REQUESTED, context)
        
        completed = manager.complete_handoff("test", return_to_bot=True)
        
        assert completed.status == HandoffStatus.RETURNED_TO_BOT
        assert completed.returned_to_bot
    
    def test_cancel_handoff(self, manager):
        """Test cancelling a queued handoff."""
        context = ConversationContext(session_id="test")
        manager.request_handoff("test", EscalationReason.USER_REQUESTED, context)
        
        cancelled = manager.cancel_handoff("test")
        
        assert cancelled is not None
        assert cancelled.status == HandoffStatus.CANCELLED
        assert manager.queue_size == 0
    
    def test_get_handoff_status(self, manager):
        """Test getting handoff status."""
        context = ConversationContext(session_id="test")
        manager.request_handoff("test", EscalationReason.USER_REQUESTED, context)
        
        status = manager.get_handoff_status("test")
        
        assert status is not None
        assert status.session_id == "test"
    
    def test_process_queue(self, manager, agent):
        """Test processing the queue."""
        # Add requests to queue
        for i in range(3):
            context = ConversationContext(session_id=f"session{i}")
            manager.request_handoff(f"session{i}", EscalationReason.USER_REQUESTED, context)
        
        assert manager.queue_size == 3
        
        # Register agent
        manager.register_agent(agent)
        
        # Process queue
        assigned = manager.process_queue()
        
        assert assigned == 1  # One agent can take one
        assert manager.queue_size == 2
    
    def test_get_queue_position(self, manager):
        """Test getting queue position."""
        for i in range(3):
            context = ConversationContext(session_id=f"session{i}")
            manager.request_handoff(f"session{i}", EscalationReason.USER_REQUESTED, context)
        
        position = manager.get_queue_position("session1")
        assert position == 2  # 1-indexed
    
    def test_get_metrics(self, manager, agent):
        """Test getting handoff metrics."""
        manager.register_agent(agent)
        
        # Create some handoffs
        for i in range(3):
            context = ConversationContext(session_id=f"session{i}")
            manager.request_handoff(f"session{i}", EscalationReason.USER_REQUESTED, context)
            manager.complete_handoff(f"session{i}")
        
        metrics = manager.get_metrics()
        
        assert metrics.total_requests == 3
        assert metrics.successful_handoffs == 3
    
    def test_callbacks(self, manager, agent):
        """Test callback functionality."""
        requests_received = []
        assignments_received = []
        completions_received = []
        
        manager.on_handoff_requested(lambda r: requests_received.append(r))
        manager.on_handoff_assigned(lambda r, a: assignments_received.append((r, a)))
        manager.on_handoff_completed(lambda r: completions_received.append(r))
        
        manager.register_agent(agent)
        
        context = ConversationContext(session_id="test")
        manager.request_handoff("test", EscalationReason.USER_REQUESTED, context)
        manager.complete_handoff("test")
        
        assert len(requests_received) == 1
        assert len(assignments_received) == 1
        assert len(completions_received) == 1
    
    def test_get_dashboard_data(self, manager, agent):
        """Test getting dashboard data."""
        manager.register_agent(agent)
        
        context = ConversationContext(session_id="test")
        manager.request_handoff("test", EscalationReason.USER_REQUESTED, context)
        
        data = manager.get_dashboard_data()
        
        assert "queue_size" in data
        assert "active_handoffs" in data
        assert "available_agents" in data
        assert "agents" in data


class TestHandoffProtocol:
    """Test HandoffProtocol class."""
    
    @pytest.fixture
    def protocol(self):
        """Create protocol instance."""
        manager = HumanHandoffManager()
        return HandoffProtocol(manager)
    
    def test_get_handoff_message_pending(self, protocol):
        """Test message for pending status."""
        message = protocol.get_handoff_message(HandoffStatus.PENDING)
        assert "connecting" in message.lower()
    
    def test_get_handoff_message_queued(self, protocol):
        """Test message for queued status."""
        message = protocol.get_handoff_message(
            HandoffStatus.QUEUED,
            queue_position=3,
            estimated_wait=120
        )
        assert "3" in message
        assert "minute" in message.lower()
    
    def test_get_handoff_message_assigned(self, protocol):
        """Test message for assigned status."""
        message = protocol.get_handoff_message(
            HandoffStatus.ASSIGNED,
            agent_name="John"
        )
        assert "John" in message
    
    def test_generate_handoff_summary(self, protocol):
        """Test generating handoff summary."""
        context = ConversationContext(
            session_id="test-123",
            user_id="user-456",
            detected_intents=["create_workflow", "help"],
            extracted_entities={"organism": "human"},
            conversation_history=[
                {"role": "user", "content": "I need help"},
                {"role": "assistant", "content": "How can I help?"}
            ]
        )
        
        summary = protocol.generate_handoff_summary(context)
        
        assert "test-123" in summary
        assert "user-456" in summary
        assert "create_workflow" in summary
        assert "human" in summary
    
    def test_format_time(self, protocol):
        """Test time formatting."""
        assert "minute" in protocol._format_time(30).lower()
        assert "minute" in protocol._format_time(120).lower()
        assert "hour" in protocol._format_time(3700).lower()


class TestSingletonFunctions:
    """Test singleton factory functions."""
    
    def setup_method(self):
        """Reset singleton before each test."""
        reset_handoff_manager()
    
    def test_get_handoff_manager(self):
        """Test getting singleton manager."""
        manager1 = get_handoff_manager()
        manager2 = get_handoff_manager()
        
        assert manager1 is manager2
    
    def test_reset_handoff_manager(self):
        """Test resetting singleton."""
        manager1 = get_handoff_manager()
        reset_handoff_manager()
        manager2 = get_handoff_manager()
        
        assert manager1 is not manager2


class TestIntegration:
    """Integration tests for handoff system."""
    
    def test_full_handoff_flow(self):
        """Test a complete handoff flow."""
        manager = HumanHandoffManager()
        protocol = HandoffProtocol(manager)
        
        # Register agent
        agent = HumanAgent(
            id="agent-1",
            name="Support Agent",
            status=AgentStatus.AVAILABLE
        )
        manager.register_agent(agent)
        
        # Check escalation trigger
        context_data = {
            "confidence": 0.2,
            "consecutive_errors": 3
        }
        
        trigger_result = manager.check_triggers("session-1", context_data)
        assert trigger_result is not None
        reason, priority = trigger_result
        
        # Request handoff
        conversation_context = ConversationContext(
            session_id="session-1",
            user_id="user-123",
            conversation_history=[
                {"role": "user", "content": "Help me!"}
            ],
            detected_intents=["help"],
            error_history=["Error 1", "Error 2", "Error 3"]
        )
        
        request = manager.request_handoff(
            "session-1",
            reason,
            conversation_context,
            priority
        )
        
        # Should be assigned immediately
        assert request.status == HandoffStatus.ASSIGNED
        assert request.assigned_agent_id == agent.id
        
        # Get handoff message
        message = protocol.get_handoff_message(
            request.status,
            agent_name=agent.name
        )
        assert "Support Agent" in message
        
        # Generate summary for agent
        summary = protocol.generate_handoff_summary(conversation_context)
        assert "session-1" in summary
        assert "help" in summary.lower()
        
        # Complete handoff
        completed = manager.complete_handoff(
            "session-1",
            resolution_notes="Helped user with their issue"
        )
        
        assert completed.status == HandoffStatus.COMPLETED
        
        # Check metrics
        metrics = manager.get_metrics()
        assert metrics.total_requests == 1
        assert metrics.successful_handoffs == 1
    
    def test_queue_and_assignment_flow(self):
        """Test queue processing flow."""
        manager = HumanHandoffManager()
        
        # Add multiple requests without agents
        for i in range(5):
            context = ConversationContext(session_id=f"session-{i}")
            priority = Priority.HIGH if i == 2 else Priority.NORMAL
            manager.request_handoff(f"session-{i}", EscalationReason.USER_REQUESTED, context, priority)
        
        assert manager.queue_size == 5
        
        # Add agents one by one and process (each with max 1 concurrent)
        for i in range(3):
            agent = HumanAgent(
                id=f"agent-{i}",
                name=f"Agent {i}",
                status=AgentStatus.AVAILABLE,
                max_concurrent_sessions=1  # Only one session per agent
            )
            manager.register_agent(agent)
            manager.process_queue()
        
        # High priority should have been processed first
        assert manager.active_count == 3
        assert manager.queue_size == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
