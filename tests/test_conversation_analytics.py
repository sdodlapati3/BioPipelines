"""
Tests for Conversation Analytics System.

Phase 3 of Professional Chat Agent implementation.
"""

import pytest
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from workflow_composer.agents.intent.conversation_analytics import (
    ConversationAnalytics,
    AnalyticsDashboard,
    ConversationMetrics,
    IntentMetrics,
    StateMetrics,
    AggregateMetrics,
    MetricPoint,
    MetricType,
    ConversationOutcome,
    MetricsStorage,
    InMemoryMetricsStorage,
    FileMetricsStorage,
    get_conversation_analytics,
    get_analytics_dashboard,
    reset_analytics,
)
from workflow_composer.agents.intent.dialog_state_machine import DialogState, DialogEvent


class TestMetricType:
    """Test MetricType enum."""
    
    def test_all_types_defined(self):
        """Test all metric types exist."""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.TIMER.value == "timer"
        assert MetricType.RATE.value == "rate"


class TestConversationOutcome:
    """Test ConversationOutcome enum."""
    
    def test_all_outcomes_defined(self):
        """Test all outcomes exist."""
        outcomes = [
            ConversationOutcome.SUCCESS,
            ConversationOutcome.PARTIAL_SUCCESS,
            ConversationOutcome.ABANDONED,
            ConversationOutcome.ERROR,
            ConversationOutcome.ESCALATED,
            ConversationOutcome.TIMEOUT,
            ConversationOutcome.USER_CANCELLED,
        ]
        assert len(outcomes) == 7


class TestMetricPoint:
    """Test MetricPoint dataclass."""
    
    def test_basic_metric_point(self):
        """Test creating a basic metric point."""
        point = MetricPoint(
            timestamp=datetime.now(),
            value=42.0
        )
        assert point.value == 42.0
        assert point.tags == {}
        assert point.metadata == {}
    
    def test_metric_point_with_tags(self):
        """Test metric point with tags."""
        point = MetricPoint(
            timestamp=datetime.now(),
            value=1.0,
            tags={"session_id": "abc123", "intent": "greeting"}
        )
        assert point.tags["session_id"] == "abc123"


class TestConversationMetrics:
    """Test ConversationMetrics dataclass."""
    
    def test_basic_metrics(self):
        """Test creating conversation metrics."""
        metrics = ConversationMetrics(
            session_id="test-session",
            start_time=datetime.now()
        )
        assert metrics.session_id == "test-session"
        assert metrics.turn_count == 0
        assert metrics.outcome is None
    
    def test_duration_calculation(self):
        """Test duration calculation."""
        start = datetime.now()
        metrics = ConversationMetrics(
            session_id="test",
            start_time=start,
            end_time=start + timedelta(seconds=30)
        )
        assert metrics.duration == pytest.approx(30.0, abs=0.1)
    
    def test_duration_none_when_not_ended(self):
        """Test duration is None when conversation not ended."""
        metrics = ConversationMetrics(
            session_id="test",
            start_time=datetime.now()
        )
        assert metrics.duration is None
    
    def test_avg_response_time(self):
        """Test average response time calculation."""
        metrics = ConversationMetrics(
            session_id="test",
            start_time=datetime.now(),
            response_times=[1.0, 2.0, 3.0]
        )
        assert metrics.avg_response_time == pytest.approx(2.0)
    
    def test_success_rate_contribution(self):
        """Test success rate contribution."""
        success = ConversationMetrics(
            session_id="test",
            start_time=datetime.now(),
            outcome=ConversationOutcome.SUCCESS
        )
        assert success.success_rate_contribution == 1.0
        
        partial = ConversationMetrics(
            session_id="test",
            start_time=datetime.now(),
            outcome=ConversationOutcome.PARTIAL_SUCCESS
        )
        assert partial.success_rate_contribution == 0.5
        
        failed = ConversationMetrics(
            session_id="test",
            start_time=datetime.now(),
            outcome=ConversationOutcome.ABANDONED
        )
        assert failed.success_rate_contribution == 0.0
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        metrics = ConversationMetrics(
            session_id="test-123",
            start_time=datetime.now(),
            turn_count=5,
            intents_detected=["greeting", "workflow_create"]
        )
        
        data = metrics.to_dict()
        
        assert data["session_id"] == "test-123"
        assert data["turn_count"] == 5
        assert "greeting" in data["intents_detected"]


class TestIntentMetrics:
    """Test IntentMetrics dataclass."""
    
    def test_basic_intent_metrics(self):
        """Test creating intent metrics."""
        metrics = IntentMetrics(intent_name="greeting")
        assert metrics.intent_name == "greeting"
        assert metrics.detection_count == 0
    
    def test_record_detection(self):
        """Test recording intent detections."""
        metrics = IntentMetrics(intent_name="help")
        
        metrics.record_detection(0.9)
        metrics.record_detection(0.85)
        metrics.record_detection(0.95)
        
        assert metrics.detection_count == 3
        assert metrics.avg_confidence == pytest.approx(0.9, abs=0.01)
    
    def test_record_completion(self):
        """Test recording intent completions."""
        metrics = IntentMetrics(intent_name="workflow")
        
        metrics.record_completion(success=True, turns=3)
        metrics.record_completion(success=True, turns=5)
        metrics.record_completion(success=False, turns=10)
        
        assert metrics.successful_completions == 2
        assert metrics.failed_completions == 1
        assert metrics.completion_rate == pytest.approx(2/3)
        assert metrics.avg_turns_to_complete == 6.0


class TestStateMetrics:
    """Test StateMetrics dataclass."""
    
    def test_basic_state_metrics(self):
        """Test creating state metrics."""
        metrics = StateMetrics(state=DialogState.IDLE)
        assert metrics.state == DialogState.IDLE
        assert metrics.entry_count == 0
    
    def test_record_entry_exit(self):
        """Test recording state entry and exit."""
        metrics = StateMetrics(state=DialogState.SLOT_FILLING)
        
        metrics.record_entry()
        metrics.record_exit(duration=5.0, next_state=DialogState.CONFIRMING)
        
        assert metrics.entry_count == 1
        assert metrics.exit_count == 1
        assert metrics.avg_time_in_state == pytest.approx(5.0)
        assert metrics.transitions_to[DialogState.CONFIRMING] == 1


class TestAggregateMetrics:
    """Test AggregateMetrics dataclass."""
    
    def test_basic_aggregate(self):
        """Test creating aggregate metrics."""
        aggregate = AggregateMetrics(
            window_start=datetime.now() - timedelta(hours=1),
            window_end=datetime.now(),
            total_conversations=100,
            successful_conversations=80
        )
        assert aggregate.success_rate == 0.8
    
    def test_success_rate_zero_conversations(self):
        """Test success rate with zero conversations."""
        aggregate = AggregateMetrics(
            window_start=datetime.now(),
            window_end=datetime.now(),
            total_conversations=0
        )
        assert aggregate.success_rate == 0.0
    
    def test_to_dict(self):
        """Test serialization."""
        aggregate = AggregateMetrics(
            window_start=datetime.now() - timedelta(hours=1),
            window_end=datetime.now(),
            total_conversations=50,
            successful_conversations=40
        )
        
        data = aggregate.to_dict()
        
        assert data["total_conversations"] == 50
        assert data["success_rate"] == 0.8


class TestInMemoryMetricsStorage:
    """Test InMemoryMetricsStorage class."""
    
    def test_store_and_get_conversation(self):
        """Test storing and retrieving conversations."""
        storage = InMemoryMetricsStorage()
        
        metrics = ConversationMetrics(
            session_id="test-1",
            start_time=datetime.now(),
            turn_count=5
        )
        
        storage.store_conversation(metrics)
        
        result = storage.get_conversations()
        assert len(result) == 1
        assert result[0].session_id == "test-1"
    
    def test_max_conversations_limit(self):
        """Test that conversations are trimmed at max limit."""
        storage = InMemoryMetricsStorage(max_conversations=5)
        
        for i in range(10):
            storage.store_conversation(ConversationMetrics(
                session_id=f"test-{i}",
                start_time=datetime.now()
            ))
        
        result = storage.get_conversations(limit=100)
        assert len(result) == 5
        # Should have the last 5
        assert result[0].session_id == "test-5"
    
    def test_get_conversations_with_time_filter(self):
        """Test filtering conversations by time."""
        storage = InMemoryMetricsStorage()
        
        old_time = datetime.now() - timedelta(hours=2)
        recent_time = datetime.now() - timedelta(minutes=30)
        
        storage.store_conversation(ConversationMetrics(
            session_id="old",
            start_time=old_time
        ))
        storage.store_conversation(ConversationMetrics(
            session_id="recent",
            start_time=recent_time
        ))
        
        # Filter to last hour
        result = storage.get_conversations(
            start_time=datetime.now() - timedelta(hours=1)
        )
        
        assert len(result) == 1
        assert result[0].session_id == "recent"
    
    def test_store_and_get_metric(self):
        """Test storing and retrieving metrics."""
        storage = InMemoryMetricsStorage()
        
        point = MetricPoint(
            timestamp=datetime.now(),
            value=42.0,
            tags={"type": "test"}
        )
        
        storage.store_metric("test.metric", point)
        
        result = storage.get_metrics("test.metric")
        assert len(result) == 1
        assert result[0].value == 42.0
    
    def test_clear(self):
        """Test clearing storage."""
        storage = InMemoryMetricsStorage()
        
        storage.store_conversation(ConversationMetrics(
            session_id="test",
            start_time=datetime.now()
        ))
        storage.store_metric("test", MetricPoint(timestamp=datetime.now(), value=1))
        
        storage.clear()
        
        assert len(storage.get_conversations()) == 0
        assert len(storage.get_metrics("test")) == 0


class TestFileMetricsStorage:
    """Test FileMetricsStorage class."""
    
    def test_store_and_get_conversation(self):
        """Test storing and retrieving conversations from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileMetricsStorage(tmpdir)
            
            metrics = ConversationMetrics(
                session_id="file-test",
                start_time=datetime.now(),
                turn_count=3,
                outcome=ConversationOutcome.SUCCESS
            )
            
            storage.store_conversation(metrics)
            
            result = storage.get_conversations()
            assert len(result) == 1
            assert result[0].session_id == "file-test"
    
    def test_store_and_get_metric(self):
        """Test storing and retrieving metrics from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileMetricsStorage(tmpdir)
            
            point = MetricPoint(
                timestamp=datetime.now(),
                value=100.0,
                tags={"source": "test"}
            )
            
            storage.store_metric("response_time", point)
            
            result = storage.get_metrics("response_time")
            assert len(result) == 1
            assert result[0].value == 100.0


class TestConversationAnalytics:
    """Test ConversationAnalytics class."""
    
    @pytest.fixture
    def analytics(self):
        """Create analytics instance for testing."""
        return ConversationAnalytics()
    
    def test_start_conversation(self, analytics):
        """Test starting a conversation."""
        metrics = analytics.start_conversation("session-1")
        
        assert metrics.session_id == "session-1"
        assert analytics.get_active_conversation_count() == 1
    
    def test_end_conversation(self, analytics):
        """Test ending a conversation."""
        analytics.start_conversation("session-1")
        
        metrics = analytics.end_conversation(
            "session-1",
            outcome=ConversationOutcome.SUCCESS,
            user_satisfaction=4.5
        )
        
        assert metrics.outcome == ConversationOutcome.SUCCESS
        assert metrics.user_satisfaction == 4.5
        assert analytics.get_active_conversation_count() == 0
    
    def test_end_nonexistent_conversation(self, analytics):
        """Test ending a conversation that doesn't exist."""
        result = analytics.end_conversation("nonexistent", ConversationOutcome.SUCCESS)
        assert result is None
    
    def test_record_turn(self, analytics):
        """Test recording a conversation turn."""
        analytics.start_conversation("session-1")
        
        analytics.record_turn(
            "session-1",
            response_time=0.5,
            intent="greeting",
            confidence=0.95,
            entities=["user_name"]
        )
        
        # Check metrics were updated (indirectly via storage)
        analytics.record_turn(
            "session-1",
            response_time=0.3,
            intent="help"
        )
    
    def test_record_state_transition(self, analytics):
        """Test recording state transitions."""
        analytics.start_conversation("session-1")
        
        analytics.record_state_transition(
            "session-1",
            from_state=DialogState.IDLE,
            to_state=DialogState.UNDERSTANDING
        )
        
        # Check state metrics
        state_metrics = analytics.get_state_metrics(DialogState.UNDERSTANDING)
        assert state_metrics.entry_count == 1
    
    def test_record_clarification(self, analytics):
        """Test recording clarifications."""
        analytics.start_conversation("session-1")
        analytics.record_clarification("session-1")
        
        # End and check
        metrics = analytics.end_conversation("session-1", ConversationOutcome.SUCCESS)
        assert metrics.clarification_count == 1
    
    def test_record_error(self, analytics):
        """Test recording errors."""
        analytics.start_conversation("session-1")
        analytics.record_error("session-1", "parse_error")
        
        metrics = analytics.end_conversation("session-1", ConversationOutcome.ERROR)
        assert metrics.error_count == 1
    
    def test_record_recovery(self, analytics):
        """Test recording recoveries."""
        analytics.start_conversation("session-1")
        analytics.record_error("session-1", "timeout")
        analytics.record_recovery("session-1", "retry")
        
        metrics = analytics.end_conversation("session-1", ConversationOutcome.SUCCESS)
        assert metrics.recovery_count == 1
    
    def test_record_slot_prompt(self, analytics):
        """Test recording slot prompts."""
        analytics.start_conversation("session-1")
        analytics.record_slot_prompt("session-1", "organism")
        analytics.record_slot_prompt("session-1", "genome_build")
        
        metrics = analytics.end_conversation("session-1", ConversationOutcome.SUCCESS)
        assert metrics.slot_prompts == 2
    
    def test_record_disambiguation(self, analytics):
        """Test recording disambiguation requests."""
        analytics.start_conversation("session-1")
        analytics.record_disambiguation("session-1", options_count=3)
        
        metrics = analytics.end_conversation("session-1", ConversationOutcome.SUCCESS)
        assert metrics.disambiguation_count == 1
    
    def test_record_intent_completion(self, analytics):
        """Test recording intent completions."""
        analytics.record_intent_completion("create_workflow", success=True, turns=5)
        analytics.record_intent_completion("create_workflow", success=True, turns=3)
        analytics.record_intent_completion("create_workflow", success=False, turns=8)
        
        intent_metrics = analytics.get_intent_metrics("create_workflow")
        assert intent_metrics.successful_completions == 2
        assert intent_metrics.failed_completions == 1
    
    def test_get_aggregate_metrics(self, analytics):
        """Test getting aggregate metrics."""
        # Create some conversations
        for i in range(10):
            analytics.start_conversation(f"session-{i}")
            analytics.record_turn(f"session-{i}", response_time=0.5)
            analytics.record_turn(f"session-{i}", response_time=0.3)
            outcome = ConversationOutcome.SUCCESS if i < 8 else ConversationOutcome.ABANDONED
            analytics.end_conversation(f"session-{i}", outcome=outcome)
        
        aggregate = analytics.get_aggregate_metrics(window_hours=1)
        
        assert aggregate.total_conversations == 10
        assert aggregate.successful_conversations == 8
        assert aggregate.success_rate == 0.8
    
    def test_get_all_intent_metrics(self, analytics):
        """Test getting all intent metrics."""
        analytics.record_intent_completion("intent1", True, 3)
        analytics.record_intent_completion("intent2", True, 5)
        
        all_metrics = analytics.get_all_intent_metrics()
        assert "intent1" in all_metrics
        assert "intent2" in all_metrics
    
    def test_get_all_state_metrics(self, analytics):
        """Test getting all state metrics."""
        state_metrics = analytics.get_all_state_metrics()
        
        # Should have metrics for all states
        assert DialogState.IDLE in state_metrics
        assert DialogState.EXECUTING in state_metrics
    
    def test_alert_callback(self):
        """Test alert callbacks are triggered."""
        alerts_received = []
        
        def alert_handler(metric, data):
            alerts_received.append((metric, data))
        
        analytics = ConversationAnalytics(alert_callbacks=[alert_handler])
        analytics.set_alert_threshold("error_rate", 0.05, "High errors!")
        
        # Create conversations with high error rate
        for i in range(10):
            analytics.start_conversation(f"session-{i}")
            analytics.record_error(f"session-{i}", "test_error")
            analytics.end_conversation(f"session-{i}", ConversationOutcome.ERROR)
        
        # Get aggregate to trigger alert check
        analytics.get_aggregate_metrics(window_hours=1)
        
        assert len(alerts_received) > 0
        assert alerts_received[0][0] == "error_rate"
    
    def test_export_report(self, analytics):
        """Test exporting analytics report."""
        analytics.start_conversation("test-1")
        analytics.record_turn("test-1", response_time=0.5, intent="greeting")
        analytics.end_conversation("test-1", ConversationOutcome.SUCCESS)
        
        report = analytics.export_report(window_hours=1)
        
        assert "generated_at" in report
        assert "aggregate" in report
        assert "intent_metrics" in report
        assert "state_metrics" in report


class TestAnalyticsDashboard:
    """Test AnalyticsDashboard class."""
    
    @pytest.fixture
    def dashboard(self):
        """Create dashboard instance for testing."""
        analytics = ConversationAnalytics()
        
        # Add some test data
        for i in range(5):
            analytics.start_conversation(f"session-{i}")
            analytics.record_turn(f"session-{i}", response_time=0.5, intent="test")
            outcome = ConversationOutcome.SUCCESS if i < 4 else ConversationOutcome.ABANDONED
            analytics.end_conversation(f"session-{i}", outcome=outcome, user_satisfaction=4.0)
        
        return AnalyticsDashboard(analytics)
    
    def test_get_summary(self, dashboard):
        """Test getting text summary."""
        summary = dashboard.get_summary(window_hours=1)
        
        assert "Conversation Analytics Summary" in summary
        assert "Conversations:" in summary
        assert "Success Rate:" in summary
    
    def test_get_intent_report(self, dashboard):
        """Test getting intent report."""
        report = dashboard.get_intent_report()
        
        assert "Intent Performance Report" in report
    
    def test_get_state_report(self, dashboard):
        """Test getting state report."""
        report = dashboard.get_state_report()
        
        assert "Dialog State Report" in report
    
    def test_get_health_status(self, dashboard):
        """Test getting health status."""
        status = dashboard.get_health_status()
        
        assert "status" in status
        assert "active_conversations" in status
        assert "success_rate" in status
        assert "issues" in status
    
    def test_health_status_degraded(self):
        """Test health status when degraded."""
        analytics = ConversationAnalytics()
        
        # Create failing conversations
        for i in range(10):
            analytics.start_conversation(f"session-{i}")
            analytics.record_error(f"session-{i}", "error")
            analytics.end_conversation(f"session-{i}", ConversationOutcome.ERROR)
        
        dashboard = AnalyticsDashboard(analytics)
        status = dashboard.get_health_status()
        
        assert status["status"] in ["degraded", "critical"]
        assert len(status["issues"]) > 0


class TestSingletonFunctions:
    """Test singleton factory functions."""
    
    def setup_method(self):
        """Reset singletons before each test."""
        reset_analytics()
    
    def test_get_conversation_analytics(self):
        """Test getting singleton analytics instance."""
        analytics1 = get_conversation_analytics()
        analytics2 = get_conversation_analytics()
        
        assert analytics1 is analytics2
    
    def test_get_analytics_dashboard(self):
        """Test getting singleton dashboard instance."""
        dashboard1 = get_analytics_dashboard()
        dashboard2 = get_analytics_dashboard()
        
        assert dashboard1 is dashboard2
    
    def test_reset_analytics(self):
        """Test resetting singleton instances."""
        analytics1 = get_conversation_analytics()
        reset_analytics()
        analytics2 = get_conversation_analytics()
        
        assert analytics1 is not analytics2


class TestIntegration:
    """Integration tests for analytics system."""
    
    def test_full_conversation_flow(self):
        """Test tracking a complete conversation flow."""
        analytics = ConversationAnalytics()
        session_id = "integration-test"
        
        # Start
        analytics.start_conversation(session_id)
        
        # Simulate conversation
        analytics.record_state_transition(
            session_id,
            DialogState.IDLE,
            DialogState.UNDERSTANDING
        )
        
        analytics.record_turn(
            session_id,
            response_time=0.2,
            intent="create_workflow",
            confidence=0.9
        )
        
        analytics.record_state_transition(
            session_id,
            DialogState.UNDERSTANDING,
            DialogState.SLOT_FILLING
        )
        
        analytics.record_slot_prompt(session_id, "organism")
        analytics.record_turn(session_id, response_time=0.1)
        
        analytics.record_slot_prompt(session_id, "genome")
        analytics.record_turn(session_id, response_time=0.1)
        
        analytics.record_state_transition(
            session_id,
            DialogState.SLOT_FILLING,
            DialogState.CONFIRMING
        )
        
        analytics.record_turn(session_id, response_time=0.3)
        
        analytics.record_state_transition(
            session_id,
            DialogState.CONFIRMING,
            DialogState.EXECUTING
        )
        
        # End successfully
        metrics = analytics.end_conversation(
            session_id,
            ConversationOutcome.SUCCESS,
            user_satisfaction=5.0
        )
        
        # Verify
        assert metrics.turn_count == 4
        assert metrics.slot_prompts == 2
        assert len(metrics.states_visited) == 4
        assert metrics.user_satisfaction == 5.0
        
        # Check intent completion
        analytics.record_intent_completion("create_workflow", True, 4)
        intent_metrics = analytics.get_intent_metrics("create_workflow")
        assert intent_metrics.successful_completions == 1
    
    def test_error_and_recovery_flow(self):
        """Test tracking error and recovery scenarios."""
        analytics = ConversationAnalytics()
        session_id = "error-test"
        
        analytics.start_conversation(session_id)
        
        # Simulate error
        analytics.record_state_transition(
            session_id,
            DialogState.EXECUTING,
            DialogState.ERROR_RECOVERY
        )
        analytics.record_error(session_id, "execution_failed")
        
        # Recovery
        analytics.record_recovery(session_id, "retry")
        analytics.record_state_transition(
            session_id,
            DialogState.ERROR_RECOVERY,
            DialogState.EXECUTING
        )
        
        # Success after retry
        metrics = analytics.end_conversation(
            session_id,
            ConversationOutcome.SUCCESS
        )
        
        assert metrics.error_count == 1
        assert metrics.recovery_count == 1
    
    def test_analytics_with_file_storage(self):
        """Test analytics with file-based storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileMetricsStorage(tmpdir)
            analytics = ConversationAnalytics(storage=storage)
            
            # Create conversation
            analytics.start_conversation("file-test")
            analytics.record_turn("file-test", response_time=0.5, intent="test")
            analytics.end_conversation("file-test", ConversationOutcome.SUCCESS)
            
            # Verify file was created
            conversations_file = Path(tmpdir) / "conversations.jsonl"
            assert conversations_file.exists()
            
            # Create new analytics instance with same storage
            analytics2 = ConversationAnalytics(storage=FileMetricsStorage(tmpdir))
            
            # Should be able to retrieve
            conversations = analytics2.storage.get_conversations()
            assert len(conversations) >= 1


class TestConcurrency:
    """Test concurrent access to analytics."""
    
    def test_concurrent_conversations(self):
        """Test multiple concurrent conversations."""
        analytics = ConversationAnalytics()
        
        # Start multiple conversations
        for i in range(10):
            analytics.start_conversation(f"concurrent-{i}")
        
        assert analytics.get_active_conversation_count() == 10
        
        # Record turns concurrently
        for i in range(10):
            analytics.record_turn(f"concurrent-{i}", response_time=0.1 * i)
        
        # End all
        for i in range(10):
            analytics.end_conversation(f"concurrent-{i}", ConversationOutcome.SUCCESS)
        
        assert analytics.get_active_conversation_count() == 0
        
        # Check aggregate
        aggregate = analytics.get_aggregate_metrics(window_hours=1)
        assert aggregate.total_conversations == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
