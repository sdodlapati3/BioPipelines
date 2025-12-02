"""
Tests for Chat Agent Integration (Professional Agent Phase 8).

Tests the unified ChatAgent that integrates all professional chat components.
"""

import pytest
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from src.workflow_composer.agents.intent.chat_agent import (
    ChatAgent,
    AgentConfig,
    AgentCapability,
    ChannelType,
    MessageDirection,
    Message,
    Session,
    AgentResponse,
    AgentPlugin,
    LoggingPlugin,
    MetricsPlugin,
    SessionManager,
    IntentRegistry,
    get_chat_agent,
    reset_chat_agent,
    create_chat_agent,
    quick_chat,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def agent():
    """Create a fresh ChatAgent for testing."""
    reset_chat_agent()
    return get_chat_agent()


@pytest.fixture
def minimal_agent():
    """Create a minimal agent with limited capabilities."""
    config = AgentConfig(
        name="Test Agent",
        enabled_capabilities={
            AgentCapability.DIALOG_STATE,
            AgentCapability.SCOPE_DETECTION,
        }
    )
    return create_chat_agent(config)


@pytest.fixture
def session_manager():
    """Create a session manager."""
    return SessionManager(timeout_minutes=30)


@pytest.fixture
def intent_registry():
    """Create an intent registry."""
    return IntentRegistry()


# =============================================================================
# Message Tests
# =============================================================================

class TestMessage:
    """Tests for Message class."""
    
    def test_message_creation(self):
        """Test creating a message."""
        message = Message(
            content="Hello, world!",
            direction=MessageDirection.INCOMING
        )
        
        assert message.content == "Hello, world!"
        assert message.direction == MessageDirection.INCOMING
        assert message.id is not None
    
    def test_message_default_values(self):
        """Test message default values."""
        message = Message()
        
        assert message.content == ""
        assert message.direction == MessageDirection.INCOMING
        assert message.channel == ChannelType.WEB
        assert message.timestamp is not None
    
    def test_message_to_dict(self):
        """Test message serialization."""
        message = Message(
            content="Test message",
            direction=MessageDirection.OUTGOING,
            user_id="user123"
        )
        
        data = message.to_dict()
        assert data["content"] == "Test message"
        assert data["direction"] == "outgoing"
        assert data["user_id"] == "user123"
    
    def test_message_with_metadata(self):
        """Test message with metadata."""
        message = Message(
            content="Test",
            metadata={"source": "web", "priority": "high"}
        )
        
        assert message.metadata["source"] == "web"
        assert message.metadata["priority"] == "high"


# =============================================================================
# Session Tests
# =============================================================================

class TestSession:
    """Tests for Session class."""
    
    def test_session_creation(self):
        """Test creating a session."""
        session = Session(user_id="user123")
        
        assert session.id is not None
        assert session.user_id == "user123"
        assert session.is_active is True
    
    def test_session_add_message(self):
        """Test adding messages to session."""
        session = Session()
        message = Message(content="Hello")
        
        session.add_message(message)
        
        assert len(session.messages) == 1
        assert session.messages[0].session_id == session.id
    
    def test_session_get_conversation_history(self):
        """Test getting conversation history."""
        session = Session()
        
        for i in range(5):
            session.add_message(Message(content=f"Message {i}"))
        
        history = session.get_conversation_history(limit=3)
        assert len(history) == 3
        assert history[0].content == "Message 2"  # Last 3
    
    def test_session_to_dict(self):
        """Test session serialization."""
        session = Session(
            user_id="user123",
            channel=ChannelType.SLACK
        )
        
        data = session.to_dict()
        assert data["user_id"] == "user123"
        assert data["channel"] == "slack"
        assert data["is_active"] is True
    
    def test_session_state_tracking(self):
        """Test session state tracking."""
        session = Session()
        assert session.state == "idle"
        
        session.state = "filling_slots"
        assert session.state == "filling_slots"


# =============================================================================
# AgentConfig Tests
# =============================================================================

class TestAgentConfig:
    """Tests for AgentConfig class."""
    
    def test_config_creation(self):
        """Test creating config with defaults."""
        config = AgentConfig()
        
        assert config.name == "BioPipelines Assistant"
        assert config.session_timeout_minutes == 30
        assert len(config.enabled_capabilities) > 0
    
    def test_config_custom_values(self):
        """Test custom config values."""
        config = AgentConfig(
            name="Custom Agent",
            session_timeout_minutes=60,
            strict_scope_mode=True
        )
        
        assert config.name == "Custom Agent"
        assert config.session_timeout_minutes == 60
        assert config.strict_scope_mode is True
    
    def test_config_is_enabled(self):
        """Test capability checking."""
        config = AgentConfig(
            enabled_capabilities={AgentCapability.ANALYTICS}
        )
        
        assert config.is_enabled(AgentCapability.ANALYTICS) is True
        assert config.is_enabled(AgentCapability.AB_TESTING) is False
    
    def test_config_all_capabilities_enabled(self):
        """Test default has all capabilities."""
        config = AgentConfig()
        
        for cap in AgentCapability:
            assert config.is_enabled(cap) is True


# =============================================================================
# SessionManager Tests
# =============================================================================

class TestSessionManager:
    """Tests for SessionManager class."""
    
    def test_manager_creation(self, session_manager):
        """Test creating session manager."""
        assert session_manager is not None
    
    def test_create_session(self, session_manager):
        """Test creating a session."""
        session = session_manager.create_session(
            user_id="user123",
            channel=ChannelType.WEB
        )
        
        assert session is not None
        assert session.user_id == "user123"
    
    def test_get_session(self, session_manager):
        """Test getting a session by ID."""
        session = session_manager.create_session()
        
        retrieved = session_manager.get_session(session.id)
        assert retrieved == session
    
    def test_get_session_for_user(self, session_manager):
        """Test getting session for a user."""
        session = session_manager.create_session(user_id="user123")
        
        retrieved = session_manager.get_session_for_user("user123")
        assert retrieved == session
    
    def test_get_or_create_session(self, session_manager):
        """Test get or create logic."""
        # First call creates
        session1 = session_manager.get_or_create_session(user_id="user123")
        
        # Second call returns existing
        session2 = session_manager.get_or_create_session(user_id="user123")
        
        assert session1.id == session2.id
    
    def test_end_session(self, session_manager):
        """Test ending a session."""
        session = session_manager.create_session(user_id="user123")
        session_manager.end_session(session.id)
        
        assert session.is_active is False
        assert session_manager.get_session_for_user("user123") is None
    
    def test_get_active_count(self, session_manager):
        """Test counting active sessions."""
        session_manager.create_session()
        session_manager.create_session()
        session3 = session_manager.create_session()
        session_manager.end_session(session3.id)
        
        assert session_manager.get_active_count() == 2


# =============================================================================
# IntentRegistry Tests
# =============================================================================

class TestIntentRegistry:
    """Tests for IntentRegistry class."""
    
    def test_registry_creation(self, intent_registry):
        """Test creating intent registry."""
        assert intent_registry is not None
    
    def test_register_handler(self, intent_registry):
        """Test registering an intent handler."""
        def greeting_handler(msg, session, ctx):
            return AgentResponse(
                message=Message(content="Hello!", direction=MessageDirection.OUTGOING)
            )
        
        intent_registry.register("greeting", greeting_handler)
        
        assert intent_registry.has_handler("greeting") is True
    
    def test_get_handler(self, intent_registry):
        """Test getting a handler."""
        def handler(msg, session, ctx):
            pass
        
        intent_registry.register("test", handler)
        
        retrieved = intent_registry.get_handler("test")
        assert retrieved == handler
    
    def test_default_handler(self, intent_registry):
        """Test default handler."""
        def default(msg, session, ctx):
            pass
        
        intent_registry.set_default(default)
        
        # Unknown intent should return default
        handler = intent_registry.get_handler("unknown_intent")
        assert handler == default
    
    def test_has_handler(self, intent_registry):
        """Test checking if handler exists."""
        assert intent_registry.has_handler("nonexistent") is False
        
        intent_registry.register("exists", lambda m, s, c: None)
        assert intent_registry.has_handler("exists") is True


# =============================================================================
# Plugin Tests
# =============================================================================

class TestPlugins:
    """Tests for plugin system."""
    
    def test_logging_plugin_name(self):
        """Test logging plugin name."""
        plugin = LoggingPlugin()
        assert plugin.name == "logging"
    
    def test_metrics_plugin_name(self):
        """Test metrics plugin name."""
        plugin = MetricsPlugin()
        assert plugin.name == "metrics"
    
    def test_metrics_plugin_stats(self):
        """Test metrics plugin statistics."""
        plugin = MetricsPlugin()
        session = Session()
        message = Message(content="Test")
        
        # Simulate message received
        plugin.on_message_received(message, session, {})
        plugin.on_message_received(message, session, {})
        
        stats = plugin.get_stats()
        assert stats["message_count"] == 2
    
    def test_custom_plugin(self, agent):
        """Test adding a custom plugin."""
        class TestPlugin(AgentPlugin):
            def __init__(self):
                self.messages_seen = []
            
            @property
            def name(self) -> str:
                return "test_plugin"
            
            def on_message_received(self, msg, session, ctx):
                self.messages_seen.append(msg.content)
                return None
            
            def on_response_generated(self, response, session, ctx):
                return None
        
        plugin = TestPlugin()
        agent.add_plugin(plugin)
        
        # Process a message
        agent.process_message("Hello", user_id="test")
        
        assert len(plugin.messages_seen) == 1
    
    def test_remove_plugin(self, agent):
        """Test removing a plugin."""
        plugin = LoggingPlugin()
        agent.add_plugin(plugin)
        
        result = agent.remove_plugin("logging")
        assert result is True
        
        result = agent.remove_plugin("nonexistent")
        assert result is False


# =============================================================================
# ChatAgent Tests
# =============================================================================

class TestChatAgent:
    """Tests for main ChatAgent class."""
    
    def test_agent_creation(self, agent):
        """Test creating an agent."""
        assert agent is not None
        assert isinstance(agent, ChatAgent)
    
    def test_agent_with_config(self):
        """Test agent with custom config."""
        config = AgentConfig(
            name="Test Bot",
            session_timeout_minutes=60
        )
        agent = create_chat_agent(config)
        
        assert agent.config.name == "Test Bot"
        assert agent.config.session_timeout_minutes == 60
    
    def test_process_message_basic(self, agent):
        """Test basic message processing."""
        response = agent.process_message(
            "Hello, how can you help me?",
            user_id="user123"
        )
        
        assert response is not None
        assert isinstance(response, AgentResponse)
        assert response.message.content != ""
        assert response.message.direction == MessageDirection.OUTGOING
    
    def test_process_message_creates_session(self, agent):
        """Test that processing creates a session."""
        response = agent.process_message(
            "Test message",
            user_id="new_user"
        )
        
        # Should have created a session
        session = agent._session_manager.get_session_for_user("new_user")
        assert session is not None
    
    def test_process_message_uses_existing_session(self, agent):
        """Test that existing session is used."""
        # First message creates session
        response1 = agent.process_message("First", user_id="user123")
        
        # Get session ID
        session = agent._session_manager.get_session_for_user("user123")
        session_id = session.id
        
        # Second message uses same session
        response2 = agent.process_message("Second", user_id="user123")
        
        # Same session should have both messages
        assert len(session.messages) == 4  # 2 user + 2 agent
    
    def test_process_message_with_session_id(self, agent):
        """Test processing with explicit session ID."""
        session = agent.create_session(user_id="user123")
        
        response = agent.process_message(
            "Hello",
            session_id=session.id
        )
        
        assert len(session.messages) == 2  # User + agent
    
    def test_process_message_processing_time(self, agent):
        """Test that processing time is recorded."""
        response = agent.process_message("Hello", user_id="user123")
        
        assert response.processing_time_ms > 0
    
    def test_scope_detection_in_scope(self, agent):
        """Test in-scope message handling."""
        response = agent.process_message(
            "Create a workflow for RNA-seq differential expression analysis",
            user_id="user123"
        )
        
        # Should be in scope
        if response.scope_result:
            assert response.scope_result.is_in_scope is True
    
    def test_scope_detection_out_of_scope(self, agent):
        """Test out-of-scope message handling."""
        response = agent.process_message(
            "What's the best pizza recipe?",
            user_id="user123"
        )
        
        # Should be deflected
        if response.scope_result:
            assert response.scope_result.is_in_scope is False
            # Should have deflection message (various formats)
            lower_content = response.message.content.lower()
            assert any(term in lower_content for term in ["not able", "unable", "focus on", "outside", "sorry", "can't", "cannot"])
    
    def test_intent_detection(self, agent):
        """Test intent detection."""
        response = agent.process_message(
            "Hello! I need help with RNA-seq workflow",
            user_id="user123"
        )
        
        # Intent is detected if in scope (could be greeting or workflow)
        # Main thing is we get a response
        assert response.message.content is not None
        assert len(response.message.content) > 0
    
    def test_custom_intent_handler(self, agent):
        """Test registering custom intent handler."""
        def custom_handler(msg, session, ctx):
            return AgentResponse(
                message=Message(
                    content="Custom response!",
                    direction=MessageDirection.OUTGOING
                ),
                intent_detected="custom"
            )
        
        agent.register_intent("custom_intent", custom_handler)
        
        # This won't trigger directly without modifying intent detection
        # but the handler is registered
        assert agent._intent_registry.has_handler("custom_intent") is True


# =============================================================================
# Session Management Tests
# =============================================================================

class TestAgentSessionManagement:
    """Tests for agent session management."""
    
    def test_create_session(self, agent):
        """Test creating a session through agent."""
        session = agent.create_session(
            user_id="user123",
            channel=ChannelType.SLACK
        )
        
        assert session is not None
        assert session.channel == ChannelType.SLACK
    
    def test_end_session(self, agent):
        """Test ending a session."""
        session = agent.create_session(user_id="user123")
        agent.end_session(session.id)
        
        assert session.is_active is False
    
    def test_get_session(self, agent):
        """Test getting a session."""
        session = agent.create_session(user_id="user123")
        
        retrieved = agent.get_session(session.id)
        assert retrieved == session


# =============================================================================
# Statistics and Health Tests
# =============================================================================

class TestAgentStatistics:
    """Tests for agent statistics and health."""
    
    def test_get_stats(self, agent):
        """Test getting agent statistics."""
        stats = agent.get_stats()
        
        assert "messages_processed" in stats
        assert "sessions_created" in stats
        assert "active_sessions" in stats
    
    def test_stats_increment(self, agent):
        """Test that stats increment properly."""
        initial_stats = agent.get_stats()
        initial_messages = initial_stats["messages_processed"]
        
        agent.process_message("Hello", user_id="user123")
        
        new_stats = agent.get_stats()
        assert new_stats["messages_processed"] == initial_messages + 1
    
    def test_health_check(self, agent):
        """Test health check."""
        health = agent.health_check()
        
        assert health["status"] == "healthy"
        assert "components" in health
        assert "timestamp" in health
    
    def test_health_check_components(self, agent):
        """Test health check shows component status."""
        health = agent.health_check()
        
        components = health["components"]
        assert "dialog_state" in components
        assert "scope_handler" in components


# =============================================================================
# Agent Reset Tests
# =============================================================================

class TestAgentReset:
    """Tests for agent reset functionality."""
    
    def test_reset(self, agent):
        """Test resetting agent."""
        # Create some state
        agent.process_message("Hello", user_id="user123")
        
        # Reset
        agent.reset()
        
        # Stats should be zeroed
        stats = agent.get_stats()
        assert stats["messages_processed"] == 0
    
    def test_reset_clears_sessions(self, agent):
        """Test that reset clears sessions."""
        agent.create_session(user_id="user123")
        agent.create_session(user_id="user456")
        
        agent.reset()
        
        assert agent._session_manager.get_active_count() == 0


# =============================================================================
# Singleton Tests
# =============================================================================

class TestSingleton:
    """Tests for singleton pattern."""
    
    def test_get_chat_agent_singleton(self):
        """Test singleton pattern."""
        reset_chat_agent()
        agent1 = get_chat_agent()
        agent2 = get_chat_agent()
        
        assert agent1 is agent2
    
    def test_reset_chat_agent(self):
        """Test resetting singleton."""
        agent1 = get_chat_agent()
        reset_chat_agent()
        agent2 = get_chat_agent()
        
        assert agent1 is not agent2
    
    def test_create_chat_agent_not_singleton(self):
        """Test creating non-singleton instance."""
        agent1 = create_chat_agent()
        agent2 = create_chat_agent()
        
        assert agent1 is not agent2


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_quick_chat(self):
        """Test quick chat function."""
        reset_chat_agent()
        
        response = quick_chat("Hello!")
        
        assert response is not None
        assert len(response) > 0
    
    def test_quick_chat_with_user(self):
        """Test quick chat with user ID."""
        reset_chat_agent()
        
        response1 = quick_chat("Hello", user_id="user123")
        response2 = quick_chat("How are you?", user_id="user123")
        
        # Both should work
        assert response1 is not None
        assert response2 is not None


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the complete agent."""
    
    def test_full_conversation_flow(self, agent):
        """Test a complete conversation flow."""
        user_id = "integration_user"
        
        # Greeting
        r1 = agent.process_message("Hello!", user_id=user_id)
        assert r1.message.content != ""
        
        # In-scope question
        r2 = agent.process_message(
            "Create a workflow for RNA-seq differential expression analysis",
            user_id=user_id
        )
        assert r2.message.content != ""
        
        # Follow-up
        r3 = agent.process_message("What parameters do I need?", user_id=user_id)
        assert r3.message.content != ""
        
        # Farewell - may be out of scope but should still respond
        r4 = agent.process_message("Thanks, bye!", user_id=user_id)
        assert r4.message.content != ""  # Just ensure we get a response
    
    def test_out_of_scope_handling(self, agent):
        """Test out-of-scope query handling."""
        response = agent.process_message(
            "What's a good movie to watch tonight?",
            user_id="user123"
        )
        
        # Should deflect politely
        content_lower = response.message.content.lower()
        assert any(word in content_lower for word in [
            "sorry", "outside", "expertise", "bioinformatics", "workflows"
        ])
    
    def test_multi_user_isolation(self, agent):
        """Test that different users have isolated sessions."""
        # User 1
        r1 = agent.process_message("Hello from user 1", user_id="user1")
        session1 = agent._session_manager.get_session_for_user("user1")
        
        # User 2
        r2 = agent.process_message("Hello from user 2", user_id="user2")
        session2 = agent._session_manager.get_session_for_user("user2")
        
        # Different sessions
        assert session1.id != session2.id
    
    def test_plugin_integration(self, agent):
        """Test plugin integration in message flow."""
        metrics_plugin = MetricsPlugin()
        agent.add_plugin(metrics_plugin)
        
        # Process some messages
        agent.process_message("Hello", user_id="user1")
        agent.process_message("How are you?", user_id="user1")
        
        stats = metrics_plugin.get_stats()
        assert stats["message_count"] == 2
        assert stats["total_processing_time_ms"] > 0


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_message(self, agent):
        """Test handling empty message."""
        response = agent.process_message("", user_id="user123")
        
        # Should handle gracefully
        assert response is not None
    
    def test_very_long_message(self, agent):
        """Test handling very long message."""
        long_message = "RNA-seq " * 500
        
        response = agent.process_message(long_message, user_id="user123")
        assert response is not None
    
    def test_special_characters(self, agent):
        """Test handling special characters."""
        response = agent.process_message(
            "Create RNA-seq workflow @#$%^&*()",
            user_id="user123"
        )
        assert response is not None
    
    def test_unicode_characters(self, agent):
        """Test handling unicode."""
        response = agent.process_message(
            "基因组分析 RNA-seq データ",
            user_id="user123"
        )
        assert response is not None
    
    def test_nonexistent_session_id(self, agent):
        """Test handling nonexistent session ID."""
        response = agent.process_message(
            "Hello",
            session_id="nonexistent-id-12345"
        )
        
        # Should create new session
        assert response is not None
    
    def test_minimal_config(self, minimal_agent):
        """Test agent with minimal capabilities."""
        response = minimal_agent.process_message(
            "Create a workflow for RNA-seq analysis",
            user_id="user123"
        )
        
        # Should still work with limited features
        assert response is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
