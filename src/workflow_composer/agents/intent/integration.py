"""
Intent System Integration
=========================

This module provides integration helpers for the enhanced intent system.

It bridges the DialogueManager with the BioPipelines facade.
"""

import logging
from typing import Any, Dict, List, Optional, Callable, Tuple

from .parser import IntentParser, IntentType, IntentResult, Entity, EntityType
from .context import ConversationContext, ContextMemory, EntityTracker
from .dialogue import (
    DialogueManager,
    DialogueResult,
    ConversationState,
    ConversationPhase,
    TaskState,
)

logger = logging.getLogger(__name__)


class ChatIntegration:
    """
    Integration layer between DialogueManager and BioPipelines facade.
    
    Usage:
        from workflow_composer.agents.intent.integration import ChatIntegration
        
        intent_system = ChatIntegration()
        result = intent_system.process_message(message, session_id)
        
        if result.should_execute_tool:
            tool_result = execute_tool(result.tool_name, result.tool_args)
            follow_up = intent_system.handle_tool_result(
                session_id, result.tool_name, tool_result
            )
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize the integration layer.
        
        Args:
            llm_client: Optional LLM client for complex intent parsing
        """
        # Create shared components
        self.parser = IntentParser(llm_client=llm_client)
        
        # Session-based dialogue managers
        self._sessions: Dict[str, DialogueManager] = {}
    
    def get_dialogue_manager(self, session_id: str) -> DialogueManager:
        """Get or create dialogue manager for session."""
        if session_id not in self._sessions:
            context = ConversationContext()
            self._sessions[session_id] = DialogueManager(
                intent_parser=self.parser,
                context=context
            )
        return self._sessions[session_id]
    
    def process_message(
        self, 
        message: str, 
        session_id: str = "default"
    ) -> DialogueResult:
        """
        Process a user message and return the dialogue result.
        
        Args:
            message: User's message
            session_id: Session identifier
            
        Returns:
            DialogueResult with intent, tool to call, and any prompts
        """
        dm = self.get_dialogue_manager(session_id)
        return dm.process_message(message)
    
    def handle_tool_result(
        self,
        session_id: str,
        tool_name: str,
        result: Any
    ) -> Optional[DialogueResult]:
        """
        Handle the result of a tool execution.
        
        For multi-step tasks, this may return a follow-up action.
        
        Args:
            session_id: Session identifier
            tool_name: Name of the tool that was executed
            result: Result from the tool
            
        Returns:
            Follow-up DialogueResult if needed, None otherwise
        """
        dm = self.get_dialogue_manager(session_id)
        return dm.handle_tool_result(tool_name, result)
    
    def get_context_for_llm(self, session_id: str) -> Dict[str, Any]:
        """
        Get conversation context for LLM prompting.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with context summary, recent messages, entities
        """
        dm = self.get_dialogue_manager(session_id)
        return dm.get_context_for_llm()
    
    def set_context_state(
        self, 
        session_id: str, 
        key: str, 
        value: Any
    ):
        """
        Set a context state variable (e.g., data_path, current_workflow).
        
        Args:
            session_id: Session identifier
            key: State key
            value: State value
        """
        dm = self.get_dialogue_manager(session_id)
        dm.context.set_state(key, value)
    
    def add_entity(
        self, 
        session_id: str, 
        entity: Entity
    ):
        """
        Add an entity to the conversation context.
        
        Useful when tools discover entities that should be tracked.
        
        Args:
            session_id: Session identifier
            entity: Entity to add
        """
        dm = self.get_dialogue_manager(session_id)
        dm.context.memory.add_entity(entity)
    
    def clear_session(self, session_id: str):
        """Clear session data."""
        if session_id in self._sessions:
            del self._sessions[session_id]


# =============================================================================
# TOOL ROUTING HELPER
# =============================================================================

def route_to_tool(result: DialogueResult) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Extract tool name and arguments from a dialogue result.
    
    Args:
        result: DialogueResult from dialogue manager
        
    Returns:
        Tuple of (tool_name, tool_args) or None if no tool should be called
    """
    if result.should_execute_tool:
        return (result.tool_name, result.tool_args)
    return None


def format_clarification_response(result: DialogueResult) -> str:
    """
    Format a clarification/error message from dialogue result.
    
    Args:
        result: DialogueResult that needs clarification
        
    Returns:
        Formatted message string
    """
    if result.message:
        return result.message
    
    if result.missing_slots:
        from .dialogue import SlotFiller
        return SlotFiller.get_prompts_for_slots(result.missing_slots)
    
    return "I need more information to help you. Could you be more specific?"


# =============================================================================
# EXAMPLE INTEGRATION
# =============================================================================

def example_integration():
    """
    Example of how to integrate with BioPipelines facade.
    
    This shows the pattern for using DialogueManager alongside
    the BioPipelines.chat() method.
    """
    
    # Initialize
    integration = ChatIntegration()
    session_id = "user_123"
    
    # Simulate conversation
    messages = [
        "Check if we have brain RNA-seq data, if not search online",
        "Download the first one",  # Uses coreference
        "Create a workflow for it",
        "yes",  # Confirmation
    ]
    
    for msg in messages:
        print(f"\nUser: {msg}")
        
        result = integration.process_message(msg, session_id)
        
        if result.action == "execute_tool":
            print(f"  → Calling tool: {result.tool_name}")
            print(f"  → Args: {result.tool_args}")
            
            # Simulate tool result
            if result.tool_name == "scan_data":
                # No local data found, check for follow-up
                mock_result = {"samples": []}
                follow_up = integration.handle_tool_result(
                    session_id, result.tool_name, mock_result
                )
                if follow_up and follow_up.should_execute_tool:
                    print(f"  → Follow-up: {follow_up.tool_name}")
        
        elif result.action == "slot_fill":
            print(f"  → Need info: {result.message}")
        
        elif result.action == "greeting":
            print(f"  → Response: {result.message[:100]}...")
        
        elif result.action == "llm_response":
            print(f"  → Defer to LLM")
            context = integration.get_context_for_llm(session_id)
            print(f"  → Context: {context['summary'][:100]}...")


if __name__ == "__main__":
    example_integration()
