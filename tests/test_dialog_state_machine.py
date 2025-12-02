"""
Tests for Dialog State Machine
==============================

Comprehensive tests for the formal FSM dialog management.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from workflow_composer.agents.intent.dialog_state_machine import (
    DialogStateMachine,
    DialogState,
    DialogEvent,
    DialogContext,
    StateTransition,
    StateConfig,
    StateHistoryEntry,
    StateHandler,
    IdleHandler,
    SlotFillingHandler,
    ConfirmingHandler,
    ErrorRecoveryHandler,
    get_dialog_state_machine,
    create_dialog_state_machine,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def fsm():
    """Create a fresh state machine for each test."""
    return DialogStateMachine()


@pytest.fixture
def context():
    """Create a fresh context for each test."""
    return DialogContext(session_id="test_session")


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestDialogStateMachineInit:
    """Tests for FSM initialization."""
    
    def test_initial_state_is_idle(self, fsm):
        """FSM should start in IDLE state."""
        assert fsm.current_state == DialogState.IDLE
    
    def test_custom_initial_state(self):
        """Can start in a different state."""
        fsm = DialogStateMachine(initial_state=DialogState.SLOT_FILLING)
        assert fsm.current_state == DialogState.SLOT_FILLING
    
    def test_has_transitions(self, fsm):
        """FSM should have default transitions."""
        assert len(fsm._transitions) > 0
    
    def test_has_state_configs(self, fsm):
        """FSM should have state configurations."""
        assert DialogState.IDLE in fsm._state_configs
        assert DialogState.SLOT_FILLING in fsm._state_configs
    
    def test_history_is_empty(self, fsm):
        """History should be empty initially."""
        assert len(fsm._history) == 0
    
    def test_time_in_state_starts_at_zero(self, fsm):
        """Time in state should be near zero initially."""
        assert fsm.time_in_state.total_seconds() < 1


# =============================================================================
# STATE TRANSITION TESTS
# =============================================================================

class TestStateTransitions:
    """Tests for state transitions."""
    
    def test_idle_to_understanding_on_user_input(self, fsm, context):
        """IDLE -> UNDERSTANDING on USER_INPUT."""
        context.query = "scan my data"
        new_state = fsm.process_event(DialogEvent.USER_INPUT, context)
        
        assert new_state == DialogState.UNDERSTANDING
        assert fsm.current_state == DialogState.UNDERSTANDING
    
    def test_understanding_to_executing_high_confidence(self, fsm, context):
        """UNDERSTANDING -> EXECUTING when intent is clear."""
        # Setup: move to UNDERSTANDING
        fsm.transition_to(DialogState.UNDERSTANDING, context)
        
        # Set high confidence intent
        context.intent = "DATA_SCAN"
        context.intent_confidence = 0.9
        context.missing_slots = []
        
        new_state = fsm.process_event(DialogEvent.INTENT_RECOGNIZED, context)
        
        assert new_state == DialogState.EXECUTING
    
    def test_understanding_to_slot_filling_missing_slots(self, fsm, context):
        """UNDERSTANDING -> SLOT_FILLING when slots are missing."""
        fsm.transition_to(DialogState.UNDERSTANDING, context)
        
        context.intent = "DATA_SEARCH"
        context.intent_confidence = 0.8
        context.missing_slots = ["query", "organism"]
        
        new_state = fsm.process_event(DialogEvent.SLOTS_MISSING, context)
        
        assert new_state == DialogState.SLOT_FILLING
    
    def test_understanding_to_disambiguation_ambiguous(self, fsm, context):
        """UNDERSTANDING -> DISAMBIGUATION when multiple intents possible."""
        fsm.transition_to(DialogState.UNDERSTANDING, context)
        
        context.intent = "DATA_SCAN"
        context.intent_confidence = 0.5
        context.alternative_intents = [
            ("DATA_SCAN", 0.5),
            ("DATA_SEARCH", 0.45),
        ]
        
        new_state = fsm.process_event(DialogEvent.INTENT_AMBIGUOUS, context)
        
        assert new_state == DialogState.DISAMBIGUATION
    
    def test_understanding_to_confirming_medium_confidence(self, fsm, context):
        """UNDERSTANDING -> CONFIRMING for medium confidence."""
        fsm.transition_to(DialogState.UNDERSTANDING, context)
        
        context.intent = "WORKFLOW_CREATE"
        context.intent_confidence = 0.55  # Medium confidence
        context.missing_slots = []
        
        new_state = fsm.process_event(DialogEvent.INTENT_RECOGNIZED, context)
        
        assert new_state == DialogState.CONFIRMING
    
    def test_slot_filling_loop(self, fsm, context):
        """SLOT_FILLING stays in SLOT_FILLING while slots missing."""
        fsm.transition_to(DialogState.SLOT_FILLING, context)
        
        context.missing_slots = ["query"]
        
        new_state = fsm.process_event(DialogEvent.USER_INPUT, context)
        
        assert new_state == DialogState.SLOT_FILLING
    
    def test_slot_filling_to_confirming(self, fsm, context):
        """SLOT_FILLING -> CONFIRMING when slots complete."""
        fsm.transition_to(DialogState.SLOT_FILLING, context)
        
        context.missing_slots = []
        
        new_state = fsm.process_event(DialogEvent.SLOTS_COMPLETE, context)
        
        assert new_state == DialogState.CONFIRMING
    
    def test_confirming_to_executing_on_confirm(self, fsm, context):
        """CONFIRMING -> EXECUTING on USER_CONFIRM."""
        fsm.transition_to(DialogState.CONFIRMING, context)
        
        new_state = fsm.process_event(DialogEvent.USER_CONFIRM, context)
        
        assert new_state == DialogState.EXECUTING
    
    def test_confirming_to_idle_on_deny(self, fsm, context):
        """CONFIRMING -> IDLE on USER_DENY."""
        fsm.transition_to(DialogState.CONFIRMING, context)
        
        new_state = fsm.process_event(DialogEvent.USER_DENY, context)
        
        assert new_state == DialogState.IDLE
    
    def test_executing_to_presenting_on_success(self, fsm, context):
        """EXECUTING -> PRESENTING on TOOL_SUCCESS."""
        fsm.transition_to(DialogState.EXECUTING, context)
        
        new_state = fsm.process_event(DialogEvent.TOOL_SUCCESS, context)
        
        assert new_state == DialogState.PRESENTING
    
    def test_executing_to_error_on_failure(self, fsm, context):
        """EXECUTING -> ERROR_RECOVERY on TOOL_FAILED."""
        fsm.transition_to(DialogState.EXECUTING, context)
        
        new_state = fsm.process_event(DialogEvent.TOOL_FAILED, context)
        
        assert new_state == DialogState.ERROR_RECOVERY
    
    def test_cancel_returns_to_idle(self, fsm, context):
        """CANCEL event should return to IDLE from most states."""
        # Test from SLOT_FILLING
        fsm.transition_to(DialogState.SLOT_FILLING, context)
        new_state = fsm.process_event(DialogEvent.CANCEL, context)
        assert new_state == DialogState.IDLE


# =============================================================================
# GUARD CONDITION TESTS
# =============================================================================

class TestGuardConditions:
    """Tests for transition guard conditions."""
    
    def test_high_confidence_guard(self, fsm, context):
        """High confidence transitions require >= 0.7 confidence."""
        fsm.transition_to(DialogState.UNDERSTANDING, context)
        
        # Low confidence - should not transition to EXECUTING
        context.intent = "DATA_SCAN"
        context.intent_confidence = 0.5
        context.missing_slots = []
        
        # This should go to CONFIRMING, not EXECUTING
        new_state = fsm.process_event(DialogEvent.INTENT_RECOGNIZED, context)
        
        assert new_state == DialogState.CONFIRMING
    
    def test_missing_slots_guard(self, fsm, context):
        """Slot filling only triggers when slots are actually missing."""
        fsm.transition_to(DialogState.UNDERSTANDING, context)
        
        context.intent = "DATA_SCAN"
        context.intent_confidence = 0.8
        context.missing_slots = []  # No missing slots
        
        # Should NOT go to SLOT_FILLING
        new_state = fsm.process_event(DialogEvent.SLOTS_MISSING, context)
        
        # Guard fails, so no transition
        assert new_state is None or fsm.current_state != DialogState.SLOT_FILLING
    
    def test_ambiguous_intent_guard(self, fsm, context):
        """Disambiguation requires multiple alternative intents."""
        fsm.transition_to(DialogState.UNDERSTANDING, context)
        
        context.alternative_intents = [("DATA_SCAN", 0.5)]  # Only one
        
        # Should NOT go to DISAMBIGUATION
        new_state = fsm.process_event(DialogEvent.INTENT_AMBIGUOUS, context)
        
        # Guard should fail
        assert new_state is None


# =============================================================================
# HISTORY AND ROLLBACK TESTS
# =============================================================================

class TestHistoryAndRollback:
    """Tests for state history and rollback."""
    
    def test_history_recorded_on_transition(self, fsm, context):
        """State history should be recorded on transitions."""
        fsm.process_event(DialogEvent.USER_INPUT, context)
        
        assert len(fsm._history) == 1
        assert fsm._history[0].state == DialogState.IDLE
    
    def test_rollback_one_step(self, fsm, context):
        """Can rollback one step."""
        # Make a transition
        fsm.process_event(DialogEvent.USER_INPUT, context)
        assert fsm.current_state == DialogState.UNDERSTANDING
        
        # Rollback
        old_state = fsm.rollback()
        
        assert old_state == DialogState.IDLE
        assert fsm.current_state == DialogState.IDLE
    
    def test_rollback_multiple_steps(self, fsm, context):
        """Can rollback multiple steps."""
        # Make multiple transitions
        fsm.process_event(DialogEvent.USER_INPUT, context)  # IDLE -> UNDERSTANDING
        
        context.intent = "DATA_SCAN"
        context.intent_confidence = 0.55
        fsm.process_event(DialogEvent.INTENT_RECOGNIZED, context)  # -> CONFIRMING
        
        # Rollback 2 steps
        old_state = fsm.rollback(steps=2)
        
        assert old_state == DialogState.IDLE
    
    def test_rollback_fails_if_not_enough_history(self, fsm, context):
        """Rollback fails gracefully if not enough history."""
        result = fsm.rollback(steps=5)
        
        assert result is None
    
    def test_history_is_trimmed(self, fsm, context):
        """History should be trimmed to max length."""
        fsm._max_history = 3
        
        # Make many transitions
        for i in range(10):
            fsm._record_history(DialogState.IDLE, DialogEvent.USER_INPUT, context)
        
        assert len(fsm._history) <= 3


# =============================================================================
# TIMEOUT TESTS
# =============================================================================

class TestTimeouts:
    """Tests for state timeouts."""
    
    def test_state_has_timeout_config(self, fsm):
        """States should have timeout configurations."""
        config = fsm._state_configs.get(DialogState.SLOT_FILLING)
        
        assert config is not None
        assert config.timeout_seconds > 0
    
    def test_idle_has_no_timeout(self, fsm):
        """IDLE state should have no timeout."""
        config = fsm._state_configs.get(DialogState.IDLE)
        
        assert config.timeout_seconds == 0
    
    def test_timeout_detection(self, fsm, context):
        """Timeout should be detected."""
        fsm.transition_to(DialogState.SLOT_FILLING, context)
        
        # Simulate time passing
        fsm._state_entered_at = datetime.now() - timedelta(seconds=200)
        
        timed_out = fsm._check_timeout(context)
        
        assert timed_out
    
    def test_no_timeout_when_within_limit(self, fsm, context):
        """No timeout when within limit."""
        fsm.transition_to(DialogState.SLOT_FILLING, context)
        
        # Recent entry
        fsm._state_entered_at = datetime.now() - timedelta(seconds=5)
        
        timed_out = fsm._check_timeout(context)
        
        assert not timed_out


# =============================================================================
# FORCED TRANSITION TESTS
# =============================================================================

class TestForcedTransitions:
    """Tests for forced transitions."""
    
    def test_force_transition(self, fsm, context):
        """Can force transition to any state."""
        result = fsm.transition_to(DialogState.ERROR_RECOVERY, context)
        
        assert result is True
        assert fsm.current_state == DialogState.ERROR_RECOVERY
    
    def test_can_transition_check(self, fsm):
        """Can check if transition is possible."""
        # From IDLE, should be able to go to UNDERSTANDING via USER_INPUT
        assert fsm.can_transition_to(DialogState.UNDERSTANDING)
        
        # From IDLE, should not directly go to PRESENTING (no direct transition)
        assert not fsm.can_transition_to(DialogState.PRESENTING)


# =============================================================================
# RESET TESTS
# =============================================================================

class TestReset:
    """Tests for reset functionality."""
    
    def test_reset_returns_to_idle(self, fsm, context):
        """Reset should return to IDLE state."""
        fsm.transition_to(DialogState.EXECUTING, context)
        
        fsm.reset(context)
        
        assert fsm.current_state == DialogState.IDLE
    
    def test_reset_clears_history(self, fsm, context):
        """Reset should clear history."""
        fsm.process_event(DialogEvent.USER_INPUT, context)
        fsm.reset(context)
        
        assert len(fsm._history) == 0
    
    def test_reset_clears_context(self, fsm, context):
        """Reset should clear context slots."""
        context.slots = {"query": "test"}
        context.pending_action = "scan"
        
        fsm.reset(context)
        
        assert context.slots == {}
        assert context.pending_action is None


# =============================================================================
# INTROSPECTION TESTS
# =============================================================================

class TestIntrospection:
    """Tests for FSM introspection methods."""
    
    def test_get_available_events(self, fsm):
        """Can get events available from current state."""
        events = fsm.get_available_events()
        
        assert DialogEvent.USER_INPUT in events
    
    def test_get_reachable_states(self, fsm):
        """Can get states reachable from current state."""
        states = fsm.get_reachable_states()
        
        assert DialogState.UNDERSTANDING in states
    
    def test_get_state_info(self, fsm):
        """Can get info about current state."""
        info = fsm.get_state_info()
        
        assert "state" in info
        assert info["state"] == "IDLE"
        assert "available_events" in info
        assert "reachable_states" in info


# =============================================================================
# STATE HANDLER TESTS
# =============================================================================

class TestStateHandlers:
    """Tests for state handlers."""
    
    def test_idle_handler_process(self):
        """IdleHandler should set query and increment turn."""
        handler = IdleHandler()
        context = DialogContext()
        
        event, message = handler.process_input("test query", context)
        
        assert event == DialogEvent.USER_INPUT
        assert context.query == "test query"
        assert context.turn_count == 1
    
    def test_slot_filling_handler_enter(self):
        """SlotFillingHandler should prompt for first missing slot."""
        handler = SlotFillingHandler()
        context = DialogContext()
        context.missing_slots = ["query", "organism"]
        
        message = handler.on_enter(context)
        
        assert "query" in message
        assert context.current_slot == "query"
    
    def test_slot_filling_handler_fills_slot(self):
        """SlotFillingHandler should fill slots from input."""
        handler = SlotFillingHandler()
        context = DialogContext()
        context.missing_slots = ["query"]
        context.current_slot = "query"
        
        event, message = handler.process_input("RNA-seq data", context)
        
        assert context.slots["query"] == "RNA-seq data"
        assert "query" not in context.missing_slots
    
    def test_slot_filling_complete(self):
        """SlotFillingHandler should signal SLOTS_COMPLETE when done."""
        handler = SlotFillingHandler()
        context = DialogContext()
        context.missing_slots = ["query"]
        context.current_slot = "query"
        
        event, _ = handler.process_input("test", context)
        
        assert event == DialogEvent.SLOTS_COMPLETE
    
    def test_confirming_handler_yes(self):
        """ConfirmingHandler should handle 'yes' confirmation."""
        handler = ConfirmingHandler()
        context = DialogContext()
        context.pending_action = "scan_data"
        
        event, _ = handler.process_input("yes", context)
        
        assert event == DialogEvent.USER_CONFIRM
    
    def test_confirming_handler_no(self):
        """ConfirmingHandler should handle 'no' denial."""
        handler = ConfirmingHandler()
        context = DialogContext()
        context.pending_action = "scan_data"
        
        event, _ = handler.process_input("no", context)
        
        assert event == DialogEvent.USER_DENY
    
    def test_confirming_handler_correction(self):
        """ConfirmingHandler should treat other input as correction."""
        handler = ConfirmingHandler()
        context = DialogContext()
        context.pending_action = "scan_data"
        
        event, _ = handler.process_input("actually search online", context)
        
        assert event == DialogEvent.USER_CORRECTION
        assert context.query == "actually search online"
    
    def test_error_recovery_handler_enter(self):
        """ErrorRecoveryHandler should show error message."""
        handler = ErrorRecoveryHandler()
        context = DialogContext()
        context.error = Exception("Test error")
        
        message = handler.on_enter(context)
        
        assert "issue" in message.lower() or "error" in message.lower()


# =============================================================================
# CONTEXT TESTS
# =============================================================================

class TestDialogContext:
    """Tests for DialogContext."""
    
    def test_reset_for_new_turn(self):
        """Context should reset transient fields."""
        context = DialogContext()
        context.query = "test"
        context.response_message = "response"
        context.error = Exception("test")
        
        context.reset_for_new_turn()
        
        assert context.query == ""
        assert context.response_message is None
        assert context.error is None
    
    def test_clear_slots(self):
        """Context should clear slot state."""
        context = DialogContext()
        context.slots = {"query": "test"}
        context.missing_slots = ["organism"]
        context.current_slot = "organism"
        
        context.clear_slots()
        
        assert context.slots == {}
        assert context.missing_slots == []
        assert context.current_slot is None
    
    def test_clear_pending(self):
        """Context should clear pending action."""
        context = DialogContext()
        context.pending_action = "test"
        context.pending_args = {"key": "value"}
        
        context.clear_pending()
        
        assert context.pending_action is None
        assert context.pending_args == {}


# =============================================================================
# FACTORY TESTS
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_get_singleton(self):
        """get_dialog_state_machine returns singleton."""
        fsm1 = get_dialog_state_machine()
        fsm2 = get_dialog_state_machine()
        
        assert fsm1 is fsm2
    
    def test_create_new_instance(self):
        """create_dialog_state_machine returns new instance."""
        fsm1 = create_dialog_state_machine()
        fsm2 = create_dialog_state_machine()
        
        assert fsm1 is not fsm2
    
    def test_create_with_initial_state(self):
        """Can create FSM with custom initial state."""
        fsm = create_dialog_state_machine(initial_state=DialogState.PRESENTING)
        
        assert fsm.current_state == DialogState.PRESENTING


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for complete flows."""
    
    def test_simple_query_flow(self, fsm, context):
        """Test: simple query -> understanding -> executing -> presenting."""
        # User input
        context.query = "scan my data"
        fsm.process_event(DialogEvent.USER_INPUT, context)
        assert fsm.current_state == DialogState.UNDERSTANDING
        
        # Intent recognized (high confidence)
        context.intent = "DATA_SCAN"
        context.intent_confidence = 0.9
        context.missing_slots = []
        fsm.process_event(DialogEvent.INTENT_RECOGNIZED, context)
        assert fsm.current_state == DialogState.EXECUTING
        
        # Tool succeeds
        fsm.process_event(DialogEvent.TOOL_SUCCESS, context)
        assert fsm.current_state == DialogState.PRESENTING
    
    def test_slot_filling_flow(self, fsm, context):
        """Test: query with missing slots -> slot filling -> confirmation."""
        # User input
        context.query = "search for data"
        fsm.process_event(DialogEvent.USER_INPUT, context)
        
        # Intent recognized but slots missing
        context.intent = "DATA_SEARCH"
        context.intent_confidence = 0.8
        context.missing_slots = ["query"]
        fsm.process_event(DialogEvent.SLOTS_MISSING, context)
        assert fsm.current_state == DialogState.SLOT_FILLING
        
        # User provides slot value (simulated)
        context.slots["query"] = "human RNA-seq"
        context.missing_slots = []
        fsm.process_event(DialogEvent.SLOTS_COMPLETE, context)
        assert fsm.current_state == DialogState.CONFIRMING
        
        # User confirms
        fsm.process_event(DialogEvent.USER_CONFIRM, context)
        assert fsm.current_state == DialogState.EXECUTING
    
    def test_error_recovery_flow(self, fsm, context):
        """Test: error -> recovery -> retry."""
        # Setup: in EXECUTING state
        fsm.transition_to(DialogState.EXECUTING, context)
        
        # Tool fails
        context.error = Exception("Connection error")
        fsm.process_event(DialogEvent.TOOL_FAILED, context)
        assert fsm.current_state == DialogState.ERROR_RECOVERY
        
        # User provides new input
        context.query = "try again"
        fsm.process_event(DialogEvent.USER_INPUT, context)
        assert fsm.current_state == DialogState.UNDERSTANDING
    
    def test_cancellation_flow(self, fsm, context):
        """Test: user cancels mid-flow."""
        # Setup: in SLOT_FILLING state
        fsm.transition_to(DialogState.SLOT_FILLING, context)
        context.slots = {"partial": "data"}
        
        # User cancels
        fsm.process_event(DialogEvent.CANCEL, context)
        
        assert fsm.current_state == DialogState.IDLE
