"""
Dialog State Machine
====================

Formal finite state machine for dialog management.

Professional chat agents (Rasa, Dialogflow, Alexa) use FSMs because they provide:
1. Predictable behavior - Every state has defined transitions
2. Debuggability - Easy to trace what happened
3. Rollback capability - Can undo states
4. Timeout handling - States can expire
5. Modularity - Each state is self-contained

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    DialogStateMachine                        │
    │  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
    │  │   IDLE   │───▶│UNDERSTAND│───▶│SLOT_FILL │──┐           │
    │  └──────────┘    └──────────┘    └──────────┘  │           │
    │       ▲               │               │        │           │
    │       │               ▼               ▼        │           │
    │       │          ┌──────────┐    ┌──────────┐  │           │
    │       │          │DISAMBIG  │    │CONFIRMING│◀─┘           │
    │       │          └──────────┘    └──────────┘              │
    │       │               │               │                    │
    │       │               ▼               ▼                    │
    │       │          ┌──────────┐    ┌──────────┐              │
    │       └──────────│EXECUTING │◀───│PRESENT   │              │
    │                  └──────────┘    └──────────┘              │
    │                       │                                    │
    │                       ▼                                    │
    │                  ┌──────────┐                              │
    │                  │ERROR_REC │                              │
    │                  └──────────┘                              │
    └─────────────────────────────────────────────────────────────┘

Usage:
    from workflow_composer.agents.intent import DialogStateMachine, DialogState
    
    fsm = DialogStateMachine()
    
    # Process event
    result = fsm.process_event("user_input", context={"query": "scan data"})
    print(fsm.current_state)  # DialogState.UNDERSTANDING
    
    # Check if transition is valid
    if fsm.can_transition_to(DialogState.EXECUTING):
        fsm.transition_to(DialogState.EXECUTING)
    
    # Rollback on error
    fsm.rollback()
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

logger = logging.getLogger(__name__)


# =============================================================================
# DIALOG STATES
# =============================================================================

class DialogState(Enum):
    """
    All possible conversation states.
    
    State Descriptions:
    - IDLE: Ready for new input, no active task
    - UNDERSTANDING: Processing/parsing user input
    - SLOT_FILLING: Collecting required information
    - DISAMBIGUATION: Clarifying ambiguous intent
    - CONFIRMING: Waiting for user confirmation
    - EXECUTING: Running a tool/action
    - PRESENTING: Showing results to user
    - FOLLOW_UP: Expecting follow-up from user
    - ERROR_RECOVERY: Handling an error
    - HUMAN_HANDOFF: Escalated to human (future)
    """
    IDLE = auto()
    UNDERSTANDING = auto()
    SLOT_FILLING = auto()
    DISAMBIGUATION = auto()
    CONFIRMING = auto()
    EXECUTING = auto()
    PRESENTING = auto()
    FOLLOW_UP = auto()
    ERROR_RECOVERY = auto()
    HUMAN_HANDOFF = auto()


class DialogEvent(Enum):
    """
    Events that can trigger state transitions.
    
    Events are triggered by:
    - User actions (input, confirmation, cancellation)
    - System actions (parsing complete, tool finished)
    - Timeouts and errors
    """
    # User Events
    USER_INPUT = auto()          # User sent a message
    USER_CONFIRM = auto()        # User confirmed action
    USER_DENY = auto()           # User denied/cancelled
    USER_CORRECTION = auto()     # User corrected intent
    
    # System Events
    INTENT_RECOGNIZED = auto()   # Intent parsed successfully
    INTENT_AMBIGUOUS = auto()    # Multiple possible intents
    INTENT_UNCLEAR = auto()      # Couldn't understand
    SLOTS_COMPLETE = auto()      # All required slots filled
    SLOTS_MISSING = auto()       # Still need more slots
    TOOL_SUCCESS = auto()        # Tool executed successfully
    TOOL_FAILED = auto()         # Tool execution failed
    RESULTS_READY = auto()       # Results ready to present
    
    # Control Events
    TIMEOUT = auto()             # State timed out
    CANCEL = auto()              # Conversation cancelled
    RESET = auto()               # Reset to idle
    ESCALATE = auto()            # Escalate to human


# =============================================================================
# TRANSITION DEFINITIONS
# =============================================================================

@dataclass
class StateTransition:
    """
    Defines a valid state transition.
    
    Attributes:
        from_state: Source state
        to_state: Destination state
        event: Event that triggers transition
        guard: Optional condition that must be true
        action: Optional action to execute on transition
        priority: Higher priority transitions checked first
    """
    from_state: DialogState
    to_state: DialogState
    event: DialogEvent
    guard: Optional[Callable[["DialogContext"], bool]] = None
    action: Optional[Callable[["DialogContext"], None]] = None
    priority: int = 0
    
    def can_fire(self, context: "DialogContext") -> bool:
        """Check if this transition can fire."""
        if self.guard is None:
            return True
        try:
            return self.guard(context)
        except Exception as e:
            logger.warning(f"Guard check failed: {e}")
            return False
    
    def execute(self, context: "DialogContext"):
        """Execute transition action."""
        if self.action:
            try:
                self.action(context)
            except Exception as e:
                logger.error(f"Transition action failed: {e}")


@dataclass
class StateConfig:
    """
    Configuration for a single state.
    
    Attributes:
        state: The dialog state
        timeout_seconds: Max time in this state (0 = no timeout)
        on_enter: Action to execute when entering state
        on_exit: Action to execute when leaving state
        on_timeout: Action when state times out
    """
    state: DialogState
    timeout_seconds: int = 0
    on_enter: Optional[Callable[["DialogContext"], Optional[str]]] = None
    on_exit: Optional[Callable[["DialogContext"], None]] = None
    on_timeout: Optional[Callable[["DialogContext"], DialogEvent]] = None


# =============================================================================
# DIALOG CONTEXT
# =============================================================================

@dataclass
class DialogContext:
    """
    Context passed through the state machine.
    
    Contains all information needed for state handling and transitions.
    """
    # Current query/input
    query: str = ""
    resolved_query: str = ""  # After coreference resolution
    
    # Intent parsing results
    intent: Optional[str] = None
    intent_confidence: float = 0.0
    alternative_intents: List[Tuple[str, float]] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    
    # Slot filling state
    slots: Dict[str, Any] = field(default_factory=dict)
    missing_slots: List[str] = field(default_factory=list)
    current_slot: Optional[str] = None
    
    # Confirmation state
    pending_action: Optional[str] = None
    pending_args: Dict[str, Any] = field(default_factory=dict)
    
    # Execution state
    tool_name: Optional[str] = None
    tool_result: Optional[Any] = None
    error: Optional[Exception] = None
    
    # Turn tracking
    turn_count: int = 0
    session_id: str = ""
    
    # Response to send
    response_message: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def reset_for_new_turn(self):
        """Reset transient fields for new turn."""
        self.query = ""
        self.resolved_query = ""
        self.response_message = None
        self.suggestions = []
        self.error = None
    
    def clear_slots(self):
        """Clear slot filling state."""
        self.slots = {}
        self.missing_slots = []
        self.current_slot = None
    
    def clear_pending(self):
        """Clear pending confirmation."""
        self.pending_action = None
        self.pending_args = {}


# =============================================================================
# STATE HISTORY
# =============================================================================

@dataclass
class StateHistoryEntry:
    """An entry in state history for rollback."""
    state: DialogState
    timestamp: datetime
    event: Optional[DialogEvent] = None
    context_snapshot: Optional[Dict[str, Any]] = None


# =============================================================================
# DIALOG STATE MACHINE
# =============================================================================

class DialogStateMachine:
    """
    Formal state machine for dialog management.
    
    Features:
    - Defined states and transitions
    - Guard conditions for conditional transitions
    - Entry/exit actions for states
    - State history for rollback
    - Timeout handling
    - Event-driven transitions
    
    Usage:
        fsm = DialogStateMachine()
        context = DialogContext(query="scan my data")
        
        # Process an event
        new_state = fsm.process_event(DialogEvent.USER_INPUT, context)
        
        # Check current state
        print(fsm.current_state)
        
        # Rollback
        fsm.rollback()
    """
    
    def __init__(self, initial_state: DialogState = DialogState.IDLE):
        """
        Initialize the state machine.
        
        Args:
            initial_state: Starting state (default: IDLE)
        """
        self._current_state = initial_state
        self._state_entered_at = datetime.now()
        
        # Transition table: (from_state, event) -> [transitions]
        self._transitions: Dict[Tuple[DialogState, DialogEvent], List[StateTransition]] = {}
        
        # State configurations
        self._state_configs: Dict[DialogState, StateConfig] = {}
        
        # History for rollback
        self._history: List[StateHistoryEntry] = []
        self._max_history = 10
        
        # Build default transitions
        self._build_default_transitions()
        self._build_state_configs()
        
        logger.info(f"DialogStateMachine initialized in state: {self._current_state.name}")
    
    @property
    def current_state(self) -> DialogState:
        """Get current state."""
        return self._current_state
    
    @property
    def time_in_state(self) -> timedelta:
        """Get time spent in current state."""
        return datetime.now() - self._state_entered_at
    
    @property
    def state_config(self) -> Optional[StateConfig]:
        """Get configuration for current state."""
        return self._state_configs.get(self._current_state)
    
    # =========================================================================
    # TRANSITION MANAGEMENT
    # =========================================================================
    
    def _build_default_transitions(self):
        """Build the default transition table."""
        transitions = [
            # From IDLE
            StateTransition(DialogState.IDLE, DialogState.UNDERSTANDING, DialogEvent.USER_INPUT),
            
            # From UNDERSTANDING
            StateTransition(
                DialogState.UNDERSTANDING, DialogState.EXECUTING,
                DialogEvent.INTENT_RECOGNIZED,
                guard=lambda ctx: ctx.intent_confidence >= 0.7 and not ctx.missing_slots
            ),
            StateTransition(
                DialogState.UNDERSTANDING, DialogState.SLOT_FILLING,
                DialogEvent.SLOTS_MISSING,
                guard=lambda ctx: bool(ctx.missing_slots)
            ),
            StateTransition(
                DialogState.UNDERSTANDING, DialogState.DISAMBIGUATION,
                DialogEvent.INTENT_AMBIGUOUS,
                guard=lambda ctx: len(ctx.alternative_intents) > 1
            ),
            StateTransition(
                DialogState.UNDERSTANDING, DialogState.CONFIRMING,
                DialogEvent.INTENT_RECOGNIZED,
                guard=lambda ctx: ctx.intent_confidence >= 0.4 and ctx.intent_confidence < 0.7
            ),
            StateTransition(
                DialogState.UNDERSTANDING, DialogState.ERROR_RECOVERY,
                DialogEvent.INTENT_UNCLEAR
            ),
            
            # From SLOT_FILLING
            StateTransition(
                DialogState.SLOT_FILLING, DialogState.SLOT_FILLING,
                DialogEvent.USER_INPUT,
                guard=lambda ctx: bool(ctx.missing_slots)
            ),
            StateTransition(
                DialogState.SLOT_FILLING, DialogState.CONFIRMING,
                DialogEvent.SLOTS_COMPLETE
            ),
            StateTransition(DialogState.SLOT_FILLING, DialogState.IDLE, DialogEvent.CANCEL),
            StateTransition(DialogState.SLOT_FILLING, DialogState.IDLE, DialogEvent.TIMEOUT),
            
            # From DISAMBIGUATION
            StateTransition(DialogState.DISAMBIGUATION, DialogState.EXECUTING, DialogEvent.USER_INPUT),
            StateTransition(DialogState.DISAMBIGUATION, DialogState.IDLE, DialogEvent.CANCEL),
            StateTransition(DialogState.DISAMBIGUATION, DialogState.IDLE, DialogEvent.TIMEOUT),
            
            # From CONFIRMING
            StateTransition(DialogState.CONFIRMING, DialogState.EXECUTING, DialogEvent.USER_CONFIRM),
            StateTransition(DialogState.CONFIRMING, DialogState.IDLE, DialogEvent.USER_DENY),
            StateTransition(
                DialogState.CONFIRMING, DialogState.UNDERSTANDING,
                DialogEvent.USER_CORRECTION
            ),
            StateTransition(DialogState.CONFIRMING, DialogState.IDLE, DialogEvent.TIMEOUT),
            
            # From EXECUTING
            StateTransition(DialogState.EXECUTING, DialogState.PRESENTING, DialogEvent.TOOL_SUCCESS),
            StateTransition(DialogState.EXECUTING, DialogState.ERROR_RECOVERY, DialogEvent.TOOL_FAILED),
            
            # From PRESENTING
            StateTransition(DialogState.PRESENTING, DialogState.FOLLOW_UP, DialogEvent.RESULTS_READY),
            StateTransition(DialogState.PRESENTING, DialogState.IDLE, DialogEvent.RESET),
            
            # From FOLLOW_UP
            StateTransition(DialogState.FOLLOW_UP, DialogState.UNDERSTANDING, DialogEvent.USER_INPUT),
            StateTransition(DialogState.FOLLOW_UP, DialogState.IDLE, DialogEvent.TIMEOUT),
            
            # From ERROR_RECOVERY
            StateTransition(DialogState.ERROR_RECOVERY, DialogState.IDLE, DialogEvent.RESET),
            StateTransition(DialogState.ERROR_RECOVERY, DialogState.UNDERSTANDING, DialogEvent.USER_INPUT),
            StateTransition(DialogState.ERROR_RECOVERY, DialogState.HUMAN_HANDOFF, DialogEvent.ESCALATE),
            
            # Global transitions (from any state)
            StateTransition(DialogState.IDLE, DialogState.IDLE, DialogEvent.RESET, priority=-1),
        ]
        
        # Register transitions
        for t in transitions:
            self.add_transition(t)
    
    def _build_state_configs(self):
        """Build state configurations."""
        self._state_configs = {
            DialogState.IDLE: StateConfig(
                state=DialogState.IDLE,
                timeout_seconds=0,  # No timeout when idle
            ),
            DialogState.UNDERSTANDING: StateConfig(
                state=DialogState.UNDERSTANDING,
                timeout_seconds=30,
                on_timeout=lambda ctx: DialogEvent.INTENT_UNCLEAR,
            ),
            DialogState.SLOT_FILLING: StateConfig(
                state=DialogState.SLOT_FILLING,
                timeout_seconds=120,  # 2 minutes to fill slots
                on_timeout=lambda ctx: DialogEvent.TIMEOUT,
            ),
            DialogState.DISAMBIGUATION: StateConfig(
                state=DialogState.DISAMBIGUATION,
                timeout_seconds=60,
                on_timeout=lambda ctx: DialogEvent.TIMEOUT,
            ),
            DialogState.CONFIRMING: StateConfig(
                state=DialogState.CONFIRMING,
                timeout_seconds=60,
                on_timeout=lambda ctx: DialogEvent.TIMEOUT,
            ),
            DialogState.EXECUTING: StateConfig(
                state=DialogState.EXECUTING,
                timeout_seconds=300,  # 5 minutes for tool execution
                on_timeout=lambda ctx: DialogEvent.TOOL_FAILED,
            ),
            DialogState.PRESENTING: StateConfig(
                state=DialogState.PRESENTING,
                timeout_seconds=0,  # No timeout
            ),
            DialogState.FOLLOW_UP: StateConfig(
                state=DialogState.FOLLOW_UP,
                timeout_seconds=300,  # 5 minutes for follow-up
                on_timeout=lambda ctx: DialogEvent.TIMEOUT,
            ),
            DialogState.ERROR_RECOVERY: StateConfig(
                state=DialogState.ERROR_RECOVERY,
                timeout_seconds=120,
                on_timeout=lambda ctx: DialogEvent.RESET,
            ),
        }
    
    def add_transition(self, transition: StateTransition):
        """
        Add a transition to the state machine.
        
        Args:
            transition: The transition to add
        """
        key = (transition.from_state, transition.event)
        if key not in self._transitions:
            self._transitions[key] = []
        
        self._transitions[key].append(transition)
        # Sort by priority (highest first)
        self._transitions[key].sort(key=lambda t: -t.priority)
    
    def get_valid_transitions(
        self, 
        event: DialogEvent, 
        context: DialogContext
    ) -> List[StateTransition]:
        """
        Get all valid transitions for an event.
        
        Args:
            event: The event
            context: Current context
            
        Returns:
            List of valid transitions (guard conditions satisfied)
        """
        key = (self._current_state, event)
        transitions = self._transitions.get(key, [])
        
        return [t for t in transitions if t.can_fire(context)]
    
    # =========================================================================
    # STATE TRANSITIONS
    # =========================================================================
    
    def process_event(
        self, 
        event: DialogEvent, 
        context: DialogContext
    ) -> Optional[DialogState]:
        """
        Process an event and potentially transition.
        
        Args:
            event: The event to process
            context: Current dialog context
            
        Returns:
            New state if transitioned, None otherwise
        """
        return self._process_event_internal(event, context, _handling_timeout=False)
    
    def _process_event_internal(
        self,
        event: DialogEvent,
        context: DialogContext,
        _handling_timeout: bool = False,
    ) -> Optional[DialogState]:
        """
        Internal event processing with timeout recursion guard.
        """
        # Check for timeout first (unless we're already handling a timeout)
        if not _handling_timeout and self._check_timeout(context, _handling_timeout=False):
            return self._current_state
        
        # Find valid transitions
        valid_transitions = self.get_valid_transitions(event, context)
        
        if not valid_transitions:
            logger.debug(
                f"No valid transition from {self._current_state.name} "
                f"on event {event.name}"
            )
            return None
        
        # Take the highest priority transition
        transition = valid_transitions[0]
        
        # Execute transition
        return self._execute_transition(transition, context)
    
    def _execute_transition(
        self, 
        transition: StateTransition, 
        context: DialogContext
    ) -> DialogState:
        """Execute a state transition."""
        old_state = self._current_state
        new_state = transition.to_state
        
        # Record history
        self._record_history(old_state, transition.event, context)
        
        # Execute exit action
        old_config = self._state_configs.get(old_state)
        if old_config and old_config.on_exit:
            old_config.on_exit(context)
        
        # Execute transition action
        transition.execute(context)
        
        # Change state
        self._current_state = new_state
        self._state_entered_at = datetime.now()
        
        # Execute entry action
        new_config = self._state_configs.get(new_state)
        if new_config and new_config.on_enter:
            message = new_config.on_enter(context)
            if message:
                context.response_message = message
        
        logger.info(
            f"State transition: {old_state.name} -> {new_state.name} "
            f"(event: {transition.event.name})"
        )
        
        return new_state
    
    def transition_to(
        self, 
        state: DialogState, 
        context: DialogContext = None
    ) -> bool:
        """
        Force transition to a specific state.
        
        Use with caution - bypasses normal event flow.
        
        Args:
            state: Target state
            context: Optional context
            
        Returns:
            True if transition was made
        """
        if context is None:
            context = DialogContext()
        
        old_state = self._current_state
        
        # Record history
        self._record_history(old_state, None, context)
        
        # Execute exit action
        old_config = self._state_configs.get(old_state)
        if old_config and old_config.on_exit:
            old_config.on_exit(context)
        
        # Change state
        self._current_state = state
        self._state_entered_at = datetime.now()
        
        # Execute entry action
        new_config = self._state_configs.get(state)
        if new_config and new_config.on_enter:
            message = new_config.on_enter(context)
            if message:
                context.response_message = message
        
        logger.info(f"Forced transition: {old_state.name} -> {state.name}")
        return True
    
    def can_transition_to(self, state: DialogState) -> bool:
        """
        Check if direct transition to state is possible.
        
        Args:
            state: Target state
            
        Returns:
            True if any event can trigger this transition
        """
        for (from_state, _), transitions in self._transitions.items():
            if from_state == self._current_state:
                for t in transitions:
                    if t.to_state == state:
                        return True
        return False
    
    # =========================================================================
    # TIMEOUT HANDLING
    # =========================================================================
    
    def _check_timeout(self, context: DialogContext, _handling_timeout: bool = False) -> bool:
        """
        Check if current state has timed out.
        
        Returns:
            True if timed out and handled
        """
        # Prevent recursion when processing timeout events
        if _handling_timeout:
            return False
            
        config = self._state_configs.get(self._current_state)
        if not config or config.timeout_seconds == 0:
            return False
        
        if self.time_in_state.total_seconds() > config.timeout_seconds:
            logger.info(f"State {self._current_state.name} timed out")
            
            if config.on_timeout:
                timeout_event = config.on_timeout(context)
                # Use internal method to avoid recursion
                self._process_event_internal(timeout_event, context, _handling_timeout=True)
                return True
            else:
                # Default: go to IDLE
                self.transition_to(DialogState.IDLE, context)
                return True
        
        return False
    
    def check_and_handle_timeout(self, context: DialogContext) -> bool:
        """
        Explicitly check for timeout.
        
        Call this periodically if you want proactive timeout handling.
        """
        return self._check_timeout(context)
    
    # =========================================================================
    # HISTORY & ROLLBACK
    # =========================================================================
    
    def _record_history(
        self, 
        state: DialogState, 
        event: Optional[DialogEvent],
        context: DialogContext
    ):
        """Record state in history."""
        entry = StateHistoryEntry(
            state=state,
            timestamp=datetime.now(),
            event=event,
            context_snapshot={
                "intent": context.intent,
                "slots": dict(context.slots),
                "turn": context.turn_count,
            }
        )
        
        self._history.append(entry)
        
        # Trim history
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
    
    def rollback(self, steps: int = 1) -> Optional[DialogState]:
        """
        Rollback to a previous state.
        
        Args:
            steps: Number of states to go back
            
        Returns:
            State rolled back to, or None if not possible
        """
        if len(self._history) < steps:
            logger.warning(f"Cannot rollback {steps} steps, only {len(self._history)} in history")
            return None
        
        # Get target entry
        target_entry = self._history[-(steps + 1)] if len(self._history) > steps else self._history[0]
        
        # Remove rolled-back entries
        self._history = self._history[:-(steps)]
        
        # Restore state
        old_state = self._current_state
        self._current_state = target_entry.state
        self._state_entered_at = datetime.now()
        
        logger.info(f"Rolled back from {old_state.name} to {self._current_state.name}")
        
        return self._current_state
    
    def get_history(self) -> List[StateHistoryEntry]:
        """Get state history."""
        return list(self._history)
    
    # =========================================================================
    # RESET
    # =========================================================================
    
    def reset(self, context: Optional[DialogContext] = None):
        """
        Reset state machine to IDLE.
        
        Args:
            context: Optional context to update
        """
        if context:
            context.clear_slots()
            context.clear_pending()
        
        self._current_state = DialogState.IDLE
        self._state_entered_at = datetime.now()
        self._history = []
        
        logger.info("State machine reset to IDLE")
    
    # =========================================================================
    # INTROSPECTION
    # =========================================================================
    
    def get_available_events(self) -> Set[DialogEvent]:
        """Get events that can trigger a transition from current state."""
        events = set()
        for (from_state, event), _ in self._transitions.items():
            if from_state == self._current_state:
                events.add(event)
        return events
    
    def get_reachable_states(self) -> Set[DialogState]:
        """Get states reachable from current state."""
        states = set()
        for (from_state, _), transitions in self._transitions.items():
            if from_state == self._current_state:
                for t in transitions:
                    states.add(t.to_state)
        return states
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get information about current state."""
        config = self._state_configs.get(self._current_state)
        return {
            "state": self._current_state.name,
            "entered_at": self._state_entered_at.isoformat(),
            "time_in_state_seconds": self.time_in_state.total_seconds(),
            "timeout_seconds": config.timeout_seconds if config else 0,
            "available_events": [e.name for e in self.get_available_events()],
            "reachable_states": [s.name for s in self.get_reachable_states()],
            "history_depth": len(self._history),
        }


# =============================================================================
# STATE HANDLERS (Abstract Base)
# =============================================================================

class StateHandler(ABC):
    """
    Base class for state-specific handlers.
    
    Each state can have a handler that:
    - Generates entry messages
    - Processes input in that state
    - Determines next event
    """
    
    @abstractmethod
    def on_enter(self, context: DialogContext) -> Optional[str]:
        """
        Called when entering this state.
        
        Returns:
            Optional message to send to user
        """
        pass
    
    @abstractmethod
    def process_input(
        self, 
        user_input: str, 
        context: DialogContext
    ) -> Tuple[DialogEvent, Optional[str]]:
        """
        Process user input in this state.
        
        Args:
            user_input: User's message
            context: Dialog context
            
        Returns:
            Tuple of (event_to_trigger, optional_response_message)
        """
        pass
    
    def on_exit(self, context: DialogContext):
        """Called when exiting this state."""
        pass


class IdleHandler(StateHandler):
    """Handler for IDLE state."""
    
    def on_enter(self, context: DialogContext) -> Optional[str]:
        return None  # No message when going idle
    
    def process_input(
        self, 
        user_input: str, 
        context: DialogContext
    ) -> Tuple[DialogEvent, Optional[str]]:
        context.query = user_input
        context.turn_count += 1
        return DialogEvent.USER_INPUT, None


class SlotFillingHandler(StateHandler):
    """Handler for SLOT_FILLING state."""
    
    def on_enter(self, context: DialogContext) -> Optional[str]:
        if context.missing_slots:
            slot = context.missing_slots[0]
            context.current_slot = slot
            return f"Please provide {slot}:"
        return None
    
    def process_input(
        self, 
        user_input: str, 
        context: DialogContext
    ) -> Tuple[DialogEvent, Optional[str]]:
        # Try to fill current slot
        if context.current_slot:
            context.slots[context.current_slot] = user_input
            context.missing_slots.remove(context.current_slot)
        
        if not context.missing_slots:
            return DialogEvent.SLOTS_COMPLETE, "Got all the information I need."
        else:
            context.current_slot = context.missing_slots[0]
            return DialogEvent.USER_INPUT, f"Please provide {context.current_slot}:"
    
    def on_exit(self, context: DialogContext):
        context.current_slot = None


class ConfirmingHandler(StateHandler):
    """Handler for CONFIRMING state."""
    
    def on_enter(self, context: DialogContext) -> Optional[str]:
        action = context.pending_action or context.intent
        return f"Just to confirm - you want me to {action}? (yes/no)"
    
    def process_input(
        self, 
        user_input: str, 
        context: DialogContext
    ) -> Tuple[DialogEvent, Optional[str]]:
        response = user_input.lower().strip()
        
        if response in ("yes", "y", "yeah", "yep", "sure", "ok", "proceed"):
            return DialogEvent.USER_CONFIRM, "Proceeding..."
        elif response in ("no", "n", "nope", "cancel", "stop"):
            context.clear_pending()
            return DialogEvent.USER_DENY, "Cancelled."
        else:
            # Treat as correction
            context.query = user_input
            return DialogEvent.USER_CORRECTION, None


class ErrorRecoveryHandler(StateHandler):
    """Handler for ERROR_RECOVERY state."""
    
    def on_enter(self, context: DialogContext) -> Optional[str]:
        if context.error:
            return (
                f"I ran into an issue: {str(context.error)[:100]}\n\n"
                "Would you like me to try again, or can I help with something else?"
            )
        return "I'm not sure how to help with that. Could you rephrase?"
    
    def process_input(
        self, 
        user_input: str, 
        context: DialogContext
    ) -> Tuple[DialogEvent, Optional[str]]:
        response = user_input.lower().strip()
        
        if response in ("try again", "retry", "yes"):
            return DialogEvent.USER_INPUT, None
        else:
            context.reset_for_new_turn()
            context.query = user_input
            return DialogEvent.USER_INPUT, None


# =============================================================================
# SINGLETON & FACTORY
# =============================================================================

_dialog_fsm: Optional[DialogStateMachine] = None


def get_dialog_state_machine() -> DialogStateMachine:
    """Get the singleton dialog state machine."""
    global _dialog_fsm
    if _dialog_fsm is None:
        _dialog_fsm = DialogStateMachine()
    return _dialog_fsm


def reset_dialog_state_machine() -> None:
    """Reset the singleton dialog state machine."""
    global _dialog_fsm
    _dialog_fsm = None


def create_dialog_state_machine(
    initial_state: DialogState = DialogState.IDLE
) -> DialogStateMachine:
    """Create a new dialog state machine instance."""
    return DialogStateMachine(initial_state=initial_state)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core classes
    "DialogStateMachine",
    "DialogState",
    "DialogEvent",
    "DialogContext",
    "StateTransition",
    "StateConfig",
    "StateHistoryEntry",
    
    # Handlers
    "StateHandler",
    "IdleHandler",
    "SlotFillingHandler",
    "ConfirmingHandler",
    "ErrorRecoveryHandler",
    
    # Factory
    "get_dialog_state_machine",
    "create_dialog_state_machine",
    "reset_dialog_state_machine",
]
