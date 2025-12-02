# Professional Chat Agent - Complete Implementation Plan

## Executive Summary

This document provides a comprehensive audit of our chat agent capabilities and a detailed implementation plan to reach production-grade quality comparable to Rasa, Dialogflow, and ChatGPT-based assistants.

**Current Status**: ~70% complete  
**Target**: Production-ready professional chat agent  
**Estimated Work**: 4-6 development phases

---

## Part 1: Current Implementation Audit

### ✅ What We Have (Implemented)

#### 1. Intent Classification System
| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| `IntentParser` | ✅ Complete | `intent/parser.py` | Pattern-based with regex |
| `SemanticIntentClassifier` | ✅ Complete | `intent/semantic.py` | FAISS + sentence-transformers |
| `UnifiedIntentParser` | ✅ Complete | `intent/unified_parser.py` | Hierarchical with LLM arbiter |
| `IntentArbiter` | ✅ Complete | `intent/arbiter.py` | LLM for ambiguous cases |
| `HybridQueryParser` | ✅ Complete | `intent/semantic.py` | Pattern + semantic + NER |
| `BioinformaticsNER` | ✅ Complete | `intent/semantic.py` | Domain-specific NER |

#### 2. Context & Memory System
| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| `ConversationContext` | ✅ Complete | `intent/context.py` | Turn tracking, entity tracking |
| `SessionMemory` | ✅ Complete | `intent/session_memory.py` | Session-wide persistence |
| `AgentMemory` | ✅ Complete | `agents/memory/memory.py` | Vector-based RAG memory |
| `CoreferenceResolver` | ✅ Complete | `intent/context.py` | "it", "that data" resolution |
| Entity Tracking | ✅ Complete | `intent/context.py` | Salience decay |

#### 3. Dialogue Management (Partial)
| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| `DialogueManager` | ⚠️ Partial | `intent/dialogue.py` | Has task tracking, needs FSM |
| `TaskState` | ✅ Complete | `intent/dialogue.py` | Multi-step task tracking |
| `ConversationPhase` | ✅ Complete | `intent/dialogue.py` | IDLE, GATHERING, CONFIRMING |
| `SlotFiller` | ⚠️ Basic | `intent/dialogue.py` | Simple prompts only |

#### 4. Slot/Form System (Partial)
| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| `SlotPrompter` | ✅ Complete | `intent/slot_prompting.py` | Slot completeness checking |
| `SlotDefinition` | ✅ Complete | `intent/slot_prompting.py` | Required/optional/defaults |
| `IntentSlotSchema` | ✅ Complete | `intent/slot_prompting.py` | Per-intent slot schemas |
| `DialogueState` | ⚠️ Basic | `intent/slot_prompting.py` | Multi-turn tracking |
| Form Validation | ❌ Missing | - | Need systematic validation |

#### 5. Entity System
| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| `Entity` | ✅ Complete | `intent/parser.py` | Basic entity structure |
| `EntityType` | ✅ Complete | `intent/parser.py` | 15+ entity types |
| `EntityRoleResolver` | ✅ Complete | `intent/entity_roles.py` | Source/destination roles |
| Entity Validation | ❌ Missing | - | Type-specific validation |
| Composite Entities | ❌ Missing | - | Nested structures |

#### 6. Error Recovery
| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| `ConversationRecovery` | ✅ Complete | `intent/conversation_recovery.py` | Clarification, fallback |
| Error Templates | ✅ Complete | `intent/conversation_recovery.py` | Category-specific messages |
| Correction Handling | ✅ Complete | `intent/conversation_recovery.py` | "No I meant X" |
| LLM Retry Strategy | ⚠️ Partial | `intent/conversation_recovery.py` | Infrastructure exists |

#### 7. Training & Learning
| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| `TrainingDataLoader` | ✅ Complete | `intent/training_data.py` | YAML loading |
| `ActiveLearner` | ✅ Complete | `intent/active_learning.py` | Correction tracking |
| `BalanceMetrics` | ✅ Complete | `intent/balance_metrics.py` | Training data analysis |
| YAML Intent Definitions | ✅ Complete | `config/nlu/intents/` | 2 files, expandable |
| YAML Entity Definitions | ✅ Complete | `config/nlu/entities/` | 3 files |

#### 8. Analytics (Partial)
| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| `QueryAnalytics` | ✅ Complete | `observability/query_analytics.py` | Query tracking |
| `LearningMetrics` | ✅ Complete | `intent/active_learning.py` | Correction rates |
| Conversation Analytics | ❌ Missing | - | Success rates, drop-off |
| Dashboard | ❌ Missing | - | No UI for insights |

---

### ❌ What's Missing (Gaps)

#### Gap 1: Formal Dialog State Machine
**Current**: Ad-hoc phase tracking  
**Need**: Formal FSM with defined states, transitions, and policies

```
Missing States:
- SLOT_FILLING (systematic)
- DISAMBIGUATION
- CONFIRMATION_PENDING
- EXECUTION_IN_PROGRESS
- AWAITING_FOLLOW_UP
- ERROR_RECOVERY
- HUMAN_HANDOFF

Missing Transitions:
- State transition rules
- Guard conditions
- Rollback capability
```

#### Gap 2: Form System with Validation
**Current**: Basic slot prompting  
**Need**: Complete form lifecycle with validation

```
Missing:
- Form definition schema
- Multi-slot validation
- Cross-slot dependencies
- Form cancellation
- Form timeout
- Partial submission
```

#### Gap 3: Response Generation
**Current**: Hardcoded templates  
**Need**: Dynamic response generator with variations

```
Missing:
- Response variation selection
- Persona consistency
- Conditional responses
- Rich responses (cards, buttons)
- Response post-processing
```

#### Gap 4: Policy Layer
**Current**: Rule-based only  
**Need**: Hybrid rule + ML policy

```
Missing:
- Policy configuration
- A/B testing support
- Policy metrics
- Policy fallback chain
```

#### Gap 5: Conversation Analytics Dashboard
**Current**: Raw metrics  
**Need**: Actionable insights UI

```
Missing:
- Success rate visualization
- Intent confusion matrix
- Drop-off analysis
- User satisfaction tracking
- Export capabilities
```

#### Gap 6: Human Handoff System
**Current**: None  
**Need**: Escalation to human when needed

```
Missing:
- Handoff detection
- Context packaging
- Handoff routing
- Return flow
```

---

## Part 2: Implementation Plan

### Phase 1: Dialog State Machine (Priority: CRITICAL)

**Goal**: Replace ad-hoc flow with formal state machine

#### 1.1 Create `dialog_state_machine.py`

```python
# New file: src/workflow_composer/agents/intent/dialog_state_machine.py

class DialogState(Enum):
    """All possible conversation states."""
    IDLE = auto()                    # Ready for new input
    UNDERSTANDING = auto()           # Parsing intent
    SLOT_FILLING = auto()            # Collecting required info
    DISAMBIGUATION = auto()          # Clarifying ambiguous intent
    CONFIRMING = auto()              # Awaiting user confirmation
    EXECUTING = auto()               # Running tool/action
    PRESENTING_RESULTS = auto()      # Showing results
    FOLLOW_UP = auto()               # Awaiting follow-up
    ERROR_RECOVERY = auto()          # Handling error
    HUMAN_HANDOFF = auto()           # Escalated to human


class DialogTransition:
    """A state transition with guards and actions."""
    from_state: DialogState
    to_state: DialogState
    trigger: str                     # Event that triggers transition
    guard: Optional[Callable]        # Condition that must be true
    action: Optional[Callable]       # Action to execute on transition


class DialogStateMachine:
    """
    Formal state machine for dialog management.
    
    Features:
    - Defined states and transitions
    - Guard conditions
    - Entry/exit actions
    - State history for rollback
    - Timeout handling
    """
    
    def __init__(self):
        self.current_state = DialogState.IDLE
        self.state_history: List[DialogState] = []
        self.transitions: Dict[tuple, DialogTransition] = {}
        self._build_transitions()
    
    def _build_transitions(self):
        """Define all valid state transitions."""
        # IDLE -> UNDERSTANDING (on user input)
        # UNDERSTANDING -> SLOT_FILLING (if slots missing)
        # UNDERSTANDING -> CONFIRMING (if high-stakes action)
        # UNDERSTANDING -> EXECUTING (if ready)
        # ... etc
    
    def transition(self, event: str, context: Dict) -> bool:
        """Attempt a state transition."""
        pass
    
    def can_transition(self, to_state: DialogState) -> bool:
        """Check if transition is valid."""
        pass
    
    def rollback(self, steps: int = 1):
        """Rollback to previous state."""
        pass
```

#### 1.2 Implement State Handlers

Each state needs entry/exit handlers:

```python
class StateHandler(ABC):
    """Base class for state handlers."""
    
    @abstractmethod
    def on_enter(self, context: DialogContext) -> Optional[str]:
        """Called when entering state. Returns message if any."""
        pass
    
    @abstractmethod
    def process(self, user_input: str, context: DialogContext) -> StateResult:
        """Process input in this state."""
        pass
    
    @abstractmethod
    def on_exit(self, context: DialogContext):
        """Called when exiting state."""
        pass


class SlotFillingHandler(StateHandler):
    """Handler for SLOT_FILLING state."""
    
    def on_enter(self, context):
        missing = context.get_missing_slots()
        return f"I need a few more details: {self._format_prompts(missing)}"
    
    def process(self, user_input, context):
        # Try to fill slots from input
        filled = self._extract_slots(user_input, context)
        context.update_slots(filled)
        
        if context.all_slots_filled():
            return StateResult(next_state=DialogState.CONFIRMING)
        else:
            return StateResult(
                next_state=DialogState.SLOT_FILLING,
                message=self._get_next_prompt(context)
            )
```

#### 1.3 Integration Points

```python
# Update DialogueManager to use state machine
class DialogueManager:
    def __init__(self):
        self.fsm = DialogStateMachine()
        self.handlers = {
            DialogState.IDLE: IdleHandler(),
            DialogState.SLOT_FILLING: SlotFillingHandler(),
            # ...
        }
    
    def process_message(self, message: str) -> DialogueResult:
        handler = self.handlers[self.fsm.current_state]
        result = handler.process(message, self.context)
        
        if result.next_state != self.fsm.current_state:
            self.fsm.transition(result.trigger, self.context)
        
        return DialogueResult(
            message=result.message,
            state=self.fsm.current_state,
            # ...
        )
```

---

### Phase 2: Form System Enhancement (Priority: HIGH)

**Goal**: Systematic form handling with validation

#### 2.1 Create `forms.py`

```python
# New file: src/workflow_composer/agents/intent/forms.py

@dataclass
class FormField:
    """A single form field."""
    name: str
    field_type: str
    required: bool = True
    prompt: str = ""
    validation: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    options: Optional[List[str]] = None
    default: Any = None


@dataclass
class FormDefinition:
    """Complete form definition."""
    name: str
    intent: str
    fields: List[FormField]
    submit_action: str
    cancel_message: str = "Cancelled."
    timeout_seconds: int = 300
    
    def get_required_fields(self) -> List[FormField]:
        return [f for f in self.fields if f.required]
    
    def get_unfilled_required(self, values: Dict) -> List[FormField]:
        return [f for f in self.get_required_fields() 
                if f.name not in values or values[f.name] is None]


class FormManager:
    """
    Manages form lifecycle.
    
    Features:
    - Form activation/deactivation
    - Field-by-field filling
    - Validation
    - Timeout handling
    - Cancellation
    """
    
    def __init__(self):
        self.forms: Dict[str, FormDefinition] = {}
        self.active_form: Optional[str] = None
        self.form_values: Dict[str, Any] = {}
        self.current_field: Optional[str] = None
        self.started_at: Optional[datetime] = None
    
    def activate_form(self, form_name: str) -> str:
        """Activate a form and return initial prompt."""
        pass
    
    def process_input(self, user_input: str) -> FormResult:
        """Process user input for current field."""
        pass
    
    def validate_field(self, field: FormField, value: Any) -> ValidationResult:
        """Validate a single field value."""
        pass
    
    def submit(self) -> Dict[str, Any]:
        """Submit the form and return collected values."""
        pass
    
    def cancel(self) -> str:
        """Cancel the form."""
        pass
    
    def is_timeout(self) -> bool:
        """Check if form has timed out."""
        pass
```

#### 2.2 Field Validators

```python
# Built-in validators
class FieldValidators:
    """Common field validators."""
    
    @staticmethod
    def file_path(value: str) -> ValidationResult:
        """Validate file path exists."""
        path = Path(value).expanduser()
        if path.exists():
            return ValidationResult(valid=True, normalized=str(path))
        return ValidationResult(
            valid=False, 
            error=f"Path not found: {value}"
        )
    
    @staticmethod
    def organism(value: str) -> ValidationResult:
        """Validate organism name."""
        from .training_data import get_training_data_loader
        loader = get_training_data_loader()
        entities = loader.get_entities()
        
        organism_def = next(
            (e for e in entities if e.name == "organism"), 
            None
        )
        if organism_def:
            alias_map = organism_def.get_alias_map()
            if value.lower() in alias_map:
                return ValidationResult(
                    valid=True,
                    normalized=alias_map[value.lower()]
                )
        
        return ValidationResult(
            valid=False,
            error=f"Unknown organism: {value}. Try: human, mouse, zebrafish"
        )
    
    @staticmethod
    def dataset_id(value: str) -> ValidationResult:
        """Validate dataset ID format."""
        import re
        patterns = [
            r'^GSE\d+$',      # GEO
            r'^ENCSR\w+$',    # ENCODE
            r'^SRP\d+$',      # SRA project
            r'^PRJNA\d+$',    # BioProject
        ]
        for pattern in patterns:
            if re.match(pattern, value, re.IGNORECASE):
                return ValidationResult(valid=True, normalized=value.upper())
        
        return ValidationResult(
            valid=False,
            error=f"Invalid dataset ID format: {value}"
        )
```

#### 2.3 Form Definitions (YAML)

```yaml
# config/nlu/forms/search_form.yaml
form:
  name: "data_search_form"
  intent: "DATA_SEARCH"
  
  fields:
    - name: query
      type: text
      required: true
      prompt: "What are you searching for?"
      
    - name: organism
      type: organism
      required: false
      prompt: "Which organism? (or 'any' for all)"
      default: "any"
      
    - name: assay_type
      type: assay_type
      required: false
      prompt: "What type of data? (RNA-seq, ChIP-seq, etc.)"
      
    - name: source
      type: choice
      required: false
      options: ["GEO", "ENCODE", "SRA", "all"]
      default: "all"
      prompt: "Which database to search?"
  
  submit_action: "search_databases"
  cancel_message: "Search cancelled. What else can I help with?"
  timeout_seconds: 120
```

---

### Phase 3: Response Generator (Priority: HIGH)

**Goal**: Professional, varied, context-aware responses

#### 3.1 Create `response_generator.py`

```python
# New file: src/workflow_composer/agents/intent/response_generator.py

class ResponseType(Enum):
    """Types of responses."""
    ACKNOWLEDGMENT = auto()      # "Got it", "Sure thing"
    CONFIRMATION = auto()        # "You want me to X, right?"
    CLARIFICATION = auto()       # "Did you mean X or Y?"
    RESULT = auto()              # Tool execution results
    ERROR = auto()               # Error messages
    SUGGESTION = auto()          # Follow-up suggestions
    GREETING = auto()            # Hello/goodbye
    HELP = auto()                # Help text


@dataclass
class ResponseTemplate:
    """A response template with variations."""
    response_type: ResponseType
    intent: Optional[str] = None          # Specific intent or None for general
    condition: Optional[str] = None       # Condition for selection
    templates: List[str] = field(default_factory=list)
    priority: int = 0                     # Higher = preferred
    
    def render(self, context: Dict[str, Any]) -> str:
        """Render template with context."""
        import random
        template = random.choice(self.templates)
        return template.format(**context)


class ResponseGenerator:
    """
    Generate natural, varied responses.
    
    Features:
    - Multiple variations per response type
    - Context-aware selection
    - Persona consistency
    - Rich response support (future)
    """
    
    def __init__(self, persona: str = "helpful_assistant"):
        self.persona = persona
        self.templates: Dict[str, List[ResponseTemplate]] = {}
        self._used_templates: Dict[str, set] = {}  # Avoid repetition
        self._load_templates()
    
    def generate(
        self,
        response_type: ResponseType,
        context: Dict[str, Any],
        intent: str = None,
    ) -> str:
        """Generate a response."""
        candidates = self._get_candidates(response_type, intent, context)
        
        # Avoid recently used templates
        fresh_candidates = self._filter_recent(candidates, response_type.name)
        
        if fresh_candidates:
            template = self._select_best(fresh_candidates, context)
        else:
            # Reset if all used
            self._used_templates[response_type.name] = set()
            template = self._select_best(candidates, context)
        
        # Mark as used
        self._used_templates.setdefault(response_type.name, set()).add(
            id(template)
        )
        
        return template.render(context)
    
    def _get_candidates(
        self, 
        response_type: ResponseType,
        intent: str,
        context: Dict
    ) -> List[ResponseTemplate]:
        """Get candidate templates."""
        key = f"{response_type.name}:{intent or 'general'}"
        candidates = self.templates.get(key, [])
        
        # Also check general templates
        general_key = f"{response_type.name}:general"
        candidates.extend(self.templates.get(general_key, []))
        
        # Filter by conditions
        return [c for c in candidates if self._check_condition(c, context)]
    
    def _check_condition(self, template: ResponseTemplate, context: Dict) -> bool:
        """Check if template condition is met."""
        if not template.condition:
            return True
        
        # Simple condition evaluation
        # Example: "success == True", "result_count > 0"
        try:
            return eval(template.condition, {"__builtins__": {}}, context)
        except:
            return True
```

#### 3.2 Response Templates (YAML)

```yaml
# config/nlu/responses/templates.yaml
version: "1.0"

responses:
  # ==========================================================================
  # ACKNOWLEDGMENTS
  # ==========================================================================
  acknowledgments:
    general:
      - "Got it!"
      - "Sure thing."
      - "On it!"
      - "Understood."
      - "Let me do that for you."
      - "Working on it..."
    
    data:
      - "Let me look at your data."
      - "Scanning your files..."
      - "Checking your data now."
    
    search:
      - "Searching the databases..."
      - "Let me find that for you."
      - "Looking for {query}..."
  
  # ==========================================================================
  # CONFIRMATIONS
  # ==========================================================================
  confirmations:
    general:
      - "Just to confirm - you want me to {action}?"
      - "Before I proceed, is this right: {action}?"
      - "I'll {action}. Sound good?"
    
    high_stakes:
      - "⚠️ This will {action}. Are you sure you want to proceed?"
      - "This action ({action}) cannot be undone. Please confirm with 'yes'."
  
  # ==========================================================================
  # RESULTS
  # ==========================================================================
  results:
    success:
      data_scan:
        - "Found {count} files in {path}.\n\n{summary}"
        - "Scan complete! {count} samples discovered.\n\n{summary}"
      
      data_search:
        - "Found {count} datasets matching your search.\n\n{summary}"
        - "Here are {count} relevant datasets:\n\n{summary}"
        - "Your search returned {count} results:\n\n{summary}"
      
      workflow_create:
        - "✅ Workflow generated successfully!\n\n{summary}"
        - "Your {workflow_type} workflow is ready:\n\n{summary}"
    
    empty:
      data_scan:
        - "No files found in {path}."
        - "The directory {path} appears to be empty or inaccessible."
      
      data_search:
        - "No datasets found matching '{query}'."
        - "I couldn't find any {query} data. Try:\n• Broadening your search\n• Checking a different database"
  
  # ==========================================================================
  # ERRORS
  # ==========================================================================
  errors:
    general:
      - "Something went wrong: {error}"
      - "I ran into an issue: {error}\n\nWould you like me to try again?"
    
    not_found:
      - "I couldn't find {resource}."
      - "{resource} doesn't seem to exist. Want me to help locate it?"
    
    permission:
      - "I don't have permission to {action}."
      - "Access denied for {action}. You may need to run with different permissions."
  
  # ==========================================================================
  # SUGGESTIONS
  # ==========================================================================
  suggestions:
    after_scan:
      - "What would you like to do with these files?"
      - "You can now:\n• Search for more data\n• Create a workflow\n• Download additional files"
    
    after_search:
      - "Would you like me to download any of these datasets?"
      - "Say 'download {first_result}' to get the first result."
    
    after_workflow:
      - "Ready to run this workflow?"
      - "You can now:\n• Submit the job\n• Modify parameters\n• Preview the workflow"
```

#### 3.3 Personality Profiles

```yaml
# config/nlu/responses/personas.yaml
personas:
  helpful_assistant:
    name: "BioPipelines Assistant"
    traits:
      - concise
      - friendly
      - professional
    
    style:
      emoji_frequency: low        # Occasional emoji for visual clarity
      formality: medium          # Professional but approachable
      verbosity: concise         # Get to the point
      error_handling: supportive # Acknowledge, apologize, suggest
    
    greetings:
      - "Hello! I'm your BioPipelines assistant. How can I help?"
      - "Hi there! Ready to help with your bioinformatics work."
    
    farewells:
      - "Goodbye! Feel free to come back anytime."
      - "Happy analyzing! Let me know if you need anything else."
  
  expert_mode:
    name: "BioPipelines Expert"
    traits:
      - technical
      - efficient
      - minimal
    
    style:
      emoji_frequency: none
      formality: high
      verbosity: minimal
      error_handling: direct
```

---

### Phase 4: Policy Layer (Priority: MEDIUM)

**Goal**: Intelligent action selection with hybrid rules + ML

#### 4.1 Create `policy.py`

```python
# New file: src/workflow_composer/agents/intent/policy.py

class PolicyType(Enum):
    """Types of policies."""
    RULE_BASED = auto()       # Deterministic rules
    ML_BASED = auto()         # Machine learning
    HYBRID = auto()           # Combination


@dataclass
class PolicyDecision:
    """A policy decision."""
    action: str
    confidence: float
    reason: str
    alternatives: List[str] = field(default_factory=list)


class Policy(ABC):
    """Base class for policies."""
    
    @abstractmethod
    def decide(self, context: PolicyContext) -> PolicyDecision:
        """Make a policy decision."""
        pass


class RuleBasedPolicy(Policy):
    """
    Rule-based policy with configurable rules.
    
    Rules are evaluated in priority order.
    """
    
    def __init__(self):
        self.rules: List[PolicyRule] = []
        self._load_rules()
    
    def decide(self, context: PolicyContext) -> PolicyDecision:
        for rule in sorted(self.rules, key=lambda r: -r.priority):
            if rule.matches(context):
                return PolicyDecision(
                    action=rule.action,
                    confidence=rule.confidence,
                    reason=f"Rule: {rule.name}"
                )
        
        return self._default_decision(context)


@dataclass
class PolicyRule:
    """A single policy rule."""
    name: str
    conditions: Dict[str, Any]
    action: str
    priority: int = 0
    confidence: float = 1.0
    
    def matches(self, context: PolicyContext) -> bool:
        """Check if rule matches context."""
        for key, expected in self.conditions.items():
            actual = getattr(context, key, None)
            if actual != expected:
                return False
        return True


class HybridPolicy(Policy):
    """
    Hybrid rule + ML policy.
    
    Uses rules for common/critical cases, ML for edge cases.
    """
    
    def __init__(self):
        self.rule_policy = RuleBasedPolicy()
        self.ml_policy: Optional[MLPolicy] = None
        self.rule_threshold = 0.8  # Confidence threshold to use rule
    
    def decide(self, context: PolicyContext) -> PolicyDecision:
        # Try rules first
        rule_decision = self.rule_policy.decide(context)
        
        if rule_decision.confidence >= self.rule_threshold:
            return rule_decision
        
        # Fall back to ML
        if self.ml_policy:
            ml_decision = self.ml_policy.decide(context)
            
            # Combine or choose
            if ml_decision.confidence > rule_decision.confidence:
                return ml_decision
        
        return rule_decision
```

#### 4.2 Policy Configuration (YAML)

```yaml
# config/nlu/policies/default.yaml
policy:
  type: hybrid
  rule_threshold: 0.8
  
  rules:
    # High-confidence patterns
    - name: "scan_local_data"
      conditions:
        intent: "DATA_SCAN"
        has_path: true
      action: "scan_data"
      priority: 100
      confidence: 0.95
    
    - name: "confirm_before_download"
      conditions:
        intent: "DATA_DOWNLOAD"
        size_mb_gt: 1000
      action: "confirm_then_download"
      priority: 90
      confidence: 0.9
    
    - name: "clarify_ambiguous"
      conditions:
        intent_confidence_lt: 0.5
      action: "clarify_intent"
      priority: 80
      confidence: 0.85
    
    # Fallback
    - name: "default_help"
      conditions: {}
      action: "show_help"
      priority: 0
      confidence: 0.5
```

---

### Phase 5: Conversation Analytics (Priority: MEDIUM)

**Goal**: Track, analyze, and visualize conversation quality

#### 5.1 Enhance `ConversationAnalytics`

```python
# New file: src/workflow_composer/agents/intent/conversation_analytics.py

@dataclass
class ConversationMetrics:
    """Metrics for a single conversation."""
    session_id: str
    turn_count: int
    success: bool
    duration_seconds: float
    intents_detected: List[str]
    tools_used: List[str]
    clarifications_needed: int
    errors_encountered: int
    user_satisfaction: Optional[float] = None  # 1-5 rating


class ConversationAnalytics:
    """
    Track and analyze conversation quality.
    
    Metrics:
    - Task completion rate
    - Average turns per task
    - Clarification rate
    - Intent confusion rate
    - Error recovery success rate
    - User satisfaction (if feedback available)
    """
    
    def __init__(self, db_path: str = "~/.biopipelines/conversation_analytics.db"):
        self.db_path = Path(db_path).expanduser()
        self._init_db()
    
    def record_conversation(self, metrics: ConversationMetrics):
        """Record a completed conversation."""
        pass
    
    def record_turn(self, session_id: str, turn: TurnRecord):
        """Record a single turn."""
        pass
    
    def get_completion_rate(self, days: int = 30) -> float:
        """Get task completion rate."""
        pass
    
    def get_confusion_matrix(self) -> Dict[str, Dict[str, int]]:
        """Get intent confusion matrix."""
        pass
    
    def get_drop_off_analysis(self) -> List[DropOffPoint]:
        """Analyze where users abandon conversations."""
        pass
    
    def get_dashboard_data(self) -> DashboardData:
        """Get all data needed for analytics dashboard."""
        return DashboardData(
            completion_rate=self.get_completion_rate(),
            avg_turns=self.get_average_turns(),
            top_intents=self.get_top_intents(),
            confusion_matrix=self.get_confusion_matrix(),
            error_rate=self.get_error_rate(),
            satisfaction=self.get_satisfaction_score(),
            trends=self.get_trends(),
        )
```

#### 5.2 Analytics Dashboard (Gradio)

```python
# New file: src/workflow_composer/web/analytics_dashboard.py

import gradio as gr
from workflow_composer.agents.intent.conversation_analytics import ConversationAnalytics

def create_analytics_dashboard():
    """Create Gradio analytics dashboard."""
    analytics = ConversationAnalytics()
    
    with gr.Blocks(title="BioPipelines Analytics") as dashboard:
        gr.Markdown("# 📊 Conversation Analytics Dashboard")
        
        with gr.Row():
            # KPI Cards
            with gr.Column():
                completion_rate = gr.Number(
                    label="Completion Rate",
                    value=lambda: analytics.get_completion_rate()
                )
            with gr.Column():
                avg_turns = gr.Number(
                    label="Avg Turns per Task",
                    value=lambda: analytics.get_average_turns()
                )
            with gr.Column():
                error_rate = gr.Number(
                    label="Error Rate",
                    value=lambda: analytics.get_error_rate()
                )
        
        with gr.Row():
            # Intent Distribution
            intent_plot = gr.Plot(label="Intent Distribution")
            
            # Confusion Matrix
            confusion_plot = gr.Plot(label="Intent Confusion")
        
        with gr.Row():
            # Trends
            trend_plot = gr.Plot(label="Weekly Trends")
        
        # Refresh button
        refresh_btn = gr.Button("Refresh")
        refresh_btn.click(
            fn=lambda: analytics.get_dashboard_data(),
            outputs=[completion_rate, avg_turns, error_rate, 
                    intent_plot, confusion_plot, trend_plot]
        )
    
    return dashboard
```

---

### Phase 6: Integration & Testing (Priority: HIGH)

**Goal**: Wire everything together with comprehensive tests

#### 6.1 Updated Agent Architecture

```python
# Updated: src/workflow_composer/agents/unified_agent.py

class UnifiedAgent:
    """
    Production-grade unified agent.
    
    Architecture:
        User Input
            ↓
        DialogStateMachine (state tracking)
            ↓
        UnifiedIntentParser (NLU)
            ↓
        SlotPrompter/FormManager (slot filling)
            ↓
        HybridPolicy (action selection)
            ↓
        ResponseGenerator (response)
            ↓
        ConversationAnalytics (tracking)
    """
    
    def __init__(
        self,
        autonomy_level: AutonomyLevel = AutonomyLevel.ASSISTED,
        use_arbiter: bool = True,
        persona: str = "helpful_assistant",
    ):
        # Core components
        self.fsm = DialogStateMachine()
        self.parser = UnifiedIntentParser(use_cascade=True)
        self.form_manager = FormManager()
        self.policy = HybridPolicy()
        self.response_gen = ResponseGenerator(persona=persona)
        
        # Memory & Context
        self.session_memory = get_session_memory()
        self.conversation_context = ConversationContext()
        self.recovery = ConversationRecovery()
        
        # Analytics
        self.analytics = ConversationAnalytics()
        
        # Tools & Execution
        self.tools = get_agent_tools()
        self.permissions = PermissionManager(autonomy_level)
    
    async def process(self, query: str, session_id: str = None) -> AgentResponse:
        """Process a user query through the full pipeline."""
        
        # 1. State Check
        current_state = self.fsm.current_state
        
        # 2. Reference Resolution
        resolved_query = self.session_memory.resolve_references_in_query(query)
        
        # 3. Form Handling (if active)
        if self.form_manager.active_form:
            form_result = self.form_manager.process_input(query)
            if form_result.needs_more:
                return self._build_response(form_result.prompt)
            # Form complete, proceed with values
        
        # 4. Intent Parsing
        parse_result = self.parser.parse(
            resolved_query,
            context=self.conversation_context.get_context_for_llm()
        )
        
        # 5. Low Confidence Handling
        if parse_result.confidence < 0.4:
            recovery_response = self.recovery.handle_low_confidence(
                query, parse_result.confidence
            )
            return self._build_response(recovery_response.message)
        
        # 6. Slot Checking
        slot_result = get_slot_prompter().check_slots(
            parse_result.intent.name,
            parse_result.slots
        )
        if slot_result.needs_prompting:
            self.form_manager.activate_form_for_intent(parse_result.intent.name)
            return self._build_response(slot_result.prompt)
        
        # 7. Policy Decision
        policy_context = self._build_policy_context(parse_result)
        decision = self.policy.decide(policy_context)
        
        # 8. Confirmation (if needed)
        if decision.action.startswith("confirm_"):
            self.fsm.transition("needs_confirmation", {})
            return self._build_confirmation_response(decision)
        
        # 9. Tool Execution
        tool_result = await self._execute_tool(decision.action, parse_result.slots)
        
        # 10. Response Generation
        response = self.response_gen.generate(
            ResponseType.RESULT if tool_result.success else ResponseType.ERROR,
            context={
                "result": tool_result,
                **parse_result.slots
            },
            intent=parse_result.intent.name
        )
        
        # 11. Analytics Recording
        self.analytics.record_turn(session_id, TurnRecord(
            query=query,
            intent=parse_result.intent.name,
            confidence=parse_result.confidence,
            action=decision.action,
            success=tool_result.success
        ))
        
        # 12. Memory Update
        self.session_memory.record_action(
            action_type=parse_result.intent.name,
            query=query,
            tool_used=decision.action,
            success=tool_result.success
        )
        
        return self._build_response(response, tool_result)
```

#### 6.2 Test Suite Structure

```
tests/
├── test_dialog_state_machine.py    # FSM unit tests
├── test_form_system.py             # Form lifecycle tests
├── test_response_generator.py      # Response variation tests
├── test_policy.py                  # Policy decision tests
├── test_integration.py             # End-to-end tests
├── test_conversation_flows.py      # Multi-turn scenario tests
└── fixtures/
    ├── conversations.yaml          # Test conversation flows
    └── expected_responses.yaml     # Expected outputs
```

#### 6.3 Integration Tests

```python
# tests/test_conversation_flows.py

@pytest.fixture
def agent():
    return UnifiedAgent(autonomy_level=AutonomyLevel.READONLY)


class TestConversationFlows:
    """Test complete conversation flows."""
    
    @pytest.mark.asyncio
    async def test_simple_scan_flow(self, agent):
        """Test: user asks to scan data."""
        response = await agent.process("scan my data")
        
        assert response.success
        assert "scan" in response.message.lower() or "files" in response.message.lower()
    
    @pytest.mark.asyncio
    async def test_slot_filling_flow(self, agent):
        """Test: user needs to provide missing info."""
        # Incomplete query
        r1 = await agent.process("create a workflow")
        assert "workflow" in r1.message.lower()
        assert "type" in r1.message.lower()  # Asks for workflow type
        
        # Provide missing info
        r2 = await agent.process("RNA-seq")
        assert r2.success or "data" in r2.message.lower()  # Either completes or asks for data
    
    @pytest.mark.asyncio
    async def test_clarification_flow(self, agent):
        """Test: ambiguous query triggers clarification."""
        response = await agent.process("do the thing")
        
        # Should ask for clarification, not fail
        assert not response.success or "what" in response.message.lower()
    
    @pytest.mark.asyncio
    async def test_error_recovery_flow(self, agent):
        """Test: error is handled gracefully."""
        response = await agent.process("scan /nonexistent/path/12345")
        
        # Should acknowledge error helpfully
        assert "not found" in response.message.lower() or "couldn't" in response.message.lower()
    
    @pytest.mark.asyncio
    async def test_correction_flow(self, agent):
        """Test: user corrects agent."""
        r1 = await agent.process("search for data")  # Might trigger scan
        r2 = await agent.process("no, I meant search online databases")
        
        # Should acknowledge correction
        assert "search" in r2.message.lower() or "database" in r2.message.lower()
    
    @pytest.mark.asyncio
    async def test_context_persistence(self, agent):
        """Test: context persists across turns."""
        r1 = await agent.process("scan /data/methylation")
        r2 = await agent.process("what's in that folder?")  # Reference to previous
        
        # Should understand "that folder"
        assert r2.success or "methylation" in r2.message.lower()
```

---

## Part 3: Implementation Priority & Timeline

### Priority Matrix

| Phase | Component | Priority | Complexity | Dependencies | Est. Time |
|-------|-----------|----------|------------|--------------|-----------|
| 1 | Dialog State Machine | CRITICAL | High | None | 3-4 days |
| 2 | Form System | HIGH | Medium | Phase 1 | 2-3 days |
| 3 | Response Generator | HIGH | Medium | None | 2 days |
| 4 | Policy Layer | MEDIUM | Medium | Phase 1, 2 | 2 days |
| 5 | Conversation Analytics | MEDIUM | Low | None | 1-2 days |
| 6 | Integration & Testing | HIGH | High | All | 2-3 days |

### Recommended Order

1. **Phase 1**: Dialog State Machine (foundation for everything)
2. **Phase 3**: Response Generator (quick win, visible improvement)
3. **Phase 2**: Form System (builds on Phase 1)
4. **Phase 4**: Policy Layer (optional enhancement)
5. **Phase 5**: Analytics (can be added anytime)
6. **Phase 6**: Integration (final polish)

---

## Part 4: Success Criteria

### Functional Requirements
- [ ] All states have defined transitions
- [ ] Form validation catches invalid input
- [ ] Responses never repeat the same phrasing 3x in a row
- [ ] 95%+ of intents resolve without clarification
- [ ] Error recovery provides actionable suggestions

### Quality Metrics
- [ ] Conversation completion rate > 80%
- [ ] Average clarifications per task < 1.5
- [ ] Intent classification accuracy > 90%
- [ ] User satisfaction > 4.0/5.0 (when tracked)

### Test Coverage
- [ ] Unit tests for each component
- [ ] Integration tests for common flows
- [ ] Regression tests for fixed bugs
- [ ] Performance benchmarks

---

## Part 5: Files to Create

```
New Files:
├── src/workflow_composer/agents/intent/
│   ├── dialog_state_machine.py     # Phase 1
│   ├── forms.py                    # Phase 2
│   ├── response_generator.py       # Phase 3
│   ├── policy.py                   # Phase 4
│   └── conversation_analytics.py   # Phase 5
│
├── config/nlu/
│   ├── forms/
│   │   ├── search_form.yaml
│   │   ├── workflow_form.yaml
│   │   └── download_form.yaml
│   │
│   ├── responses/
│   │   ├── templates.yaml
│   │   └── personas.yaml
│   │
│   └── policies/
│       └── default.yaml
│
├── src/workflow_composer/web/
│   └── analytics_dashboard.py      # Phase 5
│
└── tests/
    ├── test_dialog_state_machine.py
    ├── test_form_system.py
    ├── test_response_generator.py
    ├── test_policy.py
    └── test_conversation_flows.py
```

---

## Next Steps

**Shall I proceed with Phase 1 (Dialog State Machine)?**

This is the foundation that enables:
- Systematic slot filling
- Proper confirmation handling
- Error recovery flows
- Multi-turn conversations

Once the FSM is in place, all other enhancements become much easier to implement.
