"""
Intent Parsing & Context Management
====================================

Enhanced natural language understanding for the BioPipelines agent.

Components:
- IntentParser: Pattern-based intent detection with entity extraction
- HybridQueryParser: Production-grade hybrid parser (pattern + semantic + NER)
- UnifiedIntentParser: Hierarchical parser with LLM arbiter (RECOMMENDED)
- SemanticIntentClassifier: FAISS-based semantic similarity classification
- BioinformaticsNER: Domain-specific named entity recognition
- ConversationContext: Semantic memory and coreference resolution
- DialogueManager: Conversation flow and multi-turn task tracking
- ChatIntegration: Integration layer for BioPipelines facade

Architecture (UnifiedIntentParser - Recommended):

    +-------------------------------------------------------------+
    |              UnifiedIntentParser                             |
    |  +-------------------------------------------------------+  |
    |  |               Fast Methods (~15ms)                     |  |
    |  |  +----------+----------+----------+                    |  |
    |  |  | Pattern  | Semantic | Entity   |                    |  |
    |  |  | Matching | FAISS    | Extraction|                   |  |
    |  |  +----------+----------+----------+                    |  |
    |  |                     |                                  |  |
    |  |           Agreement Check (80% pass here)              |  |
    |  |                     |                                  |  |
    |  |  +-----------------------------------------+           |  |
    |  |  | LLM Arbiter (20% - only complex cases)  |           |  |
    |  |  | - Disagreement detection                 |           |  |
    |  |  | - Negation handling                      |           |  |
    |  |  | - Context-aware reasoning                |           |  |
    |  |  +-----------------------------------------+           |  |
    |  |                     |                                  |  |
    |  |          Final Intent + Confidence                     |  |
    |  +-------------------------------------------------------+  |
    +-------------------------------------------------------------+

Usage:
    # Recommended: Use UnifiedIntentParser for best accuracy
    from workflow_composer.agents.intent import UnifiedIntentParser
    
    parser = UnifiedIntentParser()
    result = parser.parse("search for human brain RNA-seq data")
    print(result.primary_intent)  # IntentType.DATA_SEARCH
    print(result.confidence)      # 0.92
    print(result.method)          # "unanimous" or "llm_arbiter"
    
    # With chat integration (uses arbiter automatically)
    from workflow_composer.agents.intent import ChatIntegration
    
    intent_system = ChatIntegration(use_arbiter=True)
    result = intent_system.process_message(message, session_id)
"""

from .parser import IntentParser, IntentType, IntentResult, Entity, EntityType
from .context import ConversationContext, ContextMemory, EntityTracker, MemoryItem, EntityReference
from .dialogue import (
    DialogueManager, 
    DialogueResult,
    ConversationState, 
    ConversationPhase,
    TaskState,
    TaskStatus,
    SlotFiller,
)
from .integration import ChatIntegration, route_to_tool, format_clarification_response
from .semantic import (
    HybridQueryParser,
    SemanticIntentClassifier,
    BioinformaticsNER,
    BioEntity,
    QueryParseResult,
    INTENT_EXAMPLES,
    FAISS_AVAILABLE,
    SENTENCE_TRANSFORMERS_AVAILABLE,
)
from .learning import (
    LearningHybridParser,
    QueryLogger,
    FeedbackManager,
    LLMIntentClassifier,
    FineTuningExporter,
)
from .arbiter import (
    IntentArbiter,
    ArbiterResult,
    ArbiterStrategy,
    ParserVote,
)
from .unified_parser import (
    UnifiedIntentParser,
    UnifiedParseResult,
    create_unified_parser,
)

# Professional NLU components (Phase 1-5)
from .training_data import (
    TrainingDataLoader,
    SlotValidator,
    SlotValidationResult,
    IntentDefinition,
    EntityDefinition,
    EntityValue,
    get_training_data_loader,
    reload_training_data,
)
from .active_learning import (
    ActiveLearner,
    CorrectionRecord,
    ConfirmationRecord,
    LearningMetrics,
    get_active_learner,
)
from .slot_prompting import (
    SlotPrompter,
    SlotCheckResult,
    SlotDefinition as SlotSchema,
    IntentSlotSchema,
    DialogueState,
    get_slot_prompter,
    SlotPriority,
    PromptStyle,
)
from .balance_metrics import (
    TrainingDataAnalyzer,
    BalanceReport,
    IntentStats,
    analyze_training_balance,
)
from .entity_roles import (
    EntityRoleResolver,
    ResolvedEntity,
    RoleDefinition,
    get_role_resolver,
)

# Session Memory & Recovery (Robust Agent Features)
from .session_memory import (
    SessionMemory,
    MemoryEntry,
    ActionRecord,
    MemoryType,
    MemoryPriority,
    get_session_memory,
)
from .conversation_recovery import (
    ConversationRecovery,
    RecoveryResponse,
    RecoveryStrategy,
    ErrorCategory,
    ErrorContext,
    get_conversation_recovery,
)

# Dialog State Machine (Professional Agent Phase 1)
from .dialog_state_machine import (
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

# Response Generation System (Professional Agent Phase 2)
from .response_generator import (
    ResponseGenerator,
    ResponseTemplate,
    Response,
    ResponseComponent,
    ResponseType,
    ResponseTone,
    Suggestion,
    Card,
    CodeBlock,
    TableData,
    ProgressIndicator,
    TemplateRenderer,
    ResponseHistoryTracker,
    create_response_generator,
)

# Conversation Analytics (Professional Agent Phase 3)
from .conversation_analytics import (
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

# Human Handoff System (Professional Agent Phase 4)
from .human_handoff import (
    HumanHandoffManager,
    HandoffRequest,
    HandoffStatus,
    HandoffQueue,
    HandoffProtocol,
    HandoffMetrics,
    EscalationReason,
    EscalationTrigger,
    ConversationContext as HandoffContext,
    HumanAgent,
    AgentStatus,
    Priority as HandoffPriority,
    AgentRouter,
    RoundRobinRouter,
    SkillBasedRouter,
    LoadBalancedRouter,
    get_handoff_manager,
    reset_handoff_manager,
)

# A/B Testing Framework (Professional Agent Phase 5)
from .ab_testing import (
    ABTestingManager,
    Experiment,
    ExperimentStatus,
    ExperimentReport,
    Variant,
    MetricDefinition,
    MetricType as ABMetricType,
    MetricValue,
    VariantMetrics,
    AssignmentStrategy,
    AssignmentAlgorithm,
    RandomAssignment,
    DeterministicAssignment,
    RoundRobinAssignment,
    WeightedRandomAssignment,
    StatisticalAnalyzer,
    StatisticalResult,
    ExperimentStore,
    MemoryExperimentStore,
    FileExperimentStore,
    get_ab_testing_manager,
    reset_ab_testing_manager,
    create_simple_ab_test,
    get_variant_config,
)

# Rich Response Helpers (Professional Agent Phase 6)
from .rich_responses import (
    MessageFormatter,
    MessageBuilder,
    MessageFormat,
    ComponentType,
    CalloutType,
    ButtonStyle,
    ListStyle,
    RichComponent,
    TextComponent,
    HeadingComponent,
    ListComponent,
    CodeComponent,
    TableComponent,
    ImageComponent,
    LinkComponent,
    ButtonComponent,
    CardComponent,
    CalloutComponent,
    ProgressComponent,
    QuoteComponent,
    CarouselComponent,
    FormComponent,
    InputComponent,
    FormatAdapter,
    PlainTextAdapter,
    MarkdownAdapter,
    HTMLAdapter,
    SlackAdapter,
    strip_formatting,
    truncate_text,
    word_wrap,
    format_duration,
    format_number,
    format_bytes,
    get_message_formatter,
    reset_message_formatter,
)

# Out-of-Scope Detection (Professional Agent Phase 7)
from .out_of_scope import (
    ScopeCategory,
    DeflectionStrategy,
    Topic,
    ScopeResult,
    DeflectionResponse,
    ScopeClassifier,
    KeywordScopeClassifier,
    PatternScopeClassifier,
    EnsembleScopeClassifier,
    DeflectionResponseGenerator,
    OutOfScopeHandler,
    DomainKnowledge,
    BioinformaticsDomainKnowledge,
    get_out_of_scope_handler,
    reset_out_of_scope_handler,
    check_query_scope,
)

# Chat Agent Integration (Professional Agent Phase 8)
from .chat_agent import (
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

__all__ = [
    # High-level (recommended)
    "UnifiedIntentParser",    # RECOMMENDED: Hierarchical parser with LLM arbiter
    "UnifiedParseResult",     # Result from UnifiedIntentParser
    "create_unified_parser",  # Factory for UnifiedIntentParser
    "IntentArbiter",          # LLM arbiter for complex queries
    "ArbiterResult",          # Result from arbiter
    "ArbiterStrategy",        # Strategy for LLM invocation
    "ParserVote",             # Individual parser vote
    "HybridQueryParser",      # Production-grade hybrid parser
    "LearningHybridParser",   # With active learning & feedback
    "ChatIntegration",        # Chat handler integration
    
    # Learning components
    "QueryLogger",
    "FeedbackManager",
    "LLMIntentClassifier",
    "FineTuningExporter",
    
    # Semantic components
    "SemanticIntentClassifier",
    "BioinformaticsNER",
    "BioEntity",
    "QueryParseResult",
    "INTENT_EXAMPLES",
    "FAISS_AVAILABLE",
    "SENTENCE_TRANSFORMERS_AVAILABLE",
    
    # Pattern-based parser
    "IntentParser",
    "IntentType", 
    "IntentResult",
    "Entity",
    "EntityType",
    
    # Context management
    "ConversationContext",
    "ContextMemory",
    "EntityTracker",
    "MemoryItem",
    "EntityReference",
    
    # Dialogue management
    "DialogueManager",
    "DialogueResult",
    "ConversationState",
    "ConversationPhase",
    "TaskState",
    "TaskStatus",
    "SlotFiller",
    
    # Integration helpers
    "route_to_tool",
    "format_clarification_response",
    
    # Professional NLU - Training Data (Phase 1)
    "TrainingDataLoader",
    "SlotValidator",
    "SlotValidationResult",
    "IntentDefinition",
    "EntityDefinition",
    "EntityValue",
    "get_training_data_loader",
    "reload_training_data",
    
    # Professional NLU - Active Learning (Phase 2)
    "ActiveLearner",
    "CorrectionRecord",
    "ConfirmationRecord",
    "LearningMetrics",
    "get_active_learner",
    
    # Professional NLU - Slot Prompting (Phase 3)
    "SlotPrompter",
    "SlotCheckResult",
    "SlotSchema",
    "IntentSlotSchema",
    "DialogueState",
    "get_slot_prompter",
    "SlotPriority",
    "PromptStyle",
    
    # Professional NLU - Balance Metrics (Phase 4)
    "TrainingDataAnalyzer",
    "BalanceReport",
    "IntentStats",
    "analyze_training_balance",
    
    # Professional NLU - Entity Roles (Phase 5)
    "EntityRoleResolver",
    "ResolvedEntity",
    "RoleDefinition",
    "get_role_resolver",
    
    # Session Memory & Recovery (Robust Agent)
    "SessionMemory",
    "MemoryEntry", 
    "ActionRecord",
    "MemoryType",
    "MemoryPriority",
    "get_session_memory",
    "ConversationRecovery",
    "RecoveryResponse",
    "RecoveryStrategy",
    "ErrorCategory",
    "ErrorContext",
    "get_conversation_recovery",
    
    # Dialog State Machine (Professional Agent Phase 1)
    "DialogStateMachine",
    "DialogState",
    "DialogEvent",
    "DialogContext",
    "StateTransition",
    "StateConfig",
    "StateHistoryEntry",
    "StateHandler",
    "IdleHandler",
    "SlotFillingHandler",
    "ConfirmingHandler",
    "ErrorRecoveryHandler",
    "get_dialog_state_machine",
    "create_dialog_state_machine",
    
    # Response Generation System (Professional Agent Phase 2)
    "ResponseGenerator",
    "ResponseTemplate",
    "Response",
    "ResponseComponent",
    "ResponseType",
    "ResponseTone",
    "Suggestion",
    "Card",
    "CodeBlock",
    "TableData",
    "ProgressIndicator",
    "TemplateRenderer",
    "ResponseHistoryTracker",
    "create_response_generator",
    
    # Conversation Analytics (Professional Agent Phase 3)
    "ConversationAnalytics",
    "AnalyticsDashboard",
    "ConversationMetrics",
    "IntentMetrics",
    "StateMetrics",
    "AggregateMetrics",
    "MetricPoint",
    "MetricType",
    "ConversationOutcome",
    "MetricsStorage",
    "InMemoryMetricsStorage",
    "FileMetricsStorage",
    "get_conversation_analytics",
    "get_analytics_dashboard",
    "reset_analytics",
    
    # Human Handoff System (Professional Agent Phase 4)
    "HumanHandoffManager",
    "HandoffRequest",
    "HandoffStatus",
    "HandoffQueue",
    "HandoffProtocol",
    "HandoffMetrics",
    "EscalationReason",
    "EscalationTrigger",
    "HandoffContext",
    "HumanAgent",
    "AgentStatus",
    "HandoffPriority",
    "AgentRouter",
    "RoundRobinRouter",
    "SkillBasedRouter",
    "LoadBalancedRouter",
    "get_handoff_manager",
    "reset_handoff_manager",
    
    # A/B Testing Framework (Professional Agent Phase 5)
    "ABTestingManager",
    "Experiment",
    "ExperimentStatus",
    "ExperimentReport",
    "Variant",
    "MetricDefinition",
    "ABMetricType",
    "MetricValue",
    "VariantMetrics",
    "AssignmentStrategy",
    "AssignmentAlgorithm",
    "RandomAssignment",
    "DeterministicAssignment",
    "RoundRobinAssignment",
    "WeightedRandomAssignment",
    "StatisticalAnalyzer",
    "StatisticalResult",
    "ExperimentStore",
    "MemoryExperimentStore",
    "FileExperimentStore",
    "get_ab_testing_manager",
    "reset_ab_testing_manager",
    "create_simple_ab_test",
    "get_variant_config",
    
    # Rich Response Helpers (Professional Agent Phase 6)
    "MessageFormatter",
    "MessageBuilder",
    "MessageFormat",
    "ComponentType",
    "CalloutType",
    "ButtonStyle",
    "ListStyle",
    "RichComponent",
    "TextComponent",
    "HeadingComponent",
    "ListComponent",
    "CodeComponent",
    "TableComponent",
    "ImageComponent",
    "LinkComponent",
    "ButtonComponent",
    "CardComponent",
    "CalloutComponent",
    "ProgressComponent",
    "QuoteComponent",
    "CarouselComponent",
    "FormComponent",
    "InputComponent",
    "FormatAdapter",
    "PlainTextAdapter",
    "MarkdownAdapter",
    "HTMLAdapter",
    "SlackAdapter",
    "strip_formatting",
    "truncate_text",
    "word_wrap",
    "format_duration",
    "format_number",
    "format_bytes",
    "get_message_formatter",
    "reset_message_formatter",
    
    # Out-of-Scope Detection (Professional Agent Phase 7)
    "ScopeCategory",
    "DeflectionStrategy",
    "Topic",
    "ScopeResult",
    "DeflectionResponse",
    "ScopeClassifier",
    "KeywordScopeClassifier",
    "PatternScopeClassifier",
    "EnsembleScopeClassifier",
    "DeflectionResponseGenerator",
    "OutOfScopeHandler",
    "DomainKnowledge",
    "BioinformaticsDomainKnowledge",
    "get_out_of_scope_handler",
    "reset_out_of_scope_handler",
    "check_query_scope",
    
    # Chat Agent Integration (Professional Agent Phase 8)
    "ChatAgent",
    "AgentConfig",
    "AgentCapability",
    "ChannelType",
    "MessageDirection",
    "Message",
    "Session",
    "AgentResponse",
    "AgentPlugin",
    "LoggingPlugin",
    "MetricsPlugin",
    "SessionManager",
    "IntentRegistry",
    "get_chat_agent",
    "reset_chat_agent",
    "create_chat_agent",
    "quick_chat",
]
