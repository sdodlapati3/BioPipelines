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
]
