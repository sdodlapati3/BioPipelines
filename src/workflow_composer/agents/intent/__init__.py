"""
Intent Parsing & Context Management
====================================

Enhanced natural language understanding for the BioPipelines agent.

Components:
- IntentParser: Pattern-based intent detection with entity extraction
- HybridQueryParser: Production-grade hybrid parser (pattern + semantic + NER)
- SemanticIntentClassifier: FAISS-based semantic similarity classification
- BioinformaticsNER: Domain-specific named entity recognition
- ConversationContext: Semantic memory and coreference resolution
- DialogueManager: Conversation flow and multi-turn task tracking
- ChatIntegration: Integration layer for BioPipelines facade

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    HybridQueryParser                         │
    │  ┌──────────────┬───────────────────┬───────────────────┐   │
    │  │ Pattern      │ Semantic          │ NER               │   │
    │  │ Matching     │ Similarity        │ (BioinfoNER)      │   │
    │  │ (regex)      │ (FAISS/cosine)    │                   │   │
    │  └──────────────┴───────────────────┴───────────────────┘   │
    │                         ↓                                    │
    │              Confidence-Weighted Fusion                      │
    │                         ↓                                    │
    │              Intent + Entities + Slots                       │
    └─────────────────────────────────────────────────────────────┘

Usage:
    # Simple usage
    from workflow_composer.agents.intent import HybridQueryParser
    
    parser = HybridQueryParser()
    result = parser.parse("search for human brain RNA-seq data")
    print(result.intent)      # "DATA_SEARCH"
    print(result.entities)    # [ORGANISM:Homo sapiens, TISSUE:brain, ASSAY_TYPE:RNA-seq]
    print(result.slots)       # {"organism": "Homo sapiens", "tissue": "brain", ...}
    
    # With chat integration
    from workflow_composer.agents.intent import ChatIntegration
    
    intent_system = ChatIntegration()
    result = intent_system.process_message(message, session_id)
    
    if result.should_execute_tool:
        tool_result = call_tool(result.tool_name, result.tool_args)
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

__all__ = [
    # High-level (recommended)
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
