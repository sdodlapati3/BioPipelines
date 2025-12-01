"""RAG-Enhanced Tool System for BioPipelines.

This module provides a layered RAG (Retrieval-Augmented Generation) system
for improving tool selection and argument optimization based on past
successful executions.

Architecture:
    Layer 1 (memory.py):       ToolMemory - Records all executions
    Layer 2 (arg_memory.py):   ArgumentMemory - Learns optimal arguments
    Layer 3 (tool_selector.py): RAGToolSelector - Selects best tool
    Orchestrator:              RAGOrchestrator - Coordinates all layers
    
Enhanced Features (Phase 2.6):
    KnowledgeBase:             Multi-source document indexing and retrieval
    NFCoreIndexer:             nf-core module discovery and search
    ErrorPatternDB:            Error pattern matching and solutions

Example:
    >>> from workflow_composer.agents.rag import RAGOrchestrator
    >>> rag = RAGOrchestrator()
    >>> 
    >>> # Enhance tool execution
    >>> best_tool, optimized_args = rag.enhance(
    ...     query="find H3K4me3 ChIP-seq data",
    ...     candidate_tools=["search_databases", "search_encode"],
    ...     base_args={"limit": 50}
    ... )
    
    >>> # Use knowledge base for context
    >>> from workflow_composer.agents.rag import KnowledgeBase
    >>> kb = KnowledgeBase()
    >>> docs = await kb.search("RNA-seq alignment STAR", top_k=5)
    
    >>> # Get error solutions
    >>> from workflow_composer.agents.rag import ErrorPatternDB
    >>> error_db = ErrorPatternDB()
    >>> solution = error_db.find_solution("Out of memory error")
"""

from .arg_memory import (
    ArgumentMemory,
    ArgumentMemoryConfig,
    ArgumentSuggestion,
    get_argument_memory,
)
from .memory import (
    ToolExecutionRecord,
    ToolMemory,
    ToolMemoryConfig,
    ToolStats,
    get_tool_memory,
)
from .orchestrator import (
    RAGConfig,
    RAGOrchestrator,
    get_rag_orchestrator,
)
from .tool_selector import (
    RAGSelectorConfig,
    RAGToolSelector,
    ToolBoost,
    get_rag_selector,
)

# Phase 2.6: Enhanced RAG
from .knowledge_base import (
    KnowledgeBase,
    KnowledgeSource,
    KnowledgeDocument,
    KnowledgeBaseConfig,
)
from .error_patterns import (
    ErrorPatternDB,
    ErrorSolution,
)
from .nf_core_indexer import (
    NFCoreIndexer,
    NFCoreModule,
)

__all__ = [
    # Layer 1: Memory
    "ToolMemory",
    "ToolMemoryConfig",
    "ToolExecutionRecord",
    "ToolStats",
    "get_tool_memory",
    # Layer 2: Argument Optimization
    "ArgumentMemory",
    "ArgumentMemoryConfig",
    "ArgumentSuggestion",
    "get_argument_memory",
    # Layer 3: Tool Selection
    "RAGToolSelector",
    "RAGSelectorConfig",
    "ToolBoost",
    "get_rag_selector",
    # Orchestrator
    "RAGOrchestrator",
    "RAGConfig",
    "get_rag_orchestrator",
    # Phase 2.6: Enhanced RAG
    "KnowledgeBase",
    "KnowledgeSource",
    "KnowledgeDocument",
    "KnowledgeBaseConfig",
    "ErrorPatternDB",
    "ErrorSolution",
    "NFCoreIndexer",
    "NFCoreModule",
]
