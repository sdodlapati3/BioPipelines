"""RAG Orchestrator: Coordinates all RAG layers.

Provides a unified interface for the RAG system, coordinating:
- Layer 1 (ToolMemory): Recording executions
- Layer 2 (ArgumentMemory): Optimizing arguments
- Layer 3 (RAGToolSelector): Selecting tools

Example:
    >>> from workflow_composer.agents.rag import RAGOrchestrator, get_rag_orchestrator
    >>> 
    >>> rag = get_rag_orchestrator()
    >>> 
    >>> # Full enhancement pipeline
    >>> result = rag.enhance(
    ...     query="find H3K4me3 ChIP-seq data from human",
    ...     candidate_tools=["search_databases", "search_encode"],
    ...     base_args={"limit": 50}
    ... )
    >>> print(result.selected_tool)
    >>> print(result.enhanced_args)
    >>> 
    >>> # After execution, record it
    >>> rag.record_execution(
    ...     query="find H3K4me3 ChIP-seq data from human",
    ...     tool_name="search_encode",
    ...     tool_args=result.enhanced_args,
    ...     success=True,
    ...     results=search_results
    ... )
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .arg_memory import ArgumentMemory, ArgumentSuggestion, get_argument_memory
from .memory import ToolExecutionRecord, ToolMemory, get_tool_memory
from .tool_selector import RAGToolSelector, ToolBoost, get_rag_selector

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RAGConfig:
    """Configuration for RAG orchestrator.
    
    Attributes:
        enabled: Whether RAG is enabled
        use_argument_memory: Enable Layer 2 (argument optimization)
        use_tool_selector: Enable Layer 3 (tool selection)
        rag_weight: Weight for RAG in tool selection (0-1)
        uncertainty_threshold: Trigger RAG when top tools are within this
        auto_record: Automatically record executions
    """
    enabled: bool = True
    use_argument_memory: bool = True
    use_tool_selector: bool = True
    rag_weight: float = 0.3
    uncertainty_threshold: float = 0.2
    auto_record: bool = True


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EnhancementResult:
    """Result of RAG enhancement.
    
    Attributes:
        selected_tool: Best tool from combined scoring
        enhanced_args: Optimized arguments (merged with suggestions)
        tool_boost: Boost information from Layer 3
        arg_suggestion: Suggestion from Layer 2
        rag_used: Whether RAG influenced the decision
    """
    selected_tool: str
    enhanced_args: Dict[str, Any]
    tool_boost: Optional[ToolBoost] = None
    arg_suggestion: Optional[ArgumentSuggestion] = None
    rag_used: bool = False
    
    @property
    def confidence(self) -> float:
        """Overall confidence from RAG analysis."""
        if not self.rag_used:
            return 0.0
        
        confidences = []
        if self.tool_boost and self.tool_boost.confidence > 0:
            confidences.append(self.tool_boost.confidence)
        if self.arg_suggestion and self.arg_suggestion.confidence > 0:
            confidences.append(self.arg_suggestion.confidence)
        
        return sum(confidences) / len(confidences) if confidences else 0.0


# =============================================================================
# RAG Orchestrator
# =============================================================================

class RAGOrchestrator:
    """Orchestrates all RAG layers for enhanced tool execution.
    
    This is the main entry point for using the RAG system. It coordinates:
    
    1. **Recording** (always on): Every execution is recorded for learning
    2. **Enhancement**: Before execution, provides:
       - Argument suggestions (Layer 2)
       - Tool selection boost (Layer 3)
    
    Usage Patterns:
    
    1. Simple (just recording):
        >>> rag = get_rag_orchestrator()
        >>> with rag.record(query, tool, args):
        ...     result = execute_tool(tool, args)
    
    2. Full enhancement:
        >>> rag = get_rag_orchestrator()
        >>> enhanced = rag.enhance(query, tools, base_args)
        >>> result = execute_tool(enhanced.selected_tool, enhanced.enhanced_args)
        >>> rag.record_execution(query, enhanced.selected_tool, args, success=True)
    """
    
    _instance: Optional["RAGOrchestrator"] = None
    _lock = threading.Lock()
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        tool_memory: Optional[ToolMemory] = None,
        argument_memory: Optional[ArgumentMemory] = None,
        tool_selector: Optional[RAGToolSelector] = None,
    ):
        self.config = config or RAGConfig()
        
        # Initialize layers
        self.tool_memory = tool_memory or get_tool_memory()
        self.argument_memory = argument_memory or get_argument_memory(self.tool_memory)
        self.tool_selector = tool_selector or get_rag_selector(self.tool_memory)
    
    @classmethod
    def get_instance(
        cls,
        config: Optional[RAGConfig] = None,
        **kwargs,
    ) -> "RAGOrchestrator":
        """Get singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = RAGOrchestrator(config, **kwargs)
            return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None
    
    def enhance(
        self,
        query: str,
        candidate_tools: List[str],
        base_scores: Optional[Dict[str, float]] = None,
        base_args: Optional[Dict[str, Any]] = None,
    ) -> EnhancementResult:
        """Enhance tool selection and arguments using RAG.
        
        Args:
            query: User query
            candidate_tools: List of candidate tool names
            base_scores: Optional deterministic scores (uniform if not provided)
            base_args: Optional base arguments to merge with suggestions
            
        Returns:
            EnhancementResult with selected tool and enhanced args
        """
        if not self.config.enabled or not candidate_tools:
            # Return first tool with base args
            return EnhancementResult(
                selected_tool=candidate_tools[0] if candidate_tools else "",
                enhanced_args=base_args or {},
                rag_used=False,
            )
        
        # Initialize
        base_scores = base_scores or {t: 1.0 for t in candidate_tools}
        base_args = base_args or {}
        
        tool_boost = None
        arg_suggestion = None
        selected_tool = max(base_scores.items(), key=lambda x: x[1])[0]
        enhanced_args = dict(base_args)
        rag_used = False
        
        # Layer 3: Tool Selection
        if self.config.use_tool_selector:
            should_use = self.tool_selector.should_use_rag(
                base_scores,
                self.config.uncertainty_threshold,
            )
            
            if should_use:
                tool_boost = self.tool_selector.get_tool_boost(query, candidate_tools)
                
                if tool_boost.confidence > 0:
                    adjusted = tool_boost.apply_to_scores(
                        base_scores,
                        self.config.rag_weight,
                    )
                    selected_tool = max(adjusted.items(), key=lambda x: x[1])[0]
                    rag_used = True
        
        # Layer 2: Argument Optimization
        if self.config.use_argument_memory:
            arg_suggestion = self.argument_memory.suggest(
                query=query,
                tool_name=selected_tool,
                current_args=base_args,
            )
            
            if arg_suggestion.confidence > 0:
                enhanced_args = arg_suggestion.merge_with(base_args)
                rag_used = True
        
        return EnhancementResult(
            selected_tool=selected_tool,
            enhanced_args=enhanced_args,
            tool_boost=tool_boost,
            arg_suggestion=arg_suggestion,
            rag_used=rag_used,
        )
    
    def record_execution(
        self,
        query: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        success: bool,
        duration_ms: Optional[float] = None,
        result_summary: Optional[str] = None,
        error: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> ToolExecutionRecord:
        """Record a tool execution for learning.
        
        Args:
            query: User query that triggered execution
            tool_name: Name of executed tool
            tool_args: Arguments passed to tool
            success: Whether execution succeeded
            duration_ms: Execution time in milliseconds
            result_summary: Summary of results for learning
            error: Error message if failed
            user_id: Optional user identifier
            
        Returns:
            The created execution record
        """
        record = self.tool_memory.record(
            query=query,
            tool_name=tool_name,
            tool_args=tool_args,
            success=success,
            duration_ms=duration_ms or 0,
            result_summary=result_summary,
            error_message=error,
            user_id=user_id,
        )
        
        # Also update argument memory
        if self.config.use_argument_memory:
            self.argument_memory.learn_from_execution(
                query=query,
                tool_name=tool_name,
                args=tool_args,
                success=success,
            )
        
        return record
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics.
        
        Returns:
            Dict with stats from all layers
        """
        return {
            "enabled": self.config.enabled,
            "layers": {
                "tool_memory": True,
                "argument_memory": self.config.use_argument_memory,
                "tool_selector": self.config.use_tool_selector,
            },
            "tool_memory_stats": self.tool_memory.get_all_stats(),
        }
    
    def warm_up(self) -> None:
        """Warm up the RAG system by loading cached data.
        
        Call this at startup to pre-load data from the database.
        """
        # Tool memory loads from DB on first access
        _ = self.tool_memory.get_record_count()
        logger.info("RAG system warmed up")


# =============================================================================
# Singleton Accessor
# =============================================================================

_rag_orchestrator: Optional[RAGOrchestrator] = None
_rag_orchestrator_lock = threading.Lock()


def get_rag_orchestrator(
    config: Optional[RAGConfig] = None,
    reset: bool = False,
    **kwargs,
) -> RAGOrchestrator:
    """Get the singleton RAGOrchestrator instance.
    
    Args:
        config: Optional configuration
        reset: If True, create new instance
        **kwargs: Additional args for RAGOrchestrator
        
    Returns:
        RAGOrchestrator singleton instance
    """
    global _rag_orchestrator
    
    with _rag_orchestrator_lock:
        if _rag_orchestrator is None or reset:
            _rag_orchestrator = RAGOrchestrator(config, **kwargs)
        return _rag_orchestrator
