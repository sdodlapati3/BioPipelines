"""RAG Layer 3: Tool Selection.

Uses learned patterns from successful executions to help select the best
tool when multiple tools could handle a query. This layer provides value
when you have overlapping tool capabilities (granular tools).

Example:
    >>> from workflow_composer.agents.rag import RAGToolSelector, get_rag_selector
    >>> 
    >>> selector = get_rag_selector()
    >>> 
    >>> # Get tool boost based on past success
    >>> boost = selector.get_tool_boost(
    ...     query="find H3K4me3 ChIP-seq data from human",
    ...     candidate_tools=["search_databases", "search_encode", "search_geo"]
    ... )
    >>> print(boost.best_tool)  # "search_encode" if historically successful
    >>> print(boost.boosts)  # {"search_encode": 0.8, "search_geo": 0.3, ...}
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .memory import ToolMemory, get_tool_memory

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RAGSelectorConfig:
    """Configuration for RAG tool selector.
    
    Attributes:
        min_records: Minimum records needed to make confident selection
        similarity_threshold: Query similarity threshold
        success_weight: How much to weight success rate vs frequency
        recency_weight: How much to weight recent executions
        max_candidates: Maximum tools to consider for boosting
    """
    min_records: int = 5
    similarity_threshold: float = 0.3
    success_weight: float = 0.7
    recency_weight: float = 0.3
    max_candidates: int = 10


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ToolBoost:
    """Boosting information for tool selection.
    
    Attributes:
        best_tool: The tool with highest boost (or None if no data)
        boosts: Boost scores for each candidate tool (0-1)
        confidence: Overall confidence in the recommendation
        sources: Number of past executions analyzed
    """
    best_tool: Optional[str]
    boosts: Dict[str, float]
    confidence: float
    sources: int
    
    def apply_to_scores(
        self,
        base_scores: Dict[str, float],
        weight: float = 0.3,
    ) -> Dict[str, float]:
        """Apply boosts to base tool scores.
        
        Args:
            base_scores: Original tool scores (from deterministic selection)
            weight: How much weight to give RAG boosts (0-1)
            
        Returns:
            Adjusted scores combining base + RAG boost
        """
        result = dict(base_scores)
        
        for tool, boost in self.boosts.items():
            if tool in result:
                # Weighted combination
                result[tool] = (
                    (1 - weight) * result[tool] +
                    weight * boost * self.confidence
                )
        
        return result


# =============================================================================
# RAG Tool Selector
# =============================================================================

class RAGToolSelector:
    """RAG-based tool selector (Layer 3).
    
    Analyzes successful past executions to boost tool selection confidence
    when deterministic selection is uncertain.
    
    This layer is most valuable when:
    1. You have multiple granular tools with overlapping capabilities
    2. The deterministic selector returns close scores for multiple tools
    3. You have sufficient execution history to learn patterns
    
    Depends on: ToolMemory (Layer 1)
    """
    
    _instance: Optional["RAGToolSelector"] = None
    _lock = threading.Lock()
    
    def __init__(
        self,
        tool_memory: Optional[ToolMemory] = None,
        config: Optional[RAGSelectorConfig] = None,
    ):
        self.tool_memory = tool_memory or get_tool_memory()
        self.config = config or RAGSelectorConfig()
    
    @classmethod
    def get_instance(
        cls,
        tool_memory: Optional[ToolMemory] = None,
        config: Optional[RAGSelectorConfig] = None,
    ) -> "RAGToolSelector":
        """Get singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = RAGToolSelector(tool_memory, config)
            return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None
    
    def get_tool_boost(
        self,
        query: str,
        candidate_tools: List[str],
    ) -> ToolBoost:
        """Get boost scores for candidate tools based on RAG analysis.
        
        Args:
            query: The user query
            candidate_tools: List of tool names to consider
            
        Returns:
            ToolBoost with scores and recommendation
        """
        if not candidate_tools:
            return ToolBoost(
                best_tool=None,
                boosts={},
                confidence=0,
                sources=0,
            )
        
        # Find similar successful executions
        similar = self.tool_memory.find_similar(
            query=query,
            success_only=True,
            limit=50,
        )
        
        if not similar:
            return ToolBoost(
                best_tool=None,
                boosts={t: 0 for t in candidate_tools},
                confidence=0,
                sources=0,
            )
        
        # Aggregate by tool
        tool_scores: Dict[str, float] = {}
        tool_counts: Dict[str, int] = {}
        
        for record, similarity in similar:
            if record.tool_name not in candidate_tools:
                continue
            
            tool = record.tool_name
            
            if tool not in tool_scores:
                tool_scores[tool] = 0
                tool_counts[tool] = 0
            
            # Weight by similarity
            tool_scores[tool] += similarity
            tool_counts[tool] += 1
        
        # No candidates found in history
        if not tool_scores:
            return ToolBoost(
                best_tool=None,
                boosts={t: 0 for t in candidate_tools},
                confidence=0,
                sources=len(similar),
            )
        
        # Normalize to 0-1
        max_score = max(tool_scores.values()) if tool_scores else 1
        boosts = {
            tool: score / max_score
            for tool, score in tool_scores.items()
        }
        
        # Add zero for candidates not in history
        for tool in candidate_tools:
            if tool not in boosts:
                boosts[tool] = 0
        
        # Find best tool
        best_tool = max(boosts.items(), key=lambda x: x[1])[0] if boosts else None
        
        # Calculate confidence based on data quality
        total_records = sum(tool_counts.values())
        confidence = min(1.0, total_records / self.config.min_records)
        
        return ToolBoost(
            best_tool=best_tool,
            boosts=boosts,
            confidence=confidence,
            sources=total_records,
        )
    
    def should_use_rag(
        self,
        base_scores: Dict[str, float],
        uncertainty_threshold: float = 0.2,
    ) -> bool:
        """Determine if RAG selection should be used.
        
        RAG is most valuable when deterministic selection is uncertain
        (multiple tools have similar scores).
        
        Args:
            base_scores: Scores from deterministic selector
            uncertainty_threshold: Max difference between top 2 to trigger RAG
            
        Returns:
            True if RAG should be consulted
        """
        if len(base_scores) < 2:
            return False
        
        sorted_scores = sorted(base_scores.values(), reverse=True)
        top_diff = sorted_scores[0] - sorted_scores[1]
        
        return top_diff <= uncertainty_threshold
    
    def select_tool(
        self,
        query: str,
        base_scores: Dict[str, float],
        rag_weight: float = 0.3,
    ) -> Tuple[str, Dict[str, float]]:
        """Select best tool combining deterministic and RAG scores.
        
        Args:
            query: User query
            base_scores: Scores from deterministic selector
            rag_weight: Weight to give RAG scores (0-1)
            
        Returns:
            Tuple of (selected_tool, adjusted_scores)
        """
        candidate_tools = list(base_scores.keys())
        
        # Get RAG boost
        boost = self.get_tool_boost(query, candidate_tools)
        
        # If no RAG data, use base scores
        if boost.confidence == 0:
            best_tool = max(base_scores.items(), key=lambda x: x[1])[0]
            return best_tool, base_scores
        
        # Apply boost
        adjusted = boost.apply_to_scores(base_scores, rag_weight)
        best_tool = max(adjusted.items(), key=lambda x: x[1])[0]
        
        return best_tool, adjusted


# =============================================================================
# Singleton Accessor
# =============================================================================

_rag_selector: Optional[RAGToolSelector] = None
_rag_selector_lock = threading.Lock()


def get_rag_selector(
    tool_memory: Optional[ToolMemory] = None,
    config: Optional[RAGSelectorConfig] = None,
    reset: bool = False,
) -> RAGToolSelector:
    """Get the singleton RAGToolSelector instance.
    
    Args:
        tool_memory: ToolMemory to use (uses singleton if None)
        config: Optional configuration
        reset: If True, create new instance
        
    Returns:
        RAGToolSelector singleton instance
    """
    global _rag_selector
    
    with _rag_selector_lock:
        if _rag_selector is None or reset:
            _rag_selector = RAGToolSelector(tool_memory, config)
        return _rag_selector
