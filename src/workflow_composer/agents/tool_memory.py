"""RAG-Enhanced Tool Selection for BioPipelines.

.. deprecated:: 2.0
    This module is deprecated. Use :mod:`workflow_composer.agents.rag` instead.
    
    The new RAG system provides:
    - 3-layer architecture (ToolMemory, ArgumentMemory, RAGToolSelector)
    - Database persistence (PostgreSQL/SQLite)
    - Integrated RAGOrchestrator for unified access
    
    Migration:
        # Old (deprecated)
        from workflow_composer.agents.tool_memory import get_tool_memory
        memory = get_tool_memory()
        
        # New (recommended)  
        from workflow_composer.agents.rag import RAGOrchestrator
        rag = RAGOrchestrator()
        rag.record_execution(query, tool, args, success)
        enhancement = rag.enhance(query, tools, args)

This module provides memory-augmented tool selection by learning from past
successful tool executions. It uses similarity search to find relevant
historical interactions and boost confidence in tool selection.

Key Components:
    - ToolMemory: Stores and retrieves successful tool executions
    - ToolExecutionRecord: Individual execution record
    - ToolMemoryIndex: Embedding-based similarity search

Example:
    >>> memory = ToolMemory()
    >>> # Record successful execution
    >>> memory.record_success(
    ...     query="Find RNA-seq data for liver cancer",
    ...     tool_used="search_datasets",
    ...     result={"datasets": [...]},
    ...     duration_ms=150
    ... )
    >>> # Later, for similar query
    >>> similar = memory.find_similar("RNA-seq liver tumors")
    >>> # Returns past successful executions for similar queries
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
import threading

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class ToolMemoryConfig:
    """Configuration for tool memory."""
    
    # Maximum number of records to store
    max_records: int = 10000
    
    # Minimum similarity threshold for retrieval
    similarity_threshold: float = 0.75
    
    # Number of similar records to retrieve
    top_k: int = 5
    
    # Weight decay for older records (per day)
    time_decay_factor: float = 0.95
    
    # Minimum success rate to consider a tool for boosting
    min_success_rate: float = 0.7
    
    # Boost factor for confidence when similar queries succeeded
    confidence_boost: float = 0.15
    
    # Path to persist memory (None for in-memory only)
    persistence_path: Optional[str] = None
    
    # Embedding model for similarity
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    
    # Whether to use embeddings (False = keyword matching only)
    use_embeddings: bool = True


# -----------------------------------------------------------------------------
# Execution Record
# -----------------------------------------------------------------------------

@dataclass
class ToolExecutionRecord:
    """Record of a tool execution."""
    
    record_id: str
    query: str
    query_normalized: str
    tool_name: str
    tool_args: Dict[str, Any]
    success: bool
    result_summary: str
    duration_ms: float
    timestamp: datetime
    user_feedback: Optional[str] = None  # positive, negative, None
    embedding: Optional[List[float]] = None
    
    @classmethod
    def create(
        cls,
        query: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        success: bool,
        result_summary: str,
        duration_ms: float,
        embedding: Optional[List[float]] = None,
    ) -> "ToolExecutionRecord":
        """Create a new execution record."""
        normalized = cls._normalize_query(query)
        record_id = hashlib.md5(
            f"{normalized}:{tool_name}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        return cls(
            record_id=record_id,
            query=query,
            query_normalized=normalized,
            tool_name=tool_name,
            tool_args=tool_args,
            success=success,
            result_summary=result_summary,
            duration_ms=duration_ms,
            timestamp=datetime.now(),
            embedding=embedding,
        )
    
    @staticmethod
    def _normalize_query(query: str) -> str:
        """Normalize query for comparison."""
        # Lowercase, strip, remove extra whitespace
        normalized = " ".join(query.lower().strip().split())
        return normalized
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "record_id": self.record_id,
            "query": self.query,
            "query_normalized": self.query_normalized,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "success": self.success,
            "result_summary": self.result_summary,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "user_feedback": self.user_feedback,
            # Skip embedding for JSON (too large)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolExecutionRecord":
        """Create from dictionary."""
        return cls(
            record_id=data["record_id"],
            query=data["query"],
            query_normalized=data["query_normalized"],
            tool_name=data["tool_name"],
            tool_args=data["tool_args"],
            success=data["success"],
            result_summary=data["result_summary"],
            duration_ms=data["duration_ms"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            user_feedback=data.get("user_feedback"),
            embedding=None,  # Recompute if needed
        )


# -----------------------------------------------------------------------------
# Tool Statistics
# -----------------------------------------------------------------------------

@dataclass
class ToolStats:
    """Statistics for a single tool."""
    
    tool_name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_duration_ms: float = 0
    positive_feedback: int = 0
    negative_feedback: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions
    
    @property
    def avg_duration_ms(self) -> float:
        """Calculate average duration."""
        if self.total_executions == 0:
            return 0.0
        return self.total_duration_ms / self.total_executions
    
    @property
    def feedback_score(self) -> float:
        """Calculate feedback score (-1 to 1)."""
        total_feedback = self.positive_feedback + self.negative_feedback
        if total_feedback == 0:
            return 0.0
        return (self.positive_feedback - self.negative_feedback) / total_feedback
    
    def record_execution(self, record: ToolExecutionRecord) -> None:
        """Update stats with a new execution."""
        self.total_executions += 1
        self.total_duration_ms += record.duration_ms
        
        if record.success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1
        
        if record.user_feedback == "positive":
            self.positive_feedback += 1
        elif record.user_feedback == "negative":
            self.negative_feedback += 1


# -----------------------------------------------------------------------------
# Similarity Index
# -----------------------------------------------------------------------------

class ToolMemoryIndex:
    """Index for similarity-based retrieval of tool executions."""
    
    def __init__(self, config: ToolMemoryConfig):
        self.config = config
        self._embedder = None
        self._embeddings_cache: Dict[str, List[float]] = {}
    
    def _get_embedder(self):
        """Lazy load embedding model."""
        if self._embedder is None and self.config.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.config.embedding_model)
                logger.info(f"Loaded embedding model: {self.config.embedding_model}")
            except ImportError:
                logger.warning("sentence-transformers not available, using keyword matching")
                self.config.use_embeddings = False
        return self._embedder
    
    def compute_embedding(self, text: str) -> Optional[List[float]]:
        """Compute embedding for text."""
        if not self.config.use_embeddings:
            return None
        
        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._embeddings_cache:
            return self._embeddings_cache[cache_key]
        
        embedder = self._get_embedder()
        if embedder is None:
            return None
        
        try:
            embedding = embedder.encode(text).tolist()
            self._embeddings_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            logger.warning(f"Failed to compute embedding: {e}")
            return None
    
    def compute_similarity(
        self,
        query_embedding: List[float],
        record_embedding: List[float],
    ) -> float:
        """Compute cosine similarity between embeddings."""
        if not query_embedding or not record_embedding:
            return 0.0
        
        try:
            import numpy as np
            q = np.array(query_embedding)
            r = np.array(record_embedding)
            
            norm_q = np.linalg.norm(q)
            norm_r = np.linalg.norm(r)
            
            if norm_q == 0 or norm_r == 0:
                return 0.0
            
            return float(np.dot(q, r) / (norm_q * norm_r))
        except Exception:
            return 0.0
    
    def keyword_similarity(self, query1: str, query2: str) -> float:
        """Compute keyword-based similarity (Jaccard)."""
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0


# -----------------------------------------------------------------------------
# Tool Memory
# -----------------------------------------------------------------------------

class ToolMemory:
    """Memory system for tool execution history.
    
    Stores successful tool executions and uses similarity search to
    find relevant past interactions. This enables:
    - Learning from successful queries
    - Boosting confidence for similar tool selections
    - Tracking tool performance over time
    
    Example:
        >>> memory = ToolMemory()
        >>> memory.record_success(
        ...     query="Find ChIP-seq data for H3K4me3",
        ...     tool_name="search_datasets",
        ...     tool_args={"query": "ChIP-seq H3K4me3"},
        ...     result_summary="Found 15 datasets",
        ...     duration_ms=120
        ... )
        >>> # Find similar past executions
        >>> similar = memory.find_similar("ChIP-seq H3K4me3 enhancers")
        >>> for record, score in similar:
        ...     print(f"{record.tool_name}: {score:.2f}")
    """
    
    _instance: Optional["ToolMemory"] = None
    _lock = threading.Lock()
    
    def __init__(self, config: Optional[ToolMemoryConfig] = None):
        self.config = config or ToolMemoryConfig()
        self._records: List[ToolExecutionRecord] = []
        self._tool_stats: Dict[str, ToolStats] = {}
        self._index = ToolMemoryIndex(self.config)
        self._records_lock = threading.Lock()
        
        # Load persisted data if path provided
        if self.config.persistence_path:
            self._load()
    
    @classmethod
    def get_instance(cls, config: Optional[ToolMemoryConfig] = None) -> "ToolMemory":
        """Get singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = ToolMemory(config)
            return cls._instance
    
    def record_execution(
        self,
        query: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        success: bool,
        result_summary: str,
        duration_ms: float,
    ) -> ToolExecutionRecord:
        """Record a tool execution.
        
        Args:
            query: The user query that triggered the tool
            tool_name: Name of the tool executed
            tool_args: Arguments passed to the tool
            success: Whether execution was successful
            result_summary: Brief summary of the result
            duration_ms: Execution duration in milliseconds
            
        Returns:
            The created execution record
        """
        # Compute embedding if enabled
        embedding = self._index.compute_embedding(query) if success else None
        
        record = ToolExecutionRecord.create(
            query=query,
            tool_name=tool_name,
            tool_args=tool_args,
            success=success,
            result_summary=result_summary,
            duration_ms=duration_ms,
            embedding=embedding,
        )
        
        with self._records_lock:
            self._records.append(record)
            
            # Update tool stats
            if tool_name not in self._tool_stats:
                self._tool_stats[tool_name] = ToolStats(tool_name=tool_name)
            self._tool_stats[tool_name].record_execution(record)
            
            # Enforce max records
            self._evict_if_needed()
        
        # Persist if enabled
        if self.config.persistence_path:
            self._save()
        
        return record
    
    def record_success(
        self,
        query: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        result_summary: str,
        duration_ms: float,
    ) -> ToolExecutionRecord:
        """Convenience method to record a successful execution."""
        return self.record_execution(
            query=query,
            tool_name=tool_name,
            tool_args=tool_args,
            success=True,
            result_summary=result_summary,
            duration_ms=duration_ms,
        )
    
    def record_failure(
        self,
        query: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        error_message: str,
        duration_ms: float,
    ) -> ToolExecutionRecord:
        """Convenience method to record a failed execution."""
        return self.record_execution(
            query=query,
            tool_name=tool_name,
            tool_args=tool_args,
            success=False,
            result_summary=f"Error: {error_message}",
            duration_ms=duration_ms,
        )
    
    def add_feedback(self, record_id: str, feedback: str) -> bool:
        """Add user feedback to a record.
        
        Args:
            record_id: The record ID
            feedback: "positive" or "negative"
            
        Returns:
            True if record was found and updated
        """
        if feedback not in ("positive", "negative"):
            raise ValueError("Feedback must be 'positive' or 'negative'")
        
        with self._records_lock:
            for record in self._records:
                if record.record_id == record_id:
                    record.user_feedback = feedback
                    
                    # Update tool stats
                    stats = self._tool_stats.get(record.tool_name)
                    if stats:
                        if feedback == "positive":
                            stats.positive_feedback += 1
                        else:
                            stats.negative_feedback += 1
                    
                    return True
        return False
    
    def find_similar(
        self,
        query: str,
        tool_filter: Optional[str] = None,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> List[Tuple[ToolExecutionRecord, float]]:
        """Find similar past executions.
        
        Args:
            query: The query to match against
            tool_filter: Optional tool name to filter by
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (record, similarity_score) tuples, sorted by score
        """
        top_k = top_k or self.config.top_k
        min_similarity = min_similarity or self.config.similarity_threshold
        
        # Compute query embedding
        query_embedding = self._index.compute_embedding(query)
        query_normalized = " ".join(query.lower().strip().split())
        
        results: List[Tuple[ToolExecutionRecord, float]] = []
        
        with self._records_lock:
            for record in self._records:
                # Skip failures
                if not record.success:
                    continue
                
                # Apply tool filter
                if tool_filter and record.tool_name != tool_filter:
                    continue
                
                # Compute similarity
                if query_embedding and record.embedding:
                    score = self._index.compute_similarity(
                        query_embedding, record.embedding
                    )
                else:
                    # Fall back to keyword similarity
                    score = self._index.keyword_similarity(
                        query_normalized, record.query_normalized
                    )
                
                # Apply time decay
                age_days = (datetime.now() - record.timestamp).days
                decay = self.config.time_decay_factor ** age_days
                score *= decay
                
                # Apply feedback boost/penalty
                if record.user_feedback == "positive":
                    score *= 1.1
                elif record.user_feedback == "negative":
                    score *= 0.8
                
                if score >= min_similarity:
                    results.append((record, score))
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_tool_suggestion(
        self,
        query: str,
        candidate_tools: List[str],
    ) -> Dict[str, float]:
        """Get tool confidence boosts based on similar past queries.
        
        Args:
            query: The user query
            candidate_tools: List of candidate tool names
            
        Returns:
            Dictionary mapping tool name to confidence boost (0.0 to 0.15)
        """
        boosts: Dict[str, float] = {tool: 0.0 for tool in candidate_tools}
        
        # Find similar past executions
        similar = self.find_similar(query, top_k=10, min_similarity=0.6)
        
        if not similar:
            return boosts
        
        # Calculate boosts based on similar queries
        tool_scores: Dict[str, List[float]] = {tool: [] for tool in candidate_tools}
        
        for record, similarity in similar:
            if record.tool_name in tool_scores:
                # Weight by similarity and success
                tool_scores[record.tool_name].append(similarity)
        
        # Convert to boosts
        for tool, scores in tool_scores.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                # Check tool success rate
                stats = self._tool_stats.get(tool)
                if stats and stats.success_rate >= self.config.min_success_rate:
                    boosts[tool] = min(
                        avg_score * self.config.confidence_boost,
                        self.config.confidence_boost
                    )
        
        return boosts
    
    def get_tool_stats(self, tool_name: str) -> Optional[ToolStats]:
        """Get statistics for a tool."""
        return self._tool_stats.get(tool_name)
    
    def get_all_stats(self) -> Dict[str, ToolStats]:
        """Get statistics for all tools."""
        return dict(self._tool_stats)
    
    def get_record_count(self) -> int:
        """Get total number of records."""
        return len(self._records)
    
    def get_recent_records(
        self,
        limit: int = 10,
        tool_filter: Optional[str] = None,
    ) -> List[ToolExecutionRecord]:
        """Get recent execution records.
        
        Args:
            limit: Maximum number of records
            tool_filter: Optional tool name filter
            
        Returns:
            List of recent records
        """
        with self._records_lock:
            records = self._records
            if tool_filter:
                records = [r for r in records if r.tool_name == tool_filter]
            return list(reversed(records[-limit:]))
    
    def clear(self) -> None:
        """Clear all records."""
        with self._records_lock:
            self._records.clear()
            self._tool_stats.clear()
        
        if self.config.persistence_path:
            try:
                Path(self.config.persistence_path).unlink(missing_ok=True)
            except Exception:
                pass
    
    def _evict_if_needed(self) -> None:
        """Evict old records if over limit."""
        while len(self._records) > self.config.max_records:
            # Remove oldest record
            old_record = self._records.pop(0)
            
            # Update stats (decrement)
            stats = self._tool_stats.get(old_record.tool_name)
            if stats:
                stats.total_executions -= 1
                stats.total_duration_ms -= old_record.duration_ms
                if old_record.success:
                    stats.successful_executions -= 1
                else:
                    stats.failed_executions -= 1
    
    def _save(self) -> None:
        """Save records to disk."""
        if not self.config.persistence_path:
            return
        
        try:
            path = Path(self.config.persistence_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "records": [r.to_dict() for r in self._records[-1000:]],  # Save last 1000
                "stats": {
                    name: {
                        "tool_name": s.tool_name,
                        "total_executions": s.total_executions,
                        "successful_executions": s.successful_executions,
                        "failed_executions": s.failed_executions,
                        "total_duration_ms": s.total_duration_ms,
                        "positive_feedback": s.positive_feedback,
                        "negative_feedback": s.negative_feedback,
                    }
                    for name, s in self._tool_stats.items()
                },
            }
            
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save tool memory: {e}")
    
    def _load(self) -> None:
        """Load records from disk."""
        if not self.config.persistence_path:
            return
        
        try:
            path = Path(self.config.persistence_path)
            if not path.exists():
                return
            
            with open(path) as f:
                data = json.load(f)
            
            # Load records
            for record_data in data.get("records", []):
                record = ToolExecutionRecord.from_dict(record_data)
                self._records.append(record)
            
            # Load stats
            for name, stats_data in data.get("stats", {}).items():
                self._tool_stats[name] = ToolStats(
                    tool_name=stats_data["tool_name"],
                    total_executions=stats_data["total_executions"],
                    successful_executions=stats_data["successful_executions"],
                    failed_executions=stats_data["failed_executions"],
                    total_duration_ms=stats_data["total_duration_ms"],
                    positive_feedback=stats_data.get("positive_feedback", 0),
                    negative_feedback=stats_data.get("negative_feedback", 0),
                )
            
            logger.info(f"Loaded {len(self._records)} tool memory records")
            
        except Exception as e:
            logger.warning(f"Failed to load tool memory: {e}")


# -----------------------------------------------------------------------------
# RAG-Enhanced Tool Selection
# -----------------------------------------------------------------------------

@dataclass
class ToolBoost:
    """Represents a confidence boost for a tool."""
    tool_name: str
    boost: float
    reason: str
    similar_queries: List[str] = field(default_factory=list)


@dataclass
class RAGSelectorConfig:
    """Configuration for RAG-based tool selection."""
    
    # Similarity threshold for considering past executions
    similarity_threshold: float = 0.6
    
    # Maximum boost to apply
    max_boost: float = 0.25
    
    # Minimum success rate to consider tool
    min_success_rate: float = 0.5
    
    # Number of similar queries to consider
    top_k: int = 5
    
    # Weight for recency (more recent = higher weight)
    recency_weight: float = 0.8


class RAGToolSelector:
    """RAG-enhanced tool selection using past execution history.
    
    Uses similarity search over past queries to boost tools that
    have been successful for similar requests.
    
    Example:
        >>> selector = RAGToolSelector(tool_memory)
        >>> boosts = selector.get_boosts(
        ...     "search for methylation data",
        ...     ["search_databases", "search_tcga", "explain_concept"]
        ... )
        >>> # boosts = [ToolBoost("search_databases", 0.15, "Similar to past query...")]
    """
    
    def __init__(
        self,
        tool_memory: ToolMemory,
        config: Optional[RAGSelectorConfig] = None,
    ):
        self.tool_memory = tool_memory
        self.config = config or RAGSelectorConfig()
    
    def get_boosts(
        self,
        query: str,
        candidate_tools: List[str],
    ) -> List[ToolBoost]:
        """Get confidence boosts for candidate tools.
        
        Args:
            query: The user query
            candidate_tools: List of candidate tool names
            
        Returns:
            List of ToolBoost objects for tools that should be boosted
        """
        boosts = []
        
        # Find similar past queries
        similar = self.tool_memory.find_similar(
            query=query,
            top_k=self.config.top_k,
            min_similarity=self.config.similarity_threshold,
        )
        
        if not similar:
            return boosts
        
        # Group by tool
        tool_hits: Dict[str, List[Tuple[ToolExecutionRecord, float]]] = {}
        for record, score in similar:
            if record.tool_name in candidate_tools:
                if record.tool_name not in tool_hits:
                    tool_hits[record.tool_name] = []
                tool_hits[record.tool_name].append((record, score))
        
        # Calculate boosts
        for tool_name, hits in tool_hits.items():
            # Check tool success rate
            stats = self.tool_memory.get_tool_stats(tool_name)
            if stats and stats.success_rate < self.config.min_success_rate:
                continue
            
            # Calculate boost from similarity scores
            scores = [score for _, score in hits]
            avg_score = sum(scores) / len(scores)
            
            # Apply recency weighting
            recent_boost = 0.0
            for record, score in hits:
                age_days = (datetime.now() - record.timestamp).days
                recency = self.config.recency_weight ** (age_days / 7)
                recent_boost += score * recency
            recent_boost /= len(hits)
            
            # Final boost calculation
            boost = min(
                (avg_score * 0.6 + recent_boost * 0.4) * self.config.max_boost,
                self.config.max_boost
            )
            
            # Get sample similar queries
            similar_queries = [r.query[:50] for r, _ in hits[:3]]
            
            boosts.append(ToolBoost(
                tool_name=tool_name,
                boost=boost,
                reason=f"Similar to {len(hits)} past successful queries",
                similar_queries=similar_queries,
            ))
        
        # Sort by boost amount
        boosts.sort(key=lambda b: b.boost, reverse=True)
        return boosts
    
    def apply_boosts(
        self,
        tool_confidences: Dict[str, float],
        boosts: List[ToolBoost],
    ) -> Dict[str, float]:
        """Apply boosts to tool confidence scores.
        
        Args:
            tool_confidences: Original tool -> confidence mapping
            boosts: List of boosts to apply
            
        Returns:
            Updated confidence mapping
        """
        result = tool_confidences.copy()
        
        for boost in boosts:
            if boost.tool_name in result:
                result[boost.tool_name] = min(
                    1.0,
                    result[boost.tool_name] + boost.boost
                )
        
        return result
    
    def select_best_tool(
        self,
        query: str,
        tool_confidences: Dict[str, float],
    ) -> Tuple[str, float, List[ToolBoost]]:
        """Select the best tool with RAG enhancement.
        
        Args:
            query: The user query
            tool_confidences: Original tool confidence scores
            
        Returns:
            Tuple of (best_tool_name, confidence, applied_boosts)
        """
        candidate_tools = list(tool_confidences.keys())
        boosts = self.get_boosts(query, candidate_tools)
        
        boosted_confidences = self.apply_boosts(tool_confidences, boosts)
        
        best_tool = max(boosted_confidences.keys(), key=lambda t: boosted_confidences[t])
        
        return best_tool, boosted_confidences[best_tool], boosts


# -----------------------------------------------------------------------------
# Integration Decorator
# -----------------------------------------------------------------------------

def with_tool_memory(
    memory: Optional[ToolMemory] = None,
    result_summarizer: Optional[Callable[[Any], str]] = None,
):
    """Decorator to automatically record tool executions.
    
    Args:
        memory: ToolMemory instance (uses singleton if None)
        result_summarizer: Function to summarize results for storage
        
    Example:
        >>> @with_tool_memory()
        ... async def search_datasets(query: str) -> List[DatasetInfo]:
        ...     ...
    """
    def decorator(func: Callable):
        import functools
        import asyncio
        import time
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            mem = memory or ToolMemory.get_instance()
            start = time.perf_counter()
            
            # Extract query from args/kwargs
            query = kwargs.get("query") or (args[0] if args else "unknown")
            tool_name = func.__name__
            tool_args = kwargs.copy()
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                
                # Summarize result
                if result_summarizer:
                    summary = result_summarizer(result)
                elif isinstance(result, list):
                    summary = f"Returned {len(result)} items"
                elif isinstance(result, dict):
                    summary = f"Returned dict with {len(result)} keys"
                else:
                    summary = str(result)[:100]
                
                mem.record_success(
                    query=str(query),
                    tool_name=tool_name,
                    tool_args=tool_args,
                    result_summary=summary,
                    duration_ms=duration_ms,
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                mem.record_failure(
                    query=str(query),
                    tool_name=tool_name,
                    tool_args=tool_args,
                    error_message=str(e),
                    duration_ms=duration_ms,
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            mem = memory or ToolMemory.get_instance()
            start = time.perf_counter()
            
            query = kwargs.get("query") or (args[0] if args else "unknown")
            tool_name = func.__name__
            tool_args = kwargs.copy()
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                
                if result_summarizer:
                    summary = result_summarizer(result)
                elif isinstance(result, list):
                    summary = f"Returned {len(result)} items"
                elif isinstance(result, dict):
                    summary = f"Returned dict with {len(result)} keys"
                else:
                    summary = str(result)[:100]
                
                mem.record_success(
                    query=str(query),
                    tool_name=tool_name,
                    tool_args=tool_args,
                    result_summary=summary,
                    duration_ms=duration_ms,
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                mem.record_failure(
                    query=str(query),
                    tool_name=tool_name,
                    tool_args=tool_args,
                    error_message=str(e),
                    duration_ms=duration_ms,
                )
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# -----------------------------------------------------------------------------
# Singleton Accessors (DEPRECATED - delegates to new RAG system)
# -----------------------------------------------------------------------------

_tool_memory: Optional[ToolMemory] = None
_rag_selector: Optional[RAGToolSelector] = None
_rag_orchestrator = None  # New system


def get_tool_memory(
    config: Optional[ToolMemoryConfig] = None,
    reset: bool = False
) -> ToolMemory:
    """Get the singleton ToolMemory instance.
    
    .. deprecated:: 2.0
        Use :func:`workflow_composer.agents.rag.RAGOrchestrator` instead.
    
    Args:
        config: Optional configuration (only used on first call or reset)
        reset: If True, create a new instance
        
    Returns:
        ToolMemory singleton instance
    """
    import warnings
    warnings.warn(
        "get_tool_memory() is deprecated. Use RAGOrchestrator from "
        "workflow_composer.agents.rag instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    global _tool_memory
    
    if _tool_memory is None or reset:
        _tool_memory = ToolMemory(config or ToolMemoryConfig())
    
    return _tool_memory


def get_rag_selector(
    config: Optional[RAGSelectorConfig] = None,
    reset: bool = False
) -> RAGToolSelector:
    """Get the singleton RAGToolSelector instance.
    
    .. deprecated:: 2.0
        Use :func:`workflow_composer.agents.rag.RAGOrchestrator` instead.
    
    Args:
        config: Optional configuration (only used on first call or reset)
        reset: If True, create a new instance
        
    Returns:
        RAGToolSelector singleton instance
    """
    import warnings
    warnings.warn(
        "get_rag_selector() is deprecated. Use RAGOrchestrator from "
        "workflow_composer.agents.rag instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    global _rag_selector
    
    if _rag_selector is None or reset:
        tool_memory = get_tool_memory()
        _rag_selector = RAGToolSelector(
            tool_memory=tool_memory,
            config=config or RAGSelectorConfig()
        )
    
    return _rag_selector


def get_rag_orchestrator():
    """Get the new RAGOrchestrator (recommended over deprecated functions).
    
    This is the recommended way to access RAG functionality.
    
    Returns:
        RAGOrchestrator instance from the new rag module
        
    Example:
        >>> rag = get_rag_orchestrator()
        >>> rag.record_execution(query, tool, args, success=True)
        >>> enhancement = rag.enhance(query, tools, args)
    """
    global _rag_orchestrator
    
    if _rag_orchestrator is None:
        from .rag import RAGOrchestrator
        _rag_orchestrator = RAGOrchestrator()
    
    return _rag_orchestrator


# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------

__all__ = [
    # Legacy (deprecated)
    "ToolMemory",
    "ToolMemoryConfig",
    "ToolExecutionRecord",
    "ToolStats",
    "ToolMemoryIndex",
    "RAGToolSelector",
    "RAGSelectorConfig",
    "ToolBoost",
    "with_tool_memory",
    "get_tool_memory",      # Deprecated
    "get_rag_selector",     # Deprecated
    # New (recommended)
    "get_rag_orchestrator", # Use this instead
]
