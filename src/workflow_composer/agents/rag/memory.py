"""RAG Layer 1: Tool Execution Memory.

Records all tool executions for analytics and as the foundation for
RAG Layers 2 (ArgumentMemory) and 3 (RAGToolSelector).

This module can operate in two modes:
- Standalone: In-memory storage with optional JSON persistence
- Database: Uses PostgreSQL/SQLite via ToolExecutionRepository

Example:
    >>> from workflow_composer.agents.rag import ToolMemory, get_tool_memory
    >>> 
    >>> # Get singleton instance
    >>> memory = get_tool_memory()
    >>> 
    >>> # Record execution
    >>> memory.record(
    ...     query="find RNA-seq data for liver",
    ...     tool_name="search_databases",
    ...     tool_args={"query": "RNA-seq liver"},
    ...     success=True,
    ...     duration_ms=150,
    ...     result_summary="Found 25 datasets",
    ... )
    >>> 
    >>> # Get statistics
    >>> stats = memory.get_tool_stats("search_databases")
    >>> print(f"Success rate: {stats.success_rate:.0%}")
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ToolMemoryConfig:
    """Configuration for tool memory.
    
    Attributes:
        max_records: Maximum records to store in memory
        persistence_path: Path for JSON persistence (None = memory only)
        use_database: Use database backend instead of in-memory
        similarity_threshold: Minimum similarity for retrieval
        time_decay_factor: Weight decay per day (0.95 = 5% per day)
    """
    max_records: int = 10000
    persistence_path: Optional[str] = None
    use_database: bool = False
    similarity_threshold: float = 0.6
    time_decay_factor: float = 0.95
    cleanup_days: int = 90


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ToolExecutionRecord:
    """Record of a tool execution."""
    
    id: Optional[int]
    query: str
    query_normalized: str
    tool_name: str
    tool_args: Dict[str, Any]
    success: bool
    duration_ms: float
    result_summary: Optional[str] = None
    error_message: Optional[str] = None
    result_count: Optional[int] = None
    user_feedback: Optional[str] = None
    task_type: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @classmethod
    def create(
        cls,
        query: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        success: bool,
        duration_ms: float,
        result_summary: Optional[str] = None,
        error_message: Optional[str] = None,
        result_count: Optional[int] = None,
        task_type: Optional[str] = None,
    ) -> "ToolExecutionRecord":
        """Create a new execution record."""
        return cls(
            id=None,
            query=query,
            query_normalized=cls.normalize_query(query),
            tool_name=tool_name,
            tool_args=tool_args,
            success=success,
            duration_ms=duration_ms,
            result_summary=result_summary,
            error_message=error_message,
            result_count=result_count,
            task_type=task_type,
        )
    
    @staticmethod
    def normalize_query(query: str) -> str:
        """Normalize query for comparison."""
        return " ".join(query.lower().strip().split())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "query": self.query,
            "query_normalized": self.query_normalized,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "result_summary": self.result_summary,
            "error_message": self.error_message,
            "result_count": self.result_count,
            "user_feedback": self.user_feedback,
            "task_type": self.task_type,
            "timestamp": self.timestamp.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolExecutionRecord":
        """Create from dictionary."""
        return cls(
            id=data.get("id"),
            query=data["query"],
            query_normalized=data.get("query_normalized", cls.normalize_query(data["query"])),
            tool_name=data["tool_name"],
            tool_args=data.get("tool_args", {}),
            success=data["success"],
            duration_ms=data["duration_ms"],
            result_summary=data.get("result_summary"),
            error_message=data.get("error_message"),
            result_count=data.get("result_count"),
            user_feedback=data.get("user_feedback"),
            task_type=data.get("task_type"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.utcnow(),
        )


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
    
    def update(self, record: ToolExecutionRecord) -> None:
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


# =============================================================================
# Tool Memory
# =============================================================================

class ToolMemory:
    """Memory system for tool execution history (RAG Layer 1).
    
    Records all tool executions and provides:
    - Execution history for analytics
    - Foundation for ArgumentMemory (Layer 2)
    - Foundation for RAGToolSelector (Layer 3)
    
    Can operate with in-memory storage or database backend.
    """
    
    _instance: Optional["ToolMemory"] = None
    _lock = threading.Lock()
    
    def __init__(self, config: Optional[ToolMemoryConfig] = None):
        self.config = config or ToolMemoryConfig()
        self._records: List[ToolExecutionRecord] = []
        self._tool_stats: Dict[str, ToolStats] = {}
        self._records_lock = threading.Lock()
        self._db_repo = None
        
        # Load from persistence or database
        if self.config.use_database:
            self._init_database()
        elif self.config.persistence_path:
            self._load_from_file()
    
    def _init_database(self) -> None:
        """Initialize database backend."""
        try:
            from workflow_composer.db import ToolExecutionRepository, get_db
            # Database will be initialized on first use
            logger.info("ToolMemory configured with database backend")
        except ImportError:
            logger.warning("Database not available, falling back to in-memory")
            self.config.use_database = False
    
    @classmethod
    def get_instance(cls, config: Optional[ToolMemoryConfig] = None) -> "ToolMemory":
        """Get singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = ToolMemory(config)
            return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None
    
    def record(
        self,
        query: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        success: bool,
        duration_ms: float,
        result_summary: Optional[str] = None,
        error_message: Optional[str] = None,
        result_count: Optional[int] = None,
        user_id: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> ToolExecutionRecord:
        """Record a tool execution.
        
        Args:
            query: Original user query
            tool_name: Name of executed tool
            tool_args: Arguments passed to tool
            success: Whether execution succeeded
            duration_ms: Execution time in milliseconds
            result_summary: Brief summary of result
            error_message: Error message if failed
            result_count: Number of results returned
            user_id: Optional user identifier
            task_type: Task type classification
            
        Returns:
            Created execution record
        """
        record = ToolExecutionRecord.create(
            query=query,
            tool_name=tool_name,
            tool_args=tool_args,
            success=success,
            duration_ms=duration_ms,
            result_summary=result_summary,
            error_message=error_message,
            result_count=result_count,
            task_type=task_type,
        )
        
        if self.config.use_database:
            self._record_to_database(record, user_id)
        else:
            self._record_to_memory(record)
        
        return record
    
    def _record_to_memory(self, record: ToolExecutionRecord) -> None:
        """Store record in memory."""
        with self._records_lock:
            self._records.append(record)
            record.id = len(self._records)
            
            # Update stats
            if record.tool_name not in self._tool_stats:
                self._tool_stats[record.tool_name] = ToolStats(tool_name=record.tool_name)
            self._tool_stats[record.tool_name].update(record)
            
            # Enforce max records
            while len(self._records) > self.config.max_records:
                self._records.pop(0)
        
        # Persist if configured
        if self.config.persistence_path:
            self._save_to_file()
    
    def _record_to_database(self, record: ToolExecutionRecord, user_id: Optional[str]) -> None:
        """Store record in database."""
        try:
            from uuid import UUID

            from workflow_composer.db import ToolExecutionRepository, get_db
            
            with get_db() as session:
                repo = ToolExecutionRepository(session)
                db_record = repo.record(
                    tool_name=record.tool_name,
                    query=record.query,
                    success=record.success,
                    duration_ms=record.duration_ms,
                    tool_args=record.tool_args,
                    result_summary=record.result_summary,
                    error_message=record.error_message,
                    result_count=record.result_count,
                    user_id=UUID(user_id) if user_id else None,
                    task_type=record.task_type,
                )
                record.id = db_record.id
        except Exception as e:
            logger.error(f"Failed to record to database: {e}")
            # Fallback to memory
            self._record_to_memory(record)
    
    def record_success(
        self,
        query: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        result_summary: str,
        duration_ms: float,
        result_count: Optional[int] = None,
    ) -> ToolExecutionRecord:
        """Convenience method for successful executions."""
        return self.record(
            query=query,
            tool_name=tool_name,
            tool_args=tool_args,
            success=True,
            duration_ms=duration_ms,
            result_summary=result_summary,
            result_count=result_count,
        )
    
    def record_failure(
        self,
        query: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        error_message: str,
        duration_ms: float,
    ) -> ToolExecutionRecord:
        """Convenience method for failed executions."""
        return self.record(
            query=query,
            tool_name=tool_name,
            tool_args=tool_args,
            success=False,
            duration_ms=duration_ms,
            error_message=error_message,
        )
    
    def add_feedback(self, record_id: int, feedback: str) -> bool:
        """Add user feedback to a record.
        
        Args:
            record_id: The record ID
            feedback: "positive" or "negative"
            
        Returns:
            True if record found and updated
        """
        if feedback not in ("positive", "negative"):
            raise ValueError("Feedback must be 'positive' or 'negative'")
        
        if self.config.use_database:
            try:
                from workflow_composer.db import ToolExecutionRepository, get_db
                with get_db() as session:
                    repo = ToolExecutionRepository(session)
                    return repo.add_feedback(record_id, feedback)
            except Exception as e:
                logger.error(f"Failed to add feedback to database: {e}")
                return False
        
        with self._records_lock:
            for record in self._records:
                if record.id == record_id:
                    record.user_feedback = feedback
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
        success_only: bool = True,
        limit: int = 10,
    ) -> List[Tuple[ToolExecutionRecord, float]]:
        """Find similar past executions.
        
        Args:
            query: Query to match against
            tool_filter: Optional tool name filter
            success_only: Only return successful executions
            limit: Maximum results to return
            
        Returns:
            List of (record, similarity_score) tuples
        """
        query_normalized = ToolExecutionRecord.normalize_query(query)
        query_keywords = set(query_normalized.split())
        
        if not query_keywords:
            return []
        
        results: List[Tuple[ToolExecutionRecord, float]] = []
        
        with self._records_lock:
            for record in self._records:
                if success_only and not record.success:
                    continue
                if tool_filter and record.tool_name != tool_filter:
                    continue
                
                # Simple Jaccard similarity
                record_keywords = set(record.query_normalized.split())
                if not record_keywords:
                    continue
                
                intersection = len(query_keywords & record_keywords)
                union = len(query_keywords | record_keywords)
                similarity = intersection / union if union > 0 else 0
                
                # Apply time decay
                age_days = (datetime.utcnow() - record.timestamp).days
                decay = self.config.time_decay_factor ** age_days
                similarity *= decay
                
                # Apply feedback boost
                if record.user_feedback == "positive":
                    similarity *= 1.1
                elif record.user_feedback == "negative":
                    similarity *= 0.8
                
                if similarity >= self.config.similarity_threshold:
                    results.append((record, similarity))
        
        # Sort by similarity and limit
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def get_tool_stats(self, tool_name: str) -> Optional[ToolStats]:
        """Get statistics for a specific tool."""
        if self.config.use_database:
            try:
                from workflow_composer.db import ToolExecutionRepository, get_db
                with get_db() as session:
                    repo = ToolExecutionRepository(session)
                    stats_dict = repo.get_tool_stats(tool_name)
                    return ToolStats(
                        tool_name=tool_name,
                        total_executions=stats_dict["total_executions"],
                        successful_executions=stats_dict["successful_executions"],
                        failed_executions=stats_dict["total_executions"] - stats_dict["successful_executions"],
                        total_duration_ms=stats_dict["avg_duration_ms"] * stats_dict["total_executions"],
                    )
            except Exception as e:
                logger.error(f"Failed to get stats from database: {e}")
        
        return self._tool_stats.get(tool_name)
    
    def get_all_stats(self) -> Dict[str, ToolStats]:
        """Get statistics for all tools."""
        if self.config.use_database:
            try:
                from workflow_composer.db import ToolExecutionRepository, get_db
                with get_db() as session:
                    repo = ToolExecutionRepository(session)
                    stats_dict = repo.get_all_tool_stats()
                    return {
                        name: ToolStats(
                            tool_name=name,
                            total_executions=s["total_executions"],
                            successful_executions=s["successful_executions"],
                            failed_executions=s["total_executions"] - s["successful_executions"],
                            total_duration_ms=s["avg_duration_ms"] * s["total_executions"],
                        )
                        for name, s in stats_dict.items()
                    }
            except Exception as e:
                logger.error(f"Failed to get all stats from database: {e}")
        
        return dict(self._tool_stats)
    
    def get_recent_records(
        self,
        limit: int = 20,
        tool_filter: Optional[str] = None,
        success_only: bool = False,
    ) -> List[ToolExecutionRecord]:
        """Get recent execution records."""
        with self._records_lock:
            records = self._records
            if tool_filter:
                records = [r for r in records if r.tool_name == tool_filter]
            if success_only:
                records = [r for r in records if r.success]
            return list(reversed(records[-limit:]))
    
    def get_record_count(self) -> int:
        """Get total number of records."""
        return len(self._records)
    
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
    
    def _save_to_file(self) -> None:
        """Save records to JSON file."""
        if not self.config.persistence_path:
            return
        
        try:
            path = Path(self.config.persistence_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "records": [r.to_dict() for r in self._records[-1000:]],
            }
            
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save tool memory: {e}")
    
    def _load_from_file(self) -> None:
        """Load records from JSON file."""
        if not self.config.persistence_path:
            return
        
        try:
            path = Path(self.config.persistence_path)
            if not path.exists():
                return
            
            with open(path) as f:
                data = json.load(f)
            
            for record_data in data.get("records", []):
                record = ToolExecutionRecord.from_dict(record_data)
                self._records.append(record)
                
                if record.tool_name not in self._tool_stats:
                    self._tool_stats[record.tool_name] = ToolStats(tool_name=record.tool_name)
                self._tool_stats[record.tool_name].update(record)
            
            logger.info(f"Loaded {len(self._records)} records from {path}")
        except Exception as e:
            logger.warning(f"Failed to load tool memory: {e}")


# =============================================================================
# Singleton Accessor
# =============================================================================

_tool_memory: Optional[ToolMemory] = None
_tool_memory_lock = threading.Lock()


def get_tool_memory(
    config: Optional[ToolMemoryConfig] = None,
    reset: bool = False,
) -> ToolMemory:
    """Get the singleton ToolMemory instance.
    
    Args:
        config: Optional configuration (only used on first call or reset)
        reset: If True, create a new instance
        
    Returns:
        ToolMemory singleton instance
    """
    global _tool_memory
    
    with _tool_memory_lock:
        if _tool_memory is None or reset:
            _tool_memory = ToolMemory(config)
        return _tool_memory
