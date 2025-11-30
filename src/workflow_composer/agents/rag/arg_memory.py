"""RAG Layer 2: Argument Memory.

Learns optimal arguments for tools based on past successful executions.
This layer provides value even with a small number of tools by optimizing
the parameters passed to each tool.

Example:
    >>> from workflow_composer.agents.rag import ArgumentMemory, get_argument_memory
    >>> 
    >>> arg_memory = get_argument_memory()
    >>> 
    >>> # Get suggested arguments for a query
    >>> suggestion = arg_memory.suggest(
    ...     query="find H3K4me3 ChIP-seq data from human",
    ...     tool_name="search_encode"
    ... )
    >>> print(suggestion.args)
    {'assay': 'ChIP-seq', 'target': 'H3K4me3', 'organism': 'Homo sapiens'}
"""

from __future__ import annotations

import logging
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .memory import ToolExecutionRecord, ToolMemory, get_tool_memory

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ArgumentMemoryConfig:
    """Configuration for argument memory.
    
    Attributes:
        min_occurrences: Minimum times an arg value must appear to be suggested
        similarity_threshold: Query similarity threshold for learning
        max_suggestions: Maximum argument keys to suggest
        confidence_threshold: Minimum confidence to include suggestion
    """
    min_occurrences: int = 3
    similarity_threshold: float = 0.5
    max_suggestions: int = 10
    confidence_threshold: float = 0.3
    
    # Keys to ignore (usually auto-generated or internal)
    ignored_keys: Set[str] = field(default_factory=lambda: {
        "query",  # Usually the raw query
        "limit",  # Pagination
        "offset",
        "page",
        "user_id",
        "session_id",
        "trace_id",
    })
    
    # Keys that are most valuable to learn
    priority_keys: Set[str] = field(default_factory=lambda: {
        "organism",
        "assay",
        "target",
        "tissue",
        "cell_type",
        "disease",
        "assembly",
        "file_format",
        "output_type",
    })


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ArgumentSuggestion:
    """Suggested arguments for a tool execution.
    
    Attributes:
        tool_name: Name of the tool
        args: Suggested argument dictionary
        confidence: Overall confidence score (0-1)
        sources: Number of past executions this was learned from
        arg_confidences: Per-argument confidence scores
    """
    tool_name: str
    args: Dict[str, Any]
    confidence: float
    sources: int
    arg_confidences: Dict[str, float] = field(default_factory=dict)
    
    def merge_with(self, user_args: Dict[str, Any]) -> Dict[str, Any]:
        """Merge suggestions with user-provided args.
        
        User args always take precedence over suggestions.
        """
        result = dict(self.args)
        result.update(user_args)
        return result


@dataclass
class ArgumentPattern:
    """Pattern of arguments learned from successful executions."""
    
    tool_name: str
    query_keywords: Set[str]  # Keywords that trigger this pattern
    arg_counts: Dict[str, Counter]  # arg_key -> Counter of values
    total_occurrences: int = 0
    
    def add_execution(self, record: ToolExecutionRecord) -> None:
        """Add an execution to learn from."""
        self.total_occurrences += 1
        
        for key, value in record.tool_args.items():
            if key not in self.arg_counts:
                self.arg_counts[key] = Counter()
            
            # Convert value to hashable
            if isinstance(value, (list, dict)):
                value = str(value)
            
            self.arg_counts[key][value] += 1
    
    def get_best_values(self, min_ratio: float = 0.3) -> Dict[str, Tuple[Any, float]]:
        """Get the most common value for each argument.
        
        Returns:
            Dict mapping arg_key to (best_value, confidence)
        """
        result = {}
        for key, counter in self.arg_counts.items():
            if not counter:
                continue
            
            best_value, count = counter.most_common(1)[0]
            ratio = count / self.total_occurrences
            
            if ratio >= min_ratio:
                result[key] = (best_value, ratio)
        
        return result


# =============================================================================
# Argument Memory
# =============================================================================

class ArgumentMemory:
    """Memory system for learning optimal tool arguments (RAG Layer 2).
    
    Analyzes successful tool executions to learn common argument patterns
    and suggests optimal parameters for new queries.
    
    Depends on: ToolMemory (Layer 1)
    """
    
    _instance: Optional["ArgumentMemory"] = None
    _lock = threading.Lock()
    
    def __init__(
        self,
        tool_memory: Optional[ToolMemory] = None,
        config: Optional[ArgumentMemoryConfig] = None,
    ):
        self.tool_memory = tool_memory or get_tool_memory()
        self.config = config or ArgumentMemoryConfig()
        self._patterns: Dict[str, List[ArgumentPattern]] = {}  # tool -> patterns
        self._patterns_lock = threading.Lock()
    
    @classmethod
    def get_instance(
        cls,
        tool_memory: Optional[ToolMemory] = None,
        config: Optional[ArgumentMemoryConfig] = None,
    ) -> "ArgumentMemory":
        """Get singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = ArgumentMemory(tool_memory, config)
            return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None
    
    def suggest(
        self,
        query: str,
        tool_name: str,
        current_args: Optional[Dict[str, Any]] = None,
    ) -> ArgumentSuggestion:
        """Suggest optimal arguments for a tool execution.
        
        Args:
            query: The user query
            tool_name: Target tool name
            current_args: Already specified arguments (won't be overwritten)
            
        Returns:
            ArgumentSuggestion with recommended parameters
        """
        current_args = current_args or {}
        
        # Find similar successful executions
        similar = self.tool_memory.find_similar(
            query=query,
            tool_filter=tool_name,
            success_only=True,
            limit=20,
        )
        
        if not similar:
            return ArgumentSuggestion(
                tool_name=tool_name,
                args={},
                confidence=0,
                sources=0,
            )
        
        # Aggregate argument values
        arg_values: Dict[str, Counter] = {}
        total_weight = 0
        
        for record, similarity in similar:
            weight = similarity
            total_weight += weight
            
            for key, value in record.tool_args.items():
                # Skip ignored keys
                if key in self.config.ignored_keys:
                    continue
                
                # Skip keys already provided by user
                if key in current_args:
                    continue
                
                if key not in arg_values:
                    arg_values[key] = Counter()
                
                # Convert to hashable
                if isinstance(value, (list, dict)):
                    value = str(value)
                
                arg_values[key][value] += weight
        
        if total_weight == 0:
            return ArgumentSuggestion(
                tool_name=tool_name,
                args={},
                confidence=0,
                sources=0,
            )
        
        # Select best values
        suggested_args = {}
        arg_confidences = {}
        
        for key, counter in arg_values.items():
            if not counter:
                continue
            
            # Get best value
            best_value, weighted_count = counter.most_common(1)[0]
            confidence = weighted_count / total_weight
            
            # Apply priority boost
            if key in self.config.priority_keys:
                confidence *= 1.2  # 20% boost for important keys
            
            if confidence >= self.config.confidence_threshold:
                suggested_args[key] = best_value
                arg_confidences[key] = min(confidence, 1.0)
        
        # Limit suggestions
        if len(suggested_args) > self.config.max_suggestions:
            # Keep highest confidence
            sorted_keys = sorted(
                suggested_args.keys(),
                key=lambda k: arg_confidences[k],
                reverse=True
            )[:self.config.max_suggestions]
            
            suggested_args = {k: suggested_args[k] for k in sorted_keys}
            arg_confidences = {k: arg_confidences[k] for k in sorted_keys}
        
        # Calculate overall confidence
        overall_confidence = (
            sum(arg_confidences.values()) / len(arg_confidences)
            if arg_confidences else 0
        )
        
        return ArgumentSuggestion(
            tool_name=tool_name,
            args=suggested_args,
            confidence=overall_confidence,
            sources=len(similar),
            arg_confidences=arg_confidences,
        )
    
    def learn_from_execution(
        self,
        query: str,
        tool_name: str,
        args: Dict[str, Any],
        success: bool,
    ) -> None:
        """Learn from a new execution.
        
        Called automatically by ToolMemory when recording executions.
        Only learns from successful executions.
        """
        if not success:
            return
        
        # Extract query keywords
        keywords = set(query.lower().split())
        
        with self._patterns_lock:
            if tool_name not in self._patterns:
                self._patterns[tool_name] = []
            
            # Find matching pattern or create new
            pattern = self._find_or_create_pattern(tool_name, keywords)
            
            # Create a mock record for the pattern
            record = ToolExecutionRecord.create(
                query=query,
                tool_name=tool_name,
                tool_args=args,
                success=True,
                duration_ms=0,
            )
            pattern.add_execution(record)
    
    def _find_or_create_pattern(
        self,
        tool_name: str,
        keywords: Set[str],
    ) -> ArgumentPattern:
        """Find existing pattern or create new one."""
        patterns = self._patterns.get(tool_name, [])
        
        # Find best matching pattern
        best_pattern = None
        best_overlap = 0
        
        for pattern in patterns:
            overlap = len(keywords & pattern.query_keywords)
            if overlap > best_overlap:
                best_pattern = pattern
                best_overlap = overlap
        
        # Create new if no good match
        if best_pattern is None or best_overlap < 2:
            best_pattern = ArgumentPattern(
                tool_name=tool_name,
                query_keywords=keywords,
                arg_counts={},
            )
            self._patterns[tool_name].append(best_pattern)
        else:
            # Merge keywords
            best_pattern.query_keywords |= keywords
        
        return best_pattern
    
    def get_common_args(self, tool_name: str) -> Dict[str, List[Tuple[Any, int]]]:
        """Get commonly used argument values for a tool.
        
        Useful for building UI dropdowns or suggestions.
        
        Returns:
            Dict mapping arg_key to list of (value, count) tuples
        """
        result: Dict[str, Counter] = {}
        
        # Get from patterns
        with self._patterns_lock:
            patterns = self._patterns.get(tool_name, [])
            for pattern in patterns:
                for key, counter in pattern.arg_counts.items():
                    if key not in result:
                        result[key] = Counter()
                    result[key] += counter
        
        # Also check recent records
        records = self.tool_memory.get_recent_records(
            limit=100,
            tool_filter=tool_name,
            success_only=True,
        )
        
        for record in records:
            for key, value in record.tool_args.items():
                if key in self.config.ignored_keys:
                    continue
                if key not in result:
                    result[key] = Counter()
                if isinstance(value, (list, dict)):
                    value = str(value)
                result[key][value] += 1
        
        # Convert to sorted list
        return {
            key: counter.most_common(10)
            for key, counter in result.items()
            if counter
        }


# =============================================================================
# Singleton Accessor
# =============================================================================

_argument_memory: Optional[ArgumentMemory] = None
_argument_memory_lock = threading.Lock()


def get_argument_memory(
    tool_memory: Optional[ToolMemory] = None,
    config: Optional[ArgumentMemoryConfig] = None,
    reset: bool = False,
) -> ArgumentMemory:
    """Get the singleton ArgumentMemory instance.
    
    Args:
        tool_memory: ToolMemory to use (uses singleton if None)
        config: Optional configuration
        reset: If True, create new instance
        
    Returns:
        ArgumentMemory singleton instance
    """
    global _argument_memory
    
    with _argument_memory_lock:
        if _argument_memory is None or reset:
            _argument_memory = ArgumentMemory(tool_memory, config)
        return _argument_memory
