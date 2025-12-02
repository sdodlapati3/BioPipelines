"""
Session Memory & Context Persistence
=====================================

Professional-grade session memory that persists across the entire conversation.

Based on patterns from:
- Rasa: Conversation patterns, slot persistence, context carry-over
- Dialogflow: Session parameters, context lifespan
- ChatGPT: Conversation memory, entity recall

Key Features:
1. Session-wide persistence: Paths, preferences, entities remembered ALL session
2. Smart recall: "the data", "that path" resolves to last used path
3. Preference learning: Remembers user's organism, assay preferences
4. Cross-query context: Results from query 1 available in query 2
5. Explicit memory commands: "remember this path", "forget about X"

Architecture:
    SessionMemory
    ├── PersistentState: Long-lived session data (paths, preferences)
    ├── WorkingMemory: Current task context
    ├── EntityMemory: All mentioned entities with decay
    └── ActionHistory: What was done and results

Usage:
    memory = SessionMemory()
    
    # Auto-persist paths
    memory.remember_path("/scratch/data/methylation")
    
    # Later in conversation
    path = memory.get_remembered_path("data")  # Returns the methylation path
    
    # Preferences
    memory.set_preference("organism", "human")
    memory.get_preference("organism")  # Returns "human"
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum, auto
import json
import re

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class MemoryType(Enum):
    """Types of memory storage."""
    PATH = auto()           # File/directory paths
    DATASET = auto()        # Dataset IDs (GSE, ENCSR)
    PREFERENCE = auto()     # User preferences
    ENTITY = auto()         # Extracted entities
    RESULT = auto()         # Query results
    ACTION = auto()         # Completed actions


class MemoryPriority(Enum):
    """Priority levels for memory retention."""
    CRITICAL = 5     # Never forget (explicit user request)
    HIGH = 4         # Important (frequently used)
    NORMAL = 3       # Standard retention
    LOW = 2          # Can be forgotten if needed
    EPHEMERAL = 1    # Short-lived


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MemoryEntry:
    """A single memory entry."""
    key: str
    value: Any
    memory_type: MemoryType
    priority: MemoryPriority = MemoryPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 1
    context: Dict[str, Any] = field(default_factory=dict)
    aliases: Set[str] = field(default_factory=set)
    
    def access(self):
        """Record an access to this memory."""
        self.last_accessed = datetime.now()
        self.access_count += 1
        
        # Promote priority based on usage
        if self.access_count >= 5 and self.priority.value < MemoryPriority.HIGH.value:
            self.priority = MemoryPriority.HIGH
    
    def age_seconds(self) -> float:
        """Get age in seconds since creation."""
        return (datetime.now() - self.created_at).total_seconds()
    
    def staleness_seconds(self) -> float:
        """Get time since last access."""
        return (datetime.now() - self.last_accessed).total_seconds()


@dataclass
class ActionRecord:
    """Record of a completed action."""
    action_type: str  # "scan", "search", "download", "workflow", etc.
    query: str
    tool_used: str
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)
    parameters: Dict[str, Any] = field(default_factory=dict)
    result_summary: Optional[str] = None
    entities_involved: List[str] = field(default_factory=list)


# =============================================================================
# SESSION MEMORY
# =============================================================================

class SessionMemory:
    """
    Session-wide memory that persists across all queries in a conversation.
    
    This solves the "context loss" problem where the agent forgets paths,
    datasets, and preferences between queries.
    
    Design Principles:
    1. Everything mentioned is remembered (with appropriate decay)
    2. Frequently accessed items get higher priority
    3. Explicit "remember" commands set highest priority
    4. Smart resolution for "the data", "that path", etc.
    """
    
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.created_at = datetime.now()
        
        # Core storage
        self._memories: Dict[str, MemoryEntry] = {}
        self._action_history: List[ActionRecord] = []
        
        # Type-specific indices for fast lookup
        self._paths: Dict[str, MemoryEntry] = {}
        self._datasets: Dict[str, MemoryEntry] = {}
        self._preferences: Dict[str, Any] = {}
        self._entities: Dict[str, MemoryEntry] = {}
        
        # Aliases for resolution
        self._alias_map: Dict[str, str] = {}  # alias -> canonical key
        
        # Last used items (for "it", "that", etc.)
        self._last_path: Optional[str] = None
        self._last_dataset: Optional[str] = None
        self._last_result: Optional[Any] = None
        self._last_query: Optional[str] = None
        
        logger.info(f"SessionMemory initialized: {session_id}")
    
    # =========================================================================
    # PATH MEMORY
    # =========================================================================
    
    def remember_path(
        self, 
        path: str, 
        context: str = None,
        priority: MemoryPriority = MemoryPriority.NORMAL
    ):
        """
        Remember a path for later use.
        
        The path can be recalled by:
        - Full path
        - Last component ("methylation" for "/data/methylation")
        - Context ("data path", "output directory")
        
        Args:
            path: The file/directory path
            context: Optional context (e.g., "data", "output", "reference")
            priority: Memory priority
        """
        path = path.rstrip("/")
        
        # Create memory entry
        entry = MemoryEntry(
            key=path,
            value=path,
            memory_type=MemoryType.PATH,
            priority=priority,
            context={"type": context} if context else {},
        )
        
        # Add aliases
        parts = path.split("/")
        if parts:
            last_part = parts[-1]
            entry.aliases.add(last_part)
            entry.aliases.add(last_part.lower())
            
            # Add context-based aliases
            if context:
                entry.aliases.add(f"{context}_path")
                entry.aliases.add(f"{context} path")
        
        # Store
        self._memories[path] = entry
        self._paths[path] = entry
        self._last_path = path
        
        # Update alias map
        for alias in entry.aliases:
            self._alias_map[alias.lower()] = path
        
        logger.debug(f"Remembered path: {path} (aliases: {entry.aliases})")
    
    def get_remembered_path(self, hint: str = None) -> Optional[str]:
        """
        Get a remembered path.
        
        Args:
            hint: Optional hint to find specific path
                  - None: returns last used path
                  - "methylation": finds path containing "methylation"
                  - "data": finds path with "data" context
                  
        Returns:
            The matching path or None
        """
        if hint is None:
            # Return last used path
            if self._last_path and self._last_path in self._paths:
                self._paths[self._last_path].access()
                return self._last_path
            return None
        
        hint_lower = hint.lower()
        
        # Check alias map first
        if hint_lower in self._alias_map:
            key = self._alias_map[hint_lower]
            if key in self._paths:
                self._paths[key].access()
                self._last_path = key
                return key
        
        # Search by substring match
        for path, entry in self._paths.items():
            if hint_lower in path.lower():
                entry.access()
                self._last_path = path
                return path
            if hint_lower in entry.aliases or hint_lower in str(entry.context):
                entry.access()
                self._last_path = path
                return path
        
        return None
    
    def get_all_paths(self) -> List[str]:
        """Get all remembered paths."""
        return list(self._paths.keys())
    
    # =========================================================================
    # DATASET MEMORY
    # =========================================================================
    
    def remember_dataset(
        self, 
        dataset_id: str, 
        source: str = None,
        metadata: Dict = None
    ):
        """
        Remember a dataset ID.
        
        Args:
            dataset_id: The dataset ID (GSE12345, ENCSR000AAA, etc.)
            source: Source database (GEO, ENCODE, etc.)
            metadata: Additional metadata (organism, assay, etc.)
        """
        entry = MemoryEntry(
            key=dataset_id,
            value=dataset_id,
            memory_type=MemoryType.DATASET,
            context={
                "source": source,
                **(metadata or {}),
            },
        )
        
        self._memories[dataset_id] = entry
        self._datasets[dataset_id] = entry
        self._last_dataset = dataset_id
        
        # Add lowercase alias
        self._alias_map[dataset_id.lower()] = dataset_id
        
        logger.debug(f"Remembered dataset: {dataset_id}")
    
    def remember_search_results(self, results: List[Dict], query: str):
        """
        Remember results from a search query.
        
        Args:
            results: List of search results with 'id', 'title', etc.
            query: The search query
        """
        self._last_query = query
        self._last_result = results
        
        # Remember each dataset
        for result in results[:20]:  # Limit to top 20
            dataset_id = result.get("id") or result.get("accession")
            if dataset_id:
                self.remember_dataset(
                    dataset_id,
                    source=result.get("source", "unknown"),
                    metadata={
                        "title": result.get("title"),
                        "organism": result.get("organism"),
                        "assay": result.get("assay_type"),
                    }
                )
    
    def get_remembered_dataset(self, hint: str = None) -> Optional[str]:
        """Get a remembered dataset ID."""
        if hint is None:
            return self._last_dataset
        
        hint_lower = hint.lower()
        
        if hint_lower in self._alias_map:
            return self._alias_map[hint_lower]
        
        for dataset_id in self._datasets:
            if hint_lower in dataset_id.lower():
                return dataset_id
        
        return None
    
    def get_last_search_results(self) -> Tuple[Optional[str], Optional[List]]:
        """Get the last search query and results."""
        return self._last_query, self._last_result
    
    # =========================================================================
    # PREFERENCES
    # =========================================================================
    
    def set_preference(self, key: str, value: Any):
        """
        Set a user preference.
        
        Common preferences:
        - organism: "human", "mouse"
        - assay_type: "RNA-seq", "ChIP-seq"
        - output_format: "nextflow", "snakemake"
        - default_source: "ENCODE", "GEO"
        """
        self._preferences[key] = value
        logger.debug(f"Set preference: {key} = {value}")
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference."""
        return self._preferences.get(key, default)
    
    def get_all_preferences(self) -> Dict[str, Any]:
        """Get all preferences."""
        return dict(self._preferences)
    
    def infer_preferences_from_entities(self, entities: List[Any]):
        """
        Automatically infer preferences from extracted entities.
        
        If user frequently mentions "human", set organism preference.
        """
        for entity in entities:
            etype = getattr(entity, 'type', getattr(entity, 'entity_type', None))
            evalue = getattr(entity, 'canonical', getattr(entity, 'text', None))
            
            if not etype or not evalue:
                continue
            
            etype_str = etype.name if hasattr(etype, 'name') else str(etype)
            
            if etype_str == "ORGANISM" and "organism" not in self._preferences:
                self.set_preference("organism", evalue)
            elif etype_str == "ASSAY_TYPE" and "assay_type" not in self._preferences:
                self.set_preference("assay_type", evalue)
    
    # =========================================================================
    # ACTION HISTORY
    # =========================================================================
    
    def record_action(
        self,
        action_type: str,
        query: str,
        tool_used: str,
        success: bool,
        parameters: Dict = None,
        result_summary: str = None,
        entities: List[str] = None,
    ):
        """
        Record a completed action.
        
        This enables:
        - "What did I search for?"
        - "Retry the last search"
        - "Show my download history"
        """
        record = ActionRecord(
            action_type=action_type,
            query=query,
            tool_used=tool_used,
            success=success,
            parameters=parameters or {},
            result_summary=result_summary,
            entities_involved=entities or [],
        )
        
        self._action_history.append(record)
        
        # Keep only last 50 actions
        if len(self._action_history) > 50:
            self._action_history = self._action_history[-50:]
        
        logger.debug(f"Recorded action: {action_type} ({tool_used})")
    
    def get_last_action(self, action_type: str = None) -> Optional[ActionRecord]:
        """Get the last action, optionally filtered by type."""
        for action in reversed(self._action_history):
            if action_type is None or action.action_type == action_type:
                return action
        return None
    
    def get_action_history(
        self, 
        action_type: str = None, 
        limit: int = 10
    ) -> List[ActionRecord]:
        """Get action history."""
        actions = self._action_history
        if action_type:
            actions = [a for a in actions if a.action_type == action_type]
        return list(reversed(actions))[:limit]
    
    # =========================================================================
    # REFERENCE RESOLUTION
    # =========================================================================
    
    def resolve_reference(self, text: str) -> Optional[str]:
        """
        Resolve references like "that path", "the data", "it".
        
        Args:
            text: The reference text
            
        Returns:
            Resolved value or None
        """
        text_lower = text.lower().strip()
        
        # Direct references to last items
        if text_lower in ("it", "that", "this", "the path", "that path", "this path"):
            return self._last_path
        
        if text_lower in ("the data", "that data", "the dataset", "that dataset"):
            return self._last_dataset or self._last_path
        
        if text_lower in ("the result", "the results", "those results"):
            return str(self._last_result) if self._last_result else None
        
        # Check alias map
        if text_lower in self._alias_map:
            return self._alias_map[text_lower]
        
        # Partial match on paths
        for path in self._paths:
            if text_lower in path.lower():
                return path
        
        return None
    
    def resolve_references_in_query(self, query: str) -> Tuple[str, Dict[str, str]]:
        """
        Resolve all references in a query.
        
        Args:
            query: User's query
            
        Returns:
            Tuple of (resolved_query, resolution_map)
        """
        resolutions = {}
        resolved_query = query
        
        # Patterns for reference detection
        patterns = [
            (r'\b(that|the)\s+path\b', self._last_path),
            (r'\b(that|the)\s+data\b', self._last_path or self._last_dataset),
            (r'\b(that|the)\s+dataset\b', self._last_dataset),
            (r'\bit\b(?!\s+is|\s+was|\s+has)', self._last_path or self._last_dataset),
        ]
        
        for pattern, value in patterns:
            if value:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    resolutions[match.group(0)] = value
                    resolved_query = re.sub(pattern, value, resolved_query, flags=re.IGNORECASE)
        
        return resolved_query, resolutions
    
    # =========================================================================
    # CONTEXT SUMMARY
    # =========================================================================
    
    def get_context_summary(self) -> str:
        """Get a summary of current session memory for LLM context."""
        parts = []
        
        if self._last_path:
            parts.append(f"📁 Current path: {self._last_path}")
        
        if self._last_dataset:
            parts.append(f"🧬 Last dataset: {self._last_dataset}")
        
        if self._preferences:
            prefs = ", ".join(f"{k}={v}" for k, v in self._preferences.items())
            parts.append(f"⚙️ Preferences: {prefs}")
        
        if self._last_query:
            parts.append(f"🔍 Last search: {self._last_query}")
        
        recent_paths = list(self._paths.keys())[-3:]
        if recent_paths:
            parts.append(f"📂 Known paths: {', '.join(recent_paths)}")
        
        return "\n".join(parts) if parts else "No session context yet"
    
    def get_full_state(self) -> Dict[str, Any]:
        """Export full session state."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "paths": list(self._paths.keys()),
            "datasets": list(self._datasets.keys()),
            "preferences": self._preferences,
            "last_path": self._last_path,
            "last_dataset": self._last_dataset,
            "last_query": self._last_query,
            "action_count": len(self._action_history),
        }
    
    # =========================================================================
    # MEMORY MANAGEMENT
    # =========================================================================
    
    def forget(self, key: str):
        """Explicitly forget something."""
        key_lower = key.lower()
        
        # Remove from all stores
        if key in self._memories:
            del self._memories[key]
        if key in self._paths:
            del self._paths[key]
        if key in self._datasets:
            del self._datasets[key]
        if key in self._preferences:
            del self._preferences[key]
        
        # Clean alias map
        to_remove = [alias for alias, target in self._alias_map.items() if target == key]
        for alias in to_remove:
            del self._alias_map[alias]
        
        # Clear last references
        if self._last_path == key:
            self._last_path = None
        if self._last_dataset == key:
            self._last_dataset = None
        
        logger.info(f"Forgot: {key}")
    
    def clear(self):
        """Clear all session memory."""
        self._memories.clear()
        self._paths.clear()
        self._datasets.clear()
        self._preferences.clear()
        self._entities.clear()
        self._alias_map.clear()
        self._action_history.clear()
        self._last_path = None
        self._last_dataset = None
        self._last_result = None
        self._last_query = None
        
        logger.info("Session memory cleared")


# =============================================================================
# SINGLETON
# =============================================================================

_session_memory: Optional[SessionMemory] = None


def get_session_memory(session_id: str = "default") -> SessionMemory:
    """Get the singleton session memory instance."""
    global _session_memory
    if _session_memory is None or _session_memory.session_id != session_id:
        _session_memory = SessionMemory(session_id=session_id)
    return _session_memory
