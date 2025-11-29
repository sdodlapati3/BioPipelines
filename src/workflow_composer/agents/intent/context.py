"""
Unified Conversation Context Management
========================================

Professional-grade context and memory system for the BioPipelines AI agent.

Architecture:
- Short-term Memory (ContextMemory): Last N conversation turns, working memory
- Long-term Memory (AgentMemory): Persistent semantic search across sessions
- Entity Tracking: Track organisms, datasets, assays across conversation
- Coreference Resolution: Resolve "it", "that data", "the workflow"
- State Management: Current workflow, jobs, search results

Features:
- Entity tracking across turns with salience decay
- Coreference resolution for pronouns and definite references
- Conversation summarization for LLM context windows
- Working memory for active tasks and slot filling
- Integration with persistent AgentMemory for cross-session learning

Usage:
    from workflow_composer.agents.intent import ConversationContext
    
    context = ConversationContext(session_id="session_001")
    
    # Add conversation turns
    context.add_turn("user", "search for human brain methylation data", entities=[...])
    context.add_turn("assistant", "Found 10 datasets...")
    
    # Resolve references
    resolved, refs = context.resolve_references("download it")
    # resolved = "download GSE12345"
    
    # Get context for LLM
    messages = context.get_messages_for_llm(limit=10)
    summary = context.get_context_summary()
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import deque
from enum import Enum
import re

# Local imports
from .parser import Entity, EntityType, IntentType

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Salience decay rate per minute
SALIENCE_DECAY_RATE = 0.1

# Maximum entities to track per type
MAX_ENTITIES_PER_TYPE = 50

# Default context window for LLM
DEFAULT_CONTEXT_WINDOW = 20


# =============================================================================
# MEMORY STRUCTURES
# =============================================================================

@dataclass
class MemoryItem:
    """A single item in conversation memory."""
    content: str
    timestamp: datetime
    role: str  # "user", "assistant"
    entities: List[Entity] = field(default_factory=list)
    intent: Optional[IntentType] = None
    salience: float = 1.0  # How important/recent this item is
    tool_used: Optional[str] = None  # Tool that was called
    success: Optional[bool] = None  # Whether the action succeeded
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def age_seconds(self) -> float:
        """Get age in seconds."""
        return (datetime.now() - self.timestamp).total_seconds()
    
    @property
    def age_minutes(self) -> float:
        """Get age in minutes."""
        return self.age_seconds / 60
    
    @property
    def current_salience(self) -> float:
        """Calculate current salience with time decay."""
        decay = self.age_minutes * SALIENCE_DECAY_RATE
        return max(0.1, self.salience - decay)  # Never go below 0.1


@dataclass
class EntityReference:
    """A reference to an entity with context."""
    entity: Entity
    first_mentioned: datetime
    last_mentioned: datetime
    mention_count: int = 1
    aliases: Set[str] = field(default_factory=set)
    
    def update(self, new_entity: Entity):
        """Update reference with new mention."""
        self.last_mentioned = datetime.now()
        self.mention_count += 1
        if new_entity.value != self.entity.value:
            self.aliases.add(new_entity.value)


class ReferentType(Enum):
    """Types of referential expressions."""
    PRONOUN_IT = "it"           # "it", "that", "this"
    PRONOUN_THEY = "they"       # "they", "them", "those"
    DEFINITE_NP = "definite"    # "the data", "the workflow"
    DEMONSTRATIVE = "demo"      # "this data", "that file"
    ELLIPSIS = "ellipsis"       # Omitted reference


# =============================================================================
# COREFERENCE RESOLVER
# =============================================================================

class CoreferenceResolver:
    """
    Resolve pronouns and definite references to their antecedents.
    
    Handles:
    - "it" → most recent singular entity
    - "them" → most recent plural/collection
    - "the data" → most recent data-related entity
    - "this workflow" → current workflow in context
    """
    
    # Patterns for referential expressions
    REFERENT_PATTERNS = {
        # Pronouns
        ReferentType.PRONOUN_IT: [
            r'\bit\b', r'\bthis\b(?!\s+\w)', r'\bthat\b(?!\s+\w)',
        ],
        ReferentType.PRONOUN_THEY: [
            r'\bthey\b', r'\bthem\b', r'\bthose\b', r'\bthese\b(?!\s+\w)',
        ],
        # Definite NPs (the + noun)
        ReferentType.DEFINITE_NP: [
            r'\bthe\s+(data|dataset|file|sample|workflow|pipeline|job|results?|output)\b',
        ],
        # Demonstrative NPs (this/that + noun)
        ReferentType.DEMONSTRATIVE: [
            r'\b(this|that)\s+(data|dataset|file|sample|workflow|pipeline|job)\b',
        ],
    }
    
    # Entity type affinity for resolution
    NOUN_ENTITY_MAP = {
        "data": [EntityType.DATASET_ID, EntityType.FILE_PATH, EntityType.DIRECTORY_PATH],
        "dataset": [EntityType.DATASET_ID, EntityType.PROJECT_ID],
        "file": [EntityType.FILE_PATH],
        "sample": [EntityType.SAMPLE_ID],
        "workflow": [EntityType.WORKFLOW_TYPE],
        "pipeline": [EntityType.WORKFLOW_TYPE],
        "job": [EntityType.JOB_ID],
        "results": [EntityType.FILE_PATH, EntityType.DIRECTORY_PATH],
        "output": [EntityType.FILE_PATH, EntityType.DIRECTORY_PATH],
    }
    
    def __init__(self, context: "ConversationContext"):
        self.context = context
        import re
        self._compiled_patterns = {
            rtype: [re.compile(p, re.IGNORECASE) for p in patterns]
            for rtype, patterns in self.REFERENT_PATTERNS.items()
        }
    
    def resolve(self, message: str) -> Dict[str, Entity]:
        """
        Resolve referential expressions in message.
        
        Returns:
            Dict mapping surface forms to resolved entities
        """
        resolutions = {}
        
        for rtype, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(message):
                    surface = match.group(0)
                    
                    if rtype == ReferentType.PRONOUN_IT:
                        resolved = self._resolve_pronoun_it()
                    elif rtype == ReferentType.PRONOUN_THEY:
                        resolved = self._resolve_pronoun_they()
                    elif rtype == ReferentType.DEFINITE_NP:
                        noun = match.group(1).lower()
                        resolved = self._resolve_definite_np(noun)
                    elif rtype == ReferentType.DEMONSTRATIVE:
                        noun = match.group(2).lower()
                        resolved = self._resolve_definite_np(noun)
                    else:
                        resolved = None
                    
                    if resolved:
                        resolutions[surface] = resolved
        
        return resolutions
    
    def _resolve_pronoun_it(self) -> Optional[Entity]:
        """Resolve 'it' to most salient singular entity."""
        # Priority order for 'it' resolution
        priority_types = [
            EntityType.DATASET_ID,
            EntityType.WORKFLOW_TYPE,
            EntityType.JOB_ID,
            EntityType.FILE_PATH,
            EntityType.DIRECTORY_PATH,
        ]
        
        for etype in priority_types:
            entity = self.context.get_recent_entity(etype)
            if entity:
                return entity
        
        # Fall back to any recent entity
        return self.context.get_most_recent_entity()
    
    def _resolve_pronoun_they(self) -> Optional[Entity]:
        """Resolve 'they/them' to most recent collection."""
        # Look for plural entities (samples, files)
        for etype in [EntityType.SAMPLE_ID, EntityType.FILE_PATH]:
            entities = self.context.get_recent_entities(etype, limit=5)
            if len(entities) > 1:
                # Return first as representative
                return entities[0]
        return None
    
    def _resolve_definite_np(self, noun: str) -> Optional[Entity]:
        """Resolve 'the X' to most recent entity of type X."""
        target_types = self.NOUN_ENTITY_MAP.get(noun, [])
        
        for etype in target_types:
            entity = self.context.get_recent_entity(etype)
            if entity:
                return entity
        
        return None


# =============================================================================
# ENTITY TRACKER
# =============================================================================

class EntityTracker:
    """
    Track entities across conversation turns.
    
    Maintains:
    - All mentioned entities with recency
    - Active entities (likely to be referenced)
    - Entity coreference chains
    """
    
    def __init__(self, max_entities: int = 100):
        self.max_entities = max_entities
        self._entities: Dict[EntityType, List[EntityReference]] = {}
        self._all_entities: List[EntityReference] = []
    
    def add(self, entity: Entity):
        """Add or update an entity reference."""
        # Check if entity already exists
        existing = self._find_existing(entity)
        
        if existing:
            existing.update(entity)
        else:
            ref = EntityReference(
                entity=entity,
                first_mentioned=datetime.now(),
                last_mentioned=datetime.now(),
            )
            
            # Add to type-specific list
            if entity.type not in self._entities:
                self._entities[entity.type] = []
            self._entities[entity.type].append(ref)
            
            # Add to global list
            self._all_entities.append(ref)
            
            # Prune if too many
            if len(self._all_entities) > self.max_entities:
                self._prune_oldest()
    
    def _find_existing(self, entity: Entity) -> Optional[EntityReference]:
        """Find existing reference for an entity."""
        if entity.type not in self._entities:
            return None
        
        for ref in self._entities[entity.type]:
            if ref.entity.canonical == entity.canonical:
                return ref
            if entity.value in ref.aliases:
                return ref
        
        return None
    
    def _prune_oldest(self):
        """Remove oldest entities when over limit."""
        # Sort by last mentioned
        self._all_entities.sort(key=lambda r: r.last_mentioned)
        
        # Remove oldest 20%
        prune_count = len(self._all_entities) // 5
        to_remove = self._all_entities[:prune_count]
        self._all_entities = self._all_entities[prune_count:]
        
        # Remove from type-specific lists
        for ref in to_remove:
            if ref.entity.type in self._entities:
                try:
                    self._entities[ref.entity.type].remove(ref)
                except ValueError:
                    pass
    
    def get_recent(self, entity_type: EntityType, limit: int = 5) -> List[Entity]:
        """Get recent entities of a specific type."""
        if entity_type not in self._entities:
            return []
        
        refs = sorted(
            self._entities[entity_type],
            key=lambda r: r.last_mentioned,
            reverse=True
        )
        
        return [r.entity for r in refs[:limit]]
    
    def get_most_recent(self, entity_type: EntityType = None) -> Optional[Entity]:
        """Get the most recently mentioned entity."""
        if entity_type:
            entities = self.get_recent(entity_type, limit=1)
            return entities[0] if entities else None
        
        if not self._all_entities:
            return None
        
        most_recent = max(self._all_entities, key=lambda r: r.last_mentioned)
        return most_recent.entity
    
    def get_salient_entities(self, limit: int = 5) -> List[Entity]:
        """Get most salient (recent + frequently mentioned) entities."""
        if not self._all_entities:
            return []
        
        # Score by recency and frequency
        now = datetime.now()
        scored = []
        for ref in self._all_entities:
            age = (now - ref.last_mentioned).total_seconds()
            recency_score = 1.0 / (1.0 + age / 60.0)  # Decay over minutes
            frequency_score = min(ref.mention_count / 5.0, 1.0)
            score = 0.7 * recency_score + 0.3 * frequency_score
            scored.append((score, ref))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ref.entity for score, ref in scored[:limit]]


# =============================================================================
# CONTEXT MEMORY
# =============================================================================

class ContextMemory:
    """
    Working memory for the conversation.
    
    Tracks:
    - Recent messages (sliding window)
    - Active task state
    - Important facts/constraints
    """
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self._messages: deque = deque(maxlen=window_size)
        self._facts: Dict[str, Any] = {}
        self._active_task: Optional[Dict] = None
    
    def add_message(
        self, 
        role: str, 
        content: str, 
        entities: List[Entity] = None,
        intent: IntentType = None
    ):
        """Add a message to memory."""
        item = MemoryItem(
            content=content,
            timestamp=datetime.now(),
            role=role,
            entities=entities or [],
            intent=intent,
        )
        self._messages.append(item)
    
    def get_recent_messages(self, limit: int = 10) -> List[MemoryItem]:
        """Get recent messages."""
        messages = list(self._messages)[-limit:]
        return messages
    
    def get_context_for_llm(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get messages formatted for LLM."""
        return [
            {"role": m.role, "content": m.content}
            for m in self.get_recent_messages(limit)
        ]
    
    def set_fact(self, key: str, value: Any):
        """Store a fact in working memory."""
        self._facts[key] = value
    
    def get_fact(self, key: str) -> Optional[Any]:
        """Retrieve a fact from working memory."""
        return self._facts.get(key)
    
    def set_active_task(self, task: Dict):
        """Set the current active task."""
        self._active_task = task
    
    def get_active_task(self) -> Optional[Dict]:
        """Get the current active task."""
        return self._active_task
    
    def clear_active_task(self):
        """Clear the active task."""
        self._active_task = None
    
    def summarize(self) -> str:
        """Generate a summary of current context."""
        parts = []
        
        # Recent topic
        if self._messages:
            recent = self._messages[-1]
            if recent.intent:
                parts.append(f"Recent: {recent.intent.name}")
        
        # Active task
        if self._active_task:
            task_type = self._active_task.get("type", "unknown")
            parts.append(f"Task: {task_type}")
        
        # Key facts
        for key in ["data_path", "workflow", "organism"]:
            if key in self._facts:
                parts.append(f"{key}: {self._facts[key]}")
        
        return " | ".join(parts) if parts else "No context"


# =============================================================================
# CONVERSATION CONTEXT
# =============================================================================

class ConversationContext:
    """
    Complete conversation context manager - the unified context system.
    
    This is the single source of truth for conversation state, integrating:
    - Short-term memory (ContextMemory): Recent conversation turns
    - Entity tracking (EntityTracker): Organisms, datasets, assays
    - Coreference resolution: "it", "that data", "the workflow"
    - Application state: Current workflow, jobs, search results
    - Long-term memory integration (AgentMemory): Cross-session learning
    
    Architecture:
        ConversationContext
            ├── ContextMemory (short-term: last N turns)
            ├── EntityTracker (entities with salience)
            ├── CoreferenceResolver (pronoun resolution)
            ├── State (current workflow, jobs, search results)
            └── AgentMemory (optional: long-term persistence)
    """
    
    def __init__(
        self, 
        session_id: str = "default",
        enable_long_term_memory: bool = False,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
    ):
        """
        Initialize conversation context.
        
        Args:
            session_id: Unique identifier for this conversation session
            enable_long_term_memory: Whether to persist to AgentMemory
            context_window: Number of turns to keep in short-term memory
        """
        self.session_id = session_id
        self.created_at = datetime.now()
        self._turn_count = 0
        
        # Core components
        self.memory = ContextMemory(window_size=context_window)
        self.entity_tracker = EntityTracker()
        self.coreference_resolver = CoreferenceResolver(self)
        
        # Long-term memory (lazy loaded if enabled)
        self._long_term_memory = None
        self._enable_long_term = enable_long_term_memory
        
        # Application state - tracks current workflow, jobs, search results
        self.state: Dict[str, Any] = {
            # Data context
            "data_path": None,
            "samples": [],
            "last_scan_time": None,
            
            # Workflow context  
            "current_workflow": None,
            "workflow_path": None,
            "workflow_type": None,
            
            # Job context
            "jobs": {},
            "last_job_id": None,
            
            # Search context
            "last_search_results": None,
            "last_search_ids": [],
            "last_search_query": None,
            
            # Entities extracted from recent turns
            "last_entities": [],
            
            # Multi-turn task tracking
            "pending_confirmation": None,
            "active_task": None,
            "task_slots": {},  # Slot filling for incomplete requests
        }
        
        logger.debug(f"ConversationContext initialized: {session_id}")
    
    @property
    def long_term_memory(self):
        """Lazy load long-term memory."""
        if self._long_term_memory is None and self._enable_long_term:
            try:
                from ..memory import AgentMemory
                self._long_term_memory = AgentMemory()
                logger.info("Long-term AgentMemory connected")
            except Exception as e:
                logger.warning(f"Failed to initialize long-term memory: {e}")
        return self._long_term_memory
    
    @property
    def turn_count(self) -> int:
        """Get number of conversation turns."""
        return self._turn_count
    
    def add_turn(
        self, 
        role: str, 
        content: str,
        entities: List[Entity] = None,
        intent: Union[IntentType, str] = None,
        tool_used: str = None,
        success: bool = None,
        metadata: Dict[str, Any] = None,
    ):
        """
        Add a conversation turn with full context tracking.
        
        Args:
            role: "user" or "assistant"
            content: The message content
            entities: Extracted entities (organisms, datasets, etc.)
            intent: The detected intent (IntentType or string)
            tool_used: Name of tool that was called (for assistant turns)
            success: Whether the action succeeded
            metadata: Additional metadata to store
        """
        self._turn_count += 1
        
        # Normalize intent to string if needed
        intent_str = intent.name if isinstance(intent, IntentType) else intent
        
        # Create enhanced memory item
        item = MemoryItem(
            content=content,
            timestamp=datetime.now(),
            role=role,
            entities=entities or [],
            intent=intent if isinstance(intent, IntentType) else None,
            tool_used=tool_used,
            success=success,
            metadata=metadata or {},
        )
        self.memory._messages.append(item)
        
        # Track entities with type adaptation
        if entities:
            self._track_entities(entities)
            # Store in state for quick access
            self.state["last_entities"] = [
                {"type": getattr(e, 'entity_type', getattr(e, 'type', 'UNKNOWN')),
                 "value": getattr(e, 'text', getattr(e, 'value', str(e)))}
                for e in entities
            ]
        
        # Persist to long-term memory for learning
        if self.long_term_memory and role == "user":
            self._persist_to_long_term(content, intent_str, entities)
        
        logger.debug(f"Turn {self._turn_count}: {role} - {content[:50]}...")
    
    def _track_entities(self, entities: List[Any]):
        """Track entities with type adaptation for different entity classes."""
        for entity in entities:
            try:
                # Handle both Entity (from parser) and BioEntity (from semantic)
                if hasattr(entity, 'type') and isinstance(entity.type, EntityType):
                    # Standard Entity from parser.py
                    self.entity_tracker.add(entity)
                elif hasattr(entity, 'entity_type'):
                    # BioEntity from semantic.py - adapt to Entity format
                    adapted = Entity(
                        type=self._map_entity_type(entity.entity_type),
                        value=getattr(entity, 'text', str(entity)),
                        canonical=getattr(entity, 'canonical', None),
                        confidence=getattr(entity, 'confidence', 0.8),
                    )
                    self.entity_tracker.add(adapted)
            except Exception as e:
                logger.debug(f"Could not track entity {entity}: {e}")
    
    def _map_entity_type(self, type_str: str) -> EntityType:
        """Map string entity type to EntityType enum."""
        mapping = {
            "ORGANISM": EntityType.ORGANISM,
            "TISSUE": EntityType.TISSUE,
            "CELL_LINE": EntityType.CELL_LINE,
            "ASSAY_TYPE": EntityType.ASSAY_TYPE,
            "DISEASE": EntityType.DISEASE,
            "DATASET_ID": EntityType.DATASET_ID,
            "FILE_PATH": EntityType.FILE_PATH,
            "GENE": EntityType.GENE,
        }
        return mapping.get(type_str, EntityType.OTHER)
    
    def _persist_to_long_term(
        self, 
        content: str, 
        intent: str, 
        entities: List[Any]
    ):
        """Persist important information to long-term memory."""
        try:
            entity_strs = [
                getattr(e, 'text', getattr(e, 'value', str(e)))
                for e in (entities or [])
            ]
            metadata = {
                "session_id": self.session_id,
                "intent": intent,
                "entities": entity_strs,
            }
            self.long_term_memory.add(
                content=content,
                memory_type="conversation",
                metadata=metadata,
            )
        except Exception as e:
            logger.debug(f"Failed to persist to long-term memory: {e}")
    
    def resolve_references(self, message: str) -> Tuple[str, Dict[str, Entity]]:
        """
        Resolve coreferences in a message.
        
        Handles:
        - "it" → most recent singular entity
        - "them" → most recent collection
        - "the data" → most recent data-related entity
        - "all" / "everything" → last search results
        
        Returns:
            Tuple of (resolved_message, resolution_map)
        """
        resolutions = self.coreference_resolver.resolve(message)
        
        # Handle special cases
        message_lower = message.lower()
        
        # "download all" / "get everything" → last search results
        if ("all" in message_lower or "everything" in message_lower) and \
           any(word in message_lower for word in ["download", "get", "fetch"]):
            search_ids = self.state.get("last_search_ids", [])
            if search_ids:
                resolutions["all"] = search_ids
        
        return message, resolutions
    
    def get_recent_entity(self, entity_type: EntityType) -> Optional[Entity]:
        """Get most recent entity of a type."""
        return self.entity_tracker.get_most_recent(entity_type)
    
    def get_recent_entities(self, entity_type: EntityType, limit: int = 5) -> List[Entity]:
        """Get recent entities of a type."""
        return self.entity_tracker.get_recent(entity_type, limit)
    
    def get_most_recent_entity(self) -> Optional[Entity]:
        """Get the most recently mentioned entity of any type."""
        return self.entity_tracker.get_most_recent()
    
    def get_salient_entities(self, limit: int = 5) -> List[Entity]:
        """Get most salient entities for context."""
        return self.entity_tracker.get_salient_entities(limit)
    
    def update_state(self, key: str, value: Any):
        """Update application state."""
        self.state[key] = value
        self.memory.set_fact(key, value)
    
    def get_state(self, key: str) -> Optional[Any]:
        """Get application state value."""
        return self.state.get(key)
    
    def get_context_summary(self) -> str:
        """Get a summary of current context for LLM."""
        parts = []
        
        # Data info
        if self.state.get("data_path"):
            samples = self.state.get("samples", [])
            parts.append(f"📁 Data: {self.state['data_path']} ({len(samples)} samples)")
        
        # Workflow info
        if self.state.get("current_workflow"):
            parts.append(f"📋 Workflow: {self.state['current_workflow']}")
        
        # Active jobs
        jobs = self.state.get("jobs", {})
        running = len([j for j in jobs.values() if j.get("status") == "running"])
        if running:
            parts.append(f"🔄 Jobs: {running} running")
        
        # Salient entities
        entities = self.get_salient_entities(3)
        if entities:
            entity_strs = [f"{e.type.name}:{e.canonical}" for e in entities]
            parts.append(f"🏷️ Context: {', '.join(entity_strs)}")
        
        # Memory summary
        memory_summary = self.memory.summarize()
        if memory_summary and memory_summary != "No context":
            parts.append(f"💭 {memory_summary}")
        
        return " | ".join(parts) if parts else "Ready - no data loaded"
    
    def get_messages_for_llm(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get recent messages formatted for LLM."""
        return self.memory.get_context_for_llm(limit)
    
    def set_pending_confirmation(self, action: str, data: Dict):
        """Set a pending action requiring confirmation."""
        self.state["pending_confirmation"] = {
            "action": action,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }
    
    def get_pending_confirmation(self) -> Optional[Dict]:
        """Get pending confirmation if any."""
        return self.state.get("pending_confirmation")
    
    def clear_pending_confirmation(self):
        """Clear pending confirmation."""
        self.state["pending_confirmation"] = None
    
    def clear(self):
        """Clear all context."""
        self._turn_count = 0
        self.memory = ContextMemory(window_size=self.memory.window_size)
        self.entity_tracker = EntityTracker()
        self.state = {
            "data_path": None,
            "samples": [],
            "last_scan_time": None,
            "current_workflow": None,
            "workflow_path": None,
            "workflow_type": None,
            "jobs": {},
            "last_job_id": None,
            "last_search_results": None,
            "last_search_ids": [],
            "last_search_query": None,
            "last_entities": [],
            "pending_confirmation": None,
            "active_task": None,
            "task_slots": {},
        }
        logger.info(f"Context cleared for session {self.session_id}")
    
    # =========================================================================
    # Slot Filling for Multi-Turn Tasks
    # =========================================================================
    
    def start_task(self, task_type: str, required_slots: List[str]):
        """
        Start a multi-turn task that requires slot filling.
        
        Args:
            task_type: Type of task (e.g., "workflow_generation", "data_download")
            required_slots: List of slot names that must be filled
        """
        self.state["active_task"] = {
            "type": task_type,
            "required_slots": required_slots,
            "started_at": datetime.now().isoformat(),
        }
        self.state["task_slots"] = {}
        logger.info(f"Started task: {task_type}, requires: {required_slots}")
    
    def fill_slot(self, slot_name: str, value: Any):
        """Fill a slot for the current task."""
        if self.state.get("active_task"):
            self.state["task_slots"][slot_name] = value
            logger.debug(f"Filled slot: {slot_name} = {value}")
    
    def get_missing_slots(self) -> List[str]:
        """Get list of unfilled required slots."""
        task = self.state.get("active_task")
        if not task:
            return []
        
        required = task.get("required_slots", [])
        filled = set(self.state.get("task_slots", {}).keys())
        return [s for s in required if s not in filled]
    
    def is_task_complete(self) -> bool:
        """Check if all required slots are filled."""
        return len(self.get_missing_slots()) == 0
    
    def complete_task(self) -> Dict[str, Any]:
        """Complete the current task and return filled slots."""
        slots = self.state.get("task_slots", {}).copy()
        self.state["active_task"] = None
        self.state["task_slots"] = {}
        return slots
    
    # =========================================================================
    # Context Export/Import for Persistence
    # =========================================================================
    
    def export_state(self) -> Dict[str, Any]:
        """
        Export context state for persistence.
        
        Returns:
            Dictionary containing all context state
        """
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "turn_count": self._turn_count,
            "state": self.state,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                    "intent": m.intent.name if m.intent else None,
                    "tool_used": m.tool_used,
                    "success": m.success,
                }
                for m in self.memory._messages
            ],
            "entities": [
                {
                    "type": ref.entity.type.name,
                    "value": ref.entity.value,
                    "canonical": ref.entity.canonical,
                    "mention_count": ref.mention_count,
                }
                for ref in self.entity_tracker._all_entities
            ],
        }
    
    def import_state(self, data: Dict[str, Any]):
        """
        Import context state from persistence.
        
        Args:
            data: Dictionary from export_state()
        """
        self.session_id = data.get("session_id", self.session_id)
        self._turn_count = data.get("turn_count", 0)
        self.state = data.get("state", self.state)
        
        # Restore messages (simplified - doesn't restore full MemoryItem)
        for msg in data.get("messages", []):
            self.memory._messages.append(MemoryItem(
                content=msg["content"],
                timestamp=datetime.fromisoformat(msg["timestamp"]),
                role=msg["role"],
            ))
        
        logger.info(f"Imported context for session {self.session_id}")
    
    # =========================================================================
    # Semantic Search in Context
    # =========================================================================
    
    def search_context(self, query: str, limit: int = 5) -> List[MemoryItem]:
        """
        Search conversation history semantically.
        
        Uses long-term memory if available, otherwise simple text matching.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching memory items
        """
        # Try long-term memory first
        if self.long_term_memory:
            try:
                results = self.long_term_memory.search(query, limit=limit)
                # Map back to MemoryItems if possible
                return [
                    MemoryItem(
                        content=r.entry.content,
                        timestamp=datetime.fromisoformat(r.entry.timestamp),
                        role="user",
                    )
                    for r in results
                ]
            except Exception as e:
                logger.debug(f"Long-term search failed: {e}")
        
        # Fallback to simple text matching
        query_lower = query.lower()
        matches = []
        for item in self.memory._messages:
            if query_lower in item.content.lower():
                matches.append(item)
        
        return matches[:limit]
    
    def get_similar_past_interactions(self, query: str, limit: int = 3) -> List[Dict]:
        """
        Find similar past interactions for learning.
        
        Useful for showing "you asked something similar before" or
        for learning from past successful/failed interactions.
        """
        if not self.long_term_memory:
            return []
        
        try:
            results = self.long_term_memory.search(
                query, 
                memory_type="conversation",
                limit=limit
            )
            return [
                {
                    "content": r.entry.content,
                    "similarity": r.score,
                    "metadata": r.entry.metadata,
                }
                for r in results
            ]
        except Exception:
            return []
    
    # =========================================================================
    # Context for LLM Prompting
    # =========================================================================
    
    def get_system_context(self) -> str:
        """
        Get system context string for LLM prompting.
        
        This provides the LLM with awareness of:
        - Current data/workflow state
        - Recent search results
        - Active task slots
        - Salient entities
        """
        parts = []
        
        # Session info
        parts.append(f"Session: {self.session_id} (Turn {self._turn_count})")
        
        # Data context
        if self.state.get("data_path"):
            samples = self.state.get("samples", [])
            parts.append(f"Data loaded: {self.state['data_path']} ({len(samples)} samples)")
        
        # Workflow context
        if self.state.get("current_workflow"):
            parts.append(f"Active workflow: {self.state['current_workflow']}")
        
        # Search context
        if self.state.get("last_search_ids"):
            ids = self.state["last_search_ids"]
            parts.append(f"Last search returned {len(ids)} datasets: {', '.join(ids[:3])}{'...' if len(ids) > 3 else ''}")
        
        # Entity context
        entities = self.get_salient_entities(5)
        if entities:
            entity_strs = [f"{e.type.name.lower()}: {e.canonical or e.value}" for e in entities]
            parts.append(f"Context entities: {', '.join(entity_strs)}")
        
        # Active task
        if self.state.get("active_task"):
            task = self.state["active_task"]
            missing = self.get_missing_slots()
            parts.append(f"Active task: {task['type']} (missing slots: {missing})")
        
        return "\n".join(parts) if parts else "No active context"
    
    def __repr__(self) -> str:
        return f"ConversationContext(session={self.session_id}, turns={self._turn_count})"