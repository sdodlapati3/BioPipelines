"""
Memory Module
=============

Provides persistent memory, session management, and preference learning
for the BioPipelines agentic system.

Components:
- AgentMemory: Vector-based semantic memory with embeddings
- SessionManager: Conversation session handling
- UserProfile: User preferences and history
- PreferenceLearner: Learns defaults from user interactions
- TokenTracker: Token budget tracking for context management
- ConciseMemory: Clean slate memory pattern for long workflows

New in v2.0 (DeepCode-inspired):
- Token budget tracking prevents context overflow
- Concise memory reduces token usage by 70-80%
"""

from .memory import (
    AgentMemory,
    MemoryEntry,
    SearchResult,
    EmbeddingModel,
    get_memory,
    remember,
    recall,
)

from .user_profile import (
    UserProfile,
    PersistentProfileStore,
    PreferenceLearner,
    get_profile_store,
    get_preference_learner,
    get_user_profile,
    update_preferences,
)

from .session_manager import (
    Session,
    Message,
    SessionManager,
    get_session_manager,
)

from .token_tracker import (
    TokenTracker,
    TokenBudget,
    TokenUsage,
    TokenizerBackend,
    create_tracker_for_model,
    create_budget_for_task,
    MODEL_CONTEXT_SIZES,
)

from .concise_memory import (
    ConciseMemory,
    ConciseState,
    CompletedStep,
    create_concise_memory,
)

__all__ = [
    # Core memory
    "AgentMemory",
    "MemoryEntry",
    "SearchResult",
    "EmbeddingModel",
    "get_memory",
    "remember",
    "recall",
    # User profiles
    "UserProfile",
    "PersistentProfileStore",
    "PreferenceLearner",
    "get_profile_store",
    "get_preference_learner",
    "get_user_profile",
    "update_preferences",
    # Sessions
    "Session",
    "Message",
    "SessionManager",
    "get_session_manager",
    # Token tracking (new)
    "TokenTracker",
    "TokenBudget",
    "TokenUsage",
    "TokenizerBackend",
    "create_tracker_for_model",
    "create_budget_for_task",
    "MODEL_CONTEXT_SIZES",
    # Concise memory (new)
    "ConciseMemory",
    "ConciseState",
    "CompletedStep",
    "create_concise_memory",
]
