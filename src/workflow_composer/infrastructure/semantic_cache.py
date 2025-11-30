"""
Semantic Cache
==============

Intelligent caching with TTL and semantic similarity matching.

Features:
- Time-based TTL expiration
- LRU eviction when cache is full
- Semantic similarity matching for fuzzy cache hits
- Thread-safe implementation

Usage:
    from workflow_composer.infrastructure.semantic_cache import SemanticCache
    
    cache = SemanticCache(
        max_size=1000,
        default_ttl=3600,  # 1 hour
        similarity_threshold=0.85,
    )
    
    # Set with default TTL
    cache.set("search for H3K27ac ChIP-seq", results)
    
    # Get - exact match
    value, is_exact = cache.get("search for H3K27ac ChIP-seq")
    # value = results, is_exact = True
    
    # Get - semantic match (similar query)
    value, is_exact = cache.get("find H3K27ac ChIP-seq data")
    # value = results, is_exact = False (if similarity > 0.85)
    
    # Get with TTL override
    cache.set("key", value, ttl=300)  # 5 minutes

Design:
    - Exact matches checked first (O(1) via hash)
    - Semantic matches checked if no exact match (O(n) similarity scan)
    - Embeddings computed lazily and cached
    - Thread-safe for concurrent access
"""

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Generic
import json

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Cache Entry
# =============================================================================

@dataclass
class CacheEntry(Generic[T]):
    """A single cache entry with metadata."""
    
    key: str
    """The original key string."""
    
    key_hash: str
    """MD5 hash of the normalized key for fast lookup."""
    
    value: T
    """The cached value."""
    
    created_at: datetime
    """When the entry was created."""
    
    ttl_seconds: int
    """Time-to-live in seconds."""
    
    last_accessed: datetime = field(default_factory=datetime.now)
    """Last access time for LRU."""
    
    hits: int = 0
    """Number of times this entry was accessed."""
    
    embedding: Optional[List[float]] = None
    """Optional embedding for semantic matching."""
    
    @property
    def expires_at(self) -> datetime:
        """When this entry expires."""
        return self.created_at + timedelta(seconds=self.ttl_seconds)
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return datetime.now() > self.expires_at
    
    @property
    def age_seconds(self) -> float:
        """Age of the entry in seconds."""
        return (datetime.now() - self.created_at).total_seconds()
    
    def touch(self) -> None:
        """Update last accessed time and increment hits."""
        self.last_accessed = datetime.now()
        self.hits += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "key_hash": self.key_hash,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "ttl_seconds": self.ttl_seconds,
            "hits": self.hits,
            "is_expired": self.is_expired,
        }


# =============================================================================
# Embedding Model
# =============================================================================

class EmbeddingModel:
    """
    Lazy-loaded embedding model for semantic similarity.
    
    Uses sentence-transformers if available, otherwise falls back to
    simple word overlap similarity.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()
        self._load_attempted = False
        self._load_error = None
    
    def _load_model(self) -> bool:
        """Attempt to load the model."""
        if self._load_attempted:
            return self._model is not None
        
        with self._lock:
            if self._load_attempted:
                return self._model is not None
            
            self._load_attempted = True
            
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
                return True
            except ImportError:
                self._load_error = "sentence-transformers not installed"
                logger.debug("sentence-transformers not available, using fallback")
                return False
            except Exception as e:
                self._load_error = str(e)
                logger.warning(f"Failed to load embedding model: {e}")
                return False
    
    def encode(self, text: str) -> Optional[List[float]]:
        """
        Encode text to embedding vector.
        
        Returns None if model not available.
        """
        if not self._load_model():
            return None
        
        try:
            embedding = self._model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.warning(f"Failed to encode text: {e}")
            return None
    
    def similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Embeddings are assumed to be normalized.
        """
        if len(embedding1) != len(embedding2):
            return 0.0
        
        # Dot product of normalized vectors = cosine similarity
        return sum(a * b for a, b in zip(embedding1, embedding2))
    
    def text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts.
        
        Falls back to word overlap if embeddings not available.
        """
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        
        if emb1 is not None and emb2 is not None:
            return self.similarity(emb1, emb2)
        
        # Fallback: Jaccard similarity on words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0


# =============================================================================
# Semantic Cache
# =============================================================================

class SemanticCache(Generic[T]):
    """
    Cache with TTL expiration and semantic similarity matching.
    
    Features:
    - Exact match via hash lookup (O(1))
    - Semantic match via embedding similarity (O(n))
    - TTL-based expiration
    - LRU eviction when cache is full
    - Thread-safe
    
    Example:
        cache = SemanticCache[List[Dataset]](
            max_size=100,
            default_ttl=3600,
            similarity_threshold=0.85,
        )
        
        cache.set("human ChIP-seq H3K27ac", datasets)
        
        # Exact match
        result, exact = cache.get("human ChIP-seq H3K27ac")
        # result = datasets, exact = True
        
        # Semantic match (similar but not identical)
        result, exact = cache.get("H3K27ac ChIP-seq human")
        # result = datasets, exact = False
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int = 3600,
        similarity_threshold: float = 0.85,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        enable_semantic: bool = True,
    ):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
            similarity_threshold: Minimum similarity for semantic match
            embedding_model: Sentence transformer model name
            enable_semantic: Whether to enable semantic matching
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.similarity_threshold = similarity_threshold
        self.enable_semantic = enable_semantic
        
        self._cache: Dict[str, CacheEntry[T]] = {}
        self._lock = threading.RLock()
        
        # Lazy-loaded embedding model
        self._embedder: Optional[EmbeddingModel] = None
        self._embedding_model_name = embedding_model
        
        # Stats
        self._hits = 0
        self._misses = 0
        self._semantic_hits = 0
        self._evictions = 0
    
    @property
    def embedder(self) -> EmbeddingModel:
        """Get or create the embedding model."""
        if self._embedder is None:
            self._embedder = EmbeddingModel(self._embedding_model_name)
        return self._embedder
    
    def _make_hash(self, key: str) -> str:
        """Create a hash from a key."""
        normalized = key.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get(self, key: str) -> Tuple[Optional[T], bool]:
        """
        Get a value from the cache.
        
        Args:
            key: The cache key
        
        Returns:
            Tuple of (value, is_exact_match)
            - (value, True) if exact match found
            - (value, False) if semantic match found
            - (None, False) if no match
        """
        key_hash = self._make_hash(key)
        
        with self._lock:
            # First: exact match
            if key_hash in self._cache:
                entry = self._cache[key_hash]
                
                if entry.is_expired:
                    del self._cache[key_hash]
                    self._misses += 1
                else:
                    entry.touch()
                    self._hits += 1
                    return entry.value, True
            
            # Second: semantic match (if enabled)
            if self.enable_semantic:
                match = self._find_semantic_match(key)
                if match:
                    match.touch()
                    self._hits += 1
                    self._semantic_hits += 1
                    return match.value, False
            
            self._misses += 1
            return None, False
    
    def set(
        self,
        key: str,
        value: T,
        ttl: int = None,
        compute_embedding: bool = True,
    ) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
            ttl: TTL in seconds (uses default if not provided)
            compute_embedding: Whether to compute embedding for semantic matching
        """
        key_hash = self._make_hash(key)
        ttl = ttl if ttl is not None else self.default_ttl
        
        # Compute embedding if enabled
        embedding = None
        if self.enable_semantic and compute_embedding:
            embedding = self.embedder.encode(key)
        
        entry = CacheEntry(
            key=key,
            key_hash=key_hash,
            value=value,
            created_at=datetime.now(),
            ttl_seconds=ttl,
            embedding=embedding,
        )
        
        with self._lock:
            # Evict if necessary
            self._evict_if_needed()
            
            # Add entry
            self._cache[key_hash] = entry
    
    def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.
        
        Returns True if key was found and deleted.
        """
        key_hash = self._make_hash(key)
        
        with self._lock:
            if key_hash in self._cache:
                del self._cache[key_hash]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
    
    def _find_semantic_match(self, key: str) -> Optional[CacheEntry[T]]:
        """
        Find a semantically similar entry.
        
        Returns the best matching entry above the similarity threshold.
        """
        if not self.enable_semantic:
            return None
        
        # Get embedding for the query
        query_embedding = self.embedder.encode(key)
        if query_embedding is None:
            # Fallback to text similarity
            return self._find_text_similarity_match(key)
        
        best_entry: Optional[CacheEntry[T]] = None
        best_score = self.similarity_threshold
        
        for entry in self._cache.values():
            if entry.is_expired:
                continue
            
            if entry.embedding is not None:
                score = self.embedder.similarity(query_embedding, entry.embedding)
                if score > best_score:
                    best_score = score
                    best_entry = entry
        
        return best_entry
    
    def _find_text_similarity_match(self, key: str) -> Optional[CacheEntry[T]]:
        """
        Find a match using text similarity (fallback).
        """
        best_entry: Optional[CacheEntry[T]] = None
        best_score = self.similarity_threshold
        
        for entry in self._cache.values():
            if entry.is_expired:
                continue
            
            score = self.embedder.text_similarity(key, entry.key)
            if score > best_score:
                best_score = score
                best_entry = entry
        
        return best_entry
    
    def _evict_if_needed(self) -> None:
        """Evict entries if cache is full (called under lock)."""
        # First: remove expired entries
        expired = [k for k, v in self._cache.items() if v.is_expired]
        for k in expired:
            del self._cache[k]
            self._evictions += 1
        
        # Second: LRU eviction if still over size
        while len(self._cache) >= self.max_size:
            # Find LRU entry
            lru_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].last_accessed,
            )
            del self._cache[lru_key]
            self._evictions += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "semantic_hits": self._semantic_hits,
                "evictions": self._evictions,
                "hit_rate": hit_rate,
                "semantic_hit_rate": (
                    self._semantic_hits / self._hits if self._hits > 0 else 0.0
                ),
            }
    
    def get_entries(self) -> List[Dict[str, Any]]:
        """Get information about all entries (for debugging)."""
        with self._lock:
            return [
                entry.to_dict()
                for entry in self._cache.values()
                if not entry.is_expired
            ]
    
    def __len__(self) -> int:
        """Get number of entries."""
        with self._lock:
            return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        key_hash = self._make_hash(key)
        with self._lock:
            if key_hash in self._cache:
                entry = self._cache[key_hash]
                if not entry.is_expired:
                    return True
                del self._cache[key_hash]
            return False


# =============================================================================
# Global Cache Instance
# =============================================================================

_global_caches: Dict[str, SemanticCache] = {}
_cache_lock = threading.Lock()


def get_cache(
    name: str = "default",
    **kwargs,
) -> SemanticCache:
    """
    Get or create a named cache.
    
    Args:
        name: Cache name
        **kwargs: Arguments for SemanticCache constructor
    
    Returns:
        Cache instance
    """
    with _cache_lock:
        if name not in _global_caches:
            _global_caches[name] = SemanticCache(**kwargs)
        return _global_caches[name]


def clear_all_caches() -> None:
    """Clear all global caches."""
    with _cache_lock:
        for cache in _global_caches.values():
            cache.clear()


def get_all_cache_stats() -> Dict[str, Dict[str, Any]]:
    """Get stats for all caches."""
    with _cache_lock:
        return {
            name: cache.get_stats()
            for name, cache in _global_caches.items()
        }


# =============================================================================
# Cached Decorator
# =============================================================================

def cached(
    cache_name: str = "default",
    ttl: int = None,
    key_fn: Optional[callable] = None,
    enable_semantic: bool = False,  # Disabled by default for function caching
    **cache_kwargs,
):
    """
    Decorator to cache function results.
    
    Args:
        cache_name: Name of the cache to use
        ttl: TTL in seconds
        key_fn: Function to generate cache key from args
        enable_semantic: Whether to enable semantic matching (default False)
        **cache_kwargs: Arguments for cache creation
    
    Example:
        @cached("search_cache", ttl=3600)
        def search_databases(query: str) -> List[Dataset]:
            ...
        
        @cached("api_cache", key_fn=lambda q, s: f"{q}:{s}")
        def fetch_data(query: str, source: str) -> Any:
            ...
    """
    import functools
    import asyncio
    
    # Force semantic off by default for function caching
    cache_kwargs.setdefault("enable_semantic", enable_semantic)
    
    def decorator(func):
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache = get_cache(cache_name, **cache_kwargs)
            
            # Generate cache key
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                key = f"{func.__name__}:{args}:{kwargs}"
            
            # Try cache
            value, _ = cache.get(key)
            if value is not None:
                return value
            
            # Call function
            result = func(*args, **kwargs)
            
            # Cache result
            cache.set(key, result, ttl=ttl)
            
            return result
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache = get_cache(cache_name, **cache_kwargs)
            
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                key = f"{func.__name__}:{args}:{kwargs}"
            
            value, _ = cache.get(key)
            if value is not None:
                return value
            
            result = await func(*args, **kwargs)
            cache.set(key, result, ttl=ttl)
            
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "SemanticCache",
    "CacheEntry",
    "EmbeddingModel",
    "get_cache",
    "clear_all_caches",
    "get_all_cache_stats",
    "cached",
]
