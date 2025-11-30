"""
Redis-backed Semantic Cache
===========================

Provides a Redis backend for the semantic cache to enable:
- Shared cache across multiple processes/workers
- Persistence across restarts
- Horizontal scaling

This module extends the base SemanticCache with Redis storage while
maintaining the same interface.

Usage:
    from workflow_composer.infrastructure.redis_cache import (
        RedisSemanticCache,
        get_redis_cache,
    )
    
    # Create with Redis connection
    cache = RedisSemanticCache(
        redis_url="redis://localhost:6379/0",
        key_prefix="biopipelines:",
    )
    
    # Same API as SemanticCache
    cache.set("query", results)
    value, exact = cache.get("query")

Design:
    - Uses Redis for storage (key-value with TTL)
    - Embeddings stored as JSON arrays
    - Semantic search requires loading entries (expensive at scale)
    - Falls back to in-memory if Redis unavailable
"""

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Generic

from .semantic_cache import EmbeddingModel, CacheEntry

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Redis Configuration
# =============================================================================

@dataclass
class RedisConfig:
    """Redis connection configuration."""
    
    url: str = "redis://localhost:6379/0"
    """Redis connection URL."""
    
    key_prefix: str = "biopipelines:cache:"
    """Prefix for all cache keys."""
    
    max_connections: int = 10
    """Maximum connections in pool."""
    
    socket_timeout: float = 5.0
    """Socket timeout in seconds."""
    
    retry_on_timeout: bool = True
    """Whether to retry on timeout."""
    
    decode_responses: bool = True
    """Whether to decode responses as strings."""
    
    # Fallback behavior
    fallback_to_memory: bool = True
    """Fall back to in-memory cache if Redis unavailable."""
    
    fallback_max_size: int = 1000
    """Max size for fallback in-memory cache."""


# =============================================================================
# Redis Semantic Cache
# =============================================================================

class RedisSemanticCache(Generic[T]):
    """
    Redis-backed semantic cache.
    
    Provides the same interface as SemanticCache but stores data in Redis
    for shared access across processes and persistence.
    
    Features:
    - TTL handled natively by Redis
    - Atomic operations for thread-safety
    - Connection pooling
    - Automatic fallback to in-memory if Redis unavailable
    
    Limitations:
    - Semantic search requires loading entries (O(n) Redis calls or SCAN)
    - Consider using Redis Search module for large-scale semantic search
    """
    
    def __init__(
        self,
        config: Optional[RedisConfig] = None,
        default_ttl: int = 3600,
        similarity_threshold: float = 0.85,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        enable_semantic: bool = True,
    ):
        """
        Initialize Redis cache.
        
        Args:
            config: Redis configuration
            default_ttl: Default TTL in seconds
            similarity_threshold: Minimum similarity for semantic match
            embedding_model: Sentence transformer model name
            enable_semantic: Whether to enable semantic matching
        """
        self.config = config or RedisConfig()
        self.default_ttl = default_ttl
        self.similarity_threshold = similarity_threshold
        self.enable_semantic = enable_semantic
        
        # Redis client (lazy initialized)
        self._redis: Optional[Any] = None
        self._redis_available = True
        self._redis_lock = threading.Lock()
        
        # Fallback in-memory cache
        self._fallback_cache: Dict[str, CacheEntry[T]] = {}
        self._fallback_lock = threading.RLock()
        
        # Embedding model
        self._embedder: Optional[EmbeddingModel] = None
        self._embedding_model_name = embedding_model
        
        # Stats
        self._hits = 0
        self._misses = 0
        self._semantic_hits = 0
        self._redis_errors = 0
    
    @property
    def embedder(self) -> EmbeddingModel:
        """Get or create the embedding model."""
        if self._embedder is None:
            self._embedder = EmbeddingModel(self._embedding_model_name)
        return self._embedder
    
    @property
    def redis(self) -> Optional[Any]:
        """Get Redis client (lazy initialized)."""
        if self._redis is None and self._redis_available:
            with self._redis_lock:
                if self._redis is None and self._redis_available:
                    try:
                        import redis
                        self._redis = redis.from_url(
                            self.config.url,
                            max_connections=self.config.max_connections,
                            socket_timeout=self.config.socket_timeout,
                            retry_on_timeout=self.config.retry_on_timeout,
                            decode_responses=self.config.decode_responses,
                        )
                        # Test connection
                        self._redis.ping()
                        logger.info(f"Connected to Redis: {self.config.url}")
                    except ImportError:
                        logger.warning("redis package not installed, using fallback")
                        self._redis_available = False
                    except Exception as e:
                        logger.warning(f"Failed to connect to Redis: {e}")
                        self._redis_available = False
                        if not self.config.fallback_to_memory:
                            raise
        return self._redis
    
    def _make_key(self, key: str) -> str:
        """Create a Redis key with prefix."""
        normalized = key.lower().strip()
        key_hash = hashlib.md5(normalized.encode()).hexdigest()
        return f"{self.config.key_prefix}{key_hash}"
    
    def _serialize_entry(self, entry: CacheEntry[T]) -> str:
        """Serialize cache entry for Redis storage."""
        data = {
            "key": entry.key,
            "value": entry.value,
            "created_at": entry.created_at.isoformat(),
            "ttl_seconds": entry.ttl_seconds,
            "hits": entry.hits,
            "embedding": entry.embedding,
        }
        return json.dumps(data)
    
    def _deserialize_entry(self, data: str) -> Optional[CacheEntry[T]]:
        """Deserialize cache entry from Redis."""
        try:
            parsed = json.loads(data)
            return CacheEntry(
                key=parsed["key"],
                key_hash=self._make_key(parsed["key"]),
                value=parsed["value"],
                created_at=datetime.fromisoformat(parsed["created_at"]),
                ttl_seconds=parsed["ttl_seconds"],
                hits=parsed.get("hits", 0),
                embedding=parsed.get("embedding"),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to deserialize cache entry: {e}")
            return None
    
    def get(self, key: str) -> Tuple[Optional[T], bool]:
        """
        Get a value from the cache.
        
        Returns:
            Tuple of (value, is_exact_match)
        """
        redis_key = self._make_key(key)
        
        # Try Redis first
        if self.redis is not None:
            try:
                data = self.redis.get(redis_key)
                if data is not None:
                    entry = self._deserialize_entry(data)
                    if entry is not None:
                        # Update hits
                        entry.hits += 1
                        self.redis.set(
                            redis_key,
                            self._serialize_entry(entry),
                            keepttl=True,  # Keep existing TTL
                        )
                        self._hits += 1
                        return entry.value, True
                
                # Try semantic match
                if self.enable_semantic:
                    match = self._find_semantic_match_redis(key)
                    if match is not None:
                        self._hits += 1
                        self._semantic_hits += 1
                        return match.value, False
                
                self._misses += 1
                return None, False
                
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
                self._redis_errors += 1
        
        # Fallback to in-memory
        return self._get_fallback(key)
    
    def set(
        self,
        key: str,
        value: T,
        ttl: int = None,
        compute_embedding: bool = True,
    ) -> None:
        """
        Set a value in the cache.
        """
        redis_key = self._make_key(key)
        ttl = ttl if ttl is not None else self.default_ttl
        
        # Compute embedding
        embedding = None
        if self.enable_semantic and compute_embedding:
            embedding = self.embedder.encode(key)
        
        entry = CacheEntry(
            key=key,
            key_hash=redis_key,
            value=value,
            created_at=datetime.now(),
            ttl_seconds=ttl,
            embedding=embedding,
        )
        
        # Try Redis first
        if self.redis is not None:
            try:
                self.redis.setex(
                    redis_key,
                    ttl,
                    self._serialize_entry(entry),
                )
                return
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
                self._redis_errors += 1
        
        # Fallback to in-memory
        self._set_fallback(key, entry)
    
    def delete(self, key: str) -> bool:
        """Delete a key from the cache."""
        redis_key = self._make_key(key)
        
        if self.redis is not None:
            try:
                return self.redis.delete(redis_key) > 0
            except Exception as e:
                logger.warning(f"Redis delete error: {e}")
                self._redis_errors += 1
        
        return self._delete_fallback(key)
    
    def clear(self) -> None:
        """Clear all entries with our prefix."""
        if self.redis is not None:
            try:
                pattern = f"{self.config.key_prefix}*"
                cursor = 0
                while True:
                    cursor, keys = self.redis.scan(cursor, match=pattern, count=100)
                    if keys:
                        self.redis.delete(*keys)
                    if cursor == 0:
                        break
            except Exception as e:
                logger.warning(f"Redis clear error: {e}")
                self._redis_errors += 1
        
        with self._fallback_lock:
            self._fallback_cache.clear()
    
    def _find_semantic_match_redis(self, key: str) -> Optional[CacheEntry[T]]:
        """
        Find semantically similar entry in Redis.
        
        Note: This is expensive as it requires scanning all keys.
        For production with many entries, consider using Redis Search.
        """
        if not self.enable_semantic or self.redis is None:
            return None
        
        query_embedding = self.embedder.encode(key)
        if query_embedding is None:
            return None
        
        best_entry = None
        best_score = self.similarity_threshold
        
        try:
            pattern = f"{self.config.key_prefix}*"
            cursor = 0
            
            while True:
                cursor, keys = self.redis.scan(cursor, match=pattern, count=50)
                
                # Get entries in batch
                if keys:
                    values = self.redis.mget(keys)
                    for data in values:
                        if data is None:
                            continue
                        
                        entry = self._deserialize_entry(data)
                        if entry is None or entry.embedding is None:
                            continue
                        
                        score = self.embedder.similarity(
                            query_embedding,
                            entry.embedding,
                        )
                        
                        if score > best_score:
                            best_score = score
                            best_entry = entry
                
                if cursor == 0:
                    break
                
        except Exception as e:
            logger.warning(f"Redis semantic search error: {e}")
            self._redis_errors += 1
        
        return best_entry
    
    # =========================================================================
    # Fallback Methods
    # =========================================================================
    
    def _get_fallback(self, key: str) -> Tuple[Optional[T], bool]:
        """Get from in-memory fallback cache."""
        key_hash = self._make_key(key)
        
        with self._fallback_lock:
            if key_hash in self._fallback_cache:
                entry = self._fallback_cache[key_hash]
                if entry.is_expired:
                    del self._fallback_cache[key_hash]
                    self._misses += 1
                else:
                    entry.touch()
                    self._hits += 1
                    return entry.value, True
            
            if self.enable_semantic:
                match = self._find_semantic_match_fallback(key)
                if match:
                    match.touch()
                    self._hits += 1
                    self._semantic_hits += 1
                    return match.value, False
            
            self._misses += 1
            return None, False
    
    def _set_fallback(self, key: str, entry: CacheEntry[T]) -> None:
        """Set in fallback cache."""
        with self._fallback_lock:
            # Evict if needed
            while len(self._fallback_cache) >= self.config.fallback_max_size:
                lru_key = min(
                    self._fallback_cache.keys(),
                    key=lambda k: self._fallback_cache[k].last_accessed,
                )
                del self._fallback_cache[lru_key]
            
            self._fallback_cache[entry.key_hash] = entry
    
    def _delete_fallback(self, key: str) -> bool:
        """Delete from fallback cache."""
        key_hash = self._make_key(key)
        with self._fallback_lock:
            if key_hash in self._fallback_cache:
                del self._fallback_cache[key_hash]
                return True
            return False
    
    def _find_semantic_match_fallback(self, key: str) -> Optional[CacheEntry[T]]:
        """Find semantic match in fallback cache."""
        query_embedding = self.embedder.encode(key)
        if query_embedding is None:
            return None
        
        best_entry = None
        best_score = self.similarity_threshold
        
        for entry in self._fallback_cache.values():
            if entry.is_expired:
                continue
            
            if entry.embedding is not None:
                score = self.embedder.similarity(query_embedding, entry.embedding)
                if score > best_score:
                    best_score = score
                    best_entry = entry
        
        return best_entry
    
    # =========================================================================
    # Stats
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        
        stats = {
            "hits": self._hits,
            "misses": self._misses,
            "semantic_hits": self._semantic_hits,
            "redis_errors": self._redis_errors,
            "hit_rate": self._hits / total_requests if total_requests > 0 else 0.0,
            "redis_available": self._redis_available and self._redis is not None,
        }
        
        # Add Redis info if available
        if self.redis is not None:
            try:
                info = self.redis.info("memory")
                stats["redis_memory_used"] = info.get("used_memory_human", "unknown")
            except Exception:
                pass
        
        # Add fallback size
        with self._fallback_lock:
            stats["fallback_size"] = len(self._fallback_cache)
        
        return stats
    
    def __len__(self) -> int:
        """Get approximate number of entries."""
        if self.redis is not None:
            try:
                pattern = f"{self.config.key_prefix}*"
                count = 0
                cursor = 0
                while True:
                    cursor, keys = self.redis.scan(cursor, match=pattern, count=100)
                    count += len(keys)
                    if cursor == 0:
                        break
                return count
            except Exception:
                pass
        
        with self._fallback_lock:
            return len(self._fallback_cache)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        redis_key = self._make_key(key)
        
        if self.redis is not None:
            try:
                return self.redis.exists(redis_key) > 0
            except Exception:
                pass
        
        with self._fallback_lock:
            key_hash = self._make_key(key)
            if key_hash in self._fallback_cache:
                if not self._fallback_cache[key_hash].is_expired:
                    return True
        return False


# =============================================================================
# Factory Functions
# =============================================================================

_redis_caches: Dict[str, RedisSemanticCache] = {}
_redis_cache_lock = threading.Lock()


def get_redis_cache(
    name: str = "default",
    config: Optional[RedisConfig] = None,
    **kwargs,
) -> RedisSemanticCache:
    """
    Get or create a named Redis cache.
    
    Args:
        name: Cache name
        config: Redis configuration
        **kwargs: Arguments for RedisSemanticCache constructor
    
    Returns:
        RedisSemanticCache instance
    """
    with _redis_cache_lock:
        if name not in _redis_caches:
            _redis_caches[name] = RedisSemanticCache(config=config, **kwargs)
        return _redis_caches[name]


def clear_all_redis_caches() -> None:
    """Clear all Redis caches."""
    with _redis_cache_lock:
        for cache in _redis_caches.values():
            cache.clear()


# =============================================================================
# Cache Protocol
# =============================================================================

from typing import Protocol, runtime_checkable


@runtime_checkable
class CacheProtocol(Protocol[T]):
    """Protocol for cache implementations."""
    
    def get(self, key: str) -> Tuple[Optional[T], bool]:
        """Get value by key."""
        ...
    
    def set(self, key: str, value: T, ttl: int = None) -> None:
        """Set value with optional TTL."""
        ...
    
    def delete(self, key: str) -> bool:
        """Delete by key."""
        ...
    
    def clear(self) -> None:
        """Clear all entries."""
        ...
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        ...


# Factory that respects environment
def get_cache(
    name: str = "default",
    use_redis: bool = None,
    **kwargs,
) -> CacheProtocol:
    """
    Get a cache instance, choosing backend based on configuration.
    
    Args:
        name: Cache name
        use_redis: Force Redis (True) or memory (False). Auto-detect if None.
        **kwargs: Arguments for cache constructor
    
    Returns:
        Cache instance (Redis or in-memory)
    """
    import os
    
    if use_redis is None:
        # Auto-detect based on environment
        redis_url = os.environ.get("REDIS_URL")
        use_redis = bool(redis_url)
    
    if use_redis:
        redis_url = kwargs.pop("redis_url", os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
        config = RedisConfig(url=redis_url)
        return get_redis_cache(name, config=config, **kwargs)
    else:
        from .semantic_cache import get_cache as get_memory_cache
        return get_memory_cache(name, **kwargs)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "RedisSemanticCache",
    "RedisConfig",
    "get_redis_cache",
    "clear_all_redis_caches",
    "CacheProtocol",
    "get_cache",
]
