"""
Tests for the semantic cache module.

Tests TTL-based caching, LRU eviction, and semantic similarity matching.
"""

import pytest
import time
from datetime import datetime, timedelta

from workflow_composer.infrastructure.semantic_cache import (
    SemanticCache,
    CacheEntry,
    get_cache,
    clear_all_caches,
    get_all_cache_stats,
    cached,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_caches():
    """Reset global caches before each test."""
    clear_all_caches()
    yield
    clear_all_caches()


@pytest.fixture
def cache():
    """Create a fresh cache for testing."""
    return SemanticCache(
        default_ttl=1,  # Short TTL for testing
        max_size=10,
        enable_semantic=False,  # Disable for basic tests
    )


@pytest.fixture
def similarity_cache():
    """Create a cache with similarity matching enabled."""
    return SemanticCache(
        default_ttl=60,
        max_size=100,
        enable_semantic=True,
        similarity_threshold=0.8,
    )


# =============================================================================
# Basic Cache Tests
# =============================================================================

class TestSemanticCache:
    """Test the SemanticCache class."""
    
    def test_get_cache_singleton(self):
        """Test that get_cache returns cached instances by name."""
        cache1 = get_cache("test_cache")
        cache2 = get_cache("test_cache")
        cache3 = get_cache("different_cache")
        
        assert cache1 is cache2
        assert cache1 is not cache3
    
    def test_set_and_get(self, cache):
        """Test basic set and get operations."""
        cache.set("key1", "value1")
        value, exact = cache.get("key1")
        assert value == "value1"
        assert exact is True
    
    def test_get_missing_key(self, cache):
        """Test getting a non-existent key."""
        value, exact = cache.get("missing")
        assert value is None
        assert exact is False
    
    def test_ttl_expiration(self, cache):
        """Test that entries expire after TTL."""
        cache.set("key1", "value1")
        value, _ = cache.get("key1")
        assert value == "value1"
        
        # Wait for TTL to expire
        time.sleep(1.1)
        value, _ = cache.get("key1")
        assert value is None
    
    def test_delete(self, cache):
        """Test deleting a key."""
        cache.set("key1", "value1")
        value, _ = cache.get("key1")
        assert value == "value1"
        
        cache.delete("key1")
        value, _ = cache.get("key1")
        assert value is None
    
    def test_clear(self, cache):
        """Test clearing all entries."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        cache.clear()
        
        value1, _ = cache.get("key1")
        value2, _ = cache.get("key2")
        assert value1 is None
        assert value2 is None


class TestLRUEviction:
    """Test LRU eviction behavior."""
    
    def test_eviction_when_full(self):
        """Test that oldest entries are evicted when cache is full."""
        cache = SemanticCache(
            max_size=3,
            default_ttl=60,
            enable_semantic=False,
        )
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        # Cache is full, adding a new entry should evict the oldest
        cache.set("key4", "value4")
        
        # key1 should be evicted (oldest)
        value1, _ = cache.get("key1")
        value2, _ = cache.get("key2")
        value3, _ = cache.get("key3")
        value4, _ = cache.get("key4")
        
        assert value1 is None  # Evicted
        assert value2 == "value2"
        assert value3 == "value3"
        assert value4 == "value4"


class TestCacheStats:
    """Test cache statistics."""
    
    def test_hit_and_miss_tracking(self):
        """Test that hits and misses are tracked."""
        cache = SemanticCache(default_ttl=60, enable_semantic=False)
        
        cache.set("key1", "value1")
        
        # Hit
        cache.get("key1")
        # Miss
        cache.get("missing")
        cache.get("also_missing")
        
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
    
    def test_get_all_cache_stats(self):
        """Test getting stats for all caches."""
        cache1 = get_cache("cache_a")
        cache2 = get_cache("cache_b")
        
        cache1.set("key", "value")
        cache2.set("key", "value")
        
        all_stats = get_all_cache_stats()
        assert "cache_a" in all_stats
        assert "cache_b" in all_stats


# =============================================================================
# Cached Decorator Tests
# =============================================================================

class TestCachedDecorator:
    """Test the @cached decorator."""
    
    def test_cached_function_result(self):
        """Test that function results are cached."""
        call_count = 0
        
        @cached(ttl=60)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call with same args - should use cache
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # No new call
        
        # Different args - should call function
        result3 = expensive_function(3, 4)
        assert result3 == 7
        assert call_count == 2
    
    def test_cached_with_ttl_expiry(self):
        """Test that cached results expire."""
        call_count = 0
        
        @cached(ttl=1)
        def time_sensitive():
            nonlocal call_count
            call_count += 1
            return datetime.now().isoformat()
        
        # First call
        time_sensitive()
        assert call_count == 1
        
        # Immediate second call - cached
        time_sensitive()
        assert call_count == 1
        
        # Wait for expiry
        time.sleep(1.1)
        time_sensitive()
        assert call_count == 2


# =============================================================================
# Complex Value Tests
# =============================================================================

class TestComplexValues:
    """Test caching complex values."""
    
    def test_cache_dict(self, cache):
        """Test caching dictionaries."""
        data = {"name": "test", "values": [1, 2, 3]}
        cache.set("data", data)
        
        retrieved, _ = cache.get("data")
        assert retrieved == data
    
    def test_cache_list(self, cache):
        """Test caching lists."""
        data = [1, 2, 3, "four", {"five": 5}]
        cache.set("list", data)
        
        value, _ = cache.get("list")
        assert value == data
    
    def test_cache_none_value(self, cache):
        """Test caching None as a value."""
        cache.set("none_key", None)
        
        # Should return None (the cached value), not a miss
        value, exact = cache.get("none_key")
        assert value is None
        # exact should be True since we found the key
        assert exact is True


# =============================================================================
# Thread Safety Tests
# =============================================================================

class TestThreadSafety:
    """Test thread safety of cache operations."""
    
    def test_concurrent_writes(self):
        """Test concurrent writes don't cause issues."""
        import threading
        
        cache = SemanticCache(max_size=1000, default_ttl=60, enable_semantic=False)
        errors = []
        
        def writer(start, end):
            try:
                for i in range(start, end):
                    cache.set(f"key_{i}", f"value_{i}")
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=writer, args=(0, 100)),
            threading.Thread(target=writer, args=(100, 200)),
            threading.Thread(target=writer, args=(200, 300)),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
    
    def test_concurrent_reads_and_writes(self):
        """Test concurrent reads and writes."""
        import threading
        
        cache = SemanticCache(max_size=100, default_ttl=60, enable_semantic=False)
        
        # Pre-populate
        for i in range(50):
            cache.set(f"key_{i}", f"value_{i}")
        
        errors = []
        
        def reader():
            try:
                for i in range(50):
                    cache.get(f"key_{i}")
            except Exception as e:
                errors.append(e)
        
        def writer():
            try:
                for i in range(50, 100):
                    cache.set(f"key_{i}", f"value_{i}")
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0


# =============================================================================
# Semantic Similarity Tests
# =============================================================================

class TestSemanticSimilarity:
    """Test semantic similarity matching."""
    
    def test_similarity_disabled_exact_only(self, cache):
        """Test that without similarity, only exact matches work."""
        cache.set("find human RNA-seq data", "result1")
        
        # Exact match works
        value, exact = cache.get("find human RNA-seq data")
        assert value == "result1"
        assert exact is True
        
        # Similar query doesn't match without similarity enabled
        value2, _ = cache.get("search for human RNA-seq")
        assert value2 is None
    
    def test_stats_include_semantic_info(self, similarity_cache):
        """Test that stats include semantic information."""
        stats = similarity_cache.get_stats()
        assert "semantic_hits" in stats


# =============================================================================
# Cache Entry Tests
# =============================================================================

class TestCacheEntry:
    """Test the CacheEntry class."""
    
    def test_entry_expiration_check(self):
        """Test is_expired property."""
        entry = CacheEntry(
            key="test",
            key_hash="hash123",
            value="value",
            created_at=datetime.now(),
            ttl_seconds=1,  # 1 second TTL
        )
        
        assert not entry.is_expired
        
        # Wait for expiration
        time.sleep(1.1)
        assert entry.is_expired
    
    def test_entry_touch(self):
        """Test updating access time."""
        entry = CacheEntry(
            key="test",
            key_hash="hash123",
            value="value",
            created_at=datetime.now(),
            ttl_seconds=60,
        )
        
        old_access_time = entry.last_accessed
        old_hits = entry.hits
        time.sleep(0.01)
        entry.touch()
        
        assert entry.last_accessed > old_access_time
        assert entry.hits == old_hits + 1
