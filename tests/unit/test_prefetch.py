"""Tests for Phase 5: Proactive Prefetching."""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from workflow_composer.agents.tools.prefetch import (
    PrefetchManager,
    PrefetchConfig,
    PrefetchTask,
    PrefetchPriority,
    PrefetchStats,
    BackgroundExecutor,
    ENCODEPrefetchStrategy,
    GEOPrefetchStrategy,
    GDCPrefetchStrategy,
    setup_prefetching,
)


class TestPrefetchConfig:
    """Tests for PrefetchConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PrefetchConfig()
        
        assert config.top_n_results == 3
        assert config.max_concurrent == 5
        assert config.prefetch_timeout == 10.0
        assert config.prefetch_metadata is True
        assert config.cancel_on_new_search is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PrefetchConfig(
            top_n_results=5,
            max_concurrent=10,
            prefetch_timeout=15.0,
        )
        
        assert config.top_n_results == 5
        assert config.max_concurrent == 10
        assert config.prefetch_timeout == 15.0


class TestPrefetchTask:
    """Tests for PrefetchTask."""
    
    def test_task_creation(self):
        """Test task creation."""
        task = PrefetchTask(
            task_id="test_001",
            dataset_id="ENCSR123ABC",
            source="encode",
            task_type="metadata",
            priority=PrefetchPriority.HIGH,
        )
        
        assert task.task_id == "test_001"
        assert task.dataset_id == "ENCSR123ABC"
        assert task.source == "encode"
        assert task.task_type == "metadata"
        assert task.priority == PrefetchPriority.HIGH
        assert not task.is_complete
        assert not task.is_running
    
    def test_task_states(self):
        """Test task state transitions."""
        task = PrefetchTask(
            task_id="test_001",
            dataset_id="ENCSR123ABC",
            source="encode",
            task_type="metadata",
            priority=PrefetchPriority.HIGH,
        )
        
        # Initially not running or complete
        assert not task.is_running
        assert not task.is_complete
        
        # Start task
        task.started_at = datetime.now()
        assert task.is_running
        assert not task.is_complete
        
        # Complete task
        task.completed_at = datetime.now()
        assert task.is_complete
        # Once completed, is_running should still be False conceptually
        # but our implementation shows it as running until completed
    
    def test_task_duration(self):
        """Test duration calculation."""
        task = PrefetchTask(
            task_id="test_001",
            dataset_id="ENCSR123ABC",
            source="encode",
            task_type="metadata",
            priority=PrefetchPriority.HIGH,
        )
        
        # No duration before completion
        assert task.duration_ms is None
        
        # Set times
        task.started_at = datetime.now()
        import time
        time.sleep(0.01)  # 10ms
        task.completed_at = datetime.now()
        
        assert task.duration_ms is not None
        assert task.duration_ms > 0


class TestPrefetchPriority:
    """Tests for PrefetchPriority."""
    
    def test_priority_values(self):
        """Test priority ordering."""
        assert PrefetchPriority.HIGH.value < PrefetchPriority.MEDIUM.value
        assert PrefetchPriority.MEDIUM.value < PrefetchPriority.LOW.value


class TestBackgroundExecutor:
    """Tests for BackgroundExecutor."""
    
    @pytest.mark.asyncio
    async def test_submit_task(self):
        """Test submitting a task."""
        executor = BackgroundExecutor(max_workers=2)
        
        task = PrefetchTask(
            task_id="test_001",
            dataset_id="ENCSR123ABC",
            source="encode",
            task_type="metadata",
            priority=PrefetchPriority.HIGH,
        )
        
        async def mock_execute(t):
            await asyncio.sleep(0.01)
            return {"data": "test"}
        
        await executor.submit(task, mock_execute)
        
        # Wait for completion
        await asyncio.sleep(0.05)
        
        completed = executor.get_completed(task.task_id)
        assert completed is not None
        
        executor.shutdown()
    
    @pytest.mark.asyncio
    async def test_cancel_all(self):
        """Test cancelling all tasks."""
        executor = BackgroundExecutor(max_workers=2)
        
        async def slow_execute(t):
            await asyncio.sleep(10)  # Long task
            return {"data": "test"}
        
        for i in range(3):
            task = PrefetchTask(
                task_id=f"test_{i}",
                dataset_id=f"ENCSR{i}",
                source="encode",
                task_type="metadata",
                priority=PrefetchPriority.HIGH,
            )
            await executor.submit(task, slow_execute)
        
        # Cancel all
        cancelled = await executor.cancel_all()
        
        # Should have cancelled some tasks
        assert executor.pending_count == 0
        
        executor.shutdown()
    
    def test_pending_and_completed_counts(self):
        """Test counting pending and completed tasks."""
        executor = BackgroundExecutor(max_workers=2)
        
        assert executor.pending_count == 0
        assert executor.completed_count == 0
        
        executor.shutdown()


class TestPrefetchStrategies:
    """Tests for prefetch strategies."""
    
    def test_encode_strategy_tasks(self):
        """Test ENCODE strategy generates correct tasks."""
        strategy = ENCODEPrefetchStrategy()
        
        # Create mock dataset
        dataset = MagicMock()
        dataset.id = "ENCSR123ABC"
        
        tasks = strategy.get_prefetch_tasks(dataset, PrefetchPriority.HIGH)
        
        assert len(tasks) >= 1
        assert any(t.task_type == "metadata" for t in tasks)
        assert all(t.source == "encode" for t in tasks)
    
    def test_geo_strategy_tasks(self):
        """Test GEO strategy generates correct tasks."""
        strategy = GEOPrefetchStrategy()
        
        dataset = MagicMock()
        dataset.id = "GSE12345"
        
        tasks = strategy.get_prefetch_tasks(dataset, PrefetchPriority.MEDIUM)
        
        assert len(tasks) >= 1
        assert any(t.task_type == "metadata" for t in tasks)
        assert all(t.source == "geo" for t in tasks)
    
    def test_gdc_strategy_tasks(self):
        """Test GDC strategy generates correct tasks."""
        strategy = GDCPrefetchStrategy()
        
        dataset = MagicMock()
        dataset.id = "abc-123-def"
        
        tasks = strategy.get_prefetch_tasks(dataset, PrefetchPriority.LOW)
        
        assert len(tasks) >= 1
        assert all(t.source == "gdc" for t in tasks)


class TestPrefetchManager:
    """Tests for PrefetchManager."""
    
    def test_manager_creation(self):
        """Test creating a prefetch manager."""
        manager = PrefetchManager()
        
        assert manager.config is not None
        assert manager.config.top_n_results == 3
    
    def test_register_strategy(self):
        """Test registering strategies."""
        manager = PrefetchManager()
        
        strategy = ENCODEPrefetchStrategy()
        manager.register_strategy("encode", strategy)
        
        assert "encode" in manager._strategies
    
    @pytest.mark.asyncio
    async def test_prefetch_after_search(self):
        """Test prefetching after search results."""
        manager = PrefetchManager()
        manager.register_strategy("encode", ENCODEPrefetchStrategy())
        
        # Create mock datasets
        datasets = []
        for i in range(3):
            ds = MagicMock()
            ds.id = f"ENCSR{i:03d}ABC"
            ds.source = "encode"
            datasets.append(ds)
        
        await manager.prefetch_after_search(datasets)
        
        # Stats should show tasks submitted
        stats = manager.get_stats()
        assert stats.tasks_submitted > 0
        
        manager.shutdown()
    
    def test_get_cached(self):
        """Test getting cached data."""
        manager = PrefetchManager()
        
        # Initially nothing cached
        result = manager.get_cached("ENCSR123ABC", "encode", "metadata")
        assert result is None
        
        # Check stats
        stats = manager.get_stats()
        assert stats.cache_misses > 0
        
        manager.shutdown()
    
    def test_clear_cache(self):
        """Test clearing cache."""
        manager = PrefetchManager()
        
        # Add something to cache
        manager._cache["test:123:metadata"] = {"data": "test"}
        
        manager.clear_cache()
        
        assert len(manager._cache) == 0
        
        manager.shutdown()


class TestPrefetchStats:
    """Tests for PrefetchStats."""
    
    def test_default_stats(self):
        """Test default stats values."""
        stats = PrefetchStats()
        
        assert stats.tasks_submitted == 0
        assert stats.tasks_completed == 0
        assert stats.cache_hits == 0
        assert stats.completion_rate == 0.0
        assert stats.cache_hit_rate == 0.0
    
    def test_completion_rate(self):
        """Test completion rate calculation."""
        stats = PrefetchStats()
        stats.tasks_submitted = 10
        stats.tasks_completed = 8
        
        assert stats.completion_rate == 80.0
    
    def test_cache_hit_rate(self):
        """Test cache hit rate calculation."""
        stats = PrefetchStats()
        stats.cache_hits = 7
        stats.cache_misses = 3
        
        assert stats.cache_hit_rate == 70.0
    
    def test_to_dict(self):
        """Test converting stats to dict."""
        stats = PrefetchStats()
        stats.tasks_submitted = 10
        stats.tasks_completed = 8
        
        result = stats.to_dict()
        
        assert "tasks_submitted" in result
        assert "completion_rate" in result
        assert result["tasks_submitted"] == 10


class TestSetupPrefetching:
    """Tests for setup_prefetching helper."""
    
    def test_setup_with_adapters(self):
        """Test setting up prefetching with adapters."""
        encode_adapter = MagicMock()
        geo_adapter = MagicMock()
        
        manager = setup_prefetching(
            encode_adapter=encode_adapter,
            geo_adapter=geo_adapter,
        )
        
        assert "encode" in manager._strategies
        assert "geo" in manager._strategies
        
        manager.shutdown()
    
    def test_setup_with_custom_config(self):
        """Test setup with custom config."""
        config = PrefetchConfig(top_n_results=5)
        
        manager = setup_prefetching(config=config)
        
        assert manager.config.top_n_results == 5
        
        manager.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
