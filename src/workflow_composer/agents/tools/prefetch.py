"""Proactive Prefetching for BioPipelines.

This module provides background prefetching of dataset details after search
results are returned. This speeds up follow-up queries by having data ready.

Key Components:
    - PrefetchManager: Coordinates background prefetch tasks
    - PrefetchStrategy: Defines what to prefetch for each result type
    - BackgroundExecutor: Non-blocking task execution

Example:
    >>> prefetcher = PrefetchManager()
    >>> await prefetcher.prefetch_after_search(search_results)
    >>> # Later, when user asks for details, data is already cached
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor

if TYPE_CHECKING:
    from workflow_composer.data.discovery.adapters.base import DatasetInfo

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

class PrefetchPriority(Enum):
    """Priority levels for prefetch tasks."""
    HIGH = 1      # Top results, likely to be accessed
    MEDIUM = 2    # Secondary results
    LOW = 3       # Background enrichment


@dataclass
class PrefetchConfig:
    """Configuration for prefetch behavior."""
    
    # How many top results to prefetch
    top_n_results: int = 3
    
    # Maximum concurrent prefetch tasks
    max_concurrent: int = 5
    
    # Timeout for each prefetch operation (seconds)
    prefetch_timeout: float = 10.0
    
    # Whether to prefetch metadata
    prefetch_metadata: bool = True
    
    # Whether to prefetch download URLs
    prefetch_urls: bool = True
    
    # Whether to prefetch file counts
    prefetch_file_info: bool = True
    
    # Cancel pending prefetches on new search
    cancel_on_new_search: bool = True
    
    # Maximum pending tasks before dropping low priority
    max_pending_tasks: int = 20


# -----------------------------------------------------------------------------
# Prefetch Task
# -----------------------------------------------------------------------------

@dataclass
class PrefetchTask:
    """A single prefetch task."""
    
    task_id: str
    dataset_id: str
    source: str
    task_type: str  # "metadata", "files", "urls"
    priority: PrefetchPriority
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    
    @property
    def is_complete(self) -> bool:
        return self.completed_at is not None
    
    @property
    def is_running(self) -> bool:
        return self.started_at is not None and self.completed_at is None
    
    @property
    def duration_ms(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None


# -----------------------------------------------------------------------------
# Prefetch Strategies
# -----------------------------------------------------------------------------

class PrefetchStrategy(ABC):
    """Base class for prefetch strategies."""
    
    @abstractmethod
    def get_prefetch_tasks(
        self,
        dataset: "DatasetInfo",
        priority: PrefetchPriority,
    ) -> List[PrefetchTask]:
        """Generate prefetch tasks for a dataset."""
        pass
    
    @abstractmethod
    async def execute_task(self, task: PrefetchTask) -> Any:
        """Execute a prefetch task and return the result."""
        pass


class ENCODEPrefetchStrategy(PrefetchStrategy):
    """Prefetch strategy for ENCODE datasets."""
    
    def __init__(self, adapter: Any = None):
        self.adapter = adapter
        self._task_counter = 0
    
    def get_prefetch_tasks(
        self,
        dataset: "DatasetInfo",
        priority: PrefetchPriority,
    ) -> List[PrefetchTask]:
        tasks = []
        base_id = f"encode_{self._task_counter}"
        self._task_counter += 1
        
        # Metadata prefetch
        tasks.append(PrefetchTask(
            task_id=f"{base_id}_metadata",
            dataset_id=dataset.id,
            source="encode",
            task_type="metadata",
            priority=priority,
        ))
        
        # File info prefetch
        tasks.append(PrefetchTask(
            task_id=f"{base_id}_files",
            dataset_id=dataset.id,
            source="encode",
            task_type="files",
            priority=PrefetchPriority(min(priority.value + 1, 3)),
        ))
        
        return tasks
    
    async def execute_task(self, task: PrefetchTask) -> Any:
        """Execute ENCODE-specific prefetch."""
        if not self.adapter:
            return None
        
        if task.task_type == "metadata":
            # Fetch detailed metadata
            return await self._fetch_metadata(task.dataset_id)
        elif task.task_type == "files":
            # Fetch file list
            return await self._fetch_files(task.dataset_id)
        
        return None
    
    async def _fetch_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Fetch detailed metadata from ENCODE."""
        if self.adapter and hasattr(self.adapter, 'get_dataset_details'):
            try:
                return await asyncio.wait_for(
                    self.adapter.get_dataset_details(dataset_id),
                    timeout=10.0
                )
            except Exception as e:
                logger.debug(f"Prefetch metadata failed for {dataset_id}: {e}")
        return {}
    
    async def _fetch_files(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Fetch file list from ENCODE."""
        if self.adapter and hasattr(self.adapter, 'get_files'):
            try:
                return await asyncio.wait_for(
                    self.adapter.get_files(dataset_id),
                    timeout=10.0
                )
            except Exception as e:
                logger.debug(f"Prefetch files failed for {dataset_id}: {e}")
        return []


class GEOPrefetchStrategy(PrefetchStrategy):
    """Prefetch strategy for GEO datasets."""
    
    def __init__(self, adapter: Any = None):
        self.adapter = adapter
        self._task_counter = 0
    
    def get_prefetch_tasks(
        self,
        dataset: "DatasetInfo",
        priority: PrefetchPriority,
    ) -> List[PrefetchTask]:
        tasks = []
        base_id = f"geo_{self._task_counter}"
        self._task_counter += 1
        
        # GEO metadata prefetch
        tasks.append(PrefetchTask(
            task_id=f"{base_id}_metadata",
            dataset_id=dataset.id,
            source="geo",
            task_type="metadata",
            priority=priority,
        ))
        
        return tasks
    
    async def execute_task(self, task: PrefetchTask) -> Any:
        """Execute GEO-specific prefetch."""
        if not self.adapter:
            return None
        
        if task.task_type == "metadata":
            return await self._fetch_metadata(task.dataset_id)
        
        return None
    
    async def _fetch_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Fetch detailed metadata from GEO."""
        if self.adapter and hasattr(self.adapter, 'get_dataset_details'):
            try:
                return await asyncio.wait_for(
                    self.adapter.get_dataset_details(dataset_id),
                    timeout=10.0
                )
            except Exception as e:
                logger.debug(f"Prefetch GEO metadata failed for {dataset_id}: {e}")
        return {}


class GDCPrefetchStrategy(PrefetchStrategy):
    """Prefetch strategy for GDC datasets."""
    
    def __init__(self, adapter: Any = None):
        self.adapter = adapter
        self._task_counter = 0
    
    def get_prefetch_tasks(
        self,
        dataset: "DatasetInfo",
        priority: PrefetchPriority,
    ) -> List[PrefetchTask]:
        tasks = []
        base_id = f"gdc_{self._task_counter}"
        self._task_counter += 1
        
        # GDC metadata prefetch
        tasks.append(PrefetchTask(
            task_id=f"{base_id}_metadata",
            dataset_id=dataset.id,
            source="gdc",
            task_type="metadata",
            priority=priority,
        ))
        
        # GDC case info (cancer-specific)
        tasks.append(PrefetchTask(
            task_id=f"{base_id}_cases",
            dataset_id=dataset.id,
            source="gdc",
            task_type="cases",
            priority=PrefetchPriority(min(priority.value + 1, 3)),
        ))
        
        return tasks
    
    async def execute_task(self, task: PrefetchTask) -> Any:
        """Execute GDC-specific prefetch."""
        if not self.adapter:
            return None
        
        if task.task_type == "metadata":
            return await self._fetch_metadata(task.dataset_id)
        elif task.task_type == "cases":
            return await self._fetch_cases(task.dataset_id)
        
        return None
    
    async def _fetch_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Fetch detailed metadata from GDC."""
        if self.adapter and hasattr(self.adapter, 'get_file_details'):
            try:
                return await asyncio.wait_for(
                    self.adapter.get_file_details(dataset_id),
                    timeout=10.0
                )
            except Exception as e:
                logger.debug(f"Prefetch GDC metadata failed for {dataset_id}: {e}")
        return {}
    
    async def _fetch_cases(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Fetch case information from GDC."""
        if self.adapter and hasattr(self.adapter, 'get_cases'):
            try:
                return await asyncio.wait_for(
                    self.adapter.get_cases(dataset_id),
                    timeout=10.0
                )
            except Exception as e:
                logger.debug(f"Prefetch GDC cases failed for {dataset_id}: {e}")
        return []


# -----------------------------------------------------------------------------
# Background Executor
# -----------------------------------------------------------------------------

class BackgroundExecutor:
    """Executes tasks in the background without blocking."""
    
    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._pending_tasks: Dict[str, asyncio.Task] = {}
        self._completed_tasks: Dict[str, PrefetchTask] = {}
        self._lock = asyncio.Lock()
    
    async def submit(
        self,
        task: PrefetchTask,
        execute_fn: Callable[[PrefetchTask], Any],
    ) -> None:
        """Submit a task for background execution."""
        async with self._lock:
            if task.task_id in self._pending_tasks:
                return  # Already running
            
            async_task = asyncio.create_task(
                self._execute_with_tracking(task, execute_fn)
            )
            self._pending_tasks[task.task_id] = async_task
    
    async def _execute_with_tracking(
        self,
        task: PrefetchTask,
        execute_fn: Callable[[PrefetchTask], Any],
    ) -> None:
        """Execute task with timing and error tracking."""
        task.started_at = datetime.now()
        
        try:
            result = await execute_fn(task)
            task.result = result
        except asyncio.CancelledError:
            task.error = "Cancelled"
            raise
        except Exception as e:
            task.error = str(e)
            logger.debug(f"Prefetch task {task.task_id} failed: {e}")
        finally:
            task.completed_at = datetime.now()
            
            async with self._lock:
                self._pending_tasks.pop(task.task_id, None)
                self._completed_tasks[task.task_id] = task
    
    async def cancel_all(self) -> int:
        """Cancel all pending tasks. Returns count of cancelled tasks."""
        async with self._lock:
            cancelled = 0
            for task_id, async_task in list(self._pending_tasks.items()):
                if not async_task.done():
                    async_task.cancel()
                    cancelled += 1
            self._pending_tasks.clear()
            return cancelled
    
    async def cancel_by_source(self, source: str) -> int:
        """Cancel pending tasks for a specific source."""
        async with self._lock:
            cancelled = 0
            to_remove = []
            
            for task_id, async_task in self._pending_tasks.items():
                if source in task_id:
                    if not async_task.done():
                        async_task.cancel()
                        cancelled += 1
                    to_remove.append(task_id)
            
            for task_id in to_remove:
                self._pending_tasks.pop(task_id, None)
            
            return cancelled
    
    def get_completed(self, task_id: str) -> Optional[PrefetchTask]:
        """Get a completed task by ID."""
        return self._completed_tasks.get(task_id)
    
    @property
    def pending_count(self) -> int:
        """Number of pending tasks."""
        return len(self._pending_tasks)
    
    @property
    def completed_count(self) -> int:
        """Number of completed tasks."""
        return len(self._completed_tasks)
    
    def clear_completed(self) -> None:
        """Clear completed task history."""
        self._completed_tasks.clear()
    
    def shutdown(self) -> None:
        """Shutdown the executor."""
        self._executor.shutdown(wait=False)


# -----------------------------------------------------------------------------
# Prefetch Manager
# -----------------------------------------------------------------------------

class PrefetchManager:
    """Manages background prefetching of dataset information.
    
    This manager coordinates prefetch tasks after search results are returned,
    prioritizing top results that are likely to be accessed next.
    
    Example:
        >>> manager = PrefetchManager()
        >>> results = await data_discovery.search(query)
        >>> await manager.prefetch_after_search(results)
        >>> # Background prefetching starts, cached data available later
    """
    
    _instance: Optional["PrefetchManager"] = None
    
    def __init__(self, config: Optional[PrefetchConfig] = None):
        self.config = config or PrefetchConfig()
        self._executor = BackgroundExecutor(
            max_workers=self.config.max_concurrent
        )
        self._strategies: Dict[str, PrefetchStrategy] = {}
        self._cache: Dict[str, Any] = {}  # Prefetch result cache
        self._current_search_id: Optional[str] = None
        self._stats = PrefetchStats()
    
    @classmethod
    def get_instance(cls) -> "PrefetchManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = PrefetchManager()
        return cls._instance
    
    def register_strategy(self, source: str, strategy: PrefetchStrategy) -> None:
        """Register a prefetch strategy for a data source."""
        self._strategies[source.lower()] = strategy
    
    async def prefetch_after_search(
        self,
        datasets: List["DatasetInfo"],
        search_id: Optional[str] = None,
    ) -> None:
        """Start background prefetch for search results.
        
        Args:
            datasets: List of datasets from search results
            search_id: Optional ID to track this search session
        """
        # Cancel previous prefetches if configured
        if self.config.cancel_on_new_search and self._current_search_id:
            cancelled = await self._executor.cancel_all()
            if cancelled:
                logger.debug(f"Cancelled {cancelled} pending prefetch tasks")
        
        self._current_search_id = search_id or str(datetime.now().timestamp())
        
        # Generate prefetch tasks for top N results
        all_tasks: List[PrefetchTask] = []
        
        for i, dataset in enumerate(datasets[:self.config.top_n_results]):
            # Assign priority based on result position
            if i == 0:
                priority = PrefetchPriority.HIGH
            elif i < 3:
                priority = PrefetchPriority.MEDIUM
            else:
                priority = PrefetchPriority.LOW
            
            # Get source-specific strategy
            source = getattr(dataset, 'source', 'unknown').lower()
            strategy = self._strategies.get(source)
            
            if strategy:
                tasks = strategy.get_prefetch_tasks(dataset, priority)
                all_tasks.extend(tasks)
        
        # Sort by priority and submit
        all_tasks.sort(key=lambda t: t.priority.value)
        
        # Limit pending tasks
        if len(all_tasks) > self.config.max_pending_tasks:
            all_tasks = all_tasks[:self.config.max_pending_tasks]
        
        # Submit tasks
        for task in all_tasks:
            source = task.source.lower()
            strategy = self._strategies.get(source)
            
            if strategy:
                await self._executor.submit(
                    task,
                    lambda t, s=strategy: self._execute_and_cache(t, s)
                )
                self._stats.tasks_submitted += 1
    
    async def _execute_and_cache(
        self,
        task: PrefetchTask,
        strategy: PrefetchStrategy,
    ) -> Any:
        """Execute a task and cache the result."""
        try:
            result = await asyncio.wait_for(
                strategy.execute_task(task),
                timeout=self.config.prefetch_timeout
            )
            
            if result:
                cache_key = f"{task.source}:{task.dataset_id}:{task.task_type}"
                self._cache[cache_key] = {
                    "data": result,
                    "timestamp": datetime.now(),
                    "task_id": task.task_id,
                }
                self._stats.cache_entries += 1
            
            self._stats.tasks_completed += 1
            return result
            
        except asyncio.TimeoutError:
            self._stats.tasks_timeout += 1
            raise
        except Exception as e:
            self._stats.tasks_failed += 1
            raise
    
    def get_cached(
        self,
        dataset_id: str,
        source: str,
        task_type: str = "metadata",
    ) -> Optional[Any]:
        """Get prefetched data from cache.
        
        Args:
            dataset_id: The dataset ID
            source: The data source (encode, geo, gdc)
            task_type: Type of prefetched data
            
        Returns:
            Cached data if available, None otherwise
        """
        cache_key = f"{source.lower()}:{dataset_id}:{task_type}"
        entry = self._cache.get(cache_key)
        
        if entry:
            self._stats.cache_hits += 1
            return entry["data"]
        
        self._stats.cache_misses += 1
        return None
    
    async def wait_for_prefetch(
        self,
        dataset_id: str,
        source: str,
        task_type: str = "metadata",
        timeout: float = 5.0,
    ) -> Optional[Any]:
        """Wait for a specific prefetch to complete.
        
        Args:
            dataset_id: The dataset ID
            source: The data source
            task_type: Type of prefetched data
            timeout: Maximum time to wait
            
        Returns:
            Prefetched data if available within timeout
        """
        start = datetime.now()
        cache_key = f"{source.lower()}:{dataset_id}:{task_type}"
        
        while (datetime.now() - start).total_seconds() < timeout:
            if cache_key in self._cache:
                return self._cache[cache_key]["data"]
            await asyncio.sleep(0.1)
        
        return None
    
    def clear_cache(self) -> None:
        """Clear the prefetch cache."""
        self._cache.clear()
        self._stats.cache_entries = 0
    
    async def cancel_all(self) -> int:
        """Cancel all pending prefetch tasks."""
        return await self._executor.cancel_all()
    
    def get_stats(self) -> "PrefetchStats":
        """Get prefetch statistics."""
        self._stats.pending_tasks = self._executor.pending_count
        return self._stats
    
    def shutdown(self) -> None:
        """Shutdown the prefetch manager."""
        self._executor.shutdown()
        self._cache.clear()


@dataclass
class PrefetchStats:
    """Statistics for prefetch operations."""
    
    tasks_submitted: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_timeout: int = 0
    pending_tasks: int = 0
    cache_entries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    @property
    def completion_rate(self) -> float:
        """Percentage of tasks completed successfully."""
        if self.tasks_submitted == 0:
            return 0.0
        return self.tasks_completed / self.tasks_submitted * 100
    
    @property
    def cache_hit_rate(self) -> float:
        """Percentage of cache hits."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tasks_submitted": self.tasks_submitted,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "tasks_timeout": self.tasks_timeout,
            "pending_tasks": self.pending_tasks,
            "cache_entries": self.cache_entries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "completion_rate": f"{self.completion_rate:.1f}%",
            "cache_hit_rate": f"{self.cache_hit_rate:.1f}%",
        }


# -----------------------------------------------------------------------------
# Integration Helper
# -----------------------------------------------------------------------------

# Module-level singleton
_prefetch_manager: Optional[PrefetchManager] = None


def get_prefetch_manager(
    config: Optional[PrefetchConfig] = None,
    reset: bool = False
) -> PrefetchManager:
    """Get the singleton PrefetchManager instance.
    
    Args:
        config: Optional configuration (only used on first call or reset)
        reset: If True, create a new instance
        
    Returns:
        PrefetchManager singleton instance
    """
    global _prefetch_manager
    
    if _prefetch_manager is None or reset:
        _prefetch_manager = PrefetchManager(config)
        logger.info("Created new PrefetchManager instance")
    
    return _prefetch_manager


def setup_prefetching(
    encode_adapter: Any = None,
    geo_adapter: Any = None,
    gdc_adapter: Any = None,
    config: Optional[PrefetchConfig] = None,
) -> PrefetchManager:
    """Set up prefetching with data source adapters.
    
    Args:
        encode_adapter: ENCODE data adapter
        geo_adapter: GEO data adapter  
        gdc_adapter: GDC data adapter
        config: Optional prefetch configuration
        
    Returns:
        Configured PrefetchManager instance
    """
    manager = PrefetchManager(config)
    
    if encode_adapter:
        manager.register_strategy("encode", ENCODEPrefetchStrategy(encode_adapter))
    
    if geo_adapter:
        manager.register_strategy("geo", GEOPrefetchStrategy(geo_adapter))
    
    if gdc_adapter:
        manager.register_strategy("gdc", GDCPrefetchStrategy(gdc_adapter))
    
    return manager


# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------

__all__ = [
    "PrefetchManager",
    "PrefetchConfig", 
    "PrefetchTask",
    "PrefetchPriority",
    "PrefetchStrategy",
    "PrefetchStats",
    "ENCODEPrefetchStrategy",
    "GEOPrefetchStrategy",
    "GDCPrefetchStrategy",
    "BackgroundExecutor",
    "setup_prefetching",
    "get_prefetch_manager",
]
