"""
Parallel Federated Search
=========================

Execute searches across multiple data sources in parallel for faster results.

Features:
- Parallel query execution with asyncio
- Per-source timeouts with graceful degradation
- Result merging and deduplication
- Circuit breaker integration

Usage:
    from workflow_composer.data.discovery.parallel import ParallelSearchOrchestrator
    
    orchestrator = ParallelSearchOrchestrator()
    
    # Search all sources in parallel
    results = await orchestrator.search(query)
    
    print(f"Found {len(results.datasets)} datasets")
    print(f"Sources queried: {results.sources_queried}")
    print(f"Sources failed: {results.sources_failed}")
    print(f"Time: {results.total_time_ms:.0f}ms")

Performance:
    Sequential: ENCODE (2s) + GEO (1.5s) + GDC (2s) = 5.5s
    Parallel:   max(ENCODE, GEO, GDC) + merge = ~2.2s
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
import hashlib

from .models import DatasetInfo, SearchQuery, DataSource
from .adapters.base import BaseAdapter

logger = logging.getLogger(__name__)


# =============================================================================
# Result Types
# =============================================================================

@dataclass
class SourceResult:
    """Result from a single source."""
    source: str
    datasets: List[DatasetInfo]
    success: bool
    error: Optional[str] = None
    duration_ms: float = 0.0
    cache_hit: bool = False


@dataclass
class FederatedSearchResult:
    """Combined result from federated search."""
    
    datasets: List[DatasetInfo]
    """Merged and deduplicated datasets."""
    
    sources_queried: List[str]
    """Sources that were queried."""
    
    sources_succeeded: List[str]
    """Sources that returned results."""
    
    sources_failed: List[str]
    """Sources that failed."""
    
    total_time_ms: float
    """Total search time in milliseconds."""
    
    cache_hits: int = 0
    """Number of cache hits."""
    
    source_results: Dict[str, SourceResult] = field(default_factory=dict)
    """Per-source results for debugging."""
    
    @property
    def total_datasets(self) -> int:
        """Total number of datasets found."""
        return len(self.datasets)
    
    @property
    def success_rate(self) -> float:
        """Percentage of sources that succeeded."""
        if not self.sources_queried:
            return 0.0
        return len(self.sources_succeeded) / len(self.sources_queried)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_datasets": self.total_datasets,
            "sources_queried": self.sources_queried,
            "sources_succeeded": self.sources_succeeded,
            "sources_failed": self.sources_failed,
            "total_time_ms": self.total_time_ms,
            "cache_hits": self.cache_hits,
            "success_rate": self.success_rate,
        }


# =============================================================================
# Parallel Search Orchestrator
# =============================================================================

class ParallelSearchOrchestrator:
    """
    Orchestrate parallel searches across multiple data sources.
    
    Features:
    - Parallel execution with asyncio.gather
    - Per-source timeouts
    - Graceful degradation (failed sources return empty)
    - Result deduplication by dataset ID and title similarity
    - Integration with circuit breakers
    
    Example:
        from workflow_composer.data.discovery.adapters import (
            ENCODEAdapter, GEOAdapter, GDCAdapter
        )
        
        orchestrator = ParallelSearchOrchestrator({
            "encode": ENCODEAdapter(),
            "geo": GEOAdapter(),
            "gdc": GDCAdapter(),
        })
        
        result = await orchestrator.search(SearchQuery(
            organism="human",
            assay_type="ChIP-seq",
        ))
    """
    
    def __init__(
        self,
        adapters: Dict[str, BaseAdapter] = None,
        timeout_per_source: float = 10.0,
        overall_timeout: float = 15.0,
        enable_deduplication: bool = True,
        title_similarity_threshold: float = 0.9,
    ):
        """
        Initialize the parallel search orchestrator.
        
        Args:
            adapters: Dict mapping source name to adapter instance
            timeout_per_source: Timeout for each source in seconds
            overall_timeout: Overall timeout for the entire search
            enable_deduplication: Whether to deduplicate results
            title_similarity_threshold: Threshold for title-based dedup
        """
        self.adapters = adapters or {}
        self.timeout_per_source = timeout_per_source
        self.overall_timeout = overall_timeout
        self.enable_deduplication = enable_deduplication
        self.title_similarity_threshold = title_similarity_threshold
    
    def add_adapter(self, name: str, adapter: BaseAdapter) -> None:
        """Add a data source adapter."""
        self.adapters[name] = adapter
    
    def remove_adapter(self, name: str) -> None:
        """Remove a data source adapter."""
        self.adapters.pop(name, None)
    
    async def search(
        self,
        query: SearchQuery,
        sources: List[str] = None,
    ) -> FederatedSearchResult:
        """
        Search multiple sources in parallel.
        
        Args:
            query: The search query
            sources: List of source names to query (default: all)
        
        Returns:
            FederatedSearchResult with merged datasets
        """
        start_time = datetime.now()
        
        # Determine which sources to query
        sources_to_query = sources or list(self.adapters.keys())
        
        # Filter to available adapters
        sources_to_query = [s for s in sources_to_query if s in self.adapters]
        
        if not sources_to_query:
            logger.warning("No adapters available for search")
            return FederatedSearchResult(
                datasets=[],
                sources_queried=[],
                sources_succeeded=[],
                sources_failed=[],
                total_time_ms=0.0,
            )
        
        logger.info(f"Starting parallel search across: {sources_to_query}")
        
        # Create search tasks
        tasks = []
        for source_name in sources_to_query:
            task = self._search_source_with_timeout(
                source_name,
                self.adapters[source_name],
                query,
            )
            tasks.append(task)
        
        # Execute in parallel with overall timeout
        try:
            source_results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.overall_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Overall search timeout after {self.overall_timeout}s")
            source_results = []
        
        # Process results
        all_datasets: List[DatasetInfo] = []
        sources_succeeded: List[str] = []
        sources_failed: List[str] = []
        cache_hits = 0
        result_map: Dict[str, SourceResult] = {}
        
        for i, source_name in enumerate(sources_to_query):
            if i >= len(source_results):
                # Didn't get result due to timeout
                sources_failed.append(source_name)
                result_map[source_name] = SourceResult(
                    source=source_name,
                    datasets=[],
                    success=False,
                    error="Overall timeout",
                )
                continue
            
            result = source_results[i]
            
            if isinstance(result, Exception):
                logger.warning(f"Source {source_name} failed: {result}")
                sources_failed.append(source_name)
                result_map[source_name] = SourceResult(
                    source=source_name,
                    datasets=[],
                    success=False,
                    error=str(result),
                )
            elif isinstance(result, SourceResult):
                if result.success:
                    sources_succeeded.append(source_name)
                    all_datasets.extend(result.datasets)
                    if result.cache_hit:
                        cache_hits += 1
                else:
                    sources_failed.append(source_name)
                result_map[source_name] = result
        
        # Deduplicate results
        if self.enable_deduplication and all_datasets:
            unique_datasets = self._deduplicate(all_datasets)
            logger.info(
                f"Deduplication: {len(all_datasets)} -> {len(unique_datasets)} datasets"
            )
        else:
            unique_datasets = all_datasets
        
        # Calculate total time
        total_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(
            f"Parallel search complete: {len(unique_datasets)} datasets "
            f"from {len(sources_succeeded)}/{len(sources_to_query)} sources "
            f"in {total_time_ms:.0f}ms"
        )
        
        return FederatedSearchResult(
            datasets=unique_datasets,
            sources_queried=sources_to_query,
            sources_succeeded=sources_succeeded,
            sources_failed=sources_failed,
            total_time_ms=total_time_ms,
            cache_hits=cache_hits,
            source_results=result_map,
        )
    
    async def _search_source_with_timeout(
        self,
        source_name: str,
        adapter: BaseAdapter,
        query: SearchQuery,
    ) -> SourceResult:
        """
        Search a single source with timeout.
        
        Args:
            source_name: Name of the source
            adapter: The adapter instance
            query: Search query
        
        Returns:
            SourceResult with datasets or error
        """
        start_time = datetime.now()
        
        try:
            # Check circuit breaker if available
            from workflow_composer.infrastructure.resilience import (
                get_circuit_breaker,
                CircuitBreakerError,
            )
            
            breaker = get_circuit_breaker(f"search_{source_name}")
            
            if not breaker.can_execute():
                return SourceResult(
                    source=source_name,
                    datasets=[],
                    success=False,
                    error=f"Circuit breaker open for {source_name}",
                    duration_ms=0.0,
                )
            
            # Execute search with timeout
            if asyncio.iscoroutinefunction(adapter.search):
                datasets = await asyncio.wait_for(
                    adapter.search(query),
                    timeout=self.timeout_per_source,
                )
            else:
                # Run sync adapter in thread pool
                loop = asyncio.get_event_loop()
                datasets = await asyncio.wait_for(
                    loop.run_in_executor(None, adapter.search, query),
                    timeout=self.timeout_per_source,
                )
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            breaker.record_success()
            
            logger.debug(
                f"Source {source_name}: {len(datasets)} datasets in {duration_ms:.0f}ms"
            )
            
            return SourceResult(
                source=source_name,
                datasets=datasets,
                success=True,
                duration_ms=duration_ms,
            )
        
        except asyncio.TimeoutError:
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.warning(f"Source {source_name} timed out after {duration_ms:.0f}ms")
            
            try:
                breaker.record_failure()
            except:
                pass
            
            return SourceResult(
                source=source_name,
                datasets=[],
                success=False,
                error=f"Timeout after {self.timeout_per_source}s",
                duration_ms=duration_ms,
            )
        
        except Exception as e:
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.warning(f"Source {source_name} error: {e}")
            
            try:
                breaker.record_failure()
            except:
                pass
            
            return SourceResult(
                source=source_name,
                datasets=[],
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )
    
    def _deduplicate(
        self,
        datasets: List[DatasetInfo],
    ) -> List[DatasetInfo]:
        """
        Deduplicate datasets by ID and title similarity.
        
        Strategy:
        1. First pass: exact ID match
        2. Second pass: title similarity for cross-source duplicates
        
        Args:
            datasets: List of datasets to deduplicate
        
        Returns:
            Deduplicated list
        """
        if not datasets:
            return []
        
        # First pass: exact ID match
        seen_ids: Set[str] = set()
        unique: List[DatasetInfo] = []
        
        for ds in datasets:
            if ds.id not in seen_ids:
                seen_ids.add(ds.id)
                unique.append(ds)
        
        # Second pass: title similarity (optional, more expensive)
        if self.title_similarity_threshold < 1.0 and len(unique) > 1:
            final: List[DatasetInfo] = []
            seen_title_hashes: Set[str] = set()
            
            for ds in unique:
                # Create a normalized title hash
                if ds.title:
                    normalized = self._normalize_title(ds.title)
                    title_hash = hashlib.md5(normalized.encode()).hexdigest()[:16]
                    
                    if title_hash not in seen_title_hashes:
                        seen_title_hashes.add(title_hash)
                        final.append(ds)
                else:
                    final.append(ds)
            
            return final
        
        return unique
    
    def _normalize_title(self, title: str) -> str:
        """Normalize a title for deduplication."""
        # Lowercase
        title = title.lower()
        
        # Remove common prefixes/suffixes
        prefixes = ["gse", "encsr", "tcga-"]
        for prefix in prefixes:
            if title.startswith(prefix):
                title = title[len(prefix):]
        
        # Remove punctuation and extra spaces
        import re
        title = re.sub(r'[^\w\s]', ' ', title)
        title = re.sub(r'\s+', ' ', title)
        
        return title.strip()
    
    def search_sync(
        self,
        query: SearchQuery,
        sources: List[str] = None,
    ) -> FederatedSearchResult:
        """
        Synchronous wrapper for search.
        
        Creates an event loop if needed.
        """
        try:
            loop = asyncio.get_running_loop()
            # Already in async context
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.search(query, sources)
                )
                return future.result()
        except RuntimeError:
            # No running loop, create one
            return asyncio.run(self.search(query, sources))


# =============================================================================
# Factory Function
# =============================================================================

def create_parallel_orchestrator(
    include_encode: bool = True,
    include_geo: bool = True,
    include_gdc: bool = True,
    **kwargs,
) -> ParallelSearchOrchestrator:
    """
    Create a parallel search orchestrator with default adapters.
    
    Args:
        include_encode: Whether to include ENCODE adapter
        include_geo: Whether to include GEO adapter
        include_gdc: Whether to include GDC adapter
        **kwargs: Additional arguments for ParallelSearchOrchestrator
    
    Returns:
        Configured orchestrator
    """
    adapters = {}
    
    if include_encode:
        try:
            from .adapters.encode import ENCODEAdapter
            adapters["encode"] = ENCODEAdapter()
        except ImportError:
            logger.warning("ENCODE adapter not available")
    
    if include_geo:
        try:
            from .adapters.geo import GEOAdapter
            adapters["geo"] = GEOAdapter()
        except ImportError:
            logger.warning("GEO adapter not available")
    
    if include_gdc:
        try:
            from .adapters.gdc import GDCAdapter
            adapters["gdc"] = GDCAdapter()
        except ImportError:
            logger.warning("GDC adapter not available")
    
    return ParallelSearchOrchestrator(adapters=adapters, **kwargs)


# =============================================================================
# Convenience Function
# =============================================================================

async def parallel_search(
    query: SearchQuery,
    sources: List[str] = None,
    timeout: float = 15.0,
) -> FederatedSearchResult:
    """
    Convenience function for parallel federated search.
    
    Creates a default orchestrator and executes the search.
    
    Args:
        query: Search query
        sources: List of source names to query (default: all)
        timeout: Overall timeout in seconds
    
    Returns:
        FederatedSearchResult with merged datasets
    
    Example:
        results = await parallel_search(
            SearchQuery(organism="human", assay_type="ChIP-seq")
        )
    """
    orchestrator = create_parallel_orchestrator(overall_timeout=timeout)
    return await orchestrator.search(query, sources)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ParallelSearchOrchestrator",
    "FederatedSearchResult",
    "SourceResult",
    "create_parallel_orchestrator",
    "parallel_search",
]
