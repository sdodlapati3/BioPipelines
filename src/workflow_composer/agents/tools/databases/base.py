"""
Base Database Client
====================

Abstract base class for all biological database API clients.
Provides common functionality for rate limiting, error handling,
and result standardization.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import logging
import time
import threading

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DatabaseResult:
    """
    Standardized result from database queries.
    
    All database clients return this format for consistent handling
    across the BioPipelines system.
    
    Attributes:
        success: Whether the query completed successfully
        data: The actual query results (list, dict, or None)
        count: Number of results returned
        query: The original query string
        source: Name of the database (e.g., "UniProt", "STRING")
        message: Human-readable status message
        metadata: Additional information about the query
        timestamp: When the query was executed
    """
    success: bool
    data: Any
    count: int
    query: str
    source: str
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "data": self.data,
            "count": self.count,
            "query": self.query,
            "source": self.source,
            "message": self.message,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }
    
    def __repr__(self) -> str:
        return f"DatabaseResult(source={self.source}, success={self.success}, count={self.count})"
    
    def is_empty(self) -> bool:
        """Check if the result has no data."""
        return self.count == 0 or self.data is None or (isinstance(self.data, list) and len(self.data) == 0)


# =============================================================================
# Base Client
# =============================================================================

class DatabaseClient(ABC):
    """
    Abstract base class for database API clients.
    
    Provides:
    - Rate limiting to respect API quotas
    - Request caching (optional)
    - Standardized error handling
    - Common HTTP request patterns
    
    Subclasses must implement:
    - search(): Search the database
    - get_by_id(): Retrieve a specific entry
    """
    
    # Class-level configuration (override in subclasses)
    BASE_URL: str = ""
    NAME: str = "Database"
    RATE_LIMIT: float = 3.0  # requests per second (default: 3/s)
    DEFAULT_TIMEOUT: float = 30.0
    
    def __init__(
        self,
        timeout: float = None,
        enable_cache: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the database client.
        
        Args:
            timeout: Request timeout in seconds
            enable_cache: Whether to cache responses
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.enable_cache = enable_cache
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Rate limiting state
        self._last_request_time = 0.0
        self._rate_lock = threading.Lock()
        
        # Simple response cache
        self._cache: Dict[str, Any] = {}
        self._cache_lock = threading.Lock()
        
        # HTTP client (lazy initialization)
        self._client = None
        
        logger.debug(f"Initialized {self.NAME} client")
    
    @property
    def client(self):
        """Get or create HTTP client."""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.Client(timeout=self.timeout)
            except ImportError:
                import urllib.request
                # Fallback to urllib if httpx not available
                self._client = None
        return self._client
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        with self._rate_lock:
            if self.RATE_LIMIT <= 0:
                return
            
            min_interval = 1.0 / self.RATE_LIMIT
            elapsed = time.time() - self._last_request_time
            
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
            
            self._last_request_time = time.time()
    
    def _get_cache_key(self, method: str, url: str, **kwargs) -> str:
        """Generate a cache key for a request."""
        import hashlib
        import json
        
        key_parts = [method, url]
        if kwargs:
            key_parts.append(json.dumps(kwargs, sort_keys=True, default=str))
        
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[Any]:
        """Check if a cached response exists."""
        if not self.enable_cache:
            return None
        
        with self._cache_lock:
            return self._cache.get(cache_key)
    
    def _store_cache(self, cache_key: str, data: Any):
        """Store a response in cache."""
        if not self.enable_cache:
            return
        
        with self._cache_lock:
            # Simple size limit
            if len(self._cache) > 1000:
                # Remove oldest half
                keys = list(self._cache.keys())[:500]
                for k in keys:
                    del self._cache[k]
            
            self._cache[cache_key] = data
    
    def _request(
        self,
        method: str,
        url: str,
        params: Dict[str, Any] = None,
        data: Dict[str, Any] = None,
        headers: Dict[str, str] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Make a rate-limited HTTP request with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            params: URL parameters
            data: Request body data
            headers: HTTP headers
            use_cache: Whether to use caching for this request
            
        Returns:
            Response data (parsed JSON or raw text)
            
        Raises:
            Exception: If all retries fail
        """
        # Check cache first
        if use_cache and method.upper() == "GET":
            cache_key = self._get_cache_key(method, url, params=params)
            cached = self._check_cache(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for {url}")
                return cached
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Rate limit
                self._rate_limit()
                
                # Make request
                if self.client is not None:
                    # Using httpx
                    response = self.client.request(
                        method,
                        url,
                        params=params,
                        data=data,
                        headers=headers,
                    )
                    response.raise_for_status()
                    
                    # Try to parse JSON
                    content_type = response.headers.get("content-type", "")
                    if "json" in content_type:
                        result = response.json()
                    else:
                        result = response.text
                else:
                    # Fallback to urllib
                    result = self._urllib_request(method, url, params, data, headers)
                
                # Cache successful GET responses
                if use_cache and method.upper() == "GET":
                    self._store_cache(cache_key, result)
                
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(
                    f"{self.NAME} request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        
        raise last_error or Exception(f"Request to {url} failed after {self.max_retries} attempts")
    
    def _urllib_request(
        self,
        method: str,
        url: str,
        params: Dict[str, Any] = None,
        data: Dict[str, Any] = None,
        headers: Dict[str, str] = None,
    ) -> Union[Dict[str, Any], str]:
        """Fallback request using urllib."""
        import urllib.request
        import urllib.parse
        import json
        
        # Build URL with params
        if params:
            query_string = urllib.parse.urlencode(params)
            url = f"{url}?{query_string}"
        
        # Prepare request
        req = urllib.request.Request(url, method=method)
        
        if headers:
            for key, value in headers.items():
                req.add_header(key, value)
        
        if data:
            req.data = urllib.parse.urlencode(data).encode()
        
        # Make request
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            content = response.read().decode()
            content_type = response.headers.get("content-type", "")
            
            if "json" in content_type:
                return json.loads(content)
            return content
    
    def clear_cache(self):
        """Clear the response cache."""
        with self._cache_lock:
            self._cache.clear()
        logger.info(f"Cleared {self.NAME} client cache")
    
    @abstractmethod
    def search(self, query: str, **kwargs) -> DatabaseResult:
        """
        Search the database.
        
        Args:
            query: Search query string
            **kwargs: Additional search parameters
            
        Returns:
            DatabaseResult with search results
        """
        pass
    
    @abstractmethod
    def get_by_id(self, identifier: str, **kwargs) -> DatabaseResult:
        """
        Retrieve a specific entry by its identifier.
        
        Args:
            identifier: Database-specific identifier
            **kwargs: Additional parameters
            
        Returns:
            DatabaseResult with the entry data
        """
        pass
    
    def _empty_result(self, query: str, message: str = "No results found") -> DatabaseResult:
        """Create an empty result."""
        return DatabaseResult(
            success=True,
            data=[],
            count=0,
            query=query,
            source=self.NAME,
            message=message,
        )
    
    def _error_result(self, query: str, error: Exception) -> DatabaseResult:
        """Create an error result."""
        return DatabaseResult(
            success=False,
            data=None,
            count=0,
            query=query,
            source=self.NAME,
            message=f"Error: {str(error)}",
            metadata={"error_type": type(error).__name__},
        )


# =============================================================================
# Common Organism Mappings
# =============================================================================

ORGANISM_TAXONOMY_MAP = {
    # Human
    "human": "9606",
    "homo sapiens": "9606",
    "h. sapiens": "9606",
    "hs": "9606",
    "hsa": "9606",
    
    # Mouse
    "mouse": "10090",
    "mus musculus": "10090",
    "m. musculus": "10090",
    "mm": "10090",
    "mmu": "10090",
    
    # Rat
    "rat": "10116",
    "rattus norvegicus": "10116",
    "r. norvegicus": "10116",
    "rn": "10116",
    "rno": "10116",
    
    # Zebrafish
    "zebrafish": "7955",
    "danio rerio": "7955",
    "d. rerio": "7955",
    "dr": "7955",
    "dre": "7955",
    
    # Fruit fly
    "drosophila": "7227",
    "drosophila melanogaster": "7227",
    "d. melanogaster": "7227",
    "dm": "7227",
    "dme": "7227",
    "fly": "7227",
    
    # Nematode
    "c. elegans": "6239",
    "caenorhabditis elegans": "6239",
    "worm": "6239",
    "ce": "6239",
    "cel": "6239",
    
    # Yeast
    "yeast": "559292",
    "saccharomyces cerevisiae": "559292",
    "s. cerevisiae": "559292",
    "sc": "559292",
    "sce": "559292",
    
    # E. coli
    "e. coli": "83333",
    "escherichia coli": "83333",
    "ec": "83333",
    "eco": "83333",
    
    # Arabidopsis
    "arabidopsis": "3702",
    "arabidopsis thaliana": "3702",
    "a. thaliana": "3702",
    "at": "3702",
    "ath": "3702",
}


def resolve_taxonomy_id(organism: str) -> str:
    """
    Resolve an organism name to a taxonomy ID.
    
    Args:
        organism: Organism name (human, mouse, etc.) or taxonomy ID
        
    Returns:
        NCBI taxonomy ID as string
    """
    # If already a number, return as-is
    if organism.isdigit():
        return organism
    
    # Look up in mapping
    organism_lower = organism.lower().strip()
    if organism_lower in ORGANISM_TAXONOMY_MAP:
        return ORGANISM_TAXONOMY_MAP[organism_lower]
    
    # Return original (might be a valid taxonomy ID or abbreviation)
    return organism
