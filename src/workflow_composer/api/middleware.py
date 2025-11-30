"""
API Middleware
=============

Middleware components for the FastAPI application.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RAGMiddleware(BaseHTTPMiddleware):
    """
    Middleware to record API requests for RAG learning.
    
    Records successful API calls to help the RAG system learn
    which tools work best for which queries.
    """
    
    def __init__(self, app, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled
        self._rag = None
    
    @property
    def rag(self):
        """Lazy load RAG orchestrator."""
        if self._rag is None and self.enabled:
            try:
                from workflow_composer.agents.rag import get_rag_orchestrator
                self._rag = get_rag_orchestrator()
            except ImportError:
                logger.warning("RAG module not available")
                self.enabled = False
        return self._rag
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and record for RAG learning."""
        if not self.enabled:
            return await call_next(request)
        
        # Generate request ID
        request_id = str(uuid4())[:8]
        request.state.request_id = request_id
        
        # Time the request
        start_time = time.time()
        
        # Execute the request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Record for learning (non-blocking)
        if response.status_code < 400:
            await self._record_success(request, response, duration_ms)
        
        return response
    
    async def _record_success(
        self,
        request: Request,
        response: Response,
        duration_ms: float,
    ) -> None:
        """Record successful request for RAG learning."""
        try:
            # Only record certain endpoints
            path = request.url.path
            
            if not any(p in path for p in ['/search', '/workflows/generate']):
                return
            
            # Get query from body if available
            query = None
            if hasattr(request.state, 'parsed_body'):
                body = request.state.parsed_body
                query = body.get('query') if isinstance(body, dict) else None
            
            if not query:
                return
            
            # Get user from auth
            user_id = None
            if hasattr(request.state, 'user'):
                user_id = str(request.state.user.id)
            
            # Record the execution
            if self.rag:
                # Determine tool from path
                if '/search' in path:
                    tool_name = 'api_search'
                elif '/workflows' in path:
                    tool_name = 'api_workflow_generate'
                else:
                    tool_name = 'api_unknown'
                
                self.rag.record_execution(
                    query=query,
                    tool_name=tool_name,
                    tool_args={'path': path, 'method': request.method},
                    success=True,
                    duration_ms=duration_ms,
                    user_id=user_id,
                )
                
        except Exception as e:
            logger.debug(f"Failed to record for RAG: {e}")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request/response logging.
    """
    
    def __init__(self, app, log_body: bool = False):
        super().__init__(app)
        self.log_body = log_body
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response."""
        request_id = getattr(request.state, 'request_id', str(uuid4())[:8])
        
        # Log request
        logger.info(
            f"[{request_id}] {request.method} {request.url.path}"
        )
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            duration_ms = (time.time() - start_time) * 1000
            
            logger.info(
                f"[{request_id}] {response.status_code} ({duration_ms:.0f}ms)"
            )
            
            return response
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"[{request_id}] ERROR: {e} ({duration_ms:.0f}ms)"
            )
            raise


class CacheMiddleware(BaseHTTPMiddleware):
    """
    Middleware for response caching.
    
    Caches GET responses based on URL and query parameters.
    """
    
    def __init__(
        self,
        app,
        enabled: bool = True,
        ttl: int = 300,
        cache_paths: Optional[list] = None,
    ):
        super().__init__(app)
        self.enabled = enabled
        self.ttl = ttl
        self.cache_paths = cache_paths or ['/api/v1/search']
        self._cache = None
    
    @property
    def cache(self):
        """Lazy load cache."""
        if self._cache is None and self.enabled:
            try:
                from workflow_composer.infrastructure.semantic_cache import get_cache
                self._cache = get_cache("api_cache", default_ttl=self.ttl)
            except ImportError:
                self.enabled = False
        return self._cache
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check cache and serve or store response."""
        if not self.enabled:
            return await call_next(request)
        
        # Only cache GET requests to specific paths
        if request.method != "GET":
            return await call_next(request)
        
        path = request.url.path
        if not any(p in path for p in self.cache_paths):
            return await call_next(request)
        
        # Generate cache key
        cache_key = f"{path}:{request.url.query}"
        
        # Check cache
        if self.cache:
            cached, _ = self.cache.get(cache_key)
            if cached:
                logger.debug(f"Cache hit: {cache_key}")
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    content=cached,
                    headers={"X-Cache": "HIT"},
                )
        
        # Execute request
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200 and self.cache:
            # Note: Caching full response requires reading body
            # This is a simplified implementation
            pass
        
        return response


def setup_middleware(app, config=None):
    """
    Set up all middleware for the FastAPI application.
    
    Args:
        app: FastAPI application
        config: API configuration
    """
    from .config import get_api_config
    config = config or get_api_config()
    
    # Add middleware (order matters - first added is outermost)
    
    # 1. Request logging (outermost)
    app.add_middleware(RequestLoggingMiddleware)
    
    # 2. Cache middleware
    if config.enable_caching:
        app.add_middleware(
            CacheMiddleware,
            enabled=True,
            ttl=300,
        )
    
    # 3. RAG learning middleware (innermost)
    if config.enable_rag:
        app.add_middleware(
            RAGMiddleware,
            enabled=True,
        )
    
    logger.info("API middleware configured")
