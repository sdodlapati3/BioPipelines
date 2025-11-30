"""
Search Routes
=============

Endpoints for searching biological databases.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from workflow_composer.auth.dependencies import check_rate_limit, optional_api_key
from workflow_composer.auth.models import AuthResult

from ..config import get_api_config

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class SearchRequest(BaseModel):
    """Search request model."""
    
    query: str = Field(..., description="Natural language search query")
    databases: Optional[List[str]] = Field(
        None,
        description="Databases to search (e.g., 'ENCODE', 'GEO', 'SRA')"
    )
    organism: Optional[str] = Field(None, description="Filter by organism")
    data_type: Optional[str] = Field(None, description="Filter by data type")
    limit: int = Field(20, ge=1, le=100, description="Maximum results")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "H3K27ac ChIP-seq in human liver",
                "databases": ["ENCODE", "GEO"],
                "limit": 20
            }
        }


class SearchResult(BaseModel):
    """Individual search result."""
    
    id: str
    title: str
    source: str
    organism: Optional[str] = None
    data_type: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Search response model."""
    
    query: str
    results: List[SearchResult]
    total: int
    sources_searched: List[str]
    cached: bool = False


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/search", response_model=SearchResponse)
async def search_databases(
    request: SearchRequest,
    auth: AuthResult = Depends(optional_api_key),
    authorization: Optional[str] = Header(None),
):
    """
    Search biological databases using natural language.
    
    Searches across multiple databases (ENCODE, GEO, SRA, etc.) and
    returns unified results.
    """
    config = get_api_config()
    
    # Check rate limit if authenticated
    if auth.success:
        rate_info = await check_rate_limit(auth)
        if rate_info.get("exceeded"):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers=rate_info.get("headers", {}),
            )
    
    # Check if caching is enabled
    cache_key = None
    if config.enable_caching:
        cache_key = f"search:{request.query}:{request.databases}:{request.organism}:{request.limit}"
        
        try:
            from workflow_composer.infrastructure.semantic_cache import get_cache
            cache = get_cache("search_cache")
            cached_result, _ = cache.get(cache_key)
            
            if cached_result:
                return SearchResponse(
                    query=request.query,
                    results=cached_result["results"],
                    total=cached_result["total"],
                    sources_searched=cached_result["sources"],
                    cached=True,
                )
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")
    
    # Perform search
    try:
        # Use RAG-enhanced search if available
        results = []
        sources_searched = []
        
        if config.enable_rag:
            try:
                from workflow_composer.agents.rag import get_rag_orchestrator
                rag = get_rag_orchestrator()
                
                # Enhance search with RAG
                enhanced = rag.enhance(  # noqa: F841 (TODO: use in search)
                    query=request.query,
                    candidate_tools=request.databases or ["ENCODE", "GEO"],
                    base_args={
                        "organism": request.organism,
                        "limit": request.limit,
                    }
                )
                
                # Use enhanced args for search
                # (Actual search would happen here)
                
            except ImportError:
                pass
        
        # Placeholder: return empty results for now
        # Actual implementation would call database search modules
        
        response_data = {
            "results": results,
            "total": len(results),
            "sources": sources_searched or request.databases or [],
        }
        
        # Cache the results
        if config.enable_caching and cache_key:
            try:
                cache.set(cache_key, response_data, ttl=3600)
            except Exception:
                pass
        
        return SearchResponse(
            query=request.query,
            results=results,
            total=len(results),
            sources_searched=sources_searched or request.databases or [],
            cached=False,
        )
        
    except Exception as e:
        logger.exception(f"Search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/search/databases")
async def list_available_databases():
    """
    List available databases that can be searched.
    """
    return {
        "databases": [
            {
                "id": "ENCODE",
                "name": "ENCODE",
                "description": "Encyclopedia of DNA Elements",
                "data_types": ["ChIP-seq", "RNA-seq", "ATAC-seq", "DNase-seq"],
            },
            {
                "id": "GEO",
                "name": "Gene Expression Omnibus",
                "description": "NCBI gene expression database",
                "data_types": ["microarray", "RNA-seq", "ChIP-seq"],
            },
            {
                "id": "SRA",
                "name": "Sequence Read Archive",
                "description": "NCBI sequencing data archive",
                "data_types": ["all sequencing types"],
            },
            {
                "id": "ArrayExpress",
                "name": "ArrayExpress",
                "description": "EBI functional genomics data",
                "data_types": ["microarray", "RNA-seq"],
            },
        ]
    }


@router.get("/search/suggestions")
async def get_search_suggestions(
    q: str = Query(..., min_length=2, description="Partial query for suggestions"),
    limit: int = Query(10, ge=1, le=50),
):
    """
    Get search suggestions based on partial query.
    
    Uses past successful searches to suggest completions.
    """
    config = get_api_config()
    
    suggestions = []
    
    if config.enable_rag:
        try:
            from workflow_composer.agents.rag import get_tool_memory
            memory = get_tool_memory()
            
            # Find similar past queries
            similar = memory.find_similar(
                query=q,
                success_only=True,
                limit=limit,
            )
            
            for record, similarity in similar:
                if similarity > 0.3:
                    suggestions.append({
                        "query": record.query,
                        "similarity": round(similarity, 2),
                    })
                    
        except ImportError:
            pass
    
    return {
        "query": q,
        "suggestions": suggestions[:limit],
    }
