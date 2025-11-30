"""
Health Check Routes
==================

Endpoints for health and readiness checks.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..config import get_api_config

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Basic health check.
    
    Returns 200 if the service is running.
    """
    return {"status": "healthy"}


@router.get("/ready")
async def readiness_check():
    """
    Readiness check for Kubernetes probes.
    
    Checks database connectivity and other dependencies.
    """
    config = get_api_config()
    checks = {
        "api": True,
        "database": False,
        "cache": False,
        "rag": False,
    }
    
    # Check database
    if config.database_url:
        try:
            from workflow_composer.db import DatabaseSession
            db = DatabaseSession.get_instance()
            checks["database"] = db.check_health()
        except Exception:
            pass
    else:
        checks["database"] = None  # Not configured
    
    # Check cache
    if config.enable_caching and config.redis_url:
        try:
            from workflow_composer.infrastructure.redis_cache import get_redis_cache
            cache = get_redis_cache()
            checks["cache"] = cache.redis is not None
        except Exception:
            pass
    else:
        checks["cache"] = None  # Not configured
    
    # Check RAG
    if config.enable_rag:
        try:
            from workflow_composer.agents.rag import get_rag_orchestrator
            rag = get_rag_orchestrator()
            checks["rag"] = True
        except Exception:
            pass
    else:
        checks["rag"] = None  # Not configured
    
    # Determine overall status
    required_checks = ["api"]
    if config.database_url:
        required_checks.append("database")
    
    all_ready = all(
        checks.get(check, False) 
        for check in required_checks
    )
    
    status_code = 200 if all_ready else 503
    
    return JSONResponse(
        status_code=status_code,
        content={
            "ready": all_ready,
            "checks": checks,
        }
    )


@router.get("/info")
async def service_info():
    """
    Get service information.
    """
    config = get_api_config()
    
    return {
        "name": config.title,
        "version": config.version,
        "features": {
            "rag": config.enable_rag,
            "caching": config.enable_caching,
            "job_queue": config.enable_job_queue,
            "rate_limiting": config.rate_limit_enabled,
        },
    }
