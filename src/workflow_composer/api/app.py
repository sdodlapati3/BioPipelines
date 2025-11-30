"""
FastAPI Application Factory
==========================

Creates and configures the FastAPI application.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import APIConfig, get_api_config

logger = logging.getLogger(__name__)


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting BioPipelines API...")
    
    config = get_api_config()
    
    # Initialize database connection if configured
    if config.database_url:
        try:
            from workflow_composer.db import DatabaseConfig, init_db
            db_config = DatabaseConfig(url=config.database_url)
            init_db(db_config)
            logger.info("Database initialized")
        except ImportError:
            logger.warning("Database module not available")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
    
    # Initialize RAG if enabled
    if config.enable_rag:
        try:
            from workflow_composer.agents.rag import get_rag_orchestrator
            rag = get_rag_orchestrator()
            rag.warm_up()
            logger.info("RAG system initialized")
        except ImportError:
            logger.warning("RAG module not available")
        except Exception as e:
            logger.warning(f"RAG initialization failed: {e}")
    
    logger.info(f"API ready at http://{config.host}:{config.port}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down BioPipelines API...")
    
    # Cleanup database connections
    try:
        from workflow_composer.db.session import close_db
        close_db()
    except Exception:
        pass


# =============================================================================
# Application Factory
# =============================================================================

def create_app(config: APIConfig = None) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        config: Optional configuration (uses environment if not provided)
        
    Returns:
        Configured FastAPI application
    """
    if config is None:
        config = get_api_config()
    
    app = FastAPI(
        title=config.title,
        description=config.description,
        version=config.version,
        docs_url="/docs" if config.debug else "/docs",
        redoc_url="/redoc" if config.debug else None,
        lifespan=lifespan,
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=config.cors_allow_credentials,
        allow_methods=config.cors_allow_methods,
        allow_headers=config.cors_allow_headers,
    )
    
    # Add custom middleware
    from .middleware import setup_middleware
    setup_middleware(app, config)
    
    # Add exception handlers
    add_exception_handlers(app)
    
    # Add routes
    add_routes(app, config)
    
    return app


# =============================================================================
# Exception Handlers
# =============================================================================

def add_exception_handlers(app: FastAPI) -> None:
    """Add custom exception handlers."""
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=400,
            content={"error": str(exc), "type": "validation_error"},
        )
    
    @app.exception_handler(PermissionError)
    async def permission_error_handler(request: Request, exc: PermissionError):
        return JSONResponse(
            status_code=403,
            content={"error": str(exc), "type": "permission_denied"},
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "type": "internal_error",
            },
        )


# =============================================================================
# Route Registration
# =============================================================================

def add_routes(app: FastAPI, config: APIConfig) -> None:
    """Register all route handlers."""
    
    # Import routers
    from .routes import (
        health,
        jobs,
        search,
        workflows,
    )
    
    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(workflows.router, prefix="/api/v1", tags=["Workflows"])
    app.include_router(search.router, prefix="/api/v1", tags=["Search"])
    app.include_router(jobs.router, prefix="/api/v1", tags=["Jobs"])


# =============================================================================
# Default Application Instance
# =============================================================================

# Create default app instance for uvicorn
app = create_app()
