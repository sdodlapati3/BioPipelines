"""
FastAPI Application for BioPipelines Workflow Composer
======================================================

REST API for workflow composition, execution, and management.

Features:
- Workflow generation from natural language queries
- Job management and monitoring
- Data search across biological databases
- Tool execution and caching

Usage:
    # Run with uvicorn
    uvicorn workflow_composer.api:app --reload
    
    # Or programmatically
    from workflow_composer.api import create_app
    app = create_app()
"""

from .app import app, create_app
from .config import APIConfig, get_api_config

__all__ = [
    "app",
    "create_app",
    "APIConfig",
    "get_api_config",
]
