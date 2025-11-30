"""
Celery Task Definitions.

Defines async tasks for workflow operations:
- Workflow generation
- Tool search
- Workflow validation
- Workflow execution
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from celery import Task, shared_task

logger = logging.getLogger(__name__)


class BaseTaskWithRetry(Task):
    """Base task class with retry logic and error handling."""
    
    autoretry_for = (Exception,)
    retry_kwargs = {"max_retries": 3}
    retry_backoff = True
    retry_backoff_max = 600
    retry_jitter = True
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        logger.error(
            f"Task {self.name}[{task_id}] failed: {exc}",
            extra={
                "task_id": task_id,
                "task_name": self.name,
                "args": args,
                "kwargs": kwargs,
                "exception": str(exc),
                "traceback": str(einfo),
            }
        )
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success."""
        logger.info(
            f"Task {self.name}[{task_id}] completed successfully",
            extra={
                "task_id": task_id,
                "task_name": self.name,
            }
        )
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry."""
        logger.warning(
            f"Task {self.name}[{task_id}] retrying: {exc}",
            extra={
                "task_id": task_id,
                "task_name": self.name,
                "retry_count": self.request.retries,
            }
        )


@shared_task(bind=True, base=BaseTaskWithRetry, name="workflow.generate")
def generate_workflow_task(
    self,
    query: str,
    user_id: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate a workflow from a natural language query.
    
    Args:
        query: Natural language workflow description
        user_id: Optional user identifier
        options: Optional generation options
        
    Returns:
        Dict containing:
        - workflow_id: Generated workflow identifier
        - workflow_code: Generated workflow script
        - workflow_type: Detected workflow type
        - tools: List of selected tools
        - metadata: Additional workflow metadata
    """
    from src.workflow_composer.agents.query_parser import get_intent_parser
    from src.workflow_composer.agents.tool_selector import get_tool_selector
    from src.workflow_composer.agents.workflow_generator import get_workflow_generator

    from src.workflow_composer.agents.rag import get_rag_orchestrator
    
    options = options or {}
    start_time = datetime.utcnow()
    
    try:
        # Parse intent
        parser = get_intent_parser()
        intent = parser.parse(query)
        
        # Select tools
        selector = get_tool_selector()
        tools = selector.select(intent)
        
        # Generate workflow
        generator = get_workflow_generator()
        workflow = generator.generate(intent, tools)
        
        # Record in RAG
        rag = get_rag_orchestrator()
        rag.record(
            query=query,
            workflow_type=intent.workflow_type,
            tools=[t.name for t in tools],
            execution_time=(datetime.utcnow() - start_time).total_seconds(),
            success=True,
        )
        
        return {
            "workflow_id": f"wf_{self.request.id}",
            "workflow_code": workflow.code if hasattr(workflow, 'code') else str(workflow),
            "workflow_type": intent.workflow_type,
            "tools": [t.name for t in tools],
            "organism": intent.organism,
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "user_id": user_id,
                "options": options,
                "generation_time_seconds": (datetime.utcnow() - start_time).total_seconds(),
            },
            "status": "completed",
        }
        
    except Exception as e:
        logger.error(f"Workflow generation failed: {e}")
        
        # Record failure in RAG
        try:
            rag = get_rag_orchestrator()
            rag.record(
                query=query,
                workflow_type="unknown",
                tools=[],
                execution_time=(datetime.utcnow() - start_time).total_seconds(),
                success=False,
            )
        except Exception:
            pass
        
        raise


@shared_task(bind=True, base=BaseTaskWithRetry, name="search.tools")
def search_tools_task(
    self,
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Search for bioinformatics tools.
    
    Args:
        query: Search query
        filters: Optional search filters
        limit: Maximum results to return
        
    Returns:
        Dict containing:
        - tools: List of matching tools
        - total: Total number of matches
        - query_info: Parsed query information
    """
    from src.workflow_composer.infrastructure.semantic_cache import SemanticCache
    
    filters = filters or {}
    cache = SemanticCache()
    
    try:
        # Check cache first
        cache_key = f"search:{query}:{str(filters)}:{limit}"
        cached = cache.get(cache_key)
        if cached:
            return {**cached, "from_cache": True}
        
        # Perform search (simplified - would integrate with actual tool catalog)
        # This is a placeholder for the actual search implementation
        results = {
            "tools": [],
            "total": 0,
            "query_info": {
                "original_query": query,
                "filters": filters,
                "limit": limit,
            },
            "from_cache": False,
        }
        
        # Cache results
        cache.set(cache_key, results, ttl=3600)
        
        return results
        
    except Exception as e:
        logger.error(f"Tool search failed: {e}")
        raise


@shared_task(bind=True, base=BaseTaskWithRetry, name="validation.workflow")
def validate_workflow_task(
    self,
    workflow_code: str,
    workflow_type: str = "nextflow",
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Validate a workflow script.
    
    Args:
        workflow_code: Workflow script to validate
        workflow_type: Type of workflow (nextflow, snakemake)
        strict: Whether to perform strict validation
        
    Returns:
        Dict containing:
        - valid: Whether workflow is valid
        - errors: List of validation errors
        - warnings: List of validation warnings
        - suggestions: Improvement suggestions
    """
    try:
        # Try to use preflight validator if available
        try:
            from src.workflow_composer.agents.preflight_validator import (
                PreflightValidator,
            )
            validator = PreflightValidator()
            result = validator.validate(workflow_code, workflow_type)
            return {
                "valid": result.is_valid if hasattr(result, 'is_valid') else True,
                "errors": result.errors if hasattr(result, 'errors') else [],
                "warnings": result.warnings if hasattr(result, 'warnings') else [],
                "suggestions": result.suggestions if hasattr(result, 'suggestions') else [],
                "workflow_type": workflow_type,
                "validated_at": datetime.utcnow().isoformat(),
            }
        except ImportError:
            # Fallback: basic validation
            errors = []
            warnings = []
            
            # Basic syntax checks
            if not workflow_code or not workflow_code.strip():
                errors.append("Workflow code is empty")
            elif workflow_type == "nextflow":
                if "process" not in workflow_code and "workflow" not in workflow_code:
                    warnings.append("No process or workflow blocks detected")
            elif workflow_type == "snakemake":
                if "rule" not in workflow_code:
                    warnings.append("No rule blocks detected")
            
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "suggestions": [],
                "workflow_type": workflow_type,
                "validated_at": datetime.utcnow().isoformat(),
            }
    except Exception as e:
        logger.error(f"Workflow validation failed: {e}")
        return {
            "valid": False,
            "errors": [str(e)],
            "warnings": [],
            "suggestions": [],
            "workflow_type": workflow_type,
            "validated_at": datetime.utcnow().isoformat(),
        }


@shared_task(bind=True, base=BaseTaskWithRetry, name="execution.workflow")
def execute_workflow_task(
    self,
    workflow_id: str,
    workflow_code: str,
    execution_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute a workflow (prepare for execution).
    
    Note: Actual execution would be handled by the workflow engine.
    This task prepares the workflow and returns execution metadata.
    
    Args:
        workflow_id: Workflow identifier
        workflow_code: Workflow script
        execution_params: Execution parameters
        
    Returns:
        Dict containing:
        - execution_id: Execution identifier
        - status: Execution status
        - prepared_at: Preparation timestamp
        - config: Execution configuration
    """
    import uuid
    
    execution_params = execution_params or {}
    
    try:
        execution_id = str(uuid.uuid4())
        
        # Prepare execution configuration
        config = {
            "workflow_id": workflow_id,
            "execution_id": execution_id,
            "params": execution_params,
            "prepared_at": datetime.utcnow().isoformat(),
        }
        
        # In a real implementation, this would:
        # 1. Write workflow to execution directory
        # 2. Set up input/output paths
        # 3. Configure compute resources
        # 4. Submit to workflow engine (Nextflow/Snakemake)
        
        return {
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "status": "prepared",
            "prepared_at": datetime.utcnow().isoformat(),
            "config": config,
            "message": "Workflow prepared for execution",
        }
        
    except Exception as e:
        logger.error(f"Workflow execution preparation failed: {e}")
        raise


@shared_task(name="maintenance.cleanup")
def cleanup_expired_jobs() -> Dict[str, Any]:
    """
    Cleanup expired job results and temporary files.
    Runs periodically via Celery beat.
    
    Returns:
        Dict containing cleanup statistics
    """
    from src.workflow_composer.jobs.celery_app import celery_app
    
    try:
        # Get backend
        _backend = celery_app.backend  # noqa: F841 (for future cleanup logic)
        
        # In a real implementation, this would:
        # 1. Clean up expired results from Redis
        # 2. Remove temporary workflow files
        # 3. Archive old job metadata
        
        cleaned_count = 0
        
        return {
            "status": "completed",
            "cleaned_count": cleaned_count,
            "cleaned_at": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Cleanup task failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "cleaned_at": datetime.utcnow().isoformat(),
        }


# Task status helpers
def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get the status of a task by ID."""
    from celery.result import AsyncResult

    from src.workflow_composer.jobs.celery_app import celery_app
    
    result = AsyncResult(task_id, app=celery_app)
    
    return {
        "task_id": task_id,
        "status": result.status,
        "ready": result.ready(),
        "successful": result.successful() if result.ready() else None,
        "result": result.result if result.ready() and result.successful() else None,
        "error": str(result.result) if result.ready() and result.failed() else None,
        "traceback": result.traceback if result.failed() else None,
    }


def revoke_task(task_id: str, terminate: bool = False) -> Dict[str, Any]:
    """Revoke/cancel a pending task."""
    from src.workflow_composer.jobs.celery_app import celery_app
    
    celery_app.control.revoke(task_id, terminate=terminate)
    
    return {
        "task_id": task_id,
        "revoked": True,
        "terminated": terminate,
    }
