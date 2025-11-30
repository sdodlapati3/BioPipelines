"""
Job Routes
==========

Endpoints for job management and monitoring.

Jobs are long-running workflow executions that can be:
- Created and queued via Celery
- Monitored for progress
- Cancelled if needed
- Retrieved with results

Integrates with Celery for async task processing.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Header, Query
from pydantic import BaseModel, Field

from workflow_composer.auth.models import AuthResult, KeyScope
from workflow_composer.auth.dependencies import require_api_key, require_scope

logger = logging.getLogger(__name__)

router = APIRouter()

# Check if Celery is available
CELERY_ENABLED = os.environ.get("CELERY_ENABLED", "false").lower() == "true"


# =============================================================================
# Request/Response Models
# =============================================================================

class JobStatus:
    """Job status constants."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobCreateRequest(BaseModel):
    """Request to create a new job."""
    
    workflow_id: str = Field(..., description="ID of the workflow to execute")
    name: Optional[str] = Field(None, description="Job name")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Job parameters")
    priority: int = Field(5, ge=1, le=10, description="Job priority (1=lowest, 10=highest)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "workflow_id": "wf_abc123",
                "name": "RNA-seq analysis batch 1",
                "parameters": {
                    "input_files": ["sample1.fastq", "sample2.fastq"],
                    "reference_genome": "hg38",
                },
                "priority": 5,
            }
        }


class JobResponse(BaseModel):
    """Job response model."""
    
    id: str
    workflow_id: str
    name: str
    status: str
    progress: float = 0.0
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class JobListResponse(BaseModel):
    """Response listing jobs."""
    
    jobs: List[JobResponse]
    total: int
    page: int
    page_size: int


# =============================================================================
# In-Memory Job Store + Celery Integration
# =============================================================================

# In-memory store for job metadata (complements Celery task state)
_jobs: Dict[str, Dict[str, Any]] = {}


def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get job from local store or Celery backend."""
    local_job = _jobs.get(job_id)
    
    if CELERY_ENABLED and local_job and local_job.get("celery_task_id"):
        # Update status from Celery
        try:
            from src.workflow_composer.jobs.tasks import get_task_status
            task_status = get_task_status(local_job["celery_task_id"])
            
            # Map Celery status to job status
            status_map = {
                "PENDING": JobStatus.PENDING,
                "STARTED": JobStatus.RUNNING,
                "SUCCESS": JobStatus.COMPLETED,
                "FAILURE": JobStatus.FAILED,
                "REVOKED": JobStatus.CANCELLED,
            }
            
            local_job["status"] = status_map.get(task_status["status"], local_job["status"])
            
            if task_status["ready"]:
                local_job["completed_at"] = datetime.utcnow()
                if task_status["successful"]:
                    local_job["result"] = task_status["result"]
                    local_job["progress"] = 100.0
                elif task_status["error"]:
                    local_job["error"] = task_status["error"]
        except Exception as e:
            logger.warning(f"Failed to get Celery task status: {e}")
    
    return local_job


def _create_job(user_id: UUID, request: JobCreateRequest) -> Dict[str, Any]:
    """Create job and optionally submit to Celery."""
    job_id = str(uuid4())
    
    job = {
        "id": job_id,
        "user_id": str(user_id),
        "workflow_id": request.workflow_id,
        "name": request.name or f"Job {len(_jobs) + 1}",
        "parameters": request.parameters,
        "priority": request.priority,
        "status": JobStatus.PENDING,
        "progress": 0.0,
        "created_at": datetime.utcnow(),
        "started_at": None,
        "completed_at": None,
        "error": None,
        "result": None,
        "celery_task_id": None,
    }
    
    # Submit to Celery if enabled
    if CELERY_ENABLED:
        try:
            from src.workflow_composer.jobs.tasks import execute_workflow_task
            
            # Submit async task
            result = execute_workflow_task.apply_async(
                kwargs={
                    "workflow_id": request.workflow_id,
                    "workflow_code": "",  # Would be fetched from workflow store
                    "execution_params": request.parameters,
                },
                task_id=job_id,
            )
            
            job["celery_task_id"] = result.id
            job["status"] = JobStatus.PENDING
            logger.info(f"Submitted job {job_id} to Celery as task {result.id}")
            
        except Exception as e:
            logger.error(f"Failed to submit to Celery: {e}")
            job["error"] = f"Failed to queue job: {str(e)}"
            job["status"] = JobStatus.FAILED
    else:
        # Development mode - use sync executor
        try:
            from src.workflow_composer.jobs.celery_app import get_sync_executor
            from src.workflow_composer.jobs.tasks import execute_workflow_task
            
            executor = get_sync_executor()
            result = executor.apply_async(
                execute_workflow_task,
                kwargs={
                    "workflow_id": request.workflow_id,
                    "workflow_code": "",
                    "execution_params": request.parameters,
                },
                task_id=job_id,
            )
            
            job["celery_task_id"] = result.id
            job["status"] = JobStatus.COMPLETED if result.successful() else JobStatus.FAILED
            job["result"] = result.result if result.successful() else None
            job["error"] = result.error if result.failed() else None
            job["completed_at"] = datetime.utcnow()
            job["progress"] = 100.0
            
        except Exception as e:
            logger.error(f"Sync execution failed: {e}")
            job["error"] = str(e)
            job["status"] = JobStatus.FAILED
    
    _jobs[job["id"]] = job
    return job


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/jobs", response_model=JobResponse)
async def create_job(
    request: JobCreateRequest,
    auth: AuthResult = Depends(require_scope(KeyScope.EXECUTE)),
    authorization: Optional[str] = Header(None),
):
    """
    Create a new job to execute a workflow.
    
    The job will be queued and executed asynchronously.
    Use GET /jobs/{id} to monitor progress.
    """
    if not auth.success:
        raise HTTPException(status_code=401, detail=auth.error)
    
    # Validate workflow exists (placeholder)
    # In real implementation, check database
    
    job = _create_job(auth.user.id, request)
    
    logger.info(f"Created job {job['id']} for user {auth.user.email}")
    
    return JobResponse(
        id=job["id"],
        workflow_id=job["workflow_id"],
        name=job["name"],
        status=job["status"],
        progress=job["progress"],
        created_at=job["created_at"],
        started_at=job["started_at"],
        completed_at=job["completed_at"],
        error=job["error"],
        result=job["result"],
    )


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    auth: AuthResult = Depends(require_api_key),
    authorization: Optional[str] = Header(None),
):
    """
    List jobs for the authenticated user.
    """
    if not auth.success:
        raise HTTPException(status_code=401, detail=auth.error)
    
    user_id = str(auth.user.id)
    
    # Filter jobs for this user
    user_jobs = [
        j for j in _jobs.values()
        if j["user_id"] == user_id
    ]
    
    # Filter by status if specified
    if status:
        user_jobs = [j for j in user_jobs if j["status"] == status]
    
    # Sort by created_at descending
    user_jobs.sort(key=lambda j: j["created_at"], reverse=True)
    
    # Paginate
    total = len(user_jobs)
    start = (page - 1) * page_size
    end = start + page_size
    page_jobs = user_jobs[start:end]
    
    return JobListResponse(
        jobs=[
            JobResponse(
                id=j["id"],
                workflow_id=j["workflow_id"],
                name=j["name"],
                status=j["status"],
                progress=j["progress"],
                created_at=j["created_at"],
                started_at=j["started_at"],
                completed_at=j["completed_at"],
                error=j["error"],
                result=j["result"],
            )
            for j in page_jobs
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    auth: AuthResult = Depends(require_api_key),
    authorization: Optional[str] = Header(None),
):
    """
    Get a specific job by ID.
    """
    if not auth.success:
        raise HTTPException(status_code=401, detail=auth.error)
    
    job = _get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check ownership
    if job["user_id"] != str(auth.user.id):
        from workflow_composer.auth.models import UserRole
        if auth.user.role != UserRole.ADMIN:
            raise HTTPException(status_code=403, detail="Access denied")
    
    return JobResponse(
        id=job["id"],
        workflow_id=job["workflow_id"],
        name=job["name"],
        status=job["status"],
        progress=job["progress"],
        created_at=job["created_at"],
        started_at=job["started_at"],
        completed_at=job["completed_at"],
        error=job["error"],
        result=job["result"],
    )


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    auth: AuthResult = Depends(require_api_key),
    authorization: Optional[str] = Header(None),
):
    """
    Cancel a running or pending job.
    """
    if not auth.success:
        raise HTTPException(status_code=401, detail=auth.error)
    
    job = _get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check ownership
    if job["user_id"] != str(auth.user.id):
        from workflow_composer.auth.models import UserRole
        if auth.user.role != UserRole.ADMIN:
            raise HTTPException(status_code=403, detail="Access denied")
    
    # Check if cancellable
    if job["status"] not in [JobStatus.PENDING, JobStatus.RUNNING]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {job['status']}"
        )
    
    # Cancel the job
    job["status"] = JobStatus.CANCELLED
    job["completed_at"] = datetime.utcnow()
    
    # Revoke Celery task if present
    if CELERY_ENABLED and job.get("celery_task_id"):
        try:
            from src.workflow_composer.jobs.tasks import revoke_task
            revoke_task(job["celery_task_id"], terminate=True)
            logger.info(f"Revoked Celery task {job['celery_task_id']}")
        except Exception as e:
            logger.warning(f"Failed to revoke Celery task: {e}")
    
    logger.info(f"Cancelled job {job_id}")
    
    return {"status": "cancelled", "job_id": job_id}


@router.get("/jobs/{job_id}/logs")
async def get_job_logs(
    job_id: str,
    tail: int = Query(100, ge=1, le=1000, description="Number of lines from end"),
    auth: AuthResult = Depends(require_api_key),
    authorization: Optional[str] = Header(None),
):
    """
    Get logs for a job.
    """
    if not auth.success:
        raise HTTPException(status_code=401, detail=auth.error)
    
    job = _get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check ownership
    if job["user_id"] != str(auth.user.id):
        from workflow_composer.auth.models import UserRole
        if auth.user.role != UserRole.ADMIN:
            raise HTTPException(status_code=403, detail="Access denied")
    
    # Placeholder - logs would come from job execution system
    return {
        "job_id": job_id,
        "logs": [],
        "truncated": False,
    }


# =============================================================================
# Async Workflow Generation Endpoint
# =============================================================================

class AsyncWorkflowRequest(BaseModel):
    """Request for async workflow generation."""
    
    query: str = Field(..., description="Natural language workflow description")
    name: Optional[str] = Field(None, description="Job name")
    options: Dict[str, Any] = Field(default_factory=dict, description="Generation options")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "RNA-seq differential expression analysis for human samples",
                "name": "RNA-seq workflow generation",
                "options": {"output_format": "nextflow"},
            }
        }


class AsyncWorkflowResponse(BaseModel):
    """Response for async workflow generation."""
    
    job_id: str
    status: str
    message: str
    poll_url: str


@router.post("/jobs/workflows/generate", response_model=AsyncWorkflowResponse)
async def generate_workflow_async(
    request: AsyncWorkflowRequest,
    auth: AuthResult = Depends(require_scope(KeyScope.WRITE)),
    authorization: Optional[str] = Header(None),
):
    """
    Generate a workflow asynchronously.
    
    Submits workflow generation to the job queue and returns
    immediately with a job ID for polling.
    
    Use GET /jobs/{job_id} to poll for completion.
    """
    if not auth.success:
        raise HTTPException(status_code=401, detail=auth.error)
    
    job_id = str(uuid4())
    
    job = {
        "id": job_id,
        "user_id": str(auth.user.id),
        "workflow_id": None,  # Will be set after generation
        "name": request.name or f"Workflow: {request.query[:50]}...",
        "parameters": {"query": request.query, "options": request.options},
        "priority": 5,
        "status": JobStatus.PENDING,
        "progress": 0.0,
        "created_at": datetime.utcnow(),
        "started_at": None,
        "completed_at": None,
        "error": None,
        "result": None,
        "celery_task_id": None,
        "job_type": "workflow_generation",
    }
    
    if CELERY_ENABLED:
        try:
            from src.workflow_composer.jobs.tasks import generate_workflow_task
            
            result = generate_workflow_task.apply_async(
                kwargs={
                    "query": request.query,
                    "user_id": str(auth.user.id),
                    "options": request.options,
                },
                task_id=job_id,
            )
            
            job["celery_task_id"] = result.id
            logger.info(f"Submitted async workflow generation {job_id}")
            
        except Exception as e:
            logger.error(f"Failed to submit workflow generation: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to queue workflow generation: {str(e)}"
            )
    else:
        # Sync execution for development
        try:
            from src.workflow_composer.jobs.celery_app import get_sync_executor
            from src.workflow_composer.jobs.tasks import generate_workflow_task
            
            executor = get_sync_executor()
            
            # For sync mode, we need to call the actual function
            # not the Celery task wrapper
            result = executor.apply_async(
                lambda **kw: generate_workflow_task.run(**kw),
                kwargs={
                    "query": request.query,
                    "user_id": str(auth.user.id),
                    "options": request.options,
                },
                task_id=job_id,
            )
            
            job["celery_task_id"] = result.id
            job["status"] = JobStatus.COMPLETED if result.successful() else JobStatus.FAILED
            job["result"] = result.result if result.successful() else None
            job["error"] = result.error if result.failed() else None
            job["completed_at"] = datetime.utcnow()
            job["progress"] = 100.0
            
        except Exception as e:
            logger.error(f"Sync workflow generation failed: {e}")
            job["status"] = JobStatus.FAILED
            job["error"] = str(e)
    
    _jobs[job_id] = job
    
    return AsyncWorkflowResponse(
        job_id=job_id,
        status=job["status"],
        message="Workflow generation queued" if job["status"] == JobStatus.PENDING else f"Workflow generation {job['status']}",
        poll_url=f"/api/v1/jobs/{job_id}",
    )


@router.get("/jobs/queue/status")
async def get_queue_status(
    auth: AuthResult = Depends(require_api_key),
    authorization: Optional[str] = Header(None),
):
    """
    Get the status of the job queue.
    
    Returns queue lengths and worker status.
    """
    if not auth.success:
        raise HTTPException(status_code=401, detail=auth.error)
    
    if not CELERY_ENABLED:
        return {
            "celery_enabled": False,
            "mode": "synchronous",
            "message": "Running in synchronous mode (no job queue)",
        }
    
    try:
        from src.workflow_composer.jobs.celery_app import CeleryHealthCheck
        
        health = CeleryHealthCheck()
        status = health.ping()
        queue_lengths = health.get_queue_lengths()
        
        return {
            "celery_enabled": True,
            "mode": "asynchronous",
            "worker_status": status,
            "queue_lengths": queue_lengths,
        }
        
    except Exception as e:
        return {
            "celery_enabled": True,
            "mode": "asynchronous",
            "error": str(e),
        }
