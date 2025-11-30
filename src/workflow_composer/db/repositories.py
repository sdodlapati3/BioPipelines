"""Repository pattern for database access.

Provides a clean abstraction layer between the application and database.
Each repository handles CRUD operations for a specific model.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_

from .models import (
    User, 
    APIKey, 
    Job, 
    WorkflowGeneration, 
    ToolExecution,
    UserTier,
    JobStatus,
    WorkflowType,
)


# =============================================================================
# User Repository
# =============================================================================

class UserRepository:
    """Repository for User model operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create(
        self,
        email: str,
        name: Optional[str] = None,
        tier: UserTier = UserTier.FREE,
    ) -> User:
        """Create a new user."""
        user = User(
            email=email,
            name=name,
            tier=tier,
        )
        self.session.add(user)
        self.session.flush()  # Get ID without committing
        return user
    
    def get_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID."""
        return self.session.query(User).filter(User.id == user_id).first()
    
    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return self.session.query(User).filter(User.email == email).first()
    
    def get_by_api_key(self, api_key: str) -> Optional[User]:
        """Get user by API key."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        api_key_obj = (
            self.session.query(APIKey)
            .filter(APIKey.key_hash == key_hash)
            .first()
        )
        if api_key_obj and api_key_obj.is_valid():
            # Update last used timestamp
            api_key_obj.last_used_at = datetime.utcnow()
            return api_key_obj.user
        return None
    
    def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
        active_only: bool = True,
    ) -> List[User]:
        """List all users with pagination."""
        query = self.session.query(User)
        if active_only:
            query = query.filter(User.is_active == True)
        return query.offset(offset).limit(limit).all()
    
    def update(self, user: User, **kwargs) -> User:
        """Update user attributes."""
        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)
        user.updated_at = datetime.utcnow()
        return user
    
    def delete(self, user: User) -> None:
        """Delete user (soft delete by deactivating)."""
        user.is_active = False
        user.updated_at = datetime.utcnow()
    
    def create_api_key(
        self,
        user: User,
        name: str = "default",
        scopes: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None,
    ) -> Tuple[str, APIKey]:
        """Create a new API key for user.
        
        Returns:
            Tuple of (raw_key, APIKey object). The raw key is only
            available at creation time.
        """
        raw_key = f"bp_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        api_key = APIKey(
            user_id=user.id,
            key_hash=key_hash,
            name=name,
            scopes=scopes or [],
            expires_at=expires_at,
        )
        self.session.add(api_key)
        self.session.flush()
        
        return raw_key, api_key
    
    def revoke_api_key(self, key_id: UUID) -> bool:
        """Revoke an API key."""
        api_key = self.session.query(APIKey).filter(APIKey.id == key_id).first()
        if api_key:
            api_key.is_active = False
            return True
        return False
    
    def get_usage_stats(self, user: User) -> Dict[str, Any]:
        """Get usage statistics for a user."""
        job_count = self.session.query(func.count(Job.id)).filter(
            Job.user_id == user.id
        ).scalar()
        
        workflow_count = self.session.query(func.count(WorkflowGeneration.id)).filter(
            WorkflowGeneration.user_id == user.id
        ).scalar()
        
        query_count = self.session.query(func.count(ToolExecution.id)).filter(
            ToolExecution.user_id == user.id
        ).scalar()
        
        return {
            "jobs_total": job_count,
            "workflows_total": workflow_count,
            "queries_total": query_count,
            "quota_queries_remaining": user.quota_queries,
            "quota_workflows_remaining": user.quota_workflows,
        }


# =============================================================================
# Job Repository
# =============================================================================

class JobRepository:
    """Repository for Job model operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create(
        self,
        user_id: UUID,
        name: Optional[str] = None,
        workflow_type: Optional[WorkflowType] = None,
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        config: Optional[Dict] = None,
        workflow_id: Optional[UUID] = None,
    ) -> Job:
        """Create a new job."""
        job = Job(
            user_id=user_id,
            name=name,
            workflow_type=workflow_type,
            input_dir=input_dir,
            output_dir=output_dir,
            config=config or {},
            workflow_id=workflow_id,
            status=JobStatus.PENDING,
        )
        self.session.add(job)
        self.session.flush()
        return job
    
    def get_by_id(self, job_id: UUID) -> Optional[Job]:
        """Get job by ID."""
        return self.session.query(Job).filter(Job.id == job_id).first()
    
    def get_by_slurm_id(self, slurm_job_id: str) -> Optional[Job]:
        """Get job by SLURM job ID."""
        return self.session.query(Job).filter(Job.slurm_job_id == slurm_job_id).first()
    
    def list_by_user(
        self,
        user_id: UUID,
        status: Optional[JobStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Job]:
        """List jobs for a user."""
        query = self.session.query(Job).filter(Job.user_id == user_id)
        if status:
            query = query.filter(Job.status == status)
        return query.order_by(desc(Job.submitted_at)).offset(offset).limit(limit).all()
    
    def list_active(self) -> List[Job]:
        """List all active (running/pending/queued) jobs."""
        active_statuses = [JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING]
        return (
            self.session.query(Job)
            .filter(Job.status.in_(active_statuses))
            .order_by(Job.submitted_at)
            .all()
        )
    
    def update_status(
        self,
        job: Job,
        status: JobStatus,
        slurm_job_id: Optional[str] = None,
        error_message: Optional[str] = None,
        result: Optional[Dict] = None,
    ) -> Job:
        """Update job status."""
        job.status = status
        
        if slurm_job_id:
            job.slurm_job_id = slurm_job_id
        
        if status == JobStatus.RUNNING and job.started_at is None:
            job.started_at = datetime.utcnow()
        
        if status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            job.completed_at = datetime.utcnow()
        
        if error_message:
            job.error_message = error_message
        
        if result:
            job.result = result
        
        return job
    
    def get_stats(self, user_id: Optional[UUID] = None) -> Dict[str, Any]:
        """Get job statistics."""
        query = self.session.query(Job)
        if user_id:
            query = query.filter(Job.user_id == user_id)
        
        stats = {
            "total": query.count(),
            "by_status": {},
            "avg_duration_seconds": None,
        }
        
        for status in JobStatus:
            stats["by_status"][status.value] = query.filter(
                Job.status == status
            ).count()
        
        # Average duration for completed jobs
        completed = query.filter(
            and_(
                Job.status == JobStatus.COMPLETED,
                Job.started_at.isnot(None),
                Job.completed_at.isnot(None),
            )
        ).all()
        
        if completed:
            durations = [j.duration_seconds for j in completed if j.duration_seconds]
            if durations:
                stats["avg_duration_seconds"] = sum(durations) / len(durations)
        
        return stats


# =============================================================================
# Workflow Repository
# =============================================================================

class WorkflowRepository:
    """Repository for WorkflowGeneration model operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create(
        self,
        user_id: UUID,
        query: str,
        workflow_type: WorkflowType,
        workflow_path: Optional[str] = None,
        workflow_name: Optional[str] = None,
        config: Optional[Dict] = None,
        generation_time_ms: Optional[float] = None,
        llm_model: Optional[str] = None,
        llm_tokens_used: Optional[int] = None,
    ) -> WorkflowGeneration:
        """Create a new workflow generation record."""
        workflow = WorkflowGeneration(
            user_id=user_id,
            query=query,
            workflow_type=workflow_type,
            workflow_path=workflow_path,
            workflow_name=workflow_name,
            config=config or {},
            generation_time_ms=generation_time_ms,
            llm_model=llm_model,
            llm_tokens_used=llm_tokens_used,
        )
        self.session.add(workflow)
        self.session.flush()
        return workflow
    
    def get_by_id(self, workflow_id: UUID) -> Optional[WorkflowGeneration]:
        """Get workflow by ID."""
        return self.session.query(WorkflowGeneration).filter(
            WorkflowGeneration.id == workflow_id
        ).first()
    
    def list_by_user(
        self,
        user_id: UUID,
        workflow_type: Optional[WorkflowType] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[WorkflowGeneration]:
        """List workflows for a user."""
        query = self.session.query(WorkflowGeneration).filter(
            WorkflowGeneration.user_id == user_id
        )
        if workflow_type:
            query = query.filter(WorkflowGeneration.workflow_type == workflow_type)
        return query.order_by(desc(WorkflowGeneration.created_at)).offset(offset).limit(limit).all()
    
    def get_recent(self, limit: int = 10) -> List[WorkflowGeneration]:
        """Get recent workflow generations."""
        return (
            self.session.query(WorkflowGeneration)
            .order_by(desc(WorkflowGeneration.created_at))
            .limit(limit)
            .all()
        )


# =============================================================================
# Tool Execution Repository (RAG Layer 1)
# =============================================================================

class ToolExecutionRepository:
    """Repository for ToolExecution model operations.
    
    This is the foundation for RAG Layers 2 and 3.
    """
    
    def __init__(self, session: Session):
        self.session = session
    
    def record(
        self,
        tool_name: str,
        query: str,
        success: bool,
        duration_ms: float,
        tool_args: Optional[Dict] = None,
        result_summary: Optional[str] = None,
        error_message: Optional[str] = None,
        result_count: Optional[int] = None,
        user_id: Optional[UUID] = None,
        task_type: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> ToolExecution:
        """Record a tool execution."""
        execution = ToolExecution(
            user_id=user_id,
            tool_name=tool_name,
            query=query,
            query_normalized=ToolExecution.normalize_query(query),
            tool_args=tool_args or {},
            success=success,
            result_summary=result_summary,
            error_message=error_message,
            duration_ms=duration_ms,
            result_count=result_count,
            task_type=task_type,
            session_id=session_id,
        )
        self.session.add(execution)
        self.session.flush()
        return execution
    
    def add_feedback(
        self,
        execution_id: int,
        feedback: str,
    ) -> bool:
        """Add user feedback to an execution."""
        execution = self.session.query(ToolExecution).filter(
            ToolExecution.id == execution_id
        ).first()
        if execution:
            execution.user_feedback = feedback
            return True
        return False
    
    def get_tool_stats(
        self,
        tool_name: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get statistics for tools."""
        query = self.session.query(ToolExecution)
        if tool_name:
            query = query.filter(ToolExecution.tool_name == tool_name)
        if since:
            query = query.filter(ToolExecution.created_at >= since)
        
        total = query.count()
        successful = query.filter(ToolExecution.success == True).count()
        
        avg_duration = self.session.query(
            func.avg(ToolExecution.duration_ms)
        ).filter(ToolExecution.tool_name == tool_name if tool_name else True).scalar()
        
        return {
            "total_executions": total,
            "successful_executions": successful,
            "success_rate": successful / total if total > 0 else 0,
            "avg_duration_ms": float(avg_duration) if avg_duration else 0,
        }
    
    def get_all_tool_stats(self, since: Optional[datetime] = None) -> Dict[str, Dict]:
        """Get statistics for all tools."""
        query = self.session.query(
            ToolExecution.tool_name,
            func.count(ToolExecution.id).label("total"),
            func.sum(
                func.cast(ToolExecution.success, Integer)
            ).label("successful"),
            func.avg(ToolExecution.duration_ms).label("avg_duration"),
        )
        
        if since:
            query = query.filter(ToolExecution.created_at >= since)
        
        query = query.group_by(ToolExecution.tool_name)
        
        results = {}
        for row in query.all():
            results[row.tool_name] = {
                "total_executions": row.total,
                "successful_executions": row.successful or 0,
                "success_rate": (row.successful or 0) / row.total if row.total > 0 else 0,
                "avg_duration_ms": float(row.avg_duration) if row.avg_duration else 0,
            }
        
        return results
    
    def find_similar_successful(
        self,
        query_normalized: str,
        tool_name: Optional[str] = None,
        limit: int = 10,
    ) -> List[ToolExecution]:
        """Find similar successful executions.
        
        This is a simple keyword-based search. For production,
        consider adding vector similarity with embeddings.
        """
        # Simple keyword matching for now
        keywords = query_normalized.split()
        if not keywords:
            return []
        
        db_query = self.session.query(ToolExecution).filter(
            ToolExecution.success == True
        )
        
        if tool_name:
            db_query = db_query.filter(ToolExecution.tool_name == tool_name)
        
        # Match any keyword
        for kw in keywords[:5]:  # Limit keywords to avoid slow queries
            db_query = db_query.filter(
                ToolExecution.query_normalized.contains(kw)
            )
        
        return db_query.order_by(desc(ToolExecution.created_at)).limit(limit).all()
    
    def get_successful_args_for_tool(
        self,
        tool_name: str,
        limit: int = 100,
    ) -> List[Dict]:
        """Get arguments from successful executions.
        
        Useful for ArgumentMemory (RAG Layer 2) to learn common patterns.
        """
        executions = (
            self.session.query(ToolExecution)
            .filter(
                and_(
                    ToolExecution.tool_name == tool_name,
                    ToolExecution.success == True,
                )
            )
            .order_by(desc(ToolExecution.created_at))
            .limit(limit)
            .all()
        )
        
        return [e.tool_args for e in executions if e.tool_args]
    
    def get_recent_errors(
        self,
        limit: int = 20,
        tool_name: Optional[str] = None,
    ) -> List[ToolExecution]:
        """Get recent error executions for debugging."""
        query = self.session.query(ToolExecution).filter(
            ToolExecution.success == False
        )
        if tool_name:
            query = query.filter(ToolExecution.tool_name == tool_name)
        return query.order_by(desc(ToolExecution.created_at)).limit(limit).all()
    
    def cleanup_old(self, days: int = 90) -> int:
        """Delete executions older than specified days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        deleted = self.session.query(ToolExecution).filter(
            ToolExecution.created_at < cutoff
        ).delete()
        return deleted


# Import Integer for the cast
from sqlalchemy import Integer
