"""SQLAlchemy models for BioPipelines.

Defines the database schema for:
- Users and API keys
- Jobs and workflow generations
- Tool execution history (for RAG Layer 1)

All models use UUIDs as primary keys for distributed systems compatibility.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, List

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
    Index,
    JSON,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


# =============================================================================
# Enums
# =============================================================================

class UserTier(str, Enum):
    """User subscription tiers."""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class JobStatus(str, Enum):
    """Job execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowType(str, Enum):
    """Types of bioinformatics workflows."""
    RNA_SEQ = "rna-seq"
    CHIP_SEQ = "chip-seq"
    ATAC_SEQ = "atac-seq"
    DNA_SEQ = "dna-seq"
    SCRNA_SEQ = "scrna-seq"
    METHYLATION = "methylation"
    HIC = "hic"
    LONG_READ = "long-read"
    METAGENOMICS = "metagenomics"
    STRUCTURAL_VARIANTS = "sv"
    CUSTOM = "custom"


# =============================================================================
# User Models
# =============================================================================

class User(Base):
    """User account model.
    
    Stores user information, subscription tier, and usage quotas.
    
    Attributes:
        id: Unique user identifier (UUID)
        email: User's email address (unique)
        name: Display name
        tier: Subscription tier (free/pro/enterprise)
        quota_queries: Remaining query quota
        quota_workflows: Remaining workflow generation quota
        is_active: Whether the account is active
        created_at: Account creation timestamp
        updated_at: Last update timestamp
    """
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=True)
    tier = Column(SQLEnum(UserTier), default=UserTier.FREE, nullable=False)
    
    # Quotas
    quota_queries = Column(Integer, default=100, nullable=False)
    quota_workflows = Column(Integer, default=10, nullable=False)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="user", cascade="all, delete-orphan")
    tool_executions = relationship("ToolExecution", back_populates="user", cascade="all, delete-orphan")
    workflows = relationship("WorkflowGeneration", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, email='{self.email}', tier={self.tier.value})>"
    
    def has_quota(self, quota_type: str = "queries") -> bool:
        """Check if user has remaining quota."""
        if quota_type == "queries":
            return self.quota_queries > 0
        elif quota_type == "workflows":
            return self.quota_workflows > 0
        return False
    
    def consume_quota(self, quota_type: str = "queries", amount: int = 1) -> bool:
        """Consume quota if available. Returns True if successful."""
        if quota_type == "queries" and self.quota_queries >= amount:
            self.quota_queries -= amount
            return True
        elif quota_type == "workflows" and self.quota_workflows >= amount:
            self.quota_workflows -= amount
            return True
        return False


class APIKey(Base):
    """API key model for authentication.
    
    Stores hashed API keys with optional expiration and scope restrictions.
    
    Attributes:
        id: Unique key identifier
        user_id: Associated user
        key_hash: SHA-256 hash of the API key
        name: Human-readable name for the key
        scopes: JSON list of allowed scopes
        is_active: Whether the key is active
        expires_at: Optional expiration timestamp
        last_used_at: Last usage timestamp
    """
    __tablename__ = "api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    key_hash = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    scopes = Column(JSON, default=list, nullable=False)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    last_used_at = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    
    def __repr__(self) -> str:
        return f"<APIKey(id={self.id}, name='{self.name}', user_id={self.user_id})>"
    
    def is_valid(self) -> bool:
        """Check if key is valid (active and not expired)."""
        if not self.is_active:
            return False
        if self.expires_at and self.expires_at < datetime.utcnow():
            return False
        return True
    
    def has_scope(self, scope: str) -> bool:
        """Check if key has a specific scope."""
        if not self.scopes:
            return True  # Empty scopes = all access
        return scope in self.scopes


# =============================================================================
# Job Models
# =============================================================================

class Job(Base):
    """SLURM job execution model.
    
    Tracks workflow jobs submitted to the SLURM scheduler.
    
    Attributes:
        id: Unique job identifier
        user_id: Associated user
        workflow_id: Associated workflow generation
        slurm_job_id: SLURM-assigned job ID
        status: Current job status
        workflow_type: Type of workflow
        input_dir: Input data directory
        output_dir: Output results directory
        error_message: Error message if failed
        submitted_at: Submission timestamp
        started_at: Execution start timestamp
        completed_at: Completion timestamp
    """
    __tablename__ = "jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    workflow_id = Column(UUID(as_uuid=True), ForeignKey("workflow_generations.id"), nullable=True)
    slurm_job_id = Column(String(50), nullable=True, index=True)
    
    # Job details
    name = Column(String(255), nullable=True)
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING, nullable=False, index=True)
    workflow_type = Column(SQLEnum(WorkflowType), nullable=True)
    input_dir = Column(Text, nullable=True)
    output_dir = Column(Text, nullable=True)
    config = Column(JSON, default=dict, nullable=False)
    
    # Results
    result = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    submitted_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="jobs")
    workflow = relationship("WorkflowGeneration", back_populates="jobs")
    
    # Indexes
    __table_args__ = (
        Index("idx_jobs_user_status", "user_id", "status"),
        Index("idx_jobs_submitted", "submitted_at"),
    )
    
    def __repr__(self) -> str:
        return f"<Job(id={self.id}, slurm_job_id='{self.slurm_job_id}', status={self.status.value})>"
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate job duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class WorkflowGeneration(Base):
    """Workflow generation record.
    
    Tracks AI-generated workflows and their metadata.
    
    Attributes:
        id: Unique workflow identifier
        user_id: Associated user
        query: Original natural language query
        workflow_type: Detected workflow type
        workflow_path: Path to generated workflow files
        config: Generated workflow configuration
        generation_time_ms: Time to generate in milliseconds
    """
    __tablename__ = "workflow_generations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Query and type
    query = Column(Text, nullable=False)
    workflow_type = Column(SQLEnum(WorkflowType), nullable=False)
    
    # Generated output
    workflow_path = Column(Text, nullable=True)
    workflow_name = Column(String(255), nullable=True)
    config = Column(JSON, default=dict, nullable=False)
    
    # Metrics
    generation_time_ms = Column(Float, nullable=True)
    llm_model = Column(String(100), nullable=True)
    llm_tokens_used = Column(Integer, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="workflows")
    jobs = relationship("Job", back_populates="workflow")
    
    # Indexes
    __table_args__ = (
        Index("idx_workflows_user", "user_id"),
        Index("idx_workflows_type", "workflow_type"),
    )
    
    def __repr__(self) -> str:
        return f"<WorkflowGeneration(id={self.id}, type={self.workflow_type.value})>"


# =============================================================================
# RAG Layer 1: Tool Execution History
# =============================================================================

class ToolExecution(Base):
    """Tool execution record for RAG Layer 1.
    
    Records all tool executions for analytics and future RAG enhancement.
    This is the foundation for both ArgumentMemory (Layer 2) and 
    RAGToolSelector (Layer 3).
    
    Attributes:
        id: Unique execution identifier
        user_id: Associated user (nullable for anonymous)
        tool_name: Name of the executed tool
        query: Original user query
        query_normalized: Normalized query for similarity
        tool_args: Arguments passed to the tool
        success: Whether execution succeeded
        result_summary: Brief summary of result
        error_message: Error message if failed
        duration_ms: Execution time in milliseconds
        result_count: Number of results returned (for searches)
        user_feedback: Optional user feedback (positive/negative)
    """
    __tablename__ = "tool_executions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    
    # Execution details
    tool_name = Column(String(100), nullable=False, index=True)
    query = Column(Text, nullable=False)
    query_normalized = Column(Text, nullable=False)
    tool_args = Column(JSON, default=dict, nullable=False)
    
    # Results
    success = Column(Boolean, nullable=False, index=True)
    result_summary = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    duration_ms = Column(Float, nullable=False)
    result_count = Column(Integer, nullable=True)
    
    # Feedback for learning
    user_feedback = Column(String(20), nullable=True)  # positive, negative, null
    
    # Context for RAG
    task_type = Column(String(50), nullable=True)  # DATA, WORKFLOW, JOB, etc.
    session_id = Column(String(50), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationships
    user = relationship("User", back_populates="tool_executions")
    
    # Indexes for RAG queries
    __table_args__ = (
        Index("idx_executions_tool_success", "tool_name", "success"),
        Index("idx_executions_created", "created_at"),
        Index("idx_executions_user_tool", "user_id", "tool_name"),
    )
    
    def __repr__(self) -> str:
        return f"<ToolExecution(id={self.id}, tool='{self.tool_name}', success={self.success})>"
    
    @classmethod
    def normalize_query(cls, query: str) -> str:
        """Normalize query for comparison."""
        return " ".join(query.lower().strip().split())
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "tool_name": self.tool_name,
            "query": self.query,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "result_count": self.result_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
