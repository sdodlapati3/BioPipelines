"""Initial schema for BioPipelines

Revision ID: 001_initial
Revises: 
Create Date: 2025-11-30

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('email', sa.String(255), unique=True, nullable=False, index=True),
        sa.Column('name', sa.String(255), nullable=True),
        sa.Column('tier', sa.String(50), default='free'),
        sa.Column('quota_remaining', sa.Integer(), default=100),
        sa.Column('quota_reset_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )

    # Create api_keys table
    op.create_table(
        'api_keys',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('key_hash', sa.String(255), unique=True, nullable=False, index=True),
        sa.Column('key_prefix', sa.String(20), nullable=False),
        sa.Column('name', sa.String(255), nullable=True),
        sa.Column('scopes', sa.Text(), nullable=True),  # JSON array of scopes
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_api_keys_user_id', 'api_keys', ['user_id'])

    # Create jobs table
    op.create_table(
        'jobs',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('workflow_type', sa.String(100), nullable=True),
        sa.Column('query', sa.Text(), nullable=True),
        sa.Column('status', sa.String(50), default='pending', index=True),
        sa.Column('celery_task_id', sa.String(255), nullable=True, index=True),
        sa.Column('slurm_job_id', sa.String(100), nullable=True),
        sa.Column('result', sa.Text(), nullable=True),  # JSON
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('submitted_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('idx_jobs_user_id', 'jobs', ['user_id'])
    op.create_index('idx_jobs_status', 'jobs', ['status'])

    # Create tool_executions table (for RAG analytics)
    op.create_table(
        'tool_executions',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('tool_name', sa.String(100), nullable=False, index=True),
        sa.Column('query', sa.Text(), nullable=True),
        sa.Column('query_embedding', sa.LargeBinary(), nullable=True),  # For similarity search
        sa.Column('arguments', sa.Text(), nullable=True),  # JSON
        sa.Column('success', sa.Boolean(), default=True),
        sa.Column('duration_ms', sa.Float(), nullable=True),
        sa.Column('result_count', sa.Integer(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('idx_tool_executions_tool', 'tool_executions', ['tool_name'])
    op.create_index('idx_tool_executions_success', 'tool_executions', ['success'])
    op.create_index('idx_tool_executions_created', 'tool_executions', ['created_at'])

    # Create workflows table (generated workflows)
    op.create_table(
        'workflows',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('workflow_type', sa.String(100), nullable=True),
        sa.Column('query', sa.Text(), nullable=True),
        sa.Column('file_path', sa.String(500), nullable=True),
        sa.Column('content_hash', sa.String(64), nullable=True),
        sa.Column('config', sa.Text(), nullable=True),  # JSON
        sa.Column('status', sa.String(50), default='generated'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )
    op.create_index('idx_workflows_user_id', 'workflows', ['user_id'])
    op.create_index('idx_workflows_type', 'workflows', ['workflow_type'])


def downgrade() -> None:
    op.drop_table('workflows')
    op.drop_table('tool_executions')
    op.drop_table('jobs')
    op.drop_table('api_keys')
    op.drop_table('users')
