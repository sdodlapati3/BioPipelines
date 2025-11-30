"""
Celery Application Configuration.

Configures Celery with Redis broker and result backend.
Supports graceful fallback for development without Redis.
"""

import os
from typing import Optional

from celery import Celery
from kombu import Queue

# Configuration from environment
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", REDIS_URL)

# Create Celery app
celery_app = Celery(
    "workflow_composer",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        "src.workflow_composer.jobs.tasks",
    ],
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    
    # Task execution settings
    task_soft_time_limit=300,  # 5 minutes soft limit
    task_time_limit=600,  # 10 minutes hard limit
    task_acks_late=True,  # Acknowledge after task completes
    task_reject_on_worker_lost=True,
    
    # Worker settings
    worker_prefetch_multiplier=1,  # One task at a time per worker
    worker_concurrency=4,  # 4 concurrent workers
    
    # Result settings
    result_expires=3600,  # Results expire after 1 hour
    result_extended=True,  # Include additional task info
    
    # Queue configuration
    task_queues=(
        Queue("default", routing_key="default"),
        Queue("workflows", routing_key="workflow.#"),
        Queue("search", routing_key="search.#"),
        Queue("validation", routing_key="validation.#"),
        Queue("execution", routing_key="execution.#"),
    ),
    
    task_default_queue="default",
    task_default_exchange="workflow_composer",
    task_default_routing_key="default",
    
    # Task routing
    task_routes={
        "src.workflow_composer.jobs.tasks.generate_workflow_task": {
            "queue": "workflows",
            "routing_key": "workflow.generate",
        },
        "src.workflow_composer.jobs.tasks.search_tools_task": {
            "queue": "search",
            "routing_key": "search.tools",
        },
        "src.workflow_composer.jobs.tasks.validate_workflow_task": {
            "queue": "validation",
            "routing_key": "validation.workflow",
        },
        "src.workflow_composer.jobs.tasks.execute_workflow_task": {
            "queue": "execution",
            "routing_key": "execution.workflow",
        },
    },
    
    # Retry settings
    task_annotations={
        "*": {
            "rate_limit": "10/s",
            "max_retries": 3,
            "default_retry_delay": 60,
        }
    },
    
    # Beat schedule for periodic tasks (optional)
    beat_schedule={
        "cleanup-expired-jobs": {
            "task": "src.workflow_composer.jobs.tasks.cleanup_expired_jobs",
            "schedule": 3600.0,  # Every hour
        },
    },
)


def get_celery_app() -> Celery:
    """Get the configured Celery application."""
    return celery_app


class CeleryHealthCheck:
    """Health check for Celery workers."""
    
    def __init__(self, app: Optional[Celery] = None):
        self.app = app or celery_app
    
    def ping(self) -> dict:
        """Ping all workers and return their status."""
        try:
            inspect = self.app.control.inspect()
            active = inspect.active()
            registered = inspect.registered()
            stats = inspect.stats()
            
            return {
                "status": "healthy" if active else "no_workers",
                "active_tasks": active or {},
                "registered_tasks": registered or {},
                "worker_stats": stats or {},
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }
    
    def get_queue_lengths(self) -> dict:
        """Get the length of each queue."""
        try:
            with self.app.connection_or_acquire() as conn:
                queue_lengths = {}
                for queue in ["default", "workflows", "search", "validation", "execution"]:
                    try:
                        queue_lengths[queue] = conn.default_channel.queue_declare(
                            queue=queue, passive=True
                        ).message_count
                    except Exception:
                        queue_lengths[queue] = "unknown"
                return queue_lengths
        except Exception as e:
            return {"error": str(e)}


# Development fallback - in-memory task execution
class SyncTaskExecutor:
    """
    Synchronous task executor for development without Redis.
    Executes tasks immediately instead of queuing.
    """
    
    def __init__(self):
        self._results = {}
        self._task_id_counter = 0
    
    def apply_async(self, task_func, args=None, kwargs=None, task_id=None):
        """Execute task synchronously and return a mock result."""
        import uuid
        from dataclasses import dataclass
        
        @dataclass
        class MockAsyncResult:
            id: str
            status: str = "SUCCESS"
            result: any = None
            error: str = None
            
            def get(self, timeout=None):
                return self.result
            
            def ready(self) -> bool:
                return True
            
            def successful(self) -> bool:
                return self.status == "SUCCESS"
            
            def failed(self) -> bool:
                return self.status == "FAILURE"
        
        task_id = task_id or str(uuid.uuid4())
        args = args or ()
        kwargs = kwargs or {}
        
        try:
            result = task_func(*args, **kwargs)
            mock_result = MockAsyncResult(id=task_id, result=result)
        except Exception as e:
            mock_result = MockAsyncResult(
                id=task_id, 
                status="FAILURE", 
                error=str(e)
            )
        
        self._results[task_id] = mock_result
        return mock_result
    
    def get_result(self, task_id: str):
        """Get result for a task."""
        return self._results.get(task_id)


# Global sync executor for development
_sync_executor = SyncTaskExecutor()


def get_sync_executor() -> SyncTaskExecutor:
    """Get the synchronous task executor for development."""
    return _sync_executor
