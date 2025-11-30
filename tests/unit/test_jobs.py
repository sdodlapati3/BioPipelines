"""
Unit Tests for Job Queue Module.

Tests Celery configuration, task definitions, and sync executor.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4


class TestCeleryConfiguration:
    """Tests for Celery app configuration."""
    
    def test_celery_app_creation(self):
        """Test that Celery app is created correctly."""
        from src.workflow_composer.jobs.celery_app import celery_app, get_celery_app
        
        app = get_celery_app()
        
        assert app is not None
        assert app.main == "workflow_composer"
    
    def test_celery_queues_configured(self):
        """Test that queues are configured."""
        from src.workflow_composer.jobs.celery_app import celery_app
        
        queues = celery_app.conf.task_queues
        queue_names = [q.name for q in queues]
        
        assert "default" in queue_names
        assert "workflows" in queue_names
        assert "search" in queue_names
        assert "validation" in queue_names
        assert "execution" in queue_names
    
    def test_task_routing(self):
        """Test that task routing is configured."""
        from src.workflow_composer.jobs.celery_app import celery_app
        
        routes = celery_app.conf.task_routes
        
        assert routes is not None
        assert len(routes) > 0


class TestSyncTaskExecutor:
    """Tests for synchronous task executor."""
    
    @pytest.fixture
    def executor(self):
        """Create a sync executor instance."""
        from src.workflow_composer.jobs.celery_app import SyncTaskExecutor
        return SyncTaskExecutor()
    
    def test_execute_simple_task(self, executor):
        """Test executing a simple task."""
        def add(a, b):
            return a + b
        
        result = executor.apply_async(add, args=(2, 3))
        
        assert result.get() == 5
        assert result.ready() is True
        assert result.successful() is True
    
    def test_execute_with_kwargs(self, executor):
        """Test executing a task with kwargs."""
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"
        
        result = executor.apply_async(greet, kwargs={"name": "World"})
        
        assert result.get() == "Hello, World!"
    
    def test_execute_failing_task(self, executor):
        """Test executing a task that fails."""
        def fail():
            raise ValueError("Task failed")
        
        result = executor.apply_async(fail)
        
        assert result.failed() is True
        assert result.successful() is False
        assert "Task failed" in result.error
    
    def test_task_id_assignment(self, executor):
        """Test that task IDs are assigned."""
        def noop():
            return None
        
        result = executor.apply_async(noop, task_id="custom-task-id")
        
        assert result.id == "custom-task-id"
    
    def test_get_result(self, executor):
        """Test retrieving a result by ID."""
        def compute():
            return 42
        
        result = executor.apply_async(compute, task_id="result-test")
        
        retrieved = executor.get_result("result-test")
        
        assert retrieved is not None
        assert retrieved.get() == 42


class TestCeleryHealthCheck:
    """Tests for Celery health check."""
    
    def test_health_check_without_workers(self):
        """Test health check when no workers are running."""
        from src.workflow_composer.jobs.celery_app import CeleryHealthCheck
        
        health = CeleryHealthCheck()
        
        # Without Redis/workers, should return unhealthy or no_workers
        status = health.ping()
        
        assert "status" in status
        # Either unhealthy (no Redis) or no_workers
        assert status["status"] in ["unhealthy", "no_workers"]


class TestTaskDefinitions:
    """Tests for Celery task definitions."""
    
    def test_generate_workflow_task_registered(self):
        """Test that generate_workflow_task is registered."""
        from src.workflow_composer.jobs.tasks import generate_workflow_task
        
        assert generate_workflow_task.name == "workflow.generate"
    
    def test_search_tools_task_registered(self):
        """Test that search_tools_task is registered."""
        from src.workflow_composer.jobs.tasks import search_tools_task
        
        assert search_tools_task.name == "search.tools"
    
    def test_validate_workflow_task_registered(self):
        """Test that validate_workflow_task is registered."""
        from src.workflow_composer.jobs.tasks import validate_workflow_task
        
        assert validate_workflow_task.name == "validation.workflow"
    
    def test_execute_workflow_task_registered(self):
        """Test that execute_workflow_task is registered."""
        from src.workflow_composer.jobs.tasks import execute_workflow_task
        
        assert execute_workflow_task.name == "execution.workflow"
    
    def test_cleanup_task_registered(self):
        """Test that cleanup task is registered."""
        from src.workflow_composer.jobs.tasks import cleanup_expired_jobs
        
        assert cleanup_expired_jobs.name == "maintenance.cleanup"


class TestTaskExecution:
    """Tests for task execution (mocked)."""
    
    def test_execute_workflow_task_returns_expected_structure(self):
        """Test execute_workflow_task returns expected structure."""
        from src.workflow_composer.jobs.tasks import execute_workflow_task
        import uuid
        
        # Call the run method directly without mocking request
        # This tests the core logic without Celery internals
        result = execute_workflow_task.run(
            workflow_id="wf_test",
            workflow_code="process {}",
            execution_params={"input": "test.fq"},
        )
        
        assert "execution_id" in result
        assert "workflow_id" in result
        assert "status" in result
        assert result["workflow_id"] == "wf_test"
    
    def test_validate_workflow_task_structure(self):
        """Test validate_workflow_task returns expected structure."""
        from src.workflow_composer.jobs.tasks import validate_workflow_task
        
        # Call run method directly - it will handle missing modules gracefully
        result = validate_workflow_task.run(
            workflow_code="process align { ... }",
            workflow_type="nextflow",
        )
        
        # Should return a dict with validation fields even if validator not available
        assert isinstance(result, dict)
        assert "valid" in result
        assert "errors" in result
        assert "warnings" in result
    
    def test_cleanup_task_structure(self):
        """Test cleanup_expired_jobs returns expected structure."""
        from src.workflow_composer.jobs.tasks import cleanup_expired_jobs
        
        result = cleanup_expired_jobs()
        
        assert "status" in result
        assert "cleaned_at" in result


class TestTaskHelpers:
    """Tests for task helper functions."""
    
    @pytest.mark.skipif(
        True,  # Skip by default as Redis may not be available
        reason="Requires Redis connection"
    )
    def test_get_task_status(self):
        """Test get_task_status function."""
        from src.workflow_composer.jobs.tasks import get_task_status
        
        # Test with non-existent task
        status = get_task_status("non-existent-task-id")
        
        assert "task_id" in status
        assert "status" in status
    
    @pytest.mark.skipif(
        True,  # Skip by default as Redis may not be available
        reason="Requires Redis connection"
    )
    def test_revoke_task(self):
        """Test revoke_task function."""
        from src.workflow_composer.jobs.tasks import revoke_task
        
        result = revoke_task("task-to-revoke")
        
        assert "task_id" in result
        assert result["revoked"] is True
    
    def test_task_helpers_exist(self):
        """Test that task helper functions are importable."""
        from src.workflow_composer.jobs.tasks import get_task_status, revoke_task
        
        assert callable(get_task_status)
        assert callable(revoke_task)


class TestBaseTaskWithRetry:
    """Tests for BaseTaskWithRetry class."""
    
    def test_retry_configuration(self):
        """Test that retry configuration is set."""
        from src.workflow_composer.jobs.tasks import BaseTaskWithRetry
        
        assert BaseTaskWithRetry.autoretry_for == (Exception,)
        assert BaseTaskWithRetry.retry_backoff is True
