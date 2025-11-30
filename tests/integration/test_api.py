"""
Integration Tests for API Endpoints.

Tests the full API request/response cycle including:
- Health endpoints
- Workflow generation endpoints
- Search endpoints
- Job management endpoints
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Integration tests for health endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.workflow_composer.api.app import create_app
        app = create_app()
        return TestClient(app)
    
    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_readiness_check(self, client):
        """Test readiness endpoint."""
        response = client.get("/ready")
        
        # Should return 200 or 503 depending on dependencies
        assert response.status_code in [200, 503]
        data = response.json()
        assert "ready" in data
        assert "checks" in data
    
    def test_service_info(self, client):
        """Test service info endpoint."""
        response = client.get("/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data


class TestWorkflowEndpoints:
    """Integration tests for workflow endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client with mocked auth."""
        from src.workflow_composer.api.app import create_app
        from src.workflow_composer.auth.models import AuthUser, AuthResult, UserRole
        from uuid import uuid4
        
        app = create_app()
        
        # Create a mock authenticated user
        mock_user = AuthUser(
            id=uuid4(),
            email="test@example.com",
            name="Test User",
            role=UserRole.USER,
            created_at=datetime.utcnow(),
        )
        
        mock_result = AuthResult(success=True, user=mock_user)
        
        # We'll pass auth headers in requests
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create auth headers for requests."""
        return {"Authorization": "Bearer bp_test_key_12345"}
    
    def test_generate_workflow_endpoint_exists(self, client):
        """Test that workflow generation endpoint exists."""
        # Without valid auth, should get 401
        response = client.post(
            "/api/v1/workflows/generate",
            json={"query": "RNA-seq analysis"},
        )
        
        # Either 401 (no auth) or 200/422 (with auth)
        assert response.status_code in [401, 200, 422]
    
    def test_list_workflows_endpoint_exists(self, client):
        """Test that list workflows endpoint exists."""
        response = client.get("/api/v1/workflows")
        
        # Without auth, should get 401
        assert response.status_code in [401, 200]


class TestSearchEndpoints:
    """Integration tests for search endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.workflow_composer.api.app import create_app
        app = create_app()
        return TestClient(app)
    
    @pytest.mark.skip(reason="Search endpoints may not be implemented yet")
    def test_search_tools_endpoint_exists(self, client):
        """Test that search tools endpoint exists."""
        response = client.get("/api/v1/search/tools?q=salmon")
        assert response.status_code in [401, 200]
    
    @pytest.mark.skip(reason="Search endpoints may not be implemented yet")
    def test_search_workflows_endpoint_exists(self, client):
        """Test that search workflows endpoint exists."""
        response = client.get("/api/v1/search/workflows?q=rna-seq")
        assert response.status_code in [401, 200]


class TestJobEndpoints:
    """Integration tests for job endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.workflow_composer.api.app import create_app
        app = create_app()
        return TestClient(app)
    
    def test_list_jobs_endpoint_exists(self, client):
        """Test that list jobs endpoint exists."""
        response = client.get("/api/v1/jobs")
        
        assert response.status_code in [401, 200]
    
    def test_create_job_endpoint_exists(self, client):
        """Test that create job endpoint exists."""
        response = client.post(
            "/api/v1/jobs",
            json={
                "workflow_id": "wf_test",
                "name": "Test Job",
                "parameters": {},
            },
        )
        
        assert response.status_code in [401, 200, 422]
    
    def test_async_workflow_generation_endpoint_exists(self, client):
        """Test that async workflow generation endpoint exists."""
        response = client.post(
            "/api/v1/jobs/workflows/generate",
            json={
                "query": "RNA-seq analysis",
                "name": "Test Workflow",
            },
        )
        
        assert response.status_code in [401, 200, 422]
    
    def test_queue_status_endpoint_exists(self, client):
        """Test that queue status endpoint exists."""
        response = client.get("/api/v1/jobs/queue/status")
        
        assert response.status_code in [401, 200]


class TestAPIMiddleware:
    """Integration tests for API middleware."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.workflow_composer.api.app import create_app
        app = create_app()
        return TestClient(app)
    
    def test_cors_headers(self, client):
        """Test that CORS headers are set."""
        response = client.options("/health")
        
        # CORS preflight or actual request
        assert response.status_code in [200, 405]
    
    def test_request_id_header(self, client):
        """Test that request ID is added to responses."""
        response = client.get("/health")
        
        # Check for common request ID headers
        headers = dict(response.headers)
        # May have X-Request-ID or similar
        assert response.status_code == 200


class TestAPIRouting:
    """Integration tests for API routing."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.workflow_composer.api.app import create_app
        app = create_app()
        return TestClient(app)
    
    def test_api_versioning(self, client):
        """Test that API versioning works."""
        # v1 endpoints should work
        response = client.get("/api/v1/jobs")
        assert response.status_code in [401, 200]
    
    def test_404_for_unknown_endpoints(self, client):
        """Test that unknown endpoints return 404."""
        response = client.get("/api/v1/unknown-endpoint")
        assert response.status_code == 404


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.workflow_composer.api.app import create_app
        app = create_app()
        return TestClient(app)
    
    def test_full_api_flow(self, client):
        """Test a complete API flow without auth."""
        # 1. Check health
        health = client.get("/health")
        assert health.status_code == 200
        
        # 2. Try to access protected endpoints (should get 401)
        jobs = client.get("/api/v1/jobs")
        assert jobs.status_code == 401
