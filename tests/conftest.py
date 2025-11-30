"""
Pytest configuration and fixtures.

Provides common fixtures for testing the workflow composer.
"""

import os
import pytest
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch
from uuid import uuid4


# =============================================================================
# Environment Setup
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    os.environ.setdefault("TESTING", "true")
    os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
    os.environ.setdefault("CELERY_ENABLED", "false")
    os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
    yield


# =============================================================================
# Temporary Directories
# =============================================================================

@pytest.fixture
def tmp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_data_dir() -> Path:
    """Path to test data directory."""
    return Path(__file__).parent / "data"


# =============================================================================
# Database Fixtures
# =============================================================================

@pytest.fixture
def in_memory_db():
    """Create an in-memory SQLite database for testing."""
    from src.workflow_composer.infrastructure.database import init_db, get_db
    
    # Initialize in-memory database
    engine = init_db(url="sqlite:///:memory:")
    
    # Get a session
    with next(get_db()) as session:
        yield session


# =============================================================================
# Authentication Fixtures
# =============================================================================

@pytest.fixture
def mock_user():
    """Create a mock authenticated user."""
    from src.workflow_composer.auth.models import AuthUser, UserRole
    from datetime import datetime
    
    return AuthUser(
        id=uuid4(),
        email="test@example.com",
        name="Test User",
        role=UserRole.USER,
        created_at=datetime.utcnow(),
        is_active=True,
    )


@pytest.fixture
def mock_admin_user():
    """Create a mock admin user."""
    from src.workflow_composer.auth.models import AuthUser, UserRole
    from datetime import datetime
    
    return AuthUser(
        id=uuid4(),
        email="admin@example.com",
        name="Admin User",
        role=UserRole.ADMIN,
        created_at=datetime.utcnow(),
        is_active=True,
    )


@pytest.fixture
def mock_auth_result(mock_user):
    """Create a mock successful auth result."""
    from src.workflow_composer.auth.models import AuthResult
    
    return AuthResult(
        success=True,
        user=mock_user,
    )


@pytest.fixture
def auth_service():
    """Create an AuthService instance for testing."""
    from src.workflow_composer.auth.service import AuthService
    return AuthService()


# =============================================================================
# RAG Fixtures
# =============================================================================

@pytest.fixture
def tool_memory():
    """Create a ToolMemory instance with in-memory storage."""
    from src.workflow_composer.agents.rag.memory import ToolMemory, ToolMemoryConfig
    config = ToolMemoryConfig(use_database=False)
    return ToolMemory(config)


@pytest.fixture
def arg_memory(tool_memory):
    """Create an ArgumentMemory instance."""
    from src.workflow_composer.agents.rag.arg_memory import ArgumentMemory
    return ArgumentMemory(tool_memory=tool_memory)


@pytest.fixture
def rag_tool_selector(tool_memory):
    """Create a RAGToolSelector instance."""
    from src.workflow_composer.agents.rag.tool_selector import RAGToolSelector
    return RAGToolSelector(tool_memory=tool_memory)


@pytest.fixture
def rag_orchestrator(tool_memory):
    """Create a RAGOrchestrator instance."""
    from src.workflow_composer.agents.rag.orchestrator import RAGOrchestrator
    return RAGOrchestrator(tool_memory=tool_memory)


# =============================================================================
# API Fixtures
# =============================================================================

@pytest.fixture
def test_client():
    """Create a FastAPI test client."""
    from fastapi.testclient import TestClient
    from src.workflow_composer.api.app import create_app
    
    app = create_app()
    return TestClient(app)


@pytest.fixture
def authenticated_client(test_client, mock_auth_result):
    """Create an authenticated test client."""
    from src.workflow_composer.auth.dependencies import get_auth_service
    
    # This fixture provides headers that should work with mocked auth
    class AuthenticatedClient:
        def __init__(self, client, auth_headers):
            self.client = client
            self.auth_headers = auth_headers
        
        def get(self, url, **kwargs):
            kwargs.setdefault("headers", {}).update(self.auth_headers)
            return self.client.get(url, **kwargs)
        
        def post(self, url, **kwargs):
            kwargs.setdefault("headers", {}).update(self.auth_headers)
            return self.client.post(url, **kwargs)
        
        def put(self, url, **kwargs):
            kwargs.setdefault("headers", {}).update(self.auth_headers)
            return self.client.put(url, **kwargs)
        
        def delete(self, url, **kwargs):
            kwargs.setdefault("headers", {}).update(self.auth_headers)
            return self.client.delete(url, **kwargs)
    
    return AuthenticatedClient(
        test_client,
        {"Authorization": "Bearer bp_test_key_12345"},
    )


# =============================================================================
# Job Queue Fixtures
# =============================================================================

@pytest.fixture
def sync_executor():
    """Create a sync task executor for testing."""
    from src.workflow_composer.jobs.celery_app import SyncTaskExecutor
    return SyncTaskExecutor()


@pytest.fixture
def mock_celery_app():
    """Create a mock Celery app for testing."""
    mock_app = MagicMock()
    mock_app.main = "workflow_composer"
    mock_app.conf.broker_url = "redis://localhost:6379/0"
    return mock_app


# =============================================================================
# Cache Fixtures
# =============================================================================

@pytest.fixture
def semantic_cache():
    """Create a SemanticCache instance."""
    from src.workflow_composer.infrastructure.semantic_cache import SemanticCache
    return SemanticCache()


@pytest.fixture
def redis_cache():
    """Create a RedisSemanticCache with fallback."""
    from src.workflow_composer.infrastructure.redis_cache import RedisSemanticCache
    return RedisSemanticCache(fallback_to_memory=True)


# =============================================================================
# Workflow Component Fixtures
# =============================================================================

@pytest.fixture
def intent_parser():
    """Create an IntentParser instance."""
    from src.workflow_composer.agents.query_parser import get_intent_parser
    return get_intent_parser()


@pytest.fixture
def tool_selector():
    """Create a ToolSelector instance."""
    from src.workflow_composer.agents.tool_selector import get_tool_selector
    return get_tool_selector()


@pytest.fixture
def workflow_generator():
    """Create a WorkflowGenerator instance."""
    from src.workflow_composer.agents.workflow_generator import get_workflow_generator
    return get_workflow_generator()


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_query():
    """Sample workflow query."""
    return "RNA-seq differential expression analysis for human cancer samples"


@pytest.fixture
def sample_parsed_intent(sample_query):
    """Sample parsed intent."""
    from src.workflow_composer.agents.query_parser import ParsedIntent
    
    return ParsedIntent(
        original_query=sample_query,
        workflow_type="rna_seq",
        organism="human",
        data_type="rna_seq",
        analysis_goals=["differential_expression"],
        constraints={},
    )


@pytest.fixture
def sample_tools():
    """Sample tool list."""
    from src.workflow_composer.agents.tool_selector import Tool
    
    return [
        Tool(name="salmon", category="alignment"),
        Tool(name="deseq2", category="differential_expression"),
    ]


# =============================================================================
# Pytest Plugins and Markers
# =============================================================================

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_redis: marks tests that require Redis"
    )
    config.addinivalue_line(
        "markers", "requires_celery: marks tests that require Celery workers"
    )
