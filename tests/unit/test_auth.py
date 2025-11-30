"""
Unit Tests for Authentication Module.

Tests API key authentication, user management, and rate limiting.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import MagicMock, patch, AsyncMock

from src.workflow_composer.auth.models import (
    AuthUser,
    APIKey,
    AuthResult,
    KeyScope,
    UserRole,
)
from src.workflow_composer.auth.service import AuthService


class TestAuthModels:
    """Tests for authentication data models."""
    
    def test_auth_user_creation(self):
        """Test creating an AuthUser."""
        user = AuthUser(
            id=uuid4(),
            email="test@example.com",
            name="Test User",
            role=UserRole.USER,
            created_at=datetime.utcnow(),
        )
        
        assert user.email == "test@example.com"
        assert user.role == UserRole.USER
        assert user.is_active is True
    
    def test_auth_user_class_method(self):
        """Test creating user with class method."""
        user = AuthUser.create(
            email="new@example.com",
            name="New User",
            role=UserRole.USER,
        )
        
        assert user.email == "new@example.com"
        assert user.id is not None
    
    def test_api_key_generation(self):
        """Test generating an API key."""
        user_id = uuid4()
        key, raw_key = APIKey.generate(
            user_id=user_id,
            name="Test Key",
            scopes=[KeyScope.READ, KeyScope.WRITE],
        )
        
        assert key.user_id == user_id
        assert KeyScope.READ in key.scopes
        assert key.is_valid is True  # Property, not method
        assert raw_key.startswith("bpa_")
    
    def test_api_key_expired(self):
        """Test expired API key detection."""
        user_id = uuid4()
        key, _ = APIKey.generate(
            user_id=user_id,
            name="Expired Key",
            scopes=[KeyScope.READ],
            expires_in_days=-1,  # Already expired
        )
        
        # Manually set expires_at to past
        key.expires_at = datetime.utcnow() - timedelta(days=1)
        
        assert key.is_valid is False  # Property, not method
    
    def test_auth_result_success(self):
        """Test successful auth result."""
        user = AuthUser(
            id=uuid4(),
            email="test@example.com",
            name="Test",
            role=UserRole.USER,
            created_at=datetime.utcnow(),
        )
        
        result = AuthResult(
            success=True,
            user=user,
        )
        
        assert result.success is True
        assert result.user is not None
        assert result.error is None
    
    def test_auth_result_failure(self):
        """Test failed auth result."""
        result = AuthResult(
            success=False,
            error="Invalid API key",
        )
        
        assert result.success is False
        assert result.error == "Invalid API key"


class TestAuthService:
    """Tests for AuthService."""
    
    @pytest.fixture
    def auth_service(self):
        """Create AuthService instance."""
        return AuthService()
    
    def test_create_user(self, auth_service):
        """Test creating a user."""
        user = auth_service.create_user(
            email="newuser@example.com",
            name="New User",
            role=UserRole.USER,
        )
        
        assert user.email == "newuser@example.com"
        assert user.id is not None
    
    def test_create_api_key(self, auth_service):
        """Test creating an API key."""
        # First create a user
        user = auth_service.create_user(
            email="keyuser@example.com",
            name="Key User",
        )
        
        # Then create a key
        key, raw_key = auth_service.create_api_key(
            user_id=user.id,
            name="Test Key",
            scopes=[KeyScope.READ, KeyScope.WRITE],
        )
        
        assert key is not None
        assert raw_key is not None
        # Key starts with bpa_
        assert raw_key.startswith("bpa_")
    
    def test_validate_api_key(self, auth_service):
        """Test validating an API key."""
        # Create user and key
        user = auth_service.create_user(
            email="validate@example.com",
            name="Validate User",
        )
        
        key, raw_key = auth_service.create_api_key(
            user_id=user.id,
            name="Validate Key",
            scopes=[KeyScope.READ],
        )
        
        # Validate using validate_api_key method
        result = auth_service.validate_api_key(raw_key)
        
        assert result.success is True
        assert result.user is not None
        assert result.user.id == user.id
    
    def test_validate_invalid_key(self, auth_service):
        """Test validating an invalid API key."""
        result = auth_service.validate_api_key("bpa_invalid_key_12345")
        
        assert result.success is False
    
    def test_get_user(self, auth_service):
        """Test getting a user by ID."""
        # Create user
        user = auth_service.create_user(
            email="getuser@example.com",
            name="Get User",
        )
        
        # Get user
        retrieved = auth_service.get_user(user.id)
        
        assert retrieved is not None
        assert retrieved.email == user.email
    
    def test_get_user_by_email(self, auth_service):
        """Test getting a user by email."""
        # Create user
        user = auth_service.create_user(
            email="byemail@example.com",
            name="By Email User",
        )
        
        # Get by email
        retrieved = auth_service.get_user_by_email("byemail@example.com")
        
        assert retrieved is not None
        assert retrieved.id == user.id


class TestAuthDependencies:
    """Tests for FastAPI auth dependencies."""
    
    def test_require_scope_returns_callable(self):
        """Test that require_scope returns a callable."""
        from src.workflow_composer.auth.dependencies import require_scope
        
        # Get the dependency function
        scope_dep = require_scope(KeyScope.READ)
        
        # It should return a callable
        assert callable(scope_dep)
