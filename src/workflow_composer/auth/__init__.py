"""
Authentication and Authorization Module
======================================

Provides API key-based authentication for the BioPipelines workflow composer.

Features:
- API key generation and validation
- User management
- Rate limiting support
- Session management

Usage:
    from workflow_composer.auth import AuthService, get_auth_service
    
    auth = get_auth_service()
    
    # Create a user
    user = auth.create_user("researcher@example.com")
    
    # Generate API key
    api_key = auth.create_api_key(user.id, name="CLI Tool")
    
    # Validate in request
    user = auth.validate_api_key("bpa_xxxxx...")
"""

from .dependencies import (
    get_current_user,
    optional_api_key,
    require_api_key,
)
from .models import (
    APIKey,
    AuthResult,
    AuthUser,
    RateLimitInfo,
)
from .service import (
    AuthService,
    get_auth_service,
)

__all__ = [
    # Service
    "AuthService",
    "get_auth_service",
    # Models
    "AuthUser",
    "APIKey",
    "AuthResult",
    "RateLimitInfo",
    # Dependencies
    "get_current_user",
    "require_api_key",
    "optional_api_key",
]
