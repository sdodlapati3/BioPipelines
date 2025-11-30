"""
FastAPI Dependencies for Authentication
======================================

Dependency injection functions for FastAPI routes.
"""

from __future__ import annotations

import logging
from typing import Optional

from .models import AuthResult, AuthUser, KeyScope
from .service import get_auth_service

logger = logging.getLogger(__name__)


# =============================================================================
# HTTP Header Extraction
# =============================================================================

def extract_api_key_from_header(authorization: Optional[str]) -> Optional[str]:
    """
    Extract API key from Authorization header.
    
    Supports:
    - Bearer bpa_xxx...
    - ApiKey bpa_xxx...
    - bpa_xxx... (direct)
    """
    if not authorization:
        return None
    
    authorization = authorization.strip()
    
    # Bearer token format
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    
    # ApiKey format
    if authorization.lower().startswith("apikey "):
        return authorization[7:].strip()
    
    # Direct key (starts with prefix)
    if authorization.startswith("bpa_"):
        return authorization
    
    return None


# =============================================================================
# FastAPI Dependencies
# =============================================================================

async def get_current_user(
    authorization: Optional[str] = None,
    x_api_key: Optional[str] = None,
) -> Optional[AuthUser]:
    """
    FastAPI dependency to get the current authenticated user.
    
    Checks both Authorization header and X-API-Key header.
    
    Usage with FastAPI:
        from fastapi import Depends, Header
        
        @app.get("/protected")
        async def protected_route(
            user: AuthUser = Depends(get_current_user)
        ):
            if not user:
                raise HTTPException(401, "Not authenticated")
            return {"user": user.email}
    
    Note: This is an optional dependency - it returns None if not authenticated.
    Use require_api_key for mandatory authentication.
    """
    # Try to extract key
    key = x_api_key or extract_api_key_from_header(authorization)
    
    if not key:
        return None
    
    auth = get_auth_service()
    result = auth.validate_api_key(key)
    
    if result.success:
        return result.user
    
    return None


async def require_api_key(
    authorization: Optional[str] = None,
    x_api_key: Optional[str] = None,
) -> AuthResult:
    """
    FastAPI dependency that requires valid API key authentication.
    
    Returns AuthResult which contains both user and API key info,
    useful for rate limiting and scope checking.
    
    Usage with FastAPI:
        from fastapi import Depends, Header, HTTPException
        
        @app.get("/protected")
        async def protected_route(
            auth: AuthResult = Depends(require_api_key)
        ):
            if not auth.success:
                raise HTTPException(401, auth.error)
            return {"user": auth.user.email}
    """
    key = x_api_key or extract_api_key_from_header(authorization)
    
    if not key:
        return AuthResult.fail("API key required", "missing_key")
    
    auth = get_auth_service()
    return auth.validate_api_key(key)


async def optional_api_key(
    authorization: Optional[str] = None,
    x_api_key: Optional[str] = None,
) -> AuthResult:
    """
    FastAPI dependency for optional authentication.
    
    Returns AuthResult which may or may not have a user.
    Useful for endpoints that work differently for authenticated vs anonymous users.
    
    Usage:
        @app.get("/search")
        async def search(
            auth: AuthResult = Depends(optional_api_key)
        ):
            if auth.success:
                # Authenticated - full access
                return search_all(auth.user)
            else:
                # Anonymous - limited access
                return search_public()
    """
    key = x_api_key or extract_api_key_from_header(authorization)
    
    if not key:
        # Not authenticated, but that's OK
        return AuthResult(success=False)
    
    auth = get_auth_service()
    return auth.validate_api_key(key)


# =============================================================================
# Scope Checkers
# =============================================================================

def require_scope(scope: KeyScope):
    """
    Create a dependency that requires a specific permission scope.
    
    Usage:
        @app.post("/workflows")
        async def create_workflow(
            auth: AuthResult = Depends(require_scope(KeyScope.WRITE))
        ):
            # Only gets here if key has WRITE scope
            pass
    """
    async def checker(
        authorization: Optional[str] = None,
        x_api_key: Optional[str] = None,
    ) -> AuthResult:
        key = x_api_key or extract_api_key_from_header(authorization)
        
        if not key:
            return AuthResult.fail("API key required", "missing_key")
        
        auth = get_auth_service()
        result = auth.validate_api_key(key)
        
        if not result.success:
            return result
        
        # Check scope
        if result.api_key and not result.api_key.has_scope(scope):
            return AuthResult.fail(
                f"Missing required scope: {scope.value}",
                "insufficient_scope"
            )
        
        return result
    
    return checker


def require_admin():
    """
    Create a dependency that requires admin role.
    
    Usage:
        @app.delete("/users/{user_id}")
        async def delete_user(
            user_id: str,
            auth: AuthResult = Depends(require_admin())
        ):
            # Only admins get here
            pass
    """
    async def checker(
        authorization: Optional[str] = None,
        x_api_key: Optional[str] = None,
    ) -> AuthResult:
        key = x_api_key or extract_api_key_from_header(authorization)
        
        if not key:
            return AuthResult.fail("API key required", "missing_key")
        
        auth = get_auth_service()
        result = auth.validate_api_key(key)
        
        if not result.success:
            return result
        
        from .models import UserRole
        if result.user.role != UserRole.ADMIN:
            return AuthResult.fail(
                "Admin access required",
                "admin_required"
            )
        
        return result
    
    return checker


# =============================================================================
# Rate Limit Helper
# =============================================================================

async def check_rate_limit(auth_result: AuthResult) -> dict:
    """
    Check rate limit for an authenticated request.
    
    Returns dict with rate limit headers to include in response.
    
    Usage:
        @app.get("/search")
        async def search(
            auth: AuthResult = Depends(require_api_key)
        ):
            if not auth.success:
                raise HTTPException(401, auth.error)
            
            rate_info = await check_rate_limit(auth)
            if rate_info.get("exceeded"):
                raise HTTPException(429, "Rate limit exceeded")
            
            # Normal processing...
            response = do_search()
            
            # Add rate limit headers
            return JSONResponse(
                content=response,
                headers=rate_info["headers"]
            )
    """
    if not auth_result.success or not auth_result.api_key:
        return {"exceeded": False, "headers": {}}
    
    auth = get_auth_service()
    rate_info = auth.record_request(auth_result.api_key)
    
    return {
        "exceeded": rate_info.is_exceeded,
        "remaining": rate_info.remaining,
        "reset_in": rate_info.reset_in_seconds,
        "headers": rate_info.to_headers(),
    }
