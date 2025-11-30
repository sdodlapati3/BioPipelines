"""
Authentication Data Models
=========================

Data classes for authentication system.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


# =============================================================================
# Enums
# =============================================================================

class UserRole(str, Enum):
    """User roles for authorization."""
    
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"
    SERVICE = "service"  # For internal services


class KeyScope(str, Enum):
    """API key permission scopes."""
    
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"  # Can run workflows
    ADMIN = "admin"


# =============================================================================
# User Model
# =============================================================================

@dataclass
class AuthUser:
    """Authenticated user representation.
    
    This is a simplified view of the User for authentication purposes.
    """
    
    id: UUID
    """Unique user identifier."""
    
    email: str
    """User email address."""
    
    name: Optional[str] = None
    """Display name."""
    
    role: UserRole = UserRole.USER
    """User role for authorization."""
    
    is_active: bool = True
    """Whether the user account is active."""
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    """When the user was created."""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""
    
    @classmethod
    def create(
        cls,
        email: str,
        name: Optional[str] = None,
        role: UserRole = UserRole.USER,
    ) -> "AuthUser":
        """Create a new user."""
        return cls(
            id=uuid4(),
            email=email,
            name=name or email.split("@")[0],
            role=role,
        )
    
    def has_permission(self, scope: KeyScope) -> bool:
        """Check if user has a permission scope."""
        if self.role == UserRole.ADMIN:
            return True
        if self.role == UserRole.READONLY:
            return scope == KeyScope.READ
        if self.role == UserRole.USER:
            return scope in {KeyScope.READ, KeyScope.WRITE, KeyScope.EXECUTE}
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "email": self.email,
            "name": self.name,
            "role": self.role.value,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# API Key Model
# =============================================================================

@dataclass
class APIKey:
    """API key for authentication.
    
    API keys are prefixed with 'bpa_' (BioPipelines API) for identification.
    """
    
    id: UUID
    """Unique key identifier."""
    
    user_id: UUID
    """Owner user ID."""
    
    name: str
    """Descriptive name for the key."""
    
    key_hash: str
    """Hashed key value (never store plaintext)."""
    
    prefix: str
    """First 8 chars of key for identification."""
    
    scopes: List[KeyScope] = field(default_factory=lambda: [KeyScope.READ])
    """Permission scopes for this key."""
    
    is_active: bool = True
    """Whether the key is active."""
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    """When the key was created."""
    
    expires_at: Optional[datetime] = None
    """Optional expiration time."""
    
    last_used_at: Optional[datetime] = None
    """Last time the key was used."""
    
    rate_limit: Optional[int] = None
    """Requests per hour (None = unlimited)."""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""
    
    # Constants
    KEY_PREFIX = "bpa_"
    KEY_LENGTH = 32
    
    @classmethod
    def generate(
        cls,
        user_id: UUID,
        name: str,
        scopes: Optional[List[KeyScope]] = None,
        expires_in_days: Optional[int] = None,
        rate_limit: Optional[int] = None,
    ) -> tuple["APIKey", str]:
        """Generate a new API key.
        
        Returns:
            Tuple of (APIKey, plaintext_key)
            
        Note: The plaintext key is only available at creation time.
        """
        import hashlib
        
        # Generate random key
        raw_key = secrets.token_urlsafe(cls.KEY_LENGTH)
        full_key = f"{cls.KEY_PREFIX}{raw_key}"
        
        # Hash for storage
        key_hash = hashlib.sha256(full_key.encode()).hexdigest()
        
        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        key = cls(
            id=uuid4(),
            user_id=user_id,
            name=name,
            key_hash=key_hash,
            prefix=full_key[:12],
            scopes=scopes or [KeyScope.READ, KeyScope.EXECUTE],
            expires_at=expires_at,
            rate_limit=rate_limit,
        )
        
        return key, full_key
    
    @property
    def is_expired(self) -> bool:
        """Check if key has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    @property
    def is_valid(self) -> bool:
        """Check if key is valid (active and not expired)."""
        return self.is_active and not self.is_expired
    
    def has_scope(self, scope: KeyScope) -> bool:
        """Check if key has a specific scope."""
        return KeyScope.ADMIN in self.scopes or scope in self.scopes
    
    def touch(self) -> None:
        """Update last used timestamp."""
        self.last_used_at = datetime.utcnow()
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": str(self.id),
            "name": self.name,
            "prefix": self.prefix,
            "scopes": [s.value for s in self.scopes],
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
        }
        
        if include_sensitive:
            result["user_id"] = str(self.user_id)
            result["rate_limit"] = self.rate_limit
        
        return result


# =============================================================================
# Authentication Result
# =============================================================================

@dataclass
class AuthResult:
    """Result of an authentication attempt."""
    
    success: bool
    """Whether authentication succeeded."""
    
    user: Optional[AuthUser] = None
    """The authenticated user (if successful)."""
    
    api_key: Optional[APIKey] = None
    """The API key used (if applicable)."""
    
    error: Optional[str] = None
    """Error message (if failed)."""
    
    error_code: Optional[str] = None
    """Machine-readable error code."""
    
    @classmethod
    def ok(cls, user: AuthUser, api_key: Optional[APIKey] = None) -> "AuthResult":
        """Create successful result."""
        return cls(success=True, user=user, api_key=api_key)
    
    @classmethod
    def fail(cls, error: str, code: str = "auth_failed") -> "AuthResult":
        """Create failed result."""
        return cls(success=False, error=error, error_code=code)


# =============================================================================
# Rate Limiting
# =============================================================================

@dataclass
class RateLimitInfo:
    """Rate limit information for a request."""
    
    limit: int
    """Maximum requests allowed in window."""
    
    remaining: int
    """Requests remaining in current window."""
    
    reset_at: datetime
    """When the rate limit resets."""
    
    window_seconds: int = 3600
    """Size of the rate limit window."""
    
    @property
    def is_exceeded(self) -> bool:
        """Check if rate limit is exceeded."""
        return self.remaining <= 0
    
    @property
    def reset_in_seconds(self) -> int:
        """Seconds until rate limit resets."""
        delta = self.reset_at - datetime.utcnow()
        return max(0, int(delta.total_seconds()))
    
    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers."""
        return {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(int(self.reset_at.timestamp())),
        }
