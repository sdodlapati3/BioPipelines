"""
Authentication Service
=====================

Core authentication service for API key management.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

from .models import (
    AuthUser,
    APIKey,
    AuthResult,
    RateLimitInfo,
    UserRole,
    KeyScope,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Service Configuration
# =============================================================================

class AuthConfig:
    """Authentication service configuration."""
    
    def __init__(
        self,
        rate_limit_default: int = 100,
        rate_limit_window: int = 3600,
        key_expiry_days: int = 365,
        allow_anonymous: bool = False,
        use_database: bool = True,
    ):
        self.rate_limit_default = rate_limit_default
        self.rate_limit_window = rate_limit_window
        self.key_expiry_days = key_expiry_days
        self.allow_anonymous = allow_anonymous
        self.use_database = use_database


# =============================================================================
# Authentication Service
# =============================================================================

class AuthService:
    """
    Authentication service for API key management.
    
    Provides:
    - User creation and management
    - API key generation and validation
    - Rate limiting
    
    Can use either database persistence or in-memory storage.
    
    Usage:
        auth = AuthService()
        
        # Create user
        user = auth.create_user("researcher@example.com")
        
        # Generate API key
        api_key, plaintext = auth.create_api_key(user.id, "My CLI Tool")
        
        # Validate key
        result = auth.validate_api_key(plaintext)
        if result.success:
            print(f"Authenticated as {result.user.email}")
    """
    
    _instance: Optional["AuthService"] = None
    _lock = threading.Lock()
    
    def __init__(
        self,
        config: Optional[AuthConfig] = None,
        repository = None,  # Optional DB repository
    ):
        self.config = config or AuthConfig()
        self.repository = repository
        
        # In-memory storage (fallback if no DB)
        self._users: Dict[UUID, AuthUser] = {}
        self._users_by_email: Dict[str, UUID] = {}
        self._api_keys: Dict[str, APIKey] = {}  # key_hash -> APIKey
        self._keys_by_user: Dict[UUID, List[str]] = defaultdict(list)
        
        # Rate limiting state
        self._rate_limit_state: Dict[str, Dict[str, Any]] = {}
        
        self._data_lock = threading.RLock()
    
    @classmethod
    def get_instance(
        cls,
        config: Optional[AuthConfig] = None,
        repository = None,
    ) -> "AuthService":
        """Get singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = AuthService(config, repository)
            return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None
    
    # =========================================================================
    # User Management
    # =========================================================================
    
    def create_user(
        self,
        email: str,
        name: Optional[str] = None,
        role: UserRole = UserRole.USER,
    ) -> AuthUser:
        """
        Create a new user.
        
        Args:
            email: User email (must be unique)
            name: Display name
            role: User role
            
        Returns:
            Created user
            
        Raises:
            ValueError: If email already exists
        """
        with self._data_lock:
            # Check uniqueness
            if email.lower() in self._users_by_email:
                raise ValueError(f"User with email {email} already exists")
            
            user = AuthUser.create(email.lower(), name, role)
            
            self._users[user.id] = user
            self._users_by_email[email.lower()] = user.id
            
            logger.info(f"Created user: {user.email} ({user.id})")
            return user
    
    def get_user(self, user_id: UUID) -> Optional[AuthUser]:
        """Get user by ID."""
        with self._data_lock:
            return self._users.get(user_id)
    
    def get_user_by_email(self, email: str) -> Optional[AuthUser]:
        """Get user by email."""
        with self._data_lock:
            user_id = self._users_by_email.get(email.lower())
            if user_id:
                return self._users.get(user_id)
            return None
    
    def update_user(
        self,
        user_id: UUID,
        name: Optional[str] = None,
        role: Optional[UserRole] = None,
        is_active: Optional[bool] = None,
    ) -> Optional[AuthUser]:
        """Update user attributes."""
        with self._data_lock:
            user = self._users.get(user_id)
            if not user:
                return None
            
            if name is not None:
                user.name = name
            if role is not None:
                user.role = role
            if is_active is not None:
                user.is_active = is_active
            
            return user
    
    def deactivate_user(self, user_id: UUID) -> bool:
        """Deactivate a user (also deactivates their API keys)."""
        with self._data_lock:
            user = self._users.get(user_id)
            if not user:
                return False
            
            user.is_active = False
            
            # Deactivate all keys
            for key_hash in self._keys_by_user.get(user_id, []):
                if key_hash in self._api_keys:
                    self._api_keys[key_hash].is_active = False
            
            logger.info(f"Deactivated user: {user.email}")
            return True
    
    # =========================================================================
    # API Key Management
    # =========================================================================
    
    def create_api_key(
        self,
        user_id: UUID,
        name: str,
        scopes: Optional[List[KeyScope]] = None,
        expires_in_days: Optional[int] = None,
        rate_limit: Optional[int] = None,
    ) -> tuple[APIKey, str]:
        """
        Create a new API key for a user.
        
        Args:
            user_id: Owner user ID
            name: Descriptive name
            scopes: Permission scopes
            expires_in_days: Days until expiration (default from config)
            rate_limit: Requests per hour (default from config)
            
        Returns:
            Tuple of (APIKey, plaintext_key)
            
        Note: Store the plaintext key securely - it cannot be retrieved later.
        """
        with self._data_lock:
            user = self._users.get(user_id)
            if not user:
                raise ValueError(f"User {user_id} not found")
            
            if not user.is_active:
                raise ValueError("Cannot create key for inactive user")
            
            expires = expires_in_days or self.config.key_expiry_days
            limit = rate_limit or self.config.rate_limit_default
            
            api_key, plaintext = APIKey.generate(
                user_id=user_id,
                name=name,
                scopes=scopes or [KeyScope.READ, KeyScope.EXECUTE],
                expires_in_days=expires,
                rate_limit=limit,
            )
            
            self._api_keys[api_key.key_hash] = api_key
            self._keys_by_user[user_id].append(api_key.key_hash)
            
            logger.info(f"Created API key '{name}' for user {user.email}")
            return api_key, plaintext
    
    def validate_api_key(self, key: str) -> AuthResult:
        """
        Validate an API key and return the authenticated user.
        
        Args:
            key: The API key to validate
            
        Returns:
            AuthResult with user if valid, error if invalid
        """
        if not key:
            return AuthResult.fail("No API key provided", "missing_key")
        
        # Check prefix
        if not key.startswith(APIKey.KEY_PREFIX):
            return AuthResult.fail("Invalid API key format", "invalid_format")
        
        # Hash and lookup
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        with self._data_lock:
            api_key = self._api_keys.get(key_hash)
            
            if not api_key:
                return AuthResult.fail("Invalid API key", "invalid_key")
            
            if not api_key.is_active:
                return AuthResult.fail("API key is deactivated", "key_inactive")
            
            if api_key.is_expired:
                return AuthResult.fail("API key has expired", "key_expired")
            
            # Get user
            user = self._users.get(api_key.user_id)
            if not user:
                return AuthResult.fail("User not found", "user_not_found")
            
            if not user.is_active:
                return AuthResult.fail("User account is inactive", "user_inactive")
            
            # Update last used
            api_key.touch()
            
            return AuthResult.ok(user, api_key)
    
    def get_api_keys(self, user_id: UUID) -> List[APIKey]:
        """Get all API keys for a user."""
        with self._data_lock:
            key_hashes = self._keys_by_user.get(user_id, [])
            return [
                self._api_keys[h]
                for h in key_hashes
                if h in self._api_keys
            ]
    
    def revoke_api_key(self, key_id: UUID, user_id: UUID) -> bool:
        """
        Revoke an API key.
        
        Args:
            key_id: ID of the key to revoke
            user_id: Owner user ID (for authorization)
            
        Returns:
            True if revoked, False if not found
        """
        with self._data_lock:
            for key_hash, api_key in self._api_keys.items():
                if api_key.id == key_id:
                    # Check authorization
                    if api_key.user_id != user_id:
                        user = self._users.get(user_id)
                        if not user or user.role != UserRole.ADMIN:
                            return False
                    
                    api_key.is_active = False
                    logger.info(f"Revoked API key: {api_key.prefix}...")
                    return True
            
            return False
    
    # =========================================================================
    # Rate Limiting
    # =========================================================================
    
    def check_rate_limit(self, api_key: APIKey) -> RateLimitInfo:
        """
        Check rate limit status for an API key.
        
        Args:
            api_key: The API key to check
            
        Returns:
            RateLimitInfo with current status
        """
        limit = api_key.rate_limit or self.config.rate_limit_default
        window = self.config.rate_limit_window
        
        key_id = str(api_key.id)
        now = datetime.utcnow()
        
        with self._data_lock:
            state = self._rate_limit_state.get(key_id)
            
            # Initialize or reset if window expired
            if state is None or now > state["reset_at"]:
                reset_at = now + timedelta(seconds=window)
                self._rate_limit_state[key_id] = {
                    "count": 0,
                    "reset_at": reset_at,
                }
                state = self._rate_limit_state[key_id]
            
            remaining = max(0, limit - state["count"])
            
            return RateLimitInfo(
                limit=limit,
                remaining=remaining,
                reset_at=state["reset_at"],
                window_seconds=window,
            )
    
    def record_request(self, api_key: APIKey) -> RateLimitInfo:
        """
        Record a request for rate limiting.
        
        Args:
            api_key: The API key making the request
            
        Returns:
            Updated RateLimitInfo
        """
        # Check current status first
        info = self.check_rate_limit(api_key)
        
        key_id = str(api_key.id)
        
        with self._data_lock:
            if key_id in self._rate_limit_state:
                self._rate_limit_state[key_id]["count"] += 1
                info.remaining = max(0, info.remaining - 1)
        
        return info
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        with self._data_lock:
            active_users = sum(1 for u in self._users.values() if u.is_active)
            active_keys = sum(1 for k in self._api_keys.values() if k.is_valid)
            
            return {
                "total_users": len(self._users),
                "active_users": active_users,
                "total_api_keys": len(self._api_keys),
                "active_api_keys": active_keys,
            }


# =============================================================================
# Singleton Accessor
# =============================================================================

_auth_service: Optional[AuthService] = None
_auth_service_lock = threading.Lock()


def get_auth_service(
    config: Optional[AuthConfig] = None,
    reset: bool = False,
) -> AuthService:
    """
    Get the singleton AuthService instance.
    
    Args:
        config: Optional configuration
        reset: If True, create new instance
        
    Returns:
        AuthService singleton instance
    """
    global _auth_service
    
    with _auth_service_lock:
        if _auth_service is None or reset:
            _auth_service = AuthService(config)
        return _auth_service
