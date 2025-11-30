"""
API Configuration
================

Configuration for the FastAPI application.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class APIConfig:
    """FastAPI application configuration."""
    
    # Application metadata
    title: str = "BioPipelines Workflow Composer"
    description: str = "REST API for biological data workflow composition and execution"
    version: str = "1.0.0"
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # CORS settings
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = field(default_factory=lambda: ["*"])
    cors_allow_headers: List[str] = field(default_factory=lambda: ["*"])
    
    # Authentication
    require_auth: bool = True
    allow_anonymous_read: bool = True
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_default: int = 100  # requests per hour
    
    # Feature flags
    enable_rag: bool = True
    enable_caching: bool = True
    enable_job_queue: bool = False  # Will be enabled in Phase 3
    
    # Database
    database_url: Optional[str] = None
    
    # Redis
    redis_url: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "APIConfig":
        """Create configuration from environment variables."""
        return cls(
            title=os.getenv("API_TITLE", cls.title),
            version=os.getenv("API_VERSION", cls.version),
            host=os.getenv("API_HOST", cls.host),
            port=int(os.getenv("API_PORT", cls.port)),
            debug=os.getenv("API_DEBUG", "false").lower() == "true",
            require_auth=os.getenv("API_REQUIRE_AUTH", "true").lower() == "true",
            allow_anonymous_read=os.getenv("API_ALLOW_ANON_READ", "true").lower() == "true",
            rate_limit_enabled=os.getenv("API_RATE_LIMIT", "true").lower() == "true",
            rate_limit_default=int(os.getenv("API_RATE_LIMIT_DEFAULT", "100")),
            enable_rag=os.getenv("API_ENABLE_RAG", "true").lower() == "true",
            enable_caching=os.getenv("API_ENABLE_CACHE", "true").lower() == "true",
            enable_job_queue=os.getenv("API_ENABLE_JOBS", "false").lower() == "true",
            database_url=os.getenv("DATABASE_URL"),
            redis_url=os.getenv("REDIS_URL"),
            cors_origins=os.getenv("CORS_ORIGINS", "*").split(","),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "version": self.version,
            "debug": self.debug,
            "require_auth": self.require_auth,
            "rate_limit_enabled": self.rate_limit_enabled,
            "enable_rag": self.enable_rag,
            "enable_caching": self.enable_caching,
            "enable_job_queue": self.enable_job_queue,
        }


# Global configuration instance
_config: Optional[APIConfig] = None


def get_api_config() -> APIConfig:
    """Get the global API configuration."""
    global _config
    if _config is None:
        _config = APIConfig.from_env()
    return _config


def set_api_config(config: APIConfig) -> None:
    """Set the global API configuration."""
    global _config
    _config = config
