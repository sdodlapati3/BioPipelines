"""Database session management for BioPipelines.

Provides database connection and session management with support for:
- PostgreSQL (production)
- SQLite (development/testing)
- Connection pooling
- Context managers for transactions
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from typing import Generator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, scoped_session, sessionmaker
from sqlalchemy.pool import StaticPool

from .models import Base

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration.
    
    Attributes:
        url: Database connection URL
        echo: Whether to log SQL statements
        pool_size: Connection pool size (PostgreSQL only)
        max_overflow: Max overflow connections
        pool_timeout: Connection timeout in seconds
    """
    url: str = "sqlite:///biopipelines.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create config from environment variables."""
        return cls(
            url=os.getenv("DATABASE_URL", "sqlite:///biopipelines.db"),
            echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
            pool_size=int(os.getenv("DATABASE_POOL_SIZE", "5")),
            max_overflow=int(os.getenv("DATABASE_MAX_OVERFLOW", "10")),
            pool_timeout=int(os.getenv("DATABASE_POOL_TIMEOUT", "30")),
        )
    
    @property
    def is_sqlite(self) -> bool:
        """Check if using SQLite."""
        return self.url.startswith("sqlite")
    
    @property
    def is_postgres(self) -> bool:
        """Check if using PostgreSQL."""
        return self.url.startswith("postgresql")


# Global state
_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker] = None
_scoped_session: Optional[scoped_session] = None
_lock = Lock()


def _create_engine(config: DatabaseConfig) -> Engine:
    """Create SQLAlchemy engine based on config."""
    if config.is_sqlite:
        # SQLite-specific settings
        engine = create_engine(
            config.url,
            echo=config.echo,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,  # Use static pool for SQLite
        )
        
        # Enable foreign keys for SQLite
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    else:
        # PostgreSQL settings
        engine = create_engine(
            config.url,
            echo=config.echo,
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
            pool_timeout=config.pool_timeout,
            pool_pre_ping=True,  # Check connection validity
        )
    
    return engine


def init_db(config: Optional[DatabaseConfig] = None) -> Engine:
    """Initialize database connection and create tables.
    
    Args:
        config: Database configuration. Uses environment if not provided.
        
    Returns:
        SQLAlchemy engine instance
        
    Example:
        >>> from workflow_composer.db import init_db, DatabaseConfig
        >>> config = DatabaseConfig(url="sqlite:///test.db")
        >>> engine = init_db(config)
    """
    global _engine, _session_factory, _scoped_session
    
    with _lock:
        if _engine is not None:
            logger.debug("Database already initialized")
            return _engine
        
        config = config or DatabaseConfig.from_env()
        logger.info(f"Initializing database: {config.url.split('@')[-1] if '@' in config.url else config.url}")
        
        _engine = _create_engine(config)
        _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
        _scoped_session = scoped_session(_session_factory)
        
        # Create tables
        Base.metadata.create_all(_engine)
        logger.info("Database tables created successfully")
        
        return _engine


def get_engine() -> Engine:
    """Get the database engine, initializing if needed."""
    if _engine is None:
        init_db()
    return _engine


def get_session() -> Session:
    """Get a new database session.
    
    Returns a new session instance. Caller is responsible for
    committing/rolling back and closing.
    
    Returns:
        SQLAlchemy Session instance
        
    Example:
        >>> session = get_session()
        >>> try:
        ...     user = session.query(User).first()
        ...     session.commit()
        ... finally:
        ...     session.close()
    """
    if _session_factory is None:
        init_db()
    return _session_factory()


def get_scoped_session() -> scoped_session:
    """Get the scoped session for thread-local sessions.
    
    Useful for web applications where each request should have
    its own session.
    """
    if _scoped_session is None:
        init_db()
    return _scoped_session


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Context manager for database sessions.
    
    Automatically commits on success and rolls back on exception.
    
    Yields:
        SQLAlchemy Session instance
        
    Example:
        >>> with get_db() as db:
        ...     user = db.query(User).filter_by(email="test@test.com").first()
        ...     user.name = "Updated"
        ... # Automatically commits
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def close_db() -> None:
    """Close database connection and clean up resources."""
    global _engine, _session_factory, _scoped_session
    
    with _lock:
        if _scoped_session is not None:
            _scoped_session.remove()
            _scoped_session = None
        
        if _engine is not None:
            _engine.dispose()
            _engine = None
            _session_factory = None
            
        logger.info("Database connection closed")


def reset_db() -> None:
    """Reset database (drop and recreate all tables).
    
    WARNING: This will delete all data! Use only for testing.
    """
    global _engine
    
    if _engine is None:
        init_db()
    
    logger.warning("Resetting database - all data will be deleted!")
    Base.metadata.drop_all(_engine)
    Base.metadata.create_all(_engine)
    logger.info("Database reset complete")
