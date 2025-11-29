"""
Tests for Infrastructure Layer
==============================

Tests for:
- Exception hierarchy
- DI Container
- Protocols
- Structured logging
- Settings management
"""

import pytest
import logging
import os
from pathlib import Path
from typing import List
from unittest.mock import patch, MagicMock


# =============================================================================
# Exception Tests
# =============================================================================

class TestExceptions:
    """Tests for the exception hierarchy."""
    
    def test_base_exception(self):
        """BioPipelinesError is the base for all errors."""
        from workflow_composer.infrastructure.exceptions import (
            BioPipelinesError,
            LLMError,
            ToolNotFoundError,
        )
        
        # All errors inherit from BioPipelinesError
        assert issubclass(LLMError, BioPipelinesError)
        assert issubclass(ToolNotFoundError, BioPipelinesError)
    
    def test_exception_details(self):
        """Exceptions include helpful details."""
        from workflow_composer.infrastructure.exceptions import BioPipelinesError
        
        error = BioPipelinesError(
            message="Something went wrong",
            details={"key": "value"},
            suggestion="Try this instead"
        )
        
        assert error.message == "Something went wrong"
        assert error.details["key"] == "value"
        assert error.suggestion == "Try this instead"
        assert "Try this instead" in str(error)
    
    def test_tool_not_found_error(self):
        """ToolNotFoundError includes helpful suggestions."""
        from workflow_composer.infrastructure.exceptions import ToolNotFoundError
        
        error = ToolNotFoundError(
            tool_name="starr",
            available_tools=["star", "star-fusion", "starcode"]
        )
        
        assert error.tool_name == "starr"
        assert "star" in error.suggestion
        assert error.available_tools == ["star", "star-fusion", "starcode"]
    
    def test_llm_connection_error(self):
        """LLMConnectionError provides provider-specific hints."""
        from workflow_composer.infrastructure.exceptions import LLMConnectionError
        
        error = LLMConnectionError(
            provider="ollama",
            endpoint="http://localhost:11434"
        )
        
        assert error.provider == "ollama"
        assert "ollama serve" in error.suggestion.lower()
    
    def test_exception_serialization(self):
        """Exceptions can be serialized to dict."""
        from workflow_composer.infrastructure.exceptions import BioPipelinesError
        
        error = BioPipelinesError(
            message="Test error",
            details={"code": 123}
        )
        
        data = error.to_dict()
        
        assert data["error_type"] == "BioPipelinesError"
        assert data["message"] == "Test error"
        assert data["details"]["code"] == 123


# =============================================================================
# Protocol Tests
# =============================================================================

class TestProtocols:
    """Tests for protocol definitions."""
    
    def test_llm_protocol_structural_typing(self):
        """Classes can implement LLMProtocol without inheritance."""
        from workflow_composer.infrastructure.protocols import LLMProtocol, Message
        
        # This class implements the protocol without inheriting
        class MyLLM:
            @property
            def model_name(self) -> str:
                return "my-model"
            
            def complete(self, prompt: str, system_prompt=None, **kwargs) -> str:
                return f"Response to: {prompt}"
            
            def chat(self, messages: List[Message], **kwargs) -> str:
                return "Chat response"
        
        llm = MyLLM()
        
        # Should be recognized as implementing the protocol
        assert isinstance(llm, LLMProtocol)
        
        # Should work with type checking
        def use_llm(llm: LLMProtocol) -> str:
            return llm.complete("test")
        
        result = use_llm(llm)
        assert result == "Response to: test"
    
    def test_implements_protocol_helper(self):
        """implements_protocol helper works correctly."""
        from workflow_composer.infrastructure.protocols import (
            LLMProtocol,
            implements_protocol,
            Message,
        )
        
        class ValidLLM:
            @property
            def model_name(self) -> str:
                return "test"
            
            def complete(self, prompt: str, **kwargs) -> str:
                return "done"
            
            def chat(self, messages: List[Message], **kwargs) -> str:
                return "done"
        
        class InvalidLLM:
            pass
        
        assert implements_protocol(ValidLLM(), LLMProtocol)
        assert not implements_protocol(InvalidLLM(), LLMProtocol)


# =============================================================================
# Container Tests
# =============================================================================

class TestContainer:
    """Tests for the DI Container."""
    
    def test_register_and_resolve(self):
        """Can register and resolve dependencies."""
        from workflow_composer.infrastructure.container import Container
        
        container = Container()
        
        # Register instance
        container.register(str, "hello")
        
        # Resolve
        result = container.resolve(str)
        
        assert result == "hello"
    
    def test_singleton_scope(self):
        """Singleton returns same instance."""
        from workflow_composer.infrastructure.container import Container, Scope
        
        container = Container()
        
        class Service:
            pass
        
        container.register(Service, Service, scope=Scope.SINGLETON)
        
        instance1 = container.resolve(Service)
        instance2 = container.resolve(Service)
        
        assert instance1 is instance2
    
    def test_transient_scope(self):
        """Transient creates new instance each time."""
        from workflow_composer.infrastructure.container import Container, Scope
        
        container = Container()
        
        class Service:
            pass
        
        container.register(Service, Service, scope=Scope.TRANSIENT)
        
        instance1 = container.resolve(Service)
        instance2 = container.resolve(Service)
        
        assert instance1 is not instance2
    
    def test_factory_registration(self):
        """Factory functions can create dependencies."""
        from workflow_composer.infrastructure.container import Container
        
        container = Container()
        
        class Config:
            def __init__(self, value: str):
                self.value = value
        
        class Service:
            def __init__(self, config: Config):
                self.config = config
        
        container.register(Config, Config("test-value"))
        container.register_factory(
            Service,
            lambda c: Service(c.resolve(Config))
        )
        
        service = container.resolve(Service)
        
        assert service.config.value == "test-value"
    
    def test_hierarchical_resolution(self):
        """Child container inherits from parent."""
        from workflow_composer.infrastructure.container import Container
        
        parent = Container()
        parent.register(str, "from-parent")
        
        child = Container(parent=parent)
        child.register(int, 42)
        
        # Child resolves own registration
        assert child.resolve(int) == 42
        
        # Child inherits from parent
        assert child.resolve(str) == "from-parent"
    
    def test_inject_decorator(self):
        """@inject automatically resolves dependencies."""
        from workflow_composer.infrastructure.container import (
            Container,
            get_container,
            reset_container,
            inject,
        )
        
        reset_container()
        container = get_container()
        
        class Config:
            value = "injected"
        
        container.register(Config, Config())
        
        @inject
        def my_function(name: str, config: Config) -> str:
            return f"{name}:{config.value}"
        
        result = my_function("test")
        
        assert result == "test:injected"
        
        reset_container()
    
    def test_not_registered_error(self):
        """KeyError when dependency not registered."""
        from workflow_composer.infrastructure.container import Container
        
        container = Container()
        
        with pytest.raises(KeyError) as exc_info:
            container.resolve(str)
        
        assert "str" in str(exc_info.value)


# =============================================================================
# Logging Tests
# =============================================================================

class TestLogging:
    """Tests for structured logging."""
    
    def test_get_logger(self):
        """get_logger returns a StructuredLogger."""
        from workflow_composer.infrastructure.logging import get_logger
        
        logger = get_logger("test.module")
        
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "bind")
    
    def test_structured_log_output(self, caplog):
        """Logger includes structured fields."""
        from workflow_composer.infrastructure.logging import (
            get_logger,
            configure_logging,
            LogLevel,
        )
        
        # Configure with pretty format for testing
        configure_logging(level=LogLevel.DEBUG, json_output=False)
        
        logger = get_logger("test.structured")
        
        with caplog.at_level(logging.DEBUG):
            logger.info("Test message", key="value", count=42)
        
        # Check that message was logged
        assert len(caplog.records) >= 1
    
    def test_operation_context(self):
        """operation_context tracks correlation ID."""
        from workflow_composer.infrastructure.logging import (
            operation_context,
            get_context,
        )
        
        with operation_context("test_operation", user="alice") as ctx:
            current = get_context()
            
            assert current is not None
            assert current.correlation_id == ctx.correlation_id
            assert current.operation == "test_operation"
            assert current.extra["user"] == "alice"
        
        # Context cleared after exiting
        assert get_context() is None
    
    def test_bound_logger(self):
        """BoundLogger includes pre-bound context."""
        from workflow_composer.infrastructure.logging import get_logger
        
        logger = get_logger("test.bound")
        bound = logger.bind(request_id="req-123")
        
        # Should have bind method
        assert hasattr(bound, "info")
        assert hasattr(bound, "bind")
        
        # Can chain bindings
        more_bound = bound.bind(user="bob")
        assert hasattr(more_bound, "info")


# =============================================================================
# Settings Tests
# =============================================================================

class TestSettings:
    """Tests for settings management."""
    
    def test_default_settings(self):
        """Settings have sensible defaults."""
        from workflow_composer.infrastructure.settings import Settings
        
        settings = Settings()
        
        assert settings.llm.provider == "lightning"
        assert settings.llm.temperature == 0.1
        assert settings.slurm.partition == "main"
        assert settings.debug is False
    
    def test_environment_override(self):
        """Settings can be overridden by environment."""
        from workflow_composer.infrastructure.settings import (
            Settings,
            reload_settings,
        )
        
        # Set env var
        os.environ["BIOPIPELINES_DEBUG"] = "true"
        
        try:
            settings = Settings()
            assert settings.debug is True
        finally:
            os.environ.pop("BIOPIPELINES_DEBUG", None)
    
    def test_get_settings_singleton(self):
        """get_settings returns cached instance."""
        from workflow_composer.infrastructure.settings import (
            get_settings,
            reload_settings,
        )
        
        # Reset to fresh state
        reload_settings()
        
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2
    
    def test_path_resolution(self):
        """PathSettings can resolve relative paths."""
        from workflow_composer.infrastructure.settings import PathSettings
        from pathlib import Path
        
        paths = PathSettings(base_dir=Path("/home/user/project"))
        
        # Relative path is resolved
        resolved = paths.resolve(Path("data/raw"))
        assert str(resolved) == "/home/user/project/data/raw"
        
        # Absolute path is unchanged
        resolved = paths.resolve(Path("/absolute/path"))
        assert str(resolved) == "/absolute/path"


# =============================================================================
# Integration Tests
# =============================================================================

class TestInfrastructureIntegration:
    """Integration tests combining infrastructure components."""
    
    def test_container_with_settings(self):
        """Container can provide Settings."""
        from workflow_composer.infrastructure.container import Container
        from workflow_composer.infrastructure.settings import Settings
        
        container = Container()
        settings = Settings()
        
        container.register(Settings, settings)
        
        resolved = container.resolve(Settings)
        
        assert resolved.llm.provider == settings.llm.provider
    
    def test_exception_with_logging(self):
        """Exceptions work with structured logging."""
        from workflow_composer.infrastructure.exceptions import LLMError
        from workflow_composer.infrastructure.logging import (
            get_logger,
            operation_context,
        )
        
        logger = get_logger("test.integration")
        
        with operation_context("test_operation") as ctx:
            try:
                raise LLMError(
                    message="Test LLM failure",
                    provider="test-provider"
                )
            except LLMError as e:
                logger.error(
                    "LLM call failed",
                    error=str(e),
                    provider=e.provider,
                )
                
                # Exception should be serializable
                data = e.to_dict()
                assert data["error_type"] == "LLMError"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
