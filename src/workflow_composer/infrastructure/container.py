"""
Dependency Injection Container
==============================

A lightweight dependency injection container for BioPipelines.

Features:
- Simple registration and resolution
- Singleton and transient scopes
- Factory functions for lazy creation
- Thread-safe singleton management
- No heavy frameworks required

Usage:
    from workflow_composer.infrastructure.container import Container, get_container, inject

    # Get the global container
    container = get_container()
    
    # Register dependencies
    container.register(LLMProtocol, OpenAIAdapter(model="gpt-4"))
    container.register_factory(ToolSelector, lambda c: ToolSelector(c.resolve(Config).paths.tools))
    
    # Resolve dependencies
    llm = container.resolve(LLMProtocol)
    
    # Use decorator for automatic injection
    @inject
    def generate_workflow(llm: LLMProtocol, selector: ToolSelector):
        ...
    
    generate_workflow()  # Dependencies injected automatically!
"""

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Type,
    TypeVar,
    Union,
    get_type_hints,
    overload,
)
from enum import Enum
from functools import wraps
import threading
import inspect
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Scope(Enum):
    """Dependency lifecycle scope."""
    
    SINGLETON = "singleton"
    """Single instance shared across all resolutions."""
    
    TRANSIENT = "transient"
    """New instance created for each resolution."""
    
    SCOPED = "scoped"
    """Single instance within a scope (e.g., request)."""


@dataclass
class Registration:
    """A registered dependency."""
    
    interface: Type
    """The interface/protocol being registered."""
    
    implementation: Union[Type, Any, Callable[["Container"], Any]]
    """The implementation: instance, class, or factory."""
    
    scope: Scope = Scope.SINGLETON
    """Lifecycle scope."""
    
    instance: Optional[Any] = None
    """Cached singleton instance."""
    
    is_factory: bool = False
    """True if implementation is a factory function."""


class Container:
    """
    Dependency Injection Container.
    
    A simple but powerful DI container that supports:
    - Interface-to-implementation registration
    - Singleton and transient scopes
    - Factory functions for lazy/dynamic creation
    - Hierarchical containers (child inherits from parent)
    
    Example:
        container = Container()
        
        # Register an instance (singleton)
        container.register(Config, Config.load())
        
        # Register a class (will be instantiated)
        container.register(ToolSelector, ToolSelector)
        
        # Register with factory
        container.register_factory(
            LLMProtocol,
            lambda c: get_llm(c.resolve(Config).llm.provider)
        )
        
        # Resolve
        llm = container.resolve(LLMProtocol)
    """
    
    def __init__(self, parent: Optional["Container"] = None):
        """
        Initialize container.
        
        Args:
            parent: Optional parent container for hierarchical resolution
        """
        self._registrations: Dict[Type, Registration] = {}
        self._parent = parent
        self._lock = threading.RLock()  # Use RLock to allow reentrant calls (factory -> resolve)
        self._scoped_instances: Dict[Type, Any] = {}
    
    def register(
        self,
        interface: Type[T],
        implementation: Union[T, Type[T]],
        scope: Scope = Scope.SINGLETON,
    ) -> "Container":
        """
        Register an implementation for an interface.
        
        Args:
            interface: The interface/protocol type
            implementation: Instance or class
            scope: Lifecycle scope
            
        Returns:
            Self for chaining
            
        Example:
            container.register(LLMProtocol, OpenAIAdapter())
            container.register(ToolSelector, ToolSelector, scope=Scope.TRANSIENT)
        """
        with self._lock:
            # If it's an instance (not a class), store directly
            if not isinstance(implementation, type):
                self._registrations[interface] = Registration(
                    interface=interface,
                    implementation=implementation,
                    scope=Scope.SINGLETON,
                    instance=implementation,
                    is_factory=False,
                )
            else:
                self._registrations[interface] = Registration(
                    interface=interface,
                    implementation=implementation,
                    scope=scope,
                    is_factory=False,
                )
        
        logger.debug(f"Registered {interface.__name__} -> {implementation}")
        return self
    
    def register_factory(
        self,
        interface: Type[T],
        factory: Callable[["Container"], T],
        scope: Scope = Scope.SINGLETON,
    ) -> "Container":
        """
        Register a factory function for an interface.
        
        The factory receives the container and returns an instance.
        Useful for dependencies that need other dependencies.
        
        Args:
            interface: The interface/protocol type
            factory: Factory function (container) -> instance
            scope: Lifecycle scope
            
        Returns:
            Self for chaining
            
        Example:
            container.register_factory(
                Composer,
                lambda c: Composer(
                    llm=c.resolve(LLMProtocol),
                    config=c.resolve(Config),
                )
            )
        """
        with self._lock:
            self._registrations[interface] = Registration(
                interface=interface,
                implementation=factory,
                scope=scope,
                is_factory=True,
            )
        
        logger.debug(f"Registered factory for {interface.__name__}")
        return self
    
    def resolve(self, interface: Type[T]) -> T:
        """
        Resolve an implementation for an interface.
        
        Args:
            interface: The interface/protocol type
            
        Returns:
            The implementation instance
            
        Raises:
            KeyError: If interface is not registered
            
        Example:
            llm = container.resolve(LLMProtocol)
        """
        with self._lock:
            return self._resolve_internal(interface)
    
    def _resolve_internal(self, interface: Type[T]) -> T:
        """Internal resolution (must hold lock)."""
        # Check local registrations
        if interface in self._registrations:
            reg = self._registrations[interface]
            
            # Return cached singleton
            if reg.scope == Scope.SINGLETON and reg.instance is not None:
                return reg.instance
            
            # Return cached scoped instance
            if reg.scope == Scope.SCOPED and interface in self._scoped_instances:
                return self._scoped_instances[interface]
            
            # Create new instance
            instance = self._create_instance(reg)
            
            # Cache if singleton
            if reg.scope == Scope.SINGLETON:
                reg.instance = instance
            elif reg.scope == Scope.SCOPED:
                self._scoped_instances[interface] = instance
            
            return instance
        
        # Check parent container
        if self._parent is not None:
            return self._parent.resolve(interface)
        
        raise KeyError(
            f"No registration found for {interface.__name__}. "
            f"Did you forget to call container.register({interface.__name__}, ...)?"
        )
    
    def _create_instance(self, reg: Registration) -> Any:
        """Create an instance from a registration."""
        if reg.is_factory:
            # Call factory function
            return reg.implementation(self)
        elif isinstance(reg.implementation, type):
            # Instantiate class (with auto-wiring if possible)
            return self._auto_wire(reg.implementation)
        else:
            # Already an instance
            return reg.implementation
    
    def _auto_wire(self, cls: Type[T]) -> T:
        """
        Automatically inject dependencies into constructor.
        
        Inspects __init__ parameters and resolves registered types.
        """
        try:
            hints = get_type_hints(cls.__init__)
        except Exception:
            hints = {}
        
        sig = inspect.signature(cls.__init__)
        kwargs = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            
            # Get type hint
            param_type = hints.get(param_name)
            
            if param_type is not None and param_type in self._registrations:
                kwargs[param_name] = self._resolve_internal(param_type)
            elif param.default is not inspect.Parameter.empty:
                # Has default, skip
                pass
            elif param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                # *args or **kwargs, skip
                pass
        
        return cls(**kwargs)
    
    def try_resolve(self, interface: Type[T]) -> Optional[T]:
        """
        Try to resolve, returning None if not found.
        
        Args:
            interface: The interface/protocol type
            
        Returns:
            The implementation or None
        """
        try:
            return self.resolve(interface)
        except KeyError:
            return None
    
    def is_registered(self, interface: Type) -> bool:
        """Check if an interface is registered."""
        if interface in self._registrations:
            return True
        if self._parent is not None:
            return self._parent.is_registered(interface)
        return False
    
    def create_scope(self) -> "Container":
        """
        Create a child container for scoped resolution.
        
        Scoped instances are isolated to the child container.
        
        Returns:
            New child container
            
        Example:
            # In a request handler
            with container.create_scope() as scoped:
                service = scoped.resolve(RequestScopedService)
        """
        return Container(parent=self)
    
    def clear_scoped(self) -> None:
        """Clear all scoped instances."""
        with self._lock:
            self._scoped_instances.clear()
    
    def __enter__(self) -> "Container":
        return self
    
    def __exit__(self, *args) -> None:
        self.clear_scoped()
    
    def __contains__(self, interface: Type) -> bool:
        return self.is_registered(interface)


# =============================================================================
# Global Container
# =============================================================================

_global_container: Optional[Container] = None
_container_lock = threading.Lock()


def get_container() -> Container:
    """
    Get the global container instance.
    
    Thread-safe singleton access to the global DI container.
    
    Returns:
        The global Container instance
    """
    global _global_container
    
    if _global_container is None:
        with _container_lock:
            if _global_container is None:
                _global_container = Container()
    
    return _global_container


def reset_container() -> None:
    """
    Reset the global container.
    
    Useful for testing to start with a clean slate.
    """
    global _global_container
    with _container_lock:
        _global_container = None


def configure_container(configurator: Callable[[Container], None]) -> Container:
    """
    Configure the global container.
    
    Args:
        configurator: Function that registers dependencies
        
    Returns:
        The configured container
        
    Example:
        def setup_dependencies(c: Container):
            c.register(Config, Config.load())
            c.register(LLMProtocol, OllamaAdapter())
        
        configure_container(setup_dependencies)
    """
    container = get_container()
    configurator(container)
    return container


# =============================================================================
# Injection Decorator
# =============================================================================

def inject(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for automatic dependency injection.
    
    Injects registered dependencies based on type hints.
    Caller can still override by passing explicit arguments.
    
    Example:
        @inject
        def generate_workflow(
            query: str,  # Regular parameter (passed by caller)
            llm: LLMProtocol,  # Injected from container
            selector: ToolSelector,  # Injected from container
        ) -> Workflow:
            ...
        
        # Call with just the required parameter
        workflow = generate_workflow("RNA-seq analysis")
    """
    sig = inspect.signature(func)
    
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        container = get_container()
        
        # Bind provided arguments
        bound = sig.bind_partial(*args, **kwargs)
        
        # Inject missing arguments that have registered types
        for param_name, param in sig.parameters.items():
            if param_name in bound.arguments:
                continue  # Already provided
            
            param_type = hints.get(param_name)
            
            if param_type is not None and container.is_registered(param_type):
                bound.arguments[param_name] = container.resolve(param_type)
        
        return func(*bound.args, **bound.kwargs)
    
    return wrapper


# =============================================================================
# Testing Utilities
# =============================================================================

class MockContainer(Container):
    """
    Container with easy mocking for tests.
    
    Example:
        container = MockContainer()
        container.mock(LLMProtocol, MockLLM())
        
        # Now any code using inject will get MockLLM
    """
    
    def mock(self, interface: Type[T], mock_impl: T) -> "MockContainer":
        """Register a mock implementation."""
        return self.register(interface, mock_impl)
    
    def mock_factory(
        self,
        interface: Type[T],
        factory: Callable[[], T]
    ) -> "MockContainer":
        """Register a mock factory."""
        return self.register_factory(interface, lambda _: factory())
