"""
Unified Settings Management
===========================

Centralized configuration using pydantic-settings.

Features:
- Single source of truth for all configuration
- Environment variable support (BIOPIPELINES_ prefix)
- .env file support
- Type validation
- Nested configuration groups
- Easy testing with overrides

Usage:
    from workflow_composer.infrastructure.settings import Settings, get_settings
    
    # Get settings (auto-loads from env + .env)
    settings = get_settings()
    
    # Access settings
    print(settings.llm.provider)
    print(settings.paths.output_dir)
    
    # Override for testing
    with settings_override(llm_provider="mock"):
        test_something()
"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import os
import threading

# Try pydantic v2 first, fall back to v1 style
try:
    from pydantic import BaseModel, Field, field_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict
    PYDANTIC_V2 = True
except ImportError:
    try:
        from pydantic import BaseModel, BaseSettings, Field, validator
        PYDANTIC_V2 = False
    except ImportError:
        # Fallback to dataclasses if pydantic not available
        BaseModel = None
        BaseSettings = None
        PYDANTIC_V2 = None


# =============================================================================
# Settings Models (Pydantic-based if available, dataclass fallback)
# =============================================================================

if BaseSettings is not None and PYDANTIC_V2:
    # Pydantic v2 implementation
    
    class LLMSettings(BaseModel):
        """LLM provider settings."""
        
        provider: str = Field(
            default="lightning",
            description="Default LLM provider"
        )
        model: str = Field(
            default="deepseek-ai/deepseek-v3",
            description="Default model"
        )
        api_key: Optional[str] = Field(
            default=None,
            description="API key (if required)"
        )
        temperature: float = Field(
            default=0.1,
            ge=0.0,
            le=2.0,
            description="Sampling temperature"
        )
        max_tokens: int = Field(
            default=4096,
            gt=0,
            description="Max tokens in response"
        )
        timeout: int = Field(
            default=60,
            gt=0,
            description="Request timeout in seconds"
        )
        # Local model settings
        ollama_host: str = Field(
            default="http://localhost:11434",
            description="Ollama server URL"
        )
        vllm_host: str = Field(
            default="http://localhost:8000",
            description="vLLM server URL"
        )
    
    
    class SLURMSettings(BaseModel):
        """SLURM cluster settings."""
        
        partition: str = Field(
            default="main",
            description="Default partition"
        )
        default_memory: str = Field(
            default="16G",
            description="Default memory per job"
        )
        default_time: str = Field(
            default="4:00:00",
            description="Default time limit"
        )
        default_cpus: int = Field(
            default=4,
            gt=0,
            description="Default CPUs per job"
        )
        account: Optional[str] = Field(
            default=None,
            description="SLURM account name"
        )
        max_concurrent_jobs: int = Field(
            default=10,
            gt=0,
            description="Max concurrent jobs"
        )
    
    
    class PathSettings(BaseModel):
        """Path settings."""
        
        base_dir: Path = Field(
            default_factory=Path.cwd,
            description="Base directory"
        )
        data_dir: Path = Field(
            default=Path("data"),
            description="Data directory"
        )
        output_dir: Path = Field(
            default=Path("generated_workflows"),
            description="Output directory"
        )
        log_dir: Path = Field(
            default=Path("logs"),
            description="Log directory"
        )
        tool_catalog: Path = Field(
            default=Path("data/tool_catalog"),
            description="Tool catalog directory"
        )
        module_library: Path = Field(
            default=Path("nextflow-pipelines/modules"),
            description="Nextflow modules directory"
        )
        containers_dir: Path = Field(
            default=Path("containers"),
            description="Container definitions directory"
        )
        
        def resolve(self, path: Path) -> Path:
            """Resolve relative path against base_dir."""
            if path.is_absolute():
                return path
            return self.base_dir / path
    
    
    class LoggingSettings(BaseModel):
        """Logging settings."""
        
        level: str = Field(
            default="INFO",
            description="Log level"
        )
        json_output: bool = Field(
            default=False,
            description="Use JSON log format"
        )
        log_file: Optional[Path] = Field(
            default=None,
            description="Log file path"
        )
    
    
    class WebSettings(BaseModel):
        """Web interface settings."""
        
        host: str = Field(
            default="127.0.0.1",
            description="Web server host"
        )
        port: int = Field(
            default=7860,
            ge=1,
            le=65535,
            description="Web server port"
        )
        share: bool = Field(
            default=False,
            description="Create public Gradio link"
        )
        auth_enabled: bool = Field(
            default=False,
            description="Enable authentication"
        )
    
    
    class Settings(BaseSettings):
        """
        Main settings class.
        
        All configuration in one place, loaded from:
        1. Default values
        2. Environment variables (BIOPIPELINES_ prefix)
        3. .env file
        
        Example:
            settings = Settings()
            print(settings.llm.provider)
            print(settings.paths.output_dir)
        """
        
        model_config = SettingsConfigDict(
            env_prefix="BIOPIPELINES_",
            env_file=".env",
            env_file_encoding="utf-8",
            env_nested_delimiter="__",
            extra="ignore",
        )
        
        # Nested settings
        llm: LLMSettings = Field(default_factory=LLMSettings)
        slurm: SLURMSettings = Field(default_factory=SLURMSettings)
        paths: PathSettings = Field(default_factory=PathSettings)
        logging: LoggingSettings = Field(default_factory=LoggingSettings)
        web: WebSettings = Field(default_factory=WebSettings)
        
        # Top-level settings
        debug: bool = Field(
            default=False,
            description="Enable debug mode"
        )
        environment: str = Field(
            default="development",
            description="Environment (development, staging, production)"
        )
        
        @field_validator("environment")
        @classmethod
        def validate_environment(cls, v: str) -> str:
            allowed = {"development", "staging", "production", "testing"}
            if v.lower() not in allowed:
                raise ValueError(f"environment must be one of: {allowed}")
            return v.lower()

elif BaseSettings is not None:
    # Pydantic v1 implementation
    
    class LLMSettings(BaseModel):
        provider: str = "lightning"
        model: str = "deepseek-ai/deepseek-v3"
        api_key: Optional[str] = None
        temperature: float = 0.1
        max_tokens: int = 4096
        timeout: int = 60
        ollama_host: str = "http://localhost:11434"
        vllm_host: str = "http://localhost:8000"
        
        class Config:
            extra = "ignore"
    
    
    class SLURMSettings(BaseModel):
        partition: str = "main"
        default_memory: str = "16G"
        default_time: str = "4:00:00"
        default_cpus: int = 4
        account: Optional[str] = None
        max_concurrent_jobs: int = 10
        
        class Config:
            extra = "ignore"
    
    
    class PathSettings(BaseModel):
        base_dir: Path = Path.cwd()
        data_dir: Path = Path("data")
        output_dir: Path = Path("generated_workflows")
        log_dir: Path = Path("logs")
        tool_catalog: Path = Path("data/tool_catalog")
        module_library: Path = Path("nextflow-pipelines/modules")
        containers_dir: Path = Path("containers")
        
        def resolve(self, path: Path) -> Path:
            if path.is_absolute():
                return path
            return self.base_dir / path
        
        class Config:
            extra = "ignore"
    
    
    class LoggingSettings(BaseModel):
        level: str = "INFO"
        json_output: bool = False
        log_file: Optional[Path] = None
        
        class Config:
            extra = "ignore"
    
    
    class WebSettings(BaseModel):
        host: str = "127.0.0.1"
        port: int = 7860
        share: bool = False
        auth_enabled: bool = False
        
        class Config:
            extra = "ignore"
    
    
    class Settings(BaseSettings):
        llm: LLMSettings = LLMSettings()
        slurm: SLURMSettings = SLURMSettings()
        paths: PathSettings = PathSettings()
        logging: LoggingSettings = LoggingSettings()
        web: WebSettings = WebSettings()
        debug: bool = False
        environment: str = "development"
        
        class Config:
            env_prefix = "BIOPIPELINES_"
            env_file = ".env"
            env_file_encoding = "utf-8"
            extra = "ignore"

else:
    # Dataclass fallback when pydantic not available
    
    @dataclass
    class LLMSettings:
        provider: str = "lightning"
        model: str = "deepseek-ai/deepseek-v3"
        api_key: Optional[str] = None
        temperature: float = 0.1
        max_tokens: int = 4096
        timeout: int = 60
        ollama_host: str = "http://localhost:11434"
        vllm_host: str = "http://localhost:8000"
        
        def __post_init__(self):
            # Load from environment
            self.provider = os.getenv("BIOPIPELINES_LLM__PROVIDER", self.provider)
            self.model = os.getenv("BIOPIPELINES_LLM__MODEL", self.model)
            self.api_key = os.getenv("BIOPIPELINES_LLM__API_KEY", self.api_key)
    
    
    @dataclass
    class SLURMSettings:
        partition: str = "main"
        default_memory: str = "16G"
        default_time: str = "4:00:00"
        default_cpus: int = 4
        account: Optional[str] = None
        max_concurrent_jobs: int = 10
    
    
    @dataclass
    class PathSettings:
        base_dir: Path = field(default_factory=Path.cwd)
        data_dir: Path = field(default_factory=lambda: Path("data"))
        output_dir: Path = field(default_factory=lambda: Path("generated_workflows"))
        log_dir: Path = field(default_factory=lambda: Path("logs"))
        tool_catalog: Path = field(default_factory=lambda: Path("data/tool_catalog"))
        module_library: Path = field(default_factory=lambda: Path("nextflow-pipelines/modules"))
        containers_dir: Path = field(default_factory=lambda: Path("containers"))
        
        def resolve(self, path: Path) -> Path:
            if path.is_absolute():
                return path
            return self.base_dir / path
    
    
    @dataclass
    class LoggingSettings:
        level: str = "INFO"
        json_output: bool = False
        log_file: Optional[Path] = None
    
    
    @dataclass
    class WebSettings:
        host: str = "127.0.0.1"
        port: int = 7860
        share: bool = False
        auth_enabled: bool = False
    
    
    @dataclass
    class Settings:
        llm: LLMSettings = field(default_factory=LLMSettings)
        slurm: SLURMSettings = field(default_factory=SLURMSettings)
        paths: PathSettings = field(default_factory=PathSettings)
        logging: LoggingSettings = field(default_factory=LoggingSettings)
        web: WebSettings = field(default_factory=WebSettings)
        debug: bool = False
        environment: str = "development"
        
        def __post_init__(self):
            self.debug = os.getenv("BIOPIPELINES_DEBUG", "").lower() == "true"
            self.environment = os.getenv("BIOPIPELINES_ENVIRONMENT", self.environment)


# =============================================================================
# Settings Access
# =============================================================================

_settings: Optional[Settings] = None
_settings_lock = threading.Lock()


def get_settings() -> Settings:
    """
    Get the global settings instance.
    
    Settings are loaded once and cached.
    
    Returns:
        Settings instance
        
    Example:
        settings = get_settings()
        print(settings.llm.provider)
    """
    global _settings
    
    if _settings is None:
        with _settings_lock:
            if _settings is None:
                _settings = Settings()
    
    return _settings


def reload_settings() -> Settings:
    """
    Reload settings from environment.
    
    Useful after changing environment variables.
    
    Returns:
        New Settings instance
    """
    global _settings
    
    with _settings_lock:
        _settings = Settings()
    
    return _settings


class settings_override:
    """
    Context manager for temporarily overriding settings.
    
    Useful for testing.
    
    Example:
        with settings_override(debug=True, llm_provider="mock"):
            # Settings are overridden here
            ...
        # Original settings restored
    """
    
    def __init__(self, **overrides):
        self.overrides = overrides
        self.original_settings: Optional[Settings] = None
    
    def __enter__(self) -> Settings:
        global _settings
        
        self.original_settings = _settings
        
        # Create new settings with overrides
        settings_dict = {}
        
        for key, value in self.overrides.items():
            if "__" in key:
                # Nested setting like "llm__provider"
                parts = key.split("__")
                # For simplicity, set as env var and reload
                env_key = f"BIOPIPELINES_{key.upper()}"
                os.environ[env_key] = str(value)
            else:
                env_key = f"BIOPIPELINES_{key.upper()}"
                os.environ[env_key] = str(value)
        
        with _settings_lock:
            _settings = Settings()
        
        return _settings
    
    def __exit__(self, *args) -> None:
        global _settings
        
        # Restore original settings
        with _settings_lock:
            _settings = self.original_settings
        
        # Clean up env vars
        for key in self.overrides:
            env_key = f"BIOPIPELINES_{key.upper()}"
            os.environ.pop(env_key, None)


# =============================================================================
# Settings Utilities
# =============================================================================

def dump_settings() -> Dict[str, Any]:
    """
    Dump current settings as a dictionary.
    
    Useful for debugging configuration.
    
    Returns:
        Settings as dictionary
    """
    settings = get_settings()
    
    if hasattr(settings, "model_dump"):
        # Pydantic v2
        return settings.model_dump()
    elif hasattr(settings, "dict"):
        # Pydantic v1
        return settings.dict()
    else:
        # Dataclass
        import dataclasses
        return dataclasses.asdict(settings)


def print_settings() -> None:
    """Print current settings (for debugging)."""
    import json
    settings_dict = dump_settings()
    
    # Convert Path objects to strings
    def convert(obj):
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    print(json.dumps(convert(settings_dict), indent=2))
