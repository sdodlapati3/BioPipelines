"""
Configuration management for the Workflow Composer.

Supports:
- YAML configuration files
- Environment variables
- Runtime overrides
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMProviderConfig:
    """Configuration for a single LLM provider."""
    api_key: Optional[str] = None
    model: str = ""
    temperature: float = 0.1
    max_tokens: int = 4096
    host: Optional[str] = None
    device: str = "cpu"
    
    def __post_init__(self):
        # Resolve environment variables
        if self.api_key and self.api_key.startswith("${"):
            env_var = self.api_key[2:-1]
            self.api_key = os.environ.get(env_var)


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    default_provider: str = "ollama"
    providers: Dict[str, LLMProviderConfig] = field(default_factory=dict)
    
    # Component-specific provider assignments
    intent_parser_provider: str = "ollama"
    intent_parser_model: str = "llama3:8b"
    module_generator_provider: str = "ollama"
    module_generator_model: str = "codellama:13b"
    workflow_generator_provider: str = "ollama"
    workflow_generator_model: str = "llama3:8b"


@dataclass
class KnowledgeBaseConfig:
    """Paths to knowledge base resources."""
    tool_catalog: str = "data/tool_catalog/tool_catalog_20251125_003207.json"
    module_library: str = "nextflow-modules/"
    workflow_patterns: str = "docs/COMPOSITION_PATTERNS.md"
    container_images: str = "containers/images/"


@dataclass
class DataConfig:
    """Data management configuration."""
    reference_cache: str = "data/references/"
    download_sources: Dict[str, str] = field(default_factory=lambda: {
        "ensembl": "ftp://ftp.ensembl.org/pub/",
        "ucsc": "https://hgdownload.soe.ucsc.edu/",
        "gencode": "https://ftp.ebi.ac.uk/pub/databases/gencode/",
        "refseq": "https://ftp.ncbi.nlm.nih.gov/genomes/refseq/"
    })


@dataclass
class OutputConfig:
    """Output settings."""
    workflow_dir: str = "generated_workflows/"
    visualization_dir: str = "visualizations/"
    log_level: str = "INFO"


@dataclass
class Config:
    """Main configuration class."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    knowledge_base: KnowledgeBaseConfig = field(default_factory=KnowledgeBaseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Base path for relative paths
    base_path: Path = field(default_factory=lambda: Path.cwd())
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        """Load configuration from YAML file."""
        config = cls()
        
        # Default config locations
        default_paths = [
            Path("config/composer.yaml"),
            Path("composer.yaml"),
            Path.home() / ".config/biopipelines/composer.yaml"
        ]
        
        if config_path:
            default_paths.insert(0, Path(config_path))
        
        # Find and load config
        for path in default_paths:
            if path.exists():
                logger.info(f"Loading configuration from {path}")
                with open(path) as f:
                    yaml_config = yaml.safe_load(f)
                config = cls._from_dict(yaml_config)
                config.base_path = path.parent.parent  # Assume config is in config/
                break
        else:
            logger.warning("No configuration file found, using defaults")
        
        return config
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        config = cls()
        
        if "llm" in data:
            llm_data = data["llm"]
            config.llm.default_provider = llm_data.get("default_provider", "ollama")
            
            # Parse provider configs
            for name, pconfig in llm_data.get("providers", {}).items():
                config.llm.providers[name] = LLMProviderConfig(
                    api_key=pconfig.get("api_key"),
                    model=pconfig.get("model", ""),
                    temperature=pconfig.get("temperature", 0.1),
                    max_tokens=pconfig.get("max_tokens", 4096),
                    host=pconfig.get("host"),
                    device=pconfig.get("device", "cpu")
                )
            
            # Parse component assignments
            components = llm_data.get("components", {})
            if "intent_parser" in components:
                config.llm.intent_parser_provider = components["intent_parser"].get("provider", "ollama")
                config.llm.intent_parser_model = components["intent_parser"].get("model", "llama3:8b")
        
        if "knowledge_base" in data:
            kb = data["knowledge_base"]
            config.knowledge_base.tool_catalog = kb.get("tool_catalog", config.knowledge_base.tool_catalog)
            config.knowledge_base.module_library = kb.get("module_library", config.knowledge_base.module_library)
            config.knowledge_base.workflow_patterns = kb.get("workflow_patterns", config.knowledge_base.workflow_patterns)
            config.knowledge_base.container_images = kb.get("container_images", config.knowledge_base.container_images)
        
        if "data" in data:
            d = data["data"]
            config.data.reference_cache = d.get("reference_cache", config.data.reference_cache)
            if "download_sources" in d:
                config.data.download_sources.update(d["download_sources"])
        
        if "output" in data:
            o = data["output"]
            config.output.workflow_dir = o.get("workflow_dir", config.output.workflow_dir)
            config.output.visualization_dir = o.get("visualization_dir", config.output.visualization_dir)
            config.output.log_level = o.get("log_level", config.output.log_level)
        
        return config
    
    def resolve_path(self, path: str) -> Path:
        """Resolve relative path against base path."""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.base_path / p
    
    def get_llm_config(self, provider: Optional[str] = None) -> LLMProviderConfig:
        """Get LLM provider configuration."""
        provider = provider or self.llm.default_provider
        if provider in self.llm.providers:
            return self.llm.providers[provider]
        
        # Return default config for provider
        defaults = {
            "ollama": LLMProviderConfig(
                host="http://localhost:11434",
                model="llama3:8b",
                temperature=0.1
            ),
            "openai": LLMProviderConfig(
                api_key=os.environ.get("OPENAI_API_KEY"),
                model="gpt-4-turbo-preview",
                temperature=0.1
            ),
            "anthropic": LLMProviderConfig(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                model="claude-3-opus-20240229",
                temperature=0.1
            )
        }
        return defaults.get(provider, LLMProviderConfig())
    
    def to_yaml(self) -> str:
        """Export configuration to YAML string."""
        data = {
            "llm": {
                "default_provider": self.llm.default_provider,
                "providers": {
                    name: {
                        "model": p.model,
                        "temperature": p.temperature,
                        "max_tokens": p.max_tokens,
                        "host": p.host
                    }
                    for name, p in self.llm.providers.items()
                }
            },
            "knowledge_base": {
                "tool_catalog": self.knowledge_base.tool_catalog,
                "module_library": self.knowledge_base.module_library,
                "workflow_patterns": self.knowledge_base.workflow_patterns,
                "container_images": self.knowledge_base.container_images
            },
            "data": {
                "reference_cache": self.data.reference_cache,
                "download_sources": self.data.download_sources
            },
            "output": {
                "workflow_dir": self.output.workflow_dir,
                "visualization_dir": self.output.visualization_dir,
                "log_level": self.output.log_level
            }
        }
        return yaml.dump(data, default_flow_style=False)
