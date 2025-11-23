"""
Container registry for AI agent discovery and invocation
"""
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class ContainerTool:
    """Represents a tool within a container"""
    name: str
    version: str
    purpose: str
    stage: str


class ContainerManifest:
    """
    Represents a container's manifest with all metadata
    Enables AI agents to discover and understand container capabilities
    """
    
    def __init__(self, manifest_path: Path):
        """Load manifest from JSON file"""
        with open(manifest_path) as f:
            self.data = json.load(f)
        self.path = manifest_path
        self.container_dir = manifest_path.parent
    
    @property
    def name(self) -> str:
        return self.data["name"]
    
    @property
    def version(self) -> str:
        return self.data["version"]
    
    @property
    def category(self) -> str:
        return self.data.get("category", "unknown")
    
    @property
    def capabilities(self) -> List[str]:
        return self.data.get("capabilities", [])
    
    @property
    def tools(self) -> List[ContainerTool]:
        """Get list of tools in this container"""
        tools_data = self.data.get("tools", [])
        return [
            ContainerTool(
                name=t["name"],
                version=t["version"],
                purpose=t.get("purpose", ""),
                stage=t.get("stage", "")
            )
            for t in tools_data
        ]
    
    @property
    def input_formats(self) -> List[str]:
        return self.data.get("input_formats", [])
    
    @property
    def output_formats(self) -> List[str]:
        return self.data.get("output_formats", [])
    
    def matches_query(self,
                     category: Optional[str] = None,
                     capability: Optional[str] = None,
                     input_format: Optional[str] = None,
                     tool_name: Optional[str] = None) -> bool:
        """
        Check if container matches AI agent query
        
        Args:
            category: Filter by category (e.g., "transcriptomics")
            capability: Filter by capability (e.g., "alignment")
            input_format: Filter by input format (e.g., "fastq")
            tool_name: Filter by specific tool (e.g., "STAR")
        
        Returns:
            True if container matches all specified criteria
        """
        if category and self.category != category:
            return False
        
        if capability and capability not in self.capabilities:
            return False
        
        if input_format and input_format not in self.input_formats:
            return False
        
        if tool_name:
            tool_names = [t.name.lower() for t in self.tools]
            if tool_name.lower() not in tool_names:
                return False
        
        return True
    
    def get_execution_command(self, params: Dict[str, Any]) -> str:
        """
        Generate execution command for AI agent
        
        Args:
            params: Dictionary of parameters (input_dir, output_dir, genome, etc.)
        
        Returns:
            Complete command string ready for execution
        """
        entrypoint = self.data["execution"]["entrypoint"]
        singularity_uri = self.data["execution"]["singularity_uri"]
        
        cmd_parts = [
            "singularity run",
            singularity_uri
        ]
        
        # Add parameters
        for key, value in params.items():
            if value is not None:
                cmd_parts.append(f"--{key} {value}")
        
        return " ".join(cmd_parts)
    
    def get_resource_requirements(self) -> Dict[str, Any]:
        """Get resource requirements for scheduling"""
        return self.data.get("resources", {})
    
    def get_ai_hints(self) -> Dict[str, Any]:
        """Get AI-specific hints for better agent understanding"""
        return self.data.get("ai_agent_hints", {})
    
    def __repr__(self) -> str:
        return f"ContainerManifest({self.name}:{self.version})"


class ContainerRegistry:
    """
    Registry of available containers for AI agent discovery
    """
    
    def __init__(self, containers_dir: Path):
        """
        Initialize registry from containers directory
        
        Args:
            containers_dir: Path to directory containing container subdirectories
        """
        self.containers_dir = Path(containers_dir)
        self.manifests: Dict[str, ContainerManifest] = {}
        self._load_manifests()
    
    def _load_manifests(self):
        """Load all container manifests from directory"""
        for manifest_file in self.containers_dir.glob("*/manifest.json"):
            manifest = ContainerManifest(manifest_file)
            self.manifests[manifest.name] = manifest
    
    def search(self, **criteria) -> List[ContainerManifest]:
        """
        AI agent searches for containers matching criteria
        
        Args:
            **criteria: Keyword arguments for filtering
                - category: str
                - capability: str
                - input_format: str
                - tool_name: str
        
        Returns:
            List of matching container manifests
        
        Example:
            >>> registry = ContainerRegistry("containers/")
            >>> rna_containers = registry.search(category="transcriptomics")
            >>> alignment_containers = registry.search(capability="alignment")
        """
        results = []
        for manifest in self.manifests.values():
            if manifest.matches_query(**criteria):
                results.append(manifest)
        return results
    
    def get_container(self, name: str) -> Optional[ContainerManifest]:
        """
        Get specific container by name
        
        Args:
            name: Container name (e.g., "biopipelines-rna-seq")
        
        Returns:
            ContainerManifest or None if not found
        """
        return self.manifests.get(name)
    
    def list_all(self) -> List[ContainerManifest]:
        """List all available containers"""
        return list(self.manifests.values())
    
    def get_categories(self) -> List[str]:
        """Get all available categories"""
        return list(set(m.category for m in self.manifests.values()))
    
    def get_capabilities(self) -> List[str]:
        """Get all available capabilities across containers"""
        capabilities = set()
        for manifest in self.manifests.values():
            capabilities.update(manifest.capabilities)
        return sorted(capabilities)
    
    def find_by_tool(self, tool_name: str) -> List[ContainerManifest]:
        """
        Find containers that include a specific tool
        
        Args:
            tool_name: Name of tool (e.g., "STAR", "samtools")
        
        Returns:
            List of containers containing the tool
        """
        return self.search(tool_name=tool_name)
    
    def recommend_for_query(self, user_query: str) -> List[ContainerManifest]:
        """
        AI agent recommends containers based on natural language query
        
        This is a simple keyword-based approach. Can be enhanced with LLM.
        
        Args:
            user_query: Natural language description of analysis
        
        Returns:
            Recommended containers ranked by relevance
        
        Example:
            >>> registry.recommend_for_query("I want to analyze RNA-seq data")
            [ContainerManifest(biopipelines-rna-seq:1.0.0)]
        """
        query_lower = user_query.lower()
        recommendations = []
        
        # Simple keyword matching (can be enhanced with embeddings/LLM)
        keywords_map = {
            "rna": ["transcriptomics"],
            "dna": ["genomics"],
            "variant": ["variant_calling"],
            "chip": ["epigenomics"],
            "atac": ["chromatin_accessibility"],
            "methylation": ["epigenomics"],
            "single-cell": ["single_cell"],
            "metagenome": ["metagenomics"]
        }
        
        for keyword, categories in keywords_map.items():
            if keyword in query_lower:
                for category in categories:
                    recommendations.extend(self.search(category=category))
        
        return recommendations
    
    def __len__(self) -> int:
        return len(self.manifests)
    
    def __repr__(self) -> str:
        return f"ContainerRegistry({len(self)} containers)"
