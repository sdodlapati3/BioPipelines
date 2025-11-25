"""
Tool Selector
=============

Queries the tool catalog to find appropriate tools for analysis.

Supports:
- Exact matching by tool name
- Fuzzy matching for similar tools
- Category-based search
- Container-aware selection

Example:
    selector = ToolSelector(catalog_path)
    tools = selector.find_tools("rna_seq_differential_expression")
    # Returns: [Tool(star), Tool(featurecounts), Tool(deseq2)]
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """Represents a bioinformatics tool."""
    name: str
    container: str
    path: str = ""
    version: str = ""
    category: str = ""
    description: str = ""
    
    def __hash__(self):
        return hash((self.name, self.container))
    
    def __eq__(self, other):
        if isinstance(other, Tool):
            return self.name == other.name and self.container == other.container
        return False


@dataclass
class ToolMatch:
    """A tool match with relevance score."""
    tool: Tool
    score: float
    match_type: str  # "exact", "fuzzy", "category"
    reason: str = ""


# Mapping from analysis types to required tools
ANALYSIS_TOOL_MAP = {
    "rna_seq_basic": {
        "required": ["fastqc", "star", "samtools"],
        "recommended": ["multiqc", "picard"],
        "quantification": ["featurecounts", "salmon", "htseq"],
    },
    "rna_seq_differential_expression": {
        "required": ["fastqc", "star", "featurecounts"],
        "recommended": ["multiqc", "samtools"],
        "analysis": ["deseq2", "edger"],
    },
    "rna_seq_de_novo_assembly": {
        "required": ["fastqc", "trinity"],
        "recommended": ["multiqc", "rsem", "blast"],
    },
    "chip_seq_peak_calling": {
        "required": ["fastqc", "bwa", "samtools", "macs2"],
        "recommended": ["multiqc", "picard", "deeptools"],
        "analysis": ["homer"],
    },
    "atac_seq": {
        "required": ["fastqc", "bowtie2", "samtools", "macs2"],
        "recommended": ["multiqc", "picard", "deeptools"],
    },
    "wgs_variant_calling": {
        "required": ["fastqc", "bwa", "samtools", "gatk"],
        "recommended": ["multiqc", "picard", "bcftools"],
        "alternative_callers": ["freebayes", "deepvariant"],
    },
    "somatic_variant_calling": {
        "required": ["bwa", "samtools", "gatk"],
        "recommended": ["picard", "bcftools"],
        "callers": ["mutect2", "varscan", "strelka2", "manta"],
    },
    "structural_variant_detection": {
        "required": ["bwa", "samtools"],
        "callers": ["manta", "delly", "lumpy"],
    },
    "single_cell_rna_seq": {
        "required": ["cellranger"],
        "analysis": ["seurat", "scanpy"],
        "recommended": ["velocyto"],
    },
    "metagenomics_profiling": {
        "required": ["fastqc"],
        "profilers": ["kraken2", "metaphlan", "bracken"],
        "recommended": ["multiqc"],
    },
    "metagenomics_assembly": {
        "required": ["fastqc"],
        "assemblers": ["megahit", "metaspades"],
        "annotation": ["prokka"],
    },
    "bisulfite_seq_methylation": {
        "required": ["fastqc", "bismark"],
        "recommended": ["multiqc", "trim_galore"],
    },
    "hic_chromatin_interaction": {
        "required": ["bwa"],
        "processors": ["hicpro", "juicer"],
    },
    "long_read_assembly": {
        "required": ["nanoplot"],
        "assemblers": ["flye", "canu"],
        "polishing": ["racon", "medaka"],
    },
    "genome_annotation": {
        "required": [],
        "annotators": ["prokka", "augustus"],
        "recommended": ["blast"],
    },
    # Spatial transcriptomics
    "spatial_transcriptomics": {
        "required": ["spaceranger"],
        "analysis": ["scanpy", "squidpy", "giotto"],
        "visualization": ["deeptools"],
    },
    "spatial_visium": {
        "required": ["spaceranger"],
        "analysis": ["scanpy", "squidpy"],
        "recommended": ["seurat"],
    },
    "spatial_slide_seq": {
        "required": ["fastqc"],
        "alignment": ["star", "bowtie2"],
        "analysis": ["scanpy", "squidpy"],
    },
    "spatial_xenium": {
        "required": ["xeniumranger"],
        "analysis": ["scanpy", "squidpy"],
    },
    # Long-read RNA-seq
    "long_read_rna_seq": {
        "required": ["minimap2", "samtools"],
        "analysis": ["flair", "stringtie", "isoquant"],
        "qc": ["nanoplot", "rseqc"],
    },
    "long_read_isoseq": {
        "required": ["isoseq3", "minimap2", "samtools"],
        "analysis": ["pigeon", "sqanti3", "tama"],
        "clustering": ["isoclust"],
    },
    "long_read_direct_rna": {
        "required": ["minimap2", "samtools", "nanopolish"],
        "modification": ["tombo", "eligos", "m6anet"],
        "qc": ["nanoplot"],
    },
    # Multi-omics integration
    "multi_omics_integration": {
        "required": [],
        "integration": ["mofa", "mixomics", "diablo"],
        "visualization": ["deeptools"],
    },
    "rna_atac_integration": {
        "required": ["cellranger-arc"],
        "analysis": ["seurat", "scanpy", "signac"],
        "recommended": ["chromvar", "cicero"],
    },
    "proteogenomics": {
        "required": ["star", "samtools"],
        "ms_tools": ["maxquant", "msgf", "percolator"],
        "integration": ["proteogenomics_tools"],
    },
    "multi_modal_scrna": {
        "required": ["cellranger"],
        "analysis": ["seurat", "scanpy", "totalvi"],
        "cite_seq": ["cite_seq_count", "dsb"],
    },
}


class ToolSelector:
    """
    Selects appropriate tools from the catalog for a given analysis.
    """
    
    def __init__(self, catalog_path: str):
        """
        Initialize tool selector with tool catalog.
        
        Args:
            catalog_path: Path to tool_catalog JSON file or directory containing it
        """
        self.catalog_path = Path(catalog_path)
        self.tools: Dict[str, Tool] = {}
        self.container_tools: Dict[str, Set[str]] = {}
        
        self._load_catalog()
    
    def _load_catalog(self) -> None:
        """Load tool catalog from JSON."""
        catalog_file = self.catalog_path
        
        # If path is a directory, look for the latest catalog
        if self.catalog_path.is_dir():
            # Try to find tool_catalog_latest.json symlink first
            latest = self.catalog_path / "tool_catalog_latest.json"
            if latest.exists():
                catalog_file = latest
            else:
                # Find the most recent tool_catalog JSON
                json_files = list(self.catalog_path.glob("tool_catalog_*.json"))
                if json_files:
                    catalog_file = max(json_files, key=lambda p: p.stat().st_mtime)
                else:
                    logger.warning(f"No tool catalog JSON found in: {self.catalog_path}")
                    return
        
        if not catalog_file.exists():
            logger.warning(f"Tool catalog not found: {catalog_file}")
            return
        
        logger.info(f"Loading tool catalog from: {catalog_file}")
        
        with open(catalog_file) as f:
            data = json.load(f)
        
        for container, info in data.get("containers", {}).items():
            container_name = container.replace("_1.0.0", "")
            self.container_tools[container_name] = set()
            
            for tool_path in info.get("tools", []):
                # Extract tool name from path
                tool_name = Path(tool_path).name.lower()
                
                # Skip common system tools
                if tool_name in ["ls", "cat", "grep", "awk", "sed", "python", "perl"]:
                    continue
                
                tool = Tool(
                    name=tool_name,
                    container=container_name,
                    path=tool_path
                )
                
                # Store by name (may have multiple containers)
                if tool_name not in self.tools:
                    self.tools[tool_name] = tool
                
                self.container_tools[container_name].add(tool_name)
        
        logger.info(f"Loaded {len(self.tools)} unique tools from {len(self.container_tools)} containers")
    
    def find_tool(self, name: str) -> Optional[Tool]:
        """
        Find a specific tool by name.
        
        Args:
            name: Tool name (case-insensitive)
            
        Returns:
            Tool if found, None otherwise
        """
        name_lower = name.lower()
        
        # Exact match
        if name_lower in self.tools:
            return self.tools[name_lower]
        
        # Try common variations
        variations = [
            name_lower,
            name_lower.replace("-", ""),
            name_lower.replace("_", ""),
            name_lower.replace("-", "_"),
            name_lower.replace("_", "-"),
        ]
        
        for var in variations:
            if var in self.tools:
                return self.tools[var]
        
        return None
    
    def fuzzy_search(self, query: str, threshold: float = 0.6) -> List[ToolMatch]:
        """
        Search for tools using fuzzy matching.
        
        Args:
            query: Search query
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of ToolMatch sorted by score
        """
        query_lower = query.lower()
        matches = []
        
        for name, tool in self.tools.items():
            # Calculate similarity
            score = SequenceMatcher(None, query_lower, name).ratio()
            
            # Also check if query is substring
            if query_lower in name:
                score = max(score, 0.8)
            
            if score >= threshold:
                matches.append(ToolMatch(
                    tool=tool,
                    score=score,
                    match_type="fuzzy",
                    reason=f"Similarity: {score:.2f}"
                ))
        
        return sorted(matches, key=lambda m: m.score, reverse=True)
    
    def find_tools_for_analysis(self, analysis_type: str) -> Dict[str, List[Tool]]:
        """
        Find all tools needed for a specific analysis type.
        
        Args:
            analysis_type: Analysis type (e.g., "rna_seq_differential_expression")
            
        Returns:
            Dict mapping categories to lists of Tools
        """
        result = {}
        
        tool_spec = ANALYSIS_TOOL_MAP.get(analysis_type, {})
        
        for category, tool_names in tool_spec.items():
            found_tools = []
            for name in tool_names:
                tool = self.find_tool(name)
                if tool:
                    found_tools.append(tool)
                else:
                    logger.warning(f"Tool not found in catalog: {name}")
            
            if found_tools:
                result[category] = found_tools
        
        return result
    
    def get_tools_in_container(self, container: str) -> List[Tool]:
        """
        Get all tools in a specific container.
        
        Args:
            container: Container name
            
        Returns:
            List of Tools
        """
        tools = []
        container_lower = container.lower().replace("-", "_").replace("_", "-")
        
        for name, ctools in self.container_tools.items():
            if container_lower in name.lower() or name.lower() in container_lower:
                for tool_name in ctools:
                    if tool_name in self.tools:
                        tools.append(self.tools[tool_name])
        
        return tools
    
    def check_tool_availability(self, tool_names: List[str]) -> Dict[str, bool]:
        """
        Check availability of multiple tools.
        
        Args:
            tool_names: List of tool names to check
            
        Returns:
            Dict mapping tool names to availability
        """
        return {name: self.find_tool(name) is not None for name in tool_names}
    
    def suggest_alternatives(self, tool_name: str) -> List[ToolMatch]:
        """
        Suggest alternative tools if requested tool is not available.
        
        Args:
            tool_name: Tool that was requested
            
        Returns:
            List of alternative ToolMatches
        """
        # Common alternatives mapping
        alternatives = {
            "star": ["hisat2", "bowtie2", "bwa"],
            "hisat2": ["star", "bowtie2"],
            "bowtie2": ["bwa", "bowtie"],
            "bwa": ["bowtie2", "bwa-mem2"],
            "salmon": ["kallisto", "rsem"],
            "kallisto": ["salmon", "rsem"],
            "deseq2": ["edger", "limma"],
            "edger": ["deseq2", "limma"],
            "macs2": ["homer", "sicer"],
            "gatk": ["freebayes", "bcftools"],
            "freebayes": ["gatk", "bcftools"],
            "kraken2": ["metaphlan", "centrifuge"],
            "metaphlan": ["kraken2", "bracken"],
            "flye": ["canu", "wtdbg2"],
            "canu": ["flye", "wtdbg2"],
        }
        
        alt_names = alternatives.get(tool_name.lower(), [])
        
        matches = []
        for alt in alt_names:
            tool = self.find_tool(alt)
            if tool:
                matches.append(ToolMatch(
                    tool=tool,
                    score=0.7,
                    match_type="alternative",
                    reason=f"Alternative to {tool_name}"
                ))
        
        # Also try fuzzy search
        fuzzy = self.fuzzy_search(tool_name, threshold=0.5)[:3]
        matches.extend(fuzzy)
        
        return matches
    
    def list_containers(self) -> List[str]:
        """List all available containers."""
        return list(self.container_tools.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get catalog statistics."""
        return {
            "total_tools": len(self.tools),
            "containers": len(self.container_tools),
            "tools_per_container": {
                name: len(tools)
                for name, tools in self.container_tools.items()
            }
        }
