"""
nf-core Module Indexer
======================

Indexes nf-core modules for enhanced RAG context.

Features:
- Clone/update nf-core modules repository
- Parse module metadata (main.nf, meta.yml)
- Extract process definitions
- Build searchable index
"""

import asyncio
import json
import logging
import re
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class NFCoreModule:
    """Represents an nf-core module."""
    name: str
    path: str
    description: str = ""
    keywords: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    authors: List[str] = field(default_factory=list)
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    container: str = ""
    conda: str = ""
    process_code: str = ""
    analysis_types: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "path": self.path,
            "description": self.description,
            "keywords": self.keywords,
            "tools": self.tools,
            "authors": self.authors,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "container": self.container,
            "conda": self.conda,
            "process_code": self.process_code,
            "analysis_types": self.analysis_types,
        }
    
    def get_searchable_text(self) -> str:
        """Get text for search indexing."""
        parts = [
            self.name,
            self.description,
            " ".join(self.keywords),
            " ".join(self.tools),
        ]
        
        for inp in self.inputs:
            parts.append(inp.get("description", ""))
        
        for out in self.outputs:
            parts.append(out.get("description", ""))
        
        return " ".join(filter(None, parts))


class NFCoreIndexer:
    """
    Indexes nf-core modules for RAG retrieval.
    
    Clones the nf-core modules repository and parses
    module definitions for use in workflow generation.
    """
    
    NF_CORE_MODULES_URL = "https://github.com/nf-core/modules.git"
    
    # Map tool names to analysis types
    TOOL_TO_ANALYSIS = {
        # RNA-seq
        "star": ["rna-seq"],
        "salmon": ["rna-seq"],
        "kallisto": ["rna-seq"],
        "rsem": ["rna-seq"],
        "deseq2": ["rna-seq"],
        "htseq": ["rna-seq"],
        "featurecounts": ["rna-seq"],
        "stringtie": ["rna-seq"],
        "rseqc": ["rna-seq"],
        
        # DNA-seq / Variant calling
        "bwa": ["dna-seq", "chip-seq"],
        "bowtie2": ["dna-seq", "chip-seq", "methylation"],
        "gatk": ["dna-seq"],
        "bcftools": ["dna-seq"],
        "freebayes": ["dna-seq"],
        "deepvariant": ["dna-seq"],
        "strelka": ["dna-seq"],
        "mutect2": ["dna-seq"],
        "snpeff": ["dna-seq"],
        "vep": ["dna-seq"],
        
        # ChIP-seq / ATAC-seq
        "macs2": ["chip-seq", "atac-seq"],
        "macs3": ["chip-seq", "atac-seq"],
        "homer": ["chip-seq", "atac-seq"],
        "deeptools": ["chip-seq", "atac-seq"],
        "phantompeakqualtools": ["chip-seq"],
        
        # Methylation
        "bismark": ["methylation"],
        "methyldackel": ["methylation"],
        "bwameth": ["methylation"],
        
        # Metagenomics
        "kraken2": ["metagenomics"],
        "metaphlan": ["metagenomics"],
        "diamond": ["metagenomics"],
        "megahit": ["metagenomics"],
        "spades": ["metagenomics"],
        "humann": ["metagenomics"],
        
        # scRNA-seq
        "cellranger": ["scrna-seq"],
        "star_solo": ["scrna-seq"],
        "velocyto": ["scrna-seq"],
        "alevin": ["scrna-seq"],
        
        # Long-read
        "minimap2": ["long-read"],
        "nanofilt": ["long-read"],
        "nanoplot": ["long-read"],
        "medaka": ["long-read"],
        "flye": ["long-read"],
        
        # QC tools
        "fastqc": ["rna-seq", "dna-seq", "chip-seq", "atac-seq", "methylation"],
        "multiqc": ["rna-seq", "dna-seq", "chip-seq", "atac-seq", "methylation"],
        "fastp": ["rna-seq", "dna-seq", "chip-seq", "atac-seq"],
        "trimgalore": ["rna-seq", "dna-seq", "chip-seq", "methylation"],
        "cutadapt": ["rna-seq", "dna-seq"],
        
        # General
        "samtools": ["rna-seq", "dna-seq", "chip-seq", "atac-seq", "methylation"],
        "picard": ["rna-seq", "dna-seq", "chip-seq", "atac-seq"],
        "bedtools": ["chip-seq", "atac-seq", "dna-seq"],
    }
    
    def __init__(self, cache_dir: str = "~/.cache/biopipelines/nf-core"):
        """
        Initialize indexer.
        
        Args:
            cache_dir: Directory to cache cloned repo
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.modules_dir = self.cache_dir / "modules"
        self.modules: List[NFCoreModule] = []
        self.last_updated: Optional[datetime] = None
    
    async def update_repository(self, force: bool = False) -> bool:
        """
        Clone or update the nf-core modules repository.
        
        Args:
            force: Force re-clone even if exists
            
        Returns:
            True if successful
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if self.modules_dir.exists() and not force:
            # Try to pull updates
            try:
                process = await asyncio.create_subprocess_exec(
                    "git", "pull", "--quiet",
                    cwd=str(self.modules_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.wait()
                logger.info("Updated nf-core modules repository")
                return True
            except Exception as e:
                logger.warning(f"Failed to update repo: {e}")
                # Continue to clone
        
        # Clone repository
        if self.modules_dir.exists():
            import shutil
            shutil.rmtree(self.modules_dir)
        
        try:
            process = await asyncio.create_subprocess_exec(
                "git", "clone", "--depth", "1", 
                self.NF_CORE_MODULES_URL, str(self.modules_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info("Cloned nf-core modules repository")
                return True
            else:
                logger.error(f"Git clone failed: {stderr.decode()}")
                return False
        except Exception as e:
            logger.error(f"Failed to clone repository: {e}")
            return False
    
    async def index_modules(self, categories: List[str] = None) -> int:
        """
        Index all nf-core modules.
        
        Args:
            categories: Optional list of categories to index
            
        Returns:
            Number of modules indexed
        """
        modules_path = self.modules_dir / "modules" / "nf-core"
        
        if not modules_path.exists():
            logger.error(f"Modules path not found: {modules_path}")
            return 0
        
        self.modules = []
        count = 0
        
        for tool_dir in modules_path.iterdir():
            if not tool_dir.is_dir():
                continue
            
            # Handle both single-level and multi-level modules
            # e.g., modules/nf-core/star/align vs modules/nf-core/fastqc
            meta_files = list(tool_dir.rglob("meta.yml"))
            
            for meta_file in meta_files:
                try:
                    module = await self._parse_module(meta_file.parent)
                    if module:
                        self.modules.append(module)
                        count += 1
                except Exception as e:
                    logger.debug(f"Failed to parse {meta_file}: {e}")
        
        self.last_updated = datetime.now()
        logger.info(f"Indexed {count} nf-core modules")
        return count
    
    async def _parse_module(self, module_dir: Path) -> Optional[NFCoreModule]:
        """
        Parse a single module directory.
        
        Args:
            module_dir: Path to module directory
            
        Returns:
            NFCoreModule or None
        """
        meta_file = module_dir / "meta.yml"
        main_file = module_dir / "main.nf"
        
        if not meta_file.exists():
            return None
        
        # Parse meta.yml
        try:
            meta = yaml.safe_load(meta_file.read_text())
        except Exception as e:
            logger.debug(f"Failed to parse meta.yml: {e}")
            return None
        
        # Get relative path from modules dir
        rel_path = str(module_dir.relative_to(self.modules_dir / "modules" / "nf-core"))
        name = rel_path.replace("/", "_")
        
        # Extract tools from meta
        tools = []
        if "tools" in meta:
            for tool_entry in meta["tools"]:
                if isinstance(tool_entry, dict):
                    tools.extend(tool_entry.keys())
                elif isinstance(tool_entry, str):
                    tools.append(tool_entry)
        
        # Determine analysis types from tools
        analysis_types = set()
        for tool in tools:
            tool_lower = tool.lower()
            if tool_lower in self.TOOL_TO_ANALYSIS:
                analysis_types.update(self.TOOL_TO_ANALYSIS[tool_lower])
        
        # Extract inputs/outputs
        inputs = meta.get("input", [])
        outputs = meta.get("output", [])
        
        # Parse main.nf for process code
        process_code = ""
        container = ""
        conda = ""
        
        if main_file.exists():
            try:
                content = main_file.read_text()
                
                # Extract process block
                process_match = re.search(
                    r'process\s+\w+\s*\{.*?\}', 
                    content, 
                    re.DOTALL
                )
                if process_match:
                    process_code = process_match.group(0)
                
                # Extract container
                container_match = re.search(
                    r"container\s+['\"]([^'\"]+)['\"]",
                    content
                )
                if container_match:
                    container = container_match.group(1)
                
                # Extract conda
                conda_match = re.search(
                    r"conda\s+['\"]([^'\"]+)['\"]",
                    content
                )
                if conda_match:
                    conda = conda_match.group(1)
                    
            except Exception as e:
                logger.debug(f"Failed to parse main.nf: {e}")
        
        return NFCoreModule(
            name=name,
            path=rel_path,
            description=meta.get("description", ""),
            keywords=meta.get("keywords", []),
            tools=tools,
            authors=meta.get("authors", []),
            inputs=inputs if isinstance(inputs, list) else [inputs],
            outputs=outputs if isinstance(outputs, list) else [outputs],
            container=container,
            conda=conda,
            process_code=process_code,
            analysis_types=list(analysis_types),
        )
    
    def search(self, query: str, 
               analysis_type: str = None,
               limit: int = 10) -> List[Tuple[NFCoreModule, float]]:
        """
        Search for modules matching query.
        
        Args:
            query: Search query
            analysis_type: Optional analysis type filter
            limit: Maximum results
            
        Returns:
            List of (module, score) tuples
        """
        query_terms = query.lower().split()
        results = []
        
        for module in self.modules:
            # Calculate relevance score
            score = 0.0
            searchable = module.get_searchable_text().lower()
            
            for term in query_terms:
                if term in module.name.lower():
                    score += 2.0
                if term in searchable:
                    score += 1.0
                if any(term in tool.lower() for tool in module.tools):
                    score += 1.5
                if any(term in kw.lower() for kw in module.keywords):
                    score += 1.5
            
            # Boost if matches analysis type
            if analysis_type and analysis_type in module.analysis_types:
                score *= 1.5
            
            if score > 0:
                results.append((module, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def get_modules_for_analysis(self, analysis_type: str,
                                  limit: int = 20) -> List[NFCoreModule]:
        """
        Get modules relevant to an analysis type.
        
        Args:
            analysis_type: Type of analysis
            limit: Maximum modules
            
        Returns:
            List of relevant modules
        """
        results = [
            m for m in self.modules 
            if analysis_type in m.analysis_types
        ]
        return results[:limit]
    
    def get_module(self, name: str) -> Optional[NFCoreModule]:
        """Get module by name."""
        for module in self.modules:
            if module.name == name or module.path == name:
                return module
        return None
    
    def export_index(self, output_file: str):
        """Export index to JSON file."""
        data = {
            "modules": [m.to_dict() for m in self.modules],
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "count": len(self.modules),
        }
        
        Path(output_file).write_text(json.dumps(data, indent=2))
        logger.info(f"Exported {len(self.modules)} modules to {output_file}")
    
    def import_index(self, input_file: str) -> int:
        """
        Import index from JSON file.
        
        Args:
            input_file: Path to JSON index
            
        Returns:
            Number of modules imported
        """
        path = Path(input_file)
        if not path.exists():
            return 0
        
        try:
            data = json.loads(path.read_text())
            self.modules = []
            
            for module_data in data.get("modules", []):
                module = NFCoreModule(
                    name=module_data["name"],
                    path=module_data["path"],
                    description=module_data.get("description", ""),
                    keywords=module_data.get("keywords", []),
                    tools=module_data.get("tools", []),
                    authors=module_data.get("authors", []),
                    inputs=module_data.get("inputs", []),
                    outputs=module_data.get("outputs", []),
                    container=module_data.get("container", ""),
                    conda=module_data.get("conda", ""),
                    process_code=module_data.get("process_code", ""),
                    analysis_types=module_data.get("analysis_types", []),
                )
                self.modules.append(module)
            
            if data.get("last_updated"):
                self.last_updated = datetime.fromisoformat(data["last_updated"])
            
            logger.info(f"Imported {len(self.modules)} modules from {input_file}")
            return len(self.modules)
            
        except Exception as e:
            logger.error(f"Failed to import index: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get indexer statistics."""
        analysis_counts = {}
        tool_counts = {}
        
        for module in self.modules:
            for at in module.analysis_types:
                analysis_counts[at] = analysis_counts.get(at, 0) + 1
            for tool in module.tools:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
        
        return {
            "total_modules": len(self.modules),
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "analysis_type_counts": analysis_counts,
            "top_tools": sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:20],
        }
