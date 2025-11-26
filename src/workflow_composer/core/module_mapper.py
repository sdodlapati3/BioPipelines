"""
Module Mapper
=============

Maps tools to Nextflow modules and handles module creation.

Supports:
- Finding existing modules for tools
- Auto-generating new modules using LLM
- Module validation
- Container-module compatibility checking

Example:
    mapper = ModuleMapper(module_dir)
    module = mapper.find_module("star")
    
    # Auto-generate if missing
    if not module:
        module = mapper.create_module(tool, llm)
"""

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..llm.base import LLMAdapter, Message

logger = logging.getLogger(__name__)


@dataclass
class Module:
    """Represents a Nextflow module."""
    name: str
    path: Path
    tool_name: str
    container: str
    processes: List[str] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    
    def get_import_statement(self) -> str:
        """Generate Nextflow import statement."""
        process_list = "; ".join(self.processes)
        rel_path = f"./modules/{self.path.parent.name}/{self.path.name}"
        return f"include {{ {process_list} }} from '{rel_path}'"


# Mapping from tool names to module info

def load_tool_mappings(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load tool mappings from YAML config."""
    if not config_path:
        # Default location relative to this file
        base_dir = Path(__file__).parent.parent.parent.parent
        config_path = base_dir / "config" / "tool_mappings.yaml"
    
    if not config_path.exists():
        logger.warning(f"Tool mappings not found at {config_path}")
        return {}
        
    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load tool mappings: {e}")
        return {}



class ModuleMapper:
    """
    Maps tools to Nextflow modules and handles module discovery/creation.
    """
    
    def __init__(self, module_dir: str, additional_dirs: List[str] = None):
        """
        Initialize module mapper.
        
        Args:
            module_dir: Path to primary nextflow-modules directory
            additional_dirs: Additional module directories to scan
        """
        self.module_dir = Path(module_dir)
        self.additional_dirs = [Path(d) for d in (additional_dirs or [])]
        self.modules: Dict[str, Module] = {}
        
        # Load tool mappings
        mappings = load_tool_mappings()
        self.tool_module_map = mappings.get("tools", {})
        
        # Create container map from tools
        self.tool_container_map = {}
        for tool, info in self.tool_module_map.items():
            if "container" in info:
                self.tool_container_map[tool] = info["container"]
        
        # Add explicit container mappings if any (from the yaml structure I created, it's all under tools now, but I added some extras at the end)
        # Actually my yaml structure put everything under 'tools', but I also added some standalone container mappings at the end of the yaml file
        # Let's check the yaml content again.
        # Ah, I put 'stringtie' and 'medaka' at the end under 'tools' but without 'module' info.
        # So iterating over self.tool_module_map covers everything.
        
        self._scan_modules()
        
        # Scan additional directories
        for extra_dir in self.additional_dirs:
            self._scan_flat_modules(extra_dir)
    
    def _scan_modules(self) -> None:
        """Scan module directory for existing modules."""
        if not self.module_dir.exists():
            logger.warning(f"Module directory not found: {self.module_dir}")
            return
        
        # Handle both structures:
        # 1. category/tool/main.nf (nf-core style)
        # 2. category/tool.nf (flat style)
        
        for category_dir in self.module_dir.iterdir():
            if not category_dir.is_dir():
                continue
            
            # Check for nf-core style: category/tool/main.nf
            for tool_dir in category_dir.iterdir():
                if tool_dir.is_dir():
                    main_nf = tool_dir / "main.nf"
                    if main_nf.exists():
                        module_name = tool_dir.name
                        processes = self._extract_processes(main_nf)
                        tool_name = module_name.lower()
                        
                        container = self.tool_container_map.get(tool_name, "base")
                        
                        self.modules[tool_name] = Module(
                            name=module_name,
                            path=main_nf,
                            tool_name=tool_name,
                            container=container,
                            processes=processes
                        )
            
            # Also check for flat style: category/*.nf
            for module_file in category_dir.glob("*.nf"):
                module_name = module_file.stem
                tool_name = module_name.lower()
                
                # Skip if already found in subdirectory
                if tool_name in self.modules:
                    continue
                
                # Get container from mapping
                container = self.tool_container_map.get(tool_name, "base")
                
                self.modules[tool_name] = Module(
                    name=module_name,
                    path=module_file,
                    tool_name=tool_name,
                    container=container,
                    processes=processes
                )
        
        logger.info(f"Found {len(self.modules)} modules in {self.module_dir}")
    
    def _scan_flat_modules(self, module_dir: Path) -> None:
        """Scan a directory with flat .nf module files (category/tool.nf style)."""
        if not module_dir.exists():
            logger.warning(f"Additional module directory not found: {module_dir}")
            return
        
        count_before = len(self.modules)
        
        for category_dir in module_dir.iterdir():
            if not category_dir.is_dir():
                continue
            
            for module_file in category_dir.glob("*.nf"):
                module_name = module_file.stem
                tool_name = module_name.lower()
                
                # Skip if already found (nf-core style takes precedence)
                if tool_name in self.modules:
                    continue
                
                processes = self._extract_processes(module_file)
                container = self.tool_container_map.get(tool_name, "base")
                
                self.modules[tool_name] = Module(
                    name=module_name,
                    path=module_file,
                    tool_name=tool_name,
                    container=container,
                    processes=processes
                )
        
        count_added = len(self.modules) - count_before
        logger.info(f"Found {count_added} additional modules in {module_dir}")
    
    def _extract_processes(self, module_path: Path) -> List[str]:
        """Extract process names from a module file."""
        processes = []
        
        try:
            content = module_path.read_text()
            # Match process definitions
            pattern = r"process\s+(\w+)\s*{"
            matches = re.findall(pattern, content)
            processes = matches
        except Exception as e:
            logger.warning(f"Failed to parse {module_path}: {e}")
        
        return processes
    
    # Tool name aliases - map common tool names to actual module names
    TOOL_ALIASES = {
        # Alignment aliases
        "bwa": "bwamem",
        "bwa-mem": "bwamem",
        "bwa_mem": "bwamem",
        
        # Variant calling aliases
        "gatk": "gatk_haplotypecaller",
        "haplotypecaller": "gatk_haplotypecaller",
        "gatk_hc": "gatk_haplotypecaller",
        
        # Trimming aliases
        "trimgalore": "trim_galore",
        "trim-galore": "trim_galore",
        
        # Utilities aliases
        "mark_duplicates": "markduplicates",
        "picard_markduplicates": "markduplicates",
        
        # scRNA aliases
        "cellranger": "starsolo",  # fallback to starsolo for scRNA
        "cell_ranger": "starsolo",
        
        # Peak calling
        "macs": "macs2",
        
        # QC aliases
        "multi_qc": "multiqc",
        
        # deeptools aliases
        "deep_tools": "deeptools",
        "deep-tools": "deeptools",
        
        # Methylation aliases
        "bismark_methylation": "bismark_extractor",
        "methylation_extractor": "bismark_extractor",
        
        # Hi-C aliases  
        "cooler": "cooler_cload",
        "pairtools": "pairtools_parse",
    }
    
    def find_module(self, tool_name: str) -> Optional[Module]:
        """
        Find a module for a given tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Module if found, None otherwise
        """
        tool_lower = tool_name.lower()
        
        # Check aliases first
        if tool_lower in self.TOOL_ALIASES:
            alias = self.TOOL_ALIASES[tool_lower]
            if alias in self.modules:
                return self.modules[alias]
        
        # Direct match
        if tool_lower in self.modules:
            return self.modules[tool_lower]
        
        # Try variations
        variations = [
            tool_lower,
            tool_lower.replace("-", ""),
            tool_lower.replace("_", ""),
            tool_lower.replace("-", "_"),
        ]
        
        for var in variations:
            if var in self.modules:
                return self.modules[var]
        
        # Check tool_module_map for known mappings
        if tool_lower in self.tool_module_map:
            info = self.tool_module_map[tool_lower]
            # Only if it has module info (some entries might just be container mappings)
            if "module" in info and "category" in info:
                expected_path = self.module_dir / info["category"] / info["module"]
                if expected_path.exists():
                    return Module(
                        name=info["module"].replace(".nf", ""),
                        path=expected_path,
                        tool_name=tool_lower,
                        container=self.tool_container_map.get(tool_lower, "base"),
                        processes=info.get("processes", [])
                    )
        
        return None
    
    def find_modules_for_tools(self, tool_names: List[str]) -> Dict[str, Optional[Module]]:
        """
        Find modules for multiple tools.
        
        Args:
            tool_names: List of tool names
            
        Returns:
            Dict mapping tool names to Modules (or None if not found)
        """
        return {name: self.find_module(name) for name in tool_names}
    
    def get_missing_tools(self, tool_names: List[str]) -> List[str]:
        """
        Get list of tools without existing modules.
        
        Args:
            tool_names: List of tool names
            
        Returns:
            List of tool names without modules
        """
        return [name for name in tool_names if not self.find_module(name)]
    
    def create_module(
        self,
        tool_name: str,
        container: str,
        llm: LLMAdapter,
        description: str = ""
    ) -> Module:
        """
        Auto-generate a new module using LLM.
        
        Args:
            tool_name: Name of the tool
            container: Container to use
            llm: LLM adapter for code generation
            description: Optional description of desired functionality
            
        Returns:
            Generated Module
        """
        logger.info(f"Generating module for {tool_name}...")
        
        prompt = f"""Generate a Nextflow DSL2 module for the bioinformatics tool '{tool_name}'.

The module should:
1. Follow DSL2 best practices
2. Use container parameter: ${{params.containers.{container}}}
3. Have proper input/output declarations
4. Include helpful comments
5. Handle both single and paired-end data if applicable

{f"Additional requirements: {description}" if description else ""}

Generate ONLY the Nextflow code, no explanations."""

        messages = [
            Message.system("You are an expert Nextflow developer specializing in bioinformatics pipelines."),
            Message.user(prompt)
        ]
        
        response = llm.chat(messages, temperature=0.1)
        
        # Extract code from response
        code = response.content
        if "```" in code:
            # Extract from markdown code block
            lines = code.split("\n")
            in_block = False
            code_lines = []
            for line in lines:
                if line.startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    code_lines.append(line)
            code = "\n".join(code_lines)
        
        # Determine category
        category = self._guess_category(tool_name)
        
        # Create module file
        module_dir = self.module_dir / category
        module_dir.mkdir(parents=True, exist_ok=True)
        
        module_path = module_dir / f"{tool_name.lower()}.nf"
        module_path.write_text(code)
        
        logger.info(f"Created module: {module_path}")
        
        # Parse and register
        processes = self._extract_processes(module_path)
        
        module = Module(
            name=tool_name,
            path=module_path,
            tool_name=tool_name,
            container=container,
            processes=processes
        )
        
        self.modules[tool_name.lower()] = module
        
        return module
    
    def _guess_category(self, tool_name: str) -> str:
        """Guess appropriate category for a tool."""
        tool_lower = tool_name.lower()
        
        if tool_lower in self.tool_module_map:
            info = self.tool_module_map[tool_lower]
            if "category" in info:
                return info["category"]
        
        # Heuristic based on name
        if any(kw in tool_lower for kw in ["align", "map", "bwa", "star", "bowtie"]):
            return "alignment"
        if any(kw in tool_lower for kw in ["count", "quant", "salmon", "rsem"]):
            return "quantification"
        if any(kw in tool_lower for kw in ["qc", "fastqc", "multiqc"]):
            return "qc"
        if any(kw in tool_lower for kw in ["trim", "cut", "fastp"]):
            return "trimming"
        if any(kw in tool_lower for kw in ["peak", "macs", "homer"]):
            return "peaks"
        if any(kw in tool_lower for kw in ["variant", "call", "gatk", "vcf"]):
            return "variant_calling"
        
        return "utilities"
    
    def list_modules(self) -> List[str]:
        """List all available modules."""
        return list(self.modules.keys())
    
    def list_by_category(self) -> Dict[str, List[str]]:
        """List modules organized by category."""
        categories: Dict[str, List[str]] = {}
        
        for name, module in self.modules.items():
            category = module.path.parent.name
            if category not in categories:
                categories[category] = []
            categories[category].append(name)
        
        return categories
    
    def validate_module(self, module: Module) -> Dict[str, Any]:
        """
        Validate a module's syntax and structure.
        
        Args:
            module: Module to validate
            
        Returns:
            Dict with validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        if not module.path.exists():
            results["valid"] = False
            results["errors"].append(f"Module file not found: {module.path}")
            return results
        
        content = module.path.read_text()
        
        # Check for DSL2
        if "nextflow.enable.dsl=2" not in content and "nextflow.enable.dsl = 2" not in content:
            results["warnings"].append("DSL2 not explicitly enabled")
        
        # Check for process definitions
        if not module.processes:
            results["warnings"].append("No processes found in module")
        
        # Check for container definition
        if "container" not in content:
            results["warnings"].append("No container directive found")
        
        # Check for input/output
        if "input:" not in content:
            results["errors"].append("No input block found")
            results["valid"] = False
        
        if "output:" not in content:
            results["errors"].append("No output block found")
            results["valid"] = False
        
        return results
