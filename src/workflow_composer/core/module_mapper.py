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
TOOL_MODULE_MAP = {
    # Alignment
    "star": {"category": "alignment", "module": "star.nf", "processes": ["STAR_INDEX", "STAR_ALIGN"]},
    "bwa": {"category": "alignment", "module": "bwa.nf", "processes": ["BWA_INDEX", "BWA_MEM"]},
    "bowtie2": {"category": "alignment", "module": "bowtie2.nf", "processes": ["BOWTIE2_BUILD", "BOWTIE2_ALIGN"]},
    "hisat2": {"category": "alignment", "module": "hisat2.nf", "processes": ["HISAT2_BUILD", "HISAT2_ALIGN"]},
    "minimap2": {"category": "alignment", "module": "minimap2.nf", "processes": ["MINIMAP2_INDEX", "MINIMAP2_ALIGN"]},
    
    # Quantification
    "featurecounts": {"category": "quantification", "module": "featurecounts.nf", "processes": ["FEATURECOUNTS"]},
    "salmon": {"category": "quantification", "module": "salmon.nf", "processes": ["SALMON_INDEX", "SALMON_QUANT"]},
    "htseq": {"category": "quantification", "module": "htseq.nf", "processes": ["HTSEQ_COUNT"]},
    "kallisto": {"category": "quantification", "module": "kallisto.nf", "processes": ["KALLISTO_INDEX", "KALLISTO_QUANT"]},
    "rsem": {"category": "quantification", "module": "rsem.nf", "processes": ["RSEM_PREPARE_REFERENCE", "RSEM_CALCULATE_EXPRESSION"]},
    
    # QC
    "fastqc": {"category": "qc", "module": "fastqc.nf", "processes": ["FASTQC"]},
    "multiqc": {"category": "qc", "module": "multiqc.nf", "processes": ["MULTIQC"]},
    "qualimap": {"category": "qc", "module": "qualimap.nf", "processes": ["QUALIMAP_BAMQC"]},
    
    # Trimming
    "trimmomatic": {"category": "trimming", "module": "trimmomatic.nf", "processes": ["TRIMMOMATIC_PE", "TRIMMOMATIC_SE"]},
    "cutadapt": {"category": "trimming", "module": "cutadapt.nf", "processes": ["CUTADAPT_PE", "CUTADAPT_SE"]},
    "fastp": {"category": "trimming", "module": "fastp.nf", "processes": ["FASTP_PE", "FASTP_SE"]},
    "trim_galore": {"category": "trimming", "module": "trimgalore.nf", "processes": ["TRIMGALORE_PE", "TRIMGALORE_SE"]},
    
    # Peak calling
    "macs2": {"category": "peaks", "module": "macs2.nf", "processes": ["MACS2_CALLPEAK"]},
    "homer": {"category": "peaks", "module": "homer.nf", "processes": ["HOMER_FINDPEAKS", "HOMER_ANNOTATEPEAKS"]},
    
    # Variant calling
    "gatk": {"category": "variant_calling", "module": "gatk.nf", "processes": ["GATK_MARKDUPLICATES", "GATK_BASERECALIBRATOR", "GATK_HAPLOTYPECALLER"]},
    "freebayes": {"category": "variant_calling", "module": "freebayes.nf", "processes": ["FREEBAYES"]},
    "bcftools": {"category": "variant_calling", "module": "bcftools.nf", "processes": ["BCFTOOLS_CALL", "BCFTOOLS_FILTER"]},
    "varscan": {"category": "variant_calling", "module": "varscan.nf", "processes": ["VARSCAN_GERMLINE", "VARSCAN_SOMATIC"]},
    
    # Structural variants
    "manta": {"category": "structural_variants", "module": "manta.nf", "processes": ["MANTA_GERMLINE", "MANTA_SOMATIC"]},
    "delly": {"category": "structural_variants", "module": "delly.nf", "processes": ["DELLY_CALL"]},
    
    # Utilities
    "samtools": {"category": "utilities", "module": "samtools.nf", "processes": ["SAMTOOLS_SORT", "SAMTOOLS_INDEX", "SAMTOOLS_FLAGSTAT"]},
    "bedtools": {"category": "utilities", "module": "bedtools.nf", "processes": ["BEDTOOLS_INTERSECT", "BEDTOOLS_MERGE"]},
    "picard": {"category": "utilities", "module": "picard.nf", "processes": ["PICARD_MARKDUPLICATES", "PICARD_COLLECTMETRICS"]},
    
    # Analysis
    "deseq2": {"category": "analysis", "module": "deseq2.nf", "processes": ["DESEQ2_DIFFERENTIAL"]},
    "edger": {"category": "analysis", "module": "edger.nf", "processes": ["EDGER_DIFFERENTIAL"]},
    
    # Assembly
    "trinity": {"category": "assembly", "module": "trinity.nf", "processes": ["TRINITY_DENOVO"]},
    "spades": {"category": "assembly", "module": "spades.nf", "processes": ["SPADES_ASSEMBLE"]},
    "megahit": {"category": "metagenomics", "module": "megahit.nf", "processes": ["MEGAHIT_ASSEMBLE"]},
    "flye": {"category": "assembly", "module": "flye.nf", "processes": ["FLYE_ASSEMBLE"]},
    "canu": {"category": "assembly", "module": "canu.nf", "processes": ["CANU_ASSEMBLE"]},
    
    # Annotation
    "prokka": {"category": "annotation", "module": "prokka.nf", "processes": ["PROKKA_ANNOTATE"]},
    
    # Methylation
    "bismark": {"category": "methylation", "module": "bismark.nf", "processes": ["BISMARK_GENOME_PREPARATION", "BISMARK_ALIGN", "BISMARK_METHYLATION_EXTRACTOR"]},
    
    # Metagenomics
    "kraken2": {"category": "metagenomics", "module": "kraken2.nf", "processes": ["KRAKEN2_CLASSIFY"]},
    "metaphlan": {"category": "metagenomics", "module": "metaphlan.nf", "processes": ["METAPHLAN_PROFILE"]},
    
    # Hi-C
    "juicer": {"category": "hic", "module": "juicer.nf", "processes": ["JUICER_TOOLS_HIC"]},
    "hicpro": {"category": "hic", "module": "hicpro.nf", "processes": ["HICPRO_PROCESS"]},
    
    # scRNA-seq
    "seurat": {"category": "scrna", "module": "seurat.nf", "processes": ["SEURAT_QC_NORMALIZE", "SEURAT_CLUSTER"]},
    "scanpy": {"category": "scrna", "module": "scanpy.nf", "processes": ["SCANPY_ANALYSIS"]},
    "cellranger": {"category": "scrna", "module": "cellranger.nf", "processes": ["CELLRANGER_COUNT"]},
    
    # Visualization
    "deeptools": {"category": "visualization", "module": "deeptools.nf", "processes": ["DEEPTOOLS_BAMCOVERAGE", "DEEPTOOLS_PLOTHEATMAP"]},
    
    # Polishing
    "racon": {"category": "polishing", "module": "racon.nf", "processes": ["RACON_POLISH"]},
}

# Container mapping for tools
TOOL_CONTAINER_MAP = {
    "star": "rnaseq", "hisat2": "rnaseq", "salmon": "rnaseq",
    "featurecounts": "rnaseq", "deseq2": "rnaseq", "edger": "rnaseq",
    "stringtie": "rnaseq", "kallisto": "rnaseq", "rsem": "rnaseq",
    "bwa": "dnaseq", "gatk": "dnaseq", "freebayes": "dnaseq",
    "bcftools": "dnaseq", "samtools": "dnaseq", "picard": "dnaseq",
    "bowtie2": "chipseq", "macs2": "chipseq", "homer": "chipseq",
    "deeptools": "chipseq",
    "seurat": "scrnaseq", "scanpy": "scrnaseq", "cellranger": "scrnaseq",
    "kraken2": "metagenomics", "metaphlan": "metagenomics", "megahit": "metagenomics",
    "prokka": "metagenomics",
    "bismark": "methylation",
    "minimap2": "longread", "flye": "longread", "canu": "longread",
    "racon": "longread", "medaka": "longread",
    "hicpro": "hic", "juicer": "hic",
    "manta": "structuralvariants", "delly": "structuralvariants",
    "fastqc": "base", "multiqc": "base", "bedtools": "base",
}


class ModuleMapper:
    """
    Maps tools to Nextflow modules and handles module discovery/creation.
    """
    
    def __init__(self, module_dir: str):
        """
        Initialize module mapper.
        
        Args:
            module_dir: Path to nextflow-modules directory
        """
        self.module_dir = Path(module_dir)
        self.modules: Dict[str, Module] = {}
        
        self._scan_modules()
    
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
                        
                        container = TOOL_CONTAINER_MAP.get(tool_name, "base")
                        
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
                container = TOOL_CONTAINER_MAP.get(tool_name, "base")
                
                self.modules[tool_name] = Module(
                    name=module_name,
                    path=module_file,
                    tool_name=tool_name,
                    container=container,
                    processes=processes
                )
        
        logger.info(f"Found {len(self.modules)} modules in {self.module_dir}")
    
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
        "bwa": "bwamem",
        "bwa-mem": "bwamem",
        "bwa_mem": "bwamem",
        "gatk": "gatk_haplotypecaller",
        "haplotypecaller": "gatk_haplotypecaller",
        "trimgalore": "trim_galore",
        "trim-galore": "trim_galore",
        "mark_duplicates": "markduplicates",
        "picard": "markduplicates",
        "cellranger": "starsolo",  # fallback to starsolo for scRNA
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
        
        # Check TOOL_MODULE_MAP for known mappings
        if tool_lower in TOOL_MODULE_MAP:
            info = TOOL_MODULE_MAP[tool_lower]
            expected_path = self.module_dir / info["category"] / info["module"]
            if expected_path.exists():
                return Module(
                    name=info["module"].replace(".nf", ""),
                    path=expected_path,
                    tool_name=tool_lower,
                    container=TOOL_CONTAINER_MAP.get(tool_lower, "base"),
                    processes=info["processes"]
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
        
        if tool_lower in TOOL_MODULE_MAP:
            return TOOL_MODULE_MAP[tool_lower]["category"]
        
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
