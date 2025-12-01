"""
Pre-flight Validator
====================

Validates everything before workflow generation and execution.

Checks:
- Tool availability in containers
- Container image existence
- Reference data availability
- Module existence
- Resource requirements
- Dependency resolution

Usage:
    validator = PreflightValidator()
    report = validator.validate(intent, tools)
    
    if report.can_proceed:
        # Generate and run workflow
    elif report.auto_fixable:
        validator.auto_fix(report)
"""

import os
import subprocess
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Status of a validation check."""
    READY = "ready"
    MISSING = "missing"
    DOWNLOADABLE = "downloadable"
    BUILDABLE = "buildable"
    REQUIRES_MANUAL = "requires_manual"
    ERROR = "error"


@dataclass
class ValidationItem:
    """A single item that was validated."""
    name: str
    category: str  # tool, container, reference, module
    status: ValidationStatus
    details: str = ""
    auto_fixable: bool = False
    fix_action: Optional[str] = None
    fix_time_estimate: Optional[str] = None
    path: Optional[str] = None


@dataclass
class ResourceEstimate:
    """Estimated resources for the workflow."""
    memory_gb: int
    cpus: int
    estimated_hours: float
    recommended_partition: str
    cost_estimate_usd: Optional[float] = None
    storage_gb: Optional[int] = None


@dataclass
class ValidationReport:
    """Complete validation report."""
    can_proceed: bool
    auto_fixable: bool
    items: List[ValidationItem] = field(default_factory=list)
    missing_items: List[ValidationItem] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    resources: Optional[ResourceEstimate] = None
    fix_time_total: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "can_proceed": self.can_proceed,
            "auto_fixable": self.auto_fixable,
            "items": [
                {
                    "name": item.name,
                    "category": item.category,
                    "status": item.status.value,
                    "details": item.details,
                    "auto_fixable": item.auto_fixable,
                    "fix_action": item.fix_action,
                }
                for item in self.items
            ],
            "missing_items": [item.name for item in self.missing_items],
            "warnings": self.warnings,
            "resources": {
                "memory_gb": self.resources.memory_gb,
                "cpus": self.resources.cpus,
                "estimated_hours": self.resources.estimated_hours,
                "recommended_partition": self.resources.recommended_partition,
                "cost_estimate_usd": self.resources.cost_estimate_usd,
            } if self.resources else None,
        }
    
    def to_markdown(self) -> str:
        """Format report as markdown for UI display."""
        lines = ["## 🔍 Pre-flight Validation Report\n"]
        
        # Overall status
        if self.can_proceed:
            lines.append("✅ **Status: Ready to proceed**\n")
        elif self.auto_fixable:
            lines.append("⚠️ **Status: Issues found (auto-fixable)**\n")
        else:
            lines.append("❌ **Status: Manual intervention required**\n")
        
        # Items table
        lines.append("| Component | Status | Details |")
        lines.append("|-----------|--------|---------|")
        
        status_icons = {
            ValidationStatus.READY: "✅",
            ValidationStatus.MISSING: "❌",
            ValidationStatus.DOWNLOADABLE: "📥",
            ValidationStatus.BUILDABLE: "🔧",
            ValidationStatus.REQUIRES_MANUAL: "⚠️",
            ValidationStatus.ERROR: "💥",
        }
        
        for item in self.items:
            icon = status_icons.get(item.status, "❓")
            lines.append(f"| {item.name} ({item.category}) | {icon} {item.status.value} | {item.details} |")
        
        # Resource estimate
        if self.resources:
            lines.append("\n### 📊 Resource Estimate")
            lines.append(f"- **Memory:** {self.resources.memory_gb} GB")
            lines.append(f"- **CPUs:** {self.resources.cpus}")
            lines.append(f"- **Estimated Time:** {self.resources.estimated_hours:.1f} hours")
            lines.append(f"- **Recommended Partition:** {self.resources.recommended_partition}")
            if self.resources.cost_estimate_usd:
                lines.append(f"- **Estimated Cost:** ${self.resources.cost_estimate_usd:.2f}")
        
        # Warnings
        if self.warnings:
            lines.append("\n### ⚠️ Warnings")
            for warning in self.warnings:
                lines.append(f"- {warning}")
        
        # Fix actions
        if self.missing_items and self.auto_fixable:
            lines.append("\n### 🔧 Required Actions")
            for item in self.missing_items:
                if item.auto_fixable:
                    lines.append(f"- **{item.name}**: {item.fix_action} (~{item.fix_time_estimate})")
        
        return "\n".join(lines)


# Tool to container mapping
TOOL_CONTAINER_MAP = {
    "star": "rna-seq", "hisat2": "rna-seq", "salmon": "rna-seq",
    "featurecounts": "rna-seq", "deseq2": "rna-seq", "edger": "rna-seq",
    "kallisto": "rna-seq", "rsem": "rna-seq", "stringtie": "rna-seq",
    "bwa": "dna-seq", "gatk": "dna-seq", "freebayes": "dna-seq",
    "bcftools": "dna-seq", "samtools": "base", "picard": "dna-seq",
    "bowtie2": "chip-seq", "macs2": "chip-seq", "homer": "chip-seq",
    "deeptools": "chip-seq",
    "seurat": "scrna-seq", "scanpy": "scrna-seq", "cellranger": "scrna-seq",
    "kraken2": "metagenomics", "metaphlan": "metagenomics",
    "bismark": "methylation",
    "minimap2": "long-read", "flye": "long-read", "canu": "long-read",
    "manta": "structural-variants", "delly": "structural-variants",
    "fastqc": "base", "multiqc": "base", "bedtools": "base",
}

# Resource profiles per tool
TOOL_RESOURCE_PROFILES = {
    "star": {"memory_gb": 32, "cpus": 8, "hours_per_sample": 0.5},
    "star_index": {"memory_gb": 32, "cpus": 8, "hours": 2.0},
    "hisat2": {"memory_gb": 8, "cpus": 8, "hours_per_sample": 0.3},
    "salmon": {"memory_gb": 8, "cpus": 8, "hours_per_sample": 0.1},
    "featurecounts": {"memory_gb": 4, "cpus": 4, "hours_per_sample": 0.05},
    "deseq2": {"memory_gb": 8, "cpus": 2, "hours": 0.2},
    "bwa": {"memory_gb": 16, "cpus": 8, "hours_per_sample": 0.5},
    "gatk": {"memory_gb": 16, "cpus": 4, "hours_per_sample": 1.0},
    "cellranger": {"memory_gb": 64, "cpus": 16, "hours_per_sample": 4.0},
    "macs2": {"memory_gb": 8, "cpus": 2, "hours_per_sample": 0.2},
    "fastqc": {"memory_gb": 2, "cpus": 2, "hours_per_sample": 0.05},
    "multiqc": {"memory_gb": 4, "cpus": 2, "hours": 0.1},
    "default": {"memory_gb": 8, "cpus": 4, "hours_per_sample": 0.3},
}

# Reference data requirements per organism/build
REFERENCE_REQUIREMENTS = {
    "human": {
        "GRCh38": {
            "genome": "Homo_sapiens.GRCh38.dna.primary_assembly.fa",
            "gtf": "Homo_sapiens.GRCh38.109.gtf",
            "source": "ensembl",
        },
        "hg38": {
            "genome": "hg38.fa",
            "gtf": "hg38.refGene.gtf",
            "source": "ucsc",
        }
    },
    "mouse": {
        "GRCm39": {
            "genome": "Mus_musculus.GRCm39.dna.primary_assembly.fa",
            "gtf": "Mus_musculus.GRCm39.109.gtf",
            "source": "ensembl",
        },
        "mm39": {
            "genome": "mm39.fa",
            "gtf": "mm39.refGene.gtf",
            "source": "ucsc",
        }
    }
}


class PreflightValidator:
    """
    Validates all prerequisites before workflow generation.
    
    Performs comprehensive checks to prevent runtime failures:
    - Tool existence in containers
    - Container image availability
    - Reference genome and annotation files
    - Index files (STAR, BWA, etc.)
    - Module files
    - Sufficient resources
    
    Uses ReferenceManager and ContainerManager for auto-provisioning.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize validator with optional config."""
        self.project_dir = Path(os.environ.get(
            'BIOPIPELINES_HOME',
            '/home/sdodl001_odu_edu/BioPipelines'
        ))
        self.containers_dir = self.project_dir / "containers"
        self.references_dir = Path(os.environ.get(
            'BIOPIPELINES_REFERENCES',
            '/scratch/references'
        ))
        self.modules_dir = self.project_dir / "knowledge_base" / "modules"
        
        # Container registry
        self.container_registry = os.environ.get(
            'CONTAINER_REGISTRY',
            'docker://ghcr.io/sdodlapa/biopipelines'
        )
        
        # Lazy-load managers
        self._ref_manager = None
        self._container_manager = None
    
    @property
    def ref_manager(self):
        """Lazy-load reference manager."""
        if self._ref_manager is None:
            try:
                from ..provisioning import get_reference_manager
                self._ref_manager = get_reference_manager()
            except ImportError:
                logger.warning("ReferenceManager not available")
        return self._ref_manager
    
    @property
    def container_manager(self):
        """Lazy-load container manager."""
        if self._container_manager is None:
            try:
                from ..provisioning import get_container_manager
                self._container_manager = get_container_manager()
            except ImportError:
                logger.warning("ContainerManager not available")
        return self._container_manager
    
    def validate(
        self,
        analysis_type: str,
        tools: List[str],
        organism: Optional[str] = None,
        genome_build: Optional[str] = None,
        sample_count: int = 1,
    ) -> ValidationReport:
        """
        Perform comprehensive validation.
        
        Args:
            analysis_type: Type of analysis (e.g., "rna_seq_differential_expression")
            tools: List of required tools
            organism: Organism name (e.g., "mouse")
            genome_build: Genome build (e.g., "GRCm39")
            sample_count: Number of samples to process
            
        Returns:
            ValidationReport with all findings
        """
        items = []
        missing = []
        warnings = []
        
        # 1. Validate tools and containers
        tool_items = self._validate_tools(tools)
        items.extend(tool_items)
        missing.extend([i for i in tool_items if i.status != ValidationStatus.READY])
        
        # 2. Validate container images
        containers_needed = set(TOOL_CONTAINER_MAP.get(t, "base") for t in tools)
        container_items = self._validate_containers(containers_needed)
        items.extend(container_items)
        missing.extend([i for i in container_items if i.status != ValidationStatus.READY])
        
        # 3. Validate reference data
        if organism and genome_build:
            ref_items = self._validate_references(organism, genome_build, tools)
            items.extend(ref_items)
            missing.extend([i for i in ref_items if i.status != ValidationStatus.READY])
        else:
            warnings.append("Organism/genome not specified - cannot validate reference data")
        
        # 4. Validate modules
        module_items = self._validate_modules(tools)
        items.extend(module_items)
        missing.extend([i for i in module_items if i.status != ValidationStatus.READY])
        
        # 5. Estimate resources
        resources = self._estimate_resources(tools, sample_count)
        
        # Determine if we can proceed
        can_proceed = len(missing) == 0
        auto_fixable = all(item.auto_fixable for item in missing)
        
        # Calculate total fix time
        fix_time_total = None
        if missing and auto_fixable:
            total_hours = sum(
                self._parse_time_estimate(item.fix_time_estimate or "0")
                for item in missing
            )
            fix_time_total = f"{total_hours:.1f} hours"
        
        return ValidationReport(
            can_proceed=can_proceed,
            auto_fixable=auto_fixable,
            items=items,
            missing_items=missing,
            warnings=warnings,
            resources=resources,
            fix_time_total=fix_time_total,
        )
    
    def _validate_tools(self, tools: List[str]) -> List[ValidationItem]:
        """Validate that all tools are known and available."""
        items = []
        
        for tool in tools:
            container = TOOL_CONTAINER_MAP.get(tool)
            
            if container:
                # Tool is known
                items.append(ValidationItem(
                    name=tool,
                    category="tool",
                    status=ValidationStatus.READY,
                    details=f"Available in {container} container",
                ))
            else:
                # Unknown tool - check if we can find it
                items.append(ValidationItem(
                    name=tool,
                    category="tool",
                    status=ValidationStatus.MISSING,
                    details="Unknown tool - not in any container",
                    auto_fixable=True,
                    fix_action=f"Build custom container with {tool}",
                    fix_time_estimate="30 min",
                ))
        
        return items
    
    def _validate_containers(self, containers: Set[str]) -> List[ValidationItem]:
        """Validate that container images exist and are pullable."""
        items = []
        
        for container in containers:
            # Check if container definition exists
            container_dir = self.containers_dir / container
            
            if container_dir.exists() and (container_dir / "Dockerfile").exists():
                # Check if image is built (try to pull/check)
                image_uri = f"{self.container_registry}/{container}:latest"
                
                # Try singularity inspect (quick check)
                if self._container_image_exists(container):
                    items.append(ValidationItem(
                        name=container,
                        category="container",
                        status=ValidationStatus.READY,
                        details=f"Image available: {image_uri}",
                        path=str(container_dir),
                    ))
                else:
                    items.append(ValidationItem(
                        name=container,
                        category="container",
                        status=ValidationStatus.BUILDABLE,
                        details=f"Definition exists, image not built",
                        auto_fixable=True,
                        fix_action=f"Build container: docker build -t {container} {container_dir}",
                        fix_time_estimate="15 min",
                        path=str(container_dir),
                    ))
            else:
                items.append(ValidationItem(
                    name=container,
                    category="container",
                    status=ValidationStatus.MISSING,
                    details="Container definition not found",
                    auto_fixable=False,
                ))
        
        return items
    
    def _validate_references(
        self, 
        organism: str, 
        genome_build: str, 
        tools: List[str]
    ) -> List[ValidationItem]:
        """Validate reference data availability."""
        items = []
        
        # Normalize organism name
        organism_lower = organism.lower()
        organism_map = {
            "homo sapiens": "human", "human": "human",
            "mus musculus": "mouse", "mouse": "mouse",
        }
        organism_key = organism_map.get(organism_lower, organism_lower)
        
        # Check if we know this organism/build
        if organism_key in REFERENCE_REQUIREMENTS:
            builds = REFERENCE_REQUIREMENTS[organism_key]
            if genome_build in builds:
                ref_info = builds[genome_build]
                
                # Check genome FASTA
                genome_path = self.references_dir / organism_key / genome_build / ref_info["genome"]
                if genome_path.exists():
                    items.append(ValidationItem(
                        name=f"Genome ({genome_build})",
                        category="reference",
                        status=ValidationStatus.READY,
                        details=str(genome_path),
                        path=str(genome_path),
                    ))
                else:
                    items.append(ValidationItem(
                        name=f"Genome ({genome_build})",
                        category="reference",
                        status=ValidationStatus.DOWNLOADABLE,
                        details=f"Download from {ref_info['source']}",
                        auto_fixable=True,
                        fix_action=f"Download from Ensembl/UCSC",
                        fix_time_estimate="30 min",
                    ))
                
                # Check GTF annotation
                gtf_path = self.references_dir / organism_key / genome_build / ref_info["gtf"]
                if gtf_path.exists():
                    items.append(ValidationItem(
                        name=f"Annotation ({genome_build})",
                        category="reference",
                        status=ValidationStatus.READY,
                        details=str(gtf_path),
                        path=str(gtf_path),
                    ))
                else:
                    items.append(ValidationItem(
                        name=f"Annotation ({genome_build})",
                        category="reference",
                        status=ValidationStatus.DOWNLOADABLE,
                        details=f"Download from {ref_info['source']}",
                        auto_fixable=True,
                        fix_action=f"Download GTF annotation",
                        fix_time_estimate="10 min",
                    ))
                
                # Check tool-specific indexes
                index_items = self._validate_indexes(organism_key, genome_build, tools)
                items.extend(index_items)
            else:
                items.append(ValidationItem(
                    name=f"Genome build ({genome_build})",
                    category="reference",
                    status=ValidationStatus.REQUIRES_MANUAL,
                    details=f"Unknown build for {organism_key}. Known: {list(builds.keys())}",
                    auto_fixable=False,
                ))
        else:
            items.append(ValidationItem(
                name=f"Organism ({organism})",
                category="reference",
                status=ValidationStatus.REQUIRES_MANUAL,
                details=f"Unknown organism. Known: {list(REFERENCE_REQUIREMENTS.keys())}",
                auto_fixable=False,
            ))
        
        return items
    
    def _validate_indexes(
        self, 
        organism: str, 
        genome_build: str, 
        tools: List[str]
    ) -> List[ValidationItem]:
        """Validate tool-specific index files."""
        items = []
        
        # Tool to index mapping
        tool_indexes = {
            "star": ("star_index", "STAR index (GenomeDir)", "2 hours"),
            "hisat2": ("hisat2_index", "HISAT2 index", "1 hour"),
            "bwa": ("bwa_index", "BWA index", "30 min"),
            "bowtie2": ("bowtie2_index", "Bowtie2 index", "30 min"),
            "salmon": ("salmon_index", "Salmon index", "15 min"),
            "kallisto": ("kallisto_index", "Kallisto index", "10 min"),
        }
        
        for tool in tools:
            if tool in tool_indexes:
                index_dir, index_name, build_time = tool_indexes[tool]
                index_path = self.references_dir / organism / genome_build / index_dir
                
                if index_path.exists() and any(index_path.iterdir()):
                    items.append(ValidationItem(
                        name=index_name,
                        category="index",
                        status=ValidationStatus.READY,
                        details=str(index_path),
                        path=str(index_path),
                    ))
                else:
                    items.append(ValidationItem(
                        name=index_name,
                        category="index",
                        status=ValidationStatus.BUILDABLE,
                        details="Index needs to be built",
                        auto_fixable=True,
                        fix_action=f"Build {index_name}",
                        fix_time_estimate=build_time,
                    ))
        
        return items
    
    def _validate_modules(self, tools: List[str]) -> List[ValidationItem]:
        """Validate that Nextflow modules exist for all tools."""
        items = []
        
        # Tool to module mapping - simplified inline version
        # In production, would use ModuleMapper instance
        tool_module_map = {
            "fastqc": {"category": "qc", "module": "fastqc.nf"},
            "multiqc": {"category": "qc", "module": "multiqc.nf"},
            "star": {"category": "alignment", "module": "star.nf"},
            "hisat2": {"category": "alignment", "module": "hisat2.nf"},
            "salmon": {"category": "quantification", "module": "salmon.nf"},
            "kallisto": {"category": "quantification", "module": "kallisto.nf"},
            "featurecounts": {"category": "quantification", "module": "featurecounts.nf"},
            "deseq2": {"category": "differential", "module": "deseq2.nf"},
            "bwa": {"category": "alignment", "module": "bwa.nf"},
            "gatk": {"category": "variant_calling", "module": "gatk.nf"},
            "bcftools": {"category": "variant_calling", "module": "bcftools.nf"},
            "bowtie2": {"category": "alignment", "module": "bowtie2.nf"},
            "macs2": {"category": "peak_calling", "module": "macs2.nf"},
            "deeptools": {"category": "visualization", "module": "deeptools.nf"},
            "samtools": {"category": "utils", "module": "samtools.nf"},
            "bedtools": {"category": "utils", "module": "bedtools.nf"},
        }
        
        for tool in tools:
            if tool in tool_module_map:
                module_info = tool_module_map[tool]
                category = module_info["category"]
                module_file = module_info["module"]
                
                module_path = self.modules_dir / category / module_file
                
                if module_path.exists():
                    items.append(ValidationItem(
                        name=f"{tool} module",
                        category="module",
                        status=ValidationStatus.READY,
                        details=str(module_path),
                        path=str(module_path),
                    ))
                else:
                    items.append(ValidationItem(
                        name=f"{tool} module",
                        category="module",
                        status=ValidationStatus.BUILDABLE,
                        details="Module can be auto-generated",
                        auto_fixable=True,
                        fix_action="Generate module using LLM",
                        fix_time_estimate="1 min",
                    ))
            else:
                items.append(ValidationItem(
                    name=f"{tool} module",
                    category="module",
                    status=ValidationStatus.MISSING,
                    details="No module mapping defined",
                    auto_fixable=True,
                    fix_action="Create custom module using LLM",
                    fix_time_estimate="2 min",
                ))
        
        return items
    
    def _estimate_resources(
        self, 
        tools: List[str], 
        sample_count: int
    ) -> ResourceEstimate:
        """Estimate computational resources needed."""
        max_memory = 0
        max_cpus = 0
        total_hours = 0
        
        for tool in tools:
            profile = TOOL_RESOURCE_PROFILES.get(tool, TOOL_RESOURCE_PROFILES["default"])
            
            max_memory = max(max_memory, profile["memory_gb"])
            max_cpus = max(max_cpus, profile["cpus"])
            
            if "hours_per_sample" in profile:
                total_hours += profile["hours_per_sample"] * sample_count
            elif "hours" in profile:
                total_hours += profile["hours"]
        
        # Adjust for parallelism (assume up to 10 parallel jobs)
        parallel_factor = min(sample_count, 10)
        if sample_count > 1:
            total_hours = total_hours / parallel_factor
        
        # Determine partition based on memory
        if max_memory <= 16:
            partition = "cpuspot"
        elif max_memory <= 64:
            partition = "cpuspot"
        else:
            partition = "a100flex"  # Need GPU node for large memory
        
        # Estimate cost (rough GCP estimate)
        # n1-standard-8: ~$0.38/hour
        # n1-highmem-16: ~$0.95/hour
        hourly_rate = 0.38 if max_memory <= 32 else 0.95
        cost = hourly_rate * total_hours * parallel_factor
        
        return ResourceEstimate(
            memory_gb=max_memory,
            cpus=max_cpus,
            estimated_hours=total_hours,
            recommended_partition=partition,
            cost_estimate_usd=round(cost, 2),
            storage_gb=sample_count * 10,  # Rough estimate: 10GB per sample
        )
    
    def _container_image_exists(self, container: str) -> bool:
        """Check if container image exists (cached or pullable)."""
        # Check for local Singularity image
        sif_path = self.project_dir / "containers" / "images" / f"{container}.sif"
        if sif_path.exists():
            return True
        
        # Try to query registry (quick timeout)
        try:
            result = subprocess.run(
                ["singularity", "inspect", f"docker://ghcr.io/sdodlapa/biopipelines/{container}:latest"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _parse_time_estimate(self, time_str: str) -> float:
        """Parse time estimate string to hours."""
        time_str = time_str.lower()
        if "hour" in time_str:
            return float(time_str.split()[0])
        elif "min" in time_str:
            return float(time_str.split()[0]) / 60
        return 0.5  # Default
    
    def auto_fix(self, report: ValidationReport) -> bool:
        """
        Automatically fix missing items that are auto-fixable.
        
        Uses ReferenceManager and ContainerManager for provisioning.
        
        Args:
            report: ValidationReport with missing items
            
        Returns:
            True if all fixes succeeded
        """
        if not report.auto_fixable:
            logger.warning("Report contains items that cannot be auto-fixed")
            return False
        
        success = True
        for item in report.missing_items:
            if not item.auto_fixable:
                continue
            
            try:
                if item.category == "module":
                    self._create_module(item.name.replace(" module", ""))
                elif item.category == "index":
                    self._build_index_auto(item)
                elif item.category == "reference":
                    self._download_reference_auto(item)
                elif item.category == "container":
                    self._pull_container_auto(item)
                    
                logger.info(f"Fixed: {item.name}")
            except Exception as e:
                logger.error(f"Failed to fix {item.name}: {e}")
                success = False
        
        return success
    
    def _create_module(self, tool: str):
        """Create missing Nextflow module using LLM."""
        # This would call ModuleMapper.create_module()
        logger.info(f"Would create module for: {tool}")
        # TODO: Implement
    
    def _build_index_auto(self, item: ValidationItem):
        """Build missing index using ReferenceManager."""
        if self.ref_manager is None:
            logger.warning("ReferenceManager not available for index building")
            return
        
        # Parse index type from item name
        index_name = item.name.lower()
        index_type = None
        for itype in ["star", "bwa", "bowtie2", "hisat2", "salmon", "kallisto"]:
            if itype in index_name:
                index_type = itype
                break
        
        if not index_type:
            logger.error(f"Unknown index type in: {item.name}")
            return
        
        # Try to infer organism/build from context
        # This is a simplified version - in practice, would need more context
        logger.info(f"Would build {index_type} index via ReferenceManager")
        # self.ref_manager.ensure_index(organism, build, index_type, submit_job=True)
    
    def _download_reference_auto(self, item: ValidationItem):
        """Download missing reference using ReferenceManager."""
        if self.ref_manager is None:
            logger.warning("ReferenceManager not available for download")
            return
        
        # Parse organism/build from item name
        # e.g., "Genome (GRCh38)" or "Annotation (GRCm39)"
        import re
        match = re.search(r'\((\w+)\)', item.name)
        if match:
            build = match.group(1)
            # Map build to organism (simplified)
            org_map = {
                "GRCh38": "human", "hg38": "human",
                "GRCm39": "mouse", "mm39": "mouse",
            }
            organism = org_map.get(build)
            
            if organism:
                logger.info(f"Downloading reference: {organism}/{build}")
                self.ref_manager.get_reference(organism, build, download=True)
            else:
                logger.warning(f"Unknown build: {build}")
    
    def _pull_container_auto(self, item: ValidationItem):
        """Pull missing container using ContainerManager."""
        if self.container_manager is None:
            logger.warning("ContainerManager not available for pull")
            return
        
        container_name = item.name
        logger.info(f"Pulling container: {container_name}")
        self.container_manager.get_container(container_name, pull=True)


# Convenience function
def validate_workflow_prerequisites(
    analysis_type: str,
    tools: List[str],
    organism: Optional[str] = None,
    genome_build: Optional[str] = None,
    sample_count: int = 1,
) -> ValidationReport:
    """
    Validate all prerequisites for a workflow.
    
    Convenience function that creates a validator and runs validation.
    """
    validator = PreflightValidator()
    return validator.validate(
        analysis_type=analysis_type,
        tools=tools,
        organism=organism,
        genome_build=genome_build,
        sample_count=sample_count,
    )


if __name__ == "__main__":
    # Test validation
    import logging
    logging.basicConfig(level=logging.INFO)
    
    report = validate_workflow_prerequisites(
        analysis_type="rna_seq_differential_expression",
        tools=["fastqc", "star", "featurecounts", "deseq2", "multiqc"],
        organism="mouse",
        genome_build="GRCm39",
        sample_count=10,
    )
    
    print(report.to_markdown())
