"""
Workflow Tools
==============

Tools for generating and managing bioinformatics workflows.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import ToolResult

logger = logging.getLogger(__name__)


# =============================================================================
# GENERATE_WORKFLOW
# =============================================================================

GENERATE_WORKFLOW_PATTERNS = [
    r"(?:create|generate|build|make)\s+(?:a\s+)?(?:new\s+)?(.+?)\s+(?:workflow|pipeline)",
    r"(?:create|generate|build)\s+(?:a\s+)?(?:new\s+)?(?:workflow|pipeline)\s+(?:for|to)\s+(.+)",
    r"(?:i want to|help me)\s+(?:do|run|perform)\s+(.+?)\s+(?:analysis|workflow|pipeline)",
    r"(?:set up|setup)\s+(?:a\s+)?(.+?)\s+(?:analysis|workflow|pipeline)",
]


def generate_workflow_impl(description: str = None) -> ToolResult:
    """
    Generate a workflow from a natural language description.
    
    Args:
        description: What the user wants to analyze
        
    Returns:
        ToolResult with workflow generation status
    """
    if not description:
        return ToolResult(
            success=False,
            tool_name="generate_workflow",
            error="No description provided",
            message="❌ Please describe what you want to analyze. Example:\n- 'Create RNA-seq differential expression workflow'\n- 'Build methylation analysis pipeline'"
        )
    
    # Detect workflow type from description
    desc_lower = description.lower()
    
    workflow_type = None
    workflow_map = {
        "rna-seq": ["rna", "expression", "transcriptom", "mrna", "differential expression", "deseq", "edger"],
        "chip-seq": ["chip", "histone", "h3k", "peak calling", "transcription factor"],
        "atac-seq": ["atac", "chromatin access", "open chromatin"],
        "methylation": ["methyl", "wgbs", "bisulfite", "dmr", "cpg"],
        "wgs": ["wgs", "whole genome seq", "variant", "mutation", "snv", "indel"],
        "scrna-seq": ["scrna", "single cell", "10x", "seurat", "scanpy"],
        "hic": ["hic", "hi-c", "chromatin interact", "3d genome", "tad"],
    }
    
    for wf_type, keywords in workflow_map.items():
        if any(kw in desc_lower for kw in keywords):
            workflow_type = wf_type
            break
    
    if not workflow_type:
        message = f"""🤔 I couldn't determine the workflow type from "{description}".

**Available workflow types:**
- 🧬 **RNA-seq**: Gene expression, differential analysis
- 🎯 **ChIP-seq**: Protein-DNA binding, histone marks
- 🔓 **ATAC-seq**: Chromatin accessibility
- 🔬 **Methylation**: DNA methylation (WGBS/RRBS)
- 🧪 **WGS/WES**: Variant calling, mutations
- 🔴 **scRNA-seq**: Single-cell transcriptomics
- 🔗 **Hi-C**: Chromatin conformation

**Try being more specific:**
- "Create RNA-seq workflow for differential expression"
- "Build ChIP-seq peak calling pipeline"
- "Generate methylation DMR analysis"
"""
        return ToolResult(
            success=True,
            tool_name="generate_workflow",
            data={"detected_type": None},
            message=message
        )
    
    # Try to use the workflow composer
    try:
        from workflow_composer.core.workflow_generator import WorkflowGenerator
        from workflow_composer.core.query_parser import ParsedIntent, AnalysisType
        
        # Map workflow type string to AnalysisType enum
        type_map = {
            "rna-seq": AnalysisType.RNA_SEQ_DE,
            "chip-seq": AnalysisType.CHIP_SEQ,
            "atac-seq": AnalysisType.ATAC_SEQ,
            "methylation": AnalysisType.BISULFITE_SEQ,
            "wgs": AnalysisType.WGS_VARIANT_CALLING,
            "scrna-seq": AnalysisType.SCRNA_SEQ,
            "hic": AnalysisType.HIC,
        }
        
        analysis_type = type_map.get(workflow_type, AnalysisType.CUSTOM)
        
        # Create ParsedIntent object
        intent = ParsedIntent(
            analysis_type=analysis_type,
            analysis_type_raw=workflow_type,
            confidence=0.9,
            organism="human",
            paired_end=True,
        )
        
        generator = WorkflowGenerator()
        
        # Generate workflow with correct API
        workflow = generator.generate(intent=intent, modules=[])
        
        # Save the workflow
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path.cwd() / "generated_workflows" / f"{workflow_type}_{timestamp}"
        workflow_path = workflow.save(str(output_dir))
        
        message = f"""✅ **Generated {workflow_type.upper()} Workflow**

📂 **Output:** `{workflow_path}`

**What was created:**
- `main.nf` - Nextflow pipeline
- `nextflow.config` - Configuration
- `samplesheet.csv` - Sample template
- `README.md` - Documentation

**Next steps:**
1. Review the generated workflow
2. Prepare your sample sheet
3. Run it: `nextflow run {workflow_path}/main.nf -profile slurm`

Would you like me to explain what the workflow does?
"""
        return ToolResult(
            success=True,
            tool_name="generate_workflow",
            data={"workflow_type": workflow_type, "path": str(workflow_path)},
            message=message
        )
        
    except ImportError as e:
        # Generator not available - provide template info
        component_map = {
            "rna-seq": """
- **QC**: FastQC → MultiQC
- **Trimming**: Trimmomatic/fastp
- **Alignment**: STAR or HISAT2
- **Quantification**: featureCounts or Salmon
- **Analysis**: DESeq2 differential expression""",
            "chip-seq": """
- **QC**: FastQC → MultiQC
- **Alignment**: BWA-MEM or Bowtie2
- **Peak Calling**: MACS2/MACS3
- **Annotation**: ChIPseeker
- **Visualization**: deepTools""",
            "methylation": """
- **QC**: FastQC, Bismark QC
- **Alignment**: Bismark (bisulfite-aware)
- **Calling**: methylKit or MethylDackel
- **DMR Analysis**: DSS or DMRcaller""",
        }
        
        components = component_map.get(workflow_type, "\n- Standard NGS pipeline components")
        
        message = f"""📋 **Generating {workflow_type.upper()} Workflow**

Based on: *"{description}"*

**Workflow Components:**
{components}

The WorkflowGenerator module needs additional setup.
Would you like me to help configure it?
"""
        return ToolResult(
            success=True,
            tool_name="generate_workflow",
            data={"workflow_type": workflow_type, "description": description},
            message=message
        )


# =============================================================================
# LIST_WORKFLOWS
# =============================================================================

LIST_WORKFLOWS_PATTERNS = [
    r"(?:list|show|what)\s+(?:available\s+)?workflows?",
    r"(?:show|list)\s+(?:me\s+)?(?:my\s+)?(?:generated\s+)?workflows?",
]


def list_workflows_impl() -> ToolResult:
    """
    List available and generated workflows.
    
    Returns:
        ToolResult with workflow list
    """
    generated_dir = Path.cwd() / "generated_workflows"
    workflows = []
    
    if generated_dir.exists():
        for d in generated_dir.iterdir():
            if d.is_dir() and (d / "main.nf").exists():
                workflows.append({
                    "name": d.name,
                    "path": str(d),
                    "created": datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                })
    
    if workflows:
        wf_list = "\n".join([
            f"  - `{w['name']}` (created {w['created']})"
            for w in sorted(workflows, key=lambda x: x['created'], reverse=True)[:10]
        ])
        
        message = f"""📋 **Generated Workflows:**

{wf_list}

**Available Templates:**
- RNA-seq (differential expression)
- ChIP-seq (peak calling)
- ATAC-seq (accessibility)
- Methylation (WGBS/RRBS)
- WGS/WES (variant calling)
- scRNA-seq (single-cell)

Say "create <type> workflow" to generate a new one.
"""
    else:
        message = """📋 **No workflows generated yet.**

**Available Templates:**
- RNA-seq (differential expression)
- ChIP-seq (peak calling)
- ATAC-seq (accessibility)
- Methylation (WGBS/RRBS)
- WGS/WES (variant calling)
- scRNA-seq (single-cell)

Say "create RNA-seq workflow" to generate one.
"""
    
    return ToolResult(
        success=True,
        tool_name="list_workflows",
        data={"workflows": workflows},
        message=message
    )


# =============================================================================
# CHECK_REFERENCES
# =============================================================================

CHECK_REFERENCES_PATTERNS = [
    r"(?:check|verify|do i have)\s+(?:the\s+)?(?:reference|genome|index)\s+(?:for)?\s*(.+)?",
    r"(?:list|show)\s+(?:available\s+)?(?:references?|genomes?|indexes?)",
]


def check_references_impl(organism: str = None) -> ToolResult:
    """
    Check available reference genomes and indices.
    
    Args:
        organism: Organism to check (human, mouse, etc.)
        
    Returns:
        ToolResult with reference status
    """
    # Common reference locations
    ref_paths = [
        Path("/scratch/sdodl001/BioPipelines/data/references"),
        Path.home() / "BioPipelines" / "data" / "references",
        Path.cwd() / "data" / "references",
    ]
    
    found_refs = []
    
    for ref_dir in ref_paths:
        if ref_dir.exists():
            for f in ref_dir.rglob("*.fa*"):
                found_refs.append({
                    "name": f.stem,
                    "path": str(f),
                    "type": "FASTA"
                })
            for f in ref_dir.rglob("*.gtf*"):
                found_refs.append({
                    "name": f.stem,
                    "path": str(f),
                    "type": "GTF"
                })
    
    if found_refs:
        ref_list = "\n".join([
            f"  - `{r['name']}` ({r['type']})"
            for r in found_refs[:15]
        ])
        
        message = f"""🧬 **Available References:**

{ref_list}

**Need different references?**
```bash
# Download human genome (GRCh38)
aws s3 cp s3://ngi-igenomes/igenomes/Homo_sapiens/NCBI/GRCh38/Sequence/WholeGenomeFasta/genome.fa .

# Download mouse genome (GRCm38)
aws s3 cp s3://ngi-igenomes/igenomes/Mus_musculus/NCBI/GRCm38/Sequence/WholeGenomeFasta/genome.fa .
```
"""
    else:
        message = """⚠️ **No reference genomes found.**

**To download references:**
```bash
# Human GRCh38
wget ftp://ftp.ensembl.org/pub/release-109/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz

# Mouse GRCm39
wget ftp://ftp.ensembl.org/pub/release-109/fasta/mus_musculus/dna/Mus_musculus.GRCm39.dna.primary_assembly.fa.gz
```

Or use iGenomes from AWS S3.
"""
    
    return ToolResult(
        success=True,
        tool_name="check_references",
        data={"references": found_refs},
        message=message
    )
