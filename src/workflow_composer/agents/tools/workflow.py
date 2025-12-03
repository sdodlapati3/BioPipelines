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


def generate_workflow_impl(
    description: str = None,
    workflow_type: str = None,
    pipeline_type: str = None,
    **kwargs,
) -> ToolResult:
    """
    Generate a workflow from a natural language description.
    
    Args:
        description: What the user wants to analyze
        workflow_type: Type of workflow (rna-seq, chip-seq, etc.) - alternate parameter name
        pipeline_type: Type of pipeline - alternate parameter name
        **kwargs: Additional parameters (ignored)
        
    Returns:
        ToolResult with workflow generation status
    """
    # Accept workflow_type or pipeline_type as aliases for description
    if not description:
        description = workflow_type or pipeline_type
    
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
    r"(?:check|verify|do i have)\s+(?:the\s+)?(?:reference|genome|index)",
    r"(?:check|verify)\s+references?",
    r"(?:list|show)\s+(?:available\s+)?(?:references?|genomes?|indexes?)",
    r"what\s+references?\s+(?:do\s+I\s+have|are\s+available)",
]


def check_references_impl(
    organism: str = None,
    assembly: str = None,
) -> ToolResult:
    """
    Check available reference genomes and indices.
    
    Uses ReferenceManager for comprehensive reference discovery,
    shows available files, missing items, and download URLs.
    
    Args:
        organism: Organism to check (human, mouse, rat, zebrafish). 
                  If None, checks all organisms.
        assembly: Specific assembly to check (GRCh38, GRCm39, etc.)
        
    Returns:
        ToolResult with reference status
    """
    try:
        # Try to use ReferenceManager for comprehensive checking
        try:
            from workflow_composer.data.reference_manager import ReferenceManager, REFERENCE_SOURCES
            use_manager = True
        except ImportError:
            use_manager = False
            logger.debug("ReferenceManager not available, using simple scan")
        
        # Common reference locations
        ref_paths = [
            Path("/scratch/sdodl001/BioPipelines/data/references"),
            Path.home() / "BioPipelines" / "data" / "references",
            Path.cwd() / "data" / "references",
        ]
        
        base_dir = None
        for p in ref_paths:
            if p.exists():
                base_dir = p
                break
        
        if use_manager and base_dir:
            # Use ReferenceManager for comprehensive check
            manager = ReferenceManager(base_dir=base_dir)
            
            # Determine which organisms to check
            if organism:
                organisms_to_check = [organism.lower()]
            else:
                organisms_to_check = list(REFERENCE_SOURCES.keys())
            
            results = {}
            all_available = []
            all_missing = []
            
            for org in organisms_to_check:
                if org not in REFERENCE_SOURCES:
                    continue
                    
                # Determine assemblies to check
                if assembly and organism:
                    assemblies = [assembly] if assembly in REFERENCE_SOURCES[org] else []
                else:
                    assemblies = list(REFERENCE_SOURCES[org].keys())
                
                for asm in assemblies:
                    try:
                        ref_info = manager.check_references(org, asm)
                        
                        key = f"{org}/{asm}"
                        results[key] = {
                            "genome": str(ref_info.genome_fasta) if ref_info.genome_fasta else None,
                            "gtf": str(ref_info.annotation_gtf) if ref_info.annotation_gtf else None,
                            "transcriptome": str(ref_info.transcriptome_fasta) if ref_info.transcriptome_fasta else None,
                            "star_index": str(ref_info.star_index) if ref_info.star_index else None,
                            "salmon_index": str(ref_info.salmon_index) if ref_info.salmon_index else None,
                            "bwa_index": str(ref_info.bwa_index) if ref_info.bwa_index else None,
                            "missing": ref_info.missing,
                            "download_urls": ref_info.download_urls,
                        }
                        
                        # Track what's available and missing
                        if ref_info.genome_fasta:
                            all_available.append(f"`{org}/{asm}` genome ✅")
                        else:
                            all_missing.append(f"`{org}/{asm}` genome")
                            
                        if ref_info.annotation_gtf:
                            all_available.append(f"`{org}/{asm}` GTF ✅")
                        else:
                            all_missing.append(f"`{org}/{asm}` GTF")
                            
                        if ref_info.star_index:
                            all_available.append(f"`{org}/{asm}` STAR index ✅")
                        if ref_info.salmon_index:
                            all_available.append(f"`{org}/{asm}` Salmon index ✅")
                            
                    except Exception as e:
                        logger.debug(f"Error checking {org}/{asm}: {e}")
            
            # Build message
            if all_available:
                available_list = "\n".join(f"  - {item}" for item in all_available[:20])
                message = f"""🧬 **Available References:**

{available_list}
"""
                if all_missing:
                    missing_list = "\n".join(f"  - {item}" for item in all_missing[:10])
                    message += f"""

⚠️ **Missing (can be downloaded):**

{missing_list}

**To download:**
- `download reference genome for human GRCh38`
- `download reference gtf for mouse GRCm39`
- `build star index for human GRCh38`
"""
            else:
                # Show available organisms
                org_list = ", ".join(REFERENCE_SOURCES.keys())
                message = f"""⚠️ **No reference genomes found in:**
`{base_dir}`

**Available organisms:** {org_list}

**To download references:**
- `download reference genome for human GRCh38`
- `download reference gtf for mouse GRCm39`
- `download reference transcriptome for human GRCh38`

Or download from Ensembl directly:
```bash
wget ftp://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
```
"""
            
            return ToolResult(
                success=True,
                tool_name="check_references",
                data={
                    "references": results,
                    "base_dir": str(base_dir),
                    "available_count": len(all_available),
                    "missing_count": len(all_missing),
                },
                message=message
            )
        
        else:
            # Fallback to simple scan
            found_refs = []
            
            for ref_dir in ref_paths:
                if ref_dir.exists():
                    for f in ref_dir.rglob("*.fa*"):
                        if organism and organism.lower() not in f.stem.lower():
                            continue
                        found_refs.append({
                            "name": f.stem,
                            "path": str(f),
                            "type": "FASTA"
                        })
                    for f in ref_dir.rglob("*.gtf*"):
                        if organism and organism.lower() not in f.stem.lower():
                            continue
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
- `download reference genome for human GRCh38`
- `download reference gtf for mouse GRCm39`
"""
            else:
                message = """⚠️ **No reference genomes found.**

**To download references:**
- `download reference genome for human GRCh38`
- `download reference gtf for mouse GRCm39`

Or download from Ensembl:
```bash
wget ftp://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
```
"""
            
            return ToolResult(
                success=True,
                tool_name="check_references",
                data={"references": found_refs},
                message=message
            )
            
    except Exception as e:
        logger.exception("check_references failed")
        return ToolResult(
            success=False,
            tool_name="check_references",
            error=str(e),
            message=f"❌ Error checking references: {e}"
        )


# =============================================================================
# VISUALIZE_WORKFLOW
# =============================================================================

VISUALIZE_WORKFLOW_PATTERNS = [
    r"(?:visualize|show|display|draw)\s+(?:the\s+)?(?:workflow|pipeline)\s*(?:dag|diagram|graph)?",
    r"(?:create|generate|make)\s+(?:a\s+)?(?:workflow|pipeline)\s+(?:dag|diagram|graph)",
    r"(?:show|display)\s+(?:the\s+)?(?:dag|diagram|graph)\s+(?:for|of)\s+(?:the\s+)?(?:workflow|pipeline)",
]


def visualize_workflow_impl(
    workflow_dir: str = None,
    output_format: str = "png",
    show_containers: bool = True,
) -> ToolResult:
    """
    Generate a visualization (DAG diagram) of a workflow.
    
    Uses WorkflowVisualizer to create graphviz-based DAG diagrams
    showing the data flow through pipeline processes.
    
    Args:
        workflow_dir: Path to workflow directory (or uses most recent)
        output_format: Output format (png, svg, pdf)
        show_containers: Whether to show container info on nodes
        
    Returns:
        ToolResult with visualization path
    """
    try:
        # Try to import WorkflowVisualizer
        try:
            from workflow_composer.viz.visualizer import WorkflowVisualizer
            use_visualizer = True
        except ImportError:
            use_visualizer = False
            logger.debug("WorkflowVisualizer not available")
        
        # Find workflow directory
        if workflow_dir:
            wf_dir = Path(workflow_dir)
        else:
            # Find most recent generated workflow
            possible_dirs = [
                Path.cwd() / "generated_workflows",
                Path.home() / "BioPipelines" / "generated_workflows",
            ]
            
            wf_dir = None
            latest_time = None
            
            for parent in possible_dirs:
                if parent.exists():
                    for d in parent.iterdir():
                        if d.is_dir() and (d / "main.nf").exists():
                            mtime = (d / "main.nf").stat().st_mtime
                            if latest_time is None or mtime > latest_time:
                                latest_time = mtime
                                wf_dir = d
            
            if wf_dir is None:
                return ToolResult(
                    success=False,
                    tool_name="visualize_workflow",
                    error="No workflow found",
                    message="❌ No workflow found. Generate a workflow first with:\n- `create RNA-seq workflow`"
                )
        
        if not wf_dir.exists():
            return ToolResult(
                success=False,
                tool_name="visualize_workflow",
                error=f"Workflow directory not found: {wf_dir}",
                message=f"❌ Workflow not found: `{wf_dir}`"
            )
        
        # Parse workflow to extract processes
        main_nf = wf_dir / "main.nf"
        if not main_nf.exists():
            return ToolResult(
                success=False,
                tool_name="visualize_workflow",
                error="No main.nf found",
                message=f"❌ No `main.nf` found in `{wf_dir}`"
            )
        
        # Extract workflow structure
        content = main_nf.read_text()
        
        # Parse process and workflow blocks
        import re
        processes = re.findall(r'process\s+(\w+)\s*\{', content)
        includes = re.findall(r"include\s*\{\s*(\w+)\s*\}", content)
        
        workflow_name = wf_dir.name
        
        if use_visualizer:
            # Create a simple workflow object for the visualizer
            from dataclasses import dataclass
            
            @dataclass
            class SimpleWorkflow:
                name: str
                modules: list
            
            @dataclass
            class SimpleModule:
                name: str
                container: str = None
            
            modules = []
            for proc in processes + includes:
                modules.append(SimpleModule(name=proc))
            
            workflow = SimpleWorkflow(name=workflow_name, modules=modules)
            
            # Create visualizer
            viz = WorkflowVisualizer(output_dir=str(wf_dir))
            output_path = wf_dir / f"{workflow_name}_dag.{output_format}"
            
            try:
                result_path = viz.render_dag(
                    workflow,
                    output_path=str(output_path),
                    format=output_format,
                    show_containers=show_containers,
                )
                
                return ToolResult(
                    success=True,
                    tool_name="visualize_workflow",
                    data={
                        "path": str(result_path),
                        "workflow": workflow_name,
                        "processes": processes + includes,
                        "format": output_format,
                    },
                    message=f"""📊 **Workflow Visualization Generated**

**File:** `{result_path}`
**Workflow:** {workflow_name}
**Processes:** {len(processes + includes)}

View the diagram to see the data flow through your pipeline.
"""
                )
                
            except Exception as e:
                logger.warning(f"Graphviz rendering failed: {e}, falling back to text")
                # Fall through to text representation
        
        # Fallback: Generate ASCII representation
        lines = [
            f"Workflow: {workflow_name}",
            "=" * 50,
            "",
            "📥 INPUT",
            "   │",
        ]
        
        all_procs = processes + includes
        for i, proc in enumerate(all_procs):
            lines.append(f"   ▼")
            lines.append(f"┌──────────────────────────────┐")
            lines.append(f"│  {proc:<26}  │")
            lines.append(f"└──────────────────────────────┘")
            if i < len(all_procs) - 1:
                lines.append("   │")
        
        lines.append("   │")
        lines.append("   ▼")
        lines.append("📤 OUTPUT")
        
        diagram = "\n".join(lines)
        
        output_path = wf_dir / f"{workflow_name}_dag.txt"
        output_path.write_text(diagram)
        
        return ToolResult(
            success=True,
            tool_name="visualize_workflow",
            data={
                "path": str(output_path),
                "workflow": workflow_name,
                "processes": all_procs,
                "format": "txt",
            },
            message=f"""📊 **Workflow Visualization**

**File:** `{output_path}`
**Processes:** {len(all_procs)}

```
{diagram}
```

_Install graphviz for higher-quality PNG/SVG diagrams:_
```bash
pip install graphviz
apt-get install graphviz  # or: brew install graphviz
```
"""
        )
        
    except Exception as e:
        logger.exception("visualize_workflow failed")
        return ToolResult(
            success=False,
            tool_name="visualize_workflow",
            error=str(e),
            message=f"❌ Visualization error: {e}"
        )
