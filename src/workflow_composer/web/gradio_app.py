#!/usr/bin/env python3
"""
BioPipelines - Modern Gradio Web Interface
===========================================

A beautiful, interactive web UI for AI-powered bioinformatics workflow generation.

Features:
- Chat-based workflow generation with LLM
- Real-time streaming responses
- Tool and module browser with search
- Workflow visualization
- Multiple LLM provider support (OpenAI, vLLM, Ollama)
- File upload for sample sheets
- Download generated workflows
- Pipeline execution with SLURM integration
- Real-time progress monitoring

Usage:
    python -m workflow_composer.web.gradio_app
    # or
    biocomposer ui
"""

import os
import json
import tempfile
import shutil
import subprocess
import threading
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Generator, Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

import gradio as gr

# Import workflow composer components
try:
    from workflow_composer import Composer
    from workflow_composer.core import ToolSelector, ModuleMapper, AnalysisType
    from workflow_composer.llm import get_llm, check_providers, Message
    COMPOSER_AVAILABLE = True
except ImportError:
    COMPOSER_AVAILABLE = False
    print("Warning: workflow_composer not fully installed. Running in demo mode.")


# ============================================================================
# Pipeline Execution Classes
# ============================================================================

class JobStatus(Enum):
    """Pipeline job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineJob:
    """Represents a running or completed pipeline job."""
    job_id: str
    workflow_dir: str
    workflow_name: str
    slurm_job_id: Optional[str] = None
    status: JobStatus = JobStatus.PENDING
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    progress: float = 0.0
    current_process: str = ""
    completed_processes: List[str] = field(default_factory=list)
    total_processes: int = 0
    log_file: Optional[str] = None
    error_message: Optional[str] = None
    nextflow_run_name: Optional[str] = None


class PipelineExecutor:
    """Manages pipeline execution and monitoring."""
    
    def __init__(self):
        self.jobs: Dict[str, PipelineJob] = {}
        self.monitoring_threads: Dict[str, threading.Thread] = {}
        
    def submit_job(
        self,
        workflow_dir: str,
        profile: str = "slurm",
        resume: bool = False,
        params: Dict[str, str] = None,
    ) -> PipelineJob:
        """Submit a pipeline job to SLURM."""
        workflow_path = Path(workflow_dir)
        
        if not workflow_path.exists():
            raise ValueError(f"Workflow directory not found: {workflow_dir}")
        
        main_nf = workflow_path / "main.nf"
        if not main_nf.exists():
            raise ValueError(f"main.nf not found in {workflow_dir}")
        
        # Generate job ID
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        workflow_name = workflow_path.name
        
        # Create log directory
        log_dir = workflow_path / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"nextflow_{job_id}.log"
        
        # Build Nextflow command
        cmd = [
            "nextflow", "run", str(main_nf),
            "-profile", profile,
            "-with-report", str(log_dir / "report.html"),
            "-with-timeline", str(log_dir / "timeline.html"),
            "-with-dag", str(log_dir / "dag.png"),
        ]
        
        if resume:
            cmd.append("-resume")
        
        if params:
            for key, value in params.items():
                cmd.extend(["--" + key, str(value)])
        
        # Create SLURM batch script
        sbatch_script = workflow_path / f"run_{job_id}.sbatch"
        sbatch_content = f"""#!/bin/bash
#SBATCH --job-name=nf_{workflow_name}
#SBATCH --output={log_dir}/slurm_%j.out
#SBATCH --error={log_dir}/slurm_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cpuspot

# Load required modules
module load nextflow 2>/dev/null || true
module load singularity 2>/dev/null || true

# Activate conda if available
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate biopipelines 2>/dev/null || true

# Change to workflow directory
cd {workflow_path}

# Run Nextflow
{' '.join(cmd)} 2>&1 | tee {log_file}

echo "Pipeline finished at $(date)"
"""
        
        sbatch_script.write_text(sbatch_content)
        sbatch_script.chmod(0o755)
        
        # Submit to SLURM
        try:
            result = subprocess.run(
                ["sbatch", str(sbatch_script)],
                capture_output=True,
                text=True,
                cwd=workflow_path,
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"SLURM submission failed: {result.stderr}")
            
            # Extract SLURM job ID
            # Output like: "Submitted batch job 12345"
            match = re.search(r"Submitted batch job (\d+)", result.stdout)
            slurm_job_id = match.group(1) if match else None
            
        except FileNotFoundError:
            # SLURM not available, run directly
            slurm_job_id = None
            # Start in background
            subprocess.Popen(
                cmd,
                cwd=workflow_path,
                stdout=open(log_file, 'w'),
                stderr=subprocess.STDOUT,
            )
        
        # Create job record
        job = PipelineJob(
            job_id=job_id,
            workflow_dir=str(workflow_path),
            workflow_name=workflow_name,
            slurm_job_id=slurm_job_id,
            status=JobStatus.PENDING,
            started_at=datetime.now(),
            log_file=str(log_file),
        )
        
        self.jobs[job_id] = job
        
        # Start monitoring thread
        monitor_thread = threading.Thread(
            target=self._monitor_job,
            args=(job_id,),
            daemon=True,
        )
        monitor_thread.start()
        self.monitoring_threads[job_id] = monitor_thread
        
        return job
    
    def _monitor_job(self, job_id: str):
        """Monitor a running job and update its status."""
        job = self.jobs.get(job_id)
        if not job:
            return
        
        while job.status in (JobStatus.PENDING, JobStatus.RUNNING):
            time.sleep(5)  # Check every 5 seconds
            
            # Check SLURM status if we have a SLURM job ID
            if job.slurm_job_id:
                try:
                    result = subprocess.run(
                        ["squeue", "-j", job.slurm_job_id, "-h", "-o", "%T"],
                        capture_output=True,
                        text=True,
                    )
                    status_str = result.stdout.strip()
                    
                    if not status_str:
                        # Job not in queue - check if completed or failed
                        sacct_result = subprocess.run(
                            ["sacct", "-j", job.slurm_job_id, "-n", "-o", "State", "-P"],
                            capture_output=True,
                            text=True,
                        )
                        final_status = sacct_result.stdout.strip().split('\n')[0]
                        
                        if "COMPLETED" in final_status:
                            job.status = JobStatus.COMPLETED
                            job.progress = 100.0
                        elif "FAILED" in final_status or "CANCELLED" in final_status:
                            job.status = JobStatus.FAILED if "FAILED" in final_status else JobStatus.CANCELLED
                        job.finished_at = datetime.now()
                        break
                    elif status_str == "RUNNING":
                        job.status = JobStatus.RUNNING
                    elif status_str == "PENDING":
                        job.status = JobStatus.PENDING
                        
                except Exception:
                    pass
            
            # Parse log file for progress
            if job.log_file and Path(job.log_file).exists():
                self._parse_nextflow_log(job)
        
    def _parse_nextflow_log(self, job: PipelineJob):
        """Parse Nextflow log file to extract progress."""
        try:
            log_path = Path(job.log_file)
            if not log_path.exists():
                return
            
            content = log_path.read_text()
            
            # Look for process execution lines
            # Format: [hash] process > PROCESS_NAME (sample) [100%] 1 of 1
            process_pattern = r'\[[\w/]+\]\s+process\s+>\s+(\w+).*\[(\d+)%\]\s+(\d+)\s+of\s+(\d+)'
            matches = re.findall(process_pattern, content)
            
            if matches:
                # Get latest progress
                completed = set()
                current = ""
                total_done = 0
                total_all = 0
                
                for process_name, pct, done, total in matches:
                    current = process_name
                    if int(pct) == 100:
                        completed.add(process_name)
                    total_done = max(total_done, int(done))
                    total_all = max(total_all, int(total))
                
                job.current_process = current
                job.completed_processes = list(completed)
                
                if total_all > 0:
                    job.progress = (total_done / total_all) * 100
            
            # Check for completion
            if "Pipeline completed" in content or "Workflow completed" in content:
                job.status = JobStatus.COMPLETED
                job.progress = 100.0
                job.finished_at = datetime.now()
            elif "Error executing process" in content or "Pipeline failed" in content:
                job.status = JobStatus.FAILED
                job.finished_at = datetime.now()
                # Extract error
                error_match = re.search(r'Error executing process.*?(?=\n\n|\Z)', content, re.DOTALL)
                if error_match:
                    job.error_message = error_match.group()[:500]
                    
        except Exception as e:
            print(f"Error parsing log: {e}")
    
    def get_job_status(self, job_id: str) -> Optional[PipelineJob]:
        """Get status of a specific job."""
        return self.jobs.get(job_id)
    
    def list_jobs(self) -> List[PipelineJob]:
        """List all jobs."""
        return list(self.jobs.values())
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        if job.slurm_job_id:
            try:
                subprocess.run(["scancel", job.slurm_job_id], check=True)
                job.status = JobStatus.CANCELLED
                job.finished_at = datetime.now()
                return True
            except Exception:
                return False
        
        return False
    
    def get_slurm_jobs(self) -> List[Dict[str, str]]:
        """Get all SLURM jobs for current user."""
        try:
            result = subprocess.run(
                ["squeue", "-u", os.environ.get("USER", ""), "-h", 
                 "-o", "%i|%j|%T|%M|%P|%R"],
                capture_output=True,
                text=True,
            )
            
            jobs = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('|')
                    if len(parts) >= 6:
                        jobs.append({
                            "job_id": parts[0],
                            "name": parts[1],
                            "status": parts[2],
                            "time": parts[3],
                            "partition": parts[4],
                            "node": parts[5],
                        })
            return jobs
        except Exception:
            return []


# Global executor instance
pipeline_executor = PipelineExecutor()


# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).parent.parent.parent.parent
GENERATED_DIR = BASE_DIR / "generated_workflows"
GENERATED_DIR.mkdir(exist_ok=True)

# Custom CSS for styling (Gradio 6.x compatible)
CUSTOM_CSS = """
.main-header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #059669 0%, #3b82f6 100%);
    border-radius: 12px;
    margin-bottom: 20px;
}
.main-header h1 {
    color: white !important;
    margin: 0;
    font-size: 2.2em;
}
.main-header p {
    color: rgba(255,255,255,0.9) !important;
    margin: 5px 0 0 0;
}
.stat-box {
    text-align: center;
    padding: 15px;
    background: linear-gradient(135deg, #f8fafc, #f1f5f9);
    border-radius: 10px;
    border: 1px solid #e2e8f0;
}
footer {
    display: none !important;
}
"""

# Analysis type descriptions for better UX
ANALYSIS_EXAMPLES = {
    "RNA-seq": "RNA-seq differential expression analysis for mouse samples comparing treatment vs control using STAR and DESeq2",
    "ChIP-seq": "ChIP-seq peak calling for human H3K27ac samples with input controls using Bowtie2 and MACS2",
    "ATAC-seq": "ATAC-seq chromatin accessibility analysis for human cells with Bowtie2 alignment and MACS2 peak calling",
    "Variant Calling": "Whole exome sequencing variant calling for human samples using BWA-MEM2 and GATK HaplotypeCaller",
    "Single-cell RNA": "10x Genomics single-cell RNA-seq analysis with STARsolo and Seurat clustering",
    "Metagenomics": "Shotgun metagenomics analysis with Kraken2 taxonomic classification and MetaPhlAn profiling",
    "Long-read": "Oxford Nanopore long-read sequencing analysis with minimap2 alignment and structural variant calling",
    "Methylation": "Bisulfite sequencing methylation analysis with Bismark alignment and methylation calling",
}


# ============================================================================
# Global State
# ============================================================================

class AppState:
    """Application state manager."""
    
    def __init__(self):
        self.composer: Optional[Composer] = None
        self.tool_selector: Optional[ToolSelector] = None
        self.module_mapper: Optional[ModuleMapper] = None
        self.current_provider = "openai"
        self.chat_history: List[Tuple[str, str]] = []
        self.last_generated_workflow: Optional[str] = None  # Track last workflow for quick run
        
    def initialize(self, provider: str = "openai", model: str = None):
        """Initialize or reinitialize with a specific provider."""
        try:
            available = check_providers() if COMPOSER_AVAILABLE else {}
            
            if provider not in available or not available.get(provider):
                # Fall back to first available
                for p, is_available in available.items():
                    if is_available:
                        provider = p
                        break
                else:
                    return False, "No LLM providers available"
            
            llm = get_llm(provider, model=model) if model else get_llm(provider)
            self.composer = Composer(llm=llm)
            self.current_provider = provider
            
            # Initialize tool selector and module mapper
            self.tool_selector = self.composer.tool_selector
            self.module_mapper = self.composer.module_mapper
            
            return True, f"Initialized with {provider}"
        except Exception as e:
            return False, str(e)
    
    def get_stats(self) -> Dict[str, int]:
        """Get tool and module statistics."""
        stats = {
            "tools": 0,
            "modules": 0,
            "containers": 12,
            "analysis_types": len(AnalysisType) if COMPOSER_AVAILABLE else 38
        }
        
        if self.tool_selector:
            stats["tools"] = len(self.tool_selector.tools)
        if self.module_mapper:
            stats["modules"] = len(self.module_mapper.modules)
            
        return stats


# Global app state
app_state = AppState()


# ============================================================================
# Core Functions
# ============================================================================

def check_available_providers() -> Dict[str, bool]:
    """Check which LLM providers are available."""
    if not COMPOSER_AVAILABLE:
        return {"openai": False, "vllm": False, "ollama": False, "anthropic": False}
    
    try:
        return check_providers()
    except:
        return {"openai": False, "vllm": False, "ollama": False, "anthropic": False}


def get_provider_choices() -> List[str]:
    """Get list of available provider choices for dropdown."""
    available = check_available_providers()
    choices = []
    
    provider_labels = {
        "openai": "🟢 OpenAI (GPT-4o)",
        "vllm": "🟣 vLLM (Local GPU)",
        "ollama": "🟠 Ollama (Local)",
        "anthropic": "🔵 Anthropic (Claude)",
    }
    
    for provider, is_available in available.items():
        if is_available:
            choices.append(provider_labels.get(provider, provider))
        else:
            # Show unavailable options grayed out
            choices.append(f"⚫ {provider.title()} (not configured)")
    
    return choices if choices else ["⚫ No providers available"]


def extract_provider_key(choice: str) -> str:
    """Extract provider key from dropdown choice."""
    mapping = {
        "openai": "openai",
        "vllm": "vllm",
        "ollama": "ollama",
        "anthropic": "anthropic",
    }
    
    choice_lower = choice.lower()
    for key in mapping:
        if key in choice_lower:
            return key
    return "openai"


def chat_with_composer(
    message: str,
    history: List[Dict[str, str]],
    provider: str,
) -> Generator[Tuple[List[Dict[str, str]], str], None, None]:
    """
    Chat with the AI workflow composer.
    Streams responses for better UX.
    Uses Gradio 6.0 message format: [{"role": "user/assistant", "content": "..."}]
    """
    if not message.strip():
        yield history, ""
        return
    
    # Extract actual provider
    provider_key = extract_provider_key(provider)
    
    # Initialize if needed
    if app_state.composer is None or app_state.current_provider != provider_key:
        success, msg = app_state.initialize(provider_key)
        if not success:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": f"❌ Error: {msg}"})
            yield history, ""
            return
    
    # Add user message to history
    history.append({"role": "user", "content": message})
    yield history, ""
    
    # Check if this is a workflow generation request
    is_generation_request = any(kw in message.lower() for kw in [
        "generate", "create", "build", "make", "workflow", "pipeline",
        "analyze", "analysis", "process", "run"
    ])
    
    try:
        if is_generation_request and app_state.composer:
            # Generate workflow
            response_parts = []
            
            # First, parse the intent
            history.append({"role": "assistant", "content": "🔍 Parsing your request..."})
            yield history, ""
            
            intent = app_state.composer.parse_intent(message)
            intent_info = f"""
📋 **Detected Analysis:**
- Type: `{intent.analysis_type.value}`
- Organism: `{intent.organism or 'Not specified'}`
- Genome: `{intent.genome_build or 'Auto-detect'}`
- Confidence: `{intent.confidence:.0%}`

"""
            response_parts.append(intent_info)
            history[-1] = {"role": "assistant", "content": "".join(response_parts) + "🔧 Checking tool availability..."}
            yield history, ""
            
            # Check readiness
            readiness = app_state.composer.check_readiness(message)
            
            if readiness.get("ready"):
                tools_found = readiness.get("tools_found", 0)
                modules_found = readiness.get("modules_found", 0)
                
                readiness_info = f"""✅ **Ready to generate!**
- Tools available: `{tools_found}`
- Modules available: `{modules_found}`

"""
                response_parts.append(readiness_info)
                history[-1] = {"role": "assistant", "content": "".join(response_parts) + "⚙️ Generating workflow..."}
                yield history, ""
                
                # Generate the workflow
                workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                output_dir = GENERATED_DIR / workflow_id
                
                workflow = app_state.composer.generate(message, output_dir=str(output_dir))
                
                # Track last generated workflow for quick run
                app_state.last_generated_workflow = str(output_dir)
                
                # Format response
                workflow_info = f"""🎉 **Workflow Generated!**

**Name:** `{workflow.name if hasattr(workflow, 'name') else workflow_id}`
**Output:** `{output_dir}`

**Tools used:**
{', '.join(f'`{t}`' for t in (workflow.tools_used if hasattr(workflow, 'tools_used') else []))}

**Modules:**
{', '.join(f'`{m}`' for m in (workflow.modules_used if hasattr(workflow, 'modules_used') else []))}

📥 Use the **Download** tab to get your workflow files.
"""
                response_parts.append(workflow_info)
                
            else:
                issues = readiness.get("issues", ["Unknown issue"])
                response_parts.append(f"""⚠️ **Cannot generate workflow:**
{chr(10).join(f'- {issue}' for issue in issues)}

Please provide more details or check tool availability.
""")
            
            history[-1] = {"role": "assistant", "content": "".join(response_parts)}
            yield history, ""
            
        else:
            # Regular chat - use LLM for conversational response
            if app_state.composer and app_state.composer.llm:
                # Create system context
                system_msg = """You are BioPipelines AI Assistant, an expert in bioinformatics workflow design.
You help users:
1. Design and generate bioinformatics pipelines (RNA-seq, ChIP-seq, variant calling, etc.)
2. Explain bioinformatics tools and methods
3. Troubleshoot pipeline issues
4. Recommend best practices

Be concise but helpful. Use markdown formatting."""
                
                messages = [Message.system(system_msg)]
                
                # Add chat history (convert from new format)
                for msg in history[:-1]:  # Exclude last user message we just added
                    if msg["role"] == "user":
                        messages.append(Message.user(msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(Message.assistant(msg["content"]))
                
                messages.append(Message.user(message))
                
                # Get response (streaming if supported)
                response = app_state.composer.llm.chat(messages)
                history.append({"role": "assistant", "content": response.content})
                yield history, ""
            else:
                history.append({"role": "assistant", "content": "I'm not fully initialized. Please select an LLM provider."})
                yield history, ""
                
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        history.append({"role": "assistant", "content": error_msg})
        yield history, ""


def search_tools(query: str, container_filter: str = "") -> str:
    """Search for bioinformatics tools."""
    if not query or len(query) < 2:
        return "Enter at least 2 characters to search..."
    
    if not app_state.tool_selector:
        # Return demo data
        return """
| Tool | Container | Category |
|------|-----------|----------|
| fastqc | base | QC |
| multiqc | base | QC |
| star | rna-seq | Alignment |
| bwa | dna-seq | Alignment |
| gatk | dna-seq | Variant Calling |
        """
    
    try:
        results = app_state.tool_selector.fuzzy_search(
            query, 
            limit=20,
            container=container_filter if container_filter else None
        )
        
        if not results:
            return "No tools found matching your query."
        
        # Format as markdown table
        table = "| Tool | Container | Category | Score |\n|------|-----------|----------|-------|\n"
        for match in results[:15]:
            tool = match.tool
            table += f"| `{tool.name}` | {tool.container} | {tool.category or '-'} | {match.score:.2f} |\n"
        
        return table
    except Exception as e:
        return f"Search error: {e}"


def get_modules_by_category() -> str:
    """Get all modules organized by category."""
    if not app_state.module_mapper:
        # Return demo data
        return """
## 📦 Available Modules

### QC
- fastqc
- multiqc
- fastp

### Alignment
- star
- bwa
- bowtie2
- minimap2

### Quantification
- featurecounts
- salmon
- kallisto

### Variant Calling
- gatk_haplotypecaller
- bcftools
- deepvariant
        """
    
    try:
        modules = app_state.module_mapper.list_by_category()
        
        output = "## 📦 Available Modules\n\n"
        for category, mods in sorted(modules.items()):
            output += f"### {category.title()}\n"
            for mod in sorted(mods)[:10]:
                output += f"- `{mod}`\n"
            if len(mods) > 10:
                output += f"- *... and {len(mods) - 10} more*\n"
            output += "\n"
        
        return output
    except Exception as e:
        return f"Error loading modules: {e}"


def get_example_prompts() -> List[List[str]]:
    """Get example prompts for the UI."""
    return [
        [example] for example in ANALYSIS_EXAMPLES.values()
    ]


def download_latest_workflow() -> Optional[str]:
    """Get path to latest generated workflow for download."""
    try:
        workflows = sorted(GENERATED_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
        if workflows:
            latest = workflows[0]
            zip_path = GENERATED_DIR / f"{latest.name}.zip"
            shutil.make_archive(str(zip_path.with_suffix('')), 'zip', latest)
            return str(zip_path)
    except Exception as e:
        print(f"Download error: {e}")
    return None


def refresh_stats() -> Tuple[str, str, str, str]:
    """Refresh and return statistics."""
    stats = app_state.get_stats()
    return (
        f"🔧 {stats['tools']}",
        f"📦 {stats['modules']}",
        f"🐳 {stats['containers']}",
        f"🧬 {stats['analysis_types']}"
    )


# ============================================================================
# Pipeline Execution Functions
# ============================================================================

def get_available_workflows() -> List[str]:
    """Get list of available workflows to run."""
    try:
        workflows = []
        for d in sorted(GENERATED_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if d.is_dir() and (d / "main.nf").exists():
                workflows.append(d.name)
        return workflows[:20]  # Limit to 20 most recent
    except Exception:
        return []


def submit_pipeline(
    workflow_name: str,
    profile: str,
    resume: bool,
    reads_path: str,
    genome_path: str,
    outdir: str,
) -> str:
    """Submit a pipeline to SLURM."""
    if not workflow_name:
        return "❌ Please select a workflow to run"
    
    workflow_dir = GENERATED_DIR / workflow_name
    if not workflow_dir.exists():
        return f"❌ Workflow not found: {workflow_name}"
    
    # Build parameters
    params = {}
    if reads_path.strip():
        params["reads"] = reads_path.strip()
    if genome_path.strip():
        params["genome"] = genome_path.strip()
    if outdir.strip():
        params["outdir"] = outdir.strip()
    else:
        params["outdir"] = str(workflow_dir / "results")
    
    try:
        job = pipeline_executor.submit_job(
            workflow_dir=str(workflow_dir),
            profile=profile,
            resume=resume,
            params=params,
        )
        
        if job.slurm_job_id:
            return f"""✅ **Pipeline Submitted!**

| Field | Value |
|-------|-------|
| **Job ID** | `{job.job_id}` |
| **SLURM Job** | `{job.slurm_job_id}` |
| **Workflow** | `{job.workflow_name}` |
| **Status** | 🟡 {job.status.value} |
| **Log File** | `{job.log_file}` |

Use the **Monitor** section below to track progress.
"""
        else:
            return f"""✅ **Pipeline Started (Local)**

| Field | Value |
|-------|-------|
| **Job ID** | `{job.job_id}` |
| **Workflow** | `{job.workflow_name}` |
| **Status** | 🟡 {job.status.value} |
| **Log File** | `{job.log_file}` |

Note: SLURM not available, running locally.
"""
    except Exception as e:
        return f"❌ **Submission Failed**\n\nError: {str(e)}"


def get_job_status_display() -> str:
    """Get formatted display of all job statuses."""
    jobs = pipeline_executor.list_jobs()
    
    if not jobs:
        return "No pipeline jobs submitted yet. Submit a pipeline above to get started."
    
    output = "## 📊 Pipeline Jobs\n\n"
    
    # Status icons
    status_icons = {
        JobStatus.PENDING: "🟡",
        JobStatus.RUNNING: "🔵",
        JobStatus.COMPLETED: "✅",
        JobStatus.FAILED: "❌",
        JobStatus.CANCELLED: "⚪",
    }
    
    output += "| Status | Job ID | Workflow | Progress | Current Process | Time |\n"
    output += "|--------|--------|----------|----------|-----------------|------|\n"
    
    for job in sorted(jobs, key=lambda j: j.started_at or datetime.min, reverse=True):
        icon = status_icons.get(job.status, "⚪")
        
        # Calculate runtime
        if job.started_at:
            end_time = job.finished_at or datetime.now()
            duration = end_time - job.started_at
            time_str = str(duration).split('.')[0]  # Remove microseconds
        else:
            time_str = "-"
        
        progress_bar = f"{job.progress:.0f}%"
        current = job.current_process[:20] if job.current_process else "-"
        
        output += f"| {icon} {job.status.value} | `{job.job_id[-15:]}` | {job.workflow_name[:15]} | {progress_bar} | {current} | {time_str} |\n"
    
    return output


def get_slurm_queue_display() -> str:
    """Get display of SLURM queue for current user."""
    jobs = pipeline_executor.get_slurm_jobs()
    
    if not jobs:
        return "No SLURM jobs in queue."
    
    output = "## 🖥️ SLURM Queue\n\n"
    output += "| Job ID | Name | Status | Time | Partition | Node |\n"
    output += "|--------|------|--------|------|-----------|------|\n"
    
    for job in jobs:
        status_icon = "🔵" if job["status"] == "RUNNING" else "🟡"
        output += f"| {job['job_id']} | {job['name'][:15]} | {status_icon} {job['status']} | {job['time']} | {job['partition']} | {job['node']} |\n"
    
    return output


def get_job_logs(job_id: str, tail_lines: int = 50) -> str:
    """Get recent log output for a job."""
    job = pipeline_executor.get_job_status(job_id)
    
    if not job:
        return f"Job not found: {job_id}"
    
    if not job.log_file or not Path(job.log_file).exists():
        return "Log file not available yet. The job may still be starting."
    
    try:
        # Read last N lines
        with open(job.log_file, 'r') as f:
            lines = f.readlines()
            recent = lines[-tail_lines:] if len(lines) > tail_lines else lines
        
        output = f"## 📄 Logs for `{job.job_id}`\n\n"
        output += f"*Showing last {len(recent)} lines from `{job.log_file}`*\n\n"
        output += "```\n"
        output += "".join(recent)
        output += "\n```"
        
        return output
    except Exception as e:
        return f"Error reading log: {e}"


def cancel_selected_job(job_id: str) -> str:
    """Cancel a selected job."""
    if not job_id:
        return "Please enter a job ID to cancel."
    
    # Find full job ID if partial
    matching_jobs = [j for j in pipeline_executor.list_jobs() if job_id in j.job_id]
    
    if not matching_jobs:
        return f"No job found matching: {job_id}"
    
    job = matching_jobs[0]
    
    if job.status not in (JobStatus.PENDING, JobStatus.RUNNING):
        return f"Job `{job.job_id}` is already {job.status.value}."
    
    if pipeline_executor.cancel_job(job.job_id):
        return f"✅ Job `{job.job_id}` has been cancelled."
    else:
        return f"❌ Failed to cancel job `{job.job_id}`."


def get_progress_details(job_id: str) -> str:
    """Get detailed progress for a specific job."""
    job = pipeline_executor.get_job_status(job_id)
    
    if not job:
        return "Select a job to view progress details."
    
    status_icons = {
        JobStatus.PENDING: "🟡",
        JobStatus.RUNNING: "🔵",
        JobStatus.COMPLETED: "✅",
        JobStatus.FAILED: "❌",
        JobStatus.CANCELLED: "⚪",
    }
    
    icon = status_icons.get(job.status, "⚪")
    
    output = f"""## {icon} Job Details: `{job.job_id}`

| Field | Value |
|-------|-------|
| **Workflow** | `{job.workflow_name}` |
| **Status** | {job.status.value} |
| **SLURM Job** | `{job.slurm_job_id or 'N/A'}` |
| **Progress** | {job.progress:.1f}% |
| **Started** | {job.started_at.strftime('%Y-%m-%d %H:%M:%S') if job.started_at else 'N/A'} |
| **Finished** | {job.finished_at.strftime('%Y-%m-%d %H:%M:%S') if job.finished_at else 'Running...'} |

### Current Process
`{job.current_process or 'N/A'}`

### Completed Processes
"""
    
    if job.completed_processes:
        for proc in job.completed_processes:
            output += f"- ✅ `{proc}`\n"
    else:
        output += "*No processes completed yet*\n"
    
    if job.error_message:
        output += f"\n### ❌ Error\n```\n{job.error_message}\n```"
    
    # Progress bar visualization
    filled = int(job.progress / 5)
    empty = 20 - filled
    progress_bar = "█" * filled + "░" * empty
    output += f"\n### Progress Bar\n`[{progress_bar}]` {job.progress:.0f}%"
    
    return output


def refresh_monitoring() -> Tuple[str, str]:
    """Refresh all monitoring displays."""
    return get_job_status_display(), get_slurm_queue_display()


# ============================================================================
# Gradio Interface
# ============================================================================

def create_interface() -> gr.Blocks:
    """Create the Gradio interface."""
    
    with gr.Blocks(
        title="BioPipelines - AI Workflow Composer",
    ) as demo:
        
        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #059669 0%, #3b82f6 100%); border-radius: 12px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0; font-size: 2.2em;">🧬 BioPipelines</h1>
            <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0;">AI-Powered Bioinformatics Workflow Composer</p>
        </div>
        """)
        
        # Stats Row
        with gr.Row():
            tools_stat = gr.Markdown("🔧 Loading...")
            modules_stat = gr.Markdown("📦 Loading...")
            containers_stat = gr.Markdown("🐳 12")
            analyses_stat = gr.Markdown("🧬 38")
        
        # Main Tabs
        with gr.Tabs():
            
            # ========== Chat Tab ==========
            with gr.TabItem("💬 Chat & Generate", id="chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="BioPipelines AI",
                            height=450,
                        )
                        
                        with gr.Row():
                            msg_input = gr.Textbox(
                                label="Your message",
                                placeholder="Describe the bioinformatics analysis you want to perform...",
                                lines=2,
                                scale=4,
                            )
                            send_btn = gr.Button("Send 🚀", variant="primary", scale=1)
                        
                        with gr.Accordion("📝 Example Prompts", open=False):
                            gr.Examples(
                                examples=get_example_prompts(),
                                inputs=msg_input,
                                label="Click an example to use it:",
                            )
                    
                    with gr.Column(scale=1):
                        provider_dropdown = gr.Dropdown(
                            choices=get_provider_choices(),
                            value=get_provider_choices()[0] if get_provider_choices() else None,
                            label="🤖 LLM Provider",
                            interactive=True,
                        )
                        
                        gr.Markdown("### Quick Actions")
                        
                        clear_btn = gr.Button("🗑️ Clear Chat", size="sm")
                        
                        gr.Markdown("""
                        ### Tips
                        - Be specific about organism and genome
                        - Mention specific tools if preferred
                        - Include sample type (paired-end, etc.)
                        - Describe comparison groups
                        """)
            
            # ========== Tools Tab ==========
            with gr.TabItem("🔧 Tool Browser", id="tools"):
                with gr.Row():
                    tool_search = gr.Textbox(
                        label="Search Tools",
                        placeholder="e.g., alignment, variant, fastq...",
                        scale=3,
                    )
                    container_filter = gr.Dropdown(
                        choices=["", "base", "rna-seq", "dna-seq", "chip-seq", "atac-seq", 
                                 "scrna-seq", "metagenomics", "methylation", "long-read"],
                        label="Filter by Container",
                        scale=1,
                    )
                
                tool_results = gr.Markdown("Enter a search term to find tools...")
                
                tool_search.change(
                    fn=search_tools,
                    inputs=[tool_search, container_filter],
                    outputs=tool_results,
                )
                container_filter.change(
                    fn=search_tools,
                    inputs=[tool_search, container_filter],
                    outputs=tool_results,
                )
            
            # ========== Modules Tab ==========
            with gr.TabItem("📦 Modules", id="modules"):
                modules_display = gr.Markdown(get_modules_by_category())
                refresh_modules_btn = gr.Button("🔄 Refresh Modules")
                refresh_modules_btn.click(
                    fn=get_modules_by_category,
                    outputs=modules_display,
                )
            
            # ========== Run Pipeline Tab ==========
            with gr.TabItem("🚀 Run Pipeline", id="run"):
                gr.Markdown("""
                ## Run Generated Workflows
                
                Submit your generated workflows to the SLURM cluster for execution.
                Monitor progress in real-time below.
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Submit Pipeline")
                        
                        workflow_dropdown = gr.Dropdown(
                            choices=get_available_workflows(),
                            label="Select Workflow",
                            interactive=True,
                        )
                        refresh_workflows_btn = gr.Button("🔄 Refresh", size="sm")
                        
                        profile_dropdown = gr.Dropdown(
                            choices=["slurm", "local", "docker", "singularity"],
                            value="slurm",
                            label="Execution Profile",
                        )
                        
                        resume_checkbox = gr.Checkbox(
                            label="Resume from previous run",
                            value=False,
                        )
                        
                        with gr.Accordion("📁 Input Parameters (Optional)", open=False):
                            reads_input = gr.Textbox(
                                label="Reads Path",
                                placeholder="/path/to/reads/*.fastq.gz",
                            )
                            genome_input = gr.Textbox(
                                label="Genome/Reference Path",
                                placeholder="/path/to/genome.fa",
                            )
                            outdir_input = gr.Textbox(
                                label="Output Directory",
                                placeholder="Leave empty for default (workflow_dir/results)",
                            )
                        
                        submit_btn = gr.Button("🚀 Submit Pipeline", variant="primary", size="lg")
                        submission_result = gr.Markdown("")
                    
                    with gr.Column(scale=3):
                        gr.Markdown("### Job Details")
                        
                        job_selector = gr.Dropdown(
                            choices=[],
                            label="Select Job for Details",
                            interactive=True,
                        )
                        job_details_display = gr.Markdown("Select a job to view details.")
                        
                        with gr.Row():
                            refresh_details_btn = gr.Button("🔄 Refresh", size="sm")
                            cancel_btn = gr.Button("🛑 Cancel Job", variant="stop", size="sm")
                
                gr.Markdown("---")
                
                # Monitoring Section
                gr.Markdown("### 📊 Pipeline Monitor")
                
                with gr.Row():
                    with gr.Column():
                        jobs_display = gr.Markdown("No pipeline jobs yet.")
                    with gr.Column():
                        slurm_display = gr.Markdown("SLURM queue loading...")
                
                with gr.Row():
                    refresh_monitor_btn = gr.Button("🔄 Refresh All", variant="secondary")
                    auto_refresh = gr.Checkbox(label="Auto-refresh every 10s", value=False)
                
                # Logs Section
                gr.Markdown("---")
                gr.Markdown("### 📄 Job Logs")
                
                with gr.Row():
                    log_job_input = gr.Textbox(
                        label="Job ID (or partial)",
                        placeholder="Enter job ID to view logs",
                        scale=2,
                    )
                    log_lines_slider = gr.Slider(
                        minimum=20,
                        maximum=200,
                        value=50,
                        step=10,
                        label="Lines to show",
                        scale=1,
                    )
                    view_logs_btn = gr.Button("View Logs", scale=1)
                
                logs_display = gr.Markdown("Enter a job ID to view logs.")
                
                # Event handlers for Run tab
                refresh_workflows_btn.click(
                    fn=lambda: gr.update(choices=get_available_workflows()),
                    outputs=workflow_dropdown,
                )
                
                submit_btn.click(
                    fn=submit_pipeline,
                    inputs=[
                        workflow_dropdown,
                        profile_dropdown,
                        resume_checkbox,
                        reads_input,
                        genome_input,
                        outdir_input,
                    ],
                    outputs=submission_result,
                ).then(
                    fn=lambda: gr.update(choices=[j.job_id for j in pipeline_executor.list_jobs()]),
                    outputs=job_selector,
                ).then(
                    fn=refresh_monitoring,
                    outputs=[jobs_display, slurm_display],
                )
                
                job_selector.change(
                    fn=get_progress_details,
                    inputs=job_selector,
                    outputs=job_details_display,
                )
                
                refresh_details_btn.click(
                    fn=get_progress_details,
                    inputs=job_selector,
                    outputs=job_details_display,
                )
                
                cancel_btn.click(
                    fn=cancel_selected_job,
                    inputs=job_selector,
                    outputs=submission_result,
                ).then(
                    fn=refresh_monitoring,
                    outputs=[jobs_display, slurm_display],
                )
                
                refresh_monitor_btn.click(
                    fn=refresh_monitoring,
                    outputs=[jobs_display, slurm_display],
                ).then(
                    fn=lambda: gr.update(choices=[j.job_id for j in pipeline_executor.list_jobs()]),
                    outputs=job_selector,
                )
                
                view_logs_btn.click(
                    fn=get_job_logs,
                    inputs=[log_job_input, log_lines_slider],
                    outputs=logs_display,
                )
            
            # ========== Download Tab ==========
            with gr.TabItem("📥 Download", id="download"):
                gr.Markdown("""
                ## Download Generated Workflows
                
                After generating a workflow in the Chat tab, you can download it here.
                The download includes:
                - `main.nf` - Main Nextflow workflow
                - `nextflow.config` - Configuration file
                - `modules/` - Required modules
                - `README.md` - Usage instructions
                """)
                
                download_btn = gr.Button("📥 Download Latest Workflow", variant="primary")
                download_file = gr.File(label="Download")
                
                download_btn.click(
                    fn=download_latest_workflow,
                    outputs=download_file,
                )
            
            # ========== Settings Tab ==========
            with gr.TabItem("⚙️ Settings", id="settings"):
                gr.Markdown("## LLM Configuration")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### OpenAI")
                        openai_status = gr.Markdown(
                            "✅ Configured" if os.getenv("OPENAI_API_KEY") else "❌ Not configured"
                        )
                        gr.Markdown("Set `OPENAI_API_KEY` environment variable")
                    
                    with gr.Column():
                        gr.Markdown("### vLLM")
                        vllm_url = gr.Textbox(
                            label="vLLM Server URL",
                            value=os.getenv("VLLM_API_BASE", "http://localhost:8000/v1"),
                            interactive=True,
                        )
                        vllm_model = gr.Dropdown(
                            choices=["llama3.1-8b", "mistral-7b", "qwen2.5-7b", "codellama-34b"],
                            label="Model",
                            value="mistral-7b",
                        )
                
                gr.Markdown("---")
                gr.Markdown("### System Info")
                
                system_info = gr.Markdown(f"""
                - **Workflow Composer:** {'✅ Available' if COMPOSER_AVAILABLE else '❌ Not installed'}
                - **Generated Workflows Dir:** `{GENERATED_DIR}`
                - **Python Path:** `{Path(__file__).parent}`
                """)
        
        # ========== Event Handlers ==========
        
        # Chat submission
        msg_input.submit(
            fn=chat_with_composer,
            inputs=[msg_input, chatbot, provider_dropdown],
            outputs=[chatbot, msg_input],
        )
        
        send_btn.click(
            fn=chat_with_composer,
            inputs=[msg_input, chatbot, provider_dropdown],
            outputs=[chatbot, msg_input],
        )
        
        # Clear chat
        clear_btn.click(
            fn=lambda: ([], ""),
            outputs=[chatbot, msg_input],
        )
        
        # Load stats on start
        demo.load(
            fn=refresh_stats,
            outputs=[tools_stat, modules_stat, containers_stat, analyses_stat],
        )
        
        # Initialize app state on provider change
        provider_dropdown.change(
            fn=lambda p: app_state.initialize(extract_provider_key(p)),
            inputs=[provider_dropdown],
        )
    
    return demo


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Launch the Gradio web interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BioPipelines Web UI")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║        🧬 BioPipelines - AI Workflow Composer                    ║
║                      Web Interface                               ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Starting Gradio server...                                       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize app state
    available = check_available_providers()
    if any(available.values()):
        provider = next(k for k, v in available.items() if v)
        app_state.initialize(provider)
        print(f"  ✅ Initialized with {provider} provider")
    else:
        print("  ⚠️  No LLM providers available - running in demo mode")
    
    # Create and launch interface
    demo = create_interface()
    
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True,
    )


if __name__ == "__main__":
    main()
