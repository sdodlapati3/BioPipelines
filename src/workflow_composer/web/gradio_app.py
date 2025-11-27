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
import logging
from pathlib import Path
from datetime import datetime
from typing import Generator, Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

import gradio as gr

logger = logging.getLogger(__name__)

# Import workflow composer components
try:
    from workflow_composer import Composer
    from workflow_composer.core import ToolSelector, ModuleMapper, AnalysisType
    from workflow_composer.llm import get_llm, check_providers, Message
    COMPOSER_AVAILABLE = True
except ImportError:
    COMPOSER_AVAILABLE = False
    print("Warning: workflow_composer not fully installed. Running in demo mode.")

# Import model orchestrator for ensemble management
try:
    from workflow_composer.core import (
        get_model_manager,
        AdaptiveQueryParser,
        ORCHESTRATOR_AVAILABLE
    )
    # Fallback to old names for compatibility
    get_orchestrator = get_model_manager
except ImportError:
    ORCHESTRATOR_AVAILABLE = False
    get_model_manager = None
    get_orchestrator = None
    AdaptiveQueryParser = None
    print("Note: Model orchestrator not available - using standard parsing.")

# Import error diagnosis agent
try:
    from workflow_composer.diagnosis import (
        ErrorDiagnosisAgent,
        AutoFixEngine,
        FixRiskLevel,
    )
    DIAGNOSIS_AVAILABLE = True
except ImportError:
    DIAGNOSIS_AVAILABLE = False
    ErrorDiagnosisAgent = None
    AutoFixEngine = None
    FixRiskLevel = None
    print("Note: Error diagnosis agent not available.")

# Import results visualization
try:
    from workflow_composer.results import (
        ResultCollector,
        ResultViewer,
        ResultArchiver,
    )
    RESULTS_AVAILABLE = True
except ImportError:
    RESULTS_AVAILABLE = False
    ResultCollector = None
    ResultViewer = None
    ResultArchiver = None
    print("Note: Results visualization not available.")

# Import data discovery
try:
    from workflow_composer.data.discovery import DataDiscovery, SearchQuery
    from workflow_composer.data.browser import create_reference_browser_tab
    DISCOVERY_AVAILABLE = True
except ImportError:
    DISCOVERY_AVAILABLE = False
    DataDiscovery = None
    create_reference_browser_tab = None
    print("Note: Data discovery not available.")

# Import new data-first components
try:
    from workflow_composer.web.components.data_tab import (
        create_data_tab,
        create_local_scanner_ui,
        create_remote_search_ui,
        create_reference_manager_ui,
        create_data_summary_panel,
    )
    from workflow_composer.data import DataManifest, LocalSampleScanner, ReferenceManager
    DATA_TAB_AVAILABLE = True
except ImportError:
    DATA_TAB_AVAILABLE = False
    create_data_tab = None
    DataManifest = None
    LocalSampleScanner = None
    ReferenceManager = None
    print("Note: Data tab components not available.")

# Import agent tools for unified workspace
try:
    from workflow_composer.agents import AgentTools, process_tool_request, ToolResult
    from workflow_composer.agents import AgentBridge, get_agent_bridge
    from workflow_composer.agents.context import ConversationContext
    AGENT_TOOLS_AVAILABLE = True
    
    # Check if local vLLM is configured (set by start_gradio.sh --gpu)
    USE_LOCAL_LLM = os.environ.get("USE_LOCAL_LLM", "").lower() == "true"
    VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1")
    
    if USE_LOCAL_LLM:
        print(f"✓ Local LLM mode enabled: {VLLM_URL}")
except ImportError as e:
    AGENT_TOOLS_AVAILABLE = False
    AgentTools = None
    process_tool_request = None
    ToolResult = None
    AgentBridge = None
    get_agent_bridge = None
    ConversationContext = None
    USE_LOCAL_LLM = False
    VLLM_URL = None
    print(f"Note: Agent tools not available: {e}")

# Import enhanced agent components (ReAct, Memory, Self-Healing)
try:
    from workflow_composer.agents.chat_integration import (
        AgentChatHandler,
        get_chat_handler,
        AGENTS_AVAILABLE as ENHANCED_AGENTS_AVAILABLE,
    )
    from workflow_composer.agents.self_healing import SelfHealer, get_self_healer
    from workflow_composer.agents.memory import AgentMemory
    ENHANCED_AGENTS = ENHANCED_AGENTS_AVAILABLE
    if ENHANCED_AGENTS and USE_LOCAL_LLM:
        print("✓ Enhanced agent system available (ReAct, Memory, Self-Healing)")
except ImportError as e:
    ENHANCED_AGENTS = False
    AgentChatHandler = None
    get_chat_handler = None
    SelfHealer = None
    get_self_healer = None
    AgentMemory = None
    print(f"Note: Enhanced agents not available: {e}")

# Import autonomous agent panel (Phase 4)
try:
    from workflow_composer.web.components.autonomous_panel import (
        AUTONOMOUS_AVAILABLE,
        create_autonomous_panel,
        setup_autonomous_events,
        create_health_widget,
        check_health_sync,
        get_agent,
    )
    if AUTONOMOUS_AVAILABLE:
        print("✓ Autonomous agent panel available")
except ImportError as e:
    AUTONOMOUS_AVAILABLE = False
    create_autonomous_panel = None
    setup_autonomous_events = None
    create_health_widget = None
    check_health_sync = None
    get_agent = None
    print(f"Note: Autonomous panel not available: {e}")


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
                        
                        # SLURM job finished - but check Nextflow log for actual success/failure
                        job.finished_at = datetime.now()
                        
                        # Parse log first to detect Nextflow errors
                        if job.log_file and Path(job.log_file).exists():
                            self._parse_nextflow_log(job)
                        
                        # Only set completed if log parsing didn't find errors
                        if job.status not in (JobStatus.FAILED, JobStatus.CANCELLED):
                            if "COMPLETED" in final_status:
                                job.status = JobStatus.COMPLETED
                                job.progress = 100.0
                            elif "FAILED" in final_status or "CANCELLED" in final_status:
                                job.status = JobStatus.FAILED if "FAILED" in final_status else JobStatus.CANCELLED
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
            
            # Check for completion/failure patterns
            error_patterns = [
                r"Error executing process",
                r"Pipeline failed",
                r"ERROR\s*[~\-]",
                r"No such file or directory",
                r"Command error:",
                r"Execution halted",
                r"FATAL:",
                r"Exception:",
            ]
            
            success_patterns = [
                "Pipeline completed",
                "Workflow completed",
                "Succeeded   :",
                "Workflow finished successfully",
            ]
            
            # Check for errors first
            is_error = any(re.search(pattern, content, re.IGNORECASE) for pattern in error_patterns)
            is_success = any(pattern in content for pattern in success_patterns)
            
            if is_success and not is_error:
                job.status = JobStatus.COMPLETED
                job.progress = 100.0
                job.finished_at = datetime.now()
            elif is_error:
                job.status = JobStatus.FAILED
                job.finished_at = datetime.now()
                # Extract error message
                for pattern in error_patterns:
                    error_match = re.search(f'{pattern}.*?(?=\\n\\n|\\Z)', content, re.DOTALL | re.IGNORECASE)
                    if error_match:
                        job.error_message = error_match.group()[:500]
                        break
                    
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

# Consolidated status icons (used across all display functions)
STATUS_ICONS = {
    JobStatus.PENDING: "🟡",
    JobStatus.RUNNING: "🔵",
    JobStatus.COMPLETED: "✅",
    JobStatus.FAILED: "❌",
    JobStatus.CANCELLED: "⚪",
    "pending": "🟡",
    "running": "🔵",
    "completed": "✅",
    "failed": "❌",
    "cancelled": "⚪",
    "ready": "✅",
    "starting": "🟡",
    "unavailable": "⚫",
    "error": "❌",
}

# Minimal CSS for styling
CUSTOM_CSS = """
.main-header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #059669 0%, #3b82f6 100%);
    border-radius: 12px;
    margin-bottom: 20px;
}
.main-header h1 { color: white !important; margin: 0; font-size: 2.2em; }
.main-header p { color: rgba(255,255,255,0.9) !important; margin: 5px 0 0 0; }
footer { display: none !important; }
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
        
        # Conversation context for multi-turn dialogue
        if ConversationContext:
            self.conversation_context = ConversationContext()
        else:
            self.conversation_context = None
        
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


# Global app state removed - using gr.State instead
# app_state = AppState()


# ============================================================================
# Core Functions
# ============================================================================

def check_available_providers() -> Dict[str, bool]:
    """Check which LLM providers are available based on API keys and configuration."""
    if not COMPOSER_AVAILABLE:
        return {"openai": False, "vllm": False, "ollama": False, "anthropic": False, "lightning": False}
    
    try:
        import os
        # Quick check based on API keys instead of making actual API calls
        return {
            "lightning": bool(os.environ.get("LIGHTNING_API_KEY")),
            "openai": bool(os.environ.get("OPENAI_API_KEY")),
            "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "ollama": False,  # Would need to check if ollama service is running
            "vllm": False,  # Would need to check if vLLM is configured
        }
    except:
        return {"openai": False, "vllm": False, "ollama": False, "anthropic": False, "lightning": False}


def get_provider_choices() -> List[str]:
    """Get list of available provider choices for dropdown."""
    available = check_available_providers()
    choices = []
    
    provider_labels = {
        "lightning": "⚡ Lightning.ai (30M FREE tokens!)",
        "openai": "🟢 OpenAI (GPT-4o)",
        "vllm": "🟣 vLLM (Local GPU)",
        "ollama": "🟠 Ollama (Local)",
        "anthropic": "🔵 Anthropic (Claude)",
    }
    
    # Always show Lightning.ai first as it's the best value
    import os
    lightning_available = bool(os.environ.get("LIGHTNING_API_KEY"))
    if lightning_available:
        choices.append(provider_labels["lightning"])
    else:
        choices.append("⚡ Lightning.ai (set LIGHTNING_API_KEY)")
    
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
        "lightning": "lightning",
        "openai": "openai",
        "vllm": "vllm",
        "ollama": "ollama",
        "anthropic": "anthropic",
    }
    
    choice_lower = choice.lower()
    for key in mapping:
        if key in choice_lower:
            return key
    return "lightning"  # Default to Lightning.ai now


def chat_with_composer(
    message: str,
    history: List[Dict[str, str]],
    provider: str,
    app_state: AppState,
) -> Generator[Tuple[List[Dict[str, str]], str], None, None]:
    """
    Chat with the AI workflow composer.
    Streams responses for better UX.
    Uses Gradio 6.0 message format: [{"role": "user/assistant", "content": "..."}]
    
    Now includes agent tools for:
    - Data scanning and discovery
    - Job submission and monitoring
    - Error diagnosis
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
    
    # Get conversation context
    ctx = app_state.conversation_context
    
    # ===== CHECK FOR CLARIFICATION NEEDED =====
    if ctx:
        clarification = ctx.needs_clarification(message)
        if clarification:
            history.append({"role": "assistant", "content": f"🤔 {clarification}"})
            yield history, ""
            return
    
    # ===== CHECK FOR AGENT TOOL INVOCATION =====
    # Tools: scan_data, search_databases, submit_job, get_logs, cancel_job, etc.
    # Use AgentBridge with LLM routing when in GPU mode, otherwise regex-based
    if AGENT_TOOLS_AVAILABLE:
        tool_result = None
        
        # Try LLM-based routing if available
        if AgentBridge and (USE_LOCAL_LLM or os.environ.get("LIGHTNING_API_KEY")):
            try:
                bridge = get_agent_bridge(app_state)
                # Configure bridge for local or cloud
                if USE_LOCAL_LLM:
                    bridge.router.local_url = VLLM_URL
                    bridge.router.use_local = True
                
                # Get context dict for router
                ctx_dict = None
                if ctx:
                    ctx_dict = {
                        "data_loaded": ctx.last_scan_path is not None,
                        "sample_count": len(ctx.last_scan_samples) if ctx.last_scan_samples else 0,
                        "data_path": ctx.last_scan_path,
                        "last_workflow": ctx.last_workflow_name,
                    }
                
                result = bridge.process_message_sync(message, ctx_dict)
                if result:
                    if result.get("requires_generation"):
                        # Let it fall through to workflow generation
                        pass
                    elif result.get("tool_result"):
                        tool_result = result["tool_result"]
                    elif result.get("response"):
                        # Direct LLM response
                        history.append({"role": "assistant", "content": result["response"]})
                        yield history, ""
                        return
            except Exception as e:
                logger.debug(f"AgentBridge failed, falling back to regex: {e}")
        
        # Fallback to regex-based tool detection
        if tool_result is None and process_tool_request:
            tool_result = process_tool_request(message, app_state)
        
        if tool_result:
            # Tool was executed - update conversation context
            if ctx and tool_result.success:
                if tool_result.tool_name == "scan_data" and tool_result.data:
                    ctx.set_scanned_data(
                        path=tool_result.data.get("path", ""),
                        samples=tool_result.data.get("samples", []),
                    )
                elif tool_result.tool_name == "search_databases" and tool_result.data:
                    ctx.set_search_results(
                        query=tool_result.data.get("query", ""),
                        results=tool_result.data.get("results", []),
                    )
            
            # Display result with contextual follow-up suggestions
            response_msg = tool_result.message
            
            # Add contextual follow-up suggestion based on tool
            if tool_result.tool_name == "scan_data" and tool_result.success:
                # After scanning, suggest next steps based on data
                response_msg += "\n\n💡 **What's next?** You can now:\n"
                response_msg += "- Say \"create a workflow for this data\"\n"
                response_msg += "- Ask \"check reference for human\" to verify references\n"
                response_msg += "- Say \"compare samples\" to set up differential analysis"
            elif tool_result.tool_name == "search_databases" and tool_result.success:
                response_msg += "\n\n💡 **Tip:** Say \"download ENCSR...\" to add a dataset to your manifest."
            elif tool_result.tool_name == "check_references" and tool_result.success:
                if "Not found" in tool_result.message:
                    response_msg += "\n\n💡 **Tip:** Say \"download human reference\" to fetch the genome."
            
            history.append({"role": "assistant", "content": response_msg})
            yield history, ""
            return
    
    # Check if this is a workflow generation request
    # Be more specific - don't trigger on just "process" or "analysis" alone
    message_lower = message.lower()
    
    # Explicit workflow generation keywords
    explicit_generation = any(kw in message_lower for kw in [
        "generate workflow", "create workflow", "build workflow", "make workflow",
        "generate pipeline", "create pipeline", "build pipeline", "make pipeline",
        "create a workflow", "build a workflow", "generate a pipeline",
        "create workflow for this", "generate workflow for this", "build workflow for this",
    ])
    
    # Context-aware: user says "create a workflow for this data" after scanning
    if ctx and ctx.manifest_sample_count > 0:
        if any(kw in message_lower for kw in ["for this data", "for these samples", "for the data", "for this"]):
            explicit_generation = True
    
    # Analysis type mentions (only trigger if not asking about scanning/data)
    has_analysis_type = any(kw in message_lower for kw in [
        "rna-seq", "rnaseq", "chip-seq", "chipseq", "atac-seq", "atacseq",
        "methylation", "bisulfite", "wgbs", "rrbs", "dna-seq", "variant calling",
        "differential expression", "peak calling", "metagenomics", "scrna",
    ])
    
    # Check for data-first intent (scanning, finding data) - should NOT trigger workflow
    is_data_request = any(kw in message_lower for kw in [
        "scan", "find data", "find datasets", "what data", "which data",
        "check for data", "discover data", "list data", "available data",
    ])
    
    # Determine if this is truly a workflow generation request
    is_generation_request = (
        explicit_generation or 
        (has_analysis_type and not is_data_request and any(kw in message_lower for kw in ["workflow", "pipeline", "create", "build", "generate"]))
    )
    
    try:
        if is_generation_request and app_state.composer:
            # Generate workflow
            response_parts = []
            
            # First, parse the intent
            history.append({"role": "assistant", "content": "🔍 Parsing your request..."})
            yield history, ""
            
            intent = app_state.composer.parse_intent(message)
            
            # Generate a descriptive workflow name
            workflow_name = generate_workflow_name(intent, message)
            
            intent_info = f"""
📋 **Detected Analysis:**
- Type: `{intent.analysis_type.value}`
- Organism: `{intent.organism or 'Not specified'}`
- Genome: `{intent.genome_build or 'Auto-detect'}`
- Confidence: `{intent.confidence:.0%}`

"""
            response_parts.append(intent_info)
            history[-1] = {"role": "assistant", "content": "".join(response_parts) + "🔧 Running pre-flight validation..."}
            yield history, ""
            
            # Check readiness with enhanced pre-flight validation
            readiness = app_state.composer.check_readiness(message)
            
            # Display detailed validation results if available
            validation = readiness.get("validation", {})
            resources = readiness.get("resources", {})
            
            if validation:
                # Enhanced pre-flight validation available
                validation_info = """
🔍 **Pre-flight Validation:**
"""
                # Tools status
                tools_valid = validation.get("tools_found", [])
                tools_missing = validation.get("tools_missing", [])
                if tools_valid:
                    validation_info += f"- ✅ Tools ready: {', '.join(f'`{t}`' for t in tools_valid[:5])}"
                    if len(tools_valid) > 5:
                        validation_info += f" +{len(tools_valid)-5} more"
                    validation_info += "\n"
                if tools_missing:
                    validation_info += f"- ⚠️ Tools missing: {', '.join(f'`{t}`' for t in tools_missing[:5])}\n"
                
                # Containers status  
                containers_valid = validation.get("containers_available", [])
                containers_missing = validation.get("containers_missing", [])
                if containers_valid:
                    validation_info += f"- ✅ Containers ready: {', '.join(f'`{c}`' for c in containers_valid[:3])}\n"
                if containers_missing:
                    validation_info += f"- ⚠️ Containers missing: {', '.join(f'`{c}`' for c in containers_missing[:3])}\n"
                
                # Modules status
                modules_valid = validation.get("modules_available", [])
                modules_missing = validation.get("modules_missing", [])
                if modules_valid:
                    validation_info += f"- ✅ Modules ready: `{len(modules_valid)}` available\n"
                if modules_missing:
                    validation_info += f"- ⚠️ Modules to create: `{len(modules_missing)}`\n"
                
                # Resource estimates
                if resources:
                    validation_info += f"""
📊 **Resource Estimates:**
- Memory: `{resources.get('estimated_memory_gb', 'N/A')} GB`
- Time: `{resources.get('estimated_hours', 'N/A'):.1f} hours`
- Est. Cost: `${resources.get('estimated_cost_usd', 0):.2f}`

"""
                response_parts.append(validation_info)
            
            if readiness.get("ready"):
                tools_found = validation.get("valid", readiness.get("tools_found", 0))
                
                readiness_info = f"""✅ **Ready to generate!**

"""
                response_parts.append(readiness_info)
                history[-1] = {"role": "assistant", "content": "".join(response_parts) + "⚙️ Generating workflow..."}
                yield history, ""
                
                # Generate the workflow with descriptive name
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                workflow_id = f"{workflow_name}_{timestamp}"
                output_dir = GENERATED_DIR / workflow_id
                
                workflow = app_state.composer.generate(message, output_dir=str(output_dir))
                
                # Track last generated workflow for quick run
                app_state.last_generated_workflow = str(output_dir)
                
                # Update conversation context
                if ctx:
                    ctx.set_generated_workflow(path=str(output_dir), name=workflow_id)
                
                # Format response - extract clean data from workflow
                tools_list = []
                if hasattr(workflow, 'tools_used') and workflow.tools_used:
                    for t in workflow.tools_used:
                        if hasattr(t, 'name'):
                            tools_list.append(t.name)
                        elif isinstance(t, str):
                            tools_list.append(t)
                        else:
                            tools_list.append(str(t))
                
                modules_list = []
                if hasattr(workflow, 'modules_used') and workflow.modules_used:
                    for m in workflow.modules_used:
                        if hasattr(m, 'name'):
                            modules_list.append(m.name)
                        elif isinstance(m, str):
                            modules_list.append(m)
                        else:
                            modules_list.append(str(m))
                
                tools_str = ', '.join(f'`{t}`' for t in tools_list) if tools_list else '_Auto-selected based on analysis type_'
                modules_str = ', '.join(f'`{m}`' for m in modules_list) if modules_list else '_Mapped from tools_'
                
                workflow_info = f"""🎉 **Workflow Generated Successfully!**

| Property | Value |
|----------|-------|
| **Name** | `{workflow.name if hasattr(workflow, 'name') else workflow_id}` |
| **Location** | `{output_dir}` |

**🔧 Tools:** {tools_str}

**📦 Modules:** {modules_str}

---

📥 **Next Steps:**
1. Go to the **🚀 Execute** tab
2. Select your workflow from the dropdown
3. Configure parameters and submit to SLURM
4. Use the **📥 Download** button to get workflow files
"""
                response_parts.append(workflow_info)
                
            else:
                issues = readiness.get("issues", ["Unknown issue"])
                warnings = readiness.get("warnings", [])
                auto_fixable = readiness.get("auto_fixable", False)
                
                not_ready_msg = f"""⚠️ **Pre-flight Check Failed:**

**Critical Issues:**
{chr(10).join(f'- ❌ {issue}' for issue in issues)}
"""
                if warnings:
                    not_ready_msg += f"""
**Warnings:**
{chr(10).join(f'- ⚠️ {w}' for w in warnings[:5])}
"""
                
                if auto_fixable:
                    not_ready_msg += """
✨ **Auto-Fix Available!** Missing modules can be auto-generated.
Attempting auto-fix...
"""
                    response_parts.append(not_ready_msg)
                    history[-1] = {"role": "assistant", "content": "".join(response_parts)}
                    yield history, ""
                    
                    # Attempt auto-fix
                    try:
                        fix_result = app_state.composer.validate_and_prepare(message, auto_fix=True)
                        
                        if fix_result["status"] == "ready":
                            fixes = fix_result.get("fixes_applied", [])
                            success_fixes = [f for f in fixes if f.get("status") == "success"]
                            
                            response_parts.append(f"""
✅ **Auto-Fix Successful!**
- Applied `{len(success_fixes)}` fixes
- {chr(10).join(f"  - Created module: `{f['name']}`" for f in success_fixes)}

""")
                            history[-1] = {"role": "assistant", "content": "".join(response_parts) + "⚙️ Generating workflow..."}
                            yield history, ""
                            
                            # Now generate the workflow
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            workflow_id = f"{workflow_name}_{timestamp}"
                            output_dir = GENERATED_DIR / workflow_id
                            
                            workflow = app_state.composer.generate(message, output_dir=str(output_dir))
                            app_state.last_generated_workflow = str(output_dir)
                            
                            workflow_info = f"""🎉 **Workflow Generated!**

| Property | Value |
|----------|-------|
| **Name** | `{workflow.name if hasattr(workflow, 'name') else workflow_id}` |
| **Location** | `{output_dir}` |

📥 **Next Steps:**
1. Go to the **🚀 Execute** tab to run your workflow
2. Use the **📥 Download** button to get workflow files
"""
                            response_parts.append(workflow_info)
                        else:
                            response_parts.append(f"""
⚠️ **Partial Fix Applied:**
- Status: `{fix_result['status']}`
- Message: {fix_result.get('message', 'Some issues remain')}

Please address the remaining issues manually.
""")
                    except Exception as e:
                        response_parts.append(f"\n❌ Auto-fix failed: {str(e)}")
                else:
                    not_ready_msg += """
Please provide more details or ensure required containers are available.
"""
                    response_parts.append(not_ready_msg)
            
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


# =============================================================================
# Enhanced Chat with Agent System (ReAct, Memory, Self-Healing)
# =============================================================================

# Global handler instance
_enhanced_handler: Optional[AgentChatHandler] = None


def get_enhanced_handler(app_state: AppState = None) -> Optional[AgentChatHandler]:
    """Get or create the enhanced agent chat handler."""
    global _enhanced_handler
    
    if not ENHANCED_AGENTS or not USE_LOCAL_LLM:
        return None
    
    if _enhanced_handler is None:
        try:
            _enhanced_handler = get_chat_handler(
                app_state=app_state,
                vllm_url=VLLM_URL,
            )
            logger.info("Enhanced agent handler initialized")
        except Exception as e:
            logger.warning(f"Could not initialize enhanced handler: {e}")
            return None
    elif app_state:
        _enhanced_handler.app_state = app_state
    
    return _enhanced_handler


def enhanced_chat_with_composer(
    message: str,
    history: List[Dict[str, str]],
    provider: str,
    app_state: AppState,
    use_enhanced: bool = True,
    enable_memory: bool = True,
    enable_react: bool = True,
    enable_self_healing: bool = True,
) -> Generator[Tuple[List[Dict[str, str]], str], None, None]:
    """
    Enhanced chat function using the full agent system.
    
    Features:
    - ReAct reasoning for complex multi-step queries
    - Memory for learning from past interactions
    - Self-healing for auto-fixing failed jobs
    - Falls back to basic chat if enhanced not available
    """
    # Check if we should use enhanced mode
    handler = get_enhanced_handler(app_state) if use_enhanced else None
    
    if not handler:
        # Fall back to basic chat
        yield from chat_with_composer(message, history, provider, app_state)
        return
    
    if not message.strip():
        yield history, ""
        return
    
    # Configure handler features
    handler.enable_memory = enable_memory
    handler.enable_react = enable_react
    handler.enable_self_healing = enable_self_healing
    
    # Add user message
    history = history + [{"role": "user", "content": message}]
    yield history, ""
    
    # Get context
    ctx = app_state.conversation_context
    context = {}
    if ctx:
        context = {
            "data_loaded": ctx.last_scan_path is not None,
            "sample_count": len(ctx.last_scan_samples) if ctx.last_scan_samples else 0,
            "data_path": ctx.last_scan_path,
            "last_workflow": ctx.last_workflow_name,
        }
    
    # Stream response from enhanced handler
    history = history + [{"role": "assistant", "content": ""}]
    response_text = ""
    
    try:
        for chunk in handler.chat(message, history[:-2], context):
            response_text += chunk
            history[-1]["content"] = response_text
            yield history, ""
    except Exception as e:
        logger.error(f"Enhanced chat error: {e}")
        # Fall back to basic
        history = history[:-2]  # Remove empty assistant message
        yield from chat_with_composer(message, history, provider, app_state)


def search_tools(query: str, container_filter: str, app_state: AppState) -> str:
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


def get_modules_by_category(app_state: AppState) -> str:
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
    return [[example] for example in ANALYSIS_EXAMPLES.values()]


# Mapping dictionaries for workflow naming (moved outside function for reuse)
_TYPE_MAP = {
    'rna_seq': 'rnaseq', 'chip_seq': 'chipseq', 'atac_seq': 'atacseq',
    'dna_seq': 'dnaseq', 'variant_calling': 'variants', 'single_cell_rna': 'scrna',
    'metagenomics': 'metagen', 'methylation': 'methylation', 'bisulfite_seq': 'bisulfite',
    'bisulfite_seq_methylation': 'methylation', 'long_read': 'longread',
}
_ORG_MAP = {
    'homo sapiens': 'human', 'human': 'human', 'mus musculus': 'mouse', 'mouse': 'mouse',
    'drosophila': 'fly', 'zebrafish': 'zebrafish', 'yeast': 'yeast', 'rat': 'rat',
}
_TOOL_KEYWORDS = [
    'star', 'hisat2', 'salmon', 'kallisto', 'bowtie2', 'bwa', 'minimap2',
    'deseq2', 'macs2', 'gatk', 'bismark', 'seurat', 'scanpy', 'kraken',
]


def generate_workflow_name(intent, message: str) -> str:
    """Generate a descriptive workflow name based on the analysis intent."""
    parts = []
    
    # 1. Analysis type
    analysis_type = getattr(intent, 'analysis_type', None)
    if analysis_type:
        at = analysis_type.value if hasattr(analysis_type, 'value') else str(analysis_type)
        parts.append(_TYPE_MAP.get(at, at.replace('_', '')))
    
    # 2. Organism
    organism = getattr(intent, 'organism', None)
    if organism:
        parts.append(_ORG_MAP.get(organism.lower(), organism.split()[0][:8]))
    
    # 3. Key tools from message
    msg_lower = message.lower()
    found_tools = [t for t in _TOOL_KEYWORDS if t in msg_lower][:2]
    parts.extend(found_tools)
    
    # Clean up and return
    unique_parts = list(dict.fromkeys(parts))[:4]  # Remove dupes, limit to 4
    workflow_name = '_'.join(unique_parts) if unique_parts else 'workflow'
    return re.sub(r'[^a-zA-Z0-9_]', '', workflow_name)
    
    return workflow_name


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


def get_workflow_preview(app_state: AppState):
    """Get preview of the last generated workflow.
    
    Returns:
        Tuple of (accordion_update, markdown_content, code_content)
    """
    try:
        if app_state.last_generated_workflow:
            workflow_path = Path(app_state.last_generated_workflow)
        else:
            # Find latest workflow
            if not GENERATED_DIR.exists():
                return (
                    gr.update(visible=True, open=True),
                    "*No workflows directory found*",
                    ""
                )
            workflows = [w for w in GENERATED_DIR.iterdir() if w.is_dir()]
            if not workflows:
                return (
                    gr.update(visible=True, open=True),
                    "*No workflows generated yet*",
                    ""
                )
            workflows = sorted(workflows, key=lambda x: x.stat().st_mtime, reverse=True)
            workflow_path = workflows[0]
        
        if not workflow_path.exists():
            return (
                gr.update(visible=True, open=True),
                "*Workflow not found*",
                ""
            )
        
        # Build preview content
        preview = f"""### 📁 {workflow_path.name}

**Location:** `{workflow_path}`

**Files:**
"""
        files = [f for f in workflow_path.glob("*") if not f.name.startswith('.')]
        for f in files[:10]:
            size = f.stat().st_size
            size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
            emoji = "📄" if f.is_file() else "📁"
            preview += f"- {emoji} `{f.name}` ({size_str})\n"
        
        if len(files) > 10:
            preview += f"- ... and {len(files) - 10} more files\n"
        
        # Read main.nf if exists
        main_nf = workflow_path / "main.nf"
        code_content = ""
        if main_nf.exists():
            full_code = main_nf.read_text()
            code_content = full_code[:3000]  # First 3000 chars
            if len(full_code) > 3000:
                code_content += "\n\n// ... (truncated)"
            preview += f"\n**main.nf:** {len(full_code)} characters"
        
        return (
            gr.update(visible=True, open=True),
            preview,
            code_content
        )
        
    except Exception as e:
        return (
            gr.update(visible=True, open=True),
            f"*Error loading workflow: {e}*",
            ""
        )


def refresh_stats(app_state: AppState) -> Tuple[str, str, str, str]:
    """Refresh and return statistics."""
    stats = app_state.get_stats()
    return (
        f"🔧 {stats['tools']}",
        f"📦 {stats['modules']}",
        f"🐳 {stats['containers']}",
        f"🧬 {stats['analysis_types']}"
    )


# ============================================================================
# Ensemble Model Management Functions
# ============================================================================

def get_ensemble_status() -> str:
    """Get current status of ensemble models."""
    if not ORCHESTRATOR_AVAILABLE or not get_orchestrator:
        return """
### ⚠️ Ensemble Not Available

The model orchestrator is not installed. Using standard rule-based + LLM parsing.

To enable ensemble parsing with biomedical models:
```bash
pip install transformers torch
```
"""
    
    orchestrator = get_orchestrator()
    status = orchestrator.get_status_summary()
    
    # Format status display
    strategy_display = status.get('strategy_description', 'Unknown')
    
    output = f"""
### 🧬 Ensemble Status: {strategy_display}

| Model | Status | Details |
|-------|--------|---------|
"""
    
    for model_name, model_info in status.get('models', {}).items():
        icon = STATUS_ICONS.get(model_info['status'], '❓')
        details = []
        
        if model_info.get('endpoint'):
            details.append(f"URL: `{model_info['endpoint']}`")
        if model_info.get('load_time_ms'):
            details.append(f"Loaded in {model_info['load_time_ms']:.0f}ms")
        if model_info.get('job_id'):
            details.append(f"SLURM: `{model_info['job_id']}`")
        if model_info.get('error'):
            details.append(f"⚠️ {model_info['error'][:30]}")
        
        detail_str = ", ".join(details) if details else "-"
        output += f"| **{model_name.title()}** | {icon} {model_info['status']} | {detail_str} |\n"
    
    output += f"""

### Parsing Strategy
- **GPU Available:** {'✅ Yes' if status.get('gpu_available') else '❌ No'}
- **CPU Models Loaded:** {'✅ Yes' if status.get('cpu_models_loaded') else '⚫ Not yet'}

> Current strategy: `{status.get('strategy', 'unknown')}`
"""
    
    return output


def start_biomistral_service() -> str:
    """Start BioMistral GPU service via SLURM."""
    if not ORCHESTRATOR_AVAILABLE or not get_orchestrator:
        return "❌ Orchestrator not available"
    
    orchestrator = get_orchestrator()
    success, message = orchestrator.start_biomistral_gpu()
    
    if success:
        return f"✅ {message}\n\n⏳ GPU startup typically takes 2-5 minutes. Refresh status to check."
    else:
        return f"❌ {message}"


def stop_biomistral_service() -> str:
    """Stop BioMistral GPU service."""
    if not ORCHESTRATOR_AVAILABLE or not get_orchestrator:
        return "❌ Orchestrator not available"
    
    orchestrator = get_orchestrator()
    success, message = orchestrator.stop_biomistral_gpu()
    
    if success:
        return f"✅ {message}"
    else:
        return f"❌ {message}"


def preload_cpu_models() -> str:
    """Preload BERT models on CPU."""
    if not ORCHESTRATOR_AVAILABLE or not get_orchestrator:
        return "❌ Orchestrator not available"
    
    orchestrator = get_orchestrator()
    orchestrator.preload_cpu_models()
    
    return "🔄 Loading BiomedBERT and SciBERT in background...\n\nRefresh status in ~30 seconds to see results."


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
            return f"""✅ **Submitted to SLURM!**

Job `{job.job_id}` queued (SLURM #{job.slurm_job_id})

👆 *Check the Jobs table above for live status updates*
"""
        else:
            return f"""✅ **Started Locally!**

Job `{job.job_id}` running

👆 *Check the Jobs table above for live status updates*
"""
    except Exception as e:
        return f"❌ **Submission Failed**\n\nError: {str(e)}"


def get_job_status_display() -> str:
    """Get formatted display of all job statuses."""
    jobs = pipeline_executor.list_jobs()
    
    if not jobs:
        return "No pipeline jobs submitted yet. Submit a pipeline above to get started."
    
    output = "## 📊 Pipeline Jobs\n\n"
    output += "| Status | Job ID | Workflow | Progress | Current Process | Time |\n"
    output += "|--------|--------|----------|----------|-----------------|------|\n"
    
    for job in sorted(jobs, key=lambda j: j.started_at or datetime.min, reverse=True):
        icon = STATUS_ICONS.get(job.status, "⚪")
        status_text = job.status.value if hasattr(job.status, 'value') else str(job.status)
        
        # Calculate runtime
        if job.started_at:
            end_time = job.finished_at or datetime.now()
            duration = end_time - job.started_at
            time_str = str(duration).split('.')[0]  # Remove microseconds
        else:
            time_str = "-"
        
        progress_bar = f"{job.progress:.0f}%"
        current = job.current_process[:20] if job.current_process else "-"
        
        # Add error indicator if failed
        if job.status == JobStatus.FAILED and job.error_message:
            error_hint = "⚠️ " + job.error_message[:30] + "..."
        else:
            error_hint = ""
        
        output += f"| {icon} {status_text} | `{job.job_id[-15:]}` | {job.workflow_name[:15]} | {progress_bar} | {current} | {time_str} |{error_hint}\n"
    
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
    """Get recent log output for a job. If no job_id provided, shows latest job logs."""
    
    # If no job ID provided, find the latest job
    if not job_id or not job_id.strip():
        jobs = pipeline_executor.list_jobs()
        if not jobs:
            return "*No jobs found. Submit a workflow first.*"
        
        # Sort by most recent (completed/failed first, then running, then pending)
        priority = {JobStatus.COMPLETED: 0, JobStatus.FAILED: 1, JobStatus.RUNNING: 2, JobStatus.PENDING: 3}
        sorted_jobs = sorted(jobs, key=lambda j: (priority.get(j.status, 4), -(j.started_at.timestamp() if j.started_at else 0)))
        job = sorted_jobs[0] if sorted_jobs else None
        
        if not job:
            return "*No jobs found.*"
        job_id = job.job_id
    else:
        job = pipeline_executor.get_job_status(job_id.strip())
    
    if not job:
        return f"*Job not found: `{job_id}`*"
    
    # Check for log file
    log_file = None
    if job.log_file and Path(job.log_file).exists():
        log_file = Path(job.log_file)
    else:
        # Try to find log file in common locations
        workflow_dir = Path(job.workflow_dir) if job.workflow_dir else None
        possible_logs = [
            Path(f"logs/{job.job_id}.log"),
            Path(f"logs/{job.job_id}.out"),
            Path(f"logs/{job.job_id}.err"),
            workflow_dir / ".nextflow.log" if workflow_dir else None,
            workflow_dir / "nextflow.log" if workflow_dir else None,
        ]
        for p in possible_logs:
            if p and p.exists():
                log_file = p
                break
    
    if not log_file:
        started_str = job.started_at.strftime('%Y-%m-%d %H:%M:%S') if job.started_at else 'Not started'
        status_info = f"""## 📄 Job: `{job.job_id}`

**Status:** {STATUS_ICONS.get(job.status.value, '❓')} {job.status.value}
**Started:** {started_str}
**Workflow:** `{job.workflow_name}`

*Log file not available yet. The job may still be starting or queued.*
"""
        return status_info
    
    try:
        # Read last N lines
        with open(log_file, 'r') as f:
            lines = f.readlines()
            recent = lines[-tail_lines:] if len(lines) > tail_lines else lines
        
        status_icon = STATUS_ICONS.get(job.status.value, '❓')
        started_str = job.started_at.strftime('%Y-%m-%d %H:%M:%S') if job.started_at else 'N/A'
        output = f"""## 📄 Job: `{job.job_id}`

**Status:** {status_icon} {job.status.value} | **Started:** {started_str}
**Log File:** `{log_file}`
*Showing last {len(recent)} of {len(lines)} lines*

```
{"".join(recent)}
```"""
        
        return output
    except Exception as e:
        return f"*Error reading log: {e}*"


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


def diagnose_job_failure(job_id: str) -> str:
    """
    Diagnose a failed job using the AI-powered error diagnosis agent.
    
    Args:
        job_id: Job ID to diagnose (or empty for latest failed job)
        
    Returns:
        Formatted diagnosis report
    """
    if not DIAGNOSIS_AVAILABLE:
        return """## ⚠️ Diagnosis Unavailable

The error diagnosis agent is not installed. Install it with:

```bash
pip install -e ".[diagnosis]"
```

Or check the `src/workflow_composer/diagnosis/` package.
"""
    
    # Find the job
    if not job_id or not job_id.strip():
        # Find the latest failed job
        jobs = pipeline_executor.list_jobs()
        failed_jobs = [j for j in jobs if j.status == JobStatus.FAILED]
        
        if not failed_jobs:
            return "*No failed jobs to diagnose. All jobs are running or completed successfully.*"
        
        # Most recent failed job
        job = max(failed_jobs, key=lambda j: j.finished_at or j.started_at or datetime.min)
    else:
        job = pipeline_executor.get_job_status(job_id.strip())
    
    if not job:
        return f"*Job not found: `{job_id}`*"
    
    if job.status not in (JobStatus.FAILED, JobStatus.CANCELLED):
        return f"""## ℹ️ Job Not Failed

Job `{job.job_id}` has status **{job.status.value}**.

Diagnosis is only available for failed jobs. If the job is still running, 
wait for it to complete or check the logs directly.
"""
    
    # Run diagnosis
    try:
        agent = ErrorDiagnosisAgent()
        diagnosis = agent.diagnose_sync(job)
        
        # Store in cache and record to history
        _last_diagnosis[job.job_id] = diagnosis
        try:
            from workflow_composer.diagnosis import record_diagnosis
            record_diagnosis(diagnosis, job.job_id, job.workflow_name)
        except Exception as e:
            logger.warning(f"Failed to record diagnosis to history: {e}")
        
        # Format result
        risk_icons = {
            "safe": "🟢",
            "low": "🟡", 
            "medium": "🟠",
            "high": "🔴",
        }
        
        # Build fixes list
        fixes_md = ""
        auto_fix_available = False
        for i, fix in enumerate(diagnosis.suggested_fixes, 1):
            risk_key = fix.risk_level.value if hasattr(fix.risk_level, 'value') else str(fix.risk_level).lower()
            icon = risk_icons.get(risk_key, "⚪")
            auto_tag = " `[AUTO]`" if fix.auto_executable else ""
            fixes_md += f"{i}. {icon}{auto_tag} **{fix.description}**\n"
            if fix.command:
                fixes_md += f"   ```bash\n   {fix.command}\n   ```\n"
            if fix.auto_executable and risk_key == "safe":
                auto_fix_available = True
        
        confidence_bar = "█" * int(diagnosis.confidence * 10) + "░" * (10 - int(diagnosis.confidence * 10))
        
        output = f"""## 🔍 Error Diagnosis: `{job.job_id}`

### 📊 Classification
| Field | Value |
|-------|-------|
| **Error Type** | `{diagnosis.category.value}` |
| **Confidence** | [{confidence_bar}] {diagnosis.confidence:.0%} |
| **Failed Process** | `{diagnosis.failed_process or 'Unknown'}` |
| **Provider** | `{diagnosis.llm_provider_used or 'Pattern Matching'}` |

### 🎯 Root Cause
{diagnosis.root_cause}

### 💬 Explanation (for non-experts)
{diagnosis.user_explanation}

### 🛠️ Suggested Fixes
{fixes_md}

### 📄 Relevant Log Excerpt
```
{diagnosis.log_excerpt[:500] if diagnosis.log_excerpt else 'No log excerpt available'}
```
"""
        
        if auto_fix_available:
            output += """
### ✨ Auto-Fix Available
Safe fixes marked with `[AUTO]` can be applied automatically.
Click **Apply Safe Fixes** to run them.
"""
        
        return output
        
    except Exception as e:
        return f"""## ❌ Diagnosis Failed

An error occurred during diagnosis:

```
{str(e)}
```

Try viewing the logs directly for more information.
"""


# Store the last diagnosis for apply_safe_fixes
_last_diagnosis = {}


def apply_safe_fixes(job_id: str) -> str:
    """
    Apply safe auto-fixes for a diagnosed job.
    
    Args:
        job_id: Job ID to fix (uses last diagnosis)
        
    Returns:
        Formatted result report
    """
    global _last_diagnosis
    
    if not DIAGNOSIS_AVAILABLE:
        return "⚠️ Error diagnosis agent not available."
    
    try:
        from workflow_composer.diagnosis import AutoFixEngine, get_auto_fix_engine
        import asyncio
        
        # Get or create diagnosis
        if not job_id or not job_id.strip():
            jobs = pipeline_executor.list_jobs()
            failed_jobs = [j for j in jobs if j.status == JobStatus.FAILED]
            if not failed_jobs:
                return "*No failed jobs found.*"
            job = max(failed_jobs, key=lambda j: j.finished_at or j.started_at or datetime.min)
            job_id = job.job_id
        else:
            job = pipeline_executor.get_job_status(job_id.strip())
        
        if not job:
            return f"*Job not found: `{job_id}`*"
        
        # Run diagnosis if not cached
        if job_id not in _last_diagnosis:
            agent = ErrorDiagnosisAgent()
            diagnosis = agent.diagnose_sync(job)
            _last_diagnosis[job_id] = diagnosis
        else:
            diagnosis = _last_diagnosis[job_id]
        
        # Get auto-fix engine
        engine = get_auto_fix_engine(dry_run=False)
        
        # Build context for variable substitution
        context = {
            "workflow_dir": job.workflow_dir or "",
            "job_id": job.job_id,
            "slurm_job_id": job.slurm_job_id or "",
        }
        
        # Get executable fixes
        executable_fixes = engine.get_executable_fixes(diagnosis)
        
        if not executable_fixes:
            return """## ⚠️ No Auto-Executable Fixes

The suggested fixes require manual intervention. Please review the diagnosis 
and apply the recommended fixes manually.
"""
        
        # Execute safe fixes
        async def run_fixes():
            return await engine.execute_all_safe(diagnosis, context)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(run_fixes())
        loop.close()
        
        # Format results
        output = f"""## 🔧 Auto-Fix Results for `{job_id[-15:]}`

"""
        
        success_count = sum(1 for r in results if r.success)
        fail_count = sum(1 for r in results if r.status.value == 'failed')
        
        if success_count > 0:
            output += f"✅ **{success_count} fix(es) applied successfully!**\n\n"
        if fail_count > 0:
            output += f"⚠️ **{fail_count} fix(es) failed.**\n\n"
        
        output += "### Execution Details\n\n"
        
        for i, result in enumerate(results, 1):
            status_icon = "✅" if result.success else "❌" if result.status.value == 'failed' else "⏭️"
            output += f"{i}. {status_icon} **{result.fix.description}**\n"
            output += f"   - Status: {result.status.value}\n"
            if result.message:
                output += f"   - {result.message}\n"
            if result.output and result.success:
                output += f"   ```\n   {result.output[:200]}\n   ```\n"
            if result.error:
                output += f"   - Error: {result.error[:100]}\n"
        
        if success_count > 0:
            output += """
---
💡 **Next Step:** Try re-running the pipeline with `-resume` to continue from where it failed.
"""
        
        return output
        
    except Exception as e:
        return f"""## ❌ Auto-Fix Failed

Error: {str(e)}

Please apply fixes manually or check the logs for more details.
"""


def create_github_issue(job_id: str) -> str:
    """
    Create a GitHub issue from a job diagnosis.
    
    Args:
        job_id: Job ID (uses last diagnosis)
        
    Returns:
        Formatted result
    """
    global _last_diagnosis
    
    if not DIAGNOSIS_AVAILABLE:
        return "⚠️ Error diagnosis agent not available."
    
    try:
        from workflow_composer.diagnosis import get_github_copilot_agent
        
        # Get or create diagnosis
        if not job_id or not job_id.strip():
            jobs = pipeline_executor.list_jobs()
            failed_jobs = [j for j in jobs if j.status == JobStatus.FAILED]
            if not failed_jobs:
                return "*No failed jobs found.*"
            job = max(failed_jobs, key=lambda j: j.finished_at or j.started_at or datetime.min)
            job_id = job.job_id
        else:
            job = pipeline_executor.get_job_status(job_id.strip())
        
        if not job:
            return f"*Job not found: `{job_id}`*"
        
        # Run diagnosis if not cached
        if job_id not in _last_diagnosis:
            agent = ErrorDiagnosisAgent()
            diagnosis = agent.diagnose_sync(job)
            _last_diagnosis[job_id] = diagnosis
        else:
            diagnosis = _last_diagnosis[job_id]
        
        # Get GitHub agent
        github_agent = get_github_copilot_agent()
        
        if not github_agent:
            return """## ⚠️ GitHub Not Configured

GitHub token is not set. Please ensure `GITHUB_TOKEN` is configured in `.secrets/github_token`.
"""
        
        # Create the issue
        result = github_agent.create_issue(diagnosis)
        
        if result.success:
            return f"""## ✅ GitHub Issue Created!

**Issue:** #{result.issue_number}
**URL:** [{result.issue_url}]({result.issue_url})

The issue includes:
- Error classification and root cause
- Log excerpt
- Suggested fixes
- Context information

### 🤖 Next Steps
1. Open the issue in GitHub
2. Assign GitHub Copilot to fix it (`@copilot`)
3. Or manually implement the suggested fixes
"""
        else:
            return f"""## ❌ Failed to Create Issue

Error: {result.message}

Please check your GitHub token has the required permissions (`repo` scope).
"""
        
    except Exception as e:
        return f"""## ❌ Error

{str(e)}

Please check your GitHub configuration.
"""


def get_progress_details(job_id: str) -> str:
    """Get detailed progress for a specific job."""
    job = pipeline_executor.get_job_status(job_id)
    
    if not job:
        return "Select a job to view progress details."
    
    icon = STATUS_ICONS.get(job.status, "⚪")
    
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
    """Refresh all monitoring displays and update job statuses."""
    # Re-parse log files to update job status
    for job in pipeline_executor.list_jobs():
        if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
            pipeline_executor._parse_nextflow_log(job)
    
    return get_job_status_display(), get_slurm_queue_display()


# ============================================================================
# Results Visualization Functions
# ============================================================================

def get_completed_job_results_dirs() -> List[str]:
    """Get list of result directories from completed jobs."""
    result_dirs = []
    
    # Check completed jobs from executor
    for job in pipeline_executor.list_jobs():
        if job.status == JobStatus.COMPLETED and job.workflow_dir:
            results_path = Path(job.workflow_dir) / "results"
            if results_path.exists():
                result_dirs.append(str(results_path))
    
    # Also check generated workflows for results
    if GENERATED_DIR.exists():
        for workflow_dir in GENERATED_DIR.iterdir():
            if workflow_dir.is_dir():
                results_path = workflow_dir / "results"
                if results_path.exists() and str(results_path) not in result_dirs:
                    result_dirs.append(str(results_path))
    
    # Check standard data/results path
    data_results = BASE_DIR / "data" / "results"
    if data_results.exists() and str(data_results) not in result_dirs:
        result_dirs.insert(0, str(data_results))
    
    return result_dirs[:20]  # Limit to 20


def scan_results_directory(results_dir: str, pipeline_type: str = "") -> Tuple[str, str, Any]:
    """
    Scan a results directory and return summary.
    
    Returns:
        Tuple of (summary_markdown, file_tree_html, summary_data)
    """
    if not RESULTS_AVAILABLE:
        return "⚠️ Results module not available.", "", None
    
    if not results_dir:
        return "*Select a results directory to scan*", "", None
    
    results_path = Path(results_dir)
    if not results_path.exists():
        return f"❌ Directory not found: `{results_dir}`", "", None
    
    try:
        collector = ResultCollector(pipeline_type=pipeline_type if pipeline_type else None)
        summary = collector.scan(results_path)
        
        # Format summary as markdown
        md = f"""## 📊 Results Summary

**Directory:** `{results_dir}`

| Metric | Value |
|--------|-------|
| **Total Files** | {summary.total_files:,} |
| **Total Size** | {summary.size_human} |
| **Downloadable** | {summary.downloadable_size_human} |

### 📁 Files by Category
"""
        
        category_icons = {
            'qc_reports': '📋',
            'visualizations': '📊',
            'data_files': '📄',
            'alignments': '🧬',
            'logs': '📝',
            'other': '📦',
        }
        
        # Use the category summary from the model
        category_summary = summary.get_category_summary()
        for cat_name, count in category_summary.items():
            if count > 0:
                icon = category_icons.get(cat_name, '📦')
                md += f"- {icon} **{cat_name.replace('_', ' ').title()}**: {count} files\n"
        
        # Show MultiQC if available
        if summary.has_multiqc:
            md += f"\n### ✨ Key Files\n"
            md += f"- 📊 **MultiQC Report**: `{summary.multiqc_report.name}`\n"
        
        if summary.fastqc_reports:
            md += f"- 📋 **FastQC Reports**: {len(summary.fastqc_reports)} files\n"
        
        # Build file tree HTML
        file_tree = _build_file_tree_html(summary.file_tree, summary.all_files)
        
        return md, file_tree, summary
        
    except Exception as e:
        import traceback
        logger.error(f"Error scanning {results_dir}: {traceback.format_exc()}")
        return f"❌ Error scanning directory: {str(e)}", "", None


def _build_file_tree_html(tree_node, files: list, depth: int = 0) -> str:
    """Build HTML representation of file tree."""
    if tree_node is None:
        return ""
    
    indent = "  " * depth
    html = ""
    
    if tree_node.is_dir:
        icon = "📁" if depth > 0 else "📂"
        html += f"{indent}<details open><summary>{icon} <b>{tree_node.name}</b></summary>\n"
        
        # Sort children: directories first, then files
        sorted_children = sorted(
            tree_node.children,
            key=lambda x: (not x.is_dir, x.name.lower())
        )
        
        for child in sorted_children[:50]:  # Limit to 50 items per level
            html += _build_file_tree_html(child, files, depth + 1)
        
        if len(tree_node.children) > 50:
            html += f"{indent}  <div>... and {len(tree_node.children) - 50} more items</div>\n"
        
        html += f"{indent}</details>\n"
    else:
        # Find file info for size
        size_str = ""
        for f in files:
            if f.name == tree_node.name:
                size_str = f" ({f.size_human})"
                break
        
        type_icons = {
            'html': '🌐', 'png': '🖼️', 'jpg': '🖼️', 'pdf': '📕',
            'csv': '📊', 'tsv': '📊', 'txt': '📄', 'log': '📝',
            'bam': '🧬', 'vcf': '🧬', 'h5ad': '🔬', 'gz': '📦',
        }
        
        ext = tree_node.name.split('.')[-1].lower() if '.' in tree_node.name else ''
        icon = type_icons.get(ext, '📄')
        
        html += f'{indent}<div style="margin-left: 20px;">{icon} {tree_node.name}{size_str}</div>\n'
    
    return html


def view_result_file(results_dir: str, file_path: str) -> Tuple[str, str, Any]:
    """
    View a specific result file.
    
    Returns:
        Tuple of (content_type, content_markdown, content_component)
    """
    if not RESULTS_AVAILABLE:
        return "text", "⚠️ Results module not available.", None
    
    if not file_path:
        return "text", "*Select a file to view*", None
    
    try:
        from workflow_composer.results.result_types import ResultFile, FileType, ResultCategory
        from datetime import datetime
        
        viewer = ResultViewer()
        
        # Handle relative paths
        if not os.path.isabs(file_path):
            full_path = Path(results_dir) / file_path
        else:
            full_path = Path(file_path)
        
        if not full_path.exists():
            return "text", f"❌ File not found: `{file_path}`", None
        
        # Create a ResultFile object for the viewer
        from workflow_composer.results.detector import detect_file_type
        file_type, category = detect_file_type(full_path)
        stat = full_path.stat()
        
        result_file = ResultFile(
            path=full_path,
            name=full_path.name,
            relative_path=file_path,
            size=stat.st_size,
            file_type=file_type,
            category=category,
            modified=datetime.fromtimestamp(stat.st_mtime),
        )
        
        # Render the file
        viewer_content = viewer.render(result_file)
        
        content_type = viewer_content.content_type
        content = viewer_content.content
        
        # Handle errors
        if viewer_content.is_error:
            return "text", f"❌ {viewer_content.error}", None
        
        # Format based on type
        if content_type == 'html':
            return "html", content, None
        elif content_type == 'image':
            return "image", f"📷 **Image:** `{full_path.name}`", str(content)
        elif content_type == 'table':
            # Convert DataFrame to markdown
            try:
                md_table = content.head(50).to_markdown() if hasattr(content, 'to_markdown') else str(content)
                return "table", md_table, None
            except Exception:
                return "text", str(content), None
        elif content_type == 'pdf':
            return "text", f"📕 **PDF Document:** `{full_path.name}`\n\n*Download to view.*", None
        elif content_type in ('markdown', 'json', 'text'):
            return "text", f"```\n{content}\n```" if content_type != 'markdown' else content, None
        else:
            # Default text rendering
            if isinstance(content, str):
                return "text", f"```\n{content[:5000]}\n```", None
            else:
                return "text", f"```\n{str(content)[:5000]}\n```", None
            
    except Exception as e:
        import traceback
        logger.error(f"Error viewing file: {traceback.format_exc()}")
        return "text", f"❌ Error viewing file: {str(e)}", None


def create_results_archive(results_dir: str, categories: List[str] = None) -> Optional[str]:
    """
    Create a downloadable archive of results.
    
    Returns:
        Path to the created ZIP file, or None on error
    """
    if not RESULTS_AVAILABLE:
        return None
    
    if not results_dir:
        return None
    
    try:
        archiver = ResultArchiver()
        zip_path = archiver.create_archive(
            Path(results_dir),
            output_dir=Path(tempfile.gettempdir()),
            categories=categories,
        )
        return str(zip_path)
    except Exception as e:
        print(f"Error creating archive: {e}")
        return None


def get_file_list_for_dropdown(results_dir: str) -> List[str]:
    """Get list of viewable files for dropdown selection."""
    if not results_dir or not RESULTS_AVAILABLE:
        return []
    
    results_path = Path(results_dir)
    if not results_path.exists():
        return []
    
    try:
        collector = ResultCollector()
        summary = collector.scan(results_path)
        
        # Get prioritized files (QC reports and visualizations first)
        priority_files = []
        other_files = []
        
        for f in summary.all_files:
            rel_path = str(f.path.relative_to(results_path))
            cat = f.category.value if hasattr(f.category, 'value') else str(f.category)
            
            if cat in ('qc_reports', 'visualizations'):
                priority_files.append(rel_path)
            else:
                other_files.append(rel_path)
        
        # Limit total files shown
        all_files = priority_files[:30] + other_files[:20]
        return sorted(all_files)
        
    except Exception as e:
        logger.warning(f"Error listing files: {e}")
        return []


# ============================================================================
# Gradio Interface
# ============================================================================

def create_interface() -> gr.Blocks:
    """Create the Gradio interface."""
    
    with gr.Blocks(
        title="BioPipelines - AI Workflow Composer",
    ) as demo:
        # Initialize session state
        app_state = gr.State(AppState)
        
        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #059669 0%, #3b82f6 100%); border-radius: 12px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0; font-size: 2.2em;">🧬 BioPipelines</h1>
            <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0;">AI-Powered Bioinformatics Workflow Composer</p>
        </div>
        """)
        
        # Main Tabs - Unified Workspace Design (3 main tabs)
        with gr.Tabs() as main_tabs:
            
            # ========== WORKSPACE TAB (UNIFIED - Main Entry Point) ==========
            with gr.TabItem("💬 Workspace", id="workspace") as workspace_tab:
                with gr.Row():
                    # Main chat area (70%)
                    with gr.Column(scale=7):
                        # Stats dashboard at top
                        with gr.Row():
                            with gr.Column(scale=1):
                                tools_stat = gr.Markdown("📊 **9,909** Tools")
                            with gr.Column(scale=1):
                                modules_stat = gr.Markdown("📦 **71** Modules")
                            with gr.Column(scale=1):
                                containers_stat = gr.Markdown("🐳 **10** Containers")
                            with gr.Column(scale=1):
                                analyses_stat = gr.Markdown("🧬 **15** Analysis Types")
                        
                        chatbot = gr.Chatbot(
                            label="BioPipelines AI Assistant",
                            height=500,
                            show_label=False,
                        )
                        
                        with gr.Row():
                            msg_input = gr.Textbox(
                                label="Your message",
                                placeholder="Describe your analysis, scan data, run workflows... (e.g., 'Scan data in /path' or 'Create RNA-seq pipeline')",
                                lines=2,
                                scale=5,
                                show_label=False,
                            )
                            send_btn = gr.Button("Send 🚀", variant="primary", scale=1, size="lg")
                        
                        with gr.Accordion("📝 Example Prompts", open=False):
                            gr.Examples(
                                examples=get_example_prompts(),
                                inputs=msg_input,
                                label="Click an example to use it:",
                            )
                    
                    # Sidebar (30%) - UNIFIED with Data + Jobs
                    with gr.Column(scale=3):
                        gr.Markdown("### 🤖 LLM Provider")
                        provider_dropdown = gr.Dropdown(
                            choices=get_provider_choices(),
                            value=get_provider_choices()[0] if get_provider_choices() else None,
                            label="Model",
                            interactive=True,
                            show_label=False,
                        )
                        
                        # ===== AGENT MODE CONTROLS =====
                        with gr.Accordion("🧠 Agent Mode", open=False, visible=ENHANCED_AGENTS and USE_LOCAL_LLM):
                            use_enhanced_toggle = gr.Checkbox(
                                label="Enhanced Agent",
                                value=True,
                                info="Use ReAct reasoning",
                            )
                            with gr.Row():
                                enable_memory_toggle = gr.Checkbox(label="Memory", value=True, scale=1)
                                enable_react_toggle = gr.Checkbox(label="ReAct", value=True, scale=1)
                            enable_self_healing_toggle = gr.Checkbox(
                                label="Self-Healing",
                                value=True,
                                info="Auto-fix failed jobs",
                            )
                            agent_status = gr.Markdown(
                                "✅ **Enhanced mode active**" if (ENHANCED_AGENTS and USE_LOCAL_LLM) 
                                else "⚠️ *Basic mode (no GPU)*"
                            )
                        
                        # ===== AUTONOMOUS AGENT CONTROLS =====
                        if AUTONOMOUS_AVAILABLE and create_autonomous_panel:
                            with gr.Accordion("🤖 Autonomous Agent", open=False, visible=True):
                                # Autonomy Level Control
                                with gr.Row():
                                    autonomy_level = gr.Dropdown(
                                        choices=[
                                            ("Read Only", "READONLY"),
                                            ("Monitored", "MONITORED"),
                                            ("Assisted", "ASSISTED"),
                                            ("Supervised", "SUPERVISED"),
                                            ("Full Autonomous", "AUTONOMOUS"),
                                        ],
                                        value="SUPERVISED",
                                        label="Autonomy Level",
                                        info="Control agent permissions",
                                        scale=2,
                                    )
                                    agent_running = gr.Checkbox(
                                        label="Running",
                                        value=False,
                                        interactive=False,
                                        scale=1,
                                    )
                                
                                # Health Status Widget
                                with gr.Row():
                                    health_status = gr.HTML(
                                        value=check_health_sync() if check_health_sync else "<span style='color:gray'>⚫ Not available</span>"
                                    )
                                
                                # Agent Controls
                                with gr.Row():
                                    start_agent_btn = gr.Button("▶️ Start", size="sm", variant="primary", scale=1)
                                    stop_agent_btn = gr.Button("⏹️ Stop", size="sm", variant="secondary", scale=1)
                                    quick_diagnose_btn = gr.Button("🔍 Diagnose", size="sm", scale=1)
                                
                                # Auto-refresh timer for health
                                health_timer = gr.Timer(30, active=True)
                                
                                # Agent Action Log (last 5 actions)
                                agent_action_log = gr.Markdown("*No recent actions*")
                                
                                gr.Markdown("""
**Commands:**
- `fix [job_id]` - Auto-fix failed job
- `watch [job_id]` - Monitor job
- `health` - System status
- `diagnose [error]` - Analyze error
                                """)
                        
                        gr.Markdown("---")
                        
                        # ===== DATA MANIFEST PANEL =====
                        with gr.Accordion("📁 Data Manifest", open=True):
                            with gr.Row():
                                sidebar_sample_count = gr.Markdown("**0** samples")
                                sidebar_paired_count = gr.Markdown("**0** paired")
                            sidebar_organisms = gr.Markdown("🧬 Not set")
                            sidebar_reference = gr.Markdown("📚 Not configured")
                            
                            gr.Markdown("*Chat: 'scan data in /path' or 'search for...'*")
                        
                        gr.Markdown("---")
                        
                        # ===== ACTIVE JOBS PANEL =====
                        with gr.Accordion("🚀 Active Jobs", open=True):
                            sidebar_jobs_display = gr.HTML(
                                value="<div style='font-size:0.9em;color:#666'><em>No active jobs</em></div>"
                            )
                            with gr.Row():
                                sidebar_refresh_jobs_btn = gr.Button("🔄", size="sm", scale=1)
                                sidebar_view_logs_btn = gr.Button("📄", size="sm", scale=1)
                            
                            # Auto-refresh timer for jobs
                            sidebar_job_timer = gr.Timer(15, active=True)
                        
                        gr.Markdown("---")
                        
                        # ===== RECENT WORKFLOWS =====
                        with gr.Accordion("📋 Recent Workflows", open=False):
                            recent_workflows_display = gr.Markdown(
                                "\n".join([f"- `{w}`" for w in get_available_workflows()[:5]]) 
                                if get_available_workflows() else "*No workflows yet*"
                            )
                            sidebar_workflow_dropdown = gr.Dropdown(
                                choices=get_available_workflows(),
                                label="Select",
                                interactive=True,
                                show_label=False,
                            )
                            with gr.Row():
                                sidebar_run_btn = gr.Button("▶️ Run", size="sm", variant="primary", scale=1)
                                sidebar_view_btn = gr.Button("👁️ View", size="sm", scale=1)
                        
                        gr.Markdown("---")
                        gr.Markdown("### ⚡ Quick Actions")
                        with gr.Row():
                            clear_btn = gr.Button("🗑️ Clear", size="sm", variant="secondary", scale=1)
                            goto_results_btn = gr.Button("📊 Results", size="sm", scale=1)
                        
                        # Workflow preview accordion
                        with gr.Accordion("📄 Workflow Preview", open=False, visible=False) as workflow_preview_accordion:
                            workflow_preview_content = gr.Markdown("*No workflow generated yet*")
                            workflow_preview_code = gr.Code(label="main.nf")
                        
                        gr.Markdown("---")
                        gr.Markdown("""
### 💡 Chat Commands
- **"scan data in /path"** - Find FASTQ files
- **"search for RNA-seq"** - Search databases
- **"create pipeline"** - Generate workflow
- **"run it on SLURM"** - Execute workflow
- **"show logs"** - View job output

**🤖 Autonomous:**
- **"fix [job_id]"** - Auto-fix failed job
- **"watch [job_id]"** - Monitor job
- **"health"** - System status
- **"help"** - Show all commands
                        """)
            
            # ========== EXECUTE TAB (Hidden - functionality in Workspace sidebar) ==========
            # Keep this tab but simplify - all controls are now in Workspace sidebar
            with gr.TabItem("🚀 Execute", id="execute", visible=False) as execute_tab:
                with gr.Row():
                    # Left Panel: Submit (40%)
                    with gr.Column(scale=4):
                        gr.Markdown("### 📤 Submit Workflow")
                        
                        workflow_dropdown = gr.Dropdown(
                            choices=get_available_workflows(),
                            label="Select Workflow",
                            interactive=True,
                        )
                        
                        with gr.Row():
                            refresh_workflows_btn = gr.Button("🔄", size="sm", scale=1)
                            profile_dropdown = gr.Dropdown(
                                choices=["slurm", "local", "docker"],
                                value="slurm",
                                label="Run On",
                                scale=3,
                                show_label=False,
                            )
                        
                        submit_btn = gr.Button("🚀 Submit", variant="primary", size="lg")
                        submission_result = gr.Markdown("")
                        
                        gr.Markdown("---")
                        
                        # Quick actions
                        with gr.Row():
                            download_workflow_btn = gr.Button("📥 Download", size="sm")
                            resume_checkbox = gr.Checkbox(label="Resume", value=False)
                        
                        # Download file (hidden until needed)
                        download_file = gr.File(label="Download", visible=False)
                        
                        # Hidden inputs for compatibility (use defaults)
                        reads_input = gr.Textbox(visible=False, value="")
                        genome_input = gr.Textbox(visible=False, value="")
                        outdir_input = gr.Textbox(visible=False, value="")
                    
                    # Right Panel: Monitor (60%)
                    with gr.Column(scale=6):
                        gr.Markdown("### 📊 Jobs")
                        
                        jobs_display = gr.Markdown("*No jobs yet. Submit a workflow to get started.*")
                        
                        with gr.Row():
                            refresh_monitor_btn = gr.Button("🔄 Refresh", size="sm")
                            auto_refresh_toggle = gr.Checkbox(label="Auto (15s)", value=True, scale=1)
                            job_selector = gr.Dropdown(
                                choices=[],
                                label="Job",
                                interactive=True,
                                scale=3,
                                show_label=False,
                            )
                            cancel_btn = gr.Button("🛑 Cancel", variant="stop", size="sm")
                        
                        # Timer for auto-refresh (15 seconds)
                        auto_refresh_timer = gr.Timer(15, active=True)
                        
                        gr.Markdown("---")
                        
                        # Logs section (expandable)
                        with gr.Accordion("📄 Logs", open=True):
                            with gr.Row():
                                log_lines_slider = gr.Slider(
                                    minimum=20, maximum=200, value=50, step=10,
                                    label="Lines", scale=2,
                                )
                                view_logs_btn = gr.Button("📄 View", size="sm", scale=1)
                            
                            logs_display = gr.Markdown("*Click 'View' to see job logs*")
                        
                        # AI Diagnosis section (for failed jobs)
                        with gr.Accordion("🔍 AI Diagnosis", open=False):
                            gr.Markdown("*Analyze failed jobs with AI-powered error diagnosis*")
                            with gr.Row():
                                diagnose_btn = gr.Button("🔍 Diagnose", variant="secondary", size="sm", scale=1)
                                apply_fixes_btn = gr.Button("✨ Apply Safe Fixes", size="sm", scale=1)
                                create_issue_btn = gr.Button("🐙 Create GitHub Issue", size="sm", scale=1)
                            
                            diagnosis_display = gr.Markdown("*Click 'Diagnose' to analyze a failed job*")
                        
                        # Hidden components for compatibility
                        job_details_display = gr.Markdown(visible=False, value="")
                        slurm_display = gr.Markdown(visible=False, value="")
                        log_job_input = gr.Textbox(visible=False, value="")
                        refresh_details_btn = gr.Button(visible=False)
                        auto_refresh = gr.Checkbox(visible=False, value=False)
            
            # ========== RESULTS TAB ==========
            with gr.TabItem("📊 Results", id="results") as results_tab:
                gr.Markdown("## 📊 Results Visualization & Download")
                
                with gr.Row():
                    # Left Panel: Directory Selection (40%)
                    with gr.Column(scale=4):
                        gr.Markdown("### 📁 Select Results Directory")
                        
                        results_dir_dropdown = gr.Dropdown(
                            choices=get_completed_job_results_dirs(),
                            label="Results Directory",
                            interactive=True,
                            allow_custom_value=True,
                        )
                        
                        with gr.Row():
                            refresh_results_dirs_btn = gr.Button("🔄", size="sm", scale=1)
                            pipeline_type_dropdown = gr.Dropdown(
                                choices=["", "rna_seq", "chip_seq", "dna_seq", "scrna_seq", 
                                         "atac_seq", "metagenomics", "methylation", "long_read"],
                                label="Pipeline Type",
                                scale=3,
                                value="",
                            )
                        
                        scan_results_btn = gr.Button("🔍 Scan Directory", variant="primary")
                        
                        gr.Markdown("---")
                        
                        # Summary display
                        results_summary_display = gr.Markdown("*Select a directory and click 'Scan' to see results*")
                        
                        gr.Markdown("---")
                        
                        # Download section
                        gr.Markdown("### 📥 Download")
                        with gr.Row():
                            download_all_btn = gr.Button("📦 Download All", size="sm", variant="primary")
                            download_qc_btn = gr.Button("📋 QC Reports Only", size="sm")
                        
                        results_download_file = gr.File(label="Download Archive", visible=False)
                    
                    # Right Panel: File Browser & Viewer (60%)
                    with gr.Column(scale=6):
                        gr.Markdown("### 📄 File Browser")
                        
                        # File tree (collapsible)
                        with gr.Accordion("📁 File Tree", open=True):
                            file_tree_display = gr.HTML("*Scan a directory to see file tree*")
                        
                        gr.Markdown("---")
                        
                        # File selection and viewer
                        gr.Markdown("### 👁️ File Viewer")
                        file_selector_dropdown = gr.Dropdown(
                            choices=[],
                            label="Select File to View",
                            interactive=True,
                        )
                        
                        view_file_btn = gr.Button("👁️ View File", size="sm")
                        
                        # Content display area (handles different types)
                        with gr.Tabs() as viewer_tabs:
                            with gr.TabItem("📝 Text/Markdown", id="text_viewer"):
                                text_content_display = gr.Markdown("*Select a file to view its contents*")
                            
                            with gr.TabItem("🖼️ Image", id="image_viewer"):
                                image_content_display = gr.Image(label="Image Preview", visible=True)
                            
                            with gr.TabItem("🌐 HTML Report", id="html_viewer"):
                                html_content_display = gr.HTML("*HTML reports will be displayed here*")
                
                # Hidden state for results summary
                results_summary_state = gr.State(None)
            
            # ========== ADVANCED TAB ==========
            with gr.TabItem("⚙️ Advanced", id="advanced"):
                gr.Markdown("## Advanced Configuration & Exploration")
                
                # LLM Configuration
                with gr.Accordion("🤖 LLM Configuration", open=True):
                    gr.Markdown("### Provider Settings")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### ⚡ Lightning.ai")
                            lightning_status = gr.Markdown(
                                "✅ API Key Loaded (30M FREE tokens/month!)" if os.getenv("LIGHTNING_API_KEY") 
                                else "❌ Not configured - Visit: https://lightning.ai/models"
                            )
                        
                        with gr.Column():
                            gr.Markdown("#### 🟢 OpenAI")
                            openai_status = gr.Markdown(
                                "✅ API Key Loaded" if os.getenv("OPENAI_API_KEY") 
                                else "❌ Not configured - Set OPENAI_API_KEY"
                            )
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### 🟣 vLLM (Local GPU)")
                            vllm_url = gr.Textbox(
                                label="Server URL",
                                value=os.getenv("VLLM_API_BASE", "http://localhost:8000/v1"),
                                interactive=True,
                            )
                        with gr.Column():
                            gr.Markdown("#### Model Selection")
                            vllm_model = gr.Dropdown(
                                choices=["llama3.1-8b", "mistral-7b", "qwen2.5-7b", "codellama-34b"],
                                label="vLLM Model",
                                value="mistral-7b",
                            )
                
                # Ensemble Models
                with gr.Accordion("🧬 Biomedical Model Ensemble", open=False):
                    gr.Markdown("""
                    Multi-model ensemble for accurate biomedical intent parsing:
                    - **BioMistral-7B** (GPU): Primary intent parsing
                    - **BiomedBERT** (CPU): Entity extraction
                    - **SciBERT** (CPU): Scientific term recognition
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            ensemble_status_display = gr.Markdown(get_ensemble_status())
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### Controls")
                            refresh_ensemble_btn = gr.Button("🔄 Refresh Status", size="sm")
                            
                            with gr.Row():
                                start_gpu_btn = gr.Button("▶️ Start GPU", variant="primary", size="sm")
                                stop_gpu_btn = gr.Button("⏹️ Stop GPU", variant="stop", size="sm")
                            
                            preload_cpu_btn = gr.Button("📥 Preload BERT Models", size="sm")
                            ensemble_action_result = gr.Markdown("")
                
                # Tool Browser
                with gr.Accordion("🔧 Tool Browser (9,909 tools)", open=False):
                    gr.Markdown("Search and explore available bioinformatics tools")
                    with gr.Row():
                        tool_search = gr.Textbox(
                            label="Search Tools",
                            placeholder="e.g., alignment, variant, fastq...",
                            scale=3,
                        )
                        container_filter = gr.Dropdown(
                            choices=["", "base", "rna-seq", "dna-seq", "chip-seq", "atac-seq", 
                                     "scrna-seq", "metagenomics", "methylation", "long_read"],
                            label="Container",
                            scale=1,
                        )
                    
                    tool_results = gr.Markdown("*Enter a search term to find tools...*")
                
                # Modules Browser
                with gr.Accordion("📦 Nextflow Modules (71 modules)", open=False):
                    gr.Markdown("Browse available Nextflow modules by category")
                    modules_display = gr.Markdown(get_modules_by_category(AppState()))
                    refresh_modules_btn = gr.Button("🔄 Refresh Modules", size="sm")
                
                gr.Markdown("---")
                gr.Markdown("### 📊 System Information")
                system_info = gr.Markdown(f"""
                - **Workflow Composer:** {'✅ Available' if COMPOSER_AVAILABLE else '❌ Not installed'}
                - **Model Orchestrator:** {'✅ Available' if ORCHESTRATOR_AVAILABLE else '❌ Not installed'}
                - **Data Discovery:** {'✅ Available' if DISCOVERY_AVAILABLE else '❌ Not installed'}
                - **Generated Workflows:** `{GENERATED_DIR}`
                - **Environment:** `~/envs/biopipelines`
                - **Python:** `{Path(__file__).parent}`
                """)
        
        # ========== EVENT HANDLERS ==========
        
        # Create wrapper for chat that respects agent mode toggles
        def smart_chat(message, history, provider, app_state, use_enhanced, enable_memory, enable_react, enable_self_healing):
            """Route to enhanced or basic chat based on settings."""
            if use_enhanced and ENHANCED_AGENTS and USE_LOCAL_LLM:
                yield from enhanced_chat_with_composer(
                    message, history, provider, app_state,
                    use_enhanced=True,
                    enable_memory=enable_memory,
                    enable_react=enable_react,
                    enable_self_healing=enable_self_healing,
                )
            else:
                yield from chat_with_composer(message, history, provider, app_state)
        
        # Workspace Tab - Chat handlers
        msg_input.submit(
            fn=smart_chat,
            inputs=[msg_input, chatbot, provider_dropdown, app_state, 
                    use_enhanced_toggle, enable_memory_toggle, enable_react_toggle, enable_self_healing_toggle],
            outputs=[msg_input, chatbot],
        ).then(
            fn=lambda: gr.update(choices=[j.job_id for j in pipeline_executor.list_jobs()]),
            outputs=job_selector,
        ).then(
            fn=lambda: gr.update(choices=get_available_workflows()),
            outputs=workflow_dropdown,
        ).then(
            fn=lambda: "\n".join([f"- `{w}`" for w in get_available_workflows()[:5]]) 
                if get_available_workflows() else "*No workflows yet*",
            outputs=recent_workflows_display,
        )
        
        send_btn.click(
            fn=smart_chat,
            inputs=[msg_input, chatbot, provider_dropdown, app_state,
                    use_enhanced_toggle, enable_memory_toggle, enable_react_toggle, enable_self_healing_toggle],
            outputs=[chatbot, msg_input],
        ).then(
            fn=lambda: gr.update(choices=[j.job_id for j in pipeline_executor.list_jobs()]),
            outputs=job_selector,
        ).then(
            fn=lambda: gr.update(choices=get_available_workflows()),
            outputs=workflow_dropdown,
        ).then(
            fn=lambda: "\n".join([f"- `{w}`" for w in get_available_workflows()[:5]]) 
                if get_available_workflows() else "*No workflows yet*",
            outputs=recent_workflows_display,
        )
        
        clear_btn.click(
            fn=lambda: ([], ""),
            outputs=[chatbot, msg_input],
        )
        
        # Go to Results tab button (renamed from goto_execute_btn)
        goto_results_btn.click(
            fn=lambda: gr.Tabs(selected="results"),
            outputs=main_tabs,
        )
        
        # Sidebar workflow run button
        sidebar_run_btn.click(
            fn=submit_pipeline,
            inputs=[
                sidebar_workflow_dropdown,
                gr.State("slurm"),  # Default profile
                gr.State(False),    # No resume
                gr.State(""),       # reads
                gr.State(""),       # genome
                gr.State(""),       # outdir
            ],
            outputs=sidebar_jobs_display,
        )
        
        # Sidebar job refresh
        sidebar_refresh_jobs_btn.click(
            fn=refresh_monitoring,
            outputs=[sidebar_jobs_display, gr.State(None)],
        )
        
        # Sidebar auto-refresh timer for jobs
        sidebar_job_timer.tick(
            fn=refresh_monitoring,
            outputs=[sidebar_jobs_display, gr.State(None)],
        )
        
        # ===== AUTONOMOUS AGENT EVENT HANDLERS =====
        if AUTONOMOUS_AVAILABLE and create_autonomous_panel:
            # Health timer refresh
            health_timer.tick(
                fn=check_health_sync,
                outputs=[health_status],
            )
            
            # Start agent button
            def start_autonomous_agent(level: str):
                """Start the autonomous agent with specified level."""
                try:
                    agent = get_agent()
                    if agent:
                        # Agent already running
                        return (
                            True,
                            check_health_sync(),
                            "🟢 Agent already running",
                        )
                    # Create new agent
                    from workflow_composer.agents.autonomous import create_agent, AutonomyLevel
                    level_enum = AutonomyLevel[level.upper()]
                    agent = create_agent(level=level_enum.name.lower())
                    return (
                        True,
                        check_health_sync(),
                        f"🟢 Agent started at **{level}** level",
                    )
                except Exception as e:
                    return (
                        False,
                        f"<span style='color:red'>⚠️ Error: {e}</span>",
                        f"❌ Failed to start: {e}",
                    )
            
            start_agent_btn.click(
                fn=start_autonomous_agent,
                inputs=[autonomy_level],
                outputs=[agent_running, health_status, agent_action_log],
            )
            
            # Stop agent button
            def stop_autonomous_agent():
                """Stop the autonomous agent."""
                try:
                    agent = get_agent()
                    if agent:
                        # Clear global instance
                        import workflow_composer.web.components.autonomous_panel as panel
                        panel._agent_instance = None
                        return (
                            False,
                            "<span style='color:gray'>⚫ Agent stopped</span>",
                            "⏹️ Agent stopped",
                        )
                    return (
                        False,
                        check_health_sync(),
                        "ℹ️ Agent was not running",
                    )
                except Exception as e:
                    return (
                        False,
                        check_health_sync(),
                        f"❌ Error stopping: {e}",
                    )
            
            stop_agent_btn.click(
                fn=stop_autonomous_agent,
                outputs=[agent_running, health_status, agent_action_log],
            )
            
            # Diagnose button (sidebar version)
            def quick_diagnose():
                """Quick system diagnosis."""
                try:
                    health = check_health_sync()
                    return health
                except Exception as e:
                    return f"<span style='color:red'>⚠️ Diagnosis error: {e}</span>"
            
            quick_diagnose_btn.click(
                fn=quick_diagnose,
                outputs=[health_status],
            )
            
            # Autonomy level change
            def change_autonomy_level(level: str):
                """Update agent autonomy level."""
                agent = get_agent()
                if agent:
                    try:
                        from workflow_composer.agents.autonomous import AutonomyLevel
                        level_enum = AutonomyLevel[level.upper()]
                        agent.autonomy_level = level_enum
                        return f"✅ Autonomy set to **{level}**"
                    except Exception as e:
                        return f"❌ Error: {e}"
                return "ℹ️ Start agent first"
            
            autonomy_level.change(
                fn=change_autonomy_level,
                inputs=[autonomy_level],
                outputs=[agent_action_log],
            )
        
        # View workflow button - shows preview accordion
        sidebar_view_btn.click(
            fn=get_workflow_preview,
            inputs=[app_state],
            outputs=[workflow_preview_accordion, workflow_preview_content, workflow_preview_code],
        )
        
        # Execute Tab - Submission handlers
        refresh_workflows_btn.click(
            fn=lambda: gr.update(choices=get_available_workflows()),
            outputs=workflow_dropdown,
        )
        
        download_workflow_btn.click(
            fn=download_latest_workflow,
            outputs=download_file,
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=download_file,
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
            fn=lambda: refresh_monitoring()[0],  # Only return jobs_display content
            outputs=jobs_display,
        ).then(
            fn=lambda: gr.update(choices=[j.job_id for j in pipeline_executor.list_jobs()]),
            outputs=job_selector,
        )
        
        view_logs_btn.click(
            fn=lambda lines: get_job_logs("", lines),  # Auto-get latest job
            inputs=[log_lines_slider],
            outputs=logs_display,
        )
        
        # Diagnose button - AI error analysis
        diagnose_btn.click(
            fn=lambda job_id: diagnose_job_failure(job_id),
            inputs=[job_selector],
            outputs=diagnosis_display,
        )
        
        # Apply safe fixes
        apply_fixes_btn.click(
            fn=apply_safe_fixes,
            inputs=[job_selector],
            outputs=diagnosis_display,
        )
        
        # Create GitHub issue
        create_issue_btn.click(
            fn=create_github_issue,
            inputs=[job_selector],
            outputs=diagnosis_display,
        )
        
        # Auto-refresh timer for job status
        def auto_refresh_jobs():
            """Refresh job status and return updated display."""
            jobs_content = refresh_monitoring()[0]
            job_choices = [j.job_id for j in pipeline_executor.list_jobs()]
            return jobs_content, gr.update(choices=job_choices)
        
        auto_refresh_timer.tick(
            fn=auto_refresh_jobs,
            outputs=[jobs_display, job_selector],
        )
        
        # Toggle auto-refresh on/off
        auto_refresh_toggle.change(
            fn=lambda active: gr.Timer(15, active=active),
            inputs=auto_refresh_toggle,
            outputs=auto_refresh_timer,
        )
        
        # ========== Results Tab Event Handlers ==========
        
        # Refresh results directories list
        refresh_results_dirs_btn.click(
            fn=lambda: gr.update(choices=get_completed_job_results_dirs()),
            outputs=results_dir_dropdown,
        )
        
        # Scan results directory
        def handle_scan_results(results_dir, pipeline_type):
            """Handle scanning a results directory."""
            summary_md, tree_html, summary_data = scan_results_directory(results_dir, pipeline_type)
            file_choices = get_file_list_for_dropdown(results_dir)
            return summary_md, tree_html, summary_data, gr.update(choices=file_choices)
        
        scan_results_btn.click(
            fn=handle_scan_results,
            inputs=[results_dir_dropdown, pipeline_type_dropdown],
            outputs=[results_summary_display, file_tree_display, results_summary_state, file_selector_dropdown],
        )
        
        # Also scan when directory changes
        results_dir_dropdown.change(
            fn=handle_scan_results,
            inputs=[results_dir_dropdown, pipeline_type_dropdown],
            outputs=[results_summary_display, file_tree_display, results_summary_state, file_selector_dropdown],
        )
        
        # View selected file
        def handle_view_file(results_dir, file_path):
            """Handle viewing a file."""
            content_type, content, image_path = view_result_file(results_dir, file_path)
            
            # Update the appropriate viewer
            if content_type == 'image' and image_path:
                return content, image_path, "", gr.Tabs(selected="image_viewer")
            elif content_type == 'html':
                return "", None, content, gr.Tabs(selected="html_viewer")
            else:
                return content, None, "", gr.Tabs(selected="text_viewer")
        
        view_file_btn.click(
            fn=handle_view_file,
            inputs=[results_dir_dropdown, file_selector_dropdown],
            outputs=[text_content_display, image_content_display, html_content_display, viewer_tabs],
        )
        
        # Also view on file selection change
        file_selector_dropdown.change(
            fn=handle_view_file,
            inputs=[results_dir_dropdown, file_selector_dropdown],
            outputs=[text_content_display, image_content_display, html_content_display, viewer_tabs],
        )
        
        # Download all results
        def handle_download_all(results_dir):
            """Handle downloading all results."""
            zip_path = create_results_archive(results_dir)
            if zip_path:
                return zip_path, gr.update(visible=True)
            return None, gr.update(visible=False)
        
        download_all_btn.click(
            fn=handle_download_all,
            inputs=[results_dir_dropdown],
            outputs=[results_download_file, results_download_file],
        )
        
        # Download QC reports only
        def handle_download_qc(results_dir):
            """Handle downloading QC reports only."""
            zip_path = create_results_archive(results_dir, categories=['qc_reports', 'visualizations'])
            if zip_path:
                return zip_path, gr.update(visible=True)
            return None, gr.update(visible=False)
        
        download_qc_btn.click(
            fn=handle_download_qc,
            inputs=[results_dir_dropdown],
            outputs=[results_download_file, results_download_file],
        )
        
        # Advanced Tab - Tool/Module search
        tool_search.change(
            fn=search_tools,
            inputs=[tool_search, container_filter, app_state],
            outputs=tool_results,
        )
        container_filter.change(
            fn=search_tools,
            inputs=[tool_search, container_filter, app_state],
            outputs=tool_results,
        )
        
        refresh_modules_btn.click(
            fn=get_modules_by_category,
            inputs=[app_state],
            outputs=modules_display,
        )
        
        # Advanced Tab - Ensemble controls
        refresh_ensemble_btn.click(
            fn=get_ensemble_status,
            outputs=ensemble_status_display,
        )
        
        start_gpu_btn.click(
            fn=start_biomistral_service,
            outputs=ensemble_action_result,
        ).then(
            fn=get_ensemble_status,
            outputs=ensemble_status_display,
        )
        
        stop_gpu_btn.click(
            fn=stop_biomistral_service,
            outputs=ensemble_action_result,
        ).then(
            fn=get_ensemble_status,
            outputs=ensemble_status_display,
        )
        
        preload_cpu_btn.click(
            fn=preload_cpu_models,
            outputs=ensemble_action_result,
        ).then(
            fn=get_ensemble_status,
            outputs=ensemble_status_display,
        )
        
        # ========== Autonomous Agent Event Handlers ==========
        if AUTONOMOUS_AVAILABLE and check_health_sync:
            # Health check timer
            health_timer.tick(
                fn=check_health_sync,
                outputs=[health_status],
            )
            
            # Diagnose button (uses sidebar diagnose_btn)
            diagnose_btn.click(
                fn=lambda: check_health_sync() if check_health_sync else "<span style='color:gray'>Health check not available</span>",
                outputs=[health_status],
            )
            
            # Start agent button
            def start_autonomous_agent(level: str):
                """Start the autonomous agent with specified level."""
                try:
                    agent = get_agent() if get_agent else None
                    if agent:
                        # Agent is already created, update level
                        return True, f"✅ Agent started at {level} level"
                    return False, "⚠️ Agent not available"
                except Exception as e:
                    return False, f"❌ Error: {e}"
            
            start_agent_btn.click(
                fn=start_autonomous_agent,
                inputs=[autonomy_level],
                outputs=[agent_running, agent_action_log],
            )
            
            # Stop agent button
            def stop_autonomous_agent():
                """Stop the autonomous agent."""
                try:
                    agent = get_agent() if get_agent else None
                    if agent:
                        # Signal agent to stop
                        return False, "⏹️ Agent stopped"
                    return False, "Agent was not running"
                except Exception as e:
                    return False, f"❌ Error: {e}"
            
            stop_agent_btn.click(
                fn=stop_autonomous_agent,
                outputs=[agent_running, agent_action_log],
            )
            
            # Autonomy level change
            def update_autonomy_level(level: str):
                """Update the agent's autonomy level."""
                try:
                    agent = get_agent() if get_agent else None
                    if agent:
                        return f"✅ Autonomy level set to: {level}"
                    return f"Level will be {level} when agent starts"
                except Exception as e:
                    return f"❌ Error: {e}"
            
            autonomy_level.change(
                fn=update_autonomy_level,
                inputs=[autonomy_level],
                outputs=[agent_action_log],
            )
        
    return demo


def main():
    """Launch the Gradio web interface."""
    import argparse
    import socket
    
    def find_available_port(start_port: int, max_attempts: int = 100) -> int:
        """Find an available port starting from start_port."""
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                continue
        return start_port  # Fall back to original port
    
    parser = argparse.ArgumentParser(description="BioPipelines Web UI")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--port-range", type=int, default=100, help="Range of ports to try if default is busy")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Find available port
    actual_port = find_available_port(args.port, args.port_range)
    if actual_port != args.port:
        print(f"  ℹ️  Port {args.port} in use, using port {actual_port}")
    
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
    
    print(f"  🌐 Server: http://localhost:{actual_port}")
    print(f"  🌐 Network: http://{args.host}:{actual_port}")
    print()
    
    # Initialize app state
    available = check_available_providers()
    if any(available.values()):
        provider = next(k for k, v in available.items() if v)
        print(f"  ✅ Default provider: {provider}")
    else:
        print("  ⚠️  No LLM providers available - running in demo mode")
    
    # Create and launch interface
    demo = create_interface()
    
    demo.launch(
        server_name=args.host,
        server_port=actual_port,
        share=args.share,
        debug=args.debug,
        show_error=True,
    )


if __name__ == "__main__":
    main()

