# Error Diagnosis Agent - Detailed Implementation Guide

**Created:** November 25, 2025  
**Status:** 🚧 Active Implementation  
**Priority:** 🔴 CRITICAL

---

## Overview

This document provides the complete implementation specification for the Error Diagnosis Agent - an AI-powered system that analyzes pipeline failures, identifies root causes, and provides auto-fix capabilities.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         ERROR DIAGNOSIS AGENT                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌─────────────────┐                                                            │
│   │  Failed Job     │                                                            │
│   │  (PipelineJob)  │                                                            │
│   └────────┬────────┘                                                            │
│            │                                                                     │
│            ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                       LOG COLLECTOR                                      │   │
│   │  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  ┌────────────┐   │   │
│   │  │ .nextflow   │  │ slurm_*.err  │  │ .command.err  │  │ main.nf    │   │   │
│   │  │ .log        │  │ slurm_*.out  │  │ .command.out  │  │ config     │   │   │
│   │  └─────────────┘  └──────────────┘  └───────────────┘  └────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│            │                                                                     │
│            ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  TIER 1: PATTERN MATCHER (Fast - No LLM)                                 │   │
│   │  ┌────────────────────────────────────────────────────────────────────┐ │   │
│   │  │  Pattern Database (50+ patterns)                                   │ │   │
│   │  │  • File errors: No such file, Permission denied                   │ │   │
│   │  │  • Memory errors: OOM, Killed, exceeded limit                     │ │   │
│   │  │  • Container errors: FATAL, image not found                       │ │   │
│   │  │  • Tool errors: Command failed, exit code !=0                     │ │   │
│   │  │  • Network errors: Connection refused, timeout                    │ │   │
│   │  └────────────────────────────────────────────────────────────────────┘ │   │
│   │                          │                                               │   │
│   │           ┌──────────────┴──────────────┐                               │   │
│   │           ▼                              ▼                               │   │
│   │     [Match Found]                  [No Match]                           │   │
│   │           │                              │                               │   │
│   │           ▼                              ▼                               │   │
│   │  ┌────────────────┐           ┌────────────────────────────────────┐   │   │
│   │  │ Quick Response │           │ TIER 2: LLM DEEP ANALYSIS          │   │   │
│   │  │ (No LLM call)  │           │                                    │   │   │
│   │  └────────────────┘           │  Provider Priority:                │   │   │
│   │                               │  1. GitHub Copilot (if available)  │   │   │
│   │                               │  2. Lightning.ai Free Tier         │   │   │
│   │                               │  3. Gemini Free API                │   │   │
│   │                               │  4. Local vLLM/Ollama              │   │   │
│   │                               │  5. OpenAI (paid backup)           │   │   │
│   │                               └────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│            │                                                                     │
│            ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                         FIX SUGGESTIONS                                  │   │
│   │  ┌─────────────────────────────────────────────────────────────────────┐│   │
│   │  │ 🟢 SAFE (Auto)        🟡 LOW (Notify)        🔴 HIGH (Confirm)      ││   │
│   │  │ • mkdir -p dir        • pull container       • modify workflow      ││   │
│   │  │ • retry job           • download reference   • install package      ││   │
│   │  │ • increase memory     • fix permissions      • edit config          ││   │
│   │  └─────────────────────────────────────────────────────────────────────┘│   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│            │                                                                     │
│            ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  TIER 3: CODING AGENT (For Code Fixes)                                   │   │
│   │  ┌─────────────────────────────────────────────────────────────────────┐│   │
│   │  │ GitHub Copilot Coding Agent                                         ││   │
│   │  │ → Creates PR with workflow fix                                      ││   │
│   │  │ → mcp_github_create_pull_request_with_copilot()                    ││   │
│   │  └─────────────────────────────────────────────────────────────────────┘│   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## LLM Provider Strategy

### Priority Order for Diagnosis

| Priority | Provider | Cost | Speed | Best For |
|----------|----------|------|-------|----------|
| 1 | GitHub Copilot | Subscription | Fast | Code-related errors |
| 2 | Lightning.ai Free | Free | Medium | General diagnosis |
| 3 | Google Gemini Flash | Free tier | Fast | Quick analysis |
| 4 | Local Ollama/vLLM | Free | Variable | Offline fallback |
| 5 | OpenAI GPT-4o | Paid | Fast | Complex cases |

### Provider Configuration

```python
DIAGNOSIS_PROVIDERS = {
    "github_copilot": {
        "priority": 1,
        "cost": "subscription",
        "use_for": ["code_fix", "workflow_error"],
        "requires": "GITHUB_TOKEN",
    },
    "lightning": {
        "priority": 2,
        "cost": "free_tier",
        "use_for": ["general_diagnosis", "error_analysis"],
        "requires": "LIGHTNING_API_KEY",
        "models": ["meta-llama/Llama-3.2-3B-Instruct"],
    },
    "gemini": {
        "priority": 3,
        "cost": "free_tier",
        "use_for": ["quick_analysis"],
        "requires": "GOOGLE_API_KEY",
        "models": ["gemini-1.5-flash"],
    },
    "ollama": {
        "priority": 4,
        "cost": "free",
        "use_for": ["offline_fallback"],
        "models": ["codellama", "mistral", "qwen2.5-coder"],
    },
    "openai": {
        "priority": 5,
        "cost": "paid",
        "use_for": ["complex_analysis"],
        "requires": "OPENAI_API_KEY",
    },
}
```

---

## File Structure

```
src/workflow_composer/diagnosis/
├── __init__.py              # Package exports
├── agent.py                 # Main ErrorDiagnosisAgent class
├── categories.py            # Error category taxonomy
├── patterns.py              # Pattern database (50+ patterns)
├── log_collector.py         # Log file aggregation
├── llm_diagnosis.py         # LLM-based analysis
├── auto_fix.py              # Auto-fix execution engine
├── prompts.py               # LLM prompt templates
├── github_agent.py          # GitHub Copilot integration
└── gemini_adapter.py        # Google Gemini adapter (new)
```

---

## Implementation Details

### 1. Error Categories (`categories.py`)

```python
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

class ErrorCategory(Enum):
    """Classification of pipeline errors."""
    FILE_NOT_FOUND = "file_not_found"
    PERMISSION_DENIED = "permission_denied"
    OUT_OF_MEMORY = "out_of_memory"
    CONTAINER_ERROR = "container_error"
    DEPENDENCY_MISSING = "dependency_missing"
    TOOL_ERROR = "tool_error"
    SYNTAX_ERROR = "syntax_error"
    NETWORK_ERROR = "network_error"
    RESOURCE_LIMIT = "resource_limit"
    DATA_FORMAT_ERROR = "data_format_error"
    REFERENCE_MISSING = "reference_missing"
    INDEX_MISSING = "index_missing"
    CONFIGURATION_ERROR = "configuration_error"
    SLURM_ERROR = "slurm_error"
    UNKNOWN = "unknown"

class FixRiskLevel(Enum):
    """Risk level for auto-fix actions."""
    SAFE = "safe"       # Can execute automatically
    LOW = "low"         # Execute with notification
    MEDIUM = "medium"   # Require user confirmation  
    HIGH = "high"       # Show instructions only

@dataclass
class ErrorPattern:
    """Definition of an error pattern."""
    category: ErrorCategory
    patterns: List[str]  # Regex patterns
    description: str
    common_causes: List[str]
    suggested_fixes: List['FixSuggestion']

@dataclass
class FixSuggestion:
    """A suggested fix for an error."""
    description: str
    command: Optional[str] = None
    risk_level: FixRiskLevel = FixRiskLevel.MEDIUM
    auto_executable: bool = False
    
@dataclass
class ErrorDiagnosis:
    """Result of error diagnosis."""
    category: ErrorCategory
    confidence: float
    root_cause: str
    user_explanation: str
    log_excerpt: str
    suggested_fixes: List[FixSuggestion]
    llm_provider_used: Optional[str] = None
    pattern_matched: bool = False
```

### 2. Pattern Database (`patterns.py`)

```python
from .categories import ErrorCategory, ErrorPattern, FixSuggestion, FixRiskLevel

ERROR_PATTERNS = {
    ErrorCategory.FILE_NOT_FOUND: ErrorPattern(
        category=ErrorCategory.FILE_NOT_FOUND,
        patterns=[
            r"No such file or directory: (.+)",
            r"FileNotFoundError: \[Errno 2\] (.+)",
            r"cannot open file '(.+)'",
            r"Error: Unable to open (.+)",
            r"Input file (.+) does not exist",
            r"Path not found: (.+)",
            r"Failed to open (.+)",
        ],
        description="A required file or directory was not found",
        common_causes=[
            "Reference genome not downloaded",
            "Incorrect sample sheet paths",
            "Typo in file path",
            "Symlink pointing to deleted file",
            "Input files moved or renamed",
        ],
        suggested_fixes=[
            FixSuggestion(
                description="Check if the file path exists",
                command="ls -la {path}",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Download missing reference genome",
                command="python -m workflow_composer.data.downloader download-reference {organism} {build}",
                risk_level=FixRiskLevel.LOW,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Create missing directory",
                command="mkdir -p {directory}",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
        ],
    ),
    
    ErrorCategory.OUT_OF_MEMORY: ErrorPattern(
        category=ErrorCategory.OUT_OF_MEMORY,
        patterns=[
            r"MemoryError",
            r"Out of memory",
            r"Killed.*memory",
            r"slurmstepd: error:.*oom-kill",
            r"exceeded memory limit",
            r"Cannot allocate memory",
            r"std::bad_alloc",
            r"java\.lang\.OutOfMemoryError",
            r"FATAL:   While making image.*memory",
        ],
        description="The process ran out of available memory",
        common_causes=[
            "Input files too large for allocated memory",
            "Too many parallel processes",
            "Memory leak in tool",
            "SLURM memory limit too low",
        ],
        suggested_fixes=[
            FixSuggestion(
                description="Increase SLURM memory allocation",
                command="# Edit slurm config: --mem=32G → --mem=64G",
                risk_level=FixRiskLevel.MEDIUM,
                auto_executable=False,
            ),
            FixSuggestion(
                description="Reduce parallel threads",
                command="# Set --cpus 4 instead of 8",
                risk_level=FixRiskLevel.LOW,
                auto_executable=False,
            ),
            FixSuggestion(
                description="Process files in smaller batches",
                risk_level=FixRiskLevel.MEDIUM,
                auto_executable=False,
            ),
        ],
    ),
    
    ErrorCategory.CONTAINER_ERROR: ErrorPattern(
        category=ErrorCategory.CONTAINER_ERROR,
        patterns=[
            r"FATAL:.*container",
            r"singularity: command not found",
            r"Failed to pull container",
            r"Image not found: (.+\.sif)",
            r"FATAL:.*image file",
            r"Error loading image",
            r"container runtime error",
            r"OCI runtime error",
        ],
        description="Container (Singularity/Docker) error",
        common_causes=[
            "Container image not built",
            "Singularity module not loaded",
            "Container registry unavailable",
            "Corrupted image file",
        ],
        suggested_fixes=[
            FixSuggestion(
                description="Load Singularity module",
                command="module load singularity",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Build missing container",
                command="sbatch scripts/containers/build_{container}_container.slurm",
                risk_level=FixRiskLevel.LOW,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Pull container from registry",
                command="singularity pull {image_url}",
                risk_level=FixRiskLevel.LOW,
                auto_executable=True,
            ),
        ],
    ),
    
    ErrorCategory.PERMISSION_DENIED: ErrorPattern(
        category=ErrorCategory.PERMISSION_DENIED,
        patterns=[
            r"Permission denied",
            r"EACCES",
            r"Operation not permitted",
            r"Access denied",
            r"cannot write to",
        ],
        description="Insufficient permissions to access file or directory",
        common_causes=[
            "File owned by another user",
            "Directory not writable",
            "Scratch space permissions",
        ],
        suggested_fixes=[
            FixSuggestion(
                description="Fix file permissions",
                command="chmod u+rw {path}",
                risk_level=FixRiskLevel.LOW,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Check directory ownership",
                command="ls -la {directory}",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
        ],
    ),
    
    ErrorCategory.DEPENDENCY_MISSING: ErrorPattern(
        category=ErrorCategory.DEPENDENCY_MISSING,
        patterns=[
            r"ModuleNotFoundError: No module named '(.+)'",
            r"ImportError: (.+)",
            r"command not found: (.+)",
            r"Package (.+) is not installed",
            r"Cannot find (.+) in PATH",
            r"Error: Unable to find (.+)",
        ],
        description="A required software dependency is missing",
        common_causes=[
            "Python package not installed",
            "Tool not in PATH",
            "Module not loaded",
            "Conda environment not activated",
        ],
        suggested_fixes=[
            FixSuggestion(
                description="Install missing Python package",
                command="pip install {package}",
                risk_level=FixRiskLevel.MEDIUM,
                auto_executable=False,
            ),
            FixSuggestion(
                description="Load required module",
                command="module load {module}",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Activate conda environment",
                command="conda activate biopipelines",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
        ],
    ),
    
    ErrorCategory.REFERENCE_MISSING: ErrorPattern(
        category=ErrorCategory.REFERENCE_MISSING,
        patterns=[
            r"Reference genome not found",
            r"Cannot find reference",
            r"FASTA file not found",
            r"GTF/GFF file missing",
            r"Annotation file not found",
        ],
        description="Reference genome or annotation file is missing",
        common_causes=[
            "Reference not downloaded",
            "Wrong genome build specified",
            "Path misconfigured",
        ],
        suggested_fixes=[
            FixSuggestion(
                description="Download reference genome",
                command="python -m workflow_composer.data.downloader download-reference {organism} {build}",
                risk_level=FixRiskLevel.LOW,
                auto_executable=True,
            ),
        ],
    ),
    
    ErrorCategory.INDEX_MISSING: ErrorPattern(
        category=ErrorCategory.INDEX_MISSING,
        patterns=[
            r"Index (.+) not found",
            r"BWA index missing",
            r"STAR index not found",
            r"Bowtie2 index missing",
            r"Genome index does not exist",
            r"\.bwt file not found",
            r"Genome directory not found",
        ],
        description="Aligner index files are missing",
        common_causes=[
            "Index not built",
            "Wrong index path",
            "Incomplete index build",
        ],
        suggested_fixes=[
            FixSuggestion(
                description="Build BWA index",
                command="bwa index {reference}",
                risk_level=FixRiskLevel.LOW,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Build STAR index",
                command="STAR --runMode genomeGenerate --genomeDir {outdir} --genomeFastaFiles {fasta} --sjdbGTFfile {gtf}",
                risk_level=FixRiskLevel.LOW,
                auto_executable=True,
            ),
        ],
    ),
    
    ErrorCategory.SLURM_ERROR: ErrorPattern(
        category=ErrorCategory.SLURM_ERROR,
        patterns=[
            r"SLURM_JOB.*CANCELLED",
            r"slurmstepd: error",
            r"DUE TO TIME LIMIT",
            r"JOB.*TIMEOUT",
            r"Exceeded job memory limit",
            r"srun: error",
            r"sbatch: error",
        ],
        description="SLURM scheduler error",
        common_causes=[
            "Job exceeded time limit",
            "Job exceeded memory limit",
            "Node failure",
            "Preemption",
        ],
        suggested_fixes=[
            FixSuggestion(
                description="Increase time limit",
                command="# Modify --time parameter",
                risk_level=FixRiskLevel.MEDIUM,
                auto_executable=False,
            ),
            FixSuggestion(
                description="Retry job",
                command="sbatch {script}",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
        ],
    ),
    
    ErrorCategory.NETWORK_ERROR: ErrorPattern(
        category=ErrorCategory.NETWORK_ERROR,
        patterns=[
            r"Connection refused",
            r"Connection timed out",
            r"Network is unreachable",
            r"Could not resolve host",
            r"Failed to connect",
            r"SSL certificate problem",
            r"curl: \(\d+\)",
            r"wget: unable to resolve",
        ],
        description="Network connectivity error",
        common_causes=[
            "Server unreachable",
            "Firewall blocking",
            "DNS resolution failure",
            "SSL/TLS issues",
        ],
        suggested_fixes=[
            FixSuggestion(
                description="Check network connectivity",
                command="ping -c 3 {host}",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Retry download",
                command="wget --retry-connrefused --waitretry=5 --timeout=60 {url}",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
        ],
    ),
    
    ErrorCategory.TOOL_ERROR: ErrorPattern(
        category=ErrorCategory.TOOL_ERROR,
        patterns=[
            r"Error executing process.*>.*\((\d+)\)",
            r"Command exited with non-zero status",
            r"Exit status: (\d+)",
            r"Command error:",
            r"failed with exit code",
            r"returned non-zero exit status",
        ],
        description="Bioinformatics tool returned an error",
        common_causes=[
            "Invalid input data",
            "Wrong parameters",
            "Tool bug",
            "Incompatible file format",
        ],
        suggested_fixes=[
            FixSuggestion(
                description="Check tool logs for details",
                command="cat {work_dir}/.command.err",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Validate input files",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=False,
            ),
        ],
    ),
    
    ErrorCategory.DATA_FORMAT_ERROR: ErrorPattern(
        category=ErrorCategory.DATA_FORMAT_ERROR,
        patterns=[
            r"Invalid FASTQ",
            r"Malformed BAM",
            r"VCF format error",
            r"Invalid BED format",
            r"Truncated file",
            r"Unexpected end of file",
            r"gzip: (.+): not in gzip format",
        ],
        description="Input data format is invalid or corrupted",
        common_causes=[
            "Corrupted file download",
            "Wrong file format",
            "Incomplete file transfer",
            "Compression issue",
        ],
        suggested_fixes=[
            FixSuggestion(
                description="Check file integrity",
                command="md5sum {file}",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Re-download file",
                risk_level=FixRiskLevel.LOW,
                auto_executable=False,
            ),
        ],
    ),
}

def get_pattern(category: ErrorCategory) -> ErrorPattern:
    """Get pattern definition for a category."""
    return ERROR_PATTERNS.get(category)

def get_all_patterns() -> dict:
    """Get all error patterns."""
    return ERROR_PATTERNS
```

### 3. Log Collector (`log_collector.py`)

```python
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
import re

@dataclass
class CollectedLogs:
    """Aggregated log content from various sources."""
    nextflow_log: str = ""
    slurm_err: str = ""
    slurm_out: str = ""
    command_err: str = ""
    command_out: str = ""
    workflow_file: str = ""
    config_file: str = ""
    failed_process: Optional[str] = None
    work_directory: Optional[str] = None
    
    def get_combined_error_context(self, max_lines: int = 100) -> str:
        """Get combined error context for LLM analysis."""
        sections = []
        
        if self.nextflow_log:
            # Extract last N lines with errors
            lines = self.nextflow_log.split('\n')
            error_lines = [l for l in lines if any(kw in l.lower() for kw in ['error', 'fail', 'exception', 'fatal'])]
            sections.append(f"### Nextflow Log (errors):\n```\n{chr(10).join(error_lines[-30:])}\n```")
        
        if self.slurm_err:
            sections.append(f"### SLURM Error:\n```\n{self.slurm_err[-2000:]}\n```")
        
        if self.command_err:
            sections.append(f"### Command Error:\n```\n{self.command_err[-2000:]}\n```")
        
        return "\n\n".join(sections)


class LogCollector:
    """Collects and aggregates log files for error analysis."""
    
    def __init__(self, max_log_size: int = 50000):
        self.max_log_size = max_log_size
    
    def collect(self, job) -> CollectedLogs:
        """Collect all relevant logs for a job."""
        logs = CollectedLogs()
        
        # Collect Nextflow log
        if hasattr(job, 'log_file') and job.log_file:
            log_path = Path(job.log_file)
            if log_path.exists():
                logs.nextflow_log = self._read_tail(log_path, self.max_log_size)
                logs.work_directory = self._extract_work_dir(logs.nextflow_log)
                logs.failed_process = self._extract_failed_process(logs.nextflow_log)
        
        # Collect SLURM logs
        if hasattr(job, 'output_dir') and job.output_dir:
            output_dir = Path(job.output_dir)
            
            # Find .err files
            err_files = list(output_dir.glob("*.err")) + list(output_dir.glob("slurm*.err"))
            if err_files:
                logs.slurm_err = self._read_tail(err_files[0], self.max_log_size)
            
            # Find .out files
            out_files = list(output_dir.glob("*.out")) + list(output_dir.glob("slurm*.out"))
            if out_files:
                logs.slurm_out = self._read_tail(out_files[0], self.max_log_size)
        
        # Collect work directory logs (Nextflow process-specific)
        if logs.work_directory:
            work_path = Path(logs.work_directory)
            command_err = work_path / ".command.err"
            command_out = work_path / ".command.out"
            
            if command_err.exists():
                logs.command_err = self._read_tail(command_err, self.max_log_size)
            if command_out.exists():
                logs.command_out = self._read_tail(command_out, self.max_log_size)
        
        # Collect workflow file
        if hasattr(job, 'workflow_file') and job.workflow_file:
            wf_path = Path(job.workflow_file)
            if wf_path.exists():
                logs.workflow_file = wf_path.read_text()[:self.max_log_size]
        
        return logs
    
    def _read_tail(self, path: Path, max_bytes: int) -> str:
        """Read last N bytes of a file."""
        try:
            with open(path, 'r', errors='replace') as f:
                f.seek(0, 2)  # End of file
                size = f.tell()
                f.seek(max(0, size - max_bytes))
                return f.read()
        except Exception as e:
            return f"Error reading {path}: {e}"
    
    def _extract_work_dir(self, log_content: str) -> Optional[str]:
        """Extract work directory from Nextflow log."""
        # Pattern: work_dir: /path/to/work/xx/xxxxxx
        match = re.search(r'work[-_]?dir:\s*(\S+)', log_content, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Alternative: look for [hash] pattern
        match = re.search(r'\[([a-f0-9]{2}/[a-f0-9]+)\]', log_content)
        if match:
            # Construct work path
            return f"work/{match.group(1)}"
        
        return None
    
    def _extract_failed_process(self, log_content: str) -> Optional[str]:
        """Extract the name of the failed process."""
        # Pattern: Error executing process > 'PROCESS_NAME'
        match = re.search(r"Error executing process > '?(\w+)'?", log_content)
        if match:
            return match.group(1)
        return None
```

### 4. Main Agent (`agent.py`)

```python
import re
from typing import Optional, List
from dataclasses import dataclass
import logging

from .categories import (
    ErrorCategory, ErrorDiagnosis, FixSuggestion, 
    FixRiskLevel
)
from .patterns import ERROR_PATTERNS
from .log_collector import LogCollector, CollectedLogs
from .prompts import DIAGNOSIS_PROMPT
from ..llm import get_llm, check_providers, Message

logger = logging.getLogger(__name__)


class ErrorDiagnosisAgent:
    """
    AI-powered error diagnosis agent for bioinformatics workflows.
    
    Uses a tiered approach:
    1. Pattern matching (fast, offline)
    2. LLM analysis (comprehensive, contextual)
    """
    
    def __init__(self, llm=None, provider_priority: List[str] = None):
        """
        Initialize the diagnosis agent.
        
        Args:
            llm: Pre-configured LLM adapter (optional)
            provider_priority: Ordered list of LLM providers to try
        """
        self.llm = llm
        self.provider_priority = provider_priority or [
            "lightning",  # Free tier
            "gemini",     # Free tier
            "ollama",     # Local/free
            "openai",     # Paid backup
        ]
        self.log_collector = LogCollector()
    
    async def diagnose(self, job) -> ErrorDiagnosis:
        """
        Perform full error diagnosis on a failed job.
        
        Args:
            job: PipelineJob object with failure information
            
        Returns:
            ErrorDiagnosis with root cause and fix suggestions
        """
        # Step 1: Collect all logs
        logs = self.log_collector.collect(job)
        
        # Step 2: Try pattern matching first (fast)
        pattern_result = self._match_patterns(logs)
        if pattern_result and pattern_result.confidence > 0.8:
            logger.info(f"Pattern match found: {pattern_result.category}")
            return pattern_result
        
        # Step 3: Use LLM for complex errors
        llm = self._get_available_llm()
        if llm:
            try:
                llm_result = await self._llm_diagnosis(logs, job, llm)
                if llm_result:
                    return llm_result
            except Exception as e:
                logger.warning(f"LLM diagnosis failed: {e}")
        
        # Step 4: Return pattern result or unknown
        if pattern_result:
            return pattern_result
        
        return ErrorDiagnosis(
            category=ErrorCategory.UNKNOWN,
            confidence=0.0,
            root_cause="Unable to determine root cause",
            user_explanation="The error could not be automatically diagnosed. Please check the logs manually.",
            log_excerpt=logs.get_combined_error_context(50),
            suggested_fixes=[
                FixSuggestion(
                    description="Check Nextflow log for details",
                    command=f"cat {job.log_file}",
                    risk_level=FixRiskLevel.SAFE,
                    auto_executable=True,
                )
            ],
        )
    
    def _match_patterns(self, logs: CollectedLogs) -> Optional[ErrorDiagnosis]:
        """Match error patterns against log content."""
        combined_logs = f"{logs.nextflow_log}\n{logs.slurm_err}\n{logs.command_err}"
        
        best_match = None
        best_confidence = 0.0
        matched_text = ""
        
        for category, pattern_def in ERROR_PATTERNS.items():
            for pattern in pattern_def.patterns:
                matches = re.findall(pattern, combined_logs, re.IGNORECASE)
                if matches:
                    # Calculate confidence based on number of matches
                    confidence = min(0.5 + (len(matches) * 0.1), 0.95)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = pattern_def
                        matched_text = matches[0] if isinstance(matches[0], str) else str(matches[0])
        
        if best_match:
            return ErrorDiagnosis(
                category=best_match.category,
                confidence=best_confidence,
                root_cause=f"{best_match.description}: {matched_text}",
                user_explanation=best_match.description,
                log_excerpt=matched_text[:500],
                suggested_fixes=best_match.suggested_fixes,
                pattern_matched=True,
            )
        
        return None
    
    def _get_available_llm(self):
        """Get the first available LLM from priority list."""
        if self.llm:
            return self.llm
        
        available = check_providers()
        for provider in self.provider_priority:
            if available.get(provider):
                try:
                    return get_llm(provider)
                except Exception as e:
                    logger.debug(f"Failed to initialize {provider}: {e}")
                    continue
        
        return None
    
    async def _llm_diagnosis(self, logs: CollectedLogs, job, llm) -> Optional[ErrorDiagnosis]:
        """Use LLM for deep error analysis."""
        # Build context
        error_context = logs.get_combined_error_context()
        
        prompt = DIAGNOSIS_PROMPT.format(
            workflow_name=getattr(job, 'name', 'Unknown'),
            analysis_type=getattr(job, 'analysis_type', 'Unknown'),
            failed_process=logs.failed_process or 'Unknown',
            nextflow_log=logs.nextflow_log[-3000:] if logs.nextflow_log else "N/A",
            slurm_err=logs.slurm_err[-2000:] if logs.slurm_err else "N/A",
            workflow_config=logs.config_file[:2000] if logs.config_file else "N/A",
            error_categories=", ".join([c.value for c in ErrorCategory]),
        )
        
        messages = [
            Message.system("You are an expert bioinformatics pipeline debugger."),
            Message.user(prompt),
        ]
        
        response = llm.chat(messages)
        
        # Parse structured response
        return self._parse_llm_response(response.content, llm.__class__.__name__)
    
    def _parse_llm_response(self, content: str, provider: str) -> Optional[ErrorDiagnosis]:
        """Parse LLM response into ErrorDiagnosis."""
        import json
        
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            return None
        
        try:
            data = json.loads(json_match.group())
            
            # Map category string to enum
            category_str = data.get('error_category', 'unknown')
            try:
                category = ErrorCategory(category_str)
            except ValueError:
                category = ErrorCategory.UNKNOWN
            
            # Parse fixes
            fixes = []
            for fix_data in data.get('suggested_fixes', []):
                risk_str = fix_data.get('risk_level', 'medium')
                try:
                    risk = FixRiskLevel(risk_str)
                except ValueError:
                    risk = FixRiskLevel.MEDIUM
                
                fixes.append(FixSuggestion(
                    description=fix_data.get('description', ''),
                    command=fix_data.get('command'),
                    risk_level=risk,
                    auto_executable=fix_data.get('auto_executable', False),
                ))
            
            return ErrorDiagnosis(
                category=category,
                confidence=float(data.get('confidence', 0.7)),
                root_cause=data.get('root_cause', ''),
                user_explanation=data.get('user_explanation', ''),
                log_excerpt=data.get('log_excerpt', '')[:500],
                suggested_fixes=fixes,
                llm_provider_used=provider,
                pattern_matched=False,
            )
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return None
```

### 5. LLM Prompts (`prompts.py`)

```python
DIAGNOSIS_PROMPT = """Analyze this bioinformatics pipeline failure and provide a diagnosis.

## Error Context

**Workflow:** {workflow_name}
**Analysis Type:** {analysis_type}
**Failed Process:** {failed_process}

## Log Files

### Nextflow Log (last section):
```
{nextflow_log}
```

### SLURM Error File:
```
{slurm_err}
```

### Workflow Configuration:
```
{workflow_config}
```

## Your Task

1. **Identify the root cause** of this failure
2. **Classify the error** into one of: {error_categories}
3. **Explain** what went wrong in simple terms (for non-experts)
4. **Suggest fixes** ranked by likelihood of success
5. **Provide shell commands** to fix if applicable

## Required Output Format (JSON)

```json
{{
  "error_category": "category_name",
  "root_cause": "Brief technical explanation",
  "user_explanation": "Simple explanation for non-technical users",
  "confidence": 0.85,
  "log_excerpt": "Key error line from logs",
  "suggested_fixes": [
    {{
      "description": "What to do",
      "command": "shell command if applicable (or null)",
      "auto_executable": true,
      "risk_level": "safe"
    }},
    {{
      "description": "Alternative fix",
      "command": null,
      "auto_executable": false,
      "risk_level": "medium"
    }}
  ]
}}
```

Only output the JSON, no additional text."""


GITHUB_COPILOT_FIX_PROMPT = """## Bug Fix Request

A bioinformatics Nextflow workflow failed with the following error:

**Error Category:** {category}
**Root Cause:** {root_cause}

### Error Log
```
{log_excerpt}
```

### Workflow File: {workflow_file}
```nextflow
{workflow_content}
```

### Task

1. Identify the bug in the workflow code
2. Fix the issue
3. Add appropriate error handling
4. Ensure the fix follows Nextflow DSL2 best practices

Please create a pull request with the fix.
"""
```

---

## Integration with Gradio UI

### Add Diagnose Button to Jobs Table

```python
# In gradio_app.py

from workflow_composer.diagnosis import ErrorDiagnosisAgent, ErrorDiagnosis

async def diagnose_job(job_id: str) -> str:
    """Diagnose a failed job and return formatted results."""
    job = pipeline_executor.get_job_status(job_id)
    if not job:
        return "❌ Job not found"
    
    if job.status != JobStatus.FAILED:
        return "ℹ️ Only failed jobs can be diagnosed"
    
    agent = ErrorDiagnosisAgent(llm=app_state.composer.llm if app_state.composer else None)
    
    try:
        diagnosis = await agent.diagnose(job)
        return format_diagnosis(diagnosis)
    except Exception as e:
        return f"❌ Diagnosis failed: {e}"

def format_diagnosis(d: ErrorDiagnosis) -> str:
    """Format diagnosis for display."""
    risk_icons = {
        "safe": "🟢",
        "low": "🟡", 
        "medium": "🟠",
        "high": "🔴",
    }
    
    fixes_md = ""
    for i, fix in enumerate(d.suggested_fixes, 1):
        icon = risk_icons.get(fix.risk_level.value, "⚪")
        cmd = f"\n   ```bash\n   {fix.command}\n   ```" if fix.command else ""
        auto = " *(auto-executable)*" if fix.auto_executable else ""
        fixes_md += f"\n{i}. {icon} **{fix.description}**{auto}{cmd}\n"
    
    return f"""## 🔍 Error Diagnosis

**Category:** {d.category.value.replace('_', ' ').title()}
**Confidence:** {d.confidence:.0%}
{"**Provider:** " + d.llm_provider_used if d.llm_provider_used else "**Method:** Pattern Matching"}

### Root Cause
{d.root_cause}

### Explanation
{d.user_explanation}

### Log Excerpt
```
{d.log_excerpt[:500]}
```

### Suggested Fixes
{fixes_md}
"""
```

---

## Testing Plan

### Unit Tests

```python
# tests/test_diagnosis.py
import pytest
from workflow_composer.diagnosis import (
    ErrorDiagnosisAgent, 
    LogCollector,
    ERROR_PATTERNS,
    ErrorCategory,
)

class TestPatternMatching:
    def test_file_not_found_pattern(self):
        """Test file not found error detection."""
        log_content = "Error: No such file or directory: /data/reference/genome.fa"
        agent = ErrorDiagnosisAgent()
        logs = MockLogs(nextflow_log=log_content)
        result = agent._match_patterns(logs)
        
        assert result is not None
        assert result.category == ErrorCategory.FILE_NOT_FOUND
        assert result.confidence >= 0.5
    
    def test_oom_pattern(self):
        """Test out of memory error detection."""
        log_content = "slurmstepd: error: Detected 1 oom-kill event"
        agent = ErrorDiagnosisAgent()
        logs = MockLogs(slurm_err=log_content)
        result = agent._match_patterns(logs)
        
        assert result is not None
        assert result.category == ErrorCategory.OUT_OF_MEMORY
    
    def test_container_error_pattern(self):
        """Test container error detection."""
        log_content = "FATAL: container creation failed"
        agent = ErrorDiagnosisAgent()
        logs = MockLogs(nextflow_log=log_content)
        result = agent._match_patterns(logs)
        
        assert result is not None
        assert result.category == ErrorCategory.CONTAINER_ERROR


class TestLogCollector:
    def test_collect_logs(self, tmp_path):
        """Test log file collection."""
        # Create mock log files
        log_file = tmp_path / "test.log"
        log_file.write_text("Error executing process > 'ALIGN'")
        
        collector = LogCollector()
        job = MockJob(log_file=str(log_file))
        logs = collector.collect(job)
        
        assert "Error executing process" in logs.nextflow_log
        assert logs.failed_process == "ALIGN"
```

---

## Next Steps

1. ✅ Create `src/workflow_composer/diagnosis/` package
2. ✅ Implement pattern database
3. ✅ Implement log collector
4. ✅ Implement main agent
5. 🔲 Add Gemini adapter
6. 🔲 Integrate with Gradio UI
7. 🔲 Add auto-fix execution
8. 🔲 Add GitHub Copilot integration
9. 🔲 Write tests
10. 🔲 Documentation

---

**Document Version:** 1.0  
**Last Updated:** November 25, 2025
