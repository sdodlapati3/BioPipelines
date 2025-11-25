"""
Log collector for aggregating error information from multiple sources.

Collects logs from:
- Nextflow log (.nextflow.log)
- SLURM error files (.err)
- Process-specific logs (.command.err, .command.out)
- Workflow and configuration files
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class CollectedLogs:
    """Aggregated log content from various sources."""
    nextflow_log: str = ""
    slurm_err: str = ""
    slurm_out: str = ""
    command_err: str = ""
    command_out: str = ""
    command_script: str = ""
    workflow_file: str = ""
    config_file: str = ""
    failed_process: Optional[str] = None
    work_directory: Optional[str] = None
    exit_code: Optional[int] = None
    
    def get_combined_error_context(self, max_lines: int = 100) -> str:
        """
        Get combined error context for LLM analysis.
        
        Extracts the most relevant error information from all log sources.
        """
        sections = []
        
        # Extract error lines from Nextflow log
        if self.nextflow_log:
            error_keywords = ['error', 'fail', 'exception', 'fatal', 'abort', 'halt']
            lines = self.nextflow_log.split('\n')
            error_lines = [
                l for l in lines 
                if any(kw in l.lower() for kw in error_keywords)
            ]
            if error_lines:
                sections.append(
                    f"### Nextflow Errors:\n```\n{chr(10).join(error_lines[-30:])}\n```"
                )
        
        # Include SLURM error
        if self.slurm_err:
            # Get last portion of error file
            slurm_excerpt = self.slurm_err[-2000:].strip()
            if slurm_excerpt:
                sections.append(f"### SLURM Error:\n```\n{slurm_excerpt}\n```")
        
        # Include command-specific errors
        if self.command_err:
            cmd_excerpt = self.command_err[-2000:].strip()
            if cmd_excerpt:
                sections.append(f"### Process Error (.command.err):\n```\n{cmd_excerpt}\n```")
        
        # Include the command that was run
        if self.command_script:
            sections.append(f"### Command Script:\n```bash\n{self.command_script[-1000:]}\n```")
        
        return "\n\n".join(sections) if sections else "No error context available"
    
    def get_full_log_text(self) -> str:
        """Get all logs combined as plain text."""
        return f"{self.nextflow_log}\n{self.slurm_err}\n{self.command_err}".strip()
    
    def has_errors(self) -> bool:
        """Check if any error content was collected."""
        return bool(self.nextflow_log or self.slurm_err or self.command_err)


class LogCollector:
    """
    Collects and aggregates log files for error analysis.
    
    Searches multiple locations for relevant log files and extracts
    key error information for diagnosis.
    """
    
    def __init__(self, max_log_size: int = 50000):
        """
        Initialize log collector.
        
        Args:
            max_log_size: Maximum bytes to read from each log file
        """
        self.max_log_size = max_log_size
    
    def collect(self, job) -> CollectedLogs:
        """
        Collect all relevant logs for a job.
        
        Args:
            job: PipelineJob or similar object with job metadata
            
        Returns:
            CollectedLogs with aggregated log content
        """
        logs = CollectedLogs()
        
        # 1. Collect Nextflow log
        log_file = self._get_attr(job, 'log_file')
        if log_file:
            log_path = Path(log_file)
            if log_path.exists():
                logs.nextflow_log = self._read_tail(log_path)
                logs.work_directory = self._extract_work_dir(logs.nextflow_log)
                logs.failed_process = self._extract_failed_process(logs.nextflow_log)
                logs.exit_code = self._extract_exit_code(logs.nextflow_log)
        
        # 2. Try .nextflow.log in output directory
        output_dir = self._get_attr(job, 'output_dir')
        if output_dir and not logs.nextflow_log:
            nf_log = Path(output_dir) / ".nextflow.log"
            if nf_log.exists():
                logs.nextflow_log = self._read_tail(nf_log)
                logs.work_directory = self._extract_work_dir(logs.nextflow_log)
                logs.failed_process = self._extract_failed_process(logs.nextflow_log)
        
        # 3. Collect SLURM logs
        if output_dir:
            output_path = Path(output_dir)
            
            # Find .err files
            err_patterns = ["*.err", "slurm*.err", "slurm_*.err"]
            for pattern in err_patterns:
                err_files = list(output_path.glob(pattern))
                if err_files:
                    logs.slurm_err = self._read_tail(err_files[0])
                    break
            
            # Find .out files
            out_patterns = ["*.out", "slurm*.out", "slurm_*.out"]
            for pattern in out_patterns:
                out_files = list(output_path.glob(pattern))
                if out_files:
                    logs.slurm_out = self._read_tail(out_files[0])
                    break
        
        # 4. Collect from logs directory
        logs_dir = self._get_attr(job, 'logs_dir') or (
            Path(output_dir).parent / "logs" if output_dir else None
        )
        if logs_dir and Path(logs_dir).exists():
            job_id = self._get_attr(job, 'slurm_job_id')
            if job_id:
                # Look for SLURM log with job ID
                for pattern in [f"*{job_id}*.err", f"*{job_id}*.out"]:
                    files = list(Path(logs_dir).glob(pattern))
                    if files:
                        content = self._read_tail(files[0])
                        if pattern.endswith(".err"):
                            logs.slurm_err = logs.slurm_err or content
                        else:
                            logs.slurm_out = logs.slurm_out or content
        
        # 5. Collect work directory logs (Nextflow process-specific)
        if logs.work_directory:
            work_path = Path(logs.work_directory)
            if not work_path.is_absolute() and output_dir:
                work_path = Path(output_dir) / logs.work_directory
            
            if work_path.exists():
                command_err = work_path / ".command.err"
                command_out = work_path / ".command.out"
                command_sh = work_path / ".command.sh"
                
                if command_err.exists():
                    logs.command_err = self._read_tail(command_err)
                if command_out.exists():
                    logs.command_out = self._read_tail(command_out)
                if command_sh.exists():
                    logs.command_script = self._read_tail(command_sh, 5000)
        
        # 6. Collect workflow file
        workflow_file = self._get_attr(job, 'workflow_file')
        if workflow_file:
            wf_path = Path(workflow_file)
            if wf_path.exists():
                logs.workflow_file = self._read_full(wf_path, self.max_log_size)
        
        # 7. Collect config file
        if output_dir:
            config_path = Path(output_dir) / "nextflow.config"
            if config_path.exists():
                logs.config_file = self._read_full(config_path, 10000)
        
        return logs
    
    def collect_from_path(self, path: str) -> CollectedLogs:
        """
        Collect logs from a directory path.
        
        Args:
            path: Path to workflow output directory
            
        Returns:
            CollectedLogs with aggregated content
        """
        @dataclass
        class MockJob:
            output_dir: str = path
            log_file: Optional[str] = None
            
        return self.collect(MockJob(output_dir=path))
    
    def _get_attr(self, obj, attr: str, default=None):
        """Safely get attribute from object."""
        return getattr(obj, attr, default) if obj else default
    
    def _read_tail(self, path: Path, max_bytes: int = None) -> str:
        """
        Read last N bytes of a file.
        
        Args:
            path: Path to file
            max_bytes: Maximum bytes to read (defaults to self.max_log_size)
            
        Returns:
            File content as string
        """
        max_bytes = max_bytes or self.max_log_size
        try:
            with open(path, 'r', errors='replace') as f:
                f.seek(0, 2)  # End of file
                size = f.tell()
                f.seek(max(0, size - max_bytes))
                content = f.read()
                # If we started mid-file, skip to first newline
                if size > max_bytes:
                    newline_pos = content.find('\n')
                    if newline_pos > 0:
                        content = content[newline_pos + 1:]
                return content
        except Exception as e:
            logger.warning(f"Error reading {path}: {e}")
            return ""
    
    def _read_full(self, path: Path, max_bytes: int = None) -> str:
        """
        Read full file content (up to max_bytes).
        
        Args:
            path: Path to file
            max_bytes: Maximum bytes to read
            
        Returns:
            File content as string
        """
        max_bytes = max_bytes or self.max_log_size
        try:
            content = path.read_text(errors='replace')
            return content[:max_bytes]
        except Exception as e:
            logger.warning(f"Error reading {path}: {e}")
            return ""
    
    def _extract_work_dir(self, log_content: str) -> Optional[str]:
        """
        Extract work directory from Nextflow log.
        
        Looks for patterns indicating the failing process work directory.
        """
        if not log_content:
            return None
        
        # Pattern 1: Direct work_dir reference
        match = re.search(r'work[-_]?dir[:\s]+(\S+)', log_content, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Pattern 2: Error executing process with hash
        match = re.search(
            r"Error executing process.*\[([a-f0-9]{2}/[a-f0-9]+)\]", 
            log_content
        )
        if match:
            return f"work/{match.group(1)}"
        
        # Pattern 3: Just the hash in brackets
        match = re.search(r'\[([a-f0-9]{2}/[a-f0-9]+)\]', log_content)
        if match:
            return f"work/{match.group(1)}"
        
        return None
    
    def _extract_failed_process(self, log_content: str) -> Optional[str]:
        """
        Extract the name of the failed process.
        
        Args:
            log_content: Nextflow log content
            
        Returns:
            Process name or None
        """
        if not log_content:
            return None
        
        # Pattern: Error executing process > 'PROCESS_NAME (sample)'
        patterns = [
            r"Error executing process > '?(\w+)'?",
            r"process > (\w+).*\[.*FAILED\]",
            r"Failed to execute process (\w+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, log_content)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_exit_code(self, log_content: str) -> Optional[int]:
        """
        Extract exit code from log content.
        
        Args:
            log_content: Log content to search
            
        Returns:
            Exit code or None
        """
        if not log_content:
            return None
        
        patterns = [
            r"exit status[:\s]+(\d+)",
            r"Exit status: (\d+)",
            r"returned (\d+)",
            r"exited with code (\d+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, log_content, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        
        return None
