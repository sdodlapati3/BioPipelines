"""
Workflow Monitor
================

Monitor Nextflow workflow execution status and collect metrics.

Features:
- Track running workflows
- Collect execution metrics
- Parse Nextflow logs
- Generate reports
"""

import os
import re
import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessStatus(Enum):
    """Process execution status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"


@dataclass
class ProcessExecution:
    """Single process execution record."""
    name: str
    task_id: str
    status: ProcessStatus
    hash: str = ""
    native_id: str = ""  # SLURM job ID
    submit_time: Optional[datetime] = None
    start_time: Optional[datetime] = None
    complete_time: Optional[datetime] = None
    exit_code: Optional[int] = None
    cpus: int = 1
    memory: str = ""
    duration: float = 0.0
    realtime: float = 0.0
    peak_rss: str = ""
    peak_vmem: str = ""
    read_bytes: int = 0
    write_bytes: int = 0
    workdir: str = ""


@dataclass
class WorkflowExecution:
    """Workflow execution record."""
    workflow_id: str
    name: str
    status: WorkflowStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    workdir: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    processes: List[ProcessExecution] = field(default_factory=list)
    error_message: str = ""
    
    @property
    def duration(self) -> Optional[float]:
        """Get workflow duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def process_counts(self) -> Dict[str, int]:
        """Count processes by status."""
        counts = {s.value: 0 for s in ProcessStatus}
        for p in self.processes:
            counts[p.status.value] += 1
        return counts
    
    @property
    def progress(self) -> float:
        """Get completion progress (0-100)."""
        if not self.processes:
            return 0.0
        completed = sum(1 for p in self.processes 
                       if p.status in [ProcessStatus.COMPLETED, ProcessStatus.CACHED])
        return (completed / len(self.processes)) * 100


class WorkflowMonitor:
    """
    Monitor Nextflow workflow execution.
    
    Supports:
    - Parsing .nextflow.log files
    - Reading trace files
    - Monitoring work directories
    - Collecting metrics from SLURM
    """
    
    def __init__(self, work_dir: str = None):
        """
        Initialize workflow monitor.
        
        Args:
            work_dir: Base work directory to monitor
        """
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()
        self.executions: Dict[str, WorkflowExecution] = {}
    
    def scan_workflow(self, workflow_dir: str) -> Optional[WorkflowExecution]:
        """
        Scan a workflow directory for execution info.
        
        Args:
            workflow_dir: Path to workflow directory
            
        Returns:
            WorkflowExecution if found
        """
        workflow_path = Path(workflow_dir)
        
        # Look for .nextflow.log
        log_file = workflow_path / ".nextflow.log"
        if log_file.exists():
            return self._parse_nextflow_log(log_file)
        
        # Look for trace file
        trace_file = list(workflow_path.glob("*trace*.txt"))
        if trace_file:
            return self._parse_trace_file(trace_file[0])
        
        return None
    
    def _parse_nextflow_log(self, log_path: Path) -> WorkflowExecution:
        """Parse .nextflow.log file."""
        content = log_path.read_text()
        
        # Extract workflow ID
        workflow_id = "unknown"
        match = re.search(r"Session ID: ([a-f0-9-]+)", content)
        if match:
            workflow_id = match.group(1)
        
        # Extract workflow name
        name = log_path.parent.name
        match = re.search(r"Launching `([^`]+)`", content)
        if match:
            name = match.group(1)
        
        # Determine status
        status = WorkflowStatus.RUNNING
        if "Execution complete" in content:
            status = WorkflowStatus.COMPLETED
        elif "Error executing" in content or "Pipeline failed" in content:
            status = WorkflowStatus.FAILED
        
        # Extract start time
        start_time = datetime.now()
        match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", content)
        if match:
            start_time = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
        
        # Extract end time
        end_time = None
        if status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
            matches = re.findall(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", content)
            if matches:
                end_time = datetime.strptime(matches[-1], "%Y-%m-%d %H:%M:%S")
        
        # Extract processes
        processes = self._extract_processes_from_log(content)
        
        # Extract error message
        error_message = ""
        if status == WorkflowStatus.FAILED:
            match = re.search(r"Error executing process.*?(\n.*?)+?(?=\n\n|\Z)", 
                            content, re.DOTALL)
            if match:
                error_message = match.group(0)[:500]
        
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            name=name,
            status=status,
            start_time=start_time,
            end_time=end_time,
            workdir=str(log_path.parent),
            processes=processes,
            error_message=error_message
        )
        
        self.executions[workflow_id] = execution
        return execution
    
    def _extract_processes_from_log(self, content: str) -> List[ProcessExecution]:
        """Extract process executions from log content."""
        processes = []
        
        # Pattern for process submissions
        pattern = r"\[([a-f0-9]+)\] process > (\w+) \(([^)]+)\) \[(.+?)\]"
        
        for match in re.finditer(pattern, content):
            hash_val = match.group(1)
            process_name = match.group(2)
            task_info = match.group(3)
            status_str = match.group(4).strip()
            
            # Map status string
            status_map = {
                "100%": ProcessStatus.COMPLETED,
                "COMPLETED": ProcessStatus.COMPLETED,
                "FAILED": ProcessStatus.FAILED,
                "CACHED": ProcessStatus.CACHED,
                "RUNNING": ProcessStatus.RUNNING,
            }
            status = status_map.get(status_str, ProcessStatus.PENDING)
            
            processes.append(ProcessExecution(
                name=process_name,
                task_id=task_info,
                status=status,
                hash=hash_val
            ))
        
        return processes
    
    def _parse_trace_file(self, trace_path: Path) -> WorkflowExecution:
        """Parse Nextflow trace file."""
        import csv
        
        processes = []
        workflow_id = trace_path.stem
        
        with open(trace_path) as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            for row in reader:
                status_map = {
                    'COMPLETED': ProcessStatus.COMPLETED,
                    'FAILED': ProcessStatus.FAILED,
                    'CACHED': ProcessStatus.CACHED,
                    'RUNNING': ProcessStatus.RUNNING,
                }
                
                processes.append(ProcessExecution(
                    name=row.get('name', ''),
                    task_id=row.get('task_id', ''),
                    status=status_map.get(row.get('status', ''), ProcessStatus.PENDING),
                    hash=row.get('hash', ''),
                    native_id=row.get('native_id', ''),
                    exit_code=int(row.get('exit', -1)) if row.get('exit') else None,
                    cpus=int(row.get('cpus', 1)),
                    memory=row.get('memory', ''),
                    duration=float(row.get('duration', 0)) / 1000,  # ms to seconds
                    realtime=float(row.get('realtime', 0)) / 1000,
                    peak_rss=row.get('peak_rss', ''),
                    peak_vmem=row.get('peak_vmem', ''),
                    read_bytes=int(row.get('rchar', 0)),
                    write_bytes=int(row.get('wchar', 0)),
                    workdir=row.get('workdir', '')
                ))
        
        # Determine overall status
        has_failed = any(p.status == ProcessStatus.FAILED for p in processes)
        all_complete = all(p.status in [ProcessStatus.COMPLETED, ProcessStatus.CACHED] 
                         for p in processes)
        
        if has_failed:
            status = WorkflowStatus.FAILED
        elif all_complete:
            status = WorkflowStatus.COMPLETED
        else:
            status = WorkflowStatus.RUNNING
        
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            name=trace_path.parent.name,
            status=status,
            start_time=datetime.now(),  # Would need to parse from trace
            processes=processes,
            workdir=str(trace_path.parent)
        )
        
        self.executions[workflow_id] = execution
        return execution
    
    def get_execution(self, workflow_id: str) -> Optional[WorkflowExecution]:
        """Get execution by ID."""
        return self.executions.get(workflow_id)
    
    def list_executions(self) -> List[WorkflowExecution]:
        """List all tracked executions."""
        return list(self.executions.values())
    
    def generate_report(self, execution: WorkflowExecution) -> Dict[str, Any]:
        """
        Generate execution report.
        
        Args:
            execution: WorkflowExecution to report on
            
        Returns:
            Dict with report data
        """
        # Calculate metrics
        total_duration = execution.duration or 0
        total_cpu_time = sum(p.realtime * p.cpus for p in execution.processes)
        total_read = sum(p.read_bytes for p in execution.processes)
        total_write = sum(p.write_bytes for p in execution.processes)
        
        # Process stats
        process_stats = {}
        for p in execution.processes:
            if p.name not in process_stats:
                process_stats[p.name] = {
                    'count': 0,
                    'completed': 0,
                    'failed': 0,
                    'cached': 0,
                    'total_duration': 0
                }
            stats = process_stats[p.name]
            stats['count'] += 1
            stats['total_duration'] += p.duration
            if p.status == ProcessStatus.COMPLETED:
                stats['completed'] += 1
            elif p.status == ProcessStatus.FAILED:
                stats['failed'] += 1
            elif p.status == ProcessStatus.CACHED:
                stats['cached'] += 1
        
        return {
            'workflow_id': execution.workflow_id,
            'name': execution.name,
            'status': execution.status.value,
            'start_time': execution.start_time.isoformat(),
            'end_time': execution.end_time.isoformat() if execution.end_time else None,
            'duration_seconds': total_duration,
            'progress': execution.progress,
            'process_counts': execution.process_counts,
            'total_processes': len(execution.processes),
            'total_cpu_time': total_cpu_time,
            'total_read_bytes': total_read,
            'total_write_bytes': total_write,
            'process_stats': process_stats,
            'error_message': execution.error_message
        }
    
    def export_to_json(self, output_path: str) -> None:
        """Export all executions to JSON."""
        data = {
            wid: self.generate_report(ex) 
            for wid, ex in self.executions.items()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported {len(data)} executions to {output_path}")


def monitor_slurm_jobs(job_ids: List[str]) -> Dict[str, Dict]:
    """
    Monitor SLURM jobs status.
    
    Args:
        job_ids: List of SLURM job IDs
        
    Returns:
        Dict mapping job IDs to status info
    """
    import subprocess
    
    results = {}
    
    for job_id in job_ids:
        try:
            output = subprocess.check_output(
                ['sacct', '-j', job_id, '--format=JobID,State,ExitCode,Elapsed,MaxRSS', 
                 '-n', '-P'],
                text=True
            )
            
            lines = output.strip().split('\n')
            if lines:
                parts = lines[0].split('|')
                results[job_id] = {
                    'job_id': parts[0] if len(parts) > 0 else '',
                    'state': parts[1] if len(parts) > 1 else '',
                    'exit_code': parts[2] if len(parts) > 2 else '',
                    'elapsed': parts[3] if len(parts) > 3 else '',
                    'max_rss': parts[4] if len(parts) > 4 else ''
                }
        except Exception as e:
            results[job_id] = {'error': str(e)}
    
    return results
