"""
Job Status Panel Component
==========================

A side panel showing active SLURM jobs with real-time status updates,
log viewing, and quick actions (cancel, resubmit).
"""

import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import gradio as gr

logger = logging.getLogger(__name__)


# ============================================================================
# Job Status Functions
# ============================================================================

def get_user_jobs() -> List[Dict[str, Any]]:
    """Get current user's SLURM jobs."""
    jobs = []
    
    try:
        result = subprocess.run(
            ["squeue", "-u", subprocess.getoutput("whoami"), 
             "--format=%i|%j|%P|%T|%M|%l|%D|%R", "--noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                parts = line.split("|")
                if len(parts) >= 7:
                    jobs.append({
                        "job_id": parts[0].strip(),
                        "name": parts[1].strip(),
                        "partition": parts[2].strip(),
                        "state": parts[3].strip(),
                        "time": parts[4].strip(),
                        "time_limit": parts[5].strip(),
                        "nodes": parts[6].strip(),
                        "reason": parts[7].strip() if len(parts) > 7 else "",
                    })
    except Exception as e:
        logger.warning(f"Failed to get jobs: {e}")
    
    return jobs


def get_job_log(job_id: str) -> str:
    """Get log content for a job."""
    if not job_id:
        return "No job selected"
    
    # Try common log patterns
    log_patterns = [
        f"/home/*/BioPipelines/data/raw/download_jobs/*{job_id}*.log",
        f"/home/*/BioPipelines/generated_workflows/*/slurm-{job_id}.out",
        f"slurm-{job_id}.out",
    ]
    
    import glob
    for pattern in log_patterns:
        matches = glob.glob(pattern)
        if matches:
            try:
                log_path = Path(matches[0])
                if log_path.exists():
                    content = log_path.read_text()
                    # Return last 100 lines
                    lines = content.split("\n")
                    if len(lines) > 100:
                        return f"... (showing last 100 lines)\n\n" + "\n".join(lines[-100:])
                    return content
            except Exception as e:
                return f"Error reading log: {e}"
    
    # Try sacct for completed jobs
    try:
        result = subprocess.run(
            ["sacct", "-j", job_id, "--format=JobID,State,Start,End,ExitCode,Elapsed", "--noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return f"Job {job_id} info:\n{result.stdout}"
    except Exception:
        pass
    
    return f"No log found for job {job_id}"


def cancel_job(job_id: str) -> str:
    """Cancel a SLURM job."""
    if not job_id:
        return "No job ID provided"
    
    try:
        result = subprocess.run(
            ["scancel", job_id],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return f"✅ Job {job_id} cancelled"
        else:
            return f"❌ Failed to cancel: {result.stderr}"
    except Exception as e:
        return f"❌ Error: {e}"


def get_recent_jobs(days: int = 1) -> List[Dict[str, Any]]:
    """Get recently completed jobs."""
    jobs = []
    
    try:
        result = subprocess.run(
            ["sacct", "--starttime=now-1day", "-u", subprocess.getoutput("whoami"),
             "--format=JobID,JobName,State,Start,End,Elapsed,ExitCode", "--noheader", "-P"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                parts = line.split("|")
                if len(parts) >= 6 and not parts[0].endswith(".batch"):
                    jobs.append({
                        "job_id": parts[0],
                        "name": parts[1],
                        "state": parts[2],
                        "start": parts[3],
                        "end": parts[4],
                        "elapsed": parts[5],
                        "exit_code": parts[6] if len(parts) > 6 else "",
                    })
    except Exception as e:
        logger.warning(f"Failed to get recent jobs: {e}")
    
    return jobs[:20]  # Limit to 20 most recent


def format_jobs_table(jobs: List[Dict]) -> str:
    """Format jobs as HTML table."""
    if not jobs:
        return "<p>No active jobs</p>"
    
    state_icons = {
        "RUNNING": "🟢",
        "PENDING": "🟡",
        "COMPLETED": "✅",
        "FAILED": "❌",
        "CANCELLED": "⚪",
        "CONFIGUR": "🔵",
        "COMPLETING": "🔄",
    }
    
    rows = []
    for job in jobs:
        state = job.get("state", "UNKNOWN")
        icon = state_icons.get(state, "⚪")
        job_id = job.get("job_id", "")
        name = job.get("name", "")[:20]  # Truncate long names
        time = job.get("time", job.get("elapsed", ""))
        
        rows.append(f"""
        <tr>
            <td><code>{job_id}</code></td>
            <td>{name}</td>
            <td>{icon} {state}</td>
            <td>{time}</td>
        </tr>
        """)
    
    return f"""
    <table style="width:100%; font-size:12px; border-collapse:collapse;">
        <thead>
            <tr style="background:#f0f0f0;">
                <th style="padding:4px; text-align:left;">ID</th>
                <th style="padding:4px; text-align:left;">Name</th>
                <th style="padding:4px; text-align:left;">Status</th>
                <th style="padding:4px; text-align:left;">Time</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


def refresh_job_panel() -> tuple:
    """Refresh the job panel data."""
    active_jobs = get_user_jobs()
    recent_jobs = get_recent_jobs()
    
    active_html = format_jobs_table(active_jobs)
    recent_html = format_jobs_table(recent_jobs)
    
    # Build job ID dropdown choices
    job_choices = [j["job_id"] for j in active_jobs + recent_jobs]
    
    return active_html, recent_html, gr.update(choices=job_choices)


# ============================================================================
# Panel Component
# ============================================================================

def create_job_panel() -> gr.Column:
    """Create the job status panel component."""
    
    with gr.Column(scale=1, min_width=300) as panel:
        gr.Markdown("### 📊 Job Status")
        
        # Refresh button
        refresh_btn = gr.Button("🔄 Refresh", size="sm")
        
        # Active jobs
        with gr.Accordion("🟢 Active Jobs", open=True):
            active_jobs_html = gr.HTML(value="<p>Loading...</p>")
        
        # Recent jobs
        with gr.Accordion("📋 Recent Jobs (24h)", open=False):
            recent_jobs_html = gr.HTML(value="<p>Loading...</p>")
        
        # Job details
        with gr.Accordion("📄 Job Details", open=False):
            job_select = gr.Dropdown(
                label="Select Job",
                choices=[],
                interactive=True,
            )
            with gr.Row():
                view_log_btn = gr.Button("View Log", size="sm")
                cancel_btn = gr.Button("Cancel", size="sm", variant="stop")
            
            log_output = gr.Code(
                label="Log Output",
                language="shell",
                lines=15,
            )
            action_result = gr.Markdown("")
        
        # Auto-refresh toggle
        auto_refresh = gr.Checkbox(label="Auto-refresh (30s)", value=True)
        
        # Wire up events
        refresh_btn.click(
            refresh_job_panel,
            outputs=[active_jobs_html, recent_jobs_html, job_select]
        )
        
        view_log_btn.click(
            get_job_log,
            inputs=[job_select],
            outputs=[log_output]
        )
        
        cancel_btn.click(
            cancel_job,
            inputs=[job_select],
            outputs=[action_result]
        ).then(
            refresh_job_panel,
            outputs=[active_jobs_html, recent_jobs_html, job_select]
        )
        
        # Initial load
        panel.load(
            refresh_job_panel,
            outputs=[active_jobs_html, recent_jobs_html, job_select]
        )
    
    return panel


def create_job_tab() -> gr.Tab:
    """Create a full job management tab."""
    
    with gr.Tab("📊 Jobs") as tab:
        gr.Markdown("## Job Management")
        
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh All", variant="primary")
            
        with gr.Row():
            # Active jobs panel
            with gr.Column(scale=1):
                gr.Markdown("### 🟢 Active Jobs")
                active_jobs_html = gr.HTML(value="<p>Click refresh to load...</p>")
            
            # Recent jobs panel
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Recent Jobs (24h)")
                recent_jobs_html = gr.HTML(value="<p>Click refresh to load...</p>")
        
        gr.Markdown("---")
        
        # Job details section
        gr.Markdown("### 📄 Job Details")
        
        with gr.Row():
            job_id_input = gr.Textbox(
                label="Job ID",
                placeholder="Enter job ID...",
                scale=2,
            )
            view_log_btn = gr.Button("View Log", scale=1)
            cancel_btn = gr.Button("Cancel Job", variant="stop", scale=1)
        
        log_output = gr.Code(
            label="Log Output",
            language="shell",
            lines=20,
        )
        
        action_result = gr.Markdown("")
        
        # Wire up events
        def refresh_all():
            active = get_user_jobs()
            recent = get_recent_jobs()
            return format_jobs_table(active), format_jobs_table(recent)
        
        refresh_btn.click(
            refresh_all,
            outputs=[active_jobs_html, recent_jobs_html]
        )
        
        view_log_btn.click(
            get_job_log,
            inputs=[job_id_input],
            outputs=[log_output]
        )
        
        cancel_btn.click(
            cancel_job,
            inputs=[job_id_input],
            outputs=[action_result]
        ).then(
            refresh_all,
            outputs=[active_jobs_html, recent_jobs_html]
        )
    
    return tab
