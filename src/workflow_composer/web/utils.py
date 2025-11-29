"""
Shared utilities for the web module.
"""

import os
import subprocess
from typing import Optional


def detect_vllm_endpoint() -> str:
    """
    Dynamically detect vLLM endpoint from running SLURM jobs.
    
    Checks for running vLLM/BioPipelines jobs and extracts the node URL.
    Falls back to VLLM_URL environment variable or localhost.
    
    Returns:
        vLLM API endpoint URL
    """
    try:
        result = subprocess.run(
            ["squeue", "--me", "-h", "-o", "%i %j %T %N"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.strip().split('\n'):
            if line and ('vllm' in line.lower() or 'biopipelines' in line.lower()) and 'RUNNING' in line:
                parts = line.split()
                if len(parts) >= 4:
                    node = parts[3]
                    return f"http://{node}:8000/v1"
    except Exception:
        pass
    return os.environ.get("VLLM_URL", "http://localhost:8000/v1")


def get_default_port() -> int:
    """Get the default Gradio port from environment."""
    return int(os.environ.get("GRADIO_PORT", "7860"))


def use_local_llm() -> bool:
    """Check if local LLM should be used."""
    return os.environ.get("USE_LOCAL_LLM", "false").lower() == "true"
