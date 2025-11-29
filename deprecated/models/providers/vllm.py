"""
vLLM Local Provider Implementation.

vLLM provides high-performance local inference for open-source models.
This provider connects to a local vLLM server running on the cluster GPUs.
"""

import os
import time
import aiohttp
import subprocess
from typing import Optional, Dict, Any, List

from .base import BaseProvider
from ..registry import ProviderConfig, get_registry


class VLLMProvider(BaseProvider):
    """
    Provider for local vLLM server.
    
    Connects to a vLLM OpenAI-compatible server running on local GPUs.
    Can optionally start the server if not running.
    """
    
    DEFAULT_PORT = 8000
    DEFAULT_HOST = "localhost"
    
    # Available models (configured in registry)
    RECOMMENDED_MODELS = {
        "qwen-coder-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "deepseek-coder-v2": "deepseek-ai/DeepSeek-Coder-V2-Instruct",
        "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
        "minimax-m2": "MiniMaxAI/MiniMax-M2",
        "codellama-34b": "codellama/CodeLlama-34b-Instruct-hf",
    }
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.host = self.DEFAULT_HOST
        self.port = int(os.environ.get("VLLM_PORT", self.DEFAULT_PORT))
        self.base_url = config.base_url or f"http://{self.host}:{self.port}/v1"
        self._current_model: Optional[str] = None
    
    @property
    def server_url(self) -> str:
        """Get the vLLM server URL."""
        return self.base_url.rstrip("/v1").rstrip("/")
    
    async def complete_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate completion using local vLLM server."""
        # Check if server is running
        health = await self.health_check()
        if not health.get("available"):
            raise Exception(
                f"vLLM server not available: {health.get('error', 'Unknown error')}"
            )
        
        # Use the currently loaded model or default
        model = model or self._current_model or self._get_default_model()
        messages = self._build_messages(prompt, system_prompt)
        
        headers = {
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),  # 5 min timeout for local
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"vLLM API error: {error_text}")
                
                data = await response.json()
        
        # Extract response
        content = data["choices"][0]["message"]["content"]
        tokens_used = data.get("usage", {}).get("total_tokens", 0)
        
        return {
            "content": content,
            "tokens_used": tokens_used,
            "model": model,
            "raw_response": data,
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if vLLM server is available."""
        try:
            start = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/models",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    latency = (time.time() - start) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        models = [m["id"] for m in data.get("data", [])]
                        
                        if models:
                            self._current_model = models[0]
                        
                        return {
                            "available": True,
                            "latency_ms": latency,
                            "models_loaded": models,
                        }
                    else:
                        return {
                            "available": False,
                            "error": f"HTTP {response.status}",
                            "latency_ms": latency,
                        }
        except aiohttp.ClientConnectorError:
            return {
                "available": False,
                "error": "Server not running. Start with: scripts/llm/start_vllm.sh",
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e),
            }
    
    def _get_default_model(self) -> str:
        """Get the default model to use."""
        registry = get_registry()
        
        # Find first enabled local model
        for model_id in self.RECOMMENDED_MODELS:
            model = registry.get_model(model_id)
            if model and model.enabled:
                return model.hf_id
        
        # Fallback
        return "Qwen/Qwen2.5-Coder-32B-Instruct"
    
    def get_start_command(
        self,
        model: str,
        tensor_parallel: int = 1,
        max_model_len: int = 32768,
        port: Optional[int] = None,
    ) -> List[str]:
        """
        Get the command to start vLLM server with a specific model.
        
        Args:
            model: HuggingFace model ID
            tensor_parallel: Number of GPUs for tensor parallelism
            max_model_len: Maximum context length
            port: Port to run on (default: 8000)
            
        Returns:
            Command list for subprocess
        """
        port = port or self.port
        
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            f"--model={model}",
            f"--tensor-parallel-size={tensor_parallel}",
            f"--max-model-len={max_model_len}",
            f"--host=0.0.0.0",
            f"--port={port}",
            "--trust-remote-code",
        ]
        
        return cmd
    
    def generate_slurm_script(
        self,
        model_id: str,
        output_dir: str = "logs",
    ) -> str:
        """
        Generate a SLURM script to start vLLM server.
        
        Args:
            model_id: Model ID from registry (e.g., "qwen-coder-32b")
            output_dir: Directory for logs
            
        Returns:
            SLURM script content
        """
        registry = get_registry()
        model = registry.get_model(model_id)
        
        if not model:
            raise ValueError(f"Unknown model: {model_id}")
        
        vllm_args = model.vllm_args or {}
        
        script = f"""#!/bin/bash
#SBATCH --job-name=vllm-{model_id}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:{model.gpus_required}
#SBATCH --time=24:00:00
#SBATCH --output={output_dir}/vllm_{model_id}_%j.out
#SBATCH --error={output_dir}/vllm_{model_id}_%j.err

# vLLM Server for {model.name}
# Model: {model.hf_id}
# GPUs Required: {model.gpus_required}
# VRAM Estimate: {model.size_gb}GB

echo "Starting vLLM server for {model.name}..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Load modules
module load cuda/12.1 2>/dev/null || true

# Activate environment
source ~/envs/biopipelines/bin/activate 2>/dev/null || \\
    source ~/.venv/bin/activate 2>/dev/null || true

# Set HuggingFace cache (optional)
export HF_HOME="${{HF_HOME:-$HOME/.cache/huggingface}}"

# Start vLLM
python -m vllm.entrypoints.openai.api_server \\
    --model {model.hf_id} \\
    --tensor-parallel-size {vllm_args.tensor_parallel_size if hasattr(vllm_args, 'tensor_parallel_size') else model.gpus_required} \\
    --max-model-len {vllm_args.max_model_len if hasattr(vllm_args, 'max_model_len') else 32768} \\
    --dtype {vllm_args.dtype if hasattr(vllm_args, 'dtype') else 'float16'} \\
    --gpu-memory-utilization {vllm_args.gpu_memory_utilization if hasattr(vllm_args, 'gpu_memory_utilization') else 0.9} \\
    --host 0.0.0.0 \\
    --port ${{VLLM_PORT:-8000}} \\
    --trust-remote-code

echo "vLLM server stopped."
"""
        return script


class VLLMModelSwitcher:
    """
    Utility to switch between vLLM models.
    
    On HPC clusters, this typically involves stopping the current
    server and starting a new SLURM job with a different model.
    """
    
    def __init__(self, provider: VLLMProvider):
        self.provider = provider
    
    async def get_loaded_model(self) -> Optional[str]:
        """Get the currently loaded model."""
        health = await self.provider.health_check()
        if health.get("available"):
            models = health.get("models_loaded", [])
            return models[0] if models else None
        return None
    
    def stop_server(self) -> bool:
        """
        Stop the currently running vLLM server.
        
        Returns:
            True if stopped successfully
        """
        # Find and kill vLLM process
        try:
            result = subprocess.run(
                ["pkill", "-f", "vllm.entrypoints.openai.api_server"],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def submit_server_job(
        self,
        model_id: str,
        script_dir: str = "scripts/llm",
    ) -> Optional[str]:
        """
        Submit a SLURM job to start vLLM with a new model.
        
        Args:
            model_id: Model ID from registry
            script_dir: Directory for the generated script
            
        Returns:
            SLURM job ID if successful
        """
        import os
        from pathlib import Path
        
        script_path = Path(script_dir) / f"start_vllm_{model_id}.sh"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate and save script
        script_content = self.provider.generate_slurm_script(model_id)
        with open(script_path, "w") as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        
        # Submit job
        try:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
            )
            
            if result.returncode == 0:
                # Parse job ID from output
                # "Submitted batch job 12345"
                parts = result.stdout.strip().split()
                if len(parts) >= 4:
                    return parts[-1]
            
            return None
        except Exception:
            return None
