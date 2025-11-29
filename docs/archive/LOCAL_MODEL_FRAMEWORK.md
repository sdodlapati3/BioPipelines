# Local Model Framework for BioPipelines

## Overview

This document outlines the strategy and implementation for using local GPU models as backup coding agents in BioPipelines. The framework provides a flexible, extensible system for integrating multiple open-source models.

## Design Principles

1. **API-First**: Use cloud APIs (free/subscribed) as primary providers
2. **Local as Backup**: GPU models only when APIs exhausted or unavailable  
3. **Flexible Registry**: Easy to add/remove models without code changes
4. **Smart Routing**: Automatically select best available provider
5. **Cost Optimization**: Prioritize free tiers, then subscriptions, then local

## Provider Priority (Waterfall)

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROVIDER SELECTION FLOW                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. FREE TIER APIs (Unlimited during dev)                       │
│     ├── Lightning.ai (30M tokens/month)  ──► PRIORITY 1         │
│     └── Google Gemini (free tier)        ──► PRIORITY 2         │
│                                                                  │
│  2. SUBSCRIBED APIs                                              │
│     ├── GitHub Copilot (2 accounts)      ──► PRIORITY 3         │
│     └── OpenAI API                       ──► PRIORITY 4         │
│                                                                  │
│  3. LOCAL GPU MODELS (Backup only)                              │
│     ├── Qwen2.5-Coder-32B               ──► PRIORITY 5          │
│     ├── DeepSeek-Coder-V2               ──► PRIORITY 6          │
│     ├── Llama-3.3-70B-Instruct          ──► PRIORITY 7          │
│     ├── MiniMax-M2                      ──► PRIORITY 8          │
│     └── CodeLlama-34B                   ──► PRIORITY 9          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Hardware Configuration

```yaml
Cluster: GCP HPC
Node Type: a3-highgpu-8g
GPUs: 8× NVIDIA H100 80GB
Total VRAM: 640GB
Network: 3.2Tbps GPU-GPU bandwidth
```

## Supported Models (Initial 5)

### 1. Qwen2.5-Coder-32B-Instruct
- **Size**: 32B parameters
- **VRAM**: ~65GB (FP16), ~35GB (INT8)
- **GPUs Needed**: 1× H100
- **Strengths**: Best coding model at this size, fast inference
- **HuggingFace**: `Qwen/Qwen2.5-Coder-32B-Instruct`

### 2. DeepSeek-Coder-V2-Instruct
- **Size**: 236B MoE (21B active)
- **VRAM**: ~120GB (FP16)
- **GPUs Needed**: 2× H100
- **Strengths**: Excellent code completion, fill-in-middle
- **HuggingFace**: `deepseek-ai/DeepSeek-Coder-V2-Instruct`

### 3. Llama-3.3-70B-Instruct
- **Size**: 70B parameters
- **VRAM**: ~140GB (FP16), ~70GB (INT8)
- **GPUs Needed**: 2× H100
- **Strengths**: Strong general + coding, well-supported
- **HuggingFace**: `meta-llama/Llama-3.3-70B-Instruct`

### 4. MiniMax-M2
- **Size**: 230B MoE (10B active)
- **VRAM**: ~230GB (FP8)
- **GPUs Needed**: 4× H100
- **Strengths**: Agentic workflows, SWE-bench optimized
- **HuggingFace**: `MiniMaxAI/MiniMax-M2`

### 5. CodeLlama-34B-Instruct
- **Size**: 34B parameters
- **VRAM**: ~70GB (FP16)
- **GPUs Needed**: 1× H100
- **Strengths**: Code-specific, good for infilling
- **HuggingFace**: `codellama/CodeLlama-34b-Instruct-hf`

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL REGISTRY                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   API        │    │   Local      │    │   Config     │       │
│  │   Providers  │    │   Models     │    │   Manager    │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │                │
│         └─────────┬─────────┴─────────┬─────────┘                │
│                   │                   │                          │
│           ┌───────▼───────┐   ┌───────▼───────┐                 │
│           │  Model Router │   │  Health Check │                 │
│           └───────┬───────┘   └───────────────┘                 │
│                   │                                              │
│           ┌───────▼───────┐                                     │
│           │  Unified API  │ ◄── OpenAI-compatible interface     │
│           └───────┬───────┘                                     │
│                   │                                              │
└───────────────────┼──────────────────────────────────────────────┘
                    │
        ┌───────────▼───────────┐
        │  ErrorDiagnosisAgent  │
        │  WorkflowComposer     │
        │  Other Consumers      │
        └───────────────────────┘
```

## File Structure

```
src/workflow_composer/models/
├── __init__.py           # Package exports
├── registry.py           # Model registry & configuration
├── router.py             # Smart provider selection
├── providers/
│   ├── __init__.py
│   ├── base.py           # Abstract provider interface
│   ├── lightning.py      # Lightning.ai adapter
│   ├── gemini.py         # Google Gemini adapter
│   ├── openai.py         # OpenAI adapter
│   ├── github_copilot.py # GitHub Copilot adapter
│   └── vllm.py           # vLLM local model adapter
├── configs/
│   ├── models.yaml       # Model definitions
│   └── providers.yaml    # Provider configurations
└── utils/
    ├── health.py         # Health checking
    └── metrics.py        # Usage tracking
```

## Configuration Format

### models.yaml
```yaml
models:
  qwen-coder-32b:
    name: "Qwen2.5-Coder-32B-Instruct"
    hf_id: "Qwen/Qwen2.5-Coder-32B-Instruct"
    type: local
    size_gb: 65
    gpus_required: 1
    context_length: 32768
    capabilities: [code, chat, fill-in-middle]
    vllm_args:
      tensor_parallel_size: 1
      max_model_len: 32768
      dtype: float16
    
  deepseek-coder-v2:
    name: "DeepSeek-Coder-V2-Instruct"
    hf_id: "deepseek-ai/DeepSeek-Coder-V2-Instruct"
    type: local
    size_gb: 120
    gpus_required: 2
    context_length: 128000
    capabilities: [code, chat, fill-in-middle, reasoning]
    vllm_args:
      tensor_parallel_size: 2
      max_model_len: 65536
      dtype: float16
      
  # ... more models
```

### providers.yaml
```yaml
providers:
  lightning:
    type: api
    priority: 1
    env_key: LIGHTNING_API_KEY
    base_url: "https://api.lightning.ai/v1"
    models: [llama-3.1-8b, mistral-7b]
    free_tier: true
    rate_limit: 100/min
    
  gemini:
    type: api
    priority: 2
    env_key: GOOGLE_API_KEY
    models: [gemini-2.0-flash, gemini-2.5-pro]
    free_tier: true
    rate_limit: 15/min
    
  vllm:
    type: local
    priority: 5
    base_url: "http://localhost:8000/v1"
    models: [qwen-coder-32b, deepseek-coder-v2, llama-3.3-70b]
    requires_gpu: true
```

## Usage Examples

### Basic Usage
```python
from workflow_composer.models import get_model_client

# Auto-selects best available provider
client = get_model_client()
response = client.complete("Fix this Python error...")

# Force specific provider
client = get_model_client(provider="vllm", model="qwen-coder-32b")
```

### With Fallback
```python
from workflow_composer.models import ModelRouter

router = ModelRouter()

# Will try providers in priority order until one works
response = router.complete(
    prompt="Analyze this error log...",
    fallback=True,
    max_retries=3
)
```

### Health Check
```python
from workflow_composer.models import check_all_providers

status = check_all_providers()
# {
#   "lightning": {"available": True, "latency_ms": 120},
#   "gemini": {"available": True, "latency_ms": 85},
#   "vllm": {"available": False, "error": "Server not running"},
# }
```

## SLURM Integration

### Launch vLLM Server Job
```bash
#!/bin/bash
#SBATCH --job-name=vllm-server
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=logs/vllm_%j.out

module load cuda/12.1
source ~/envs/biopipelines/bin/activate

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-32B-Instruct \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8000
```

## Metrics & Monitoring

Track usage across providers:
- Tokens consumed per provider
- Latency percentiles
- Error rates
- Cost estimation
- Fallback frequency

## Future Extensions

1. **Model Fine-tuning**: Fine-tune on BioPipelines error patterns
2. **Ensemble Inference**: Combine multiple model outputs
3. **Caching Layer**: Cache common error diagnoses
4. **A/B Testing**: Compare model performance on same inputs
5. **Auto-scaling**: Start/stop local models based on demand

## Implementation Phases

### Phase 1: Core Framework (Current)
- [x] Provider abstraction layer
- [x] Model registry
- [x] Smart routing
- [x] vLLM adapter
- [x] Configuration system

### Phase 2: Model Support
- [ ] Qwen2.5-Coder-32B integration
- [ ] DeepSeek-Coder-V2 integration
- [ ] Llama-3.3-70B integration
- [ ] MiniMax-M2 integration
- [ ] CodeLlama-34B integration

### Phase 3: Production Ready
- [ ] Health monitoring dashboard
- [ ] Usage metrics & alerts
- [ ] Auto-failover testing
- [ ] Load balancing
- [ ] Cost optimization

---

*Document Version: 1.0*
*Last Updated: November 2025*
*Author: BioPipelines Team*
