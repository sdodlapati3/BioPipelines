# Orchestrator-8B Integration Guide

## Overview

BioPipelines now integrates **NVIDIA's Orchestrator-8B** from the [ToolOrchestra paper](https://arxiv.org/abs/2511.21689), providing intelligent model/tool routing for cost-efficient workflow generation.

## What is Orchestrator-8B?

Orchestrator-8B is an 8B parameter model trained with reinforcement learning to:
- **Route queries** to the appropriate model tier (local vs cloud, small vs large)
- **Minimize cost** while maintaining quality
- **Coordinate tools** in multi-turn reasoning

On benchmarks, it achieves **37.1% on Humanity's Last Exam**, outperforming GPT-5 (35.1%) while being **2.5x more efficient**.

## Architecture

```
                     ┌─────────────────────────┐
                     │   User Query            │
                     │   "RNA-seq workflow"    │
                     └───────────┬─────────────┘
                                 │
                     ┌───────────▼─────────────┐
                     │   Orchestrator-8B       │
                     │   Routing Decision      │
                     └───────────┬─────────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           │                     │                     │
    ┌──────▼──────┐      ┌───────▼───────┐     ┌──────▼──────┐
    │ Local LLM   │      │  Cloud LLM    │     │  Specialist │
    │ (Ollama)    │      │  (GPT-4)      │     │  (CodeLlama)│
    │ Cost: $0    │      │  Cost: $$$    │     │  Cost: $    │
    └─────────────┘      └───────────────┘     └─────────────┘
```

## Quick Start

### Basic Usage

```python
from workflow_composer.llm import Orchestrator8B, OrchestratorConfig

# Create orchestrator with default settings
orch = Orchestrator8B()

# Get routing decision (no model needed)
decision = orch.get_routing_decision("Generate RNA-seq workflow")
print(f"Route to: {decision.target_model}")
print(f"Tier: {decision.target_tier}")
print(f"Estimated cost: ${decision.estimated_cost:.4f}")
```

### With OrchestratedSupervisor

```python
from workflow_composer.agents import OrchestratedSupervisor

# Create supervisor with orchestrator routing
supervisor = OrchestratedSupervisor(
    use_orchestrator=True,
    prefer_local=True,      # Prefer local models when possible
    max_cost=0.50           # Maximum cost per query
)

# Execute workflow generation
result = await supervisor.execute("Generate ChIP-seq pipeline with MACS2")

# Check routing details
print(f"Cost: ${result.metadata['cost']:.4f}")
print(f"Models used: {result.metadata['models_used']}")
print(f"Orchestrator reasoning: {result.metadata['orchestrator_reasoning']}")
```

## Configuration Options

### OrchestratorConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `nvidia/Orchestrator-8B` | HuggingFace model ID |
| `inference_backend` | `vllm` | Backend: `vllm`, `transformers`, `api` |
| `prefer_local` | `True` | Prefer local models over cloud |
| `max_cost_per_query` | `1.0` | Maximum cost in USD |
| `optimize_for` | `balanced` | `cost`, `speed`, `accuracy`, `balanced` |
| `max_turns` | `10` | Maximum orchestration turns |

### User Preferences

```python
config = OrchestratorConfig(
    prefer_local=True,              # Privacy-conscious
    max_cost_per_query=0.10,        # Strict budget
    optimize_for="cost"             # Minimize cost above all
)
```

## Model Tiers

| Tier | Examples | Cost | Use Case |
|------|----------|------|----------|
| `LOCAL_SMALL` | Ollama, Llama-7B | Free | Simple queries |
| `LOCAL_LARGE` | vLLM with 70B | Free | Code generation |
| `CLOUD_SMALL` | GPT-3.5, Claude Haiku | $ | Moderate complexity |
| `CLOUD_LARGE` | GPT-4, Claude Opus | $$$ | Complex reasoning |
| `SPECIALIST` | CodeLlama | $ | Domain-specific |

## BioPipeline Tool Catalog

The orchestrator is pre-configured with BioPipeline-specific tools:

| Tool | Description | Default Tier |
|------|-------------|--------------|
| `workflow_planner` | Plans bioinformatics workflows | LOCAL_SMALL |
| `code_generator` | Generates Nextflow code | LOCAL_LARGE |
| `code_validator` | Validates generated code | LOCAL_SMALL |
| `nfcore_reference` | Searches nf-core modules | LOCAL_SMALL |
| `container_selector` | Selects containers | LOCAL_SMALL |
| `cloud_expert` | Consults GPT-4/Claude | CLOUD_LARGE |

## Deployment Options

### Option 1: vLLM Server (Recommended)

```bash
# Start vLLM server with Orchestrator-8B
pip install vllm

vllm serve nvidia/Orchestrator-8B \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9
```

```python
# Use with vLLM
orch = Orchestrator8B(OrchestratorConfig(
    inference_backend="vllm",
    api_endpoint="http://localhost:8000/v1"
))
await orch.initialize()
```

### Option 2: Transformers (Development)

```python
# Use with transformers (slower, requires more memory)
orch = Orchestrator8B(OrchestratorConfig(
    inference_backend="transformers"
))
await orch.initialize()  # Loads model into GPU memory
```

### Option 3: Remote API

```python
# Use with custom API endpoint
orch = Orchestrator8B(OrchestratorConfig(
    inference_backend="api",
    api_endpoint="https://your-orchestrator-api.com/v1"
))
```

### Option 4: Heuristic Mode (No GPU)

```python
# Works without loading the model
orch = Orchestrator8B()
# Don't call initialize() - uses heuristic routing

decision = orch.get_routing_decision("Generate workflow")
# Lower confidence but still useful routing
```

## Fine-tuning for BioPipelines

The base Orchestrator-8B can be fine-tuned on BioPipeline-specific data:

```python
# Data format for fine-tuning
training_data = [
    {
        "query": "Generate RNA-seq workflow with STAR aligner",
        "optimal_routing": {
            "model": "local_large",
            "tools": ["workflow_planner", "code_generator"],
            "cost": 0.01
        }
    },
    {
        "query": "What is a FASTQ file?",
        "optimal_routing": {
            "model": "local_small",
            "tools": ["nfcore_reference"],
            "cost": 0.0
        }
    }
]
```

See the [ToolOrchestra training guide](https://github.com/NVlabs/ToolOrchestra/tree/main/training) for fine-tuning instructions.

## Comparison: With vs Without Orchestrator

| Metric | Without Orchestrator | With Orchestrator |
|--------|---------------------|-------------------|
| Simple queries | Cloud LLM ($0.05) | Local LLM ($0.00) |
| Code generation | Cloud LLM ($0.10) | Local vLLM ($0.01) |
| Complex reasoning | Cloud LLM ($0.15) | Cloud only when needed ($0.05) |
| **Average cost** | **$0.10/query** | **$0.02/query** |
| **Cost savings** | - | **80%** |

## Monitoring & Analytics

```python
# Get detailed metrics
result = await supervisor.execute("Generate workflow")

print(f"Routing method: {result.metadata['routing']}")
print(f"Total cost: ${result.metadata['cost']:.4f}")
print(f"Models used: {result.metadata['models_used']}")
print(f"Turns: {result.metadata['turns']}")
print(f"Tier used: {result.metadata['tier_used']}")
```

## Troubleshooting

### Model Not Loading

```python
# Check if model is available
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Orchestrator-8B needs ~16GB GPU memory
```

### Falling Back to Heuristics

```python
# If you see "Falling back to heuristic routing"
# This is normal when:
# 1. Model not loaded (vLLM server not running)
# 2. API endpoint not reachable
# 3. Initialization failed

# Heuristics still work, just with lower confidence
```

### High Latency

```python
# Use vLLM for production (faster than transformers)
config = OrchestratorConfig(
    inference_backend="vllm",  # Not "transformers"
    max_turns=5                # Reduce turns for faster response
)
```

## References

- [ToolOrchestra Paper](https://arxiv.org/abs/2511.21689)
- [Orchestrator-8B Model](https://huggingface.co/nvidia/Orchestrator-8B)
- [ToolScale Dataset](https://huggingface.co/datasets/nvidia/ToolScale)
- [GitHub Repository](https://github.com/NVlabs/ToolOrchestra)
