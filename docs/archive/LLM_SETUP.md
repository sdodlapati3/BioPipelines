# LLM Integration Guide

This guide covers setting up and using Large Language Models (LLMs) with the BioPipelines Workflow Composer. We support two main approaches:

1. **OpenAI API** - GPT-4o, GPT-4, GPT-3.5 (cloud-based)
2. **vLLM** - Open-source HuggingFace models on GPUs (self-hosted)

## Quick Start

### Option 1: OpenAI (Easiest)

```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

# Test the connection
./scripts/llm/setup_llm.sh test-openai

# Use in Python
from workflow_composer.llm import get_llm
llm = get_llm("openai", model="gpt-4o")
response = llm.complete("Create an RNA-seq workflow")
```

### Option 2: vLLM with Open-Source Models (Self-Hosted)

```bash
# On a GPU node, install vLLM
pip install vllm

# Start the server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000

# Test the connection
./scripts/llm/setup_llm.sh test-vllm

# Use in Python
from workflow_composer.llm import get_llm
llm = get_llm("vllm", model="meta-llama/Llama-3.1-8B-Instruct")
response = llm.complete("Create an RNA-seq workflow")
```

---

## Detailed Setup

### OpenAI Setup

#### 1. Get API Key

1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Create a new secret key
3. Copy the key (starts with `sk-`)

#### 2. Set Environment Variable

```bash
# Add to ~/.bashrc or ~/.zshrc for persistence
export OPENAI_API_KEY="sk-your-api-key-here"

# Or use our environment setup script
./scripts/llm/setup_llm.sh env-setup
# Then edit .env.llm and add your key
source .env.llm
```

#### 3. Test Connection

```bash
./scripts/llm/setup_llm.sh test-openai
```

#### 4. Available Models

| Model | Description | Cost | Recommended For |
|-------|-------------|------|-----------------|
| `gpt-4o` | Latest GPT-4 Omni | $$ | **Best balance of quality and cost** |
| `gpt-4-turbo` | GPT-4 Turbo | $$$ | Complex workflow generation |
| `gpt-4` | Original GPT-4 | $$$$ | Highest quality |
| `gpt-3.5-turbo` | Fast and cheap | $ | Quick iterations, simple tasks |

#### 5. Python Usage

```python
from workflow_composer.llm import get_llm, OpenAIAdapter

# Using factory (recommended)
llm = get_llm("openai", model="gpt-4o")

# Or direct instantiation
llm = OpenAIAdapter(
    model="gpt-4o",
    temperature=0.3,
    max_tokens=4096
)

# Simple completion
response = llm.complete("Explain RNA-seq analysis steps")
print(response.content)

# Chat with context
from workflow_composer.llm import Message
messages = [
    Message.system("You are a bioinformatics expert. Be concise."),
    Message.user("What tools should I use for RNA-seq differential expression?")
]
response = llm.chat(messages)
print(response.content)
```

---

### vLLM Setup (GPU Inference)

vLLM is a high-throughput inference engine for running open-source HuggingFace models on GPUs.

#### 1. Prerequisites

- NVIDIA GPU with CUDA support
- At least 16GB GPU memory for 7B models, 40GB+ for 70B models
- Python 3.9+

#### 2. Installation

```bash
# On GPU node
pip install vllm

# Or via our setup script
./scripts/llm/setup_llm.sh install-vllm
```

#### 3. Start vLLM Server

```bash
# Start with default model (Llama 3.1 8B)
./scripts/llm/setup_llm.sh start-vllm

# Or with a specific model
./scripts/llm/setup_llm.sh start-vllm meta-llama/Llama-3.1-70B-Instruct

# Or manually with full control
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 1
```

#### 4. Server Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | HuggingFace model ID | Required |
| `--port` | Server port | 8000 |
| `--host` | Bind address | localhost |
| `--gpu-memory-utilization` | GPU memory fraction | 0.9 |
| `--tensor-parallel-size` | Number of GPUs | 1 |
| `--max-model-len` | Max sequence length | Model default |
| `--quantization` | Quantization (awq, gptq) | None |

#### 5. Recommended Models

**General Purpose (Balanced Quality/Speed)**

| Model | Size | GPU Memory | Notes |
|-------|------|------------|-------|
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | ~16GB | **Best starting point** |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | ~14GB | Fast, good quality |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | ~14GB | Good multilingual |

**High Quality (More GPU Memory)**

| Model | Size | GPU Memory | Notes |
|-------|------|------------|-------|
| `meta-llama/Llama-3.1-70B-Instruct` | 70B | ~140GB | Highest quality |
| `Qwen/Qwen2.5-72B-Instruct` | 72B | ~144GB | Excellent reasoning |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | 47B | ~90GB | MoE architecture |

**Code Generation (for Workflow Scripts)**

| Model | Size | GPU Memory | Notes |
|-------|------|------------|-------|
| `codellama/CodeLlama-34b-Instruct-hf` | 34B | ~70GB | Best for code |
| `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | 16B | ~32GB | Code-focused |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | 7B | ~14GB | Compact code model |

**Small/Fast (For Testing)**

| Model | Size | GPU Memory | Notes |
|-------|------|------------|-------|
| `microsoft/Phi-3.5-mini-instruct` | 3.8B | ~8GB | Very fast |
| `google/gemma-2-2b-it` | 2B | ~4GB | Smallest capable model |

#### 6. HuggingFace Token (For Gated Models)

Some models (like Llama) require accepting license terms:

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create an access token
3. Accept the model's license on its HuggingFace page
4. Set the token:

```bash
export HF_TOKEN="hf_..."
```

#### 7. Test vLLM Connection

```bash
./scripts/llm/setup_llm.sh test-vllm
```

#### 8. Python Usage

```python
from workflow_composer.llm import get_llm, VLLMAdapter

# Using factory (recommended)
llm = get_llm("vllm", model="meta-llama/Llama-3.1-8B-Instruct")

# Or with custom URL
llm = VLLMAdapter(
    model="meta-llama/Llama-3.1-8B-Instruct",
    base_url="http://gpu-node:8000",
    temperature=0.3
)

# Use model aliases
llm = get_llm("vllm", model="llama3.1-8b")  # Resolves to full name

# Get recommended models
from workflow_composer.llm import VLLMAdapter
models = VLLMAdapter.get_recommended_models()
print(models)

# Generate vLLM launch command
cmd = VLLMAdapter.get_launch_command(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=2
)
print(cmd)
```

---

### Using HuggingFace Adapter with vLLM Backend

The HuggingFace adapter can use vLLM as its backend:

```python
from workflow_composer.llm import HuggingFaceAdapter

# Use vLLM backend
llm = HuggingFaceAdapter(
    model="meta-llama/Llama-3.1-8B-Instruct",
    backend="vllm",
    vllm_url="http://localhost:8000"
)

# Or use HuggingFace Inference API
llm = HuggingFaceAdapter(
    model="meta-llama/Llama-3.1-8B-Instruct",
    backend="api"  # Requires HF_TOKEN
)

# Or use local transformers
llm = HuggingFaceAdapter(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    backend="transformers",
    device="cuda"
)
```

---

## Configuration

### Configuration File

Edit `config/composer.yaml`:

```yaml
llm:
  # Default provider: openai, vllm, huggingface, ollama, anthropic
  default_provider: vllm
  
  providers:
    openai:
      model: gpt-4o
      temperature: 0.3
      max_tokens: 4096
    
    vllm:
      base_url: http://localhost:8000
      model: meta-llama/Llama-3.1-8B-Instruct
      temperature: 0.3
      max_tokens: 4096
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `HF_TOKEN` | HuggingFace token (for gated models) |
| `VLLM_BASE_URL` | vLLM server URL (default: http://localhost:8000) |
| `VLLM_API_KEY` | vLLM API key (optional, for secured deployments) |

---

## Running on HPC Clusters

### SLURM Job for vLLM Server

Create `scripts/llm/run_vllm_server.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=vllm-server
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/vllm_server_%j.log

module load cuda/12.1
source ~/miniconda3/bin/activate biopipelines

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9

# Keep running until job ends
sleep infinity
```

Submit:

```bash
sbatch scripts/llm/run_vllm_server.sbatch

# Get the node name
squeue -u $USER

# SSH tunnel from login node
ssh -L 8000:gpu-node:8000 login-node
```

### Multi-GPU Setup

For 70B+ models, use tensor parallelism across multiple GPUs:

```bash
#SBATCH --gres=gpu:4

python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9
```

---

## API Reference

### Common Interface (All Adapters)

```python
# All adapters implement the same interface
class LLMAdapter:
    def complete(self, prompt: str, **kwargs) -> LLMResponse
    def chat(self, messages: List[Message], **kwargs) -> LLMResponse
    def stream(self, prompt: str, **kwargs) -> Iterator[str]
    def is_available(self) -> bool
```

### LLMResponse Object

```python
response = llm.complete("...")
print(response.content)      # Generated text
print(response.model)        # Model used
print(response.provider)     # Provider name
print(response.tokens_used)  # Total tokens
print(response.finish_reason) # stop, length, etc.
```

### Message Object

```python
from workflow_composer.llm import Message

# Create messages
system_msg = Message.system("You are a bioinformatics expert.")
user_msg = Message.user("What is RNA-seq?")
assistant_msg = Message.assistant("RNA-seq is...")

# Use in chat
response = llm.chat([system_msg, user_msg])
```

### Factory Function

```python
from workflow_composer.llm import get_llm, list_providers, check_providers

# List all providers
print(list_providers())
# {'ollama': 'OllamaAdapter', 'openai': 'OpenAIAdapter', ...}

# Check which are available
print(check_providers())
# {'ollama': False, 'openai': True, 'vllm': True, ...}

# Create adapter
llm = get_llm("vllm", model="llama3.1-8b", temperature=0.3)
```

---

## Troubleshooting

### OpenAI Issues

**Error: "API key not configured"**
```bash
export OPENAI_API_KEY="sk-..."
```

**Error: "Rate limit exceeded"**
- Wait a few seconds and retry
- Reduce `max_tokens` or use a smaller model

### vLLM Issues

**Error: "Connection refused"**
```bash
# Check if server is running
curl http://localhost:8000/health

# Start the server
./scripts/llm/setup_llm.sh start-vllm
```

**Error: "CUDA out of memory"**
- Use a smaller model
- Reduce `--gpu-memory-utilization`
- Use quantization: `--quantization awq`

**Error: "Model not found"**
```bash
# Login to HuggingFace
huggingface-cli login

# Accept model license on HuggingFace website
```

### General Issues

**Testing all providers**
```bash
./scripts/llm/setup_llm.sh test-all
```

**Check logs**
```bash
# vLLM server logs
tail -f logs/vllm_server.log
```

---

## Performance Comparison

| Provider | Latency | Quality | Cost |
|----------|---------|---------|------|
| OpenAI GPT-4o | ~1-2s | ⭐⭐⭐⭐⭐ | $$$$ |
| vLLM Llama 3.1 70B | ~3-5s | ⭐⭐⭐⭐⭐ | Free (GPU) |
| vLLM Llama 3.1 8B | ~1s | ⭐⭐⭐⭐ | Free (GPU) |
| vLLM Mistral 7B | ~0.5s | ⭐⭐⭐⭐ | Free (GPU) |

---

## Next Steps

1. **Set up your preferred LLM backend** using this guide
2. **Test with the Workflow Composer**: `python -m workflow_composer.cli`
3. **Try the Web UI**: `python -m workflow_composer.web.app`
4. **Read the [Workflow Composer Guide](WORKFLOW_COMPOSER_GUIDE.md)** for full usage

## Support

- Check the [FAQ](TUTORIALS.md#faq) for common questions
- Review [Composition Patterns](COMPOSITION_PATTERNS.md) for workflow examples
- Open an issue on GitHub for bugs or feature requests
