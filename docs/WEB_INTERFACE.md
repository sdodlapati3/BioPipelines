# BioPipelines Web Interface Guide

The BioPipelines Workflow Composer provides multiple web interfaces for generating bioinformatics workflows using AI.

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Launch web UI (default: Gradio)
./scripts/launch_web_ui.sh

# Or use CLI
biocomposer ui
```

## Available Interfaces

| Interface | Port | Best For | Command |
|-----------|------|----------|---------|
| **Gradio** (recommended) | 7860 | Interactive chat, demos | `biocomposer ui` |
| **Flask** | 5000 | Simple web forms | `biocomposer ui -t flask` |
| **FastAPI** | 8080 | REST API, integrations | `biocomposer ui -t api` |

---

## 1. Gradio UI (Recommended)

Modern chat-based interface with real-time workflow generation.

### Features

- 💬 **Chat Interface**: Natural conversation with AI
- 🚀 **Streaming Responses**: See AI thinking in real-time
- 🔧 **Tool Browser**: Search 100+ bioinformatics tools
- 📦 **Module Library**: Browse Nextflow modules
- 📥 **Downloads**: Get generated workflows as ZIP
- 🔗 **Shareable Links**: Create public demos

### Launch

```bash
# Standard launch
biocomposer ui

# With public link (for demos)
biocomposer ui --share

# Custom port
biocomposer ui --port 8000

# Full options
./scripts/launch_web_ui.sh --gradio --port 7860 --share
```

### Screenshot

```
╔══════════════════════════════════════════════════════════════════╗
║        🧬 BioPipelines - AI Workflow Composer                    ║
╠══════════════════════════════════════════════════════════════════╣
║  🔧 150 Tools  │  📦 80 Modules  │  🐳 12 Containers  │  🧬 38 Types ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  [💬 Chat] [🔧 Tools] [📦 Modules] [📥 Download] [⚙️ Settings]    ║
║                                                                  ║
║  You: Generate an RNA-seq pipeline for mouse with STAR and DESeq2║
║                                                                  ║
║  AI: 🔍 Parsing your request...                                  ║
║      📋 Detected Analysis: rnaseq                                ║
║      ✅ Ready to generate! Tools: 6, Modules: 6                  ║
║      🎉 Workflow Generated!                                       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

### Example Prompts

```
# RNA-seq
RNA-seq differential expression analysis for mouse samples 
comparing treatment vs control using STAR and DESeq2

# ChIP-seq
ChIP-seq peak calling for human H3K27ac samples with input controls
using Bowtie2 and MACS2

# Single-cell
10x Genomics single-cell RNA-seq analysis with STARsolo and Seurat

# Variant Calling
Whole exome sequencing variant calling for human samples
using BWA-MEM2 and GATK HaplotypeCaller
```

---

## 2. FastAPI REST API

Programmatic access to workflow generation.

### Launch

```bash
biocomposer ui -t api

# Or with uvicorn directly
uvicorn workflow_composer.web.api:app --reload --port 8080
```

### API Documentation

- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/stats` | GET | System statistics |
| `/chat` | POST | Chat with AI |
| `/chat/stream` | POST | Streaming chat |
| `/generate` | POST | Generate workflow |
| `/parse-intent` | POST | Parse description to intent |
| `/workflows` | GET | List generated workflows |
| `/workflows/{id}` | GET | Get workflow details |
| `/workflows/{id}/download` | GET | Download workflow ZIP |
| `/tools/search` | POST | Search tools |
| `/modules` | GET | List modules |
| `/analysis-types` | GET | List analysis types |

### Example: Generate Workflow

```python
import requests

response = requests.post("http://localhost:8080/generate", json={
    "description": "RNA-seq differential expression for mouse",
    "provider": "openai",
    "name": "rnaseq_mouse"
})

data = response.json()
print(f"Workflow ID: {data['workflow_id']}")
print(f"Tools: {data['tools_used']}")
print(f"Modules: {data['modules_used']}")
```

### Example: Chat

```python
import requests

response = requests.post("http://localhost:8080/chat", json={
    "messages": [
        {"role": "user", "content": "What's the best aligner for RNA-seq?"}
    ],
    "provider": "openai"
})

print(response.json()["content"])
```

### Example: Streaming Chat (SSE)

```python
import requests

with requests.post(
    "http://localhost:8080/chat/stream",
    json={"messages": [{"role": "user", "content": "Explain STAR aligner"}]},
    stream=True
) as response:
    for line in response.iter_lines():
        if line:
            print(line.decode())
```

---

## 3. Flask UI (Legacy)

Simple form-based interface.

### Launch

```bash
biocomposer ui -t flask
./scripts/launch_web_ui.sh --flask
```

### Features

- Form-based workflow generation
- Tool and module browser
- Basic styling with Bootstrap

---

## Configuration

### Environment Variables

```bash
# Required for OpenAI
export OPENAI_API_KEY="sk-..."

# Optional: vLLM server
export VLLM_API_BASE="http://localhost:8000/v1"

# Optional: Anthropic
export ANTHROPIC_API_KEY="..."
```

### Using .secrets file

```bash
# Store API key securely
echo "sk-proj-..." > .secrets/openai_key
chmod 600 .secrets/openai_key

# The launch script loads it automatically
./scripts/launch_web_ui.sh
```

---

## Deployment

### Local Development

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run with auto-reload
biocomposer ui --debug
```

### Production (Gradio)

```bash
# Run without debug
biocomposer ui --port 7860

# With authentication (set in environment)
export GRADIO_USERNAME=admin
export GRADIO_PASSWORD=secure123
biocomposer ui
```

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -e .

EXPOSE 7860

CMD ["biocomposer", "ui", "--host", "0.0.0.0", "--port", "7860"]
```

### HPC Cluster

For running on HPC with GPU nodes:

```bash
# Submit job to start UI on GPU node
sbatch scripts/llm/serve_vllm.sbatch

# Then access via SSH tunnel
ssh -L 7860:gpu-node:7860 cluster.example.com

# Open http://localhost:7860
```

---

## Troubleshooting

### "No LLM providers available"

```bash
# Check which providers are configured
biocomposer providers --check

# Set up OpenAI
export OPENAI_API_KEY="sk-..."

# Or start local Ollama
ollama serve
```

### "Gradio not installed"

```bash
pip install gradio>=4.0.0
```

### "Connection refused" on API

```bash
# Check if server is running
curl http://localhost:8080/health

# Check port availability
lsof -i :8080
```

### Slow first response

The first request may be slow as models are loaded. Subsequent requests will be faster.

---

## See Also

- [LLM Setup Guide](LLM_SETUP.md) - Configure OpenAI, vLLM, Ollama
- [Workflow Composer Guide](WORKFLOW_COMPOSER_GUIDE.md) - Full documentation
- [Tutorials](TUTORIALS.md) - Step-by-step examples
