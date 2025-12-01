# BioPipelines

> **AI-Powered Bioinformatics Workflow Generation**  
> Natural language → Production Nextflow pipelines → Results

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Nextflow](https://img.shields.io/badge/nextflow-DSL2-green.svg)](https://www.nextflow.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 Quick Start (30 seconds)

```bash
# 1. Activate environment
conda activate ~/envs/biopipelines

# 2. Set API keys (choose any provider)
export GEMINI_API_KEY="your-key"      # Google - FREE 1,500/day
export CEREBRAS_API_KEY="your-key"    # Fastest - FREE 14,400/day
export GROQ_API_KEY="your-key"        # Fast - FREE 14,400/day

# 3. Launch web interface
./scripts/start_gradio.sh

# 4. Open browser and type:
#    "RNA-seq differential expression for mouse, paired-end reads"
```

📖 **[Complete Architecture Guide](docs/ARCHITECTURE.md)** - Technical deep-dive

---

## 🌐 LLM Provider Cascade

BioPipelines uses intelligent provider routing with automatic failover:

| Priority | Provider | Free Tier | Speed | Best For |
|----------|----------|-----------|-------|----------|
| 1 | **Google Gemini** | 1,500 req/day | ~500ms | High-quality generation |
| 2 | **Cerebras** | 14,400 req/day, 1M tokens | ~170ms | **Fastest inference** |
| 3 | **Groq** | 14,400 req/day | ~170ms | Fast inference |
| 4 | **OpenRouter** | 50 req/day (:free models) | ~2.8s | Model variety |
| 5 | **Lightning.ai** | 1,000 credits | ~400ms | DeepSeek models |
| 6 | **GitHub Models** | Requires approval | ~1s | GPT-4 access |
| 15 | **Ollama** | Unlimited (local) | Variable | Privacy/offline |
| 16 | **vLLM** | Unlimited (local) | GPU-dependent | Custom models |
| 99 | **OpenAI** | Pay-per-use | ~800ms | Best quality |

**Automatic Failover**: If Gemini is unavailable, system cascades to Cerebras → Groq → etc.

See [FREE_LLM_PROVIDERS.md](docs/FREE_LLM_PROVIDERS.md) for detailed API key setup.

---

## ✨ Features

### 🤖 AI-Powered Workflow Composer

Generate production-ready Nextflow pipelines from natural language:

```python
from workflow_composer import Composer

composer = Composer()  # Uses provider cascade automatically
workflow = composer.generate(
    "RNA-seq differential expression for mouse, treatment vs control"
)
workflow.save("my_rnaseq_workflow/")
```

### 🧬 UnifiedIntentParser (87.4% Accuracy)

Advanced natural language understanding:
- Multi-model ensemble with arbiter voting
- Automatic model selection based on query complexity
- 40% LLM augmentation for ambiguous queries

### 🧬 10 Production-Ready Pipelines

All pipelines fully validated and containerized:

| Pipeline | Description | Status |
|----------|-------------|--------|
| **DNA-seq** | Variant calling with GATK, FreeBayes | ✅ Validated |
| **RNA-seq** | Differential expression with DESeq2 | ✅ Validated |
| **scRNA-seq** | Single-cell analysis with Scanpy | ✅ Validated |
| **ChIP-seq** | Peak calling with MACS2 | ✅ Validated |
| **ATAC-seq** | Chromatin accessibility | ✅ Validated |
| **Methylation** | WGBS/RRBS bisulfite analysis | ⚠️ Core Complete |
| **Hi-C** | 3D genome organization | ⚠️ Core Complete |
| **Long-read** | Nanopore/PacBio SV detection | ✅ Validated |
| **Metagenomics** | Taxonomic profiling (Kraken2) | ✅ Validated |
| **Structural Variants** | Multi-tool SV calling | ✅ Validated |

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/sdodlapati3/BioPipelines.git
cd BioPipelines

# Create conda environment
conda env create -f environment.yml
conda activate biopipelines

# Install Python package
pip install -e .

# Set up API keys (copy and edit)
cp docker/.env.example .env
# Edit .env with your API keys
```

---

## 🎯 Usage

### Web Interface (Recommended)

```bash
./scripts/start_gradio.sh
# Open http://localhost:7860
```

### CLI

```bash
# Generate workflow from natural language
biocomposer generate "ChIP-seq peak calling for human H3K4me3" -o chipseq_workflow/

# Interactive chat mode
biocomposer chat

# Search available tools
biocomposer tools --search "alignment"

# Check LLM providers
biocomposer providers --check
```

### Python API

```python
from workflow_composer import Composer
from workflow_composer.providers import check_providers

# Check available providers
status = check_providers()
print(status)
# {'gemini': True, 'cerebras': True, 'groq': True, ...}

# Generate workflow
composer = Composer()
workflow = composer.generate(
    "WGS germline variant calling for human samples"
)
workflow.save("variants_workflow/")
```

---

## 📁 Project Structure

```
BioPipelines/
├── src/workflow_composer/     # AI Workflow Composer (main package)
│   ├── providers/             # LLM providers (Gemini, Cerebras, Groq, etc.)
│   ├── core/                  # Intent parsing, tool selection, generation
│   ├── agents/                # ChatAgent, multi-agent orchestration
│   ├── cli.py                 # biocomposer CLI
│   └── composer.py            # Main Composer class
├── nextflow-pipelines/        # Production Nextflow pipelines
│   └── modules/               # Reusable Nextflow modules
├── containers/                # Singularity container definitions
│   ├── base/                  # Base bioinformatics container
│   ├── rna-seq/               # RNA-seq tools container
│   └── ...                    # Pipeline-specific containers
├── config/                    # Configuration files
│   ├── composer.yaml          # Workflow Composer config
│   └── tool_mappings.yaml     # Tool catalog
├── scripts/                   # Utility scripts
│   ├── start_gradio.sh        # Launch web interface
│   ├── start_server.sh        # Start API server
│   └── llm/                   # vLLM server scripts
├── data/                      # Data directory (gitignored)
├── docs/                      # Documentation
│   ├── ARCHITECTURE.md        # System architecture
│   ├── FREE_LLM_PROVIDERS.md  # Free LLM API guide
│   └── tutorials/             # Step-by-step guides
├── examples/                  # Example workflows
│   └── generated/             # AI-generated examples
├── tests/                     # Test suite
└── logs/                      # Runtime logs
```

---

## 🔧 Requirements

- **Python** >= 3.10
- **Conda/Mamba** (for environment management)
- **Nextflow** >= 23.0 (for pipeline execution)
- **Singularity** >= 3.8 (for containerized tools)
- **SLURM** (optional, for HPC execution)

---

## 📚 Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)** - System design and components
- **[Free LLM Providers](docs/FREE_LLM_PROVIDERS.md)** - API key setup for all providers
- **[Container Architecture](docs/infrastructure/CONTAINER_ARCHITECTURE.md)** - Singularity containers

---

## 🤝 Contributing

Contributions welcome! Please open an issue first to discuss changes.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 📧 Contact

For questions, please open an issue on GitHub.

**Repository**: [github.com/sdodlapati3/BioPipelines](https://github.com/sdodlapati3/BioPipelines)

