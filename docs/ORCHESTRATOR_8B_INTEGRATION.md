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

---

## Future Work: Implementation Roadmap

This section documents planned enhancements for future implementation.

### Phase 1: Production Deployment

#### 1.1 Deploy vLLM Server with Orchestrator-8B

**Objective:** Set up production-ready vLLM server for Orchestrator-8B inference.

**Prerequisites:**
- GPU with 16GB+ VRAM (A100, A10G, RTX 4090, etc.)
- CUDA 12.1+ installed
- Python 3.10+

**Implementation Steps:**

```bash
# 1. Install vLLM
pip install vllm

# 2. Download model (optional, vLLM can do this automatically)
huggingface-cli download nvidia/Orchestrator-8B

# 3. Start vLLM server
vllm serve nvidia/Orchestrator-8B \
    --port 8000 \
    --host 0.0.0.0 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 24000 \
    --dtype bfloat16

# 4. For multi-GPU setup (if available)
vllm serve nvidia/Orchestrator-8B \
    --port 8000 \
    --tensor-parallel-size 2 \  # Use 2 GPUs
    --pipeline-parallel-size 1
```

**Configuration for BioPipelines:**

```python
# In config/orchestrator.yaml
orchestrator:
  enabled: true
  backend: vllm
  api_endpoint: "http://localhost:8000/v1"
  
  # User preferences
  prefer_local: true
  max_cost_per_query: 0.50
  optimize_for: balanced  # cost, speed, accuracy, balanced
  
  # Inference settings
  max_turns: 10
  temperature: 0.0
  
  # Fallback
  fallback_to_heuristics: true
```

**Systemd Service (for production):**

```ini
# /etc/systemd/system/orchestrator-vllm.service
[Unit]
Description=Orchestrator-8B vLLM Server
After=network.target

[Service]
Type=simple
User=biopipelines
WorkingDirectory=/opt/biopipelines
ExecStart=/opt/biopipelines/venv/bin/vllm serve nvidia/Orchestrator-8B \
    --port 8000 \
    --gpu-memory-utilization 0.9
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Health Check Integration:**

```python
# Add to workflow_composer/llm/orchestrator_8b.py
async def health_check(self) -> Dict[str, Any]:
    """Check if Orchestrator-8B server is healthy."""
    try:
        response = await self._client.models.list()
        return {
            "status": "healthy",
            "model": self.config.model_name,
            "backend": self.config.inference_backend
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
```

---

### Phase 2: Data Collection for Fine-tuning

#### 2.1 Collect Routing Data

**Objective:** Gather BioPipelines-specific query → routing decision pairs for fine-tuning.

**Data Schema:**

```python
@dataclass
class RoutingTrainingExample:
    """Training example for fine-tuning orchestrator."""
    query: str                    # User query
    analysis_type: str            # Detected analysis type
    optimal_tier: ModelTier       # Best tier for this query
    optimal_tools: List[str]      # Tools that should be used
    actual_cost: float            # Actual cost incurred
    success: bool                 # Did the workflow succeed?
    latency_ms: float             # Total latency
    user_satisfaction: int        # 1-5 rating (optional)
```

**Collection Implementation:**

```python
# Add to OrchestratedSupervisor
class RoutingDataCollector:
    """Collects routing decisions for fine-tuning."""
    
    def __init__(self, output_path: str = "data/routing_training.jsonl"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log_decision(
        self,
        query: str,
        decision: RoutingDecision,
        result: OrchestratedResult,
        feedback: Optional[int] = None
    ):
        """Log a routing decision with outcome."""
        example = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "decision": {
                "target_model": decision.target_model,
                "target_tier": decision.target_tier.value,
                "tools_planned": decision.tool_calls_planned,
                "estimated_cost": decision.estimated_cost,
                "reasoning": decision.reasoning
            },
            "outcome": {
                "success": result.success,
                "actual_cost": result.metadata.get("cost", 0),
                "models_used": result.metadata.get("models_used", []),
                "tier_used": result.metadata.get("tier_used", "unknown")
            },
            "feedback": feedback
        }
        
        with open(self.output_path, "a") as f:
            f.write(json.dumps(example) + "\n")
```

**Recommended Collection Period:**
- Minimum: 2 weeks of production usage
- Ideal: 1000+ diverse queries
- Include: Success/failure outcomes, user feedback if available

---

### Phase 3: Fine-tuning (Optional)

#### 3.1 Fine-tune for BioPipelines

**Objective:** Improve routing accuracy for bioinformatics-specific queries.

**Training Data Format (ToolOrchestra-compatible):**

```json
{
  "instruction": "Generate a ChIP-seq workflow with MACS2 peak calling",
  "tools": ["workflow_planner", "code_generator", "nfcore_reference"],
  "preference": {
    "prefer_local": true,
    "optimize_for": "balanced"
  },
  "golden_routing": {
    "tier": "local_large",
    "model": "vllm/codellama-34b",
    "tools_sequence": [
      {"name": "nfcore_reference", "reason": "Get ChIP-seq module patterns"},
      {"name": "workflow_planner", "reason": "Design pipeline structure"},
      {"name": "code_generator", "reason": "Generate Nextflow code"}
    ]
  },
  "cost": 0.01,
  "success": true
}
```

**Fine-tuning Script (based on ToolOrchestra):**

```bash
# Clone ToolOrchestra training code
git clone https://github.com/NVlabs/ToolOrchestra
cd ToolOrchestra/training

# Prepare BioPipelines-specific training data
python prepare_data.py \
    --input ../data/routing_training.jsonl \
    --output ./biopipelines_train.json \
    --format toolorchestra

# Fine-tune (requires 16x H100 for full training, or use LoRA)
python resume_h100.py \
    --base_model nvidia/Orchestrator-8B \
    --train_data ./biopipelines_train.json \
    --output_dir ./biopipelines-orchestrator \
    --lora_rank 8 \  # Use LoRA for cheaper fine-tuning
    --learning_rate 1e-5 \
    --epochs 3
```

**LoRA Fine-tuning (Lower Resource Alternative):**

```python
# Using PEFT for efficient fine-tuning
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

# Requires only ~24GB GPU memory instead of 128GB+
```

---

### Phase 4: Production Monitoring

#### 4.1 Cost Savings Dashboard

**Metrics to Track:**

```python
class OrchestratorMetrics:
    """Metrics for orchestrator monitoring."""
    
    # Cost metrics
    total_cost_saved: float = 0.0
    queries_routed_local: int = 0
    queries_routed_cloud: int = 0
    
    # Routing metrics
    routing_accuracy: float = 0.0  # % of optimal routing decisions
    tier_distribution: Dict[str, int] = {}
    
    # Performance metrics
    avg_latency_ms: float = 0.0
    success_rate: float = 0.0
    fallback_rate: float = 0.0  # % using heuristics
```

**Prometheus Metrics (for production):**

```python
from prometheus_client import Counter, Histogram, Gauge

# Counters
orchestrator_queries_total = Counter(
    'orchestrator_queries_total',
    'Total queries processed',
    ['tier', 'success']
)

# Histograms
orchestrator_cost = Histogram(
    'orchestrator_query_cost_dollars',
    'Cost per query in dollars',
    buckets=[0.001, 0.01, 0.05, 0.10, 0.50, 1.0]
)

# Gauges
orchestrator_cost_savings = Gauge(
    'orchestrator_cost_savings_percent',
    'Percentage cost savings vs always-cloud'
)
```

**Weekly Report Template:**

```markdown
## Orchestrator-8B Weekly Report

### Cost Summary
- Total queries: {total_queries}
- Cost with orchestrator: ${actual_cost:.2f}
- Estimated cost without: ${baseline_cost:.2f}
- **Savings: ${savings:.2f} ({savings_pct:.1f}%)**

### Routing Distribution
- Local Small: {local_small_pct}%
- Local Large: {local_large_pct}%
- Cloud Small: {cloud_small_pct}%
- Cloud Large: {cloud_large_pct}%

### Performance
- Success rate: {success_rate:.1f}%
- Avg latency: {avg_latency_ms:.0f}ms
- Fallback rate: {fallback_rate:.1f}%

### Recommendations
- {recommendations}
```

---

### Implementation Priority

| Phase | Task | Priority | Effort | Dependencies |
|-------|------|----------|--------|--------------|
| 1.1 | Deploy vLLM server | P0 | 1 day | GPU available |
| 1.2 | Configure systemd service | P1 | 2 hours | Phase 1.1 |
| 2.1 | Implement data collector | P1 | 4 hours | None |
| 2.2 | Collect 1000+ examples | P1 | 2 weeks | Phase 2.1 |
| 3.1 | Fine-tune with LoRA | P2 | 1 day | Phase 2.2 |
| 4.1 | Add Prometheus metrics | P1 | 4 hours | None |
| 4.2 | Create dashboard | P2 | 1 day | Phase 4.1 |

---

### Notes

- The base Orchestrator-8B works well out-of-box for general queries
- Fine-tuning is optional but recommended after collecting 1000+ examples
- LoRA fine-tuning is cost-effective (single A100 vs 16x H100)
- Monitor fallback rate - if >20%, model loading may have issues

---

## Related Research: Puppeteer (Multi-Agent Evolving Orchestration)

### Paper Overview

**Title:** Multi-Agent Collaboration via Evolving Orchestration  
**Venue:** NeurIPS 2025  
**Authors:** Dang et al. (Tsinghua, SJTU, Tencent)  
**arXiv:** [2505.19591](https://arxiv.org/abs/2505.19591)  
**Code:** [github.com/OpenBMB/ChatDev/tree/puppeteer](https://github.com/OpenBMB/ChatDev/tree/puppeteer)

### Key Concepts

The "Puppeteer" paradigm proposes a **centralized orchestrator** that dynamically directs multiple LLM agents:

```
                    ┌─────────────────────────┐
                    │      Puppeteer          │
                    │  (Centralized Policy)   │
                    └───────────┬─────────────┘
                                │ Dynamically selects
           ┌────────────────────┼────────────────────┐
           │                    │                    │
    ┌──────▼──────┐      ┌──────▼──────┐     ┌──────▼──────┐
    │ Reasoning   │      │ Critique    │      │ Tool-use   │
    │   Agent     │      │   Agent     │      │   Agent    │
    └─────────────┘      └─────────────┘      └────────────┘
```

**Core Innovation:**
1. **Serialized Orchestration**: Unfolds multi-agent graph into sequential decisions
2. **Markov Decision Process**: $a_t \sim \pi(S_t, \tau)$ - select agent based on state
3. **RL-based Evolution**: REINFORCE with reward balancing accuracy and efficiency
4. **Emergent Topologies**: Learns compact, cyclic reasoning structures

### Reward Design

```
R_t = {
    r - λ·C_T,     if t = T (terminal)
    γ·R_{t+1} - λ·C_t,  if t < T
}

where:
- r ∈ {0,1} = task success
- λ = cost-accuracy tradeoff weight
- C_t = F·log(1 + t/φ) = step-wise cost
- γ = discount factor
```

### Comparison: ToolOrchestra vs Puppeteer

| Aspect | ToolOrchestra (NVIDIA) | Puppeteer (Tsinghua) |
|--------|------------------------|----------------------|
| **Focus** | Tool/Model routing | Agent sequencing |
| **Orchestrator** | Fixed 8B model | Learnable policy |
| **Agents** | Heterogeneous LLMs | Specialized reasoning patterns |
| **Training** | GRPO with preferences | REINFORCE |
| **Topology** | Implicit | Explicit graph evolution |
| **Cost Control** | Per-tool pricing | Step-wise penalty |
| **Key Insight** | Small models can route | Cyclic structures emerge |

### What BioPipelines Can Learn from Puppeteer

#### 1. **Emergent Cyclic Structures** ⭐⭐⭐

Puppeteer discovers that effective multi-agent reasoning naturally forms **cyclic topologies** - agents revisit previous collaborators for refinement:

```python
# Current BioPipelines flow (linear):
Planner → CodeGen → Validator → Docs

# Puppeteer-inspired (cyclic):
Planner → CodeGen → Validator ─┐
    ↑                          │
    └──── (if issues) ─────────┘
```

**Implementation Idea:**
```python
class CyclicOrchestrator:
    """Allow agents to revisit previous agents for refinement."""
    
    async def execute(self, query: str, max_cycles: int = 3):
        state = {"query": query, "cycle": 0}
        
        while state["cycle"] < max_cycles:
            # Forward pass
            plan = await self.planner.execute(state)
            code = await self.codegen.execute(plan)
            validation = await self.validator.execute(code)
            
            if validation.passed:
                break
            
            # Cyclic refinement
            state["feedback"] = validation.issues
            state["previous_code"] = code
            state["cycle"] += 1
        
        return code
```

#### 2. **Compaction Over Time** ⭐⭐⭐

The paper shows that as training progresses:
- Graph **density increases** (more focused agent interactions)
- **Hub agents** emerge (frequently activated experts)
- Unnecessary agents are **pruned**

**Application to BioPipelines:**
```python
# Track agent activation frequency
agent_stats = {
    "planner": {"activations": 0, "success_rate": 0.0},
    "codegen": {"activations": 0, "success_rate": 0.0},
    "validator": {"activations": 0, "success_rate": 0.0},
    "cloud_expert": {"activations": 0, "success_rate": 0.0}  # May be pruned
}

# Over time, learn which agents are essential
# Prune low-value agents, promote hub agents
```

#### 3. **Dynamic Agent Pool** ⭐⭐

Puppeteer uses diverse agent types:

| Agent Type | Reasoning Pattern | BioPipelines Equivalent |
|------------|-------------------|------------------------|
| `planning` | Decompose task | `PlannerAgent` |
| `reasoning` | Synthesize solutions | `CodeGenAgent` |
| `critique` | Identify flaws | `ValidatorAgent` |
| `reflect` | Analyze trajectory | Could add |
| `question` | Generate sub-questions | Could add |
| `modify` | Correct errors | Validation loop |
| `summarize` | Condense results | `DocAgent` |
| `conclude` | Final answer | Aggregation |

**Potential New Agents:**
```python
class ReflectAgent:
    """Analyze failed workflows and propose improvements."""
    
    async def reflect(self, failed_result: WorkflowResult) -> str:
        prompt = f"""
        The workflow failed with these issues:
        {failed_result.validation_issues}
        
        Previous code:
        {failed_result.code[:1000]}
        
        Diagnose the cause and propose a high-level fix.
        """
        return await self.llm.complete(prompt)

class QuestionAgent:
    """Generate clarifying sub-questions when query is ambiguous."""
    
    async def clarify(self, query: str) -> List[str]:
        # Generate questions like:
        # - What organism is this for?
        # - Single-end or paired-end reads?
        # - What reference genome version?
        pass
```

#### 4. **Serialized Orchestration as MDP** ⭐⭐

The key insight: Multi-agent collaboration can be modeled as sequential decisions:

```
State S_t = (query, current_plan, generated_code, validation_results, ...)
Action a_t = select_next_agent(S_t)
Reward R_t = accuracy - λ·cost
```

**Integration with Orchestrator-8B:**

Our current Orchestrator-8B decides which **model** to use. Puppeteer's insight suggests we could also learn which **agent sequence** is optimal:

```python
class SequenceOrchestrator:
    """Learn optimal agent sequences for different query types."""
    
    def __init__(self):
        self.policy = OrchestrationPolicy()  # Learnable
        
    async def orchestrate(self, query: str):
        state = self._init_state(query)
        trajectory = []
        
        while not state.is_terminal:
            # Policy decides next agent
            agent = self.policy.select_agent(state)
            
            # Execute agent
            output = await agent.execute(state.context)
            
            # Update state
            state = self._update_state(state, output)
            trajectory.append((state, agent, output))
        
        return trajectory[-1].output
```

### Implementation Roadmap

| Phase | Feature | From Paper | Priority | Effort |
|-------|---------|------------|----------|--------|
| 1 | Cyclic validation loop | Cyclic structures | P0 | Low |
| 2 | Agent activation tracking | Compaction analysis | P1 | Low |
| 3 | ReflectAgent for failures | Agent types | P1 | Medium |
| 4 | QuestionAgent for ambiguity | Agent types | P2 | Medium |
| 5 | Learnable agent sequencing | MDP formulation | P3 | High |

### Combining ToolOrchestra + Puppeteer

The two papers are **complementary**:

```
User Query
    │
    ▼
┌───────────────────────────────┐
│  Orchestrator-8B              │  ← ToolOrchestra: Which MODEL?
│  (Model/Tool Routing)         │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  Puppeteer Policy             │  ← Puppeteer: Which AGENT sequence?
│  (Agent Sequencing)           │
└───────────────┬───────────────┘
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
 Planner    CodeGen    Validator
    │           │           │
    └───────────┴───────────┘
                │
                ▼
            Result
```

**Unified Reward:**
```python
reward = accuracy_reward - λ_model * model_cost - λ_agent * agent_steps
```

### Code Reference

Puppeteer implementation available at:
- GitHub: [OpenBMB/ChatDev/tree/puppeteer](https://github.com/OpenBMB/ChatDev/tree/puppeteer)
- Key files: Agent prompts, policy training, topology analysis

### Citation

```bibtex
@inproceedings{puppeteer2025,
  title={Multi-Agent Collaboration via Evolving Orchestration},
  author={Dang, Yufan and Qian, Chen and Luo, Xueheng and others},
  booktitle={NeurIPS},
  year={2025}
}
```
