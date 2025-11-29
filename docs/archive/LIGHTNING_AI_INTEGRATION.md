# Lightning.ai Integration Analysis for BioPipelines

## Executive Summary

Lightning.ai offers a unified API to access multiple LLM providers with **30 million free tokens per month** and **up to 70% off** during their Black Friday promotion. This could significantly reduce our LLM costs and simplify our multi-model ensemble architecture.

---

## Lightning.ai Offerings

### 1. Model API Access (Key Feature for BioPipelines)

**Free Tier:**
- 30 million tokens/month FREE
- No credit card required
- Single API key for ALL models
- Pay-as-you-go after free tier

**Key Benefits:**
- One API, multiple models (OpenAI, Anthropic, open-source)
- Compatible with OpenAI SDK (drop-in replacement)
- Pay per token, no infrastructure costs

### 2. GPU Cloud Access

| GPU | VRAM | Price/hr | Spot Price |
|-----|------|----------|------------|
| T4 | 16 GB | $0.19 | $0.57-$0.68 |
| L4 | 24 GB | $0.48 | $0.47-$0.56 |
| L40S | 48 GB | $2.89 | $1.68-$2.02 |
| A100 | 40 GB | $1.29 | - |
| A100 | 80 GB | $2.71 | $2.54-$3.05 |
| H100 | 80 GB | $1.99 | - |
| H200 | 141 GB | $3.50 | - |

**Free Tier GPU Access:**
- 15 free credits/month = ~80 GPU hours on spot instances
- T4 access unlimited
- A100/H100/H200 limited to 4 hour sessions

### 3. Open Source Tools

Lightning.ai provides several open-source libraries:
- **LitServe**: Custom inference engines for models/agents
- **LitAI**: LLM router and agent framework
- **LitGPT**: 20+ high-performance LLMs for training/finetuning
- **PyTorch Lightning**: Model training at scale

---

## Models Useful for BioPipelines

### For Intent Parsing (Replace/Augment BioMistral)

Based on the Lightning.ai model marketplace:

| Model | Use Case | Est. Cost (per 1M tokens) |
|-------|----------|---------------------------|
| **DeepSeek-V3** | Best for complex reasoning, code | ~$0.14 input / $0.28 output |
| **Llama 3.3 70B** | General purpose, open weights | ~$0.80 |
| **Mistral Large** | Good balance of speed/quality | ~$2.00 |
| **GPT-4o** | Highest quality | ~$2.50 input / $10 output |
| **Claude 3.5 Sonnet** | Best for scientific text | ~$3.00 input / $15 output |

### Recommended for BioPipelines

1. **Primary: DeepSeek-V3** (cheapest, excellent reasoning)
   - Great for JSON extraction
   - Strong scientific knowledge
   - Cost: ~$0.14-0.28 per 1M tokens

2. **Fallback: Llama 3.3 70B** (open weights)
   - Can run locally if needed
   - Good biological knowledge
   - Cost: ~$0.80 per 1M tokens

3. **Specialist: Claude 3.5 Sonnet** (when highest quality needed)
   - Excellent at scientific text understanding
   - Best for complex workflow generation
   - Cost: ~$3-15 per 1M tokens

---

## Cost Estimation for BioPipelines

### Current Usage Estimate

| Operation | Tokens/Query | Queries/Month | Monthly Tokens |
|-----------|--------------|---------------|----------------|
| Intent parsing | ~500 | 1000 | 500,000 |
| Workflow generation | ~2000 | 500 | 1,000,000 |
| Module creation | ~1500 | 200 | 300,000 |
| Chat interactions | ~300 | 2000 | 600,000 |
| **Total** | | | **2.4M tokens** |

### Cost Comparison

| Provider | Monthly Cost (2.4M tokens) |
|----------|---------------------------|
| Lightning.ai (DeepSeek) | **FREE** (under 30M free) |
| Lightning.ai (Llama 70B) | **FREE** (under 30M free) |
| OpenAI GPT-4o | ~$24 input + $96 output = **$120** |
| Anthropic Claude 3.5 | ~$7.20 input + $36 output = **$43** |
| Self-hosted BioMistral | ~$15-30 (GPU hours) |

**Recommendation:** Use Lightning.ai free tier with DeepSeek-V3 for most operations.

---

## Integration Plan

### Phase 1: Add Lightning.ai LLM Adapter

```python
# New file: src/workflow_composer/llm/lightning_adapter.py

from litai import LLM

class LightningAdapter(LLMAdapter):
    """Lightning.ai unified API adapter."""
    
    def __init__(self, model: str = "deepseek/deepseek-v3", api_key: str = None):
        self.client = LLM(
            model=model,
            api_key=api_key or os.environ.get("LIGHTNING_API_KEY")
        )
    
    def chat(self, messages, **kwargs):
        # Compatible with OpenAI message format
        response = self.client.chat(messages)
        return response
```

### Phase 2: Model Router

Create intelligent routing based on task:

| Task | Recommended Model | Fallback |
|------|-------------------|----------|
| Intent parsing | DeepSeek-V3 | Llama 70B |
| Workflow generation | Claude 3.5 | DeepSeek-V3 |
| Module creation | DeepSeek-V3 | Llama 70B |
| Chat | Llama 70B | DeepSeek-V3 |
| Code generation | DeepSeek-V3 | Claude 3.5 |

### Phase 3: Replace Self-Hosted BioMistral

With Lightning.ai free tier:
- No need for SLURM GPU jobs
- No waiting for GPU allocation
- Instant access to better models
- Zero infrastructure cost

---

## Updated Cost Estimation (for PreflightValidator)

### Realistic Resource Costs

Based on Lightning.ai + HPC pricing:

```python
# Updated cost estimation
RESOURCE_COSTS = {
    # LLM Costs (per 1M tokens)
    "llm_deepseek": 0.14,      # Lightning.ai DeepSeek
    "llm_llama_70b": 0.80,     # Lightning.ai Llama
    "llm_claude": 15.00,       # Lightning.ai Claude (output)
    
    # GPU Compute (per hour)
    "gpu_t4": 0.19,            # Lightning.ai T4
    "gpu_l4": 0.48,            # Lightning.ai L4  
    "gpu_a100": 1.29,          # Lightning.ai A100
    "hpc_cpu": 0.03,           # University HPC CPU
    "hpc_gpu_t4": 0.10,        # University HPC T4 (subsidized)
    
    # Storage (per GB/month)
    "storage_scratch": 0.00,   # HPC scratch (free)
    "storage_cloud": 0.023,    # S3/GCS
}

def estimate_workflow_cost(tools, samples, organism):
    """Estimate total cost for a workflow."""
    
    # LLM costs (parsing + generation)
    llm_tokens = 3000  # Typical workflow
    llm_cost = (llm_tokens / 1_000_000) * RESOURCE_COSTS["llm_deepseek"]
    
    # Compute costs
    hours = estimate_compute_hours(tools, samples)
    compute_cost = hours * RESOURCE_COSTS["hpc_cpu"]
    
    # Free tier consideration
    if llm_tokens < 30_000_000:  # Within free tier
        llm_cost = 0.0
    
    return {
        "llm_cost": llm_cost,
        "compute_cost": compute_cost,
        "total": llm_cost + compute_cost,
        "free_tier_eligible": llm_tokens < 30_000_000
    }
```

---

## Action Items

### Immediate (This Session)
1. ✅ Analyze Lightning.ai offerings
2. ⏳ Create Lightning adapter
3. ⏳ Update cost estimation in PreflightValidator
4. ⏳ Add model router for intelligent selection

### Short-term (Next Week)
1. Sign up for Lightning.ai free account
2. Test API with bioinformatics queries
3. Compare quality: DeepSeek vs BioMistral
4. Integrate into Gradio UI

### Long-term (Future)
1. Fine-tune open model on bioinformatics data
2. Deploy custom model on Lightning.ai
3. Build agentic workflow with LitAI

---

## Configuration

### Environment Variables

```bash
# Lightning.ai API key (get from https://lightning.ai)
export LIGHTNING_API_KEY="your-api-key"

# Default model (cheapest with good quality)
export LIGHTNING_DEFAULT_MODEL="deepseek/deepseek-v3"

# Fallback model
export LIGHTNING_FALLBACK_MODEL="meta-llama/llama-3.3-70b"
```

### Config File Update (config/composer.yaml)

```yaml
llm:
  default_provider: lightning
  
  lightning:
    api_key: ${LIGHTNING_API_KEY}
    models:
      intent_parsing: deepseek/deepseek-v3
      workflow_generation: anthropic/claude-3.5-sonnet
      module_creation: deepseek/deepseek-v3
      chat: meta-llama/llama-3.3-70b
    fallback: meta-llama/llama-3.3-70b
    free_tier_limit: 30000000  # 30M tokens
```

---

## Conclusion

Lightning.ai provides an excellent opportunity to:

1. **Eliminate LLM hosting costs** - 30M free tokens/month covers typical usage
2. **Simplify architecture** - One API replaces multiple providers
3. **Access better models** - DeepSeek-V3, Claude 3.5, GPT-4o all available
4. **Scale easily** - Pay-as-you-go beyond free tier

**Recommended Strategy:**
- Use Lightning.ai as primary LLM provider
- Keep BERT models for local entity extraction (CPU, no API cost)
- Fall back to local Ollama for offline/air-gapped scenarios
