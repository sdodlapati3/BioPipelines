# Lightning.ai Setup Instructions

## Getting Your Free API Key

1. **Visit**: https://lightning.ai/models
2. **Sign up** for a free account
3. **Verify** your email address
4. **Create** a teamspace (even for free tier)
5. **Generate** an API key from your dashboard

## Free Tier Benefits

✅ **30 million tokens/month** - No credit card required!  
✅ Access to **all models**: Llama, DeepSeek, Mistral, GPT-4o, Claude  
✅ **15 free credits/month** for GPU Studios (~80 GPU hours)  
✅ OpenAI-compatible API for easy integration  

## Current Status

Your API key has been saved to `.secrets/lightning_key` but needs activation:

```bash
# Check if key is loaded
python -c "from workflow_composer import secrets; print(secrets.check_secrets())"

# Test the adapter (once activated)
python -c "
from workflow_composer.llm import LightningAdapter, Message
llm = LightningAdapter()
response = llm.chat([Message.user('Hello!')])
print(response.content)
"
```

## Activation Checklist

- [ ] Account created at lightning.ai
- [ ] Email verified
- [ ] Teamspace created (Settings → Billing)
- [ ] API key generated (Settings → API Keys)
- [ ] Key saved to `.secrets/lightning_key` ✅ (Already done!)
- [ ] Test successful: `Error 401` = needs activation, `Response: ...` = working!

## Usage in Code

Once activated, Lightning.ai will be available in:

### Gradio UI
- **Provider dropdown**: ⚡ Lightning.ai (30M FREE tokens!)
- Models auto-selected by task

### Python API
```python
from workflow_composer.llm import get_llm, LightningAdapter

# Via factory (recommended)
llm = get_llm('lightning')  # Uses DeepSeek-V3 by default

# Task-specific model selection
llm = LightningAdapter(task='workflow_generation')  # Auto-picks best model

# Specific model
llm = LightningAdapter(model='meta-llama/Llama-3.1-8B-Instruct')
```

## Recommended Models

| Task | Model | Cost per 1M tokens |
|------|-------|-------------------|
| Workflow generation | DeepSeek-V3 | $0.14 in / $0.28 out |
| Code generation | DeepSeek-V3 | $0.14 in / $0.28 out |
| Scientific analysis | Qwen2.5-72B | $0.80 in / $0.80 out |
| General chat | Llama-3.3-70B | $0.80 in / $0.80 out |
| Quick responses | Llama-3.1-8B | $0.10 in / $0.10 out |

## Cost Comparison

**30M free tokens** = Approximately:
- 214,000 workflow generations (with DeepSeek-V3)
- 37,500 complex scientific analyses (with Qwen)
- **~6 months** of typical BioPipelines usage

**After free tier**: Still 50-70% cheaper than direct API access!

## Troubleshooting

### Error 401: Unauthorized
- ✅ API key saved correctly
- ❌ Key not activated on Lightning.ai
- **Fix**: Complete activation steps above

### Model not found
- Update model names: Use `deepseek-ai/DeepSeek-V3` not `deepseek/deepseek-v3`
- Check available models: https://lightning.ai/models

### Package errors
```bash
# Install required packages
pip install openai litai
```

## Next Steps

1. **Activate your key** at https://lightning.ai/models
2. **Test it**: Run test command above
3. **Use it**: Start generating workflows with 30M free tokens!

---

**Questions?** Check docs/LIGHTNING_AI_INTEGRATION.md for full details.
