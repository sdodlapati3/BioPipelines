# Provider Framework Consolidation Summary

## Overview

This document summarizes the consolidation of three separate LLM abstraction layers into a single unified `providers/` package.

## Problem Statement

Before consolidation, the codebase had **3 separate LLM abstraction layers**:

1. **`llm/`** - Original adapters for workflow generation (8 files)
2. **`diagnosis/`** - Separate adapters for error diagnosis (gemini_adapter.py, lightning_adapter.py)
3. **`models/`** - Another model registry with yet more providers (providers/ subdirectory)

This caused:
- Lightning adapter duplicated 3 times
- vLLM adapter duplicated 2 times
- Gemini adapter duplicated 2 times
- No unified provider registry
- Inconsistent interfaces across layers
- Maintenance nightmare

## Solution

Created a new **unified `providers/` package** as the single source of truth.

### New Package Structure

```
src/workflow_composer/providers/
├── __init__.py          # Main exports and API
├── base.py              # BaseProvider ABC, Message, ProviderResponse
├── registry.py          # ProviderRegistry singleton with configs
├── router.py            # ProviderRouter with fallback logic
├── factory.py           # get_provider(), complete(), chat() helpers
├── lightning.py         # Lightning.ai implementation
├── gemini.py            # Google Gemini implementation
├── openai.py            # OpenAI implementation
├── anthropic.py         # Anthropic Claude implementation
├── ollama.py            # Local Ollama implementation
├── vllm.py              # Local vLLM server implementation
└── utils/
    ├── __init__.py
    ├── health.py        # Health checking utilities
    └── metrics.py       # Usage tracking
```

### Provider Priority

1. **Lightning.ai** - 30M FREE tokens/month
2. **Gemini** - FREE tier with rate limits
3. **Ollama** - Local, free
4. **vLLM** - Local GPU, free
5. **OpenAI** - Paid backup
6. **Anthropic** - Paid backup

## Usage

### Simple Completion (Auto-fallback)

```python
from workflow_composer.providers import complete

response = complete("Explain RNA-seq analysis")
print(response.content)
```

### Specific Provider

```python
from workflow_composer.providers import get_provider

gemini = get_provider("gemini")
response = gemini.complete("Debug this STAR error")
```

### Chat Conversation

```python
from workflow_composer.providers import chat, Message

messages = [
    Message.system("You are a bioinformatics expert."),
    Message.user("What causes OOM errors in STAR?"),
]
response = chat(messages)
```

### Check Available Providers

```python
from workflow_composer.providers import print_provider_status

print_provider_status()
# Output:
# ==================================================
# PROVIDER STATUS
# ==================================================
#   ✅ GEMINI (39ms)
#   ❌ LIGHTNING (166ms)
#   ❌ OLLAMA (2ms)
#   ✅ OPENAI (585ms)
#   ...
# Available: 2/6
```

## Backward Compatibility

The existing packages are preserved:

- **`llm/`** - Still works, used by Gradio app, CLI, and composer
- **`diagnosis/`** - Updated to prefer providers/ with fallback to llm/
- **`models/`** - Still available for local model configurations

New code should import from `providers/`:

```python
# New way (preferred)
from workflow_composer.providers import get_provider, complete, chat

# Old way (still works)
from workflow_composer.llm import get_llm, check_providers
```

## Key Classes

### ProviderResponse
```python
@dataclass
class ProviderResponse:
    content: str
    model: str
    provider: str
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0
    raw_response: Optional[dict] = None
```

### Message
```python
@dataclass
class Message:
    role: Role
    content: str
    
    @classmethod
    def system(cls, content: str) -> "Message"
    
    @classmethod
    def user(cls, content: str) -> "Message"
    
    @classmethod
    def assistant(cls, content: str) -> "Message"
```

### BaseProvider
```python
class BaseProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> ProviderResponse
    
    @abstractmethod
    def chat(self, messages: List[Message], **kwargs) -> ProviderResponse
    
    def is_available(self) -> bool
    
    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]
```

## Migration Plan

For gradual migration:

1. ✅ New unified providers/ package created
2. ✅ diagnosis/ updated to use providers/
3. 🔄 llm/ remains for backward compatibility
4. ⏳ Future: Update all imports to use providers/
5. ⏳ Future: Deprecate llm/ package

## Files Created

| File | Lines | Description |
|------|-------|-------------|
| `providers/__init__.py` | ~80 | Main API exports |
| `providers/base.py` | ~200 | Core classes |
| `providers/registry.py` | ~350 | Provider configs |
| `providers/router.py` | ~300 | Smart routing |
| `providers/factory.py` | ~280 | Convenience functions |
| `providers/lightning.py` | ~150 | Lightning.ai |
| `providers/gemini.py` | ~180 | Google Gemini |
| `providers/openai.py` | ~140 | OpenAI |
| `providers/anthropic.py` | ~180 | Anthropic |
| `providers/ollama.py` | ~130 | Ollama |
| `providers/vllm.py` | ~130 | vLLM |
| `providers/utils/health.py` | ~190 | Health checks |
| `providers/utils/metrics.py` | ~250 | Usage tracking |

**Total: ~2,560 lines of clean, modular code**

## Testing

```bash
# Test imports
python -c "from workflow_composer.providers import complete, get_provider, print_provider_status"

# Check provider status
python -c "from workflow_composer.providers import print_provider_status; print_provider_status()"

# Test backward compatibility
python -c "from workflow_composer.llm import get_llm, check_providers"
```

## Next Steps

1. Monitor usage of both packages
2. Add deprecation warnings to llm/ in future release
3. Eventually remove duplicate code from diagnosis/
4. Consider merging models/registry.py with providers/registry.py
