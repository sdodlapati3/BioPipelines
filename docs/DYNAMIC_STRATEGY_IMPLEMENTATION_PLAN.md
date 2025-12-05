# Dynamic Strategy Selection Implementation Plan
## BioPipelines Multi-Model Orchestration Enhancement

**Created:** December 5, 2025  
**Updated:** December 5, 2025 (v2.0 - Post Peer Review)  
**Status:** Planning (Revised After ChatGPT Peer Review)  
**Priority:** High  
**Estimated Effort:** ~3 weeks (simplified from 5-10 days after refinement)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Peer Review & Refined Strategy](#peer-review--refined-strategy) ← **NEW**
3. [Current State Analysis](#current-state-analysis)
4. [Gap Analysis (Revised)](#gap-analysis-revised-after-peer-review)
5. [Proposed Architecture](#proposed-architecture) (reference only)
6. [Implementation Phases](#implementation-phases) (reference only)
7. [Detailed Component Specifications](#detailed-component-specifications)
8. [Integration Points](#integration-points)
9. [Testing Strategy](#testing-strategy)
10. [Risk Assessment](#risk-assessment)
11. [Research Topics for Further Exploration](#research-topics-for-further-exploration)
12. [Implementation Timeline (Revised)](#implementation-timeline-revised) ← **UPDATED**
13. [Peer Review Synthesis](#appendix-peer-review-synthesis) ← **NEW**

---

## Executive Summary

### Goal

Implement a **dynamic strategy selection system** that:
1. Detects available hardware resources (GPUs, SLURM partitions, cloud APIs)
2. Loads pre-configured strategy profiles from YAML files
3. Allows users to select or override strategies at session start
4. Routes requests to appropriate models based on the active strategy

### Key Principle

> **"Configure once, run anywhere"** - The same BioPipelines codebase should work optimally whether deployed on 10× T4s, a single H100, or cloud-only mode.

### Success Criteria

| Metric | Target |
|--------|--------|
| Strategy selection time | < 2 seconds at session start |
| Profile switching | Zero code changes required |
| Fallback reliability | 99.9% request success rate |
| New profile creation | < 30 minutes by non-developer |

---

## Peer Review Integration & Refined Strategy

> **Review Date:** December 5, 2025  
> **Reviewer:** ChatGPT (GPT-4o)  
> **Synthesis:** Claude (Opus 4.5)

### Key Insight: Avoid Over-Engineering

The original plan proposed 4 new major components (ResourceDetector, StrategyProfile, StrategySelector, UnifiedRouter) as a parallel routing stack. The peer review correctly identified this as **scope creep for a 1-2 person team**.

### Adopted Recommendations

| Recommendation | Source | Action |
|---------------|--------|--------|
| Extend existing components, don't duplicate | ChatGPT | ✅ **Adopt** - Integrate into `ModelOrchestrator` + `TaskRouter` |
| 3-4 vLLM servers, not 10 specialized models | ChatGPT | ✅ **Adopt** - Generalist + Coder + Math + Embeddings |
| Static long-running vLLM (not dynamic SLURM) | ChatGPT | ✅ **Adopt** - Interactive use needs persistent servers |
| Skip Semantic Router (have UnifiedIntentParser) | ChatGPT | ⚠️ **Partial** - Skip for now, revisit for tool routing |
| LiteLLM as optional provider, not full migration | ChatGPT | ✅ **Adopt** - Add as `LiteLLMProvider` option |
| 2-stage routing (Task → Complexity) | ChatGPT | ✅ **Adopt** - Only for CODE_GEN, BIOMEDICAL |
| Add `allow_cloud` data governance flag | ChatGPT | ✅ **Adopt** - Critical for university PHI concerns |
| Debug routing logs | ChatGPT | ✅ **Adopt** - Essential for troubleshooting |

### Refined Recommendations (Claude's Additions)

| Recommendation | Rationale |
|---------------|-----------|
| Keep embeddings server (BGE-M3) | ChatGPT suggested 3 models; we need 4 including embeddings for RAG |
| Use Qwen2.5-Math-7B over Phi-3.5-mini | Significantly better math benchmarks; fits T4 with INT8 |
| RouteLLM for Phase 2 | 85% cost savings too significant to ignore; add after v1 stable |
| Profile immutability during session | Changing strategy mid-session causes state inconsistencies |
| Health-check based ResourceDetector | Simpler than GPU detection; just "is vLLM endpoint alive?" |

### Rejected Recommendations

| Recommendation | Why Rejected |
|---------------|--------------|
| "vLLM Semantic Router" | **Does not exist.** ChatGPT hallucinated this component. |
| Skip all domain models | BioMistral provides measurable quality gains for bio tasks |
| Store profile in SessionManager only | Need profile at orchestrator level, not just session |

### Revised Architecture

```
BEFORE (Over-Engineered):
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ ResourceDetector│───▶│ StrategyProfile │───▶│ StrategySelector│───▶│  UnifiedRouter  │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
        ↑                                                                    │
        │                     (4 NEW CLASSES)                                │
        │                                                                    ▼
        └──────────────────────────────────────────────────────────── [Response]

AFTER (Integrated):
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           EXISTING COMPONENTS (Extended)                             │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   config/llm_profiles.yaml  ───▶  ResourceDetector (NEW, simple)                    │
│            │                              │                                          │
│            ▼                              ▼                                          │
│   ┌─────────────────┐           ┌─────────────────┐                                 │
│   │ SessionManager  │◀──────────│  Profile Match  │                                 │
│   │ .strategy_profile│           └─────────────────┘                                 │
│   └────────┬────────┘                                                               │
│            │                                                                         │
│            ▼                                                                         │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐        │
│   │   TaskRouter    │───▶│ModelOrchestrator│───▶│CascadingProviderRouter  │        │
│   │ (task classify) │    │ (uses profile)  │    │ (fallback chains)       │        │
│   └─────────────────┘    └─────────────────┘    └─────────────────────────┘        │
│            │                      │                         │                       │
│            │         ┌────────────┼────────────┐           │                       │
│            │         ▼            ▼            ▼           ▼                       │
│            │    [T4 vLLM]   [T4 vLLM]   [T4 vLLM]   [Cloud APIs]                   │
│            │    Generalist    Coder       Math      DeepSeek/Claude                │
│            │                                                                        │
└─────────────────────────────────────────────────────────────────────────────────────┘

NEW CODE: ~300 lines (ResourceDetector + profile loading)
vs ORIGINAL PLAN: ~1500 lines (4 major new classes)
```

### Revised Model Deployment (4 Servers, Not 10)

| Server | Model | Quantization | VRAM | Tasks |
|--------|-------|--------------|------|-------|
| `t4-generalist` | Qwen2.5-7B-Instruct | AWQ (4-bit) | ~8GB | Intent, Docs, General |
| `t4-coder` | Qwen2.5-Coder-7B-Instruct | AWQ (4-bit) | ~8GB | CodeGen, Validation |
| `t4-math` | Qwen2.5-Math-7B-Instruct | INT8 | ~10GB | Math, Statistics |
| `t4-embeddings` | BAAI/bge-m3 | FP16 | ~4GB | RAG, Semantic Search |

**Optional (Phase 2):**
| Server | Model | Quantization | VRAM | Tasks |
|--------|-------|--------------|------|-------|
| `t4-bio` | BioMistral-7B | INT8 | ~10GB | Biomedical reasoning |
| `t4-safety` | Llama-Guard-3-1B | FP16 | ~3GB | Content moderation |

### Data Governance Pattern (New)

```yaml
# config/llm_profiles.yaml
routes:
  biomedical:
    model: "t4-bio"
    allow_cloud: false      # NEVER send to cloud - potential PHI
    fallback_to_cloud: false
    local_only_reason: "May contain sample IDs or clinical metadata"
  
  code_generation:
    model: "t4-coder"
    allow_cloud: true       # Safe - no sensitive data in code requests
    fallback_chain:
      - deepseek-v3
      - claude-3.5-sonnet
```

### 2-Stage Routing Pattern (Complexity-Aware)

Only for high-value tasks where complexity significantly affects quality:

```python
# Tasks that benefit from complexity-based routing
COMPLEXITY_ROUTING_ENABLED = {
    "code_generation",   # Simple script vs complex pipeline
    "biomedical",        # Basic lookup vs multi-step reasoning
}

# Phase 1: Simple heuristics
def estimate_complexity(query: str, task: str) -> str:
    """Returns 'low', 'medium', or 'high'."""
    indicators = {
        "high": ["optimize", "debug", "compare", "analyze multiple", 
                 "step by step", "complex", "advanced"],
        "medium": ["explain", "modify", "extend", "improve"],
    }
    query_lower = query.lower()
    
    if any(ind in query_lower for ind in indicators["high"]):
        return "high"
    if any(ind in query_lower for ind in indicators["medium"]):
        return "medium"
    if len(query.split()) > 100:  # Long queries often complex
        return "high"
    return "low"

# Phase 2: RouteLLM integration (future)
# from routellm import Controller
# complexity = routellm_controller.predict(query)
```

### Debug Routing Pattern (New)

```python
# Add to every routing decision
class RoutingDecision:
    """Captures full routing context for debugging."""
    task_type: str
    active_profile: str
    complexity: str  # low/medium/high
    primary_model: str
    fallback_chain: List[str]
    chosen_model: str
    chosen_provider: str
    fallback_depth: int  # 0 = primary, 1+ = fallback
    latency_ms: float
    reason: str  # Why this model was chosen

# Usage
if os.getenv("BIOPIPELINES_DEBUG_ROUTING"):
    logger.info(f"Routing: {decision.to_json()}")
```

---

## Current State Analysis

### What Exists ✅

```
src/workflow_composer/
├── llm/
│   ├── orchestrator.py          # ModelOrchestrator with Strategy enum
│   ├── strategies.py            # Strategy, StrategyConfig, PRESETS
│   ├── task_router.py           # TaskRouter with task classification
│   └── providers.py             # LocalProvider, CloudProvider
│
├── providers/
│   ├── router.py                # ProviderRouter (cloud cascade)
│   ├── t4_router.py             # T4ModelRouter (task-based routing)
│   ├── local_model_registry.py  # Model catalog access
│   └── registry.py              # ProviderRegistry
│
└── config/
    └── local_model_catalog.yaml # Model definitions with T4 compatibility
```

### Current Flow

```
User Request
     │
     ▼
ModelOrchestrator.__init__(strategy=Strategy.LOCAL_FIRST)  ← Fixed at startup
     │
     ├── Strategy determines: local vs cloud preference
     │
     ▼
LocalProvider or CloudProvider
     │
     ▼
Response
```

**Problem**: Strategy is fixed at `__init__`, no hardware detection, no profile loading.

---

## Gap Analysis (Revised After Peer Review)

### What We Need vs Original Plan

**Original Plan**: 6 new components (ResourceDetector, StrategyProfile, StrategySelector, UnifiedRouter, SessionManager, Strategy CLI)

**Revised Plan**: Extend 3 existing components, add 2 minimal new ones

### Revised Missing Components

| Component | Approach | Priority | Effort |
|-----------|----------|----------|--------|
| **`ResourceDetector`** | New but minimal - GPU/SLURM detection only | P0 | 0.5 days |
| **`StrategyConfig` extension** | Add `profile_name`, `allow_cloud` to existing dataclass | P0 | 0.5 days |
| **`ModelOrchestrator` extension** | Add `switch_strategy()`, resource-aware init | P0 | 1 day |
| **Profile YAML files** | 4-5 simple YAML configs in `config/strategies/` | P1 | 0.5 days |
| **CLI flag** | `--strategy` flag in entry points | P2 | 0.5 days |

**Total Effort: ~3 days** (down from 6.5 days)

### Components We DON'T Need (ChatGPT + Claude Agreement)

| Avoided Component | Why Not Needed |
|-------------------|----------------|
| ~~UnifiedRouter~~ | CascadingProviderRouter already does this |
| ~~SessionManager~~ | ModelOrchestrator can hold state |
| ~~StrategyProfile class~~ | Just use StrategyConfig + YAML |
| ~~Dynamic vLLM launcher~~ | Static long-running servers are better |
| ~~Semantic Router~~ | UnifiedIntentParser already handles this |
| ~~10 specialized models~~ | 4-5 models cover 95% of use cases |

### Integration Gaps

1. `T4ModelRouter` and `ProviderRouter` are not connected
2. No SLURM partition detection
3. No cloud API availability checking at startup
4. Strategy profiles are Python dicts (PRESETS), not external YAML

---

## Proposed Architecture

### ⚠️ Architecture Note (Post Peer Review)

The detailed architecture below was the **original plan** before ChatGPT's review. 
After the peer review (see "Peer Review & Refined Strategy" section above), we recommend 
a **simplified approach**:

1. **Don't create new `UnifiedRouter`** → Use `CascadingProviderRouter` 
2. **Don't create `SessionManager`** → Extend `ModelOrchestrator`
3. **Don't create complex `StrategyProfile`** → Use `StrategyConfig` + simple YAML

The detailed specs below remain as **reference architecture** showing one possible 
implementation, but the actual implementation should follow the Phase 1-3 approach 
in "Refined Architecture" above.

---

### Original High-Level Design (Reference Only)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SESSION INITIALIZATION                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │ ResourceDetector │───▶│ StrategySelector │───▶│  SessionManager  │       │
│  │                  │    │                  │    │                  │       │
│  │ - GPU detection  │    │ - Load profiles  │    │ - Active profile │       │
│  │ - SLURM check    │    │ - Match resources│    │ - Runtime state  │       │
│  │ - API key check  │    │ - User override  │    │ - Metrics        │       │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (StrategyProfile)
┌─────────────────────────────────────────────────────────────────────────────┐
│                           REQUEST ROUTING                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        UnifiedRouter                                  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │ Task        │  │ Route       │  │ Execute     │  │ Fallback    │  │   │
│  │  │ Classifier  │─▶│ Selector    │─▶│ Request     │─▶│ Handler     │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│              ┌─────────────────────┼─────────────────────┐                  │
│              ▼                     ▼                     ▼                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │   T4 vLLM Fleet  │  │   Cloud APIs     │  │   H100/L4 Local  │          │
│  │   (T4ModelRouter)│  │   (ProviderRouter)│  │   (LocalProvider)│          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Directory Structure (New Files)

```
src/workflow_composer/
├── strategy/                    # NEW: Strategy selection module
│   ├── __init__.py
│   ├── resource_detector.py     # Hardware/API detection
│   ├── profile.py               # StrategyProfile dataclass
│   ├── selector.py              # StrategySelector
│   ├── session.py               # SessionManager
│   └── unified_router.py        # UnifiedRouter
│
config/
├── strategies/                  # NEW: Strategy profile YAML files
│   ├── t4_hybrid.yaml           # 10× T4 + DeepSeek fallback
│   ├── h100_local.yaml          # Single H100, all local
│   ├── l4_t4_combined.yaml      # 4× L4 + 4× T4
│   ├── cloud_only.yaml          # No local GPUs
│   └── development.yaml         # Minimal, fast iteration
```

---

## Implementation Phases

### ⚠️ Phase Note (Post Peer Review)

The detailed implementation phases below were the **original plan**. After peer review, 
we recommend the **simplified 3-phase approach** from the "Refined Architecture" section:

1. **Week 1**: ResourceDetector + extend StrategyConfig + YAML profiles
2. **Week 2**: Extend ModelOrchestrator with `switch_strategy()` + wire T4ModelRouter
3. **Week 3**: Add complexity routing for CODE_GEN/BIO + debug logging

The detailed specs below remain as **reference** for understanding what each component does.

---

### Original Phase 1: Core Infrastructure (Days 1-2)

**Goal**: Build foundational components without breaking existing code.

#### 1.1 ResourceDetector

```python
# src/workflow_composer/strategy/resource_detector.py

@dataclass
class ResourceProfile:
    """Detected hardware and API resources."""
    
    # GPU Information
    gpu_available: bool
    gpu_type: Optional[str]  # "T4", "L4", "H100", "A100", None
    gpu_count: int
    gpu_memory_gb: float
    
    # SLURM Information
    slurm_available: bool
    slurm_partitions: List[str]  # ["t4flex", "h100flex", "a100flex"]
    current_partition: Optional[str]
    
    # Cloud API Availability
    cloud_apis: Dict[str, bool]  # {"deepseek": True, "openai": False, ...}
    
    # System Information
    hostname: str
    cpu_count: int
    memory_gb: float
    
    # Derived
    @property
    def deployment_mode(self) -> str:
        """Infer deployment mode: 'slurm', 'local', 'cloud_only'."""
        if self.slurm_available:
            return "slurm"
        elif self.gpu_available:
            return "local"
        else:
            return "cloud_only"


class ResourceDetector:
    """Detects available hardware and cloud resources."""
    
    def detect(self) -> ResourceProfile:
        """Run all detection methods and return profile."""
        return ResourceProfile(
            gpu_available=self._detect_gpu(),
            gpu_type=self._detect_gpu_type(),
            gpu_count=self._detect_gpu_count(),
            gpu_memory_gb=self._detect_gpu_memory(),
            slurm_available=self._detect_slurm(),
            slurm_partitions=self._detect_slurm_partitions(),
            current_partition=self._detect_current_partition(),
            cloud_apis=self._detect_cloud_apis(),
            hostname=socket.gethostname(),
            cpu_count=os.cpu_count() or 1,
            memory_gb=self._detect_system_memory(),
        )
    
    def _detect_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0 and bool(result.stdout.strip())
        except Exception:
            return False
    
    def _detect_gpu_type(self) -> Optional[str]:
        """Detect GPU type (T4, L4, H100, etc.)."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                gpu_name = result.stdout.strip().split("\n")[0]
                # Parse common GPU types
                if "T4" in gpu_name:
                    return "T4"
                elif "L4" in gpu_name:
                    return "L4"
                elif "H100" in gpu_name:
                    return "H100"
                elif "A100" in gpu_name:
                    return "A100"
                elif "V100" in gpu_name:
                    return "V100"
                return gpu_name  # Return full name if not recognized
        except Exception:
            pass
        return None
    
    def _detect_slurm(self) -> bool:
        """Check if running in SLURM environment."""
        return "SLURM_JOB_ID" in os.environ or shutil.which("squeue") is not None
    
    def _detect_slurm_partitions(self) -> List[str]:
        """Get available SLURM partitions."""
        try:
            result = subprocess.run(
                ["sinfo", "-h", "-o", "%P"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return [p.strip().rstrip("*") for p in result.stdout.strip().split("\n") if p.strip()]
        except Exception:
            pass
        return []
    
    def _detect_cloud_apis(self) -> Dict[str, bool]:
        """Check which cloud APIs have keys configured."""
        return {
            "deepseek": bool(os.getenv("DEEPSEEK_API_KEY")),
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
            "google": bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
            "groq": bool(os.getenv("GROQ_API_KEY")),
            "cerebras": bool(os.getenv("CEREBRAS_API_KEY")),
        }
```

#### 1.2 StrategyProfile

```python
# src/workflow_composer/strategy/profile.py

@dataclass
class ModelRoute:
    """Routing configuration for a single task category."""
    model: str                    # Model ID or HuggingFace path
    local: bool                   # True = use local vLLM, False = use cloud
    quantization: Optional[str]   # "fp16", "int8", "int4"
    max_tokens: int = 4096
    temperature: float = 0.7
    
    # Fallback chain
    fallback_model: Optional[str] = None
    fallback_local: bool = False


@dataclass  
class StrategyProfile:
    """Complete strategy configuration loaded from YAML."""
    
    # Metadata
    name: str
    description: str
    version: str = "1.0"
    
    # Hardware requirements
    min_gpu_memory_gb: float = 0
    required_gpu_types: List[str] = field(default_factory=list)  # Empty = any
    requires_slurm: bool = False
    required_cloud_apis: List[str] = field(default_factory=list)
    
    # Task routing configuration
    routes: Dict[str, ModelRoute] = field(default_factory=dict)
    
    # Default fallback (used when task-specific fallback not defined)
    default_fallback_model: str = "deepseek-v3"
    default_fallback_local: bool = False
    
    # Behavior settings
    prefer_local: bool = True
    max_retries: int = 2
    timeout_seconds: float = 60.0
    enable_caching: bool = True
    
    # Cost controls
    max_cost_per_request: Optional[float] = None
    monthly_budget_limit: Optional[float] = None
    
    def matches_resources(self, resources: ResourceProfile) -> Tuple[bool, str]:
        """Check if this profile is compatible with detected resources."""
        # Check GPU memory
        if resources.gpu_memory_gb < self.min_gpu_memory_gb:
            return False, f"Insufficient GPU memory: {resources.gpu_memory_gb}GB < {self.min_gpu_memory_gb}GB"
        
        # Check GPU type
        if self.required_gpu_types and resources.gpu_type not in self.required_gpu_types:
            return False, f"GPU type {resources.gpu_type} not in {self.required_gpu_types}"
        
        # Check SLURM
        if self.requires_slurm and not resources.slurm_available:
            return False, "SLURM required but not available"
        
        # Check cloud APIs
        for api in self.required_cloud_apis:
            if not resources.cloud_apis.get(api, False):
                return False, f"Cloud API '{api}' not configured"
        
        return True, "Compatible"
    
    def get_route(self, task: str) -> ModelRoute:
        """Get routing configuration for a task category."""
        if task in self.routes:
            return self.routes[task]
        # Return default route
        return ModelRoute(
            model=self.default_fallback_model,
            local=self.default_fallback_local,
        )
    
    @classmethod
    def from_yaml(cls, path: Path) -> "StrategyProfile":
        """Load profile from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        
        # Parse routes
        routes = {}
        for task, route_data in data.get("routes", {}).items():
            routes[task] = ModelRoute(**route_data)
        
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            min_gpu_memory_gb=data.get("hardware", {}).get("min_gpu_memory_gb", 0),
            required_gpu_types=data.get("hardware", {}).get("gpu_types", []),
            requires_slurm=data.get("hardware", {}).get("requires_slurm", False),
            required_cloud_apis=data.get("hardware", {}).get("required_cloud_apis", []),
            routes=routes,
            default_fallback_model=data.get("fallback", {}).get("model", "deepseek-v3"),
            default_fallback_local=data.get("fallback", {}).get("local", False),
            prefer_local=data.get("behavior", {}).get("prefer_local", True),
            max_retries=data.get("behavior", {}).get("max_retries", 2),
            timeout_seconds=data.get("behavior", {}).get("timeout_seconds", 60.0),
            enable_caching=data.get("behavior", {}).get("enable_caching", True),
            max_cost_per_request=data.get("cost", {}).get("max_per_request"),
            monthly_budget_limit=data.get("cost", {}).get("monthly_limit"),
        )
```

#### 1.3 Strategy Profile YAML Files

```yaml
# config/strategies/t4_hybrid.yaml
name: "T4 Hybrid"
description: "10× T4 GPUs locally + DeepSeek cloud fallback"
version: "1.0"

hardware:
  min_gpu_memory_gb: 16
  gpu_types: ["T4"]
  requires_slurm: true
  required_cloud_apis: ["deepseek"]

routes:
  intent_parsing:
    model: "meta-llama/Llama-3.2-3B-Instruct"
    local: true
    quantization: "fp16"
    max_tokens: 2048
    fallback_model: "deepseek-v3"
    fallback_local: false
  
  code_generation:
    model: "Qwen/Qwen2.5-Coder-7B-Instruct"
    local: true
    quantization: "int8"
    max_tokens: 4096
    fallback_model: "deepseek-v3"
    fallback_local: false
  
  code_validation:
    model: "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    local: true
    quantization: "fp16"
    max_tokens: 2048
  
  data_analysis:
    model: "microsoft/Phi-3.5-mini-instruct"
    local: true
    quantization: "fp16"
    max_tokens: 4096
  
  math_statistics:
    model: "Qwen/Qwen2.5-Math-7B-Instruct"
    local: true
    quantization: "int8"
    max_tokens: 2048
  
  biomedical:
    model: "BioMistral/BioMistral-7B"
    local: true
    quantization: "int8"
    fallback_model: "claude-3.5-sonnet"
    fallback_local: false
  
  documentation:
    model: "google/gemma-2-9b-it"
    local: true
    quantization: "int8"
    max_tokens: 8192
  
  embeddings:
    model: "BAAI/bge-m3"
    local: true
    quantization: "fp16"
  
  safety:
    model: "meta-llama/Llama-Guard-3-1B"
    local: true
    quantization: "fp16"
  
  orchestration:
    model: "deepseek-v3"
    local: false  # Cloud only - too complex for small models

fallback:
  model: "deepseek-v3"
  local: false

behavior:
  prefer_local: true
  max_retries: 2
  timeout_seconds: 60
  enable_caching: true

cost:
  max_per_request: 0.10
  monthly_limit: 50.00
```

```yaml
# config/strategies/cloud_only.yaml
name: "Cloud Only"
description: "No local GPUs - all requests go to cloud APIs"
version: "1.0"

hardware:
  min_gpu_memory_gb: 0
  gpu_types: []
  requires_slurm: false
  required_cloud_apis: ["deepseek"]

routes:
  intent_parsing:
    model: "deepseek-v3"
    local: false
  
  code_generation:
    model: "deepseek-v3"
    local: false
  
  code_validation:
    model: "deepseek-v3"
    local: false
  
  data_analysis:
    model: "deepseek-v3"
    local: false
  
  math_statistics:
    model: "deepseek-v3"
    local: false
  
  biomedical:
    model: "claude-3.5-sonnet"
    local: false
  
  documentation:
    model: "claude-3.5-sonnet"
    local: false
  
  embeddings:
    model: "openai/text-embedding-3-small"
    local: false
  
  safety:
    model: "deepseek-v3"
    local: false
  
  orchestration:
    model: "deepseek-v3"
    local: false

fallback:
  model: "gpt-4o"
  local: false

behavior:
  prefer_local: false
  max_retries: 3
  timeout_seconds: 120
  enable_caching: true

cost:
  max_per_request: 1.00
  monthly_limit: 200.00
```

### Phase 2: Strategy Selection (Days 3-4)

#### 2.1 StrategySelector

```python
# src/workflow_composer/strategy/selector.py

class StrategySelector:
    """Selects optimal strategy profile based on resources and user preferences."""
    
    def __init__(self, profiles_dir: Optional[Path] = None):
        self.profiles_dir = profiles_dir or Path(__file__).parent.parent.parent.parent / "config" / "strategies"
        self.detector = ResourceDetector()
        self.profiles: Dict[str, StrategyProfile] = {}
        self._load_profiles()
    
    def _load_profiles(self):
        """Load all strategy profiles from YAML files."""
        if not self.profiles_dir.exists():
            logger.warning(f"Profiles directory not found: {self.profiles_dir}")
            return
        
        for yaml_file in self.profiles_dir.glob("*.yaml"):
            try:
                profile = StrategyProfile.from_yaml(yaml_file)
                self.profiles[profile.name.lower().replace(" ", "_")] = profile
                logger.info(f"Loaded strategy profile: {profile.name}")
            except Exception as e:
                logger.error(f"Failed to load {yaml_file}: {e}")
    
    def detect_resources(self) -> ResourceProfile:
        """Detect current hardware resources."""
        return self.detector.detect()
    
    def find_compatible_profiles(self, resources: ResourceProfile) -> List[Tuple[str, StrategyProfile, str]]:
        """Find all profiles compatible with current resources."""
        compatible = []
        for name, profile in self.profiles.items():
            matches, reason = profile.matches_resources(resources)
            if matches:
                compatible.append((name, profile, reason))
        return compatible
    
    def recommend_profile(self, resources: ResourceProfile) -> Tuple[str, StrategyProfile]:
        """Recommend the best profile for current resources."""
        compatible = self.find_compatible_profiles(resources)
        
        if not compatible:
            # No compatible profiles - use cloud_only as ultimate fallback
            if "cloud_only" in self.profiles:
                return "cloud_only", self.profiles["cloud_only"]
            raise RuntimeError("No compatible strategy profiles found")
        
        # Priority order for recommendation
        priority = ["h100_local", "l4_t4_combined", "t4_hybrid", "cloud_only", "development"]
        
        for pname in priority:
            for name, profile, _ in compatible:
                if name == pname:
                    return name, profile
        
        # Return first compatible
        return compatible[0][0], compatible[0][1]
    
    def select(
        self,
        override_profile: Optional[str] = None,
        interactive: bool = False,
    ) -> Tuple[str, StrategyProfile, ResourceProfile]:
        """
        Select a strategy profile.
        
        Args:
            override_profile: Force use of specific profile
            interactive: Prompt user for confirmation/selection
        
        Returns:
            Tuple of (profile_name, StrategyProfile, ResourceProfile)
        """
        resources = self.detect_resources()
        
        # If override specified, use it
        if override_profile:
            if override_profile not in self.profiles:
                available = ", ".join(self.profiles.keys())
                raise ValueError(f"Unknown profile: {override_profile}. Available: {available}")
            profile = self.profiles[override_profile]
            matches, reason = profile.matches_resources(resources)
            if not matches:
                logger.warning(f"Profile '{override_profile}' may not be compatible: {reason}")
            return override_profile, profile, resources
        
        # Recommend best profile
        recommended_name, recommended_profile = self.recommend_profile(resources)
        
        if interactive:
            return self._interactive_select(resources, recommended_name, recommended_profile)
        
        return recommended_name, recommended_profile, resources
    
    def _interactive_select(
        self,
        resources: ResourceProfile,
        recommended_name: str,
        recommended_profile: StrategyProfile,
    ) -> Tuple[str, StrategyProfile, ResourceProfile]:
        """Interactive profile selection."""
        print("\n" + "="*60)
        print("BioPipelines Strategy Selection")
        print("="*60)
        
        print(f"\nDetected Resources:")
        print(f"  GPU: {resources.gpu_type or 'None'} × {resources.gpu_count}")
        print(f"  GPU Memory: {resources.gpu_memory_gb:.1f} GB")
        print(f"  SLURM: {'Yes' if resources.slurm_available else 'No'}")
        print(f"  Cloud APIs: {[k for k, v in resources.cloud_apis.items() if v]}")
        
        print(f"\nRecommended: {recommended_name} ({recommended_profile.description})")
        
        compatible = self.find_compatible_profiles(resources)
        print(f"\nCompatible profiles ({len(compatible)}):")
        for i, (name, profile, _) in enumerate(compatible, 1):
            marker = "→" if name == recommended_name else " "
            print(f"  {marker} [{i}] {name}: {profile.description}")
        
        print(f"\nPress Enter to use '{recommended_name}', or type profile name/number:")
        
        try:
            choice = input().strip()
        except EOFError:
            choice = ""
        
        if not choice:
            return recommended_name, recommended_profile, resources
        
        # Try as number
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(compatible):
                name = compatible[idx][0]
                return name, compatible[idx][1], resources
        except ValueError:
            pass
        
        # Try as name
        if choice in self.profiles:
            return choice, self.profiles[choice], resources
        
        print(f"Unknown selection '{choice}', using recommended.")
        return recommended_name, recommended_profile, resources
```

#### 2.2 SessionManager

```python
# src/workflow_composer/strategy/session.py

@dataclass
class SessionState:
    """Runtime state for the current session."""
    profile_name: str
    profile: StrategyProfile
    resources: ResourceProfile
    started_at: datetime
    request_count: int = 0
    total_cost: float = 0.0
    errors: int = 0
    
    # Model availability cache (updated by health checks)
    model_health: Dict[str, bool] = field(default_factory=dict)


class SessionManager:
    """Manages the active strategy session."""
    
    _instance: Optional["SessionManager"] = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._state: Optional[SessionState] = None
        self._selector = StrategySelector()
    
    @property
    def is_active(self) -> bool:
        """Check if a session is active."""
        return self._state is not None
    
    @property
    def state(self) -> SessionState:
        """Get current session state."""
        if not self._state:
            raise RuntimeError("No active session. Call start_session() first.")
        return self._state
    
    @property
    def profile(self) -> StrategyProfile:
        """Get current strategy profile."""
        return self.state.profile
    
    def start_session(
        self,
        profile_override: Optional[str] = None,
        interactive: bool = False,
    ) -> SessionState:
        """
        Start a new strategy session.
        
        Args:
            profile_override: Force specific profile
            interactive: Prompt for selection
        """
        profile_name, profile, resources = self._selector.select(
            override_profile=profile_override,
            interactive=interactive,
        )
        
        self._state = SessionState(
            profile_name=profile_name,
            profile=profile,
            resources=resources,
            started_at=datetime.now(),
        )
        
        logger.info(f"Session started with profile: {profile_name}")
        return self._state
    
    def end_session(self) -> Dict[str, Any]:
        """End the current session and return summary."""
        if not self._state:
            return {}
        
        summary = {
            "profile": self._state.profile_name,
            "duration_seconds": (datetime.now() - self._state.started_at).total_seconds(),
            "requests": self._state.request_count,
            "total_cost": self._state.total_cost,
            "errors": self._state.errors,
        }
        
        self._state = None
        logger.info(f"Session ended: {summary}")
        return summary
    
    def record_request(self, cost: float = 0.0, error: bool = False):
        """Record a request in the session."""
        if self._state:
            self._state.request_count += 1
            self._state.total_cost += cost
            if error:
                self._state.errors += 1


# Global accessor
def get_session() -> SessionManager:
    """Get the global session manager."""
    return SessionManager()
```

### Phase 3: Unified Routing (Days 5-6)

#### 3.1 UnifiedRouter

```python
# src/workflow_composer/strategy/unified_router.py

class UnifiedRouter:
    """
    Single entry point for all model routing.
    
    Combines T4ModelRouter, ProviderRouter, and LocalProvider
    under a unified interface driven by the active StrategyProfile.
    """
    
    def __init__(self, session: Optional[SessionManager] = None):
        self.session = session or get_session()
        
        # Lazy-loaded routers
        self._t4_router: Optional[T4ModelRouter] = None
        self._provider_router: Optional[ProviderRouter] = None
        self._local_provider: Optional[LocalProvider] = None
    
    @property
    def profile(self) -> StrategyProfile:
        """Get active strategy profile."""
        return self.session.profile
    
    @property
    def t4_router(self) -> T4ModelRouter:
        """Get or create T4 router."""
        if self._t4_router is None:
            self._t4_router = T4ModelRouter()
        return self._t4_router
    
    @property
    def provider_router(self) -> ProviderRouter:
        """Get or create provider router."""
        if self._provider_router is None:
            self._provider_router = ProviderRouter()
        return self._provider_router
    
    async def route(
        self,
        task: str,
        prompt: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Route a request based on task category and active profile.
        
        Args:
            task: Task category (intent_parsing, code_generation, etc.)
            prompt: The prompt to process
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            Response dictionary with content, model, latency, cost, etc.
        """
        route_config = self.profile.get_route(task)
        
        # Merge kwargs with route config defaults
        params = {
            "max_tokens": route_config.max_tokens,
            "temperature": route_config.temperature,
            **kwargs,
        }
        
        try:
            if route_config.local:
                result = await self._route_local(task, prompt, route_config, params)
            else:
                result = await self._route_cloud(prompt, route_config, params)
            
            self.session.record_request(cost=result.get("cost", 0))
            return result
            
        except Exception as e:
            logger.error(f"Primary route failed for {task}: {e}")
            
            # Try fallback
            if route_config.fallback_model:
                try:
                    if route_config.fallback_local:
                        result = await self._route_local_model(
                            route_config.fallback_model, prompt, params
                        )
                    else:
                        result = await self._route_cloud_model(
                            route_config.fallback_model, prompt, params
                        )
                    result["fallback_used"] = True
                    self.session.record_request(cost=result.get("cost", 0))
                    return result
                except Exception as e2:
                    logger.error(f"Fallback also failed: {e2}")
            
            self.session.record_request(error=True)
            raise
    
    async def _route_local(
        self,
        task: str,
        prompt: str,
        route_config: ModelRoute,
        params: Dict,
    ) -> Dict[str, Any]:
        """Route to local vLLM server."""
        # Map task to T4ModelRouter category
        task_category = self._map_task_to_category(task)
        
        result = await self.t4_router.complete(
            task=task_category,
            prompt=prompt,
            **params,
        )
        
        result["route_type"] = "local"
        result["model_config"] = route_config
        return result
    
    async def _route_cloud(
        self,
        prompt: str,
        route_config: ModelRoute,
        params: Dict,
    ) -> Dict[str, Any]:
        """Route to cloud provider."""
        response = self.provider_router.complete(
            prompt=prompt,
            preferred_model=route_config.model,
            **params,
        )
        
        return {
            "content": response.content,
            "model": response.model,
            "provider": response.provider,
            "cost": response.cost or 0,
            "latency_ms": response.latency_ms,
            "route_type": "cloud",
        }
    
    def _map_task_to_category(self, task: str) -> str:
        """Map profile task names to T4ModelRouter categories."""
        mapping = {
            "intent_parsing": "intent",
            "code_generation": "codegen",
            "code_validation": "validation",
            "data_analysis": "analysis",
            "math_statistics": "math",
            "biomedical": "biomedical",
            "documentation": "docs",
            "embeddings": "embeddings",
            "safety": "safety",
            "orchestration": "orchestration",
        }
        return mapping.get(task, task)
    
    # Convenience methods
    async def complete(self, prompt: str, task: str = "general", **kwargs):
        """Simple completion with auto task classification."""
        return await self.route(task, prompt, **kwargs)
    
    async def embed(self, texts: Union[str, List[str]]) -> Dict[str, Any]:
        """Generate embeddings."""
        return await self.route("embeddings", str(texts) if isinstance(texts, str) else "\n".join(texts))
```

### Phase 4: Integration & CLI (Days 7-8)

#### 4.1 CLI Commands

```python
# src/workflow_composer/cli/strategy_cli.py

import click
from ..strategy import get_session, StrategySelector, ResourceDetector

@click.group()
def strategy():
    """Strategy management commands."""
    pass

@strategy.command()
def detect():
    """Detect available hardware resources."""
    detector = ResourceDetector()
    resources = detector.detect()
    
    click.echo("\n" + "="*50)
    click.echo("Detected Resources")
    click.echo("="*50)
    click.echo(f"GPU Available: {resources.gpu_available}")
    click.echo(f"GPU Type: {resources.gpu_type or 'N/A'}")
    click.echo(f"GPU Count: {resources.gpu_count}")
    click.echo(f"GPU Memory: {resources.gpu_memory_gb:.1f} GB")
    click.echo(f"SLURM Available: {resources.slurm_available}")
    click.echo(f"SLURM Partitions: {', '.join(resources.slurm_partitions) or 'N/A'}")
    click.echo(f"Deployment Mode: {resources.deployment_mode}")
    click.echo("\nCloud APIs Configured:")
    for api, available in resources.cloud_apis.items():
        status = "✓" if available else "✗"
        click.echo(f"  {status} {api}")

@strategy.command()
def list():
    """List available strategy profiles."""
    selector = StrategySelector()
    
    click.echo("\n" + "="*50)
    click.echo("Available Strategy Profiles")
    click.echo("="*50)
    
    for name, profile in selector.profiles.items():
        click.echo(f"\n{name}:")
        click.echo(f"  Description: {profile.description}")
        click.echo(f"  Min GPU Memory: {profile.min_gpu_memory_gb} GB")
        click.echo(f"  GPU Types: {profile.required_gpu_types or 'Any'}")
        click.echo(f"  Requires SLURM: {profile.requires_slurm}")

@strategy.command()
@click.option("--profile", "-p", help="Force specific profile")
@click.option("--interactive", "-i", is_flag=True, help="Interactive selection")
def start(profile, interactive):
    """Start a strategy session."""
    session = get_session()
    state = session.start_session(
        profile_override=profile,
        interactive=interactive,
    )
    
    click.echo(f"\n✓ Session started with profile: {state.profile_name}")
    click.echo(f"  Description: {state.profile.description}")

@strategy.command()
def status():
    """Show current session status."""
    session = get_session()
    if not session.is_active:
        click.echo("No active session.")
        return
    
    state = session.state
    click.echo("\n" + "="*50)
    click.echo("Current Session")
    click.echo("="*50)
    click.echo(f"Profile: {state.profile_name}")
    click.echo(f"Started: {state.started_at}")
    click.echo(f"Requests: {state.request_count}")
    click.echo(f"Total Cost: ${state.total_cost:.4f}")
    click.echo(f"Errors: {state.errors}")
```

---

## Integration Points

### 1. Modify ModelOrchestrator

```python
# In src/workflow_composer/llm/orchestrator.py

class ModelOrchestrator:
    def __init__(
        self,
        strategy: Strategy = Strategy.AUTO,
        config: Optional[StrategyConfig] = None,
        use_unified_router: bool = True,  # NEW
        ...
    ):
        if use_unified_router:
            # Use new strategy-aware routing
            self._router = UnifiedRouter()
        else:
            # Legacy mode
            self.local = local_provider or LocalProvider()
            self.cloud = cloud_provider or CloudProvider()
```

### 2. Update Chat Agent Initialization

```python
# In chat agent startup code

from workflow_composer.strategy import get_session

def initialize_agent():
    # Start strategy session (with interactive selection)
    session = get_session()
    session.start_session(interactive=True)
    
    # Now create orchestrator - it will use the active session
    orchestrator = ModelOrchestrator()
```

---

## Testing Strategy

### Unit Tests

```python
# tests/strategy/test_resource_detector.py

def test_gpu_detection_with_gpu(mocker):
    mocker.patch("subprocess.run", return_value=Mock(
        returncode=0,
        stdout="Tesla T4\n"
    ))
    detector = ResourceDetector()
    resources = detector.detect()
    assert resources.gpu_available
    assert resources.gpu_type == "T4"

def test_gpu_detection_without_gpu(mocker):
    mocker.patch("subprocess.run", side_effect=FileNotFoundError)
    detector = ResourceDetector()
    resources = detector.detect()
    assert not resources.gpu_available
```

### Integration Tests

```python
# tests/strategy/test_integration.py

@pytest.mark.integration
def test_full_session_flow():
    session = get_session()
    
    # Start session
    state = session.start_session(profile_override="t4_hybrid")
    assert state.profile_name == "t4_hybrid"
    
    # Create router
    router = UnifiedRouter(session)
    
    # Route a request (mocked)
    result = await router.route("intent_parsing", "Test prompt")
    assert "content" in result
    
    # End session
    summary = session.end_session()
    assert summary["requests"] == 1
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Profile YAML syntax errors | Medium | High | Validate on load, provide schema |
| GPU detection fails | Low | Medium | Graceful fallback to cloud_only |
| Session state corruption | Low | High | Singleton pattern, clear state methods |
| Performance overhead | Low | Low | Lazy loading, caching |
| Backwards compatibility | Medium | Medium | Keep legacy mode as fallback |

---

## Research Topics for Further Exploration

### 1. **LiteLLM Integration**

LiteLLM provides a battle-tested routing layer. Consider:
- Using LiteLLM as the underlying router instead of ProviderRouter
- Benefits: 100+ provider support, built-in retries, cost tracking
- Concern: Additional dependency, may be overkill

**Research questions:**
- Can LiteLLM route to local vLLM servers?
- How does LiteLLM handle task-based routing (not just load balancing)?

### 2. **Model Context Protocol (MCP)**

Anthropic's MCP allows LLMs to access external tools/resources. For BioPipelines:
- Could MCP servers expose bioinformatics tools directly?
- Strategy profiles could define which MCP servers to connect

**Research:**
- https://modelcontextprotocol.io/
- Is MCP mature enough for production?

### 3. **Semantic Router**

[Semantic Router](https://github.com/aurelio-labs/semantic-router) by Aurelio Labs:
- Uses embeddings to classify queries into categories
- Could replace keyword-based task classification
- More robust to variations in user input

**Integration idea:**
```python
from semantic_router import Route, SemanticRouter

intent_route = Route(name="intent", utterances=["parse this query", "understand what I mean", ...])
code_route = Route(name="code", utterances=["write a script", "generate code", ...])

router = SemanticRouter(routes=[intent_route, code_route])
task = router(query)  # Returns "intent" or "code"
```

### 4. **Mixture of Experts (MoE) Patterns**

Modern MoE models (DeepSeek-V3, Mixtral) use sparse expert routing. Apply same concept:
- Each "expert" is a specialized small model
- Router selects 1-2 experts per query
- Reduces compute while maintaining quality

**Research:**
- How do MoE models decide which experts to activate?
- Can we learn routing weights from query patterns?

### 5. **Cost-Aware Routing (RouteLLM)**

[RouteLLM](https://github.com/lm-sys/RouteLLM) by LMSYS:
- Routes queries between strong (expensive) and weak (cheap) models
- Learns routing from preference data
- Could reduce costs 50%+ with minimal quality loss

**Integration idea:**
```python
from routellm import RouteLLMController

controller = RouteLLMController(
    strong_model="deepseek-v3",
    weak_model="llama-3.2-3b",
    threshold=0.5  # Route to strong if confidence < 0.5
)

response = controller.chat(messages)  # Auto-selects model
```

### 6. **Speculative Decoding**

Use small model to generate draft, large model to verify:
- Small local model (Llama-3.2-3B) generates candidate tokens
- Large cloud model (DeepSeek-V3) accepts/rejects in batch
- Reduces latency while maintaining quality

**BioPipelines application:**
- Use T4 models for initial generation
- Route to cloud for verification only when needed

### 7. **Adaptive Batch Sizing**

Dynamic batching based on:
- Current GPU memory usage
- Queue depth
- Time-of-day patterns

**Research:**
- vLLM already does continuous batching - can we tune it?
- SLURM job priority vs batch size tradeoffs

### 8. **Federated Model Selection**

If you scale to multiple universities/clusters:
- Each site has different hardware
- Central registry of model capabilities
- Query routes to nearest capable site

**Probably overkill for now**, but interesting for future.

---

## Deep Research Analysis

### RouteLLM - Cost-Quality Routing (In-Depth)

**Source:** https://github.com/lm-sys/RouteLLM (4.5k⭐)  
**Paper:** [arXiv:2406.18665](https://arxiv.org/abs/2406.18665)

#### Key Insights

RouteLLM addresses the exact problem we face: routing between expensive high-quality models and cheaper models while maintaining quality.

**Core Concept:**
```
User Query → Router (ML model) → Prediction: "Strong or Weak?"
     │                                    │
     │         ┌─────────────────────────┐│
     ▼         ▼                         ▼▼
    Strong Model (expensive)    OR    Weak Model (cheap)
    (GPT-4, Claude-3.5)              (Mixtral, Llama)
```

**Router Types Available:**
1. **Matrix Factorization (`mf`)** - Recommended, lightweight
2. **Similarity-Weighted Ranking (`sw_ranking`)** - Uses Elo scores
3. **BERT Classifier (`bert`)** - BERT-based classification
4. **Causal LLM (`causal_llm`)** - LLM-based routing

**Reported Results:**
- MT Bench: 85% cost reduction while maintaining 95% GPT-4 quality
- MMLU: 45% cost reduction at 95% performance
- GSM8K: 35% cost reduction at 95% performance

**BioPipelines Integration Pattern:**
```python
from routellm.controller import Controller
import os

# RouteLLM can replace our cloud cascade
controller = Controller(
    routers=["mf"],  # Matrix factorization router
    strong_model="deepseek-v3",
    weak_model="local/qwen2.5-coder-7b",  # Our T4-hosted model
)

# Threshold controls cost-quality tradeoff
# Lower threshold = more strong model calls = higher quality
# Higher threshold = more weak model calls = lower cost
response = controller.chat.completions.create(
    model="router-mf-0.11593",  # mf router with 0.11593 threshold
    messages=[{"role": "user", "content": query}]
)
```

**Key Insight for BioPipelines:**
- RouteLLM uses threshold calibration: you specify "I want 50% calls to strong model"
- It returns the threshold value to achieve that ratio
- This enables budget-aware routing

**Threshold Calibration:**
```bash
python -m routellm.calibrate_threshold --routers mf --strong-model-pct 0.5
# Output: For 50.0% strong model calls for mf, threshold = 0.11593
```

**Adaptation for Task-Based Routing:**
Our system differs because we route by TASK, not just query complexity:
```python
# BioPipelines hybrid approach
class HybridRouter:
    def route(self, task: str, query: str):
        # Step 1: Task determines MODEL CAPABILITY requirements
        required_capability = self.task_capabilities[task]
        
        # Step 2: Within capability tier, use RouteLLM for cost optimization
        if required_capability == "high":
            # Always use strong model (biomedical, orchestration)
            return self.strong_model
        elif required_capability == "medium":
            # Use RouteLLM to decide (code_gen, analysis)
            return self.routellm.route(query)
        else:
            # Always use weak model (intent, safety check)
            return self.weak_model
```

---

### Semantic Router - Embedding-Based Classification

**Source:** https://github.com/aurelio-labs/semantic-router (2.9k⭐)  
**Docs:** https://docs.aurelio.ai/semantic-router

#### Why This Matters for BioPipelines

Our current task classification uses keyword matching or simple heuristics. Semantic Router provides **semantic understanding** of user queries.

**Current Approach (Fragile):**
```python
if "code" in query.lower() or "write" in query.lower():
    task = "code_generation"
elif "analyze" in query.lower():
    task = "data_analysis"
# Fails: "I need a Python script" → misses "code"
```

**Semantic Router Approach (Robust):**
```python
from semantic_router import Route, SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder

# Define routes with example utterances
code_route = Route(
    name="code_generation",
    utterances=[
        "write a Python script",
        "generate code for",
        "I need a function that",
        "can you code this up",
        "implement an algorithm",
    ]
)

bio_route = Route(
    name="biomedical",
    utterances=[
        "analyze this FASTQ file",
        "what does this gene do",
        "run differential expression",
        "variant calling pipeline",
    ]
)

# Use local embedding model (runs on T4)
encoder = HuggingFaceEncoder(name="BAAI/bge-m3")
router = SemanticRouter(
    encoder=encoder,
    routes=[code_route, bio_route, ...]
)

# Now queries are matched by semantic similarity
task = router("I need to create a snakemake workflow")  # → "code_generation"
task = router("What variants are in this VCF")  # → "biomedical"
```

**Performance:** < 10ms per routing decision (embedding lookup)

**Integration with StrategyProfile:**
```yaml
# config/strategies/t4_hybrid.yaml
classification:
  method: "semantic_router"  # or "keyword" for legacy
  embedding_model: "BAAI/bge-m3"
  routes:
    code_generation:
      utterances_file: "config/utterances/code_gen.txt"
    biomedical:
      utterances_file: "config/utterances/biomedical.txt"
```

---

### Speculative Decoding & Medusa

**Sources:**
- DeepMind Speculative Sampling: [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
- HuggingFace Assisted Generation: https://huggingface.co/blog/assisted-generation
- Medusa: https://github.com/FasterDecoding/Medusa (2.7k⭐)

#### Concept

Instead of generating one token at a time with a large model:
1. **Draft Model** (small, fast): Generates N candidate tokens
2. **Target Model** (large, accurate): Verifies/rejects candidates in ONE forward pass
3. **Accept verified prefix**, repeat

**Why This Matters for BioPipelines:**
- Our T4 models could serve as draft models
- Cloud models verify only when needed
- 2-3x speedup without quality loss

**Medusa Variant (Self-Speculation):**
- Adds extra "heads" to predict future tokens
- No separate draft model needed
- Can achieve 2.2-3.6x speedup

**vLLM Support:**
vLLM v0.12+ has native speculative decoding:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --speculative-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --num-speculative-tokens 5
```

**BioPipelines Application:**
```yaml
# T4 node configuration
servers:
  - name: "code_gen_speculative"
    main_model: "Qwen/Qwen2.5-Coder-7B-Instruct"
    draft_model: "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    speculative_tokens: 5
    # Uses 7B for quality, 1.5B for speed
```

---

### LiteLLM vs Custom Router - Comparison

| Feature | LiteLLM | Our Custom Router |
|---------|---------|-------------------|
| **Provider Support** | 100+ providers | DeepSeek, OpenAI, Anthropic, Groq, Cerebras |
| **Load Balancing** | ✅ Multiple strategies | ❌ Not implemented |
| **Fallback** | ✅ Automatic | ✅ Manual cascade |
| **Cost Tracking** | ✅ Built-in | ❌ Basic |
| **Task Routing** | ❌ | ✅ Core feature |
| **Local Models** | ✅ OpenAI-compatible | ✅ vLLM native |
| **Dependency Weight** | ~20 deps | 0 (custom) |
| **Customization** | Config-driven | Code-driven |

**Recommendation:** Use LiteLLM as the *underlying* provider layer, build task routing on top.

```python
# Proposed architecture
class UnifiedRouter:
    def __init__(self):
        # LiteLLM handles provider complexity
        self.litellm_router = litellm.Router(
            model_list=[
                {"model_name": "local-coder", "litellm_params": {"model": "openai/local", "api_base": "http://t4-coder:8000"}},
                {"model_name": "deepseek", "litellm_params": {"model": "deepseek/deepseek-chat"}},
                {"model_name": "claude", "litellm_params": {"model": "claude-3-5-sonnet-20241022"}},
            ]
        )
        
        # Our layer handles task classification
        self.task_classifier = SemanticRouter(...)
    
    async def route(self, query: str):
        task = self.task_classifier(query)  # "code_gen", "biomedical", etc.
        model = self.profile.get_route(task)  # Get model for task
        return await self.litellm_router.acompletion(model=model, messages=[...])
```

---

### vLLM Advanced Features (v0.12+)

Based on the vLLM documentation structure, key features for BioPipelines:

1. **Automatic Prefix Caching:**
   - Caches KV for repeated prompt prefixes
   - Useful for system prompts in chat
   - Enable: `--enable-prefix-caching`

2. **Speculative Decoding:**
   - Built-in draft model support
   - MLP Speculator for self-speculation
   - Medusa heads support

3. **Multi-LoRA Inference:**
   - Switch LoRA adapters per request
   - Could fine-tune for bioinformatics tasks
   - Enable: `--enable-lora`

4. **Disaggregated Prefilling:**
   - Separate prefill and decode across GPUs
   - Better for long-context inputs
   - Experimental but promising

5. **Quantized KV Cache:**
   - FP8 or INT8 KV cache
   - Reduces memory for longer contexts
   - Enable: `--kv-cache-dtype fp8`

---

### Recommended Integration Priority

Based on research, prioritized integration path:

| Priority | Technology | Effort | Impact | Recommendation |
|----------|------------|--------|--------|----------------|
| P1 | **LiteLLM** | 1 day | High | Replace ProviderRouter |
| P1 | **Semantic Router** | 1 day | High | Replace keyword classification |
| P2 | **RouteLLM** | 2 days | Medium | Add cost-aware routing layer |
| P2 | **vLLM Speculative** | 1 day | Medium | Enable on T4 servers |
| P3 | **Prefix Caching** | 0.5 day | Medium | Enable in vLLM config |
| P4 | **Medusa** | 3 days | Low | Requires training heads |

---

## Implementation Timeline (Revised)

### After Peer Review: Simplified 3-Week Plan

| Week | Focus | Deliverables |
|------|-------|--------------|
| **Week 1** | Foundation | ResourceDetector (minimal), extend StrategyConfig, 4 YAML profiles |
| **Week 2** | Integration | ModelOrchestrator.switch_strategy(), wire T4ModelRouter |
| **Week 3** | Polish | Complexity routing (CODE_GEN/BIO), debug logging, tests |

### Original 4-Week Plan (Reference)

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | Core (Phase 1-2) | ResourceDetector, StrategyProfile, Selector |
| 2 | Routing (Phase 3) | UnifiedRouter, Session integration |
| 3 | Polish (Phase 4) | CLI, tests, documentation |
| 4 | Research | Evaluate LiteLLM, RouteLLM, SemanticRouter |

---

## Next Steps (Updated After Peer Review)

### Immediate (This Week)
1. **Deploy 4 static vLLM servers** on T4 nodes:
   - Generalist: Qwen-2.5-7B-Instruct-AWQ
   - Coder: Qwen-Coder-7B-AWQ  
   - Math: Qwen-Math-7B-AWQ
   - Embeddings: BGE-M3

2. **Extend StrategyConfig** in `llm/strategies.py`:
   - Add `profile_name: Optional[str]`
   - Add `allow_cloud: bool = True` (data governance)
   - Add `debug_routing: bool = False`

3. **Create minimal ResourceDetector** (50 lines):
   - Just health-check vLLM endpoints
   - Check for cloud API keys

### Short-Term (Next 2 Weeks)
4. **Add `switch_strategy()` to ModelOrchestrator**
5. **Wire T4ModelRouter into CascadingProviderRouter**
6. **Create 4 YAML profiles** in `config/strategies/`

### Future (Month 2+)
7. **Add RouteLLM** for complexity routing (CODE_GEN, BIO)
8. **Evaluate speculative decoding** on T4 (Medusa heads)
9. **Prefix caching** for bioinformatics reference data

---

## Appendix: Quick Reference

### Profile Selection Decision Tree

```
START
  │
  ├── User specified --profile?
  │     └── YES → Use that profile
  │
  ├── H100 detected?
  │     └── YES → h100_local
  │
  ├── L4 + T4 detected?
  │     └── YES → l4_t4_combined
  │
  ├── T4 detected + DeepSeek API?
  │     └── YES → t4_hybrid
  │
  ├── Any cloud API configured?
  │     └── YES → cloud_only
  │
  └── FAIL → Error: No viable strategy
```

### Task Category Mapping (Revised: 4-5 Models)

**Instead of 10 specialized models, use 4 models that cover 95% of tasks:**

| Task Category | Primary Model | Fallback |
|---------------|---------------|----------|
| intent_parsing | Qwen-2.5-7B-AWQ (Generalist) | DeepSeek-V3 |
| code_generation | Qwen-Coder-7B-AWQ | DeepSeek-V3 |
| code_validation | Qwen-Coder-7B-AWQ | DeepSeek-V3 |
| data_analysis | Qwen-2.5-7B-AWQ (Generalist) | DeepSeek-V3 |
| math_statistics | Qwen-Math-7B-AWQ | DeepSeek-V3 |
| biomedical | Qwen-2.5-7B-AWQ (Generalist) | Claude-3.5 |
| documentation | Qwen-2.5-7B-AWQ (Generalist) | Claude-3.5 |
| embeddings | BGE-M3 | OpenAI |
| safety | Qwen-2.5-7B-AWQ (Generalist) | Claude-3.5 |
| orchestration | DeepSeek-V3 | Claude-3.5 |

### Original Task Mapping (Reference)

| Task | Local Model (T4) | Cloud Fallback |
|------|------------------|----------------|
| intent_parsing | Llama-3.2-3B | DeepSeek-V3 |
| code_generation | Qwen-Coder-7B | DeepSeek-V3 |
| code_validation | Qwen-Coder-1.5B | DeepSeek-V3 |
| data_analysis | Phi-3.5-mini | DeepSeek-V3 |
| math_statistics | Qwen-Math-7B | DeepSeek-V3 |
| biomedical | BioMistral-7B | Claude-3.5 |
| documentation | Gemma-2-9B | Claude-3.5 |
| embeddings | BGE-M3 | OpenAI |
| safety | Llama-Guard-3 | Claude-3.5 |
| orchestration | (cloud only) | DeepSeek-V3 |

---

## Appendix: Peer Review Synthesis

### Key Insights from ChatGPT Review

ChatGPT's most valuable observation:

> "You're on the edge of over-engineering for a 1-2 person team."

This led to significant simplifications:
1. Reduce models from 10 to 4-5 (Generalist, Coder, Math, Embeddings)
2. Extend existing components instead of creating parallel systems
3. Static vLLM servers instead of dynamic SLURM jobs
4. Simple YAML profiles instead of complex StrategyProfile class

### Claude's Additional Considerations

Where I diverge from or extend ChatGPT's recommendations:

1. **Keep Math Model (Qwen-Math-7B)**: ChatGPT suggested it might not be needed, 
   but bioinformatics has heavy statistical workloads (p-values, fold changes, 
   normalization). The math-specialized model is worth one T4 slot.

2. **Keep Embeddings Model (BGE-M3)**: Essential for semantic search in 
   workflow documentation and error pattern matching. Not just a "nice to have."

3. **"vLLM Semantic Router" Warning**: ChatGPT mentioned this as if it exists.
   It doesn't. vLLM is an inference server, not a routing system. Don't go 
   searching for this feature - it was a hallucination.

4. **Health-Check Simplification**: ChatGPT suggested replacing GPU detection
   with simple HTTP health checks. I agree this is sufficient for Phase 1:
   ```python
   def is_model_available(endpoint: str) -> bool:
       try:
           return requests.get(f"{endpoint}/health", timeout=2).ok
       except:
           return False
   ```

5. **LiteLLM Position**: ChatGPT correctly identified LiteLLM as optional.
   Use it only if you need unified API across many cloud providers. For our
   case (primarily DeepSeek + local vLLM), direct OpenAI-compatible calls
   are simpler.

### Agreed Priority Order

Both Claude and ChatGPT agree on this implementation order:

1. **Deploy static vLLM servers** (4 models, long-running)
2. **Minimal ResourceDetector** (health checks, not GPU introspection)
3. **Extend StrategyConfig** (add profile_name, allow_cloud)
4. **Wire T4ModelRouter** into existing provider cascade
5. **Add complexity routing later** (RouteLLM for CODE_GEN/BIO)

### Final Recommendation

**Start simple, measure, then optimize.**

The original 10-model, 4-component architecture was designed for scale we don't 
yet have. The refined approach:
- Gets us running in 1 week instead of 4
- Uses existing code patterns
- Leaves room to add complexity when metrics prove it's needed

The detailed specifications in this document remain valuable as a roadmap for 
future phases, but Phase 1 should be minimal viable: 4 vLLM servers + simple 
profile switching + cloud fallback.

---

*Document Version: 2.0 (Post Peer Review)*
*Last Updated: $(date)*
*Authors: Claude (primary), ChatGPT (peer review)*