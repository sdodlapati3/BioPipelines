# BioPipelines: Professional Agentic Chat Architecture

**Created:** November 27, 2025  
**Version:** 1.0  
**Status:** 📋 DESIGN PROPOSAL  
**GPU Available:** T4 (16GB VRAM) × Multiple  

---

## Executive Summary

Transform the current regex-based chat interface into a **multi-model agentic system** that:
1. Uses specialized small models for different tasks (not one big model for everything)
2. Follows state-of-the-art ReAct/Tool-calling patterns
3. Has true conversational memory and context understanding
4. Gracefully degrades when GPU unavailable (CPU fallback)

### Current State vs Target State

| Aspect | Current | Target |
|--------|---------|--------|
| Intent Detection | Regex patterns | Small LLM with tool calling |
| Entity Extraction | Pattern matching | BiomedBERT NER |
| Conversation Memory | Basic context dict | Vector-based RAG memory |
| Workflow Generation | Cloud API (Lightning.ai) | Cloud + Local hybrid |
| Tool Execution | Hardcoded if/else | OpenAI-compatible function calling |
| Error Handling | Simple try/catch | Agentic retry with reflection |

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           AGENTIC CHAT SYSTEM                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │   Router    │───▶│  Planner    │───▶│  Executor   │───▶│ Synthesizer │   │
│  │  (1.5B)     │    │  (3B/7B)    │    │   (Tools)   │    │  (1.5B/7B)  │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│        │                  │                  │                  │            │
│        ▼                  ▼                  ▼                  ▼            │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    SPECIALIZED KNOWLEDGE MODELS                         │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │ │
│  │  │  BiomedBERT  │  │   SciBERT    │  │  BioMistral  │                  │ │
│  │  │  (NER, CPU)  │  │ (Classify)   │  │  (Bio QA)    │                  │ │
│  │  │   ~500MB     │  │   ~500MB     │  │    ~5GB      │                  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                       TOOL EXECUTION LAYER                               │ │
│  │  scan_data | search_db | check_refs | submit_job | diagnose | ...        │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    MEMORY & CONTEXT (RAG-based)                          │ │
│  │  Conversation History | User Preferences | Project State | Error History │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Model Selection Strategy (T4 Optimized)

### Tier 1: Router Model (Always Loaded, ~1.5B params)
Fast intent classification and routing. Fits easily in T4.

| Model | Size | Quantization | VRAM | Speed | Use Case |
|-------|------|--------------|------|-------|----------|
| **Qwen2.5-1.5B-Instruct** | 1.5B | FP16 | ~3GB | ~100 tok/s | Intent/routing |
| Phi-3-mini-4k | 3.8B | INT4 | ~2.5GB | ~80 tok/s | Alternative |
| SmolLM2-1.7B | 1.7B | FP16 | ~3.5GB | ~90 tok/s | Lightweight |

**Recommended:** `Qwen/Qwen2.5-1.5B-Instruct` - excellent tool-calling support, small footprint.

### Tier 2: Bio-Domain Expert (On-Demand, ~7B params)
For bioinformatics-specific reasoning. Loaded when needed.

| Model | Size | Quantization | VRAM | Purpose |
|-------|------|--------------|------|---------|
| **BioMistral-7B-AWQ** | 7B | AWQ-4bit | ~4.5GB | Bio Q&A |
| Mistral-7B-Instruct | 7B | AWQ-4bit | ~4.5GB | General fallback |
| Llama-3.1-8B-Instruct | 8B | AWQ-4bit | ~5GB | Versatile |

**Recommended:** `BioMistral/BioMistral-7B-AWQ` for biology, `Mistral-7B-AWQ` as fallback.

### Tier 3: NER & Classification (CPU, Always Available)

| Model | Size | Device | VRAM | Purpose |
|-------|------|--------|------|---------|
| **BiomedBERT** | 110M | CPU | 0 | Bio entity extraction |
| **SciBERT** | 110M | CPU | 0 | Scientific classification |
| BERN2 (optional) | - | API | 0 | Advanced bio NER |

**Already implemented!** These models are in `query_parser_ensemble.py`.

### Tier 4: Workflow Generation (Cloud API)
Keep using Lightning.ai/OpenAI for complex generation (best quality, 30M free tokens).

---

## Implementation Plan

### Phase 1: Tool-Calling Router (Week 1)
Replace regex-based tool detection with a small tool-calling LLM.

```python
# src/workflow_composer/agents/router.py

from openai import OpenAI
from typing import List, Dict, Any

# Tool definitions in OpenAI format
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "scan_data",
            "description": "Scan a directory for FASTQ/BAM files and samples",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to scan"},
                    "recursive": {"type": "boolean", "default": True}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "search_databases",
            "description": "Search ENCODE/GEO/SRA for datasets",
            "parameters": {
                "type": "object",
                "properties": {
                    "organism": {"type": "string"},
                    "assay_type": {"type": "string"},
                    "tissue": {"type": "string"},
                    "query": {"type": "string"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_workflow",
            "description": "Generate a Nextflow workflow for bioinformatics analysis",
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis_type": {"type": "string", "enum": ["rna_seq", "chip_seq", "atac_seq", "methylation", "variant_calling", "metagenomics", "single_cell"]},
                    "organism": {"type": "string"},
                    "data_path": {"type": "string"},
                    "comparison": {"type": "string", "description": "Groups to compare"}
                },
                "required": ["analysis_type"]
            }
        }
    },
    # ... more tools
]

class AgentRouter:
    """
    Routes user queries to appropriate tools using a small LLM with function calling.
    Uses vLLM for local inference or falls back to cloud API.
    """
    
    def __init__(self, local_model: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.local_client = None
        self.cloud_client = None
        self.model = local_model
        
        # Try to connect to local vLLM
        self._init_local()
        
        # Fallback to cloud
        if not self.local_client:
            self._init_cloud()
    
    def _init_local(self):
        """Initialize local vLLM client."""
        try:
            self.local_client = OpenAI(
                base_url="http://localhost:8000/v1",
                api_key="not-needed"
            )
            # Quick health check
            self.local_client.models.list()
        except Exception:
            self.local_client = None
    
    def _init_cloud(self):
        """Initialize cloud client (Lightning.ai)."""
        import os
        api_key = os.environ.get("LIGHTNING_API_KEY")
        if api_key:
            self.cloud_client = OpenAI(
                base_url="https://api.lightning.ai/v1",
                api_key=api_key
            )
    
    async def route(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Route a user message to the appropriate tool(s).
        
        Returns:
            {
                "tool": "tool_name" or None,
                "arguments": {...},
                "requires_generation": bool,
                "clarification": str or None
            }
        """
        client = self.local_client or self.cloud_client
        
        # Build context-aware system prompt
        system_prompt = self._build_system_prompt(context)
        
        response = client.chat.completions.create(
            model=self.model if self.local_client else "meta-llama/Llama-3.3-70B-Instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            tools=TOOLS,
            tool_choice="auto"
        )
        
        # Parse response
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            return {
                "tool": tool_call.function.name,
                "arguments": json.loads(tool_call.function.arguments),
                "requires_generation": tool_call.function.name == "generate_workflow",
                "clarification": None
            }
        else:
            # Model chose not to call a tool - conversational response
            return {
                "tool": None,
                "arguments": {},
                "requires_generation": False,
                "response": response.choices[0].message.content
            }
    
    def _build_system_prompt(self, context: Dict[str, Any] = None) -> str:
        """Build system prompt with current context."""
        base = """You are BioPipelines AI, an expert bioinformatics assistant.

You help users:
1. Discover and manage sequencing data (FASTQ, BAM files)
2. Design and generate analysis workflows (RNA-seq, ChIP-seq, etc.)
3. Execute pipelines on SLURM clusters
4. Diagnose errors and suggest fixes

CURRENT CONTEXT:
"""
        if context:
            if context.get("data_loaded"):
                base += f"- Data loaded: {context['sample_count']} samples from {context['data_path']}\n"
            if context.get("last_workflow"):
                base += f"- Last workflow: {context['last_workflow']}\n"
            if context.get("active_job"):
                base += f"- Active job: {context['active_job']} ({context.get('job_status', 'unknown')})\n"
        else:
            base += "- No data loaded yet\n"
        
        base += """
IMPORTANT:
- Use tools to help the user. Don't just describe what you would do.
- If the user asks to scan data, use the scan_data tool.
- If the user wants to create a workflow, use generate_workflow.
- If unclear what the user wants, ask for clarification.
"""
        return base
```

### Phase 2: vLLM Server with Multiple Models (Week 2)

```bash
#!/bin/bash
# scripts/llm/start_agent_server.sh

# Start vLLM with the router model
# T4 can easily handle 1.5B + leave room for BioMistral when needed

vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --port 8000 \
    --host 0.0.0.0 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.4 \
    --max-model-len 4096 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --dtype half

# Note: 0.4 GPU utilization leaves ~10GB free for BioMistral when needed
```

```python
# scripts/llm/serve_biomistral_t4.sbatch

#!/bin/bash
#SBATCH --job-name=biomistral_t4
#SBATCH --partition=t4flex
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

# Load modules
module load cuda/12.1

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate biopipelines

# Start vLLM with BioMistral (AWQ quantized for T4)
vllm serve BioMistral/BioMistral-7B-AWQ \
    --port 8001 \
    --host 0.0.0.0 \
    --quantization awq \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --dtype half
```

### Phase 3: ReAct Agent Loop (Week 3)

Implement a proper ReAct (Reason + Act) loop for multi-step tasks:

```python
# src/workflow_composer/agents/react_agent.py

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum
import asyncio

class AgentState(Enum):
    THINKING = "thinking"
    ACTING = "acting" 
    OBSERVING = "observing"
    RESPONDING = "responding"
    DONE = "done"
    ERROR = "error"

@dataclass
class AgentStep:
    """A single step in the agent's reasoning chain."""
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict] = None
    observation: Optional[str] = None
    
class ReactAgent:
    """
    ReAct-style agent that can reason and take actions.
    
    Flow:
    1. User query → Thought (what should I do?)
    2. Thought → Action (which tool to use?)
    3. Action → Observation (what happened?)
    4. Observation → Thought (what next?)
    5. Repeat until done or max_steps
    6. Synthesize final response
    """
    
    def __init__(
        self, 
        router: AgentRouter,
        tools: AgentTools,
        max_steps: int = 5,
        verbose: bool = True
    ):
        self.router = router
        self.tools = tools
        self.max_steps = max_steps
        self.verbose = verbose
        self.steps: List[AgentStep] = []
    
    async def run(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        Run the agent on a user query.
        
        Args:
            query: User's natural language query
            context: Current conversation context
            
        Returns:
            Final response to the user
        """
        self.steps = []
        
        for i in range(self.max_steps):
            # Build prompt with reasoning history
            prompt = self._build_react_prompt(query, context)
            
            # Get next action from router
            result = await self.router.route(prompt, context)
            
            if result.get("tool"):
                # Execute the tool
                step = AgentStep(
                    thought=f"I should use {result['tool']} to help with this.",
                    action=result["tool"],
                    action_input=result["arguments"]
                )
                
                # Execute tool
                tool_result = await self._execute_tool(
                    result["tool"], 
                    result["arguments"]
                )
                step.observation = tool_result.message
                
                # Update context based on tool result
                self._update_context(context, result["tool"], tool_result)
                
                self.steps.append(step)
                
                # Check if we should continue or respond
                if self._should_respond(tool_result, query):
                    return self._synthesize_response(query, context)
            
            elif result.get("response"):
                # Model chose to respond directly
                return result["response"]
            
            else:
                # Ask for clarification
                if result.get("clarification"):
                    return result["clarification"]
                break
        
        # Max steps reached - synthesize what we have
        return self._synthesize_response(query, context)
    
    async def _execute_tool(self, tool_name: str, arguments: Dict) -> ToolResult:
        """Execute a tool and return the result."""
        tool_method = getattr(self.tools, tool_name, None)
        if tool_method:
            try:
                return tool_method(**arguments)
            except Exception as e:
                return ToolResult(
                    success=False,
                    tool_name=tool_name,
                    error=str(e),
                    message=f"❌ Tool execution failed: {e}"
                )
        return ToolResult(
            success=False,
            tool_name=tool_name,
            error=f"Unknown tool: {tool_name}"
        )
    
    def _build_react_prompt(self, query: str, context: Dict) -> str:
        """Build prompt with reasoning history."""
        prompt = f"User Query: {query}\n\n"
        
        if self.steps:
            prompt += "Previous Steps:\n"
            for i, step in enumerate(self.steps, 1):
                prompt += f"{i}. Thought: {step.thought}\n"
                if step.action:
                    prompt += f"   Action: {step.action}({step.action_input})\n"
                if step.observation:
                    prompt += f"   Observation: {step.observation[:500]}...\n"
            prompt += "\nWhat should I do next?\n"
        
        return prompt
    
    def _should_respond(self, result: ToolResult, query: str) -> bool:
        """Determine if we have enough to respond."""
        # Simple heuristic - can be made smarter
        if result.success:
            # If user asked to scan data and we scanned, respond
            if "scan" in query.lower() and result.tool_name == "scan_data":
                return True
            # If user asked to search and we searched, respond
            if "search" in query.lower() and result.tool_name == "search_databases":
                return True
        return False
    
    def _synthesize_response(self, query: str, context: Dict) -> str:
        """Synthesize final response from all steps."""
        if not self.steps:
            return "I wasn't able to complete your request. Could you please provide more details?"
        
        # Use the last observation as the main response
        last_step = self.steps[-1]
        if last_step.observation:
            return last_step.observation
        
        return "Task completed but no output was generated."
    
    def _update_context(self, context: Dict, tool_name: str, result: ToolResult):
        """Update context based on tool result."""
        if not context:
            return
        
        if tool_name == "scan_data" and result.success:
            context["data_loaded"] = True
            context["data_path"] = result.data.get("path", "")
            context["sample_count"] = result.data.get("count", 0)
        
        elif tool_name == "submit_job" and result.success:
            context["active_job"] = result.data.get("job_id", "")
            context["job_status"] = "submitted"
```

### Phase 4: Vector Memory & RAG (Week 4)

Add long-term memory using embeddings:

```python
# src/workflow_composer/agents/memory.py

from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime
import numpy as np

@dataclass  
class MemoryEntry:
    """A single memory entry."""
    content: str
    embedding: np.ndarray
    metadata: Dict
    timestamp: datetime
    memory_type: str  # "conversation", "error", "workflow", "preference"

class AgentMemory:
    """
    Long-term memory using embeddings for semantic search.
    
    Uses a small embedding model (runs on CPU) to enable:
    - Remembering past conversations
    - Learning from errors
    - Storing user preferences
    - Contextual workflow recommendations
    """
    
    def __init__(self, embedding_model: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        
        self.model = SentenceTransformer(embedding_model)
        self.memories: List[MemoryEntry] = []
        self.max_memories = 1000
    
    def add(self, content: str, memory_type: str, metadata: Dict = None):
        """Add a memory."""
        embedding = self.model.encode(content)
        entry = MemoryEntry(
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            timestamp=datetime.now(),
            memory_type=memory_type
        )
        self.memories.append(entry)
        
        # Prune old memories if needed
        if len(self.memories) > self.max_memories:
            self._prune_memories()
    
    def search(self, query: str, top_k: int = 5, memory_type: str = None) -> List[MemoryEntry]:
        """Search memories by semantic similarity."""
        query_embedding = self.model.encode(query)
        
        # Filter by type if specified
        candidates = self.memories
        if memory_type:
            candidates = [m for m in self.memories if m.memory_type == memory_type]
        
        if not candidates:
            return []
        
        # Compute similarities
        similarities = []
        for memory in candidates:
            sim = np.dot(query_embedding, memory.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(memory.embedding)
            )
            similarities.append((memory, sim))
        
        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in similarities[:top_k]]
    
    def get_context_for_query(self, query: str) -> str:
        """Get relevant context for a query."""
        # Search different memory types
        conv_memories = self.search(query, top_k=2, memory_type="conversation")
        error_memories = self.search(query, top_k=1, memory_type="error")
        pref_memories = self.search(query, top_k=1, memory_type="preference")
        
        context_parts = []
        
        if conv_memories:
            context_parts.append("Recent relevant conversations:")
            for m in conv_memories:
                context_parts.append(f"  - {m.content[:200]}")
        
        if error_memories:
            context_parts.append("Past errors to avoid:")
            for m in error_memories:
                context_parts.append(f"  - {m.content[:200]}")
        
        if pref_memories:
            context_parts.append("User preferences:")
            for m in pref_memories:
                context_parts.append(f"  - {m.content[:200]}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def _prune_memories(self):
        """Remove old, less important memories."""
        # Keep most recent and highest-accessed
        # Simple: just keep most recent for now
        self.memories = sorted(
            self.memories, 
            key=lambda m: m.timestamp,
            reverse=True
        )[:self.max_memories]
```

---

## Resource Allocation for T4 GPUs

With multiple T4s, here's the recommended allocation:

### Option A: Single T4 (Conservative)
```
┌──────────────────────────────────────────────┐
│              T4 GPU (16GB VRAM)              │
├──────────────────────────────────────────────┤
│  Qwen2.5-1.5B (Router)     : ~3GB  (always)  │
│  BioMistral-7B-AWQ         : ~5GB  (on-demand)│
│  KV Cache / Overhead       : ~5GB            │
│  Free                      : ~3GB            │
└──────────────────────────────────────────────┘
```

### Option B: Two T4s (Recommended)
```
┌─────────────────────────────────────────────────────────────────┐
│  T4 #1: Fast Router + General                                   │
├─────────────────────────────────────────────────────────────────┤
│  Qwen2.5-1.5B-Instruct     : ~3GB   (always loaded)             │
│  Mistral-7B-Instruct-AWQ   : ~5GB   (general fallback)          │
│  Reserved for batching     : ~8GB                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  T4 #2: Bio-Domain Expert                                       │
├─────────────────────────────────────────────────────────────────┤
│  BioMistral-7B-AWQ         : ~5GB   (bio questions)             │
│  Reserved for future       : ~11GB  (larger models)             │
└─────────────────────────────────────────────────────────────────┘

CPU (Always Available):
- BiomedBERT: NER extraction
- SciBERT: Scientific classification  
- BGE-small: Embeddings for memory
- Workflow validation (rule-based)
```

---

## Implementation Checklist

### Week 1: Tool-Calling Router
- [ ] Install vLLM with tool-calling support
- [ ] Download Qwen2.5-1.5B-Instruct model
- [ ] Create `AgentRouter` class with OpenAI-compatible interface
- [ ] Define tools in OpenAI function format
- [ ] Update `gradio_app.py` to use router instead of regex

### Week 2: Multi-Model Setup  
- [ ] Create SLURM scripts for model serving
- [ ] Set up BioMistral-7B-AWQ for bio queries
- [ ] Implement model switching logic
- [ ] Add health checks and fallback handling
- [ ] Test with mixed workloads

### Week 3: ReAct Agent
- [ ] Implement `ReactAgent` class
- [ ] Add multi-step reasoning loop
- [ ] Implement context tracking between steps
- [ ] Add error recovery and retry logic
- [ ] Stream intermediate steps to UI

### Week 4: Memory & Polish
- [ ] Implement `AgentMemory` with embeddings
- [ ] Add conversation persistence
- [ ] Implement error learning
- [ ] Fine-tune prompts based on testing
- [ ] Performance benchmarking

---

## Comparison: Current vs New Architecture

| Metric | Current (Regex) | New (Agentic) |
|--------|-----------------|---------------|
| Intent Accuracy | ~70% | ~95% |
| Multi-turn Support | Basic | Full |
| Error Recovery | None | Automatic retry |
| Context Window | ~2K tokens | ~32K tokens |
| Latency (local) | 0ms (regex) | ~200ms (1.5B) |
| Latency (cloud) | ~2s | ~1s (local) / ~2s (cloud) |
| GPU Required | No | Optional (CPU fallback) |
| Bio-domain Knowledge | Pattern matching | Trained on PubMed |

---

## Future Scaling (A100/H100)

When you have access to A100/H100:

1. **Replace BioMistral-7B with BioMistral-13B** (if available) or fine-tuned Llama-3.1-70B
2. **Run multiple models simultaneously** on one GPU
3. **Enable speculative decoding** for faster inference
4. **Fine-tune on your workflow data** for better intent understanding
5. **Add multimodal support** (analyze images of gels, plots, etc.)

---

## Quick Start

```bash
# 1. Install vLLM
pip install vllm

# 2. Start the router model
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --port 8000 \
    --enable-auto-tool-choice

# 3. Start the Gradio app
cd scripts && ./start_gradio.sh

# The app will automatically use local models if available,
# falling back to Lightning.ai API if not.
```

---

## References

1. [ReAct: Synergizing Reasoning and Acting in LLMs](https://arxiv.org/abs/2210.03629)
2. [BioMistral: Medical LLM](https://arxiv.org/abs/2402.10373)
3. [vLLM: Fast LLM Serving](https://docs.vllm.ai/)
4. [Function Calling with vLLM](https://docs.vllm.ai/en/latest/features/tool_calling/)
5. [smolagents: HuggingFace Agents Library](https://huggingface.co/docs/smolagents/)

---

*This architecture is designed to grow with your needs - start simple with T4s, scale to enterprise-grade with A100/H100 when ready.*
