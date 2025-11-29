# BioPipelines: H100 Professional Multi-Agent Architecture

**Created:** November 27, 2025  
**Version:** 2.0  
**Status:** 📋 DESIGN & IMPLEMENTATION PLAN  
**GPU Available:** H100 80GB × 2-4  

---

## Executive Summary

Build a **professional agentic AI chat system** using:
- **Multi-model orchestration** on H100 GPUs
- **Specialized coding agents** for error diagnosis and fixes
- **Data discovery agents** for intelligent data scanning
- **Workflow generation agents** with domain expertise
- **Self-healing pipelines** that diagnose and fix their own errors

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                        H100 MULTI-AGENT ORCHESTRATION SYSTEM                         │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                       │
│    ┌────────────────────────────────────────────────────────────────────────────┐    │
│    │                           SUPERVISOR AGENT                                  │    │
│    │               Llama-3.3-70B-Instruct (GPU 0-1, vLLM)                       │    │
│    │                                                                             │    │
│    │   • Routes queries to specialized agents                                    │    │
│    │   • Maintains conversation context                                          │    │
│    │   • Orchestrates multi-step workflows                                       │    │
│    │   • Synthesizes final responses                                             │    │
│    └───────┬────────────────────────┬────────────────────────┬───────────────────┘    │
│            │                        │                        │                        │
│            ▼                        ▼                        ▼                        │
│    ┌──────────────┐         ┌──────────────┐         ┌──────────────┐                │
│    │    DATA      │         │  WORKFLOW    │         │   CODING     │                │
│    │   AGENT      │         │   AGENT      │         │   AGENT      │                │
│    │              │         │              │         │              │                │
│    │ • Scan dirs  │         │ • Generate   │         │ • Diagnose   │                │
│    │ • Search DBs │         │   pipelines  │         │   errors     │                │
│    │ • Check refs │         │ • Configure  │         │ • Fix code   │                │
│    │ • Download   │         │ • Validate   │         │ • Debug logs │                │
│    └──────────────┘         └──────────────┘         └──────────────┘                │
│            │                        │                        │                        │
│            │                        │                        ▼                        │
│            │                        │                ┌──────────────┐                │
│            │                        │                │   CODING     │                │
│            │                        │                │   MODEL      │                │
│            │                        │                │              │                │
│            │                        │                │ Qwen2.5-     │                │
│            │                        │                │ Coder-32B    │                │
│            │                        │                │ (GPU 2-3)    │                │
│            │                        │                └──────────────┘                │
│            ▼                        ▼                        ▼                        │
│    ┌─────────────────────────────────────────────────────────────────────────────┐   │
│    │                           TOOL EXECUTION LAYER                               │   │
│    │                                                                              │   │
│    │  scan_data | search_db | check_refs | submit_job | diagnose | fix_code      │   │
│    │  generate_workflow | run_nextflow | parse_logs | validate_config             │   │
│    └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                       │
│    ┌─────────────────────────────────────────────────────────────────────────────┐   │
│    │                         MEMORY & CONTEXT                                     │   │
│    │                                                                              │   │
│    │  Conversation History | Error History | User Preferences | Project State    │   │
│    │  (SQLite + Vector Embeddings with BGE-small)                                 │   │
│    └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                       │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Model Deployment Strategy

### Configuration A: 2× H100 80GB (Default)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                             2× H100 80GB DEPLOYMENT                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  GPU 0 + GPU 1: Llama-3.3-70B-Instruct (Tensor Parallel = 2)               │   │
│  │                                                                              │   │
│  │  VRAM Usage: ~70GB × 2 GPUs = 140GB model in FP16                           │   │
│  │  Port: 8000                                                                  │   │
│  │  Role: Supervisor + Workflow Generation + Bio-domain reasoning              │   │
│  │                                                                              │   │
│  │  Capabilities:                                                               │   │
│  │  • Tool calling (OpenAI function format)                                    │   │
│  │  • 128K context window                                                       │   │
│  │  • Strong reasoning and planning                                             │   │
│  │  • Bio/medical knowledge from training                                       │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
│  CPU Services (Always Available):                                                    │
│  • BiomedBERT - Entity extraction                                                    │
│  • BGE-small - Embeddings for memory                                                 │
│  • FastAPI server for orchestration                                                  │
│                                                                                      │
│  Cloud Fallback (When Local Unavailable):                                            │
│  • Lightning.ai - 30M free tokens/month                                              │
│  • DeepSeek-V3 via Lightning                                                         │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**For 2 GPUs - Single Model Strategy:**
- Run Llama-3.3-70B with tensor parallelism across both GPUs
- Use prompt-based specialization (coding prompts, bio prompts)
- Cloud fallback for specialized coding tasks

### Configuration B: 4× H100 80GB (Advanced)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                             4× H100 80GB DEPLOYMENT                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌───────────────────────────────────────────┐  ┌───────────────────────────────┐   │
│  │  GPU 0 + GPU 1: Llama-3.3-70B-Instruct   │  │  GPU 2 + GPU 3:               │   │
│  │                                           │  │  Qwen2.5-Coder-32B × 2       │   │
│  │  Tensor Parallel = 2                      │  │  OR DeepSeek-Coder-V2        │   │
│  │  Port: 8000                               │  │                               │   │
│  │                                           │  │  Tensor Parallel = 2          │   │
│  │  Role:                                    │  │  Port: 8001                   │   │
│  │  • Supervisor agent                       │  │                               │   │
│  │  • Query routing                          │  │  Role:                        │   │
│  │  • Workflow planning                      │  │  • Code generation            │   │
│  │  • Bio-domain Q&A                         │  │  • Error diagnosis            │   │
│  │  • Response synthesis                     │  │  • Log parsing                │   │
│  │                                           │  │  • Code fixes                 │   │
│  └───────────────────────────────────────────┘  └───────────────────────────────┘   │
│                                                                                      │
│  Alternative 4-GPU Config (Best Coding):                                             │
│  • GPU 0-3: DeepSeek-Coder-V2-Instruct (236B MoE, TP=4)                             │
│    - Best-in-class coding performance                                               │
│    - Handles orchestration + coding in one model                                    │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Coding Agent Design

### Purpose
The coding agent specializes in:
1. **Error Diagnosis** - Parse logs, identify root causes
2. **Code Fixes** - Generate patches for broken workflows
3. **Workflow Debugging** - Fix Nextflow/Snakemake configs
4. **Self-Healing** - Automatic retry with fixes

### Architecture

```python
# src/workflow_composer/agents/coding_agent.py

class CodingAgent:
    """
    Specialized agent for code-related tasks.
    
    Uses a coding-optimized model (Qwen2.5-Coder or DeepSeek-Coder)
    for tasks requiring code understanding and generation.
    """
    
    # Capabilities
    TASKS = [
        "diagnose_error",     # Parse error logs, identify cause
        "fix_code",           # Generate code patches
        "explain_error",      # Human-friendly error explanation
        "optimize_workflow",  # Suggest performance improvements
        "validate_config",    # Check Nextflow/Snakemake configs
        "generate_test",      # Create test cases for workflows
    ]
    
    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        vllm_url: str = "http://localhost:8001/v1",  # Separate from supervisor
    ):
        self.model = model
        self.client = OpenAI(base_url=vllm_url, api_key="not-needed")
        
    async def diagnose_error(
        self, 
        error_log: str, 
        workflow_config: str = None,
        context: dict = None
    ) -> DiagnosisResult:
        """
        Analyze an error and provide diagnosis + fix.
        
        Returns:
            DiagnosisResult with:
            - root_cause: str
            - explanation: str (human-friendly)
            - suggested_fix: str (code or command)
            - confidence: float
            - auto_fixable: bool
        """
        prompt = self._build_diagnosis_prompt(error_log, workflow_config)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CODING_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low for precise code
            max_tokens=2048
        )
        
        return self._parse_diagnosis(response.choices[0].message.content)
    
    async def generate_fix(
        self,
        diagnosis: DiagnosisResult,
        file_content: str,
        file_type: str  # "nextflow", "snakemake", "python", etc.
    ) -> CodeFix:
        """
        Generate a code fix based on diagnosis.
        
        Returns:
            CodeFix with:
            - original_code: str
            - fixed_code: str
            - diff: str
            - explanation: str
        """
        prompt = f"""
Based on this error diagnosis:
{diagnosis.explanation}

Fix the following {file_type} code:
```{file_type}
{file_content}
```

Provide the corrected code with explanatory comments.
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CODING_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=4096
        )
        
        return self._parse_code_fix(response.choices[0].message.content, file_content)
```

### Coding Model Options

| Model | HuggingFace ID | Size | GPUs | Best For |
|-------|----------------|------|------|----------|
| **Qwen2.5-Coder-32B** | `Qwen/Qwen2.5-Coder-32B-Instruct` | 32B | 1 | General coding, fits single H100 |
| **DeepSeek-Coder-V2** | `deepseek-ai/DeepSeek-Coder-V2-Instruct` | 236B MoE | 4 | Best performance, needs 4 GPUs |
| **CodeLlama-70B** | `codellama/CodeLlama-70b-Instruct-hf` | 70B | 2 | Alternative to Qwen |
| **Starcoder2-15B** | `bigcode/starcoder2-15b` | 15B | 1 | Lightweight alternative |

---

## Agent Orchestrator Design

```python
# src/workflow_composer/agents/orchestrator.py

from enum import Enum
from typing import Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass

class AgentType(Enum):
    SUPERVISOR = "supervisor"     # Routes and plans
    DATA = "data"                 # Data discovery
    WORKFLOW = "workflow"         # Pipeline generation  
    CODING = "coding"             # Error diagnosis/fixes
    MEMORY = "memory"             # Context management

@dataclass
class AgentMessage:
    """Message passed between agents."""
    from_agent: AgentType
    to_agent: AgentType
    content: str
    metadata: Dict[str, Any]
    requires_response: bool = True

class AgentOrchestrator:
    """
    Coordinates multiple specialized agents for complex tasks.
    
    Flow:
    1. User query → Supervisor
    2. Supervisor plans and routes to specialized agents
    3. Specialized agents execute and return results
    4. Supervisor synthesizes final response
    
    Supports:
    - Parallel agent execution
    - Multi-step workflows
    - Error recovery with coding agent
    - Streaming responses
    """
    
    def __init__(
        self,
        supervisor_url: str = "http://localhost:8000/v1",
        coding_url: str = "http://localhost:8001/v1",
        supervisor_model: str = "meta-llama/Llama-3.3-70B-Instruct",
        coding_model: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
    ):
        # Agent instances
        self.supervisor = SupervisorAgent(supervisor_url, supervisor_model)
        self.data_agent = DataAgent(supervisor_url, supervisor_model)
        self.workflow_agent = WorkflowAgent(supervisor_url, supervisor_model)
        self.coding_agent = CodingAgent(coding_url, coding_model)
        self.memory = AgentMemory()
        
        # Routing table
        self.agents = {
            AgentType.DATA: self.data_agent,
            AgentType.WORKFLOW: self.workflow_agent,
            AgentType.CODING: self.coding_agent,
        }
    
    async def process(
        self, 
        query: str, 
        context: Dict[str, Any] = None,
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Process a user query through the agent system.
        
        Yields intermediate updates and final response.
        """
        # Add relevant memories to context
        if context is None:
            context = {}
        context["memories"] = self.memory.get_context_for_query(query)
        
        # Step 1: Supervisor plans the approach
        yield "🤔 Analyzing your request...\n"
        plan = await self.supervisor.plan(query, context)
        
        if plan.requires_agents:
            yield f"📋 Plan: {plan.summary}\n\n"
            
            # Step 2: Execute agent tasks
            results = {}
            for task in plan.tasks:
                agent = self.agents.get(task.agent_type)
                if agent:
                    yield f"⚙️ {task.agent_type.value}: {task.description}...\n"
                    result = await agent.execute(task.params, context)
                    results[task.agent_type] = result
                    
                    # Check for errors that need coding agent
                    if result.has_error and task.agent_type != AgentType.CODING:
                        yield "🔧 Detected error, invoking coding agent...\n"
                        fix = await self.coding_agent.diagnose_error(
                            result.error_log,
                            context.get("workflow_config")
                        )
                        results[AgentType.CODING] = fix
                        yield f"💡 Diagnosis: {fix.explanation}\n"
            
            # Step 3: Synthesize response
            yield "\n📝 Preparing response...\n\n"
            response = await self.supervisor.synthesize(query, results, context)
            
        else:
            # Direct response from supervisor
            response = plan.direct_response
        
        # Save to memory
        self.memory.add(f"Q: {query}\nA: {response[:500]}", "conversation")
        
        yield response
    
    async def diagnose_and_fix(
        self,
        job_id: str,
        auto_fix: bool = False
    ) -> DiagnosisResult:
        """
        Diagnose a failed job and optionally apply fixes.
        
        This is a key feature for self-healing pipelines:
        1. Get job logs
        2. Send to coding agent for diagnosis
        3. Generate fix if possible
        4. Optionally apply and retry
        """
        # Get logs
        logs = await self.data_agent.get_job_logs(job_id)
        config = await self.data_agent.get_workflow_config(job_id)
        
        # Diagnose
        diagnosis = await self.coding_agent.diagnose_error(logs, config)
        
        if diagnosis.auto_fixable and auto_fix:
            # Generate and apply fix
            fix = await self.coding_agent.generate_fix(
                diagnosis,
                config,
                file_type="nextflow"
            )
            
            # Apply fix
            await self._apply_fix(job_id, fix)
            
            # Retry job
            new_job_id = await self.data_agent.retry_job(job_id)
            diagnosis.retry_job_id = new_job_id
        
        return diagnosis
```

---

## Multi-Model Server Script

```bash
#!/bin/bash
# scripts/llm/start_multi_model.sh
# 
# Start multiple vLLM servers for multi-agent system
# Usage: ./start_multi_model.sh [2gpu|4gpu]

set -e

MODE="${1:-2gpu}"

echo "🚀 Starting BioPipelines Multi-Agent System"
echo "   Mode: $MODE"
echo ""

# Common settings
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true

case "$MODE" in
    "2gpu")
        echo "📦 Configuration: 2× H100 80GB"
        echo "   - Llama-3.3-70B (GPU 0-1, port 8000)"
        echo ""
        
        # Single model spanning both GPUs
        CUDA_VISIBLE_DEVICES="0,1" python -m vllm.entrypoints.openai.api_server \
            --model meta-llama/Llama-3.3-70B-Instruct \
            --port 8000 \
            --host 0.0.0.0 \
            --tensor-parallel-size 2 \
            --max-model-len 32768 \
            --gpu-memory-utilization 0.90 \
            --enable-auto-tool-choice \
            --tool-call-parser hermes \
            --dtype float16
        ;;
        
    "4gpu")
        echo "📦 Configuration: 4× H100 80GB"
        echo "   - Llama-3.3-70B (GPU 0-1, port 8000) - Supervisor"
        echo "   - Qwen2.5-Coder-32B (GPU 2-3, port 8001) - Coding Agent"
        echo ""
        
        # Start supervisor model (background)
        CUDA_VISIBLE_DEVICES="0,1" python -m vllm.entrypoints.openai.api_server \
            --model meta-llama/Llama-3.3-70B-Instruct \
            --port 8000 \
            --host 0.0.0.0 \
            --tensor-parallel-size 2 \
            --max-model-len 32768 \
            --gpu-memory-utilization 0.90 \
            --enable-auto-tool-choice \
            --tool-call-parser hermes \
            --dtype float16 &
        
        SUPERVISOR_PID=$!
        echo "   Supervisor PID: $SUPERVISOR_PID"
        
        # Wait for supervisor to be ready
        sleep 30
        
        # Start coding model
        CUDA_VISIBLE_DEVICES="2,3" python -m vllm.entrypoints.openai.api_server \
            --model Qwen/Qwen2.5-Coder-32B-Instruct \
            --port 8001 \
            --host 0.0.0.0 \
            --tensor-parallel-size 2 \
            --max-model-len 65536 \
            --gpu-memory-utilization 0.90 \
            --dtype float16 &
        
        CODER_PID=$!
        echo "   Coder PID: $CODER_PID"
        
        # Wait for both
        wait
        ;;
        
    "4gpu-deepseek")
        echo "📦 Configuration: 4× H100 80GB (DeepSeek-Coder-V2)"
        echo "   - DeepSeek-Coder-V2-Instruct (GPU 0-3, port 8000)"
        echo "   - Best coding performance, handles all tasks"
        echo ""
        
        # DeepSeek-Coder-V2 is a 236B MoE model - needs all 4 GPUs
        python -m vllm.entrypoints.openai.api_server \
            --model deepseek-ai/DeepSeek-Coder-V2-Instruct \
            --port 8000 \
            --host 0.0.0.0 \
            --tensor-parallel-size 4 \
            --max-model-len 65536 \
            --gpu-memory-utilization 0.95 \
            --trust-remote-code \
            --enable-auto-tool-choice \
            --tool-call-parser hermes \
            --dtype bfloat16
        ;;
        
    *)
        echo "Usage: $0 [2gpu|4gpu|4gpu-deepseek]"
        exit 1
        ;;
esac
```

---

## Tool Definitions for Coding Agent

```python
# Additional tools for coding agent

CODING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "diagnose_error",
            "description": "Analyze error logs from a failed pipeline job and identify root cause",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "SLURM job ID or pipeline run ID"
                    },
                    "error_type": {
                        "type": "string",
                        "enum": ["slurm", "nextflow", "tool", "memory", "unknown"],
                        "description": "Type of error if known"
                    }
                },
                "required": ["job_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fix_workflow",
            "description": "Generate a fix for a broken Nextflow or Snakemake workflow",
            "parameters": {
                "type": "object",
                "properties": {
                    "workflow_path": {
                        "type": "string",
                        "description": "Path to the workflow file"
                    },
                    "error_description": {
                        "type": "string",
                        "description": "Description of the error"
                    },
                    "auto_apply": {
                        "type": "boolean",
                        "description": "Whether to automatically apply the fix",
                        "default": False
                    }
                },
                "required": ["workflow_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "explain_code",
            "description": "Explain what a piece of bioinformatics code does",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Code snippet to explain"
                    },
                    "language": {
                        "type": "string",
                        "enum": ["nextflow", "snakemake", "python", "bash", "r"],
                        "description": "Programming language"
                    },
                    "detail_level": {
                        "type": "string",
                        "enum": ["brief", "detailed", "beginner"],
                        "default": "detailed"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "validate_config",
            "description": "Validate a workflow configuration file",
            "parameters": {
                "type": "object",
                "properties": {
                    "config_path": {
                        "type": "string",
                        "description": "Path to config file"
                    },
                    "config_type": {
                        "type": "string",
                        "enum": ["nextflow.config", "snakemake.yaml", "params.json"],
                        "description": "Type of configuration"
                    }
                },
                "required": ["config_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_optimization",
            "description": "Suggest performance optimizations for a workflow",
            "parameters": {
                "type": "object",
                "properties": {
                    "workflow_path": {
                        "type": "string",
                        "description": "Path to workflow"
                    },
                    "focus": {
                        "type": "string",
                        "enum": ["speed", "memory", "cost", "all"],
                        "default": "all"
                    }
                },
                "required": ["workflow_path"]
            }
        }
    }
]
```

---

## Integration with Gradio Chat

```python
# Update to src/workflow_composer/web/gradio_app.py

async def chat_with_multi_agent(
    message: str,
    history: List[Dict[str, str]],
    app_state: AppState,
) -> AsyncGenerator[Tuple[List[Dict[str, str]], str], None]:
    """
    Chat handler using multi-agent orchestration.
    
    Flow:
    1. Check for local vLLM servers
    2. Route to orchestrator
    3. Stream responses with progress updates
    """
    
    # Initialize orchestrator (lazy loading)
    if not hasattr(app_state, 'orchestrator'):
        from workflow_composer.agents.orchestrator import AgentOrchestrator
        
        # Detect configuration
        supervisor_url = os.environ.get("VLLM_URL", "http://localhost:8000/v1")
        coding_url = os.environ.get("VLLM_CODER_URL", supervisor_url)  # Same if single model
        
        app_state.orchestrator = AgentOrchestrator(
            supervisor_url=supervisor_url,
            coding_url=coding_url,
        )
    
    # Build context
    context = {
        "data_loaded": app_state.data_path is not None,
        "sample_count": len(app_state.samples) if app_state.samples else 0,
        "data_path": app_state.data_path,
        "last_workflow": app_state.generated_workflow,
        "active_job": app_state.active_job_id,
    }
    
    # Stream response
    history.append({"role": "user", "content": message})
    response_text = ""
    
    async for chunk in app_state.orchestrator.process(message, context, stream=True):
        response_text += chunk
        history_copy = history + [{"role": "assistant", "content": response_text}]
        yield history_copy, ""
    
    # Final history update
    history.append({"role": "assistant", "content": response_text})
    yield history, ""
```

---

## Implementation Checklist

### Phase 1: Core Infrastructure (Week 1)
- [ ] Update `start_gradio.sh` to support multi-model configuration
- [ ] Create `start_multi_model.sh` script
- [ ] Add Qwen2.5-Coder-32B to model registry
- [ ] Test 2-GPU Llama-3.3-70B deployment

### Phase 2: Coding Agent (Week 2)
- [ ] Create `src/workflow_composer/agents/coding_agent.py`
- [ ] Implement `diagnose_error()` method
- [ ] Implement `generate_fix()` method
- [ ] Add coding tools to AGENT_TOOLS list
- [ ] Write unit tests

### Phase 3: Orchestrator (Week 3)
- [ ] Create `src/workflow_composer/agents/orchestrator.py`
- [ ] Implement `SupervisorAgent` for routing
- [ ] Implement multi-agent message passing
- [ ] Add streaming support
- [ ] Test full agent loop

### Phase 4: Self-Healing (Week 4)
- [ ] Implement `diagnose_and_fix()` for failed jobs
- [ ] Add auto-retry logic
- [ ] Create error pattern database
- [ ] Add learning from past fixes
- [ ] Performance benchmarking

---

## Model Comparison

| Feature | Llama-3.3-70B | Qwen2.5-Coder-32B | DeepSeek-Coder-V2 |
|---------|---------------|-------------------|-------------------|
| **Size** | 70B | 32B | 236B MoE |
| **GPUs (H100)** | 2 | 1 | 4 |
| **Context** | 128K | 128K | 128K |
| **Tool Calling** | Excellent | Good | Excellent |
| **Code Quality** | Good | Excellent | Best |
| **Reasoning** | Excellent | Good | Excellent |
| **Bio Knowledge** | Good | Limited | Good |
| **Best For** | Orchestration | Code fixes | Everything |

---

## Quick Start

```bash
# Option 1: 2 GPU setup (recommended start)
sbatch scripts/start_gradio.sh --gpu 2

# Option 2: 4 GPU with separate coding model
sbatch scripts/start_gradio.sh --gpu 4 --multi-model

# Option 3: Manual start for testing
./scripts/llm/start_multi_model.sh 2gpu
```

---

## Summary

This architecture provides:

1. **Professional Multi-Agent System** - Specialized agents for different tasks
2. **Self-Healing Pipelines** - Coding agent diagnoses and fixes errors
3. **Flexible Deployment** - Works with 2 or 4 H100 GPUs
4. **Cloud Fallback** - Lightning.ai when GPUs unavailable
5. **Streaming Responses** - Real-time feedback in chat
6. **Memory & Context** - Learns from past interactions

The key insight is that with H100 GPUs, you can run **state-of-the-art models locally** that match or exceed cloud API quality, while having full control over data privacy and cost.
