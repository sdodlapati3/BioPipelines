# Unified Agentic System Architecture

## Overview

The BioPipelines project now has a **Unified Agentic System** that combines:
1. **AgentTools** (29 tools across 6 categories)
2. **AutonomousAgent** (orchestration, permissions, multi-step reasoning)
3. **Executor Layer** (safe command execution, audit logging)

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                              │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     UnifiedAgent.process_query()                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  1. classify_task()                      │   │
│  │    Determines: WORKFLOW, DATA, JOB, DIAGNOSIS, etc.     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                │                                │
│                                ▼                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           2. detect_tool() + check_permission()          │   │
│  │    Pattern matching → Permission check by level          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                │                                │
│              ┌─────────────────┴─────────────────┐              │
│              ▼                                   ▼              │
│   ┌──────────────────┐              ┌──────────────────────┐   │
│   │  ALLOWED         │              │  NEEDS_APPROVAL      │   │
│   │  execute_tool()  │              │  (ASSISTED level)    │   │
│   └────────┬─────────┘              └──────────────────────┘   │
│            │                                                    │
│            ▼                                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              3. AgentTools.execute_tool()                │   │
│  │    Dispatches to appropriate tool implementation         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                │                                │
│                                ▼                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              4. AuditLogger.log_tool_call()              │   │
│  │    Complete audit trail of all actions                   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       AgentResponse                             │
│  - success: bool                                                │
│  - message: str                                                 │
│  - task_type: TaskType                                          │
│  - tool_executions: List[ToolExecution]                         │
│  - data: Dict                                                   │
│  - suggestions: List[str]                                       │
└─────────────────────────────────────────────────────────────────┘
```

## Autonomy Levels

| Level | Read | Write | Execute | Delete | Use Case |
|-------|------|-------|---------|--------|----------|
| `READONLY` | ✅ | ❌ | ❌ | ❌ | Safe demos, read-only exploration |
| `MONITORED` | ✅ | ❌ | ❌ | ❌ | Logged read-only access |
| `ASSISTED` | ✅ | ✅ | 🔐 | ❌ | Interactive use with confirmations |
| `SUPERVISED` | ✅ | ✅ | ✅ | ❌ | Trusted users, most operations |
| `AUTONOMOUS` | ✅ | ✅ | ✅ | ✅ | Full autonomy with safety limits |

🔐 = Requires user approval

## Tool Permission Mapping

### Read-Only Tools (always allowed)
- `scan_data`, `search_databases`, `search_tcga`
- `describe_files`, `validate_dataset`
- `list_workflows`, `check_references`
- `get_job_status`, `get_logs`, `watch_job`, `list_jobs`
- `check_system_health`
- `explain_concept`, `compare_samples`, `show_help`
- `diagnose_error`, `analyze_results`
- `visualize_workflow`

### Write Tools (ASSISTED+ level)
- `download_dataset`, `download_reference`
- `generate_workflow`, `download_results`

### Execute Tools (SUPERVISED+ or approval at ASSISTED)
- `build_index`
- `submit_job`, `cancel_job`, `resubmit_job`
- `restart_vllm`, `recover_error`
- `run_command`

### Delete Tools (AUTONOMOUS only, still logged)
- `cleanup_data`, `confirm_cleanup`

## Usage Examples

### Basic Usage
```python
from workflow_composer.agents import UnifiedAgent, AutonomyLevel

# Create agent with desired permission level
agent = UnifiedAgent(autonomy_level=AutonomyLevel.ASSISTED)

# Process a query (async)
response = await agent.process_query("scan /data/raw for FASTQ files")

# Or synchronously
response = agent.process_sync("explain what RNA-seq is")

print(response.success)
print(response.message)
```

### Different Autonomy Levels
```python
# Read-only mode (safe for demos)
readonly_agent = UnifiedAgent(autonomy_level=AutonomyLevel.READONLY)

# Full autonomy (for trusted environments)
auto_agent = UnifiedAgent(autonomy_level=AutonomyLevel.AUTONOMOUS)

# Change level dynamically
agent.set_autonomy_level(AutonomyLevel.SUPERVISED)
```

### Handling Approvals
```python
response = agent.process_sync("submit job in /workflows/rna-seq")

if response.requires_approval:
    print(f"Approval needed: {response.approval_request}")
    
    # User approves
    agent.approve_action(response.approval_request['id'])
    
    # Or denies
    agent.deny_action(response.approval_request['id'])
```

### Checking History
```python
history = agent.get_history(limit=10)
for entry in history:
    print(f"{entry['task_type']}: {entry['success']}")
    
agent.clear_history()
```

## Components Integration

### 1. UnifiedAgent (New Entry Point)
- **File**: `agents/unified_agent.py` (~865 lines)
- **Class**: `UnifiedAgent`
- Combines orchestration + tool execution + permissions

### 2. AgentTools (Tool Implementations)
- **File**: `agents/tools/__init__.py`
- **29 tools** across 6 categories
- Pattern-based detection, OpenAI function definitions

### 3. Executor Layer (Safety + Audit)
- **CommandSandbox**: Secure command execution
- **PermissionManager**: Permission checking
- **AuditLogger**: Complete audit trail
- **FileOperations**: Safe file I/O

### 4. Autonomous Components (Advanced Features)
- **HealthChecker**: System health monitoring
- **RecoveryManager**: Automated error recovery
- **JobMonitor**: SLURM job watching
- **ReactAgent**: Multi-step reasoning (future)
- **AgentMemory**: Context persistence (future)

## Task Classification

Queries are automatically classified into task types:

| Type | Keywords | Example |
|------|----------|---------|
| WORKFLOW | workflow, pipeline, generate, rna-seq | "generate an RNA-seq workflow" |
| DIAGNOSIS | error, fail, diagnose, debug, crash | "why did my job fail" |
| DATA | scan, find, download, fastq, bam | "scan /data for FASTQ files" |
| JOB | job, submit, status, slurm, cancel | "what jobs are running" |
| EDUCATION | explain, what is, how does | "explain what ChIP-seq is" |
| SYSTEM | health, vllm, restart, server | "check system health" |
| ANALYSIS | analyze, results, compare | "analyze the results" |
| GENERAL | (fallback) | "hello" |

## Testing

```bash
# Run unified agent tests (29 tests)
pytest tests/test_unified_agent.py -v

# Run all tests
pytest tests/ -v
```

## Future Enhancements

1. **ReactAgent Integration**: Multi-step reasoning for complex tasks
2. **AgentMemory**: Persistent context across sessions
3. **CodingAgent**: Code generation for custom workflows
4. **Web Interface**: Gradio/React frontend with approval UI
5. **Background Tasks**: Async task queue for long-running operations

## Migration Guide

### Before (Direct Tool Access)
```python
from workflow_composer.agents import get_agent_tools
tools = get_agent_tools()
result = tools.execute_tool("scan_data", path="/data")
```

### After (Unified Agent with Permissions)
```python
from workflow_composer.agents import UnifiedAgent, AutonomyLevel
agent = UnifiedAgent(autonomy_level=AutonomyLevel.ASSISTED)
response = agent.process_sync("scan /data for files")
```

The old API still works for backward compatibility, but the UnifiedAgent provides:
- Permission checking
- Audit logging
- Task classification
- Approval workflows
- History tracking
