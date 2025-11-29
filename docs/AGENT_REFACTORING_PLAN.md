# Agent System Refactoring - Detailed Implementation Plan

**Created:** November 28, 2025  
**Status:** ✅ COMPLETED  
**Goal:** Address all weaknesses and implement opportunities identified in AGENT_SYSTEM_REVIEW.md

---

## Summary of Changes

### What Was Done
1. ✅ **Phase 1: Tool Modularization** - Split 3,253-line tools.py into 9 category-based modules
2. ✅ **Phase 2: Unified Interface** - Created chat_handler.py with session management and LLM fallback
3. ✅ **Phase 3: Robustness** - Added retry logic, validation utilities, argument validation
4. ✅ **Phase 4: Documentation** - Updated this plan with completion status

### Files Created
```
src/workflow_composer/agents/tools/
├── __init__.py          # Unified AgentTools class (205 lines)
├── base.py              # ToolResult, ToolName, validation utilities (240 lines)
├── registry.py          # ToolRegistry pattern (135 lines)
├── data_discovery.py    # scan_data, search_databases, describe_files (520 lines)
├── data_management.py   # download_dataset, cleanup_data (285 lines)
├── workflow.py          # generate_workflow, list_workflows (225 lines)
├── execution.py         # submit_job, get_job_status (180 lines)
├── diagnostics.py       # diagnose_error, analyze_results (165 lines)
└── education.py         # explain_concept, compare_samples, get_help (205 lines)

src/workflow_composer/web/
├── app.py               # Simplified from 38KB to 9KB
└── chat_handler.py      # New unified chat handler (620 lines)
```

### Files Archived
```
deprecated/web/
├── gradio_app.py        # Old Gradio interface
└── app_legacy.py        # Old monolithic app.py
```

---

## Overview

This plan consolidates the BioPipelines agent system into a unified, maintainable architecture.

### Scope
1. ✅ Archive gradio_app.py, use only app.py
2. ✅ Split tools.py into category-based modules
3. ✅ Create UnifiedChatHandler as single interface
4. ✅ Implement ToolRegistry pattern
5. 🔲 Integrate AgentMemory into main flow (future)
6. 🔲 Add ResponseValidator to app.py (future - partial)
7. ✅ Consolidate provider management

---

## Phase 1: Consolidation ✅ COMPLETED

### 1.1 Archive gradio_app.py
- [x] Move `gradio_app.py` to `deprecated/web/`
- [x] Update imports in `__init__.py` if any
- [x] Ensure `app.py` is the only entry point

### 1.2 Split tools.py into Modules

Current: `tools.py` (3,253 LOC)

Target structure: ✅ IMPLEMENTED
```
agents/tools/
├── __init__.py          # Exports all tools, ToolName, ToolResult
├── base.py              # ToolResult, ToolName enum, base classes
├── registry.py          # ToolRegistry pattern implementation
├── data_discovery.py    # scan_data, search_databases, search_tcga, describe_files, validate_dataset
├── data_management.py   # download_dataset, cleanup_data, confirm_cleanup
├── workflow.py          # generate_workflow, list_workflows, check_references
├── execution.py         # submit_job, get_job_status, get_logs, cancel_job
├── diagnostics.py       # diagnose_error, analyze_results
└── education.py         # explain_concept, compare_samples, get_help
```

**Implementation Details:**
- Created `AgentTools` class as unified interface in `__init__.py`
- Added `ToolName` enum and `TOOL_PATTERNS` in `base.py`
- Added validation utilities: `validate_path()`, `validate_dataset_id()`
- Added `ToolResult.success_result()` and `ToolResult.error_result()` helpers

---

## Phase 2: Unified Agent Interface ✅ COMPLETED

### 2.1 Create UnifiedChatHandler

File: `web/chat_handler.py`

```python
class UnifiedChatHandler:
    """Unified chat handler with:
    - Pattern-based tool detection (fast, no LLM cost)
    - LLM function calling (for complex queries)
    - Graceful fallbacks with retry logic
    - Session management
    """
    
    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider
        self.session_manager = SessionManager()
        self._tools = get_agent_tools()
    
    def chat(self, message: str, history: list = None) -> Generator:
        """Main chat interface with tool integration."""
        ...
```

### 2.2 Integrate into app.py

New simplified app.py (9KB vs 38KB):
```python
from workflow_composer.web.chat_handler import get_chat_handler

chat_handler = get_chat_handler()

def chat_response(message, history):
    for response in chat_handler.chat(message, history):
        yield response
```

---

## Phase 3: ToolRegistry Pattern

### 3.1 Implement ToolRegistry

File: `agents/tools/registry.py`

Features:
- Decorator-based tool registration
- Auto-generate OpenAI function calling format
- Pattern registration for regex fallback
- Validation rules per tool

```python
class ToolRegistry:
    _tools: Dict[str, RegisteredTool] = {}
    
    @classmethod
    def register(cls, 
                 name: str, 
                 description: str,
                 parameters: dict,
                 patterns: List[str] = None):
        """Decorator to register a tool."""
        def decorator(func):
            cls._tools[name] = RegisteredTool(
                name=name,
                description=description,
                parameters=parameters,
                patterns=patterns or [],
                handler=func
            )
            return func
        return decorator
    
    @classmethod
    def get_openai_tools(cls) -> List[dict]:
        """Generate OpenAI function calling definitions."""
        return [tool.to_openai_format() for tool in cls._tools.values()]
    
    @classmethod
    def get_patterns(cls) -> List[Tuple[str, str]]:
        """Get all regex patterns for fallback."""
        patterns = []
        for tool in cls._tools.values():
            for pattern in tool.patterns:
                patterns.append((pattern, tool.name))
        return patterns
```

### 3.2 Convert Tools to Use Registry

Before:
```python
# In monolithic tools.py
TOOL_PATTERNS = [
    (r"scan\s+(?:data|files)\s+(?:in|at|from)\s+(.+)", ToolName.SCAN_DATA),
    ...
]

class AgentTools:
    def scan_data(self, path: str) -> ToolResult:
        ...
```

After:
```python
# In agents/tools/data_discovery.py
from .registry import ToolRegistry, ToolResult

@ToolRegistry.register(
    name="scan_data",
    description="Scan a directory for FASTQ/BAM sequencing files",
    parameters={
        "path": {"type": "string", "description": "Directory path to scan", "required": True}
    },
    patterns=[
        r"scan\s+(?:data|files)\s+(?:in|at|from)\s+['\"]?([^\s'\"]+)['\"]?",
        r"(?:what|show)\s+(?:data|files|samples)\s+(?:are\s+)?(?:in|at)\s+['\"]?([^\s'\"]+)['\"]?",
    ]
)
def scan_data(path: str = None) -> ToolResult:
    """Scan directory for sequencing data."""
    ...
```

---

## Phase 4: Memory & Validation Integration

### 4.1 Integrate AgentMemory

Update UnifiedBioAgent to use memory:
```python
def process(self, message: str, history: list = None) -> AgentResponse:
    # Get relevant context from memory
    if self.memory:
        memory_context = self.memory.get_context(message)
        # Include in LLM prompt
    
    # Process message...
    result = self._execute(message)
    
    # Store in memory for future
    if self.memory:
        self.memory.store(message, result)
    
    return result
```

### 4.2 Integrate ResponseValidator

Add validation to response flow:
```python
def process(self, message: str, history: list = None) -> AgentResponse:
    # Update context with user message
    self.context.add_message("user", message)
    
    # Execute tool/LLM
    raw_result = self._execute(message)
    
    # Validate response matches intent
    if self.validator:
        validated = self.validator.validate(
            user_intent=self.context.current_intent,
            tool_result=raw_result
        )
        if not validated.is_valid:
            # Add warning to response
            raw_result.message = f"⚠️ {validated.issues[0]}\n\n{raw_result.message}"
    
    return raw_result
```

---

## Phase 5: Provider Consolidation

### 5.1 Single Provider Management

Remove duplicate provider logic:
- Keep: `providers/router.py` (ProviderRouter)
- Update: `app.py` to use ProviderRouter instead of LLMProvider class
- Remove: Inline provider setup in app.py

```python
# In app.py
from workflow_composer.providers import ProviderRouter, get_router

router = get_router()  # Single source of truth

def get_llm_response(messages):
    return router.chat(messages, fallback=True)
```

---

## Implementation Checklist

### Phase 1: Consolidation
- [ ] 1.1.1 Create `deprecated/web/` directory
- [ ] 1.1.2 Move `gradio_app.py` to deprecated
- [ ] 1.2.1 Create `agents/tools/` directory structure
- [ ] 1.2.2 Create `agents/tools/base.py` (ToolResult, ToolName)
- [ ] 1.2.3 Create `agents/tools/registry.py` (ToolRegistry)
- [ ] 1.2.4 Create `agents/tools/data_discovery.py`
- [ ] 1.2.5 Create `agents/tools/data_management.py`
- [ ] 1.2.6 Create `agents/tools/workflow.py`
- [ ] 1.2.7 Create `agents/tools/execution.py`
- [ ] 1.2.8 Create `agents/tools/monitoring.py`
- [ ] 1.2.9 Create `agents/tools/diagnostics.py`
- [ ] 1.2.10 Create `agents/tools/education.py`
- [ ] 1.2.11 Create `agents/tools/__init__.py` (assemble AgentTools)
- [ ] 1.2.12 Update imports in agents/__init__.py
- [ ] 1.2.13 Archive old tools.py
- [ ] 1.2.14 Test all tools still work

### Phase 2: Unified Agent
- [ ] 2.1.1 Create `agents/unified.py`
- [ ] 2.1.2 Implement UnifiedBioAgent class
- [ ] 2.1.3 Add process(), stream(), process_async() methods
- [ ] 2.2.1 Update app.py to use UnifiedBioAgent
- [ ] 2.2.2 Remove old chat logic from app.py
- [ ] 2.2.3 Test chat functionality

### Phase 3: ToolRegistry
- [ ] 3.1.1 Implement ToolRegistry.register() decorator
- [ ] 3.1.2 Implement ToolRegistry.get_openai_tools()
- [ ] 3.1.3 Implement ToolRegistry.get_patterns()
- [ ] 3.2.1 Convert all tools to use @ToolRegistry.register
- [ ] 3.2.2 Update app.py TOOLS list to use registry
- [ ] 3.2.3 Test pattern matching still works

### Phase 4: Memory & Validation
- [ ] 4.1.1 Integrate AgentMemory into UnifiedBioAgent
- [ ] 4.1.2 Add memory context to LLM prompts
- [ ] 4.1.3 Store interactions in memory
- [ ] 4.2.1 Integrate ResponseValidator into process()
- [ ] 4.2.2 Add confidence display to responses
- [ ] 4.2.3 Test validation catches mismatches

### Phase 5: Provider Consolidation
- [ ] 5.1.1 Update app.py to use ProviderRouter
- [ ] 5.1.2 Remove LLMProvider class from app.py
- [ ] 5.1.3 Test fallback chain still works

---

## Testing Plan

After each phase, run:
```bash
# Unit tests
python -m pytest tests/test_agentic_router.py -v

# Integration test
python -c "
from workflow_composer.agents import AgentTools
tools = AgentTools()
print(tools.scan_data('/tmp'))
print(tools.search_tcga('GBM'))
print(tools.generate_workflow('RNA-seq'))
"

# Web interface test
cd scripts && ./start_server.sh --cloud
```

---

## Rollback Plan

If issues arise:
1. All original files preserved in `deprecated/`
2. Git history allows reverting any commit
3. Old imports can be restored from deprecated modules

---

## Timeline

| Phase | Estimated Time | Status |
|-------|---------------|--------|
| Phase 1: Consolidation | 2-3 hours | 🔄 Starting |
| Phase 2: Unified Agent | 1-2 hours | ⏳ Pending |
| Phase 3: ToolRegistry | 1-2 hours | ⏳ Pending |
| Phase 4: Memory & Validation | 1 hour | ⏳ Pending |
| Phase 5: Provider Consolidation | 30 min | ⏳ Pending |
| Testing & Fixes | 1 hour | ⏳ Pending |

**Total: ~8 hours**

---

*Let's begin implementation!*
