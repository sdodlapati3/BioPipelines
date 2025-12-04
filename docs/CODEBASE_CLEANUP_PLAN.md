# BioPipelines Codebase Cleanup Plan

**Created**: December 4, 2025  
**Status**: Phase 1 Complete ✅  
**Author**: AI Assistant (Claude)  
**Last Updated**: December 4, 2025

---

## Completed Cleanups

### ✅ December 4, 2025 - Phase 1 Implemented
1. **Renamed duplicate class** in `arbiter.py`: `UnifiedIntentParser` → `SimpleArbiterParser`
2. **Deleted unused directory**: `web/archive/` (4 files removed)
3. **Verified imports**: All core imports working correctly

---

## Executive Summary

This document outlines the cleanup of redundant, deprecated, and legacy code in the BioPipelines codebase. Each task is verified before implementation to ensure we only remove truly unused code.

---

## Pre-Implementation Verification Checklist

Before removing any code, we verify:
1. ✅ No active imports from the code
2. ✅ No references in tests (or tests are also deprecated)
3. ✅ Functionality is available through replacement code
4. ✅ Documentation exists for migration path

---

## Task 1: Duplicate `UnifiedIntentParser` Class

### Problem
Two classes with identical names exist:
- `agents/intent/unified_parser.py:136` - Full-featured (1100+ lines)
- `agents/intent/arbiter.py:589` - Simpler version (200 lines)

### ✅ Verification Results
- [x] `agents/intent/__init__.py` imports from `unified_parser.py` (line 93-95)
- [x] `UnifiedAgent` uses the one from `unified_parser.py` via `__init__.py`
- [x] No code imports `UnifiedIntentParser` from `arbiter.py` directly
- [x] The arbiter.py version is only used internally within that file

### Proposed Action
Rename the class in `arbiter.py` from `UnifiedIntentParser` to `SimpleArbiterParser` to avoid confusion.
This is **SAFE** - the class is not exported or imported externally.

---

## Task 2: Triple `ProviderType` Enum

### Problem
Three separate definitions with inconsistent values:
- `llm/providers/base.py:44` → `LOCAL="local"`, `CLOUD="cloud"`
- `providers/registry.py:22` → `API="api"`, `LOCAL="local"`
- `models/registry.py:18` → `API="api"`, `LOCAL="local"`

### Verification Steps
- [ ] Map all usages of each ProviderType
- [ ] Determine which enum is the "canonical" one
- [ ] Check if the different values (CLOUD vs API) serve different purposes

### Proposed Action
Keep all three for now - they serve different contexts:
- `llm/providers/base.py` - For LLM routing (local GPU vs cloud API)
- `providers/registry.py` - For general provider registry
- `models/registry.py` - For model configuration

Add documentation clarifying their different purposes rather than consolidating.

### Alternative
If truly redundant after verification, consolidate into `infrastructure/enums.py`.

---

## Task 3: Archived Agent Components (`agents/_archived/`)

### Problem
Directory contains deprecated code:
- `bridge.py` - Legacy AgentBridge
- `router.py` - Legacy AgentRouter  
- `context/` - Legacy context management

### ✅ Verification Results
- [x] Imports exist ONLY in backward-compat layer (`agents/__init__.py` lines 53, 63)
- [x] These imports emit `DeprecationWarning` when accessed
- [x] `AgentRouter` in `test_human_handoff.py` is a DIFFERENT class (handoff routing, not archived)
- [x] Test file `tests/_archived/test_agentic_router.py` is also archived with README
- [x] `UnifiedAgent` provides equivalent functionality

### Proposed Action
**DO NOT DELETE YET** - The backward-compat imports with deprecation warnings are the correct approach.
The code is properly archived and will emit warnings if anyone uses it.

Keep for one more release cycle, then delete both:
- `src/workflow_composer/agents/_archived/`
- `tests/_archived/`

---

## Task 4: Archived Web Components (`web/archive/`)

### Problem
Old web implementations:
- `api.py` - Legacy FastAPI backend (600 lines)
- `app.py` - Legacy Flask UI (713 lines)
- `result_browser.py` - Old result browser
- `unified_workspace.py` - Old workspace manager

### ✅ Verification Results
- [x] NO imports from `web.archive` in any Python file
- [x] `web/__init__.py` only imports from `app.py` and `utils.py` (not archive)
- [x] Code quality report shows these files have unused imports (dead code)
- [x] Gradio `web/app.py` provides all current functionality
- [x] No Docker/deployment configs reference these files

### Proposed Action
**SAFE TO DELETE** - Remove `web/archive/` directory entirely.
These files are truly unused and have clear replacements.

---

## Task 5: Deprecated TOOL_PATTERNS

### Problem
In `agents/tools/base.py:89-107`, patterns marked as deprecated:
```python
# LEGACY TOOL_PATTERNS - DEPRECATED
# These patterns will be removed in a future version.
TOOL_PATTERNS = [...]
```

### ✅ Verification Results
- [x] `TOOL_PATTERNS` is imported in `agents/tools/__init__.py` line 22
- [x] `TOOL_PATTERNS` is re-exported in `__all__` (line 1578)
- [x] The ACTIVE patterns are `ALL_TOOL_PATTERNS` (line 219) - different variable
- [x] `AgentTools.detect_tool()` uses `ALL_TOOL_PATTERNS`, NOT the legacy one (line 910)
- [x] Legacy `TOOL_PATTERNS` appears to be a simpler fallback list

### Proposed Action
**SAFE TO REMOVE** the legacy `TOOL_PATTERNS` constant since:
1. `ALL_TOOL_PATTERNS` is the comprehensive list actually used
2. The import and export can be removed
3. The deprecation comment says "will be removed in future version"

OR keep with deprecation warning if external code might use it.

---

## Task 6: Legacy LLM Adapter Layer

### Problem
Two adapter systems coexist:
- `llm/*.py` - Legacy adapters (OllamaAdapter, OpenAIAdapter, etc.)
- `providers/*.py` - New provider system

### Verification Steps
- [ ] Check usage count of legacy adapters
- [ ] Check usage count of new providers
- [ ] Verify feature parity
- [ ] Check `get_llm()` factory function

### Proposed Action
**Do NOT remove** - These are actively used and marked as "still supported".
Add documentation clarifying:
- Legacy adapters: For direct LLM access
- Providers: For orchestrated/routed access

---

## Task 7: Duplicate `providers/` Directories

### Problem
Two provider directories:
- `llm/providers/` - Unified local/cloud providers
- `providers/` - Individual provider implementations

### Verification Steps
- [ ] Map imports from each
- [ ] Check if they serve different purposes
- [ ] Identify any circular dependencies

### Proposed Action
Keep both - they serve different architectural layers:
- `llm/providers/` - Abstract provider interface for LLM orchestrator
- `providers/` - Concrete implementations for different services

Document the distinction in `README.md` files.

---

## Implementation Order

### Phase 1: Safe Cleanups (Verified Safe)
1. ✅ Task 1: Rename duplicate `UnifiedIntentParser` in arbiter.py → `SimpleArbiterParser` **[DONE]**
2. ✅ Task 4: Delete `web/archive/` directory (no imports found) **[DONE]**
3. ⚠️ Task 5: Remove legacy `TOOL_PATTERNS` (or add deprecation warning) **[OPTIONAL]**

### Phase 2: Keep With Documentation
4. ⏸️ Task 3: Keep `agents/_archived/` - deprecation warnings already in place
5. ⏸️ Task 2: Keep triple `ProviderType` - they serve different purposes

### Phase 3: Documentation Only
6. ⏸️ Task 6: Document adapter layers (no code changes)
7. ⏸️ Task 7: Document provider directories (no code changes)

---

## Verification Commands

```bash
# Find all imports of a module
grep -r "from.*_archived" src/ tests/ --include="*.py"
grep -r "import.*_archived" src/ tests/ --include="*.py"

# Find all usages of a class
grep -r "UnifiedIntentParser" src/ tests/ --include="*.py"

# Find all usages of an enum
grep -r "ProviderType" src/ tests/ --include="*.py"

# Check for web.archive imports
grep -r "from.*web\.archive" src/ tests/ --include="*.py"
grep -r "web/archive" . --include="*.py"
```

---

## Success Criteria

- [ ] All tests pass after cleanup
- [ ] No new deprecation warnings in normal operation
- [ ] Documentation updated for any API changes
- [ ] Git history preserved for reverted code

---

## Task 8: Parser Architecture Redundancy (CRITICAL)

### Problem
**7+ parser classes exist with significant overlap**, causing:
1. Confusion about which parser is authoritative
2. Complex fallback chains that obscure failures
3. Inconsistent pattern coverage (some intents only in some parsers)
4. Log messages like "Parallel parsing failed", "Parse error", "fallback to HybridParser"

### Current Parser Inventory

| File | Class | Lines | Purpose | Status |
|------|-------|-------|---------|--------|
| `core/query_parser.py:284` | IntentParser | ~200 | Legacy LLM-based | ⚠️ Legacy |
| `core/query_parser_ensemble.py:439` | EnsembleIntentParser | ~300 | Legacy ensemble | ⚠️ Legacy |
| `core/model_service_manager.py:452` | AdaptiveIntentParser | ~150 | Adaptive routing | ⚠️ Legacy |
| `agents/intent/parser.py:1195` | IntentParser | ~800 | Pattern-based | ✅ Active |
| `agents/intent/semantic.py:978` | HybridQueryParser | ~400 | Hybrid approach | ⚠️ Fallback Only |
| `agents/intent/unified_parser.py:136` | UnifiedIntentParser | ~1100 | **Main orchestrator** | ✅ **Recommended** |
| `agents/intent/arbiter.py:590` | SimpleArbiterParser | ~200 | Internal eval | 🔧 Internal |
| `agents/intent/learning.py:802` | LearningHybridParser | ~300 | Experimental | 🧪 Experimental |
| `data/discovery/query_parser.py:77` | QueryParser | ~200 | Data domain | ✅ Domain-specific |

### Current Integration Flow

```
Frontend (web/app.py)
    ↓ bp.chat() / bp.chat_stream()
BioPipelines Facade (facade.py)
    ↓ chat_agent.process_message()
ChatAgent (intent/chat_agent.py)
    ↓ _is_tool_query() check → _execute_via_unified_agent_sync()
UnifiedAgent (unified_agent.py)
    ↓ process_query()
    ↓ self.intent_parser.parse() → UnifiedIntentParser [PRIMARY]
    ↓ (if None) → self.query_parser.parse() → HybridQueryParser [FALLBACK]
```

### ✅ Verification Results
- [x] `UnifiedIntentParser` is the PRIMARY parser used by `UnifiedAgent`
- [x] `HybridQueryParser` is only used as FALLBACK when primary returns `None`
- [x] `IntentParser` (agents/intent/parser.py) is used INTERNALLY by `UnifiedIntentParser`
- [x] Legacy `core/*.py` parsers are NOT imported by current agent system
- [x] `ChatAgent` routes tool queries to `UnifiedAgent` (correct path)
- [x] Parse failures cascade through multiple fallback layers, making debugging hard

### Root Cause of Recent Failure
The query "show me details of methylation data" failed because:
1. `DATA_DESCRIBE` intent had NO regex patterns in `IntentParser`
2. `UnifiedIntentParser` couldn't match any intent
3. Fallback to `HybridQueryParser` produced confusing result
4. Query was routed to ENCODE API instead of `describe_files` tool

### Proposed Consolidation Plan

**Phase 1: Pattern Coverage (Immediate)**
- ✅ Add missing `DATA_DESCRIBE` patterns to `parser.py` [DONE]
- ✅ Add `DESCRIBE_FILES_PATTERNS` to `data_discovery.py` [DONE]
- [ ] Audit ALL `IntentType` enums vs pattern coverage
- [ ] Add patterns for any missing intents

**Phase 2: Deprecate Legacy Parsers (Next Release)**
- [ ] Add deprecation warnings to `core/query_parser.py`
- [ ] Add deprecation warnings to `core/query_parser_ensemble.py`
- [ ] Add deprecation warnings to `core/model_service_manager.py` AdaptiveIntentParser
- [ ] Update any imports still using legacy parsers

**Phase 3: Simplify Fallback Chain (Future)**
- [ ] Remove `HybridQueryParser` fallback from `UnifiedAgent`
- [ ] Make `UnifiedIntentParser` the ONLY parser
- [ ] Improve `UnifiedIntentParser` error handling instead of fallbacks
- [ ] Add better logging for parse failures

**Phase 4: Remove Legacy Code (Major Release)**
- [ ] Delete `core/query_parser.py`
- [ ] Delete `core/query_parser_ensemble.py`  
- [ ] Remove AdaptiveIntentParser from `core/model_service_manager.py`
- [ ] Delete `HybridQueryParser` from `semantic.py`

### Architecture Target State

```
Frontend → BioPipelines → ChatAgent → UnifiedAgent
                                          ↓
                                    UnifiedIntentParser (ONLY parser)
                                          ↓
                              [Uses IntentParser patterns internally]
                                          ↓
                                    Tool Execution
```

---

## Appendix: Files to Review

### Definitely Remove (After Verification)
- ✅ `web/archive/api.py` [REMOVED]
- ✅ `web/archive/app.py` [REMOVED]
- ✅ `web/archive/result_browser.py` [REMOVED]
- ✅ `web/archive/unified_workspace.py` [REMOVED]

### Possibly Rename
- ✅ `agents/intent/arbiter.py` class `UnifiedIntentParser` → `SimpleArbiterParser` [DONE]

### Keep But Document
- `llm/*.py` legacy adapters
- `providers/*.py` 
- `llm/providers/*.py`
- Triple `ProviderType` enums

### Keep But Add Warnings
- `agents/tools/base.py` TOOL_PATTERNS

### Parser Files - Future Cleanup
- `core/query_parser.py` - Add deprecation, then delete
- `core/query_parser_ensemble.py` - Add deprecation, then delete
- `agents/intent/semantic.py` HybridQueryParser - Remove after consolidation
