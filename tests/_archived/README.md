# Archived Tests

These tests are for deprecated modules that have been archived.

## Archived Modules

| Test File | Module | Replacement |
|-----------|--------|-------------|
| `test_agentic_router.py` | `agents/router.py` | Use `UnifiedAgent` with `HybridQueryParser` |

## Why Archived?

These modules were superseded by the unified agent architecture:

- **AgentRouter**: LLM-based intent detection is now handled by `HybridQueryParser` 
  which combines semantic similarity, pattern matching, and NER for better accuracy.
  
- **AgentBridge**: The bridge between router and tools is now handled internally 
  by `UnifiedAgent` which provides permission control, audit logging, and multi-step
  task support.

## If You Need These Tests

If you need to run these archived tests:

```bash
cd tests/_archived
pytest test_agentic_router.py --no-cov
```

Note: You'll need to update the imports to use `workflow_composer.agents._archived.router`.

## Date Archived

2025-11-29
