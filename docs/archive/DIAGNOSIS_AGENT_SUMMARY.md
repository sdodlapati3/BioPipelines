# Error Diagnosis Agent - Implementation Summary

## Overview

The BioPipelines Error Diagnosis Agent provides AI-powered analysis of pipeline failures with automatic fix suggestions. It uses a tiered approach combining fast pattern matching with deep LLM analysis for unknown errors.

## Architecture

```
src/workflow_composer/diagnosis/
├── __init__.py           # Package exports
├── categories.py         # Error taxonomy (12 categories)
├── patterns.py           # 50+ regex patterns with fixes
├── log_collector.py      # Multi-source log aggregation
├── prompts.py            # LLM prompt templates
├── agent.py              # Main diagnosis orchestrator
├── auto_fix.py           # Fix execution with rollback
├── gemini_adapter.py     # Google Gemini free tier
├── lightning_adapter.py  # Lightning.ai integration
└── github_agent.py       # GitHub Copilot for code fixes
```

## Features

### ✅ Implemented

1. **Error Categories** (12 types)
   - FILE_NOT_FOUND, OUT_OF_MEMORY, CONTAINER_ERROR
   - PERMISSION_DENIED, DEPENDENCY_MISSING, TOOL_ERROR
   - NETWORK_ERROR, SLURM_ERROR, DATA_FORMAT_ERROR
   - TIMEOUT, CONFIG_ERROR, UNKNOWN

2. **Pattern Matching** (50+ patterns)
   - Fast, offline detection (<100ms)
   - Covers common bioinformatics errors
   - Each pattern has suggested fixes

3. **LLM Provider Support**
   - ⚡ Lightning.ai (30M FREE tokens/month)
   - 🟢 Google Gemini (free tier)
   - 🟠 OpenAI (paid backup)
   - 🔵 Ollama/vLLM (local/free)

4. **Log Collection**
   - Nextflow main log (.nextflow.log)
   - SLURM output/error files
   - Process-specific logs
   - Trace files

5. **Fix Suggestions**
   - Risk-level classification (SAFE/LOW/MEDIUM/HIGH)
   - Shell commands where applicable
   - Auto-executable flags for safe fixes

6. **Gradio UI Integration**
   - "🔍 Diagnose" button in Execute tab
   - AI Diagnosis accordion section
   - Formatted diagnosis reports

### 📋 Coming Soon

- Auto-fix execution with user confirmation
- GitHub Copilot PR creation for code fixes
- Historical error tracking

## Usage

### In Code

```python
from workflow_composer.diagnosis import ErrorDiagnosisAgent

agent = ErrorDiagnosisAgent()

# For a PipelineJob object
diagnosis = agent.diagnose_sync(job)

# From raw log text
diagnosis = agent.diagnose_from_logs(log_content)

print(f"Error: {diagnosis.category.value}")
print(f"Root Cause: {diagnosis.root_cause}")
print(f"Confidence: {diagnosis.confidence:.0%}")

for fix in diagnosis.suggested_fixes:
    print(f"- {fix.description}")
    if fix.command:
        print(f"  Command: {fix.command}")
```

### In Gradio UI

1. Navigate to **Execute** tab
2. Select a failed job from the dropdown
3. Expand **🔍 AI Diagnosis** section
4. Click **Diagnose** button
5. Review the diagnosis report with:
   - Error classification
   - Root cause analysis
   - User-friendly explanation
   - Suggested fixes with risk levels

## Configuration

### Environment Variables

```bash
# Lightning.ai (recommended - free tier)
export LIGHTNING_API_KEY="your-key-here"

# Google Gemini (free tier)
export GOOGLE_API_KEY="your-key-here"

# OpenAI (paid backup)
export OPENAI_API_KEY="your-key-here"

# GitHub (for code fixes)
export GITHUB_TOKEN="your-token-here"
```

### Provider Priority

The agent tries providers in this order:
1. Lightning.ai (free, fast)
2. Gemini (free, reliable)
3. Ollama (local, free)
4. OpenAI (paid, comprehensive)

## Error Categories Reference

| Category | Description | Example |
|----------|-------------|---------|
| FILE_NOT_FOUND | Missing input/reference files | "No such file or directory" |
| OUT_OF_MEMORY | Memory allocation failures | "Cannot allocate memory" |
| CONTAINER_ERROR | Singularity/Docker issues | "FATAL: container creation failed" |
| PERMISSION_DENIED | Access denied to files | "Permission denied" |
| DEPENDENCY_MISSING | Missing tools/packages | "command not found" |
| TOOL_ERROR | Tool execution failures | "STAR alignment failed" |
| NETWORK_ERROR | Connection failures | "Connection refused" |
| SLURM_ERROR | Job scheduler errors | "Job exceeded time limit" |
| DATA_FORMAT_ERROR | Invalid input formats | "not in FASTQ format" |
| TIMEOUT | Process timeouts | "Process exceeded time limit" |
| CONFIG_ERROR | Configuration problems | "Invalid parameter" |
| UNKNOWN | Unclassified errors | Other failures |

## Risk Levels for Fixes

| Level | Description | Auto-Executable |
|-------|-------------|-----------------|
| SAFE | No side effects | ✅ Yes |
| LOW | Minimal risk | ⚠️ With notification |
| MEDIUM | Moderate risk | ❌ Needs confirmation |
| HIGH | Significant risk | ❌ Manual only |

## Files Modified

- `src/workflow_composer/web/gradio_app.py` - Added diagnosis UI and handler
- `src/workflow_composer/diagnosis/` - New package (9 files)

## Testing

```bash
# Test imports
python -c "from workflow_composer.diagnosis import ErrorDiagnosisAgent; print('OK')"

# Test pattern matching
python -c "
from workflow_composer.diagnosis import ErrorDiagnosisAgent
agent = ErrorDiagnosisAgent()
result = agent.diagnose_from_logs('Error: No such file or directory: /data/ref.fa')
print(f'Category: {result.category.value}')
print(f'Confidence: {result.confidence:.0%}')
"
```

## Next Steps

1. **Test with real failures** - Run diagnosis on actual failed pipeline logs
2. **Enable auto-fix** - Implement the "Apply Safe Fixes" button
3. **GitHub PR creation** - Wire up Copilot integration for code fixes
4. **Track history** - Store diagnosis results for pattern learning

---
*Last updated: 2024 | BioPipelines Error Diagnosis Agent v1.0*
