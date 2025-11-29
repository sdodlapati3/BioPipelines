# Error Diagnosis & Auto-Fix Agent: Critical Analysis

## Current State Assessment

### What Exists Now

| Component | Status | Location |
|-----------|--------|----------|
| Error Detection | ⚠️ **Basic** | `gradio_app.py:_parse_nextflow_log()` |
| Error Display | ⚠️ **Limited** | Shows first 500 chars of error |
| Troubleshooting Chat | 🔶 **Partial** | Chatbot mentions "troubleshoot issues" |
| Auto-Fix | ❌ **Missing** | Not implemented |
| Log Analysis | ❌ **Missing** | No semantic analysis |
| Root Cause Detection | ❌ **Missing** | No pattern matching |

### Current Error Detection (What Works)

```python
# From gradio_app.py lines 326-358
error_patterns = [
    r"Error executing process",
    r"Pipeline failed",
    r"ERROR\s*[~\-]",
    r"No such file or directory",
    r"Command error:",
    r"Execution halted",
    r"FATAL:",
    r"Exception:",
]
```

**Limitations:**
1. Only detects errors, doesn't explain them
2. No categorization (file missing vs. memory vs. tool error)
3. No suggested fixes
4. No automatic remediation

---

## Proposed Solutions: Critical Analysis

### Option 1: Rule-Based Error Pattern Matcher
**Implementation Effort: 2-3 days**

```
┌─────────────────────────────────────────┐
│           RULE-BASED MATCHER            │
├─────────────────────────────────────────┤
│  Error Log → Pattern Match → Lookup     │
│  "No such file" → "Missing file"        │
│  "Memory" → "OOM - increase RAM"        │
│  "Container" → "Container issue"        │
└─────────────────────────────────────────┘
```

**Pros:**
- Fast (no LLM calls)
- Deterministic
- Works offline
- Low cost

**Cons:**
- Limited to known patterns
- Can't handle novel errors
- Requires manual pattern updates
- No contextual understanding

**Best For:** Quick fixes for common errors

---

### Option 2: LLM-Based Error Diagnosis Agent
**Implementation Effort: 5-7 days**

```
┌─────────────────────────────────────────────────────────────┐
│                LLM ERROR DIAGNOSIS AGENT                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐     ┌───────────────┐     ┌────────────────┐  │
│  │ Error Log│ ──▶ │ Context Build │ ──▶ │ LLM Analysis   │  │
│  │ + .err   │     │  + workflow   │     │  + diagnosis   │  │
│  │ + config │     │  + env info   │     │  + suggestions │  │
│  └──────────┘     └───────────────┘     └────────────────┘  │
│                                                    │         │
│                                                    ▼         │
│                            ┌─────────────────────────────┐  │
│                            │  Structured Output:         │  │
│                            │  - Error category           │  │
│                            │  - Root cause               │  │
│                            │  - Fix suggestions          │  │
│                            │  - Auto-fix commands        │  │
│                            └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Pros:**
- Understands novel errors
- Contextual analysis
- Natural language explanations
- Can learn from workflow context

**Cons:**
- Requires LLM availability
- API costs
- Potential hallucinations
- Slower response

**Best For:** Complex errors needing contextual understanding

---

### Option 3: Hybrid System (RECOMMENDED)
**Implementation Effort: 7-10 days**

```
┌────────────────────────────────────────────────────────────────────────┐
│                     HYBRID ERROR DIAGNOSIS SYSTEM                       │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  TIER 1: Fast Pattern Match (< 100ms)                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Known Patterns → Instant Fix Suggestions                        │   │
│  │  • "No such file X" → Check file path, download reference       │   │
│  │  • "MemoryError" → Increase SLURM memory                        │   │
│  │  • "Container not found" → Build/pull container                 │   │
│  │  • "Module not found" → Install package                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                           │                                             │
│                           ▼ (If no match)                               │
│  TIER 2: LLM Deep Analysis (3-10s)                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Context Collection:                                              │   │
│  │  • Nextflow log (.nextflow.log)                                  │   │
│  │  • SLURM error file (.err)                                       │   │
│  │  • Workflow definition (main.nf)                                 │   │
│  │  • Environment (modules, containers)                             │   │
│  │  • Recent changes to workspace                                   │   │
│  │                                                                   │   │
│  │  LLM Analysis:                                                    │   │
│  │  • Root cause identification                                      │   │
│  │  • Similar past errors (if logged)                               │   │
│  │  • Multi-step fix suggestions                                     │   │
│  │  • Confidence scoring                                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                           │                                             │
│                           ▼                                             │
│  TIER 3: Auto-Fix Execution (Optional, User-Approved)                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Safe Fixes (Auto):              Risky Fixes (Confirm):          │   │
│  │  • Create missing directories    • Modify workflow code          │   │
│  │  • Retry with increased memory   • Install new packages          │   │
│  │  • Pull missing container        • Change config files           │   │
│  │  • Download missing reference    • Modify SLURM params           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Design: Error Diagnosis Agent

### 1. Error Classification Taxonomy

```python
class ErrorCategory(Enum):
    FILE_NOT_FOUND = "file_not_found"      # Missing input/reference
    PERMISSION_DENIED = "permission"        # Access issues
    MEMORY_OOM = "out_of_memory"           # OOM killer
    CONTAINER_ERROR = "container"           # Singularity/Docker
    DEPENDENCY_MISSING = "dependency"       # Module/package not found
    TOOL_ERROR = "tool_error"              # Tool-specific failure
    SYNTAX_ERROR = "syntax"                 # Workflow syntax
    NETWORK_ERROR = "network"              # Download/connection
    RESOURCE_LIMIT = "resource"            # SLURM limits
    DATA_FORMAT = "data_format"            # Wrong input format
    UNKNOWN = "unknown"                    # Needs LLM analysis
```

### 2. Pattern Database

```python
ERROR_PATTERNS = {
    "file_not_found": {
        "patterns": [
            r"No such file or directory: (.+)",
            r"Path not found: (.+)",
            r"FileNotFoundError: (.+)",
            r"cannot open file '(.+)'",
        ],
        "auto_fixes": [
            {"type": "check_path", "description": "Verify file path exists"},
            {"type": "download_reference", "description": "Download missing reference"},
            {"type": "check_symlink", "description": "Check if symlink is valid"},
        ],
        "common_causes": [
            "Reference genome not downloaded",
            "Incorrect sample sheet paths",
            "Typo in file path",
            "Symlink pointing to deleted file",
        ]
    },
    "out_of_memory": {
        "patterns": [
            r"MemoryError",
            r"Out of memory",
            r"Killed.*memory",
            r"slurmstepd: error:.*oom-kill",
            r"exceeded memory limit",
        ],
        "auto_fixes": [
            {"type": "increase_memory", "description": "Double SLURM memory request"},
            {"type": "reduce_threads", "description": "Reduce parallel threads"},
            {"type": "split_input", "description": "Process files in smaller batches"},
        ],
        "common_causes": [
            "Input files too large for allocated memory",
            "Too many parallel processes",
            "Memory leak in tool",
        ]
    },
    "container_error": {
        "patterns": [
            r"FATAL:.*container",
            r"singularity: command not found",
            r"Failed to pull container",
            r"Image not found: (.+\.sif)",
        ],
        "auto_fixes": [
            {"type": "build_container", "description": "Build missing container"},
            {"type": "pull_container", "description": "Pull from registry"},
            {"type": "check_singularity", "description": "Verify Singularity is loaded"},
        ],
        "common_causes": [
            "Container image not built",
            "Singularity module not loaded",
            "Container registry unavailable",
        ]
    },
    # ... more patterns
}
```

### 3. LLM Diagnosis Prompt Template

```python
DIAGNOSIS_PROMPT = """You are an expert bioinformatics pipeline debugger.

## Error Context

**Workflow:** {workflow_name}
**Analysis Type:** {analysis_type}
**Failed Process:** {failed_process}

## Log Files

### Nextflow Log (last 100 lines):
```
{nextflow_log}
```

### SLURM Error File:
```
{slurm_err}
```

### Workflow Configuration:
```
{workflow_config}
```

## Task

1. **Identify the root cause** of this failure
2. **Classify the error** into one of: {error_categories}
3. **Explain** what went wrong in simple terms
4. **Suggest fixes** ranked by likelihood of success
5. **Provide commands** to fix if possible

## Output Format (JSON)

```json
{
  "error_category": "category_name",
  "root_cause": "Brief explanation",
  "user_explanation": "Non-technical explanation for user",
  "confidence": 0.0-1.0,
  "suggested_fixes": [
    {
      "description": "What to do",
      "command": "shell command if applicable",
      "auto_executable": true/false,
      "risk_level": "low/medium/high"
    }
  ],
  "similar_known_issues": ["list of related problems"],
  "prevention_tips": ["how to avoid this in future"]
}
```"""
```

### 4. Auto-Fix Safety Levels

```python
class FixRiskLevel(Enum):
    SAFE = "safe"           # Can execute automatically
    LOW = "low"             # Execute with notification
    MEDIUM = "medium"       # Require user confirmation
    HIGH = "high"           # Show instructions only

SAFE_AUTO_FIXES = {
    "create_directory": FixRiskLevel.SAFE,
    "retry_job": FixRiskLevel.SAFE,
    "pull_container": FixRiskLevel.LOW,
    "download_reference": FixRiskLevel.LOW,
    "increase_memory": FixRiskLevel.MEDIUM,
    "modify_config": FixRiskLevel.MEDIUM,
    "install_package": FixRiskLevel.HIGH,
    "modify_workflow": FixRiskLevel.HIGH,
}
```

---

## Integration with GitHub Copilot Coding Agent

### Option A: Use MCP GitHub for Complex Fixes
When fixes require code changes:

```python
async def request_coding_agent_fix(
    error_analysis: ErrorAnalysis,
    workflow_files: List[str]
) -> PullRequest:
    """
    For complex fixes that require code modifications,
    delegate to GitHub Copilot Coding Agent.
    """
    problem_statement = f"""
    ## Bug Fix Request
    
    A bioinformatics workflow failed with the following error:
    
    **Error Category:** {error_analysis.category}
    **Root Cause:** {error_analysis.root_cause}
    
    ### Error Log
    ```
    {error_analysis.log_excerpt}
    ```
    
    ### Files to Review
    {workflow_files}
    
    ### Suggested Fix
    {error_analysis.suggested_fixes[0].description}
    
    Please:
    1. Identify the bug in the workflow code
    2. Fix the issue
    3. Add any missing error handling
    4. Test the fix if possible
    """
    
    # Use existing MCP tool
    result = await mcp_github_create_pull_request_with_copilot(
        owner="username",
        repo="BioPipelines", 
        problem_statement=problem_statement,
        title=f"Fix: {error_analysis.root_cause[:50]}"
    )
    
    return result
```

### Option B: Local LLM Code Fixer
For immediate fixes without PR workflow:

```python
class LocalCodeFixer:
    """Use local LLM to generate code fixes."""
    
    def generate_fix(self, 
                     error: ErrorAnalysis,
                     file_content: str) -> CodePatch:
        prompt = f"""
        The following Nextflow workflow has an error:
        
        ```nextflow
        {file_content}
        ```
        
        Error: {error.root_cause}
        
        Generate a minimal fix. Output ONLY the corrected code section.
        """
        
        response = self.llm.complete(prompt)
        return self.parse_code_patch(response.content)
```

---

## Comparison Matrix

| Feature | Rule-Based | LLM-Only | Hybrid (Recommended) |
|---------|------------|----------|----------------------|
| **Speed** | ⚡ <100ms | 🐢 3-10s | ⚡/🐢 Tiered |
| **Novel Errors** | ❌ | ✅ | ✅ |
| **Offline** | ✅ | ❌ | ⚠️ Tier 1 only |
| **Cost** | Free | $0.01-0.10/query | $0-0.10/query |
| **Accuracy** | 70% known | 85% all | 90% all |
| **Context Aware** | ❌ | ✅ | ✅ |
| **Auto-Fix** | Limited | Full | Full + Safe |
| **Implementation** | 2-3 days | 5-7 days | 7-10 days |

---

## Implementation Plan

### Phase 1: Foundation (Days 1-3)
```
□ Create ErrorDiagnosisAgent class
□ Implement pattern database (30+ patterns)
□ Add error classification enum
□ Create log parsing utilities
□ Unit tests for pattern matching
```

### Phase 2: LLM Integration (Days 4-6)
```
□ Design diagnosis prompt template
□ Implement structured output parsing
□ Add context collection (logs, config, env)
□ Create confidence scoring
□ Integration with existing LLM adapters
```

### Phase 3: Auto-Fix Engine (Days 7-9)
```
□ Define safe vs. risky fixes
□ Implement auto-fix executors
□ Add user confirmation flow
□ Create fix verification checks
□ Log all auto-fix actions
```

### Phase 4: UI Integration (Days 9-10)
```
□ Add "Diagnose" button to Jobs table
□ Create diagnosis result panel
□ Add "Apply Fix" buttons
□ Integrate with chatbot for explanations
□ Add fix history tracking
```

---

## UI Mockup

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Jobs                                                    [Auto-Refresh ✓]│
├────────┬──────────┬─────────┬────────────┬─────────────────────────────┤
│ Job ID │ Pipeline │ Status  │ Progress   │ Actions                     │
├────────┼──────────┼─────────┼────────────┼─────────────────────────────┤
│ abc123 │ chipseq  │ 🔴FAILED│ 45%        │ [View Log] [🔍 Diagnose]   │
└────────┴──────────┴─────────┴────────────┴─────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 🔍 Error Diagnosis                                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ **Error Category:** 🗂️ File Not Found                                   │
│ **Confidence:** 95%                                                     │
│                                                                         │
│ ## What Happened                                                        │
│ The BWA alignment process failed because the reference genome index     │
│ files are missing. The workflow expected files at:                      │
│ `/data/references/GRCh38/bwa/genome.fa.bwt`                            │
│                                                                         │
│ ## Root Cause                                                           │
│ BWA index files were not generated or path is incorrect.               │
│                                                                         │
│ ## Suggested Fixes                                                      │
│                                                                         │
│ 1. 🟢 **Download and Index Reference** (Auto-executable)               │
│    ```bash                                                              │
│    scripts/download_reference.sh GRCh38 bwa                            │
│    ```                                                                  │
│    [▶ Apply Fix]                                                       │
│                                                                         │
│ 2. 🟡 **Check Existing Path** (Manual)                                 │
│    Verify the reference path in `nextflow.config` matches your setup.  │
│                                                                         │
│ 3. 🔴 **Create Symlink** (Risky - Confirm)                             │
│    If index exists elsewhere, create symlink:                          │
│    ```bash                                                              │
│    ln -s /actual/path/to/bwa /data/references/GRCh38/bwa               │
│    ```                                                                  │
│    [▶ Apply Fix (Confirm)]                                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Recommendation

**Implement the Hybrid System (Option 3)** because:

1. **Best of both worlds** - Fast pattern matching for common errors, LLM for complex ones
2. **Works offline** for common issues (Tier 1)
3. **Cost-effective** - Only uses LLM when needed
4. **Safe auto-fixes** - Clear risk levels prevent dangerous auto-modifications
5. **Future-proof** - Can integrate with GitHub Copilot Coding Agent for code fixes

### Priority Implementation Order

1. **Phase 1** (Days 1-3): Pattern matcher + error taxonomy
2. **Phase 2** (Days 4-6): LLM diagnosis for unknown errors
3. **Phase 3** (Days 7-9): Safe auto-fix execution
4. **Phase 4** (Days 9-10): UI integration + GitHub Copilot for code fixes

---

## Files to Create

```
src/workflow_composer/
├── diagnosis/
│   ├── __init__.py
│   ├── error_agent.py         # Main ErrorDiagnosisAgent class
│   ├── patterns.py            # Pattern database
│   ├── categories.py          # Error taxonomy
│   ├── auto_fix.py            # Fix execution engine
│   └── prompts.py             # LLM prompt templates
```

---

## Next Steps

1. Review this analysis
2. Decide on implementation scope (full hybrid vs. phased)
3. Create comprehensive implementation plan document
4. Begin Phase 1 development

Would you like me to proceed with creating the implementation plan document?
