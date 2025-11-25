"""
LLM prompt templates for error diagnosis.

Contains structured prompts for various LLM providers to analyze
pipeline failures and suggest fixes.
"""

# Main diagnosis prompt
DIAGNOSIS_PROMPT = """Analyze this bioinformatics pipeline failure and provide a diagnosis.

## Error Context

**Workflow:** {workflow_name}
**Analysis Type:** {analysis_type}
**Failed Process:** {failed_process}

## Log Files

### Nextflow Log (last section):
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

## Your Task

1. **Identify the root cause** of this failure
2. **Classify the error** into one of: {error_categories}
3. **Explain** what went wrong in simple terms (for non-experts)
4. **Suggest fixes** ranked by likelihood of success
5. **Provide shell commands** to fix if applicable

## Required Output Format (JSON)

```json
{{
  "error_category": "category_name",
  "root_cause": "Brief technical explanation",
  "user_explanation": "Simple explanation for non-technical users",
  "confidence": 0.85,
  "log_excerpt": "Key error line from logs",
  "suggested_fixes": [
    {{
      "description": "What to do",
      "command": "shell command if applicable (or null)",
      "auto_executable": true,
      "risk_level": "safe"
    }},
    {{
      "description": "Alternative fix",
      "command": null,
      "auto_executable": false,
      "risk_level": "medium"
    }}
  ]
}}
```

Only output the JSON, no additional text."""


# Simplified prompt for faster LLMs / free tiers
DIAGNOSIS_PROMPT_SIMPLE = """Analyze this error from a bioinformatics pipeline:

```
{error_log}
```

Failed process: {failed_process}

Respond with JSON only:
{{
  "error_category": "one of: file_not_found, out_of_memory, container_error, permission_denied, dependency_missing, tool_error, network_error, slurm_error, data_format_error, unknown",
  "root_cause": "brief explanation",
  "user_explanation": "simple explanation",
  "confidence": 0.0-1.0,
  "suggested_fixes": [
    {{"description": "fix description", "command": "command or null", "risk_level": "safe/low/medium/high"}}
  ]
}}"""


# GitHub Copilot code fix prompt
GITHUB_COPILOT_FIX_PROMPT = """## Bug Fix Request

A bioinformatics Nextflow workflow failed with the following error:

**Error Category:** {category}
**Root Cause:** {root_cause}

### Error Log
```
{log_excerpt}
```

### Workflow File: {workflow_file}
```nextflow
{workflow_content}
```

### Task

1. Identify the bug in the workflow code
2. Fix the issue
3. Add appropriate error handling
4. Ensure the fix follows Nextflow DSL2 best practices

Please create a pull request with the fix.
"""


# Gemini-specific prompt (optimized for free tier)
GEMINI_DIAGNOSIS_PROMPT = """You are a bioinformatics expert. Analyze this pipeline error.

Error log:
{error_log}

Workflow: {workflow_name}
Failed process: {failed_process}

Provide diagnosis as JSON:
{{"error_category":"...", "root_cause":"...", "confidence":0.8, "fix":"..."}}"""


# System prompts for different contexts
SYSTEM_PROMPT_DIAGNOSIS = """You are an expert bioinformatics pipeline debugger.
You specialize in Nextflow, Snakemake, and HPC (SLURM) environments.
Analyze errors precisely and suggest actionable fixes.
Always respond with valid JSON when requested."""

SYSTEM_PROMPT_FIX = """You are a bioinformatics code repair agent.
You fix Nextflow DSL2 workflows and shell scripts.
Make minimal, targeted fixes. Explain your changes clearly."""


def build_diagnosis_prompt(
    logs,
    workflow_name: str = "Unknown",
    analysis_type: str = "Unknown",
    simple: bool = False,
    error_categories: str = ""
) -> str:
    """
    Build a diagnosis prompt from collected logs.
    
    Args:
        logs: CollectedLogs object
        workflow_name: Name of the workflow
        analysis_type: Type of analysis (rnaseq, chipseq, etc.)
        simple: Use simplified prompt for faster/cheaper LLMs
        error_categories: Comma-separated list of valid categories
        
    Returns:
        Formatted prompt string
    """
    if simple:
        return DIAGNOSIS_PROMPT_SIMPLE.format(
            error_log=logs.get_combined_error_context(50),
            failed_process=logs.failed_process or "Unknown",
        )
    
    return DIAGNOSIS_PROMPT.format(
        workflow_name=workflow_name,
        analysis_type=analysis_type,
        failed_process=logs.failed_process or "Unknown",
        nextflow_log=(logs.nextflow_log[-3000:] if logs.nextflow_log else "N/A"),
        slurm_err=(logs.slurm_err[-2000:] if logs.slurm_err else "N/A"),
        workflow_config=(logs.config_file[:2000] if logs.config_file else "N/A"),
        error_categories=error_categories,
    )


def build_code_fix_prompt(
    diagnosis,
    workflow_content: str,
    workflow_file: str = "main.nf"
) -> str:
    """
    Build a prompt for code fix suggestions.
    
    Args:
        diagnosis: ErrorDiagnosis object
        workflow_content: Content of the workflow file
        workflow_file: Name of the workflow file
        
    Returns:
        Formatted prompt string
    """
    return GITHUB_COPILOT_FIX_PROMPT.format(
        category=diagnosis.category.value,
        root_cause=diagnosis.root_cause,
        log_excerpt=diagnosis.log_excerpt[:500],
        workflow_file=workflow_file,
        workflow_content=workflow_content[:5000],
    )
