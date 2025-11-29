"""
Diagnostics Tools
=================

Tools for error diagnosis, results analysis, and troubleshooting.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import ToolResult

logger = logging.getLogger(__name__)


# =============================================================================
# DIAGNOSE_ERROR
# =============================================================================

DIAGNOSE_ERROR_PATTERNS = [
    r"(?:diagnose|debug|troubleshoot|fix|why|what)\s+(?:did|went|is|this|the)?\s*(?:error|wrong|fail|crash|problem)",
    r"(?:help|explain)\s+(?:with\s+)?(?:this\s+)?error",
]


def diagnose_error_impl(
    error_text: str = None,
    log_file: str = None,
) -> ToolResult:
    """
    Diagnose an error from log or provided text.
    
    Args:
        error_text: Error text to diagnose
        log_file: Log file path to analyze
        
    Returns:
        ToolResult with diagnosis
    """
    # Common error patterns and solutions
    ERROR_PATTERNS = {
        "OutOfMemoryError": {
            "cause": "Process ran out of memory",
            "solutions": [
                "Increase memory allocation in workflow config",
                "Use --mem flag with higher value",
                "Try chunking input data into smaller pieces",
                "Add retry strategy with increased resources"
            ]
        },
        "No such file or directory": {
            "cause": "Required file or path not found",
            "solutions": [
                "Check input file paths are correct",
                "Ensure files exist before running workflow",
                "Check for typos in file names",
                "Verify reference data is downloaded"
            ]
        },
        "Permission denied": {
            "cause": "Insufficient permissions to access file or directory",
            "solutions": [
                "Check file/directory permissions: ls -la",
                "Run 'chmod' to fix permissions",
                "Check if output directory is writable",
                "Try running with different user privileges"
            ]
        },
        "Connection refused": {
            "cause": "Network connection failed",
            "solutions": [
                "Check network connectivity",
                "Verify server/database is running",
                "Check firewall settings",
                "Try again after brief wait"
            ]
        },
        "SLURM": {
            "cause": "SLURM job scheduling issue",
            "solutions": [
                "Check partition availability: sinfo",
                "Verify resource requirements are valid",
                "Check QOS limits: sacctmgr show qos",
                "Review SLURM configuration"
            ]
        },
        "Timeout": {
            "cause": "Process exceeded time limit",
            "solutions": [
                "Increase time limit in workflow",
                "Optimize process or use more cores",
                "Check if process is stuck",
                "Consider chunking input data"
            ]
        },
        "Invalid": {
            "cause": "Invalid input or parameter",
            "solutions": [
                "Check input format matches expected",
                "Validate input files are not corrupted",
                "Review parameter documentation",
                "Try with example/test data first"
            ]
        }
    }
    
    # Get error text
    if not error_text and log_file:
        log_path = Path(log_file)
        if log_path.exists():
            with open(log_path, 'r') as f:
                error_text = f.read()
    
    if not error_text:
        # Try to find recent log
        log_dir = Path.cwd() / "logs"
        if log_dir.exists():
            logs = sorted(log_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
            if logs:
                with open(logs[0], 'r') as f:
                    lines = f.readlines()
                    # Get last 100 lines
                    error_text = "".join(lines[-100:])
    
    if not error_text:
        return ToolResult(
            success=False,
            tool_name="diagnose_error",
            error="No error text to analyze",
            message="""❌ **No Error to Diagnose**

Please provide:
- Error message text
- Log file path
- Or paste the error directly

Example: "diagnose this error: OutOfMemoryError in process X"
"""
        )
    
    # Find matching patterns
    diagnosis = []
    matched = False
    
    for pattern, info in ERROR_PATTERNS.items():
        if pattern.lower() in error_text.lower():
            matched = True
            diagnosis.append({
                "pattern": pattern,
                "cause": info["cause"],
                "solutions": info["solutions"]
            })
    
    if not matched:
        diagnosis.append({
            "pattern": "Unknown",
            "cause": "Could not identify specific error pattern",
            "solutions": [
                "Review the complete error message",
                "Check workflow logs for context",
                "Search online for the error message",
                "Contact support with full error details"
            ]
        })
    
    # Format response
    sections = []
    for i, diag in enumerate(diagnosis, 1):
        solutions_list = "\n".join(f"   {j}. {s}" for j, s in enumerate(diag['solutions'], 1))
        sections.append(f"""
**Issue {i}: {diag['pattern']}**
- **Cause**: {diag['cause']}
- **Solutions**:
{solutions_list}
""")
    
    message = f"""🔍 **Error Diagnosis**

{chr(10).join(sections)}

**Error Text Analyzed:**
```
{error_text[:500]}{'...' if len(error_text) > 500 else ''}
```

Need more help? Ask me to explain any solution in detail.
"""
    
    return ToolResult(
        success=True,
        tool_name="diagnose_error",
        data={"diagnosis": diagnosis},
        message=message
    )


# =============================================================================
# ANALYZE_RESULTS
# =============================================================================

ANALYZE_RESULTS_PATTERNS = [
    r"(?:analyze|interpret|explain|summarize|what do)\s+(?:the\s+)?(?:these\s+)?results?\s*(?:show|mean)?",
    r"(?:look at|review|check)\s+(?:the\s+)?(?:output|results|analysis)",
    r"(?:what\s+do|can\s+you)\s+(?:these|the)\s+(?:results|output|data)\s+(?:mean|show|tell)",
]


def analyze_results_impl(
    results_path: str = None,
    result_type: str = None,
) -> ToolResult:
    """
    Analyze workflow results and provide interpretation.
    
    Args:
        results_path: Path to results directory or file
        result_type: Type of analysis (qc, alignment, variant, expression)
        
    Returns:
        ToolResult with analysis interpretation
    """
    # Find results if not specified
    if not results_path:
        # Check common locations
        for path in [
            Path.cwd() / "data" / "results",
            Path.cwd() / "results",
            Path.cwd() / "output",
        ]:
            if path.exists() and any(path.iterdir()):
                results_path = str(path)
                break
    
    if not results_path or not Path(results_path).exists():
        return ToolResult(
            success=False,
            tool_name="analyze_results",
            error="No results found",
            message="""❌ **No Results Found**

I couldn't find results to analyze. Please:
1. Run a workflow first
2. Specify the results path: "analyze results in /path/to/results"
3. Make sure your workflow completed successfully
"""
        )
    
    results_dir = Path(results_path)
    
    # Scan for common result files
    result_files = {
        "fastqc": list(results_dir.rglob("*fastqc*.html")) + list(results_dir.rglob("*fastqc*.zip")),
        "multiqc": list(results_dir.rglob("*multiqc*.html")),
        "alignment": list(results_dir.rglob("*.bam")) + list(results_dir.rglob("*.bam.bai")),
        "variants": list(results_dir.rglob("*.vcf")) + list(results_dir.rglob("*.vcf.gz")),
        "expression": list(results_dir.rglob("*counts*.txt")) + list(results_dir.rglob("*fpkm*.txt")),
        "peaks": list(results_dir.rglob("*.narrowPeak")) + list(results_dir.rglob("*.broadPeak")),
        "logs": list(results_dir.rglob("*.log")),
    }
    
    # Build summary
    found_types = {k: len(v) for k, v in result_files.items() if v}
    
    if not found_types:
        return ToolResult(
            success=True,
            tool_name="analyze_results",
            data={"path": results_path},
            message=f"""📊 **Results Directory: {results_path}**

No standard bioinformatics result files found. 
The directory may contain:
- Custom output formats
- Intermediate files
- Non-standard file extensions

**Contents:**
```
{chr(10).join(f.name for f in list(results_dir.iterdir())[:20])}
```

Specify the file type for detailed analysis.
"""
        )
    
    # Generate interpretation based on what's found
    interpretations = []
    
    if "fastqc" in found_types:
        interpretations.append(f"""
### 📈 Quality Control (FastQC)
**{found_types['fastqc']} FastQC reports found**

Key things to check:
- Per base sequence quality (green = good)
- Sequence duplication levels (high = potential PCR bias)
- Adapter content (should be low)
- GC content (should match expected for organism)
""")
    
    if "multiqc" in found_types:
        interpretations.append(f"""
### 📊 MultiQC Summary
**{found_types['multiqc']} MultiQC reports available**

Open the HTML report for an overview of all samples.
Look for outliers in the heatmap/plots.
""")
    
    if "alignment" in found_types:
        interpretations.append(f"""
### 🎯 Alignment Results
**{found_types['alignment']} BAM files found**

Good alignment typically shows:
- >80% mapping rate for DNA-seq
- >70% mapping rate for RNA-seq
- Low duplicate rate (<30%)
""")
    
    if "variants" in found_types:
        interpretations.append(f"""
### 🧬 Variant Calls
**{found_types['variants']} VCF files found**

Key quality metrics:
- QUAL score (higher = more confident)
- Filter status (PASS = high quality)
- Read depth (DP, higher = more reliable)
""")
    
    if "expression" in found_types:
        interpretations.append(f"""
### 📉 Expression Quantification
**{found_types['expression']} count files found**

Analysis suggestions:
- Check for samples with very low total counts
- Look at count distribution per sample
- Consider normalization (TPM, FPKM, CPM)
""")
    
    if "peaks" in found_types:
        interpretations.append(f"""
### ⛰️ Peak Calls
**{found_types['peaks']} peak files found**

Quality indicators:
- FRiP score (Fraction of Reads in Peaks)
- Peak number consistency across replicates
- Peak width distribution
""")
    
    message = f"""📊 **Results Analysis: {results_path}**

**Summary of Findings:**
{chr(10).join(f'- {k.title()}: {v} files' for k, v in found_types.items())}

{''.join(interpretations)}

---
**Next Steps:**
1. Review MultiQC report if available
2. Check for failed samples in logs
3. Compare metrics across samples
4. Ask me about specific files for detailed analysis
"""
    
    return ToolResult(
        success=True,
        tool_name="analyze_results",
        data={"path": results_path, "files_found": found_types},
        message=message
    )
