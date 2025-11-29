"""
Diagnostics Tools
=================

Tools for error diagnosis, results analysis, and troubleshooting.
Integrates with the full ErrorDiagnosisAgent for comprehensive error handling.
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
    r"(?:my|the)\s+(?:job|workflow|pipeline)\s+(?:failed|crashed|errored)",
]


def diagnose_error_impl(
    error_text: str = None,
    log_file: str = None,
    job_id: str = None,
    work_dir: str = None,
    auto_fix: bool = False,
) -> ToolResult:
    """
    Diagnose an error using the full ErrorDiagnosisAgent.
    
    Uses a tiered approach:
    1. Pattern matching (50+ bioinformatics-specific patterns)
    2. Historical learning (boost confidence from past diagnoses)
    3. LLM analysis (for unknown/complex errors)
    4. Auto-fix execution (optional, safe fixes only)
    
    Args:
        error_text: Error text to diagnose
        log_file: Log file path to analyze
        job_id: SLURM job ID to collect logs from
        work_dir: Nextflow work directory to scan
        auto_fix: Whether to attempt automatic fixes
        
    Returns:
        ToolResult with comprehensive diagnosis
    """
    try:
        from workflow_composer.diagnosis import (
            ErrorDiagnosisAgent,
            LogCollector,
            get_diagnosis_history,
            ERROR_PATTERNS,
            ErrorCategory,
            FixRiskLevel,
        )
        from workflow_composer.diagnosis.auto_fix import AutoFixEngine
        
        # Initialize the full diagnosis agent
        agent = ErrorDiagnosisAgent(
            enable_history=True,
            pattern_confidence_threshold=0.7,
            history_boost_factor=0.1,
        )
        
        collected_error_text = error_text or ""
        
        # Collect logs from various sources
        if job_id or work_dir or log_file:
            collector = LogCollector()
            
            if work_dir:
                logs = collector.collect_from_work_dir(Path(work_dir))
                collected_error_text += "\n" + logs.get_combined_error_context()
            
            if log_file:
                log_path = Path(log_file)
                if log_path.exists():
                    with open(log_path, 'r') as f:
                        collected_error_text += "\n" + f.read()
            
            if job_id:
                # Collect SLURM logs
                slurm_logs = collector.collect_slurm_logs(job_id)
                if slurm_logs:
                    collected_error_text += "\n" + slurm_logs
        
        # Try to find recent logs if nothing provided
        if not collected_error_text.strip():
            for log_dir in [Path.cwd() / "logs", Path.cwd() / ".nextflow.log"]:
                if log_dir.exists():
                    if log_dir.is_file():
                        with open(log_dir, 'r') as f:
                            lines = f.readlines()
                            collected_error_text = "".join(lines[-200:])
                            break
                    else:
                        logs = sorted(log_dir.glob("*.log"), 
                                     key=lambda x: x.stat().st_mtime, reverse=True)
                        if logs:
                            with open(logs[0], 'r') as f:
                                lines = f.readlines()
                                collected_error_text = "".join(lines[-200:])
                                break
        
        if not collected_error_text.strip():
            return ToolResult(
                success=False,
                tool_name="diagnose_error",
                error="No error text to analyze",
                message="""❌ **No Error to Diagnose**

Please provide one of:
- Error message text directly
- Log file path: `diagnose error in /path/to/log`
- Job ID: `diagnose job 12345`
- Work directory: `diagnose error in /path/to/work`

Example: "diagnose this error: OutOfMemoryError in STAR process"
"""
            )
        
        # Run diagnosis using pattern matching on log text
        # Use diagnose_from_logs() for text-based diagnosis
        diagnosis = agent.diagnose_from_logs(collected_error_text)
        
        # Attempt auto-fix for safe fixes if requested
        fix_results = []
        if auto_fix and diagnosis and diagnosis.suggested_fixes:
            fix_engine = AutoFixEngine(dry_run=False, track_history=True)
            for fix in diagnosis.suggested_fixes:
                if fix.risk_level == FixRiskLevel.SAFE and fix.auto_executable:
                    try:
                        result = fix_engine.execute_sync(fix)
                        fix_results.append({
                            "fix": fix.description,
                            "success": result.success,
                            "message": result.message,
                        })
                    except Exception as e:
                        logger.warning(f"Auto-fix failed: {e}")
        
        # Format the response
        if diagnosis:
            # Get historical success info
            history = get_diagnosis_history()
            success_rates = history.get_fix_success_rate() if history else {}
            category_rate = success_rates.get(diagnosis.category.value, 0)
            
            # Format fixes with risk levels
            fixes_formatted = []
            for i, fix in enumerate(diagnosis.suggested_fixes[:5], 1):
                risk_emoji = {
                    FixRiskLevel.SAFE: "🟢",
                    FixRiskLevel.LOW: "🟡", 
                    FixRiskLevel.MEDIUM: "🟠",
                    FixRiskLevel.HIGH: "🔴",
                }.get(fix.risk_level, "⚪")
                
                cmd_str = f"\n   ```bash\n   {fix.command}\n   ```" if fix.command else ""
                auto_tag = " *(auto-executable)*" if fix.auto_executable else ""
                fixes_formatted.append(f"{i}. {risk_emoji} **{fix.description}**{auto_tag}{cmd_str}")
            
            fixes_section = "\n".join(fixes_formatted) if fixes_formatted else "No specific fixes suggested"
            
            # Show auto-fix results if any
            autofix_section = ""
            if fix_results:
                autofix_lines = []
                for r in fix_results:
                    status = "✅" if r["success"] else "❌"
                    autofix_lines.append(f"  {status} {r['fix']}")
                autofix_section = f"\n\n### 🔧 Auto-Fix Results\n" + "\n".join(autofix_lines)
            
            # Historical context
            history_note = ""
            if category_rate > 0:
                history_note = f"\n\n📊 *Historical success rate for this error type: {category_rate:.0%}*"
            
            message = f"""🔍 **Error Diagnosis**

### Category: {diagnosis.category.value.replace('_', ' ').title()}
**Confidence:** {diagnosis.confidence:.0%}

### Root Cause
{diagnosis.root_cause}

### Suggested Fixes
{fixes_section}{autofix_section}{history_note}

---
**Error Context Analyzed:**
```
{collected_error_text[:600]}{'...' if len(collected_error_text) > 600 else ''}
```

💡 *Say "apply fix 1" to execute a specific fix, or "auto-fix" to run all safe fixes.*
"""
            
            return ToolResult(
                success=True,
                tool_name="diagnose_error",
                data={
                    "category": diagnosis.category.value,
                    "confidence": diagnosis.confidence,
                    "root_cause": diagnosis.root_cause,
                    "fixes": [
                        {
                            "description": f.description,
                            "command": f.command,
                            "risk_level": f.risk_level.value,
                            "auto_executable": f.auto_executable,
                        }
                        for f in diagnosis.suggested_fixes
                    ],
                    "fix_results": fix_results,
                },
                message=message
            )
        else:
            # Fallback if diagnosis failed
            return _fallback_diagnosis(collected_error_text)
            
    except ImportError as e:
        logger.warning(f"Diagnosis module not available: {e}")
        return _fallback_diagnosis(error_text or "")
    except Exception as e:
        logger.error(f"Diagnosis failed: {e}")
        return _fallback_diagnosis(error_text or "", str(e))


def _fallback_diagnosis(error_text: str, error_msg: str = None) -> ToolResult:
    """Fallback to simple pattern matching if full agent unavailable."""
    # Simple patterns for fallback
    SIMPLE_PATTERNS = {
        "OutOfMemoryError": ("Out of memory", ["Increase memory allocation", "Use chunking"]),
        "No such file": ("File not found", ["Check file paths", "Verify files exist"]),
        "Permission denied": ("Permission issue", ["Check permissions", "Use chmod"]),
        "exit status 137": ("OOM killed by system", ["Increase memory limit", "Reduce input size"]),
        "SLURM": ("SLURM scheduling issue", ["Check partition", "Verify resources"]),
        "Timeout": ("Process timeout", ["Increase time limit", "Optimize process"]),
    }
    
    diagnosis = []
    for pattern, (cause, fixes) in SIMPLE_PATTERNS.items():
        if pattern.lower() in error_text.lower():
            diagnosis.append({"pattern": pattern, "cause": cause, "solutions": fixes})
    
    if not diagnosis:
        diagnosis.append({
            "pattern": "Unknown",
            "cause": "Could not identify specific error",
            "solutions": ["Review complete error message", "Check logs for context"]
        })
    
    sections = []
    for i, d in enumerate(diagnosis, 1):
        solutions = "\n".join(f"   {j}. {s}" for j, s in enumerate(d['solutions'], 1))
        sections.append(f"**Issue {i}: {d['pattern']}**\n- Cause: {d['cause']}\n- Solutions:\n{solutions}")
    
    note = f"\n\n⚠️ *Using simplified diagnosis. Error: {error_msg}*" if error_msg else ""
    
    return ToolResult(
        success=True,
        tool_name="diagnose_error",
        data={"diagnosis": diagnosis, "fallback": True},
        message=f"""🔍 **Error Diagnosis** (Basic Mode)

{chr(10).join(sections)}{note}

**Error Text:**
```
{error_text[:400]}{'...' if len(error_text) > 400 else ''}
```
"""
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
    job_id: str = None,
) -> ToolResult:
    """
    Analyze workflow results using the full ResultCollector and ResultViewer.
    
    Provides:
    - Smart file discovery with pipeline-specific patterns
    - File categorization (QC reports, alignments, variants, etc.)
    - Rich previews for tables, HTML reports, and images
    - Summary statistics for count matrices
    
    Args:
        results_path: Path to results directory or file
        result_type: Type of analysis (rna_seq, chip_seq, variant, etc.)
        job_id: Job identifier for labeling
        
    Returns:
        ToolResult with comprehensive analysis
    """
    try:
        from workflow_composer.results import ResultCollector, ResultViewer
        from workflow_composer.results.result_types import FileType
        
        # Find results if not specified
        if not results_path:
            for path in [
                Path.cwd() / "data" / "results",
                Path.cwd() / "results",
                Path.cwd() / "output",
                Path.cwd() / "generated_workflows",
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
2. Specify the results path: `analyze results in /path/to/results`
3. Make sure your workflow completed successfully
"""
            )
        
        results_dir = Path(results_path)
        
        # Use the smart ResultCollector
        collector = ResultCollector(pipeline_type=result_type)
        summary = collector.scan(results_dir, job_id=job_id)
        
        if summary.total_files == 0:
            # Fallback to basic file listing
            contents = list(results_dir.iterdir())[:30]
            return ToolResult(
                success=True,
                tool_name="analyze_results",
                data={"path": str(results_path)},
                message=f"""📊 **Results Directory: {results_path}**

No recognized bioinformatics result files found.

**Directory Contents ({len(contents)} items):**
```
{chr(10).join(f.name for f in contents)}
```

Try specifying the pipeline type: `analyze results as rna_seq`
"""
            )
        
        # Initialize viewer for preview capability
        viewer = ResultViewer()
        
        # Get category breakdown
        category_summary = summary.get_category_summary()
        
        # Build interpretations section
        interpretations = []
        
        # QC Reports
        if summary.qc_reports:
            qc_list = "\n".join([
                f"  - [{r.name}]({r.path})" 
                for r in summary.qc_reports[:5]
            ])
            has_multiqc = "✅ MultiQC available" if summary.has_multiqc else ""
            interpretations.append(f"""
### 📈 Quality Control Reports ({len(summary.qc_reports)} files)
{has_multiqc}
{qc_list}

**Key metrics to check:**
- Per-base sequence quality
- Adapter content
- Duplication levels
- GC content distribution
""")
        
        # Alignment files
        alignments = [f for f in summary.all_files if f.file_type in [FileType.ALIGNMENT, FileType.INDEX]]
        if alignments:
            interpretations.append(f"""
### 🎯 Alignment Results ({len(alignments)} files)
Found BAM/BAI files.

**Quality indicators:**
- Mapping rate >80% (DNA-seq) / >70% (RNA-seq)
- Duplicate rate <30%
- Proper pair percentage >90%
""")
        
        # Variant files
        variants = [f for f in summary.all_files if f.file_type == FileType.VARIANT]
        if variants:
            interpretations.append(f"""
### 🧬 Variant Calls ({len(variants)} files)
Found VCF files.

**Quality checks:**
- QUAL score distribution
- FILTER status (% PASS)
- Read depth (DP) coverage
""")
        
        # Expression/count files
        tables = [f for f in summary.all_files if f.file_type == FileType.TABLE]
        count_files = [f for f in tables if 'count' in f.name.lower() or 'fpkm' in f.name.lower()]
        if count_files:
            interpretations.append(f"""
### 📉 Expression Quantification ({len(count_files)} files)

**Analysis recommendations:**
- Check for samples with very low total counts
- Assess count distribution per sample
- Apply normalization (TPM, FPKM, CPM)
""")
        
        # Peaks (ChIP-seq, ATAC-seq)
        peaks = [f for f in summary.all_files 
                 if 'peak' in f.name.lower() or f.name.endswith('.bed')]
        if peaks:
            interpretations.append(f"""
### ⛰️ Peak Files ({len(peaks)} files)

**Quality metrics:**
- FRiP score (Fraction of Reads in Peaks)
- Peak count consistency across replicates
- Peak width distribution
""")
        
        # Build the final message
        category_lines = "\n".join([
            f"  - **{cat.replace('_', ' ').title()}**: {count} files"
            for cat, count in category_summary.items() if count > 0
        ])
        
        interpretations_text = "\n".join(interpretations) if interpretations else ""
        
        message = f"""📊 **Results Analysis: {results_path}**

### Summary
| Metric | Value |
|--------|-------|
| **Total Files** | {summary.total_files} |
| **Total Size** | {summary.size_human} |
| **Job ID** | {summary.job_id or 'Unknown'} |

### Files by Category
{category_lines}
{interpretations_text}

---
### 💡 Next Steps
1. {"Open MultiQC report for overview" if summary.has_multiqc else "Generate MultiQC report"}
2. Check for failed samples in logs
3. Compare metrics across samples
4. Ask me to preview specific files

*Say `show <filename>` to view a specific result file.*
"""
        
        return ToolResult(
            success=True,
            tool_name="analyze_results",
            data={
                "path": str(results_path),
                "total_files": summary.total_files,
                "total_size": summary.total_size,
                "categories": category_summary,
                "has_multiqc": summary.has_multiqc,
                "qc_reports": [str(r.path) for r in summary.qc_reports[:10]],
            },
            message=message
        )
        
    except ImportError as e:
        logger.warning(f"Results module not fully available: {e}")
        return _fallback_analyze_results(results_path, result_type)
    except Exception as e:
        logger.error(f"Result analysis failed: {e}")
        return _fallback_analyze_results(results_path, result_type, str(e))


def _fallback_analyze_results(results_path: str, result_type: str, error_msg: str = None) -> ToolResult:
    """Fallback to simple file scanning."""
    if not results_path or not Path(results_path).exists():
        return ToolResult(
            success=False,
            tool_name="analyze_results",
            error="No results found",
            message="❌ No results directory found to analyze."
        )
    
    results_dir = Path(results_path)
    
    # Simple pattern matching
    result_files = {
        "fastqc": list(results_dir.rglob("*fastqc*.html")),
        "multiqc": list(results_dir.rglob("*multiqc*.html")),
        "alignment": list(results_dir.rglob("*.bam")),
        "variants": list(results_dir.rglob("*.vcf*")),
        "counts": list(results_dir.rglob("*count*.txt")),
    }
    
    found_types = {k: len(v) for k, v in result_files.items() if v}
    
    note = f"\n\n⚠️ *Using basic scanner. Error: {error_msg}*" if error_msg else ""
    
    return ToolResult(
        success=True,
        tool_name="analyze_results",
        data={"path": str(results_path), "files_found": found_types, "fallback": True},
        message=f"""📊 **Results Analysis (Basic Mode)**: {results_path}

**Files Found:**
{chr(10).join(f'- {k.title()}: {v} files' for k, v in found_types.items()) or 'No standard files found'}
{note}
"""
    )


# =============================================================================
# RECOVER_ERROR
# =============================================================================

RECOVER_ERROR_PATTERNS = [
    r"(?:recover|fix|repair|resolve)\s+(?:the\s+)?(?:error|failure|issue|problem)",
    r"(?:auto\s*-?\s*fix|apply\s+fix)",
    r"run\s+(?:the\s+)?recovery",
    r"execute\s+(?:the\s+)?recovery",
    r"(?:start|trigger)\s+(?:auto\s*)?recovery",
]


def recover_error_impl(
    action: str = None,
    job_id: str = None,
    error_log: str = None,
    confirm: bool = False,
) -> ToolResult:
    """
    Execute recovery actions for diagnosed errors.
    
    Wraps RecoveryManager to execute recovery actions:
    - restart_server: Restart vLLM server
    - resubmit_job: Resubmit a failed SLURM job
    - clear_cache: Clear disk caches
    - install_module: Install missing Python modules
    - scale_resources: Get suggestions for resource scaling
    
    Args:
        action: Recovery action to execute (restart_server, resubmit_job, clear_cache, etc.)
        job_id: SLURM job ID (for resubmit_job)
        error_log: Error log content (for diagnosis)
        confirm: Whether to execute without confirmation
        
    Returns:
        ToolResult with recovery status
    """
    import asyncio
    
    try:
        from workflow_composer.agents.autonomous.recovery import (
            RecoveryManager,
            RecoveryAction,
            RecoveryResult,
        )
        
        recovery = RecoveryManager(
            require_confirmation=not confirm,
            ai_diagnosis_enabled=False,  # We already have diagnosis
        )
        
        # Map action string to RecoveryAction
        action_map = {
            "restart_server": RecoveryAction.RESTART_SERVER,
            "restart_vllm": RecoveryAction.RESTART_SERVER,
            "resubmit_job": RecoveryAction.RESUBMIT_JOB,
            "resubmit": RecoveryAction.RESUBMIT_JOB,
            "clear_cache": RecoveryAction.CLEAR_CACHE,
            "install_module": RecoveryAction.APPLY_PATCH,
            "scale_resources": RecoveryAction.SCALE_RESOURCES,
        }
        
        # If no specific action, try to diagnose and recover
        if not action and (job_id or error_log):
            # Run async recovery
            async def do_recovery():
                return await recovery.handle_job_failure(
                    job_id=job_id or "unknown",
                    error_log=error_log,
                )
            
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(asyncio.run, do_recovery())
                        result = future.result(timeout=120)
                else:
                    result = loop.run_until_complete(do_recovery())
            except RuntimeError:
                result = asyncio.run(do_recovery())
            
            if result.success:
                message = f"""✅ **Recovery Successful**

**Action:** {result.action.value.replace('_', ' ').title()}
**Message:** {result.message}
"""
                if result.new_job_id:
                    message += f"\n**New Job ID:** {result.new_job_id}"
                if result.details:
                    message += f"\n**Details:** {result.details}"
            else:
                message = f"""❌ **Recovery Failed**

**Action:** {result.action.value.replace('_', ' ').title()}
**Message:** {result.message}
**Error:** {result.error or 'Unknown error'}

💡 Try specifying a recovery action:
- `recover restart_server` - Restart vLLM
- `recover resubmit_job job_id=12345` - Resubmit SLURM job
- `recover clear_cache` - Clear disk caches
"""
            
            return ToolResult(
                success=result.success,
                tool_name="recover_error",
                data=result.to_dict(),
                message=message
            )
        
        # Execute specific action
        if action:
            action_lower = action.lower().replace("-", "_").replace(" ", "_")
            recovery_action = action_map.get(action_lower)
            
            if not recovery_action:
                return ToolResult(
                    success=False,
                    tool_name="recover_error",
                    error=f"Unknown action: {action}",
                    message=f"""❌ **Unknown Recovery Action:** {action}

**Available Actions:**
- `restart_server` / `restart_vllm` - Restart the vLLM server
- `resubmit_job` - Resubmit a failed SLURM job (requires job_id)
- `clear_cache` - Clear disk caches to free space
- `install_module` - Install missing Python modules
- `scale_resources` - Get suggestions for resource scaling
"""
                )
            
            # Execute the recovery action
            async def execute_action():
                if recovery_action == RecoveryAction.RESTART_SERVER:
                    return await recovery._restart_vllm(error_log)
                elif recovery_action == RecoveryAction.RESUBMIT_JOB:
                    if not job_id:
                        return RecoveryResult(
                            success=False,
                            action=recovery_action,
                            message="job_id is required for resubmit_job",
                        )
                    from workflow_composer.agents.autonomous.job_monitor import JobMonitor
                    monitor = JobMonitor()
                    job_info = monitor.get_job_info(job_id)
                    return await recovery._resubmit_job(job_info)
                elif recovery_action == RecoveryAction.CLEAR_CACHE:
                    return await recovery._clear_cache()
                else:
                    return RecoveryResult(
                        success=False,
                        action=recovery_action,
                        message=f"Action {recovery_action.value} requires manual intervention",
                    )
            
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(asyncio.run, execute_action())
                        result = future.result(timeout=120)
                else:
                    result = loop.run_until_complete(execute_action())
            except RuntimeError:
                result = asyncio.run(execute_action())
            
            if result.success:
                message = f"""✅ **Recovery Action Completed**

**Action:** {result.action.value.replace('_', ' ').title()}
**Status:** Success
**Message:** {result.message}
"""
                if result.new_job_id:
                    message += f"\n**New Job ID:** {result.new_job_id}"
            else:
                message = f"""❌ **Recovery Action Failed**

**Action:** {result.action.value.replace('_', ' ').title()}
**Message:** {result.message}
**Error:** {result.error or 'See details'}
"""
            
            return ToolResult(
                success=result.success,
                tool_name="recover_error",
                data=result.to_dict(),
                message=message
            )
        
        # No action specified
        return ToolResult(
            success=False,
            tool_name="recover_error",
            error="No action specified",
            message="""❓ **What would you like to recover?**

**Usage:**
- `recover error` + provide error log or job_id
- `recover restart_server` - Restart vLLM
- `recover resubmit_job job_id=12345` - Resubmit failed job
- `recover clear_cache` - Free disk space

💡 First run `diagnose error` to identify the issue, then use the suggested recovery action.
"""
        )
        
    except ImportError as e:
        logger.warning(f"Recovery module not available: {e}")
        return ToolResult(
            success=False,
            tool_name="recover_error",
            error=f"Recovery module not available: {e}",
            message=f"""❌ **Recovery Module Not Available**

The autonomous recovery system couldn't be loaded.
Error: {e}

**Manual Recovery Options:**
- Restart vLLM: `pkill -f vllm && ./scripts/start_server.sh`
- Resubmit job: `sbatch <script.sh>`
- Clear cache: `rm -rf ~/.cache/huggingface/hub/*`
"""
        )
    except Exception as e:
        logger.error(f"Recovery failed: {e}")
        return ToolResult(
            success=False,
            tool_name="recover_error",
            error=str(e),
            message=f"""❌ **Recovery Failed**

Error: {e}

Please try manual recovery or provide more details.
"""
        )
