# Capability Integration Plan

**Document Version:** 1.1  
**Created:** November 29, 2025  
**Updated:** November 29, 2025  
**Status:** ✅ PHASE 1 COMPLETE

---

## Progress Summary

| Phase | Status | Tools Added | Lines Integrated |
|-------|--------|-------------|------------------|
| Phase 1: Core Integration | ✅ COMPLETE | 3 new tools | ~4,000 |
| Phase 2: Autonomous Agent | 🔲 Planned | TBD | ~3,000 |
| Phase 3: Testing & Validation | 🔲 Planned | - | - |

### Phase 1 Completion Details

**Tools Enhanced (now using full capabilities):**
- `diagnose_error` → ErrorDiagnosisAgent (50+ patterns, LLM fallback, auto-fix)
- `analyze_results` → ResultCollector + ResultViewer (smart discovery, rich previews)
- `check_references` → ReferenceManager (comprehensive status, download URLs)
- `get_job_status` → WorkflowMonitor (Nextflow trace parsing, progress bars)

**New Tools Added (23 total, up from 20):**
- `download_reference` → Download genomes/GTF from Ensembl
- `build_index` → Build STAR/Salmon/BWA/HISAT2/Kallisto indexes
- `visualize_workflow` → Generate DAG diagrams with graphviz

---

## Executive Summary

After a comprehensive code audit, we identified **~8,000+ lines** of sophisticated capability code that has been developed but NOT integrated into the unified agent system. This document provides a detailed plan for integrating these capabilities.

### Key Findings

| Category | Unused Modules | LOC | Priority | Status |
|----------|---------------|-----|----------|--------|
| Error Diagnosis | 6 modules | 2,850 | HIGH | ✅ INTEGRATED |
| Reference Management | 2 modules | 710 | HIGH | ✅ INTEGRATED |
| Results Viewer | 7 modules | 2,880 | MEDIUM | ✅ INTEGRATED |
| Workflow Monitoring | 1 module | 415 | MEDIUM | ✅ INTEGRATED |
| Workflow Visualization | 1 module | 460 | LOW | ✅ INTEGRATED |
| **TOTAL** | **17 modules** | **~7,300** | - | ✅ |

> **Note:** The autonomous agent system (~3,000 lines) is deferred as it requires architectural changes.

### Current vs. Target Architecture

```
ORIGINAL STATE (20 tools):
┌─────────────────────────────────────────────────────┐
│              UnifiedChatHandler                      │
│                     ↓                                │
│              AgentTools (20 tools)                   │
│   diagnose_error_impl() → Simple dict patterns      │
│   analyze_results_impl() → Basic file listing       │
│   check_references_impl() → Path scanning only      │
└─────────────────────────────────────────────────────┘

CURRENT STATE (23 tools) ✅ ACHIEVED:
┌─────────────────────────────────────────────────────┐
│              UnifiedChatHandler                      │
│                     ↓                                │
│              AgentTools (23 tools)                   │
│   diagnose_error_impl() → ErrorDiagnosisAgent       │
│     ├── Pattern Matching (702 lines, 50+ patterns)  │
│     ├── Historical Learning (333 lines)             │
│     ├── LLM Analysis (tiered providers)             │
│     └── AutoFixEngine (501 lines)                   │
│   analyze_results_impl() → ResultCollector+Viewer   │
│   check_references_impl() → ReferenceManager        │
│   get_job_status_impl() → WorkflowMonitor           │
│   download_reference → ReferenceManager             │
│   build_index → ReferenceManager                    │
│   visualize_workflow → WorkflowVisualizer           │
└─────────────────────────────────────────────────────┘
```

---

## Detailed Capability Analysis

### 1. Error Diagnosis System ⭐ HIGH PRIORITY

#### Current Implementation (diagnostics.py)
- **Location:** `agents/tools/diagnostics.py:diagnose_error_impl()`
- **Lines:** ~150
- **Approach:** Simple dictionary-based pattern matching (7 patterns)
- **Limitations:**
  - No LLM fallback for unknown errors
  - No historical learning
  - No log collection from Nextflow/SLURM
  - No auto-fix capability
  - Static solutions, no context-awareness

#### Available Capability (NOT USED)

| Module | Location | Lines | Purpose | Status |
|--------|----------|-------|---------|--------|
| `ErrorDiagnosisAgent` | `diagnosis/agent.py` | 809 | Main diagnosis orchestrator | ❌ Not imported |
| `LogCollector` | `diagnosis/log_collector.py` | 375 | Collect logs from Nextflow/SLURM/work dirs | ❌ Not imported |
| `ERROR_PATTERNS` | `diagnosis/patterns.py` | 702 | 50+ regex patterns with fix suggestions | ❌ Not imported |
| `DiagnosisHistory` | `diagnosis/history.py` | 333 | Track fix success, learn from past | ❌ Not imported |
| `AutoFixEngine` | `diagnosis/auto_fix.py` | 501 | Execute safe fixes with rollback | ❌ Not imported |
| Error categories | `diagnosis/categories.py` | ~200 | Rich categorization (OOM, permission, etc.) | ❌ Not imported |

#### Integration Plan

```python
# File: agents/tools/diagnostics.py

# CURRENT (REPLACE):
def diagnose_error_impl(error_text=None, log_file=None) -> ToolResult:
    ERROR_PATTERNS = {"OutOfMemoryError": {...}}  # 7 simple patterns
    ...

# NEW (INTEGRATE):
def diagnose_error_impl(
    error_text: str = None,
    log_file: str = None,
    job_id: str = None,
    auto_fix: bool = False,
) -> ToolResult:
    """
    Diagnose an error using the full ErrorDiagnosisAgent.
    """
    from workflow_composer.diagnosis import (
        ErrorDiagnosisAgent,
        LogCollector,
        get_diagnosis_history,
    )
    from workflow_composer.diagnosis.auto_fix import AutoFixEngine
    
    agent = ErrorDiagnosisAgent(
        enable_history=True,
        pattern_confidence_threshold=0.75,
    )
    
    # Collect logs if job_id provided
    if job_id:
        collector = LogCollector()
        logs = collector.collect_for_job(job_id)
        error_text = logs.get_combined_error_context()
    
    # Run diagnosis
    diagnosis = agent.diagnose_sync(error_text)
    
    # Optionally apply safe fixes
    if auto_fix and diagnosis.suggested_fixes:
        fix_engine = AutoFixEngine(dry_run=False)
        for fix in diagnosis.suggested_fixes:
            if fix.risk_level == FixRiskLevel.SAFE:
                result = fix_engine.execute_sync(fix)
                # Record to history
    
    return ToolResult(
        success=True,
        tool_name="diagnose_error",
        data={
            "category": diagnosis.category.value,
            "confidence": diagnosis.confidence,
            "root_cause": diagnosis.root_cause,
            "fixes": [f.to_dict() for f in diagnosis.suggested_fixes],
        },
        message=format_diagnosis(diagnosis)
    )
```

#### Migration Steps

1. **Phase 1: Pattern Integration** (Effort: LOW)
   - Import `ERROR_PATTERNS` from `diagnosis/patterns.py`
   - Replace simple dict with comprehensive patterns
   - Test with existing error samples

2. **Phase 2: Log Collection** (Effort: MEDIUM)
   - Add `job_id` parameter to `diagnose_error_impl`
   - Use `LogCollector` to gather Nextflow/.command.err/SLURM logs
   - Format combined context for LLM

3. **Phase 3: LLM Fallback** (Effort: MEDIUM)
   - Integrate `ErrorDiagnosisAgent` for unknown patterns
   - Use tiered providers (Lightning → Gemini → local)
   - Fall back gracefully if all LLMs unavailable

4. **Phase 4: Historical Learning** (Effort: LOW)
   - Enable `DiagnosisHistory` tracking
   - Boost confidence based on past success
   - Show "This fix worked X times before" in UI

5. **Phase 5: Auto-Fix** (Effort: HIGH)
   - Add new tool `apply_fix` or parameter `auto_fix=True`
   - Use `AutoFixEngine` for safe fixes (mkdir, retry)
   - Require confirmation for risky fixes

#### Testing Requirements
- [ ] Test with OOM error samples
- [ ] Test with file not found errors
- [ ] Test with SLURM timeout errors
- [ ] Test log collection from work directory
- [ ] Test historical learning after 5+ diagnoses
- [ ] Test auto-fix for safe operations

---

### 2. Reference Management ⭐ HIGH PRIORITY

#### Current Implementation (data_management.py)
- **Location:** `agents/tools/data_management.py:check_references_impl()`
- **Lines:** ~50
- **Approach:** Simple path scanning for reference files
- **Limitations:**
  - Cannot download missing references
  - Cannot build aligner indexes
  - No validation of reference integrity
  - No Ensembl/GENCODE URL resolution

#### Available Capability (NOT USED)

| Module | Location | Lines | Purpose | Status |
|--------|----------|-------|---------|--------|
| `ReferenceManager` | `data/reference_manager.py` | 711 | Full reference management | ❌ Not imported |
| `ReferenceBrowser` | `data/browser/reference_browser.py` | ~200 | Gradio UI for browsing | ❌ Not exposed |

#### `ReferenceManager` Capabilities (VERIFIED)

```python
# From reference_manager.py - CURRENTLY NOT USED:

class ReferenceManager:
    """
    Handles:
    - Local reference discovery
    - Reference downloads from Ensembl/GENCODE
    - Aligner index building (STAR, Salmon, BWA, HISAT2, Bowtie2)
    - Reference validation
    """
    
    def check_references(organism, assembly) -> ReferenceInfo:
        """Check local availability, return missing items with download URLs"""
    
    def download_reference(organism, assembly, resource) -> Path:
        """Download genome/gtf/transcriptome from Ensembl"""
    
    def build_index(aligner, genome_path, gtf_path) -> Path:
        """Build STAR/Salmon/BWA/HISAT2 index"""
    
    def validate_index(aligner, path) -> bool:
        """Validate index completeness"""

# Pre-configured sources:
REFERENCE_SOURCES = {
    "human": {
        "GRCh38": {
            "genome": "https://ftp.ensembl.org/pub/release-110/fasta/...",
            "gtf": "https://ftp.ensembl.org/pub/release-110/gtf/...",
            "transcriptome": "https://ftp.ensembl.org/pub/release-110/fasta/.../cdna/..."
        },
        "GRCh37": {...}
    },
    "mouse": {"GRCm39": {...}, "GRCm38": {...}}
}
```

#### Integration Plan

```python
# File: agents/tools/data_management.py

# ADD NEW FUNCTION:
def download_reference_impl(
    organism: str = "human",
    assembly: str = "GRCh38",
    resource: str = "genome",  # genome, gtf, transcriptome
    build_index: str = None,   # star, salmon, bwa, hisat2
) -> ToolResult:
    """
    Download reference data and optionally build indexes.
    """
    from workflow_composer.data.reference_manager import ReferenceManager
    
    manager = ReferenceManager(base_dir=Path("data/references"))
    
    # Check what's available
    ref_info = manager.check_references(organism, assembly)
    
    # Download missing resources
    if resource in ref_info.missing:
        path = manager.download_reference(organism, assembly, resource)
    
    # Build index if requested
    if build_index:
        manager.build_index(build_index, ref_info.genome_fasta, ref_info.annotation_gtf)
    
    return ToolResult(...)

# ENHANCE EXISTING:
def check_references_impl(...) -> ToolResult:
    """Use ReferenceManager for rich reference checking."""
    from workflow_composer.data.reference_manager import ReferenceManager
    
    manager = ReferenceManager()
    ref_info = manager.check_references(organism, assembly)
    
    # Show available, missing, and download commands
    return ToolResult(
        data={
            "available": {...},
            "missing": ref_info.missing,
            "download_urls": ref_info.download_urls,
        },
        message=format_reference_status(ref_info)
    )
```

#### New Tool Definition

```python
# In tools/__init__.py, add:
{
    "name": "download_reference",
    "description": "Download reference genomes/annotations from Ensembl and optionally build aligner indexes (STAR, Salmon, BWA)",
    "parameters": {
        "type": "object",
        "properties": {
            "organism": {"type": "string", "enum": ["human", "mouse"], "description": "Organism"},
            "assembly": {"type": "string", "description": "Genome assembly (GRCh38, GRCm39)"},
            "resource": {"type": "string", "enum": ["genome", "gtf", "transcriptome"]},
            "build_index": {"type": "string", "enum": ["star", "salmon", "bwa", "hisat2"]}
        },
        "required": ["organism"]
    }
}
```

#### Migration Steps

1. **Phase 1: Enhance check_references** (Effort: LOW)
   - Import `ReferenceManager`
   - Show download URLs for missing items
   - Display index availability

2. **Phase 2: Add download_reference tool** (Effort: MEDIUM)
   - New tool implementation
   - Progress reporting for large downloads
   - Decompress .gz files automatically

3. **Phase 3: Index Building** (Effort: HIGH)
   - Long-running job handling
   - SLURM submission for index builds
   - Validation after completion

---

### 3. Results Viewer & Collector ⭐ MEDIUM PRIORITY

#### Current Implementation (diagnostics.py)
- **Location:** `agents/tools/diagnostics.py:analyze_results_impl()`
- **Lines:** ~100
- **Approach:** Basic file type detection, simple file listing
- **Limitations:**
  - Cannot render HTML reports inline
  - Cannot parse table data (TSV/CSV)
  - No summary statistics for count matrices
  - No cloud upload capability

#### Available Capability (NOT USED)

| Module | Location | Lines | Purpose | Status |
|--------|----------|-------|---------|--------|
| `ResultViewer` | `results/viewer.py` | 470 | Render HTML, tables, images, JSON | ❌ Not imported |
| `ResultCollector` | `results/collector.py` | 439 | Smart file discovery with patterns | ❌ Not imported |
| `ResultArchiver` | `results/archiver.py` | ~200 | Archive and export results | ❌ Not imported |
| `CloudTransfer` | `results/cloud_transfer.py` | ~200 | Upload to GCS/S3 | ❌ Not imported |
| File type detection | `results/detector.py` | ~150 | Detect QC reports, tables, logs | ❌ Not imported |

#### Integration Plan

```python
# File: agents/tools/diagnostics.py

def analyze_results_impl(
    results_path: str = None,
    result_type: str = None,
    render_preview: bool = False,
) -> ToolResult:
    """
    Analyze results using ResultCollector and ResultViewer.
    """
    from workflow_composer.results import ResultCollector, ResultViewer
    
    # Use smart collector
    collector = ResultCollector(pipeline_type=result_type)
    summary = collector.scan(results_path)
    
    # Render previews if requested
    viewer = ResultViewer()
    previews = []
    for report in summary.qc_reports[:3]:  # Top 3 reports
        content = viewer.render(report)
        previews.append(content)
    
    return ToolResult(
        data={
            "summary": summary.to_dict(),
            "file_tree": summary.file_tree,
            "qc_reports": [r.to_dict() for r in summary.qc_reports],
            "previews": previews,
        },
        message=format_result_summary(summary)
    )
```

#### Migration Steps

1. **Phase 1: Smart Collection** (Effort: LOW)
   - Use `ResultCollector` for intelligent file discovery
   - Pattern-based file categorization
   - Pipeline-specific patterns

2. **Phase 2: Rich Previews** (Effort: MEDIUM)
   - Integrate `ResultViewer` for HTML/table rendering
   - Return preview content in ToolResult
   - Handle large files gracefully

3. **Phase 3: Cloud Export** (Effort: MEDIUM)
   - Add `export_results` tool using `CloudTransfer`
   - Support GCS/S3 with signed URLs
   - Archive with `ResultArchiver`

---

### 4. Workflow Monitoring ⭐ MEDIUM PRIORITY

#### Current Implementation (execution.py)
- **Location:** `agents/tools/execution.py:get_job_status_impl()`
- **Lines:** ~70
- **Approach:** SLURM `squeue` only
- **Limitations:**
  - No Nextflow-specific parsing
  - Cannot show process-level progress
  - No resource usage metrics
  - Cannot parse trace files

#### Available Capability (NOT USED)

| Module | Location | Lines | Purpose | Status |
|--------|----------|-------|---------|--------|
| `WorkflowMonitor` | `monitor/workflow_monitor.py` | 416 | Parse Nextflow logs/trace, track processes | ❌ Not imported |

#### Integration Plan

```python
# File: agents/tools/execution.py

def get_job_status_impl(
    job_id: str = None,
    workflow_dir: str = None,
    detailed: bool = False,
) -> ToolResult:
    """
    Get job status using WorkflowMonitor + SLURM.
    """
    from workflow_composer.monitor import WorkflowMonitor
    
    monitor = WorkflowMonitor()
    
    # Parse Nextflow execution if workflow_dir provided
    if workflow_dir:
        execution = monitor.scan_workflow(workflow_dir)
        if execution:
            return ToolResult(
                data={
                    "status": execution.status.value,
                    "progress": execution.progress,
                    "processes": execution.process_counts,
                    "duration": execution.duration,
                },
                message=format_workflow_status(execution)
            )
    
    # Fall back to SLURM status
    return slurm_status(job_id)
```

---

### 5. Workflow Visualization ⭐ LOW PRIORITY

#### Available Capability (NOT USED)

| Module | Location | Lines | Purpose | Status |
|--------|----------|-------|---------|--------|
| `WorkflowVisualizer` | `viz/visualizer.py` | 461 | DAG diagrams, HTML reports | ❌ Not imported |

#### Integration Plan

```python
# File: agents/tools/workflow.py

# ADD NEW TOOL:
def visualize_workflow_impl(
    workflow_path: str,
    output_format: str = "png",  # png, svg, html
) -> ToolResult:
    """
    Generate workflow DAG visualization.
    """
    from workflow_composer.viz import WorkflowVisualizer
    
    viz = WorkflowVisualizer()
    diagram_path = viz.render_dag(workflow, format=output_format)
    
    return ToolResult(
        data={"diagram": str(diagram_path)},
        message=f"📊 Workflow diagram generated: {diagram_path}"
    )
```

---

### 6. Autonomous Agent System ⭐ DEFERRED

#### Analysis

The autonomous agent system is a **complete parallel architecture** that includes:

| Module | Lines | Purpose | Integration Complexity |
|--------|-------|---------|----------------------|
| `AutonomousAgent` | 1089 | Full task loop with tool execution | VERY HIGH |
| `RecoveryManager` | 810 | Automatic error recovery | HIGH |
| `JobMonitor` | ~300 | Job event tracking | MEDIUM |
| `HealthChecker` | ~200 | System health monitoring | MEDIUM |
| `CommandSandbox` | ~400 | Safe command execution | MEDIUM |

#### Recommendation: DEFER

The autonomous agent is designed for **long-running autonomous operations**, not interactive chat. Integrating it would require:
- Architectural changes to support async task loops
- UI changes for task confirmation dialogs
- Session management for multi-step tasks

**Recommendation:** Keep as separate capability for advanced use cases. May be activated via explicit command like "run autonomously" in future.

#### Partial Integration Opportunity

The `RecoveryManager` could be partially integrated into `diagnose_error`:
- Error pattern matching overlaps with `diagnosis/patterns.py`
- Recovery actions could enhance `AutoFixEngine`

**Action:** Merge recovery patterns into `diagnosis/patterns.py` rather than full integration.

---

## Redundancy Analysis & Consolidation

### Identified Redundancies

| Capability | Location 1 | Location 2 | Resolution |
|------------|------------|------------|------------|
| Error patterns (bio) | `diagnosis/patterns.py` (702 lines) | `agents/tools/diagnostics.py` (simple dict) | USE patterns.py |
| Error patterns (infra) | `autonomous/recovery.py` (vLLM/SLURM patterns) | `diagnosis/patterns.py` | COMPLEMENTARY - keep both |
| Job monitoring | `autonomous/job_monitor.py` | `monitor/workflow_monitor.py` | USE workflow_monitor.py |
| Health checking | `autonomous/health_checker.py` | N/A | DEFER (not needed for chat) |
| LLM providers | `diagnosis/agent.py` (DIAGNOSIS_PROVIDERS) | `web/chat_handler.py` (LLMProvider) | USE chat_handler.py |
| Reference download | `data/reference_manager.py` | `core/preflight_validator.py` (TODO stub) | USE reference_manager.py |
| Result scanning | `results/collector.py` | `web/archive/result_browser.py` | ALREADY USES collector ✅ |

> **Note:** The error patterns in `autonomous/recovery.py` focus on infrastructure (vLLM, SLURM, Python imports) 
> while `diagnosis/patterns.py` focuses on bioinformatics workflows (OOM, file not found, singularity).
> These are **complementary**, not redundant. Consider merging infrastructure patterns into patterns.py for unified error handling.

### Consolidation Actions

1. **Error Patterns**
   - `diagnosis/patterns.py` has 50+ comprehensive patterns with fix suggestions
   - `autonomous/recovery.py` has simpler patterns
   - **Action:** Keep `diagnosis/patterns.py` as the canonical source

2. **Job Monitoring**
   - `monitor/workflow_monitor.py` is focused and complete
   - `autonomous/job_monitor.py` is for autonomous loop only
   - **Action:** Use `WorkflowMonitor` for agent tools, leave `JobMonitor` for future autonomous mode

3. **LLM Provider Selection**
   - Both have provider priority lists
   - **Action:** Use `chat_handler.py`'s LLM infrastructure for consistency

---

## Implementation Priority Matrix

| Phase | Capability | Tool | Effort | Impact | Dependencies |
|-------|------------|------|--------|--------|--------------|
| 1 | Error Diagnosis Patterns | `diagnose_error` | LOW | HIGH | None |
| 1 | Reference Check Enhancement | `check_references` | LOW | HIGH | None |
| 2 | Log Collection | `diagnose_error` | MEDIUM | HIGH | Phase 1 |
| 2 | Download Reference | NEW `download_reference` | MEDIUM | HIGH | Phase 1 |
| 2 | Result Collection | `analyze_results` | MEDIUM | MEDIUM | None |
| 3 | LLM Diagnosis Fallback | `diagnose_error` | MEDIUM | HIGH | Phase 2 |
| 3 | Result Viewer | `analyze_results` | MEDIUM | MEDIUM | Phase 2 |
| 3 | Workflow Monitor | `get_job_status` | LOW | MEDIUM | None |
| 4 | Historical Learning | `diagnose_error` | LOW | MEDIUM | Phase 3 |
| 4 | Auto-Fix Engine | NEW `apply_fix` | HIGH | HIGH | Phase 3 |
| 4 | Workflow Visualization | NEW `visualize_workflow` | LOW | LOW | None |
| 5 | Index Building | `download_reference` | HIGH | MEDIUM | Phase 2 |
| 5 | Cloud Export | NEW `export_results` | MEDIUM | LOW | Phase 2 |

---

## Testing Strategy

### Unit Tests Required

```python
# tests/test_integrated_tools.py

class TestDiagnoseErrorIntegration:
    def test_pattern_matching_oom(self):
        """Test OOM error detection with full patterns"""
    
    def test_log_collection_nextflow(self):
        """Test LogCollector finds .nextflow.log"""
    
    def test_historical_boost(self):
        """Test confidence increases with history"""
    
    def test_llm_fallback_unknown(self):
        """Test LLM activation for unknown patterns"""

class TestReferenceManagerIntegration:
    def test_check_references_human_grch38(self):
        """Test reference checking for human GRCh38"""
    
    def test_download_urls_populated(self):
        """Test download URLs returned for missing items"""
    
    def test_index_validation_star(self):
        """Test STAR index validation"""

class TestResultsIntegration:
    def test_collector_finds_multiqc(self):
        """Test ResultCollector finds MultiQC reports"""
    
    def test_viewer_renders_html(self):
        """Test ResultViewer renders HTML content"""
```

### Integration Tests

```python
# tests/integration/test_agent_tools_e2e.py

class TestAgentToolsE2E:
    def test_diagnose_and_fix_missing_directory(self):
        """Full flow: diagnose missing dir → suggest fix → apply fix"""
    
    def test_check_and_download_reference(self):
        """Full flow: check refs → identify missing → download"""
    
    def test_workflow_run_and_monitor(self):
        """Full flow: submit job → monitor → get results"""
```

---

## Timeline Estimate

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1 | 2 days | Pattern integration, reference check enhancement |
| Phase 2 | 3 days | Log collection, download_reference tool, result collection |
| Phase 3 | 3 days | LLM fallback, result viewer, workflow monitor |
| Phase 4 | 2 days | Historical learning, auto-fix engine |
| Phase 5 | 2 days | Index building, cloud export |
| Testing | 2 days | Unit + integration tests |
| **Total** | **14 days** | Full capability integration |

---

## Success Criteria

- [ ] `diagnose_error` uses 50+ patterns from `diagnosis/patterns.py`
- [ ] `diagnose_error` collects logs from Nextflow work directories
- [ ] `diagnose_error` falls back to LLM for unknown errors
- [ ] `check_references` shows download URLs for missing items
- [ ] New `download_reference` tool can fetch from Ensembl
- [ ] `analyze_results` uses `ResultCollector` for smart discovery
- [ ] `get_job_status` parses Nextflow trace files
- [ ] All tools have unit test coverage >80%
- [ ] Web chat successfully uses all integrated tools

---

## Appendix A: File Locations

```
src/workflow_composer/
├── agents/
│   └── tools/
│       ├── diagnostics.py      ← ENHANCE diagnose_error, analyze_results
│       ├── data_management.py  ← ENHANCE check_references, ADD download_reference
│       ├── execution.py        ← ENHANCE get_job_status
│       └── workflow.py         ← ADD visualize_workflow
├── diagnosis/
│   ├── agent.py         (809 lines) → ErrorDiagnosisAgent
│   ├── patterns.py      (702 lines) → 50+ error patterns
│   ├── log_collector.py (375 lines) → LogCollector
│   ├── history.py       (333 lines) → DiagnosisHistory
│   ├── auto_fix.py      (501 lines) → AutoFixEngine
│   └── categories.py    (~200 lines) → ErrorCategory enum
├── data/
│   └── reference_manager.py (711 lines) → ReferenceManager
├── results/
│   ├── viewer.py     (470 lines) → ResultViewer
│   ├── collector.py  (439 lines) → ResultCollector
│   ├── archiver.py   (~200 lines) → ResultArchiver
│   └── cloud_transfer.py (~200 lines) → CloudTransfer
├── monitor/
│   └── workflow_monitor.py (416 lines) → WorkflowMonitor
└── viz/
    └── visualizer.py (461 lines) → WorkflowVisualizer
```

---

## Appendix B: Import Mapping

```python
# Quick reference for imports when integrating:

# Error Diagnosis
from workflow_composer.diagnosis import ErrorDiagnosisAgent
from workflow_composer.diagnosis import LogCollector, CollectedLogs
from workflow_composer.diagnosis import get_diagnosis_history, DiagnosisHistory
from workflow_composer.diagnosis.auto_fix import AutoFixEngine, FixResult
from workflow_composer.diagnosis.patterns import ERROR_PATTERNS, get_all_patterns
from workflow_composer.diagnosis.categories import ErrorCategory, ErrorDiagnosis, FixSuggestion

# Reference Management
from workflow_composer.data.reference_manager import ReferenceManager, ReferenceInfo

# Results
from workflow_composer.results import ResultCollector, ResultViewer, ResultSummary

# Monitoring
from workflow_composer.monitor import WorkflowMonitor, WorkflowExecution

# Visualization
from workflow_composer.viz import WorkflowVisualizer
```

---

*Document maintained by: BioPipelines Development Team*  
*Next Review: After Phase 1 completion*
