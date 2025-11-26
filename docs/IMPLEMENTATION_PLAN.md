# BioPipelines: Comprehensive Implementation Plan

## Executive Summary

This document outlines the complete implementation plan to address all identified gaps in the BioPipelines framework, transforming it from a functional prototype into a production-ready bioinformatics platform.

**Total Estimated Effort:** 35-45 days  
**Priority Focus:** Error Diagnosis → Result Visualization → Auto-Fix → UX Polish

---

## Gap Analysis Summary

| Gap | Current State | Target State | Priority | Effort |
|-----|---------------|--------------|----------|--------|
| Error Diagnosis | Basic pattern detection | AI-powered root cause analysis | 🔴 CRITICAL | 10 days |
| Result Visualization | Not integrated | MultiQC/reports in UI | 🔴 CRITICAL | 8 days |
| Result Download | Missing | Archive & transfer to local | 🔴 CRITICAL | 5 days |
| Auto-Fix Engine | Not implemented | Safe auto-remediation | 🟠 HIGH | 7 days |
| Reference Browser | Manual path entry | Interactive reference manager | 🟡 MEDIUM | 5 days |
| Container Tiers | 11 tiers (complex) | 3 tiers (simplified) | 🟡 MEDIUM | 3 days |
| Dataset Discovery | Manual | Sample dataset browser | 🟢 LOW | 3 days |
| Tutorial Integration | Docs only | Interactive tutorials | 🟢 LOW | 4 days |

---

## Phase 1: Error Diagnosis & Auto-Fix Agent
**Duration: 10 days**
**Priority: 🔴 CRITICAL**

### 1.1 Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        ERROR DIAGNOSIS SYSTEM                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ┌─────────────┐         ┌─────────────────────────────────────────────┐    │
│   │ Failed Job  │────────▶│           Log Collector                     │    │
│   └─────────────┘         │  • .nextflow.log                            │    │
│                           │  • slurm_*.err                              │    │
│                           │  • work/*/.command.err                      │    │
│                           │  • main.nf (workflow)                       │    │
│                           │  • nextflow.config                          │    │
│                           └─────────────────────────────────────────────┘    │
│                                          │                                    │
│                                          ▼                                    │
│   ┌──────────────────────────────────────────────────────────────────────┐   │
│   │                    TIER 1: Pattern Matcher (< 100ms)                  │   │
│   │  ┌────────────────────────────────────────────────────────────────┐  │   │
│   │  │  30+ Known Patterns:                                           │  │   │
│   │  │  • File not found → Check path, download reference            │  │   │
│   │  │  • OOM → Increase memory, reduce threads                      │  │   │
│   │  │  • Container error → Build/pull container                     │  │   │
│   │  │  • Permission denied → Fix permissions                        │  │   │
│   │  │  • Module not found → Install package                         │  │   │
│   │  └────────────────────────────────────────────────────────────────┘  │   │
│   │              │ Match found                    │ No match              │   │
│   │              ▼                                ▼                       │   │
│   │  ┌────────────────────┐          ┌────────────────────────────────┐  │   │
│   │  │ Instant Suggestion │          │   TIER 2: LLM Deep Analysis    │  │   │
│   │  │ (No LLM required)  │          │   • Context-aware diagnosis    │  │   │
│   │  └────────────────────┘          │   • Multi-step fix plans       │  │   │
│   │                                  │   • Confidence scoring         │  │   │
│   │                                  └────────────────────────────────┘  │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│                                          │                                    │
│                                          ▼                                    │
│   ┌──────────────────────────────────────────────────────────────────────┐   │
│   │                    AUTO-FIX ENGINE                                    │   │
│   │                                                                       │   │
│   │   🟢 SAFE (Auto)          🟡 LOW (Notify)       🔴 HIGH (Confirm)     │   │
│   │   • Create directory      • Pull container      • Modify workflow     │   │
│   │   • Retry job             • Download ref        • Install packages    │   │
│   │   • Increase memory       • Fix permissions     • Change config       │   │
│   │                                                                       │   │
│   │   ┌──────────────────────────────────────────────────────────────┐   │   │
│   │   │  For Code Fixes → GitHub Copilot Coding Agent                │   │   │
│   │   │  mcp_github_create_pull_request_with_copilot()              │   │   │
│   │   └──────────────────────────────────────────────────────────────┘   │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Implementation Tasks

#### Week 1: Core Diagnosis (Days 1-5)

| Task | Description | File(s) | Est. |
|------|-------------|---------|------|
| 1.1 | Create error taxonomy (15+ categories) | `diagnosis/categories.py` | 4h |
| 1.2 | Build pattern database (30+ patterns) | `diagnosis/patterns.py` | 8h |
| 1.3 | Implement log collector | `diagnosis/log_collector.py` | 4h |
| 1.4 | Create ErrorDiagnosisAgent class | `diagnosis/error_agent.py` | 8h |
| 1.5 | Add LLM diagnosis prompts | `diagnosis/prompts.py` | 4h |
| 1.6 | Implement structured output parser | `diagnosis/parser.py` | 4h |
| 1.7 | Unit tests for pattern matching | `tests/test_diagnosis.py` | 4h |

#### Week 2: Auto-Fix & Integration (Days 6-10)

| Task | Description | File(s) | Est. |
|------|-------------|---------|------|
| 2.1 | Define fix risk levels | `diagnosis/auto_fix.py` | 4h |
| 2.2 | Implement safe fix executors | `diagnosis/auto_fix.py` | 8h |
| 2.3 | Add user confirmation flow | `diagnosis/auto_fix.py` | 4h |
| 2.4 | GitHub Copilot integration | `diagnosis/github_agent.py` | 8h |
| 2.5 | Add "Diagnose" button to UI | `web/gradio_app.py` | 4h |
| 2.6 | Create diagnosis result panel | `web/gradio_app.py` | 4h |
| 2.7 | End-to-end testing | `tests/test_e2e_diagnosis.py` | 4h |

### 1.3 New Files

```
src/workflow_composer/diagnosis/
├── __init__.py              # Package exports
├── categories.py            # ErrorCategory enum (15+ types)
├── patterns.py              # ERROR_PATTERNS database (30+)
├── log_collector.py         # Collect all relevant logs
├── error_agent.py           # Main ErrorDiagnosisAgent
├── auto_fix.py              # AutoFixEngine with risk levels
├── prompts.py               # LLM prompt templates
├── parser.py                # Structured output parsing
└── github_agent.py          # GitHub Copilot integration
```

### 1.4 Sample Code

```python
# diagnosis/error_agent.py
class ErrorDiagnosisAgent:
    """
    AI-powered error diagnosis for bioinformatics workflows.
    
    Uses tiered approach:
    1. Pattern matching (fast, offline)
    2. LLM analysis (comprehensive, contextual)
    """
    
    def __init__(self, llm: Optional[LLMAdapter] = None):
        self.llm = llm
        self.pattern_matcher = PatternMatcher()
        self.log_collector = LogCollector()
        self.auto_fixer = AutoFixEngine()
    
    async def diagnose(self, job: PipelineJob) -> ErrorDiagnosis:
        """Full diagnosis workflow."""
        # Step 1: Collect all logs
        logs = self.log_collector.collect(job)
        
        # Step 2: Try pattern matching first (fast)
        match = self.pattern_matcher.match(logs)
        if match and match.confidence > 0.8:
            return self._build_diagnosis(match, logs)
        
        # Step 3: Use LLM for complex errors
        if self.llm:
            return await self._llm_diagnosis(logs, job)
        
        return ErrorDiagnosis(
            category=ErrorCategory.UNKNOWN,
            message="Unable to diagnose. Please check logs manually."
        )
    
    async def suggest_fixes(self, diagnosis: ErrorDiagnosis) -> List[Fix]:
        """Generate fix suggestions."""
        fixes = []
        
        # Get pattern-based fixes
        if diagnosis.pattern_match:
            fixes.extend(diagnosis.pattern_match.auto_fixes)
        
        # Get LLM-suggested fixes
        if self.llm and diagnosis.requires_llm_fix:
            llm_fixes = await self._get_llm_fixes(diagnosis)
            fixes.extend(llm_fixes)
        
        return sorted(fixes, key=lambda f: f.confidence, reverse=True)
    
    async def apply_fix(self, fix: Fix, confirm: bool = True) -> FixResult:
        """Apply a suggested fix."""
        if fix.risk_level == FixRiskLevel.HIGH and not confirm:
            return FixResult(
                success=False,
                message="High-risk fix requires explicit confirmation"
            )
        
        return await self.auto_fixer.execute(fix)
```

---

## Phase 2: Result Visualization & Download
**Duration: 8 days**
**Priority: 🔴 CRITICAL**
**Status: ✅ COMPLETE** (Implemented November 26, 2025)

> **Implementation:** See `docs/RESULTS_VISUALIZATION_DESIGN.md` for full details.
> - Created `src/workflow_composer/results/` package (8 files)
> - Added "📊 Results" tab to Gradio UI
> - Supports HTML reports, images, tables, text, and download

### 2.1 Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      RESULT VISUALIZATION SYSTEM                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │                    RESULT COLLECTOR                                  │    │
│   │                                                                      │    │
│   │  Completed Job → Scan Output Directory                              │    │
│   │                                                                      │    │
│   │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐         │    │
│   │  │  MultiQC HTML  │  │  BAM/VCF/BED   │  │  Plots (PNG)   │         │    │
│   │  │  Reports       │  │  Data Files    │  │  Figures       │         │    │
│   │  └────────────────┘  └────────────────┘  └────────────────┘         │    │
│   └─────────────────────────────────────────────────────────────────────┘    │
│                                          │                                    │
│                                          ▼                                    │
│   ┌──────────────────────────────────────────────────────────────────────┐   │
│   │                    RESULT BROWSER (UI)                                │   │
│   │                                                                       │   │
│   │  ┌─────────────────────────────────────────────────────────────────┐ │   │
│   │  │  📊 QC Reports              📁 Output Files    📈 Visualizations │ │   │
│   │  ├─────────────────────────────────────────────────────────────────┤ │   │
│   │  │                                                                  │ │   │
│   │  │  🔍 MultiQC Report     │ ├── bam/                │ [View Plot]   │ │   │
│   │  │  [View in Browser]     │ │   ├── sample1.bam    │               │ │   │
│   │  │                        │ │   └── sample2.bam    │ Peak Distrib. │ │   │
│   │  │  📋 FastQC Reports     │ ├── peaks/             │ [Download]    │ │   │
│   │  │  • sample1_fastqc.html │ │   └── peaks.bed      │               │ │   │
│   │  │  • sample2_fastqc.html │ └── counts/            │ QC Summary    │ │   │
│   │  │                        │     └── matrix.tsv     │ [View]        │ │   │
│   │  └─────────────────────────────────────────────────────────────────┘ │   │
│   │                                                                       │   │
│   │  [📥 Download All Results (ZIP)]   [📤 Transfer to Cloud Storage]   │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│                                          │                                    │
│                                          ▼                                    │
│   ┌──────────────────────────────────────────────────────────────────────┐   │
│   │                    RESULT ARCHIVE & DOWNLOAD                          │   │
│   │                                                                       │   │
│   │  1. Create ZIP archive of results directory                          │   │
│   │  2. Generate download link (temporary, secure)                        │   │
│   │  3. Optional: Upload to GCS/S3 for persistent storage                │   │
│   │  4. Send notification with download link                              │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Implementation Tasks

#### Days 1-4: Result Collection & Display

| Task | Description | File(s) | Est. |
|------|-------------|---------|------|
| 1.1 | Create ResultCollector class | `results/collector.py` | 4h |
| 1.2 | Implement file type detection | `results/detector.py` | 4h |
| 1.3 | Add MultiQC HTML embedding | `web/gradio_app.py` | 8h |
| 1.4 | Create file browser component | `web/components/file_browser.py` | 8h |
| 1.5 | Add plot/image viewer | `web/components/plot_viewer.py` | 4h |

#### Days 5-8: Download & Archive

| Task | Description | File(s) | Est. |
|------|-------------|---------|------|
| 2.1 | Implement ZIP archiver | `results/archiver.py` | 4h |
| 2.2 | Add download endpoint | `web/api.py` | 4h |
| 2.3 | Create GCS/S3 uploader | `results/cloud_transfer.py` | 8h |
| 2.4 | Add "Download Results" button | `web/gradio_app.py` | 4h |
| 2.5 | Integrate with existing monitor | `monitor/workflow_monitor.py` | 4h |
| 2.6 | Email notification with link | `notification/email.py` | 4h |

### 2.3 New Files

```
src/workflow_composer/results/
├── __init__.py
├── collector.py         # Scan and categorize output files
├── detector.py          # File type detection
├── archiver.py          # ZIP creation
├── cloud_transfer.py    # GCS/S3 upload
└── metadata.py          # Result metadata (sizes, timestamps)

src/workflow_composer/web/components/
├── __init__.py
├── file_browser.py      # Gradio file browser
├── plot_viewer.py       # Image/plot display
└── multiqc_embed.py     # Embed MultiQC HTML
```

### 2.4 UI Integration (gradio_app.py)

```python
# Add "Results" tab after job completion
def view_results(job_id: str) -> Tuple[str, str, List[str]]:
    """Load results for a completed job."""
    job = pipeline_executor.get_job_status(job_id)
    if not job or job.status != JobStatus.COMPLETED:
        return "Job not completed", "", []
    
    collector = ResultCollector(job.output_dir)
    results = collector.scan()
    
    # MultiQC HTML if exists
    multiqc_html = ""
    if results.multiqc_report:
        with open(results.multiqc_report) as f:
            multiqc_html = f.read()
    
    # File tree
    file_tree = collector.get_file_tree()
    
    # Plot images
    plots = results.get_plots()
    
    return multiqc_html, file_tree, plots

# Add download functionality
def download_results(job_id: str) -> str:
    """Create downloadable archive of results."""
    job = pipeline_executor.get_job_status(job_id)
    archiver = ResultArchiver()
    
    zip_path = archiver.create_archive(
        job.output_dir,
        f"{job.name}_{job_id}.zip"
    )
    
    return zip_path  # Gradio will make this downloadable
```

---

## Phase 3: Reference & Dataset Browser
**Duration: 5 days**
**Priority: 🟡 MEDIUM**

### 3.1 Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    REFERENCE & DATASET BROWSER                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │  📚 Reference Genomes                                                │    │
│   │  ┌───────────────────────────────────────────────────────────────┐  │    │
│   │  │ Organism    │ Build   │ Source  │ Indexes       │ Status      │  │    │
│   │  ├─────────────┼─────────┼─────────┼───────────────┼─────────────┤  │    │
│   │  │ Human       │ GRCh38  │ Ensembl │ BWA ✅ STAR ✅│ Ready       │  │    │
│   │  │ Human       │ hg19    │ UCSC    │ BWA ✅ STAR ❌│ Partial     │  │    │
│   │  │ Mouse       │ GRCm39  │ Ensembl │ BWA ❌ STAR ❌│ [Download]  │  │    │
│   │  │ Zebrafish   │ GRCz11  │ Ensembl │ BWA ❌ STAR ❌│ [Download]  │  │    │
│   │  └───────────────────────────────────────────────────────────────┘  │    │
│   │  [+ Add Custom Reference]                                            │    │
│   └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │  📦 Sample Datasets                                                  │    │
│   │  ┌───────────────────────────────────────────────────────────────┐  │    │
│   │  │ Dataset          │ Type    │ Size   │ Description    │ Action │  │    │
│   │  ├──────────────────┼─────────┼────────┼────────────────┼────────┤  │    │
│   │  │ ENCODE ChIP-seq  │ ChIP    │ 2.1GB  │ H3K4me3 demo   │ [Use]  │  │    │
│   │  │ SRA RNA-seq      │ RNA-seq │ 1.5GB  │ Mouse liver    │ [Use]  │  │    │
│   │  │ 1000 Genomes     │ WGS     │ 50GB   │ NA12878 trio   │ [Use]  │  │    │
│   │  └───────────────────────────────────────────────────────────────┘  │    │
│   └─────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Implementation Tasks

| Task | Description | File(s) | Est. |
|------|-------------|---------|------|
| 1.1 | Create reference registry | `data/reference_registry.py` | 4h |
| 1.2 | Implement index checker | `data/index_checker.py` | 4h |
| 1.3 | Add sample dataset catalog | `data/sample_datasets.py` | 4h |
| 1.4 | Create browser UI component | `web/components/reference_browser.py` | 8h |
| 1.5 | Add download progress tracking | `data/downloader.py` | 4h |
| 1.6 | Integrate with workflow generator | `core/workflow_generator.py` | 4h |

---

## Phase 4: Container Simplification
**Duration: 3 days**
**Priority: 🟡 MEDIUM**

### 4.1 Current vs. Target

```
CURRENT (11 Tiers):                    TARGET (3 Tiers):
├── base                               ├── base (universal tools)
├── workflow-engine                    │   • FastQC, MultiQC, fastp
├── atac-seq                           │   • Samtools, BEDTools
├── chip-seq                           │   • BWA, Bowtie2
├── dna-seq                            │
├── hic                                ├── analysis (specialty tools)
├── long-read                          │   • STAR, Salmon (RNA-seq)
├── metagenomics                       │   • MACS2, HOMER (ChIP/ATAC)
├── methylation                        │   • GATK, DeepVariant (DNA)
├── rna-seq                            │   • Long-read tools
├── scrna-seq                          │
└── structural-variants                └── specialty (rare/large tools)
                                           • Hi-C tools
                                           • Metagenomics
                                           • Single-cell
```

### 4.2 Implementation Tasks

| Task | Description | File(s) | Est. |
|------|-------------|---------|------|
| 1.1 | Audit tool overlap across containers | `scripts/audit_containers.py` | 4h |
| 1.2 | Design consolidated container specs | `containers/consolidated/` | 4h |
| 1.3 | Update container references | `config/containers.yaml` | 4h |
| 1.4 | Test consolidated containers | `tests/test_containers.py` | 8h |
| 1.5 | Update documentation | `docs/CONTAINER_ARCHITECTURE.md` | 4h |

---

## Phase 5: UI Polish & Tutorials
**Duration: 4 days**
**Priority: 🟢 LOW**

### 5.1 Tasks

| Task | Description | Est. |
|------|-------------|------|
| Add guided tutorial mode | 8h |
| Improve error messages | 4h |
| Add keyboard shortcuts | 4h |
| Create video walkthroughs | 8h |
| Improve mobile responsiveness | 4h |

---

## Implementation Timeline

```
Week 1: Error Diagnosis Core
├── Day 1-2: Pattern database & error taxonomy
├── Day 3-4: ErrorDiagnosisAgent implementation
└── Day 5: LLM integration & prompts

Week 2: Auto-Fix & Integration
├── Day 6-7: Auto-fix engine with risk levels
├── Day 8: GitHub Copilot integration
├── Day 9: UI integration (Diagnose button)
└── Day 10: Testing & refinement

Week 3: Result Visualization
├── Day 11-12: ResultCollector & file browser
├── Day 13-14: MultiQC embedding & plot viewer
└── Day 15: Download/archive functionality

Week 4: Result Download & Transfer
├── Day 16-17: ZIP archiver & download endpoint
├── Day 18: Cloud transfer (GCS/S3)
└── Day 19: Email notifications

Week 5: Reference Browser
├── Day 20-21: Reference registry & index checker
├── Day 22-23: Sample dataset catalog
└── Day 24: UI component integration

Week 6: Container & Polish
├── Day 25-26: Container consolidation
├── Day 27-28: Tutorial integration
└── Day 29-30: Testing & documentation
```

---

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Error diagnosis accuracy | 0% | >85% | Pattern + LLM tests |
| Auto-fix success rate | N/A | >70% | Fix execution logs |
| Result visibility | 0% | 100% | All jobs have viewable results |
| Download availability | 0% | 100% | All results downloadable |
| Reference setup time | >30 min | <5 min | User testing |
| Container build time | Variable | <10 min | Build logs |

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM hallucinations in fixes | HIGH | Confidence thresholds, user confirmation |
| Auto-fix causes damage | HIGH | Risk levels, dry-run mode, backup |
| Large result downloads | MEDIUM | Streaming, chunked transfer |
| GitHub API rate limits | MEDIUM | Caching, batch operations |
| Container compatibility | MEDIUM | Extensive testing matrix |

---

## Dependencies & Prerequisites

### External Services
- [ ] GitHub API access (for Copilot Coding Agent)
- [ ] GCS/S3 credentials (for cloud transfer)
- [ ] Email SMTP settings (for notifications)

### Existing Components
- [x] LLM adapters (OpenAI, vLLM, etc.)
- [x] Workflow generator
- [x] Job submission system
- [x] Gradio UI framework

---

## Next Steps

1. **Review and approve this plan**
2. **Prioritize first sprint** (Error Diagnosis - Week 1-2)
3. **Set up development branch** for Phase 1
4. **Begin implementation** of `src/workflow_composer/diagnosis/`

---

## Appendix: File Changes Summary

### New Files to Create

```
src/workflow_composer/
├── diagnosis/                    # NEW PACKAGE (Phase 1)
│   ├── __init__.py
│   ├── categories.py             # Error taxonomy
│   ├── patterns.py               # Pattern database
│   ├── log_collector.py          # Log aggregation
│   ├── error_agent.py            # Main agent class
│   ├── auto_fix.py               # Fix execution
│   ├── prompts.py                # LLM prompts
│   ├── parser.py                 # Output parsing
│   └── github_agent.py           # Copilot integration
│
├── results/                      # NEW PACKAGE (Phase 2)
│   ├── __init__.py
│   ├── collector.py              # Result scanning
│   ├── detector.py               # File type detection
│   ├── archiver.py               # ZIP creation
│   ├── cloud_transfer.py         # GCS/S3 upload
│   └── metadata.py               # Result metadata
│
└── web/components/               # NEW SUBPACKAGE (Phase 2-3)
    ├── __init__.py
    ├── file_browser.py           # File tree component
    ├── plot_viewer.py            # Image viewer
    ├── multiqc_embed.py          # HTML embedding
    └── reference_browser.py      # Reference manager
```

### Files to Modify

```
src/workflow_composer/web/gradio_app.py     # Add Diagnose, Results, Download
src/workflow_composer/monitor/workflow_monitor.py  # Result tracking
src/workflow_composer/data/downloader.py    # Reference browser integration
src/workflow_composer/core/composer.py      # Diagnosis integration
config/defaults.yaml                         # New config options
```

---

**Document Version:** 1.0  
**Created:** November 25, 2025  
**Author:** BioPipelines Development Team
