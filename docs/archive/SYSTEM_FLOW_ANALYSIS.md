# BioPipelines: Complete System Flow Analysis & Improvement Plan

## Executive Summary

This document traces the complete sequence of events from receiving a user query in the frontend to displaying pipeline execution results. It includes a critical evaluation of the current implementation and proposes improvements for a production-ready system.

---

## 📊 Current System Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    COMPLETE SYSTEM FLOW                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

                                        USER QUERY
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: FRONTEND RECEPTION (gradio_app.py)                                                            │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  1. User enters query in Gradio Chatbot                                                                 │
│  2. chat_with_composer() function triggered                                                             │
│  3. Provider validation (OpenAI/vLLM/Ollama)                                                            │
│  4. AppState initialization check                                                                        │
│                                                                                                          │
│  FILES: src/workflow_composer/web/gradio_app.py                                                         │
│  FUNCTIONS: chat_with_composer(), AppState.initialize()                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: QUERY PARSING (query_parser.py, query_parser_ensemble.py)                                     │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  Current Flow:                                                                                           │
│  ┌────────────┐    ┌─────────────────┐    ┌────────────────┐                                            │
│  │ User Query │───►│ IntentParser    │───►│ LLM (GPT-4o)   │                                            │
│  └────────────┘    │ .parse()        │    │ JSON Response  │                                            │
│                    └─────────────────┘    └────────────────┘                                            │
│                           │                                                                              │
│                           ▼                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────┐            │
│  │ ParsedIntent:                                                                            │            │
│  │   - analysis_type: "rna_seq_differential_expression"                                     │            │
│  │   - organism: "mouse"                                                                    │            │
│  │   - genome_build: "GRCm39"                                                               │            │
│  │   - data_format: "paired_end_fastq"                                                      │            │
│  │   - confidence: 0.85                                                                     │            │
│  └─────────────────────────────────────────────────────────────────────────────────────────┘            │
│                                                                                                          │
│  ⚠️  ISSUE: Pure LLM parsing can misclassify (e.g., "long-read" → "custom")                             │
│  ✅ SOLUTION: Hybrid approach implemented (rules-first, LLM fallback)                                    │
│                                                                                                          │
│  FILES: src/workflow_composer/core/query_parser.py                                                       │
│         src/workflow_composer/core/query_parser_ensemble.py (optional)                                   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: TOOL SELECTION (tool_selector.py)                                                             │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  1. Map analysis_type → required tools via ANALYSIS_TOOL_MAP                                            │
│  2. Categorize: required, recommended, optional                                                          │
│                                                                                                          │
│  Example for "rna_seq_differential_expression":                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────┐            │
│  │ Required:     fastqc, star, featurecounts                                                │            │
│  │ Recommended:  multiqc, samtools                                                          │            │
│  │ Analysis:     deseq2, edger                                                              │            │
│  └─────────────────────────────────────────────────────────────────────────────────────────┘            │
│                                                                                                          │
│  ⚠️  ISSUE: No validation if tools exist in containers                                                  │
│  ⚠️  ISSUE: No version compatibility checking                                                            │
│                                                                                                          │
│  FILES: src/workflow_composer/core/tool_selector.py                                                      │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: MODULE MAPPING (module_mapper.py)                                                             │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  1. For each tool → find corresponding Nextflow module                                                   │
│  2. Check if module exists in: knowledge_base/modules/{container}/{module}.nf                           │
│  3. If missing: auto-generate using LLM                                                                  │
│                                                                                                          │
│  ┌──────────────────────────────────────────────────────────────────────────────────────┐               │
│  │  TOOL         │  MODULE PATH                           │  PROCESSES                  │               │
│  ├───────────────┼────────────────────────────────────────┼─────────────────────────────┤               │
│  │  star         │  modules/alignment/star.nf             │  STAR_INDEX, STAR_ALIGN     │               │
│  │  featurecounts│  modules/quantification/featurecounts.nf│  FEATURECOUNTS             │               │
│  │  deseq2       │  modules/analysis/deseq2.nf            │  DESEQ2_DIFFERENTIAL        │               │
│  └──────────────────────────────────────────────────────────────────────────────────────┘               │
│                                                                                                          │
│  ⚠️  ISSUE: No container image validation                                                               │
│  ⚠️  ISSUE: No module dependency resolution                                                              │
│  ⚠️  ISSUE: No check if required databases/indexes exist                                                 │
│                                                                                                          │
│  FILES: src/workflow_composer/core/module_mapper.py                                                      │
│         knowledge_base/modules/                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 5: WORKFLOW GENERATION (workflow_generator.py)                                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  1. Load workflow pattern template                                                                       │
│  2. Chain modules based on data flow                                                                     │
│  3. Generate main.nf with proper imports                                                                 │
│  4. Generate nextflow.config with container/SLURM settings                                               │
│  5. Save to generated_workflows/{workflow_name}_{timestamp}/                                             │
│                                                                                                          │
│  Output Structure:                                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────┐            │
│  │  generated_workflows/rnaseq_mouse_star_20251125/                                        │            │
│  │  ├── main.nf                  # Main workflow                                            │            │
│  │  ├── nextflow.config          # Configuration                                            │            │
│  │  ├── modules/                 # Copied/generated modules                                 │            │
│  │  │   ├── alignment/star.nf                                                               │            │
│  │  │   ├── quantification/featurecounts.nf                                                 │            │
│  │  │   └── analysis/deseq2.nf                                                              │            │
│  │  └── README.md                # Usage instructions                                       │            │
│  └─────────────────────────────────────────────────────────────────────────────────────────┘            │
│                                                                                                          │
│  FILES: src/workflow_composer/core/workflow_generator.py                                                 │
│         knowledge_base/patterns/                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 6: PIPELINE EXECUTION (gradio_app.py → PipelineExecutor)                                         │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  1. User clicks "Run Pipeline" in UI                                                                     │
│  2. Generate SLURM batch script (run_{job_id}.sbatch)                                                    │
│  3. Submit to SLURM: sbatch run_{job_id}.sbatch                                                          │
│  4. Start monitoring thread                                                                              │
│                                                                                                          │
│  SLURM Job Flow:                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────┐            │
│  │  SLURM Queue ──► Node Allocation ──► Nextflow Launch ──► Process Execution ──► Results  │            │
│  └─────────────────────────────────────────────────────────────────────────────────────────┘            │
│                                                                                                          │
│  Nextflow Process Execution:                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────────────────────┐             │
│  │  FASTQC → STAR_INDEX → STAR_ALIGN → FEATURECOUNTS → DESEQ2 → MULTIQC                   │             │
│  │    │           │            │              │            │         │                     │             │
│  │   QC         Index        BAM          Counts       Results    Report                   │             │
│  └────────────────────────────────────────────────────────────────────────────────────────┘             │
│                                                                                                          │
│  ⚠️  ISSUE: Container images not validated before execution                                             │
│  ⚠️  ISSUE: No reference genome/index pre-check                                                         │
│  ⚠️  ISSUE: No resource estimation                                                                      │
│                                                                                                          │
│  FILES: src/workflow_composer/web/gradio_app.py (PipelineExecutor class)                                │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 7: MONITORING & RESULTS (gradio_app.py)                                                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  1. Monitoring thread polls SLURM status (squeue, sacct)                                                │
│  2. Parse Nextflow log for process progress                                                              │
│  3. Update UI via Gradio state refresh                                                                   │
│  4. Display: progress bar, current process, completed processes                                          │
│  5. On completion: show results location, download links                                                 │
│                                                                                                          │
│  UI Updates:                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────┐            │
│  │  Status: 🔵 RUNNING                                                                      │            │
│  │  Progress: [████████████░░░░░░░░] 60%                                                    │            │
│  │  Current: STAR_ALIGN                                                                     │            │
│  │  Completed: ✅ FASTQC, ✅ STAR_INDEX                                                      │            │
│  │  Runtime: 00:45:23                                                                       │            │
│  └─────────────────────────────────────────────────────────────────────────────────────────┘            │
│                                                                                                          │
│  ⚠️  ISSUE: No intermediate result preview                                                              │
│  ⚠️  ISSUE: No email/notification on completion                                                          │
│  ⚠️  ISSUE: No cost estimation for cloud resources                                                       │
│                                                                                                          │
│  FILES: src/workflow_composer/web/gradio_app.py (_monitor_job, _parse_nextflow_log)                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

```

---

## 🔴 Critical Gaps Identified

### Gap 1: No Container/Tool Availability Validation

**Current Behavior:**
- Workflow generates assuming all tools exist
- Fails at runtime if container missing

**What's Needed:**
```
Query → Parse → Tool Selection → ⚠️ VALIDATE AVAILABILITY → Module Mapping
                                        │
                                        ├── Container exists? (Singularity pull test)
                                        ├── Tool version compatible?
                                        └── Dependencies installed?
```

### Gap 2: No Module Dependency Resolution

**Current Behavior:**
- Each module treated independently
- No check for shared dependencies (e.g., genome index)

**What's Needed:**
```
Modules Required:
├── STAR_ALIGN (needs: STAR_INDEX output)
├── STAR_INDEX (needs: genome.fa)
└── Check: Is genome.fa available? Is index built?
```

### Gap 3: No Dynamic Container Building

**Current Behavior:**
- Static container images
- User must manually request tools not in containers

**What's Needed:**
```
Missing Tool Detected → Auto-build Container Layer → Cache for Future Use → Cleanup Policy
```

### Gap 4: No Reference Data Management

**Current Behavior:**
- Assumes user provides all reference data
- No automated download/indexing

**What's Needed:**
```
Organism: mouse, Build: GRCm39
├── Genome FASTA: /references/GRCm39/genome.fa ✅ EXISTS
├── GTF Annotation: /references/GRCm39/annotation.gtf ✅ EXISTS
├── STAR Index: /references/GRCm39/star/ ❌ MISSING → BUILD
└── Estimated Build Time: ~2 hours
```

### Gap 5: No Resource Estimation

**Current Behavior:**
- Fixed SLURM resources (8G RAM, 2 CPUs)
- No scaling based on data size

**What's Needed:**
```
Input: 20 samples × 50M reads each
├── Estimated Runtime: 8 hours
├── Recommended Memory: 64GB for STAR
├── Recommended CPUs: 16 per sample
├── Estimated Cost: $45 (AWS) / $32 (GCP)
└── Auto-configure SLURM resources
```

---

## 🚀 Proposed Architecture Improvements

### 1. Pre-flight Validation System

```python
class PreflightValidator:
    """Validate everything before workflow generation."""
    
    def validate(self, intent: ParsedIntent) -> ValidationReport:
        return ValidationReport(
            tools_available=self._check_tools(intent),
            containers_ready=self._check_containers(intent),
            references_available=self._check_references(intent),
            modules_exist=self._check_modules(intent),
            estimated_resources=self._estimate_resources(intent),
            missing_items=self._get_missing(),
            auto_fixable=self._can_auto_fix(),
        )
    
    def auto_fix(self, report: ValidationReport) -> bool:
        """Automatically resolve missing dependencies."""
        for item in report.missing_items:
            if item.auto_fixable:
                self._fix_item(item)
        return True
```

### 2. Dynamic Container Manager

```python
class ContainerManager:
    """Manage container images with on-demand tool installation."""
    
    def ensure_tool_available(self, tool: str, version: str = None) -> str:
        """Ensure tool is available, building if necessary."""
        
        # Check if tool exists in base container
        if self._tool_in_container(tool, "base"):
            return self._get_container_uri("base")
        
        # Check specialized container
        container = TOOL_CONTAINER_MAP.get(tool)
        if container and self._container_exists(container):
            return self._get_container_uri(container)
        
        # Build microservice container with just this tool
        return self._build_tool_container(tool, version)
    
    def _build_tool_container(self, tool: str, version: str) -> str:
        """Build lightweight container with specific tool."""
        dockerfile = self._generate_dockerfile(tool, version)
        image_name = f"biopipelines/{tool}:{version or 'latest'}"
        
        # Build and push to registry
        self._build_image(dockerfile, image_name)
        
        # Schedule cleanup after TTL
        self._schedule_cleanup(image_name, ttl_days=7)
        
        return image_name
```

### 3. Reference Data Manager

```python
class ReferenceDataManager:
    """Manage reference genomes, indexes, and annotations."""
    
    REFERENCE_REGISTRY = {
        "human": {
            "GRCh38": {
                "genome": "gs://biopipelines/references/human/GRCh38/genome.fa",
                "gtf": "gs://biopipelines/references/human/GRCh38/annotation.gtf",
                "star_index": "gs://biopipelines/references/human/GRCh38/star/",
            }
        },
        "mouse": {
            "GRCm39": {...}
        }
    }
    
    def ensure_references(self, organism: str, build: str, tools: List[str]) -> Dict:
        """Ensure all required references are available."""
        required = self._get_required_references(organism, build, tools)
        
        status = {}
        for ref_type, ref_path in required.items():
            if self._exists(ref_path):
                status[ref_type] = {"status": "ready", "path": ref_path}
            elif self._can_download(ref_type, organism, build):
                status[ref_type] = {
                    "status": "downloadable",
                    "source": self._get_download_source(ref_type, organism, build),
                    "estimated_time": self._estimate_download_time(ref_type)
                }
            else:
                status[ref_type] = {"status": "missing", "required_action": "manual"}
        
        return status
```

### 4. Intelligent Resource Estimator

```python
class ResourceEstimator:
    """Estimate computational resources based on analysis and data."""
    
    # Resource profiles per tool (per sample)
    TOOL_PROFILES = {
        "star": {"memory_gb": 32, "cpus": 8, "time_hours": 0.5},
        "featurecounts": {"memory_gb": 4, "cpus": 4, "time_hours": 0.1},
        "deseq2": {"memory_gb": 8, "cpus": 2, "time_hours": 0.2},
        "cellranger": {"memory_gb": 64, "cpus": 16, "time_hours": 2.0},
    }
    
    def estimate(self, intent: ParsedIntent, tools: List[str], 
                 sample_count: int, reads_per_sample: int) -> ResourceEstimate:
        """Estimate total resources needed."""
        
        total_memory = 0
        total_time = 0
        max_cpus = 0
        
        for tool in tools:
            profile = self.TOOL_PROFILES.get(tool, self.DEFAULT_PROFILE)
            total_memory = max(total_memory, profile["memory_gb"])
            total_time += profile["time_hours"] * sample_count
            max_cpus = max(max_cpus, profile["cpus"])
        
        # Adjust for parallelism
        if sample_count > 1:
            parallel_factor = min(sample_count, 10)  # Max 10 parallel
            total_time = total_time / parallel_factor
        
        return ResourceEstimate(
            memory_gb=total_memory,
            cpus=max_cpus,
            estimated_hours=total_time,
            recommended_partition=self._recommend_partition(total_memory, max_cpus),
            cost_estimate=self._estimate_cost(total_memory, max_cpus, total_time),
        )
```

### 5. Enhanced Workflow Execution Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    ENHANCED SYSTEM FLOW                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

                                        USER QUERY
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: INTELLIGENT PARSING                                                                           │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐   ┌─────────────────┐   ┌─────────────────┐   ┌──────────────┐                       │
│  │ Rule-based   │──►│ Biomedical NER  │──►│ LLM Validation  │──►│ Confidence   │                       │
│  │ Keywords     │   │ (BiomedBERT)    │   │ (BioMistral)    │   │ Merger       │                       │
│  └──────────────┘   └─────────────────┘   └─────────────────┘   └──────────────┘                       │
│                                                                                                          │
│  Output: High-confidence ParsedIntent with entities                                                      │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: PRE-FLIGHT VALIDATION (NEW)                                                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  VALIDATION CHECKLIST                                                                            │   │
│  ├─────────────────────────────────────────────────────────────────────────────────────────────────┤   │
│  │  ✅ Tools:       star (v2.7.10b), featurecounts (v2.0.3), deseq2 (v1.38.0)                      │   │
│  │  ✅ Containers:  rnaseq:latest (pulled 2024-01-15)                                               │   │
│  │  ⚠️  References: GRCm39 genome ✅, STAR index ❌ (will build, ~2h)                               │   │
│  │  ✅ Modules:     All 5 modules available                                                         │   │
│  │  ℹ️  Resources:  Est. 32GB RAM, 8 CPUs, ~4 hours                                                 │   │
│  │  💰 Cost Est:    ~$12 (GCP), ~$15 (AWS)                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                          │
│  User Approval: [✓] Proceed with missing STAR index build?  [Build & Run] [Cancel]                     │
│                                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: DEPENDENCY RESOLUTION (NEW)                                                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                          │
│  Dependency Graph:                                                                                       │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                                                 │    │
│  │  DOWNLOAD_GENOME ──► BUILD_STAR_INDEX ──┬──► STAR_ALIGN ──► FEATURECOUNTS ──► DESEQ2          │    │
│  │        │                                 │         ▲              ▲              ▲             │    │
│  │        │                                 │         │              │              │             │    │
│  │        └─────────────────────────────────┴─────────┴──────────────┴──────────────┘             │    │
│  │                            (genome.fa used by multiple steps)                                   │    │
│  │                                                                                                 │    │
│  │  Parallel Execution Plan:                                                                       │    │
│  │  T=0:  DOWNLOAD_GENOME, FASTQC (parallel)                                                       │    │
│  │  T=1:  BUILD_STAR_INDEX (after genome ready)                                                    │    │
│  │  T=2:  STAR_ALIGN × N samples (parallel, after index)                                           │    │
│  │  T=3:  FEATURECOUNTS × N samples (parallel)                                                     │    │
│  │  T=4:  DESEQ2 (after all counts ready)                                                          │    │
│  │                                                                                                 │    │
│  └────────────────────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: SMART EXECUTION                                                                               │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                          │
│  Execution Modes:                                                                                        │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────┐    │
│  │  [x] SLURM Cluster (default)                                                                    │    │
│  │  [ ] Local (testing)                                                                            │    │
│  │  [ ] AWS Batch (cloud)                                                                          │    │
│  │  [ ] Google Cloud Life Sciences (cloud)                                                         │    │
│  └────────────────────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                                          │
│  Resource Auto-Scaling:                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────┐    │
│  │  Process        │ Memory  │ CPUs │ Time   │ Partition    │ Instances                          │    │
│  │  ───────────────┼─────────┼──────┼────────┼──────────────┼────────────                        │    │
│  │  STAR_ALIGN     │ 32 GB   │ 8    │ 30 min │ cpuspot      │ 10 parallel                        │    │
│  │  FEATURECOUNTS  │ 4 GB    │ 4    │ 5 min  │ cpuspot      │ 10 parallel                        │    │
│  │  DESEQ2         │ 8 GB    │ 2    │ 10 min │ cpuspot      │ 1                                  │    │
│  └────────────────────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 5: LIVE MONITORING & NOTIFICATIONS                                                               │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                          │
│  Real-time Dashboard:                                                                                    │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                                                 │    │
│  │  🧬 RNA-seq Pipeline: rnaseq_mouse_star_deseq2                                                  │    │
│  │  ─────────────────────────────────────────────────────────────────────────────────────────     │    │
│  │                                                                                                 │    │
│  │  Overall Progress: [████████████████░░░░░░░░░░░░] 65%                                           │    │
│  │                                                                                                 │    │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────┐       │    │
│  │  │ Process          │ Status      │ Progress │ Time    │ Resources │ Cost             │       │    │
│  │  │ ─────────────────┼─────────────┼──────────┼─────────┼───────────┼────────          │       │    │
│  │  │ FASTQC           │ ✅ Complete │ 10/10    │ 5:23    │ 4GB/2CPU  │ $0.15            │       │    │
│  │  │ STAR_INDEX       │ ✅ Complete │ 1/1      │ 1:45:00 │ 32GB/8CPU │ $2.10            │       │    │
│  │  │ STAR_ALIGN       │ 🔵 Running  │ 7/10     │ 25:00   │ 32GB/8CPU │ $3.50            │       │    │
│  │  │ FEATURECOUNTS    │ ⏳ Pending  │ 0/10     │ -       │ 4GB/4CPU  │ -                │       │    │
│  │  │ DESEQ2           │ ⏳ Pending  │ 0/1      │ -       │ 8GB/2CPU  │ -                │       │    │
│  │  └─────────────────────────────────────────────────────────────────────────────────────┘       │    │
│  │                                                                                                 │    │
│  │  💰 Running Cost: $5.75 | Estimated Total: $8.50                                                │    │
│  │  ⏱️  Elapsed: 2:15:23 | Remaining: ~1:30:00                                                     │    │
│  │                                                                                                 │    │
│  │  [📊 View Intermediate Results] [📥 Download Logs] [🛑 Cancel] [⏸️ Pause]                        │    │
│  │                                                                                                 │    │
│  └────────────────────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                                          │
│  Notifications:                                                                                          │
│  ├── 📧 Email on: [x] Completion  [x] Failure  [ ] Each Process                                         │
│  ├── 💬 Slack:    [ ] Enable Slack notifications                                                         │
│  └── 🔔 Browser:  [x] Desktop notifications                                                              │
│                                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 6: RESULTS & ARTIFACTS                                                                           │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────┐    │
│  │  RESULTS SUMMARY                                                                                │    │
│  │  ─────────────────────────────────────────────────────────────────────────────────────────     │    │
│  │                                                                                                 │    │
│  │  📁 Output Directory: /results/rnaseq_mouse_star_20251125/                                      │    │
│  │                                                                                                 │    │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────┐       │    │
│  │  │ Artifact                    │ Size    │ Preview │ Download                          │       │    │
│  │  │ ────────────────────────────┼─────────┼─────────┼─────────                          │       │    │
│  │  │ QC Report (MultiQC)         │ 2.5 MB  │ [👁️]    │ [📥]                               │       │    │
│  │  │ Differential Expression     │ 15 MB   │ [👁️]    │ [📥]                               │       │    │
│  │  │ Count Matrix                │ 50 MB   │ [👁️]    │ [📥]                               │       │    │
│  │  │ Volcano Plot                │ 500 KB  │ [👁️]    │ [📥]                               │       │    │
│  │  │ Pipeline Report (HTML)      │ 1 MB    │ [👁️]    │ [📥]                               │       │    │
│  │  │ Execution Timeline          │ 200 KB  │ [👁️]    │ [📥]                               │       │    │
│  │  │ Full Results Archive        │ 2.1 GB  │ -       │ [📥 ZIP]                           │       │    │
│  │  └─────────────────────────────────────────────────────────────────────────────────────┘       │    │
│  │                                                                                                 │    │
│  │  📊 Quick Stats:                                                                                │    │
│  │  ├── 1,234 significantly differentially expressed genes (padj < 0.05)                          │    │
│  │  ├── 678 upregulated, 556 downregulated                                                         │    │
│  │  └── Top gene: Gapdh (log2FC: 3.2, padj: 1e-50)                                                 │    │
│  │                                                                                                 │    │
│  │  🔄 Reproducibility:                                                                            │    │
│  │  ├── [📋 Copy Nextflow Command]                                                                 │    │
│  │  ├── [📦 Export Environment]                                                                    │    │
│  │  └── [📤 Share Results (generate link)]                                                         │    │
│  │                                                                                                 │    │
│  └────────────────────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Implementation Priority

### Phase 1: Critical (Week 1-2)
1. **Pre-flight Validation System** - Prevent runtime failures
2. **Container Availability Check** - Validate before execution
3. **Reference Data Check** - Ensure genome/indexes exist

### Phase 2: Important (Week 3-4)
4. **Resource Estimator** - Right-size SLURM jobs
5. **Module Dependency Graph** - Proper execution order
6. **Enhanced Progress Monitoring** - Better UI feedback

### Phase 3: Nice-to-have (Week 5-6)
7. **Dynamic Container Building** - On-demand tool installation
8. **Notification System** - Email/Slack alerts
9. **Cost Estimation** - Cloud cost tracking

### Phase 4: Future (Month 2+)
10. **Result Preview** - Interactive plots
11. **Result Sharing** - Shareable links
12. **Auto-retry** - Failed job recovery
13. **Multi-cloud Support** - AWS/GCP execution

---

## 🔧 Files to Create/Modify

### New Files:
```
src/workflow_composer/core/
├── preflight_validator.py      # Pre-execution validation
├── container_manager.py        # Container image management
├── reference_manager.py        # Reference data management
├── resource_estimator.py       # Resource estimation
├── dependency_resolver.py      # Module dependency resolution
└── notification_service.py     # Alerts and notifications

src/workflow_composer/web/
├── components/
│   ├── validation_panel.py     # UI for validation results
│   ├── progress_dashboard.py   # Enhanced progress UI
│   └── results_viewer.py       # Results preview
```

### Modified Files:
```
src/workflow_composer/composer.py           # Add validation step
src/workflow_composer/web/gradio_app.py     # Enhanced UI
src/workflow_composer/core/workflow_generator.py  # Dependency-aware generation
```

---

## 📊 Metrics to Track

| Metric | Current | Target |
|--------|---------|--------|
| Workflow Generation Success Rate | ~70% | >95% |
| Average User Corrections Needed | 2-3 | 0-1 |
| Time to First Run | 2-5 min | <1 min |
| Pipeline Failure Rate (runtime) | ~30% | <5% |
| User Satisfaction | Unknown | >4.5/5 |

---

## Conclusion

The current system has a solid foundation but lacks critical validation and resource management features. The proposed improvements focus on:

1. **Preventing failures** through pre-flight validation
2. **Reducing surprises** with resource estimation and cost tracking
3. **Improving UX** with better progress monitoring and notifications
4. **Enabling self-service** through dynamic container building

Implementing these changes will transform BioPipelines from a prototype into a production-ready platform.
