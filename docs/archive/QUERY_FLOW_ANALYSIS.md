# Query Flow Analysis: BioPipelines Workflow Composer

## Executive Summary

This document provides a comprehensive analysis of the query processing flow in the BioPipelines Workflow Composer system. The system transforms natural language queries into executable Nextflow DSL2 workflows through a multi-stage pipeline with AI-assisted processing at key decision points.

---

## High-Level Architecture Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    USER QUERY                                        │
│                    "I want to do RNA-seq differential expression"                    │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                               WORKFLOW COMPOSER                                      │
│                            (Orchestrator: composer.py)                               │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │                         generate(query, config) → Workflow                      │ │
│  │                                                                                 │ │
│  │   ┌─────────┐   ┌─────────────┐   ┌─────────────┐   ┌──────────────────────┐   │ │
│  │   │ Step 1  │   │   Step 2    │   │   Step 3    │   │       Step 4         │   │ │
│  │   │  Parse  │──▶│   Select    │──▶│     Map     │──▶│      Generate        │   │ │
│  │   │ Intent  │   │   Tools     │   │   Modules   │   │      Workflow        │   │ │
│  │   └─────────┘   └─────────────┘   └─────────────┘   └──────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              GENERATED WORKFLOW                                      │
│                           (Nextflow DSL2 + Execution)                                │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Component Analysis

### 1. WORKFLOW COMPOSER (Orchestrator)

**File:** `src/workflow_composer/composer.py`

**Class:** `Composer`

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                   COMPOSER                                           │
│                                                                                      │
│  Initialization:                                                                     │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │  __init__(config_path)                                                        │   │
│  │    ├── Load configuration (composer.yaml)                                     │   │
│  │    ├── Initialize IntentParser                                                │   │
│  │    ├── Initialize ToolSelector (with tool catalog)                            │   │
│  │    ├── Initialize ModuleMapper (with Nextflow modules)                        │   │
│  │    └── Initialize WorkflowGenerator                                           │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
│  Main Entry Point:                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │  generate(query: str, config: Optional[Dict]) → Workflow                      │   │
│  │                                                                               │   │
│  │    1. intent = self.intent_parser.parse(query)                                │   │
│  │         └── Returns: ParsedIntent                                             │   │
│  │                                                                               │   │
│  │    2. tools = self.tool_selector.select_tools(intent)                         │   │
│  │         └── Returns: List[ToolMatch]                                          │   │
│  │                                                                               │   │
│  │    3. modules = self.module_mapper.map_tools(tools)                           │   │
│  │         └── Returns: List[NextflowModule]                                     │   │
│  │                                                                               │   │
│  │    4. workflow = self.workflow_generator.generate(intent, modules)            │   │
│  │         └── Returns: Workflow (DSL2 code + metadata)                          │   │
│  │                                                                               │   │
│  │    return workflow                                                            │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**Capabilities:**
- Configuration-driven initialization
- Clean orchestration of 4-step workflow
- Support for custom config overrides

**Current Limitations:**
- No async/parallel processing
- Limited error recovery between steps
- No caching of intermediate results

**Enhancement Opportunities:**
- Add step-level caching for repeated queries
- Implement async processing for tool selection
- Add telemetry/logging for performance monitoring

---

### 2. INTENT PARSER (Step 1)

**File:** `src/workflow_composer/core/query_parser.py`

**Class:** `IntentParser`

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                               INTENT PARSER                                          │
│                                                                                      │
│  Input: Natural Language Query                                                       │
│  Output: ParsedIntent dataclass                                                      │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                          PARSING FLOW                                         │   │
│  │                                                                               │   │
│  │   ┌─────────────────────────────────────────────────────────────────────┐    │   │
│  │   │                        RULE-BASED PARSING                            │    │   │
│  │   │                          (Primary Path)                              │    │   │
│  │   │  ┌─────────────────────────────────────────────────────────────┐    │    │   │
│  │   │  │  Keyword Patterns:                                           │    │    │   │
│  │   │  │    "rna-seq"     → AnalysisType.RNA_SEQ                      │    │    │   │
│  │   │  │    "chip-seq"    → AnalysisType.CHIP_SEQ                     │    │    │   │
│  │   │  │    "atac-seq"    → AnalysisType.ATAC_SEQ                     │    │    │   │
│  │   │  │    "variant"     → AnalysisType.VARIANT_CALLING              │    │    │   │
│  │   │  │    "methylation" → AnalysisType.METHYLATION                  │    │    │   │
│  │   │  │    "single.?cell"→ AnalysisType.SINGLE_CELL                  │    │    │   │
│  │   │  │    "metagenomics"→ AnalysisType.METAGENOMICS                 │    │    │   │
│  │   │  │    "hi-?c"       → AnalysisType.HIC                          │    │    │   │
│  │   │  └─────────────────────────────────────────────────────────────┘    │    │   │
│  │   │                                │                                     │    │   │
│  │   │                   ┌────────────┴────────────┐                       │    │   │
│  │   │                   │     Pattern Matched?     │                       │    │   │
│  │   │                   └────────────┬────────────┘                       │    │   │
│  │   │                          │           │                               │    │   │
│  │   │                         YES          NO                              │    │   │
│  │   │                          │           │                               │    │   │
│  │   │                          ▼           ▼                               │    │   │
│  │   │                  ┌───────────┐ ┌─────────────────┐                   │    │   │
│  │   │                  │  Extract  │ │   LLM FALLBACK  │                   │    │   │
│  │   │                  │ Parameters│ │  (Ambiguous)    │                   │    │   │
│  │   │                  └───────────┘ └─────────────────┘                   │    │   │
│  │   └─────────────────────────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                      PARAMETER EXTRACTION                                     │   │
│  │                                                                               │   │
│  │  Organism Detection:                                                          │   │
│  │    "human", "homo sapiens", "hg38", "grch38" → organism: "human"             │   │
│  │    "mouse", "mus musculus", "mm10", "grcm38" → organism: "mouse"             │   │
│  │    "zebrafish", "danio"                      → organism: "zebrafish"         │   │
│  │                                                                               │   │
│  │  Task Detection (Sub-analysis):                                               │   │
│  │    "differential expression" → tasks: ["differential_expression"]            │   │
│  │    "peak calling"            → tasks: ["peak_calling"]                        │   │
│  │    "variant calling"         → tasks: ["variant_calling"]                     │   │
│  │    "alignment", "mapping"    → tasks: ["alignment"]                           │   │
│  │                                                                               │   │
│  │  Options Extraction:                                                          │   │
│  │    "strand.*specific"        → options: {"stranded": true}                    │   │
│  │    "paired.*end"             → options: {"paired_end": true}                  │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
│  Output Structure:                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │  @dataclass                                                                   │   │
│  │  class ParsedIntent:                                                          │   │
│  │      analysis_type: AnalysisType      # RNA_SEQ, CHIP_SEQ, etc.              │   │
│  │      organism: Optional[str]          # "human", "mouse", etc.               │   │
│  │      tasks: List[str]                 # ["differential_expression"]          │   │
│  │      parameters: Dict[str, Any]       # {"stranded": True}                   │   │
│  │      raw_query: str                   # Original user query                  │   │
│  │      confidence: float                # 0.0-1.0 parsing confidence           │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**Capabilities:**
- Fast rule-based parsing for common patterns
- Hybrid approach: rules + LLM fallback
- Organism detection from multiple aliases
- Task/sub-analysis detection
- Confidence scoring

**Current Limitations:**
- Limited to predefined analysis types (8 types)
- No context memory between queries
- Single-organism per query
- Basic task extraction

**Enhancement Opportunities:**
1. **Multi-organism support**: Enable pipelines comparing multiple organisms
2. **Contextual parsing**: Remember previous queries for follow-up refinement
3. **Confidence thresholds**: Configurable confidence levels for LLM fallback
4. **Custom analysis types**: Allow user-defined analysis types via config
5. **Batch query parsing**: Parse multiple related queries at once

**Test Coverage Gaps:**
- Ambiguous queries triggering LLM fallback
- Edge cases with mixed analysis types
- Non-English organism names
- Queries with conflicting parameters

---

### 3. TOOL SELECTOR (Step 2)

**File:** `src/workflow_composer/core/tool_selector.py`

**Class:** `ToolSelector`

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                               TOOL SELECTOR                                          │
│                                                                                      │
│  Input: ParsedIntent                                                                 │
│  Output: List[ToolMatch]                                                             │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                        TOOL CATALOG STRUCTURE                                 │   │
│  │                                                                               │   │
│  │  Source: data/tool_catalog/*.json                                             │   │
│  │                                                                               │   │
│  │  @dataclass                                                                   │   │
│  │  class Tool:                                                                  │   │
│  │      name: str              # "star", "salmon", "bwa"                        │   │
│  │      description: str       # Human-readable description                      │   │
│  │      category: str          # "aligner", "quantifier", "caller"              │   │
│  │      analysis_types: List[str]  # ["rna-seq", "chip-seq"]                    │   │
│  │      inputs: List[str]      # ["fastq", "bam"]                               │   │
│  │      outputs: List[str]     # ["bam", "counts"]                              │   │
│  │      version: str           # "2.7.9a"                                       │   │
│  │      container: str         # "ghcr.io/biopipelines/rna-seq"                │   │
│  │      parameters: Dict       # Tool-specific parameters                        │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                          SELECTION FLOW                                       │   │
│  │                                                                               │   │
│  │   ┌─────────────────────────────────────────────────────────────────────┐    │   │
│  │   │                    1. CATEGORY FILTERING                             │    │   │
│  │   │                                                                      │    │   │
│  │   │   ParsedIntent.analysis_type → Filter tools by analysis_types[]     │    │   │
│  │   │                                                                      │    │   │
│  │   │   Example:                                                           │    │   │
│  │   │   analysis_type=RNA_SEQ → ["star","hisat2","salmon","featurecounts"]│    │   │
│  │   └─────────────────────────────────────────────────────────────────────┘    │   │
│  │                                    │                                          │   │
│  │                                    ▼                                          │   │
│  │   ┌─────────────────────────────────────────────────────────────────────┐    │   │
│  │   │                    2. TASK MATCHING                                  │    │   │
│  │   │                                                                      │    │   │
│  │   │   ParsedIntent.tasks → Match to tool categories                      │    │   │
│  │   │                                                                      │    │   │
│  │   │   "differential_expression" → ["deseq2", "edger", "limma"]          │    │   │
│  │   │   "alignment"               → ["star", "hisat2", "bwa"]             │    │   │
│  │   │   "quantification"          → ["salmon", "kallisto", "rsem"]        │    │   │
│  │   └─────────────────────────────────────────────────────────────────────┘    │   │
│  │                                    │                                          │   │
│  │                                    ▼                                          │   │
│  │   ┌─────────────────────────────────────────────────────────────────────┐    │   │
│  │   │                    3. FUZZY MATCHING                                 │    │   │
│  │   │                                                                      │    │   │
│  │   │   Uses rapidfuzz for similarity scoring:                             │    │   │
│  │   │   - Tool name matching                                               │    │   │
│  │   │   - Description matching                                             │    │   │
│  │   │   - Alias matching                                                   │    │   │
│  │   │                                                                      │    │   │
│  │   │   Threshold: score >= 80 considered a match                          │    │   │
│  │   └─────────────────────────────────────────────────────────────────────┘    │   │
│  │                                    │                                          │   │
│  │                                    ▼                                          │   │
│  │   ┌─────────────────────────────────────────────────────────────────────┐    │   │
│  │   │                    4. PIPELINE CONSTRUCTION                          │    │   │
│  │   │                                                                      │    │   │
│  │   │   Order tools by execution sequence:                                 │    │   │
│  │   │                                                                      │    │   │
│  │   │   RNA-seq example:                                                   │    │   │
│  │   │   1. QC: fastqc                                                      │    │   │
│  │   │   2. Trimming: trimmomatic/fastp                                    │    │   │
│  │   │   3. Alignment: star/hisat2                                         │    │   │
│  │   │   4. Quantification: featurecounts/salmon                           │    │   │
│  │   │   5. Analysis: deseq2/edger                                         │    │   │
│  │   │   6. Report: multiqc                                                │    │   │
│  │   └─────────────────────────────────────────────────────────────────────┘    │   │
│  │                                    │                                          │   │
│  │                                    ▼                                          │   │
│  │   ┌─────────────────────────────────────────────────────────────────────┐    │   │
│  │   │                    5. ALTERNATIVE SUGGESTIONS                        │    │   │
│  │   │                                                                      │    │   │
│  │   │   For each selected tool, provide alternatives:                      │    │   │
│  │   │                                                                      │    │   │
│  │   │   star → alternatives: ["hisat2", "bwa"]                            │    │   │
│  │   │   salmon → alternatives: ["kallisto", "rsem"]                       │    │   │
│  │   │                                                                      │    │   │
│  │   │   @dataclass                                                         │    │   │
│  │   │   class ToolMatch:                                                   │    │   │
│  │   │       tool: Tool                                                     │    │   │
│  │   │       confidence: float           # Match confidence                 │    │   │
│  │   │       alternatives: List[Tool]    # Alternative tools               │    │   │
│  │   │       step_order: int             # Position in pipeline            │    │   │
│  │   └─────────────────────────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**Capabilities:**
- JSON-based tool catalog (extensible)
- Fuzzy matching for user-friendly queries
- Category-based filtering
- Alternative tool suggestions
- Automatic pipeline ordering

**Current Limitations:**
- Static tool catalog (manual updates required)
- No tool version negotiation
- Limited dependency resolution
- No tool compatibility checking

**Enhancement Opportunities:**
1. **Dynamic catalog updates**: Fetch latest tool versions from bioconda
2. **Tool compatibility matrix**: Track known conflicts between tools
3. **User preference learning**: Remember user's tool preferences
4. **Performance-based ranking**: Rank tools by benchmark performance
5. **Container pre-pull**: Trigger container downloads early

**Test Coverage Gaps:**
- Tool version conflicts
- Missing tools in catalog
- Circular dependencies
- Custom tool definitions

---

### 4. MODULE MAPPER (Step 3)

**File:** `src/workflow_composer/core/module_mapper.py`

**Class:** `ModuleMapper`

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                               MODULE MAPPER                                          │
│                                                                                      │
│  Input: List[ToolMatch]                                                              │
│  Output: List[NextflowModule]                                                        │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                       MODULE DISCOVERY                                        │   │
│  │                                                                               │   │
│  │   Scans: nextflow-modules/                                                    │   │
│  │                                                                               │   │
│  │   Directory Structure:                                                        │   │
│  │   nextflow-modules/                                                           │   │
│  │   ├── alignment/                                                              │   │
│  │   │   ├── star.nf                                                            │   │
│  │   │   ├── hisat2.nf                                                          │   │
│  │   │   └── bwa.nf                                                             │   │
│  │   ├── quantification/                                                         │   │
│  │   │   ├── salmon.nf                                                          │   │
│  │   │   └── featurecounts.nf                                                   │   │
│  │   ├── quality_control/                                                        │   │
│  │   │   ├── fastqc.nf                                                          │   │
│  │   │   └── multiqc.nf                                                         │   │
│  │   └── ...                                                                     │   │
│  │                                                                               │   │
│  │   @dataclass                                                                  │   │
│  │   class NextflowModule:                                                       │   │
│  │       name: str              # "STAR"                                        │   │
│  │       path: Path             # nextflow-modules/alignment/star.nf            │   │
│  │       inputs: List[str]      # ["reads", "index"]                            │   │
│  │       outputs: List[str]     # ["bam", "log"]                                │   │
│  │       params: Dict           # Module parameters                              │   │
│  │       container: str         # Container image reference                      │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                         MAPPING FLOW                                          │   │
│  │                                                                               │   │
│  │   For each ToolMatch:                                                         │   │
│  │   ┌─────────────────────────────────────────────────────────────────────┐    │   │
│  │   │  1. Search existing modules by tool name                             │    │   │
│  │   │     modules[tool.name.lower()] → NextflowModule?                     │    │   │
│  │   └─────────────────────────────────────────────────────────────────────┘    │   │
│  │                         │                                                     │   │
│  │              ┌──────────┴──────────┐                                         │   │
│  │              │    Module Found?     │                                         │   │
│  │              └──────────┬──────────┘                                         │   │
│  │                   │           │                                               │   │
│  │                  YES          NO                                              │   │
│  │                   │           │                                               │   │
│  │                   ▼           ▼                                               │   │
│  │   ┌─────────────────┐  ┌──────────────────────────────────────────────┐     │   │
│  │   │  Return module  │  │           AUTO-GENERATE MODULE               │     │   │
│  │   │   (validated)   │  │                                              │     │   │
│  │   └─────────────────┘  │  1. Generate template via LLM                │     │   │
│  │                        │  2. Add container from TOOL_CONTAINER_MAP    │     │   │
│  │                        │  3. Write to nextflow-modules/               │     │   │
│  │                        │  4. Return new module                        │     │   │
│  │                        └──────────────────────────────────────────────┘     │   │
│  │                                                                               │   │
│  │   ┌─────────────────────────────────────────────────────────────────────┐    │   │
│  │   │  2. Validate I/O Compatibility                                       │    │   │
│  │   │     Check: tool.outputs[i] connects to tool.inputs[i+1]              │    │   │
│  │   │                                                                      │    │   │
│  │   │     FASTQC.outputs["fastq"] → STAR.inputs["reads"] ✓                │    │   │
│  │   │     STAR.outputs["bam"] → FEATURECOUNTS.inputs["bam"] ✓             │    │   │
│  │   └─────────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                               │   │
│  │   ┌─────────────────────────────────────────────────────────────────────┐    │   │
│  │   │  3. Container Resolution                                             │    │   │
│  │   │                                                                      │    │   │
│  │   │     TOOL_CONTAINER_MAP:                                              │    │   │
│  │   │       "star" → "rna-seq"                                            │    │   │
│  │   │       "bwa"  → "dna-seq"                                            │    │   │
│  │   │       "macs2"→ "chip-seq"                                           │    │   │
│  │   │                                                                      │    │   │
│  │   │     Container format:                                                │    │   │
│  │   │       ghcr.io/biopipelines/{category}:{version}                     │    │   │
│  │   └─────────────────────────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**Capabilities:**
- Automatic module discovery from filesystem
- LLM-assisted module generation for missing tools
- I/O compatibility validation
- Container resolution via mapping

**Current Limitations:**
- Single module per tool (no variants)
- Limited module versioning
- No module testing/validation
- Manual container mapping maintenance

**Enhancement Opportunities:**
1. **Module versioning**: Support multiple versions of same tool
2. **Module validation**: Automated syntax checking
3. **Subworkflow support**: Combine modules into reusable subworkflows
4. **Dynamic container lookup**: Query container registries directly
5. **Module templating**: Support for parameterized module variants

**Test Coverage Gaps:**
- Module generation via LLM
- I/O validation failures
- Missing container mappings
- Module parsing edge cases

---

### 5. WORKFLOW GENERATOR (Step 4)

**File:** `src/workflow_composer/core/workflow_generator.py`

**Class:** `WorkflowGenerator`

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            WORKFLOW GENERATOR                                        │
│                                                                                      │
│  Input: ParsedIntent + List[NextflowModule]                                          │
│  Output: Workflow (DSL2 code + metadata + execution artifacts)                       │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                        GENERATION FLOW                                        │   │
│  │                                                                               │   │
│  │   ┌─────────────────────────────────────────────────────────────────────┐    │   │
│  │   │                    1. TEMPLATE SELECTION                             │    │   │
│  │   │                                                                      │    │   │
│  │   │   Based on intent.analysis_type, select base template:               │    │   │
│  │   │                                                                      │    │   │
│  │   │   WORKFLOW_TEMPLATES = {                                             │    │   │
│  │   │       AnalysisType.RNA_SEQ: "templates/rnaseq_base.nf",             │    │   │
│  │   │       AnalysisType.CHIP_SEQ: "templates/chipseq_base.nf",           │    │   │
│  │   │       AnalysisType.VARIANT_CALLING: "templates/variant_base.nf",    │    │   │
│  │   │       ...                                                            │    │   │
│  │   │   }                                                                  │    │   │
│  │   └─────────────────────────────────────────────────────────────────────┘    │   │
│  │                                    │                                          │   │
│  │                                    ▼                                          │   │
│  │   ┌─────────────────────────────────────────────────────────────────────┐    │   │
│  │   │                    2. MODULE INCLUSION                               │    │   │
│  │   │                                                                      │    │   │
│  │   │   For each NextflowModule:                                           │    │   │
│  │   │     - Add include statement: include { STAR } from './modules/...'  │    │   │
│  │   │     - Wire inputs/outputs to channels                                │    │   │
│  │   │     - Configure container directive                                  │    │   │
│  │   │                                                                      │    │   │
│  │   │   Example include block:                                             │    │   │
│  │   │   ```                                                                │    │   │
│  │   │   include { FASTQC } from './modules/quality_control/fastqc'        │    │   │
│  │   │   include { STAR } from './modules/alignment/star'                  │    │   │
│  │   │   include { FEATURECOUNTS } from './modules/quant/featurecounts'    │    │   │
│  │   │   ```                                                                │    │   │
│  │   └─────────────────────────────────────────────────────────────────────┘    │   │
│  │                                    │                                          │   │
│  │                                    ▼                                          │   │
│  │   ┌─────────────────────────────────────────────────────────────────────┐    │   │
│  │   │                    3. CHANNEL WIRING                                 │    │   │
│  │   │                                                                      │    │   │
│  │   │   Generate main workflow block:                                      │    │   │
│  │   │                                                                      │    │   │
│  │   │   ```nextflow                                                        │    │   │
│  │   │   workflow {                                                         │    │   │
│  │   │       // Input channels                                              │    │   │
│  │   │       reads_ch = Channel.fromFilePairs(params.reads)                │    │   │
│  │   │                                                                      │    │   │
│  │   │       // Process chain                                               │    │   │
│  │   │       FASTQC(reads_ch)                                              │    │   │
│  │   │       STAR(reads_ch, params.index)                                  │    │   │
│  │   │       FEATURECOUNTS(STAR.out.bam, params.gtf)                       │    │   │
│  │   │       MULTIQC(FASTQC.out.mix(STAR.out.log))                         │    │   │
│  │   │   }                                                                  │    │   │
│  │   │   ```                                                                │    │   │
│  │   └─────────────────────────────────────────────────────────────────────┘    │   │
│  │                                    │                                          │   │
│  │                                    ▼                                          │   │
│  │   ┌─────────────────────────────────────────────────────────────────────┐    │   │
│  │   │                    4. PARAMETER CONFIGURATION                        │    │   │
│  │   │                                                                      │    │   │
│  │   │   Generate nextflow.config:                                          │    │   │
│  │   │   - Default parameters from tool catalog                             │    │   │
│  │   │   - User-specified overrides from intent.parameters                  │    │   │
│  │   │   - Resource profiles (HPC, cloud, local)                            │    │   │
│  │   │   - Container configurations                                         │    │   │
│  │   │                                                                      │    │   │
│  │   │   ```                                                                │    │   │
│  │   │   params {                                                           │    │   │
│  │   │       reads = './data/*_{R1,R2}.fastq.gz'                           │    │   │
│  │   │       outdir = './results'                                          │    │   │
│  │   │       genome = 'GRCh38'                                             │    │   │
│  │   │   }                                                                  │    │   │
│  │   │                                                                      │    │   │
│  │   │   process {                                                          │    │   │
│  │   │       withLabel: 'high_memory' { memory = '32 GB' }                 │    │   │
│  │   │   }                                                                  │    │   │
│  │   │   ```                                                                │    │   │
│  │   └─────────────────────────────────────────────────────────────────────┘    │   │
│  │                                    │                                          │   │
│  │                                    ▼                                          │   │
│  │   ┌─────────────────────────────────────────────────────────────────────┐    │   │
│  │   │                    5. LLM CUSTOMIZATION (Optional)                   │    │   │
│  │   │                                                                      │    │   │
│  │   │   If query has specific requirements not covered by templates:       │    │   │
│  │   │   - Use LLM to customize workflow logic                              │    │   │
│  │   │   - Generate custom process blocks                                   │    │   │
│  │   │   - Add conditional execution paths                                  │    │   │
│  │   │                                                                      │    │   │
│  │   │   Prompt: "Generate a Nextflow DSL2 workflow for {analysis} with:   │    │   │
│  │   │           - Modules: {module_list}                                   │    │   │
│  │   │           - Requirements: {custom_requirements}"                     │    │   │
│  │   └─────────────────────────────────────────────────────────────────────┘    │   │
│  │                                    │                                          │   │
│  │                                    ▼                                          │   │
│  │   ┌─────────────────────────────────────────────────────────────────────┐    │   │
│  │   │                    6. OUTPUT PACKAGING                               │    │   │
│  │   │                                                                      │    │   │
│  │   │   @dataclass                                                         │    │   │
│  │   │   class Workflow:                                                    │    │   │
│  │   │       name: str                # "rnaseq_20231126_143022"           │    │   │
│  │   │       main_script: str         # main.nf content                    │    │   │
│  │   │       config: str              # nextflow.config content            │    │   │
│  │   │       modules: List[str]       # Module file contents               │    │   │
│  │   │       output_dir: Path         # generated_workflows/{name}/        │    │   │
│  │   │       metadata: Dict           # Provenance, intent, timestamps     │    │   │
│  │   │                                                                      │    │   │
│  │   │   Directory structure:                                               │    │   │
│  │   │   generated_workflows/{name}/                                        │    │   │
│  │   │   ├── main.nf                                                        │    │   │
│  │   │   ├── nextflow.config                                                │    │   │
│  │   │   ├── modules/                                                       │    │   │
│  │   │   │   ├── fastqc.nf                                                  │    │   │
│  │   │   │   ├── star.nf                                                    │    │   │
│  │   │   │   └── ...                                                        │    │   │
│  │   │   ├── params.yaml                                                    │    │   │
│  │   │   └── metadata.json                                                  │    │   │
│  │   └─────────────────────────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**Capabilities:**
- Template-based workflow generation
- Automatic module inclusion and wiring
- Parameter configuration generation
- LLM customization for complex requirements
- Complete workflow packaging

**Current Limitations:**
- Limited template library
- No subworkflow nesting
- Basic error handling in generated code
- No resume/checkpoint configuration

**Enhancement Opportunities:**
1. **Template library expansion**: More analysis-specific templates
2. **Subworkflow support**: Nested workflows for complex analyses
3. **Error handling**: Add try-catch blocks and retry logic
4. **Resume configuration**: Auto-configure Nextflow resume
5. **Validation**: Syntax check generated workflows before output
6. **Documentation generation**: Auto-generate README for workflows

**Test Coverage Gaps:**
- Complex channel wiring
- Multi-output processes
- Conditional execution paths
- Resource directive generation

---

### 6. PREFLIGHT VALIDATOR (Pre-Execution Check)

**File:** `src/workflow_composer/core/preflight_validator.py`

**Class:** `PreflightValidator`

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           PREFLIGHT VALIDATOR                                        │
│                                                                                      │
│  Input: ParsedIntent + List[ToolMatch]                                               │
│  Output: ValidationReport                                                            │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                        VALIDATION CHECKS                                      │   │
│  │                                                                               │   │
│  │   ┌─────────────────────────────────────────────────────────────────────┐    │   │
│  │   │  1. TOOL AVAILABILITY                                                │    │   │
│  │   │     Check: Tool exists in container image                            │    │   │
│  │   │     Status: READY / MISSING / BUILDABLE                              │    │   │
│  │   └─────────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                               │   │
│  │   ┌─────────────────────────────────────────────────────────────────────┐    │   │
│  │   │  2. CONTAINER EXISTENCE                                              │    │   │
│  │   │     Check: Container image pullable from registry                    │    │   │
│  │   │     Status: READY / DOWNLOADABLE / BUILDABLE                         │    │   │
│  │   └─────────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                               │   │
│  │   ┌─────────────────────────────────────────────────────────────────────┐    │   │
│  │   │  3. REFERENCE DATA                                                   │    │   │
│  │   │     Check: Required reference files available                        │    │   │
│  │   │     (genome, annotations, indices)                                   │    │   │
│  │   │     Status: READY / DOWNLOADABLE / REQUIRES_MANUAL                   │    │   │
│  │   └─────────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                               │   │
│  │   ┌─────────────────────────────────────────────────────────────────────┐    │   │
│  │   │  4. MODULE EXISTENCE                                                 │    │   │
│  │   │     Check: Nextflow module files present                             │    │   │
│  │   │     Status: READY / BUILDABLE (auto-generate)                        │    │   │
│  │   └─────────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                               │   │
│  │   ┌─────────────────────────────────────────────────────────────────────┐    │   │
│  │   │  5. RESOURCE ESTIMATION                                              │    │   │
│  │   │     Calculate: Memory, CPUs, runtime, cost                           │    │   │
│  │   │     Output: ResourceEstimate dataclass                               │    │   │
│  │   │                                                                      │    │   │
│  │   │     TOOL_RESOURCE_PROFILES:                                          │    │   │
│  │   │       star:        32GB RAM, 8 CPU, 0.5h/sample                      │    │   │
│  │   │       cellranger:  64GB RAM, 16 CPU, 4.0h/sample                     │    │   │
│  │   │       salmon:      8GB RAM, 8 CPU, 0.1h/sample                       │    │   │
│  │   └─────────────────────────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                         OUTPUT STRUCTURE                                      │   │
│  │                                                                               │   │
│  │   @dataclass                                                                  │   │
│  │   class ValidationReport:                                                     │   │
│  │       can_proceed: bool           # All checks passed                        │   │
│  │       auto_fixable: bool          # Issues can be auto-resolved             │   │
│  │       items: List[ValidationItem] # All validated items                      │   │
│  │       missing_items: List[...]    # Items that need attention               │   │
│  │       warnings: List[str]         # Non-blocking warnings                   │   │
│  │       resources: ResourceEstimate # Estimated requirements                  │   │
│  │       fix_time_total: str         # Total time to fix issues                │   │
│  │                                                                               │   │
│  │   Methods:                                                                    │   │
│  │       to_dict() → JSON-serializable dictionary                               │   │
│  │       to_markdown() → Human-readable report                                  │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**Capabilities:**
- Comprehensive pre-execution validation
- Resource estimation for HPC scheduling
- Auto-fix suggestions for common issues
- Multiple output formats (JSON, Markdown)

**Current Limitations:**
- Limited to predefined resource profiles
- No actual container pull validation
- Static reference requirements

**Enhancement Opportunities:**
1. **Dynamic resource profiling**: Learn from past executions
2. **Container registry integration**: Actually verify images exist
3. **Reference data indexing**: Check index compatibility
4. **Cost estimation**: Cloud cost calculator integration

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                      COMPLETE QUERY FLOW                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

User Query: "I want to analyze RNA-seq data from human samples with differential expression"
                                                    │
                                                    ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: INTENT PARSING (IntentParser)                                                          │
│                                                                                                │
│ Input:  "I want to analyze RNA-seq data from human samples with differential expression"      │
│                                                                                                │
│ Processing:                                                                                    │
│   ├── Pattern match: "rna-seq" → AnalysisType.RNA_SEQ ✓                                       │
│   ├── Pattern match: "human" → organism: "human" ✓                                            │
│   ├── Pattern match: "differential expression" → tasks: ["differential_expression"] ✓         │
│   └── Confidence: 0.95 (high, no LLM needed)                                                  │
│                                                                                                │
│ Output: ParsedIntent(                                                                          │
│           analysis_type=RNA_SEQ,                                                               │
│           organism="human",                                                                    │
│           tasks=["differential_expression"],                                                   │
│           parameters={},                                                                       │
│           confidence=0.95                                                                      │
│         )                                                                                      │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: TOOL SELECTION (ToolSelector)                                                          │
│                                                                                                │
│ Input: ParsedIntent (analysis_type=RNA_SEQ, tasks=["differential_expression"])                │
│                                                                                                │
│ Processing:                                                                                    │
│   ├── Filter by analysis_type: [star, hisat2, salmon, kallisto, featurecounts, deseq2, ...]  │
│   ├── Match tasks to categories:                                                              │
│   │     └── "differential_expression" → Include deseq2/edger                                  │
│   ├── Build pipeline order:                                                                   │
│   │     1. fastqc (QC)                                                                        │
│   │     2. star (alignment)                                                                   │
│   │     3. featurecounts (quantification)                                                     │
│   │     4. deseq2 (differential expression)                                                   │
│   │     5. multiqc (reporting)                                                                │
│   └── Add alternatives for each tool                                                          │
│                                                                                                │
│ Output: [                                                                                      │
│   ToolMatch(tool=fastqc, step=1, alternatives=[]),                                            │
│   ToolMatch(tool=star, step=2, alternatives=[hisat2, bwa]),                                   │
│   ToolMatch(tool=featurecounts, step=3, alternatives=[salmon, htseq]),                        │
│   ToolMatch(tool=deseq2, step=4, alternatives=[edger, limma]),                                │
│   ToolMatch(tool=multiqc, step=5, alternatives=[])                                            │
│ ]                                                                                              │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: MODULE MAPPING (ModuleMapper)                                                          │
│                                                                                                │
│ Input: List[ToolMatch] from Step 2                                                            │
│                                                                                                │
│ Processing:                                                                                    │
│   For each tool:                                                                               │
│   ├── fastqc → nextflow-modules/quality_control/fastqc.nf ✓                                   │
│   ├── star → nextflow-modules/alignment/star.nf ✓                                             │
│   ├── featurecounts → nextflow-modules/quantification/featurecounts.nf ✓                      │
│   ├── deseq2 → nextflow-modules/analysis/deseq2.nf ✓                                          │
│   └── multiqc → nextflow-modules/reporting/multiqc.nf ✓                                       │
│                                                                                                │
│   Container resolution:                                                                        │
│   ├── fastqc → base                                                                           │
│   ├── star → rna-seq                                                                          │
│   ├── featurecounts → rna-seq                                                                 │
│   ├── deseq2 → rna-seq                                                                        │
│   └── multiqc → base                                                                          │
│                                                                                                │
│ Output: [                                                                                      │
│   NextflowModule(name="FASTQC", path="...", container="base"),                                │
│   NextflowModule(name="STAR", path="...", container="rna-seq"),                               │
│   NextflowModule(name="FEATURECOUNTS", path="...", container="rna-seq"),                      │
│   NextflowModule(name="DESEQ2", path="...", container="rna-seq"),                             │
│   NextflowModule(name="MULTIQC", path="...", container="base")                                │
│ ]                                                                                              │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: WORKFLOW GENERATION (WorkflowGenerator)                                                │
│                                                                                                │
│ Input: ParsedIntent + List[NextflowModule]                                                    │
│                                                                                                │
│ Processing:                                                                                    │
│   ├── Select template: templates/rnaseq_base.nf                                               │
│   ├── Generate include statements                                                             │
│   ├── Wire channels between processes                                                         │
│   ├── Generate nextflow.config with:                                                          │
│   │     └── params, profiles, container configs                                               │
│   └── Package into output directory                                                           │
│                                                                                                │
│ Output: Workflow(                                                                              │
│   name="rnaseq_differential_20231126_143022",                                                 │
│   main_script="... Nextflow DSL2 code ...",                                                   │
│   config="... nextflow.config ...",                                                           │
│   output_dir=Path("generated_workflows/rnaseq_differential_20231126_143022/")                 │
│ )                                                                                              │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│ OPTIONAL: PREFLIGHT VALIDATION (PreflightValidator)                                            │
│                                                                                                │
│ Input: ParsedIntent + Tools                                                                    │
│                                                                                                │
│ Checks:                                                                                        │
│   ├── Container images available? ✓                                                           │
│   ├── Reference data (GRCh38) available? ✓                                                    │
│   ├── All modules present? ✓                                                                  │
│   └── Resource estimation: 32GB RAM, 8 CPU, ~4 hours                                          │
│                                                                                                │
│ Output: ValidationReport(can_proceed=True, resources=ResourceEstimate(...))                   │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│ FINAL OUTPUT                                                                                   │
│                                                                                                │
│ generated_workflows/rnaseq_differential_20231126_143022/                                      │
│ ├── main.nf                    # Executable Nextflow workflow                                 │
│ ├── nextflow.config            # Configuration file                                          │
│ ├── modules/                   # Included module files                                        │
│ │   ├── fastqc.nf                                                                            │
│ │   ├── star.nf                                                                              │
│ │   ├── featurecounts.nf                                                                     │
│ │   ├── deseq2.nf                                                                            │
│ │   └── multiqc.nf                                                                           │
│ ├── params.yaml                # Default parameters                                          │
│ └── metadata.json              # Provenance information                                      │
│                                                                                                │
│ Run with: nextflow run main.nf -profile slurm --reads './data/*_{R1,R2}.fq.gz'               │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Interaction Matrix

| Component | Receives From | Sends To | Data Type |
|-----------|---------------|----------|-----------|
| Composer | User/API | All components | Query string, Config dict |
| IntentParser | Composer | ToolSelector | ParsedIntent |
| ToolSelector | IntentParser | ModuleMapper | List[ToolMatch] |
| ModuleMapper | ToolSelector | WorkflowGenerator | List[NextflowModule] |
| WorkflowGenerator | ModuleMapper + Intent | Output/User | Workflow |
| PreflightValidator | Intent + Tools | Composer | ValidationReport |

---

## Enhancement Priority Matrix

| Component | Enhancement | Impact | Effort | Priority |
|-----------|------------|--------|--------|----------|
| IntentParser | Multi-organism support | High | Medium | P1 |
| IntentParser | Contextual parsing (session memory) | High | High | P2 |
| ToolSelector | Dynamic catalog from bioconda | High | High | P2 |
| ToolSelector | User preference learning | Medium | Medium | P3 |
| ModuleMapper | Module versioning | High | Medium | P1 |
| ModuleMapper | Subworkflow support | High | High | P2 |
| WorkflowGenerator | Template expansion | High | Medium | P1 |
| WorkflowGenerator | Generated workflow validation | High | Low | P1 |
| PreflightValidator | Actual container verification | Medium | Medium | P2 |
| PreflightValidator | Dynamic resource profiling | Medium | High | P3 |

---

## Recommended Test Coverage Expansion

### Unit Tests Needed

1. **IntentParser**
   - [ ] Ambiguous query handling
   - [ ] Multi-language organism names
   - [ ] Conflicting parameters
   - [ ] LLM fallback triggers
   - [ ] Edge case: empty query
   - [ ] Edge case: unknown analysis type

2. **ToolSelector**
   - [ ] Missing tools in catalog
   - [ ] Version conflict resolution
   - [ ] Empty tool selection
   - [ ] Alternative tool ordering
   - [ ] Custom tool definitions

3. **ModuleMapper**
   - [ ] Missing module generation
   - [ ] I/O validation failures
   - [ ] Container mapping gaps
   - [ ] Module syntax validation

4. **WorkflowGenerator**
   - [ ] Complex channel wiring
   - [ ] Multi-output processes
   - [ ] Conditional workflows
   - [ ] Error handling in output

### Integration Tests Needed

1. **End-to-end query processing**
   - [ ] RNA-seq full pipeline
   - [ ] ChIP-seq full pipeline
   - [ ] Variant calling pipeline
   - [ ] Multi-sample handling

2. **Error scenarios**
   - [ ] Unknown analysis type
   - [ ] Missing tools
   - [ ] Container unavailable
   - [ ] Reference data missing

---

## Conclusion

The BioPipelines query flow is well-structured with clear separation of concerns. The four-step pipeline (Parse → Select → Map → Generate) provides good modularity. Key areas for enhancement include:

1. **Immediate priorities**: Workflow validation, template expansion, module versioning
2. **Medium-term**: Multi-organism support, contextual parsing, subworkflows  
3. **Long-term**: Dynamic tool catalog, user preference learning, cost optimization

The current test coverage is good for basic functionality but needs expansion for edge cases, error scenarios, and integration testing.
