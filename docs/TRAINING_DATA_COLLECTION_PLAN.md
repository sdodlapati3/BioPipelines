# Training Data Collection Plan for BioPipelines LLM Fine-tuning

**Version:** 1.0  
**Created:** December 3, 2025  
**Status:** Ready for Implementation

---

## Executive Summary

This document details the comprehensive plan to collect high-quality training data for fine-tuning a specialized bioinformatics workflow generation LLM. The BioPipelines system is **production-ready** for training data collection based on the following assessment:

### System Readiness Assessment ✅

| Component | Status | Details |
|-----------|--------|---------|
| **Test Coverage** | ✅ Ready | 1651 tests passing, 45% coverage |
| **Golden Queries** | ✅ Ready | 100 queries across 27 categories |
| **Evaluation Framework** | ✅ Ready | Complete benchmarks, scorer, reporter |
| **Knowledge Base** | ✅ Ready | 95 documents indexed (tools, analysis, errors) |
| **Intent Parser** | ✅ Ready | 26+ analysis types, ensemble parsing |
| **Workflow Generator** | ✅ Ready | Nextflow DSL2 generation |
| **Feedback Collection** | ✅ Exists | Basic JSONL logging in data/feedback/ |
| **Memory System** | ✅ Ready | Session, user profiles, preference learning |
| **Web UI** | ✅ Ready | Gradio interface with feedback section |

---

## Part 1: Data Sources and Collection Strategy

### 1.1 Synthetic Data Generation (Primary Source)

**Goal:** Generate 5,000+ high-quality query-response pairs from golden queries.

#### Source Materials:
- 100 golden queries in `tests/test_golden_queries.py`
- 26 analysis type definitions in `config/analysis_definitions.yaml`
- 46 tool descriptions in `ToolCatalogIndexer`
- 21 error patterns in `config/error_patterns.yaml`

#### Generation Approaches:

**A. Query Variation (5,000 examples)**
```
Base Query → Template Expansion → LLM Paraphrasing → Validation
```

Each golden query generates 50 variations:
- 10 formal/technical variations
- 10 casual/conversational variations
- 10 minimal/short variations
- 10 detailed/complex variations
- 10 edge case variations (typos, incomplete, etc.)

**B. Tool Selection Examples (2,000 examples)**
```
Analysis Type + Context → Expected Tools + Rationale
```

Example:
```json
{
  "input": "RNA-seq differential expression analysis for mouse",
  "output": {
    "tools": ["star", "featurecounts", "deseq2"],
    "rationale": "STAR for splice-aware alignment, featureCounts for gene counting, DESeq2 for differential expression with biological replicates"
  }
}
```

**C. Workflow Generation Examples (1,000 examples)**
```
Complete Query → Parsed Intent → Full Nextflow Workflow
```

**D. Education/Explanation Examples (500 examples)**
```
Tool/Concept Question → Accurate Explanation + Context
```

**E. Error Handling Examples (500 examples)**
```
Error Description → Diagnosis + Solution Steps
```

### 1.2 Interaction Logging (Secondary Source)

**Goal:** Collect 1,000+ real user interactions with quality filtering.

#### Current Infrastructure:
- `data/feedback/confirmations.jsonl` - Intent confirmations (already collecting)
- Web UI feedback section - Query corrections
- Session manager - Conversation history

#### Enhanced Collection Points:

| Interaction Type | Data Captured | Quality Signal |
|-----------------|---------------|----------------|
| Query submission | Query text, timestamp, session | - |
| Intent parsing | Parsed intent, confidence | High confidence = high quality |
| Tool selection | Selected tools, alternatives | User acceptance |
| Workflow generation | Generated code | Syntax validation |
| User feedback | Accept/reject/modify | Explicit signal |
| Execution result | Success/failure | Implicit signal |

### 1.3 Expert Curation (Test Set Only)

**Goal:** 100 hand-crafted benchmark examples for evaluation.

- **Not used for training** - Reserved for testing
- Created by domain experts
- Cover edge cases and challenging scenarios
- Include ground truth workflows

---

## Part 2: Implementation Architecture

### 2.1 Module Structure

```
src/training/
├── __init__.py                 # Package exports
├── config.py                   # Training configuration
├── data_generator.py           # Synthetic data generation
├── interaction_logger.py       # User interaction logging
├── data_pipeline.py            # Processing and validation
├── export.py                   # Export to training formats
├── augmentation.py             # Data augmentation
└── quality.py                  # Quality scoring and filtering
```

### 2.2 Data Generator Implementation

```python
# src/training/data_generator.py

class TrainingDataGenerator:
    """Generate synthetic training data from golden queries."""
    
    def __init__(self, llm: LLMAdapter, config: GeneratorConfig):
        self.llm = llm
        self.config = config
        self.intent_parser = IntentParser(llm)
        self.workflow_generator = WorkflowGenerator()
        self.tool_selector = ToolSelector()
        
    async def generate_query_variations(
        self, 
        base_query: str, 
        count: int = 50
    ) -> List[Dict]:
        """Generate variations of a query with ground truth."""
        variations = []
        
        # Parse base query for ground truth
        base_intent = await self.intent_parser.parse(base_query)
        
        # Generate variations using LLM
        prompt = self._build_variation_prompt(base_query, count)
        response = await self.llm.generate(prompt)
        
        for var_query in self._parse_variations(response):
            variations.append({
                "query": var_query,
                "ground_truth_intent": base_intent.to_dict(),
                "source": "synthetic",
                "base_query": base_query,
            })
            
        return variations
    
    async def generate_full_example(
        self, 
        query: str
    ) -> Dict:
        """Generate complete training example with workflow."""
        
        # Parse intent
        intent = await self.intent_parser.parse(query)
        
        # Select tools
        tools = self.tool_selector.find_tools_for_analysis(
            intent.analysis_type
        )
        
        # Generate workflow
        workflow = self.workflow_generator.generate(intent, tools)
        
        return {
            "id": generate_id(),
            "query": query,
            "intent": intent.to_dict(),
            "tools": [t.name for t in tools],
            "workflow": workflow.code,
            "validated": self._validate_workflow(workflow),
            "timestamp": datetime.now().isoformat(),
        }
```

### 2.3 Interaction Logger Implementation

```python
# src/training/interaction_logger.py

class InteractionLogger:
    """Log all user interactions for training data collection."""
    
    def __init__(self, storage_path: Path = None):
        self.storage_path = storage_path or Path("data/training/interactions")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.db = self._init_database()
        
    def log_interaction(self, interaction: Interaction) -> str:
        """Log a complete interaction with all components."""
        interaction_id = generate_id()
        
        record = {
            "id": interaction_id,
            "session_id": interaction.session_id,
            "timestamp": datetime.now().isoformat(),
            "query": interaction.query,
            "intent": interaction.intent.to_dict() if interaction.intent else None,
            "tools_selected": interaction.tools,
            "workflow_generated": interaction.workflow,
            "user_feedback": interaction.feedback,
            "modifications": interaction.modifications,
            "execution_success": interaction.success,
            "quality_score": self._calculate_quality_score(interaction),
        }
        
        self._save_record(record)
        return interaction_id
    
    def _calculate_quality_score(self, interaction: Interaction) -> float:
        """Calculate quality score for filtering."""
        score = 0.5  # Base score
        
        # High confidence intent parsing
        if interaction.intent and interaction.intent.confidence > 0.8:
            score += 0.2
            
        # User accepted without modifications
        if interaction.feedback == "accept" and not interaction.modifications:
            score += 0.2
            
        # Successful execution
        if interaction.success:
            score += 0.1
            
        return min(score, 1.0)
```

### 2.4 Data Pipeline Implementation

```python
# src/training/data_pipeline.py

class TrainingDataPipeline:
    """Process, validate, and format training data."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.validator = DataValidator()
        self.augmenter = DataAugmenter()
        self.formatter = TrainingDataFormatter()
        
    def process_all_sources(self) -> Dataset:
        """Process all data sources into unified dataset."""
        
        # Load and process each source
        synthetic = self.process_synthetic_data()
        interactions = self.process_interaction_logs()
        
        # Combine
        combined = self._merge_datasets([synthetic, interactions])
        
        # Deduplicate
        deduplicated = self._deduplicate(combined)
        
        # Quality filter
        filtered = self._quality_filter(deduplicated)
        
        return filtered
    
    def create_splits(
        self, 
        dataset: Dataset
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """Create train/val/test splits with stratification."""
        
        # Stratify by analysis type
        stratified = self._stratify_by_type(dataset)
        
        # Split ratios: 80% train, 10% val, 10% test
        train, val_test = train_test_split(
            stratified, test_size=0.2, stratify=stratified["analysis_type"]
        )
        val, test = train_test_split(
            val_test, test_size=0.5, stratify=val_test["analysis_type"]
        )
        
        return train, val, test
```

---

## Part 3: Data Quality Standards

### 3.1 Validation Rules

| Rule | Description | Action on Failure |
|------|-------------|-------------------|
| **Intent Parseable** | Query can be parsed to valid intent | Reject |
| **Workflow Valid** | Generated workflow passes syntax check | Reject |
| **Tools Mapped** | All tools exist in tool_mappings.yaml | Reject |
| **No PII** | No personal information | Remove PII |
| **Language Quality** | No gibberish or nonsense | Reject |
| **Consistent** | Intent matches expected analysis type | Review |

### 3.2 Quality Metrics

```python
@dataclass
class QualityMetrics:
    """Quality metrics for training data."""
    
    # Intent parsing
    intent_parse_rate: float      # % of queries that parse successfully
    intent_confidence_mean: float  # Average confidence score
    
    # Tool selection
    tool_coverage: float          # % of tools with examples
    tool_accuracy: float          # % matching expected tools
    
    # Workflow generation
    workflow_valid_rate: float    # % passing syntax validation
    workflow_complete_rate: float # % with all required steps
    
    # User signals
    acceptance_rate: float        # % of user-accepted outputs
    modification_rate: float      # % requiring modifications
    
    # Dataset balance
    category_entropy: float       # Diversity across categories
    difficulty_distribution: Dict # Examples per difficulty level
```

### 3.3 Minimum Thresholds

| Metric | Minimum Threshold |
|--------|-------------------|
| Intent parse rate | 95% |
| Workflow validity | 100% |
| Tool coverage | 90% (of 46 tools) |
| Category coverage | 100% (all 27 categories) |
| Acceptance rate | 80% (for logged interactions) |

---

## Part 4: Training Data Format

### 4.1 Instruction Format (for Fine-tuning)

```json
{
  "id": "train_00001",
  "messages": [
    {
      "role": "system",
      "content": "You are BioPipelines, an expert bioinformatics assistant that helps users create analysis workflows. You understand sequencing data types, analysis methods, and bioinformatics tools. When given a query, extract the analysis intent and generate appropriate Nextflow workflows."
    },
    {
      "role": "user", 
      "content": "Create an RNA-seq differential expression workflow for mouse liver samples"
    },
    {
      "role": "assistant",
      "content": "## Analysis Intent\n- **Type**: RNA-seq Differential Expression\n- **Organism**: Mouse (Mus musculus)\n- **Tissue**: Liver\n\n## Recommended Tools\n1. **STAR** - Splice-aware RNA-seq alignment\n2. **featureCounts** - Read counting per gene\n3. **DESeq2** - Differential expression analysis\n\n## Workflow\n```nextflow\n// RNA-seq Differential Expression Pipeline\n// Organism: Mouse | Tissue: Liver\n\ninclude { STAR_ALIGN } from './modules/star'\ninclude { FEATURECOUNTS } from './modules/featurecounts'\ninclude { DESEQ2 } from './modules/deseq2'\n\nworkflow RNA_SEQ_DE {\n    take:\n    reads     // paired-end FASTQ files\n    gtf       // gene annotation\n    \n    main:\n    STAR_ALIGN(reads)\n    FEATURECOUNTS(STAR_ALIGN.out.bam, gtf)\n    DESEQ2(FEATURECOUNTS.out.counts)\n    \n    emit:\n    results = DESEQ2.out.results\n}\n```"
    }
  ],
  "metadata": {
    "source": "synthetic",
    "category": "workflow_generation",
    "analysis_type": "rna_seq_differential_expression",
    "tools": ["star", "featurecounts", "deseq2"],
    "difficulty": 2,
    "validated": true
  }
}
```

### 4.2 Export Formats

| Format | Use Case | File |
|--------|----------|------|
| **JSONL** | General training | `train.jsonl`, `val.jsonl`, `test.jsonl` |
| **HuggingFace** | 🤗 Datasets integration | `dataset/` directory |
| **Alpaca** | Instruction tuning | `alpaca_format.json` |
| **ShareGPT** | Conversation format | `sharegpt_format.json` |

---

## Part 5: Implementation Timeline

### Week 1-2: Core Infrastructure

- [ ] Create `src/training/` module structure
- [ ] Implement `TrainingDataGenerator` class
- [ ] Implement `InteractionLogger` class
- [ ] Add integration with existing feedback system
- [ ] Write unit tests for generators

### Week 3: Synthetic Data Generation

- [ ] Generate query variations from 100 golden queries
- [ ] Generate tool selection examples
- [ ] Generate workflow examples
- [ ] Generate education examples
- [ ] Validate all generated data

### Week 4: Data Pipeline

- [ ] Implement `TrainingDataPipeline` class
- [ ] Add quality validation and filtering
- [ ] Implement data augmentation
- [ ] Create train/val/test splits
- [ ] Export to multiple formats

### Week 5: Enhanced Logging

- [ ] Enhance web UI feedback collection
- [ ] Add CLI interaction logging
- [ ] Implement quality scoring
- [ ] Set up automated data ingestion

### Week 6: Evaluation & Iteration

- [ ] Run evaluation on collected data
- [ ] Analyze quality metrics
- [ ] Identify gaps and add more examples
- [ ] Prepare final dataset

---

## Part 6: Target Dataset Size

| Data Type | Target Count | Status |
|-----------|--------------|--------|
| Query variations | 5,000 | Planned |
| Tool selection examples | 2,000 | Planned |
| Workflow generation | 1,000 | Planned |
| Education/explanation | 500 | Planned |
| Error handling | 500 | Planned |
| User interactions | 1,000+ | Collecting |
| Expert curated (test only) | 100 | Planned |
| **Total** | **~10,000** | |

---

## Part 7: System Strengthening

### How Training Data Collection Strengthens BioPipelines:

1. **Improved Intent Parsing**: Diverse query variations improve robustness to natural language variations.

2. **Better Tool Recommendations**: Tool selection examples teach optimal tool combinations for each analysis type.

3. **Higher Quality Workflows**: Validated workflow examples establish quality standards.

4. **Enhanced Education**: Explanation examples improve the system's ability to teach users.

5. **Robust Error Handling**: Error examples improve troubleshooting capabilities.

6. **User-Aligned Behavior**: Logged interactions capture real user preferences and expectations.

7. **Continuous Improvement**: The logging infrastructure enables ongoing data collection for model updates.

---

## Part 8: Files to Create

```
src/training/
├── __init__.py
├── config.py
├── data_generator.py
├── interaction_logger.py
├── data_pipeline.py
├── export.py
├── augmentation.py
├── quality.py
└── tests/
    ├── __init__.py
    ├── test_generator.py
    ├── test_logger.py
    └── test_pipeline.py

data/training/
├── raw/
│   ├── synthetic/
│   └── interactions/
├── processed/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
└── metadata/
    └── quality_report.json
```

---

## Conclusion

The BioPipelines system is **ready for training data collection**. The infrastructure is in place, the evaluation framework exists, and the collection mechanisms are partially implemented. 

**Next Step**: Begin implementation of the `src/training/` module starting with the data generator.

---

## Appendix: Verification Checklist

### Pre-Implementation Verification ✅

- [x] Golden queries exist and are comprehensive (100 queries)
- [x] Intent parser correctly parses all analysis types
- [x] Workflow generator produces valid Nextflow code
- [x] Tool selector maps tools to analysis types
- [x] Evaluation framework can score outputs
- [x] Feedback collection mechanism exists
- [x] Knowledge base is indexed with tool information
- [x] Test suite passes (1651 tests)

### Implementation Readiness ✅

- [x] Clear module structure defined
- [x] Data formats specified
- [x] Quality metrics defined
- [x] Timeline established
- [x] Target dataset size calculated
