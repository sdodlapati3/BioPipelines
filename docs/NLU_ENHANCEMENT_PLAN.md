# NLU Enhancement Plan: Professional Query Parsing

## Overview

This document outlines the implementation plan for enhancing BioPipelines' Natural Language Understanding (NLU) capabilities to match professional standards used by Rasa, AWS Lex, Google Dialogflow, and Microsoft LUIS.

## Current State

Our NLU pipeline already includes:
- ✅ Pattern-based intent classification (200+ regex patterns)
- ✅ Semantic similarity using FAISS + MiniLM-L6-v2
- ✅ LLM arbiter for edge cases
- ✅ Weighted ensemble voting (Pattern: 0.35, Semantic: 0.40, Entity: 0.25)
- ✅ Entity extraction (organisms, tissues, diseases, dataset IDs)
- ✅ Synonym normalization (`_normalize_data_type()`)
- ✅ Multi-question query splitting
- ✅ LRU caching for arbiter decisions

## Gaps to Address

### 1. Formal Training Data Format (Priority: HIGH)

**Problem**: Training data is currently hardcoded in Python files, making it difficult to maintain and extend.

**Solution**: YAML-based training data like Rasa.

**File Structure**:
```
config/
├── nlu/
│   ├── intents/
│   │   ├── data_operations.yaml    # DATA_SCAN, DATA_SEARCH, DATA_DOWNLOAD
│   │   ├── workflow_operations.yaml # WORKFLOW_CREATE, WORKFLOW_LIST
│   │   ├── job_management.yaml      # JOB_SUBMIT, JOB_STATUS, JOB_CANCEL
│   │   └── education.yaml           # EDUCATION_EXPLAIN, EDUCATION_HELP
│   ├── entities/
│   │   ├── organisms.yaml           # Human, mouse, rat, etc.
│   │   ├── assay_types.yaml         # RNA-seq, ChIP-seq, ATAC-seq
│   │   ├── diseases.yaml            # Cancer, Alzheimer's, etc.
│   │   └── tissues.yaml             # Brain, liver, heart, etc.
│   ├── synonyms.yaml                # Canonical mappings
│   ├── lookup_tables.yaml           # Large entity lists
│   └── regex_patterns.yaml          # Dataset ID patterns
```

**Schema Design**:
```yaml
# config/nlu/intents/data_operations.yaml
version: "1.0"

intents:
  - intent: DATA_SCAN
    description: "Scan local filesystem for data files"
    required_slots:
      - name: path
        type: file_path
        prompt: "Which directory should I scan?"
        default: "/scratch/sdodl001/BioPipelines/data"
    optional_slots:
      - name: data_type
        type: assay_type
        prompt: "Any specific data type? (e.g., RNA-seq, ChIP-seq)"
    examples:
      - "scan my data folder"
      - "what files do we have"
      - "show me what's in [/data/raw](path)"
      - "how many [scRNA-seq](data_type) files do we have"
      - "list all [ChIP-seq](data_type) samples in [/scratch](path)"
      - "check [~/projects](path) for [ATAC-seq](data_type) data"
    
  - intent: DATA_SEARCH
    description: "Search public databases for datasets"
    required_slots:
      - name: query
        type: text
        prompt: "What are you looking for?"
    optional_slots:
      - name: organism
        type: organism
      - name: tissue
        type: tissue
      - name: disease
        type: disease
    examples:
      - "search for [human](organism) [liver](tissue) data"
      - "find [cancer](disease) [RNA-seq](data_type) datasets"
      - "look for [mouse](organism) [brain](tissue) [Alzheimer's](disease) samples"
```

**Entity Schema**:
```yaml
# config/nlu/entities/organisms.yaml
version: "1.0"

entities:
  - entity: organism
    values:
      - canonical: "Homo sapiens"
        aliases: ["human", "humans", "h. sapiens", "hg38", "hg19", "grch38"]
      - canonical: "Mus musculus"
        aliases: ["mouse", "mice", "m. musculus", "mm10", "mm39"]
      - canonical: "Rattus norvegicus"
        aliases: ["rat", "rats", "r. norvegicus", "rn6", "rn7"]
      - canonical: "Drosophila melanogaster"
        aliases: ["fly", "fruit fly", "drosophila", "dm6"]
```

**Regex Patterns**:
```yaml
# config/nlu/regex_patterns.yaml
version: "1.0"

patterns:
  - name: geo_accession
    entity_type: dataset_id
    pattern: "GSE\\d{5,8}"
    metadata:
      source: "GEO"
      
  - name: sra_accession
    entity_type: dataset_id  
    pattern: "(?:SRR|ERR|DRR)\\d{6,10}"
    metadata:
      source: "SRA"
```

---

### 2. Active Learning Loop (Priority: MEDIUM)

**Problem**: System doesn't learn from user corrections or feedback.

**Solution**: Implement correction tracking and periodic retraining.

**Components**:

1. **Correction Store** (`corrections.jsonl`):
```jsonl
{"timestamp": "2025-12-02T10:30:00", "query": "search for liver data", "predicted": "DATA_SCAN", "corrected": "DATA_SEARCH", "user": "anonymous"}
{"timestamp": "2025-12-02T10:31:00", "query": "run the pipeline", "predicted": "WORKFLOW_CREATE", "corrected": "JOB_SUBMIT", "user": "anonymous"}
```

2. **Feedback Collection API**:
```python
class ActiveLearner:
    def record_correction(self, query: str, predicted: str, corrected: str):
        """Record when user corrects an intent prediction."""
        
    def record_confirmation(self, query: str, intent: str):
        """Record when user confirms correct prediction (positive signal)."""
        
    def get_confusion_matrix(self) -> Dict[str, Dict[str, int]]:
        """Get intent confusion statistics."""
        
    def get_problematic_queries(self, min_corrections: int = 3) -> List[str]:
        """Get queries that are frequently miscategorized."""
        
    def export_for_retraining(self) -> str:
        """Export corrections as YAML training examples."""
```

3. **Learning Metrics**:
```python
@dataclass
class LearningMetrics:
    total_queries: int
    corrections_count: int
    correction_rate: float  # corrections / total
    top_confused_intents: List[Tuple[str, str, int]]  # (predicted, actual, count)
    improvement_over_time: float  # Trend of correction rate
```

---

### 3. Slot Prompting (Priority: HIGH)

**Problem**: When required information is missing, system doesn't ask follow-up questions.

**Solution**: Implement slot validation with clarification prompts.

**Architecture**:
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Intent Parse   │ --> │  Slot Validator │ --> │ Prompt Generator│
│  (slots found)  │     │ (check required)│     │ (ask for missing│
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │ Dialogue State  │
                        │ (track pending) │
                        └─────────────────┘
```

**Slot Definition**:
```python
@dataclass
class SlotDefinition:
    name: str
    type: str  # "text", "entity", "number", "boolean", "choice"
    required: bool = False
    prompt: str = ""
    default: Any = None
    validation: Optional[Callable] = None
    choices: Optional[List[str]] = None  # For choice type

@dataclass  
class IntentSlots:
    intent: str
    required_slots: List[SlotDefinition]
    optional_slots: List[SlotDefinition]
```

**Slot Validator**:
```python
class SlotValidator:
    def validate(self, intent: str, extracted_slots: Dict[str, Any]) -> SlotValidationResult:
        """
        Check if all required slots are filled.
        
        Returns:
            SlotValidationResult with:
            - is_complete: bool
            - missing_slots: List[SlotDefinition]
            - clarification_prompt: str (if incomplete)
        """

    def generate_prompt(self, missing_slot: SlotDefinition) -> str:
        """Generate natural language prompt for missing slot."""
        # Example outputs:
        # "Which directory should I scan?"
        # "What organism are you interested in? (e.g., human, mouse)"
        # "Which data type? Options: RNA-seq, ChIP-seq, ATAC-seq"
```

**Dialogue Flow**:
```
User: "scan my data"
Bot:  Intent=DATA_SCAN, slots={}
      Missing: path (required)
      → "Which directory should I scan? (default: /scratch/.../data)"

User: "the raw folder"
Bot:  Fills path="/scratch/.../data/raw"
      → Executes DATA_SCAN with path="/scratch/.../data/raw"
```

---

### 4. Intent Balance Metrics (Priority: LOW)

**Problem**: No visibility into training data distribution.

**Solution**: Track and report intent balance statistics.

**Metrics**:
```python
@dataclass
class IntentBalanceReport:
    total_examples: int
    intent_counts: Dict[str, int]
    min_examples: int
    max_examples: int
    imbalance_ratio: float  # max / min
    warnings: List[str]  # Intents with too few/many examples
    
    def is_balanced(self, threshold: float = 5.0) -> bool:
        """Return True if imbalance ratio is below threshold."""
        return self.imbalance_ratio <= threshold
```

**Balance Checker**:
```python
class IntentBalanceChecker:
    def analyze(self, training_data: Dict[str, List[str]]) -> IntentBalanceReport:
        """Analyze training data distribution."""
        
    def generate_warnings(self) -> List[str]:
        """
        Generate actionable warnings:
        - "Intent 'DATA_SCAN' has 500 examples but 'JOB_CANCEL' has only 5"
        - "Consider adding more examples to: JOB_CANCEL, REFERENCE_INDEX"
        - "Intent 'META_UNKNOWN' should have ~10% of total examples"
        """
```

---

### 5. Entity Roles Support (Priority: MEDIUM)

**Problem**: Cannot distinguish same entity type in different roles (e.g., source vs destination path).

**Solution**: Add role annotations to entities.

**Role Schema**:
```yaml
# In training examples:
examples:
  - "copy from [/data/raw](path:source) to [/data/processed](path:destination)"
  - "move [sample1.fastq](file:source) to [/archive](path:destination)"
```

**Entity with Role**:
```python
@dataclass
class Entity:
    type: EntityType
    value: str
    canonical: Optional[str] = None
    span: Tuple[int, int] = (0, 0)
    confidence: float = 1.0
    role: Optional[str] = None  # NEW: "source", "destination", "origin", etc.
    group: Optional[str] = None  # NEW: For grouping related entities
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Role Extraction**:
```python
class RoleAwareEntityExtractor:
    def extract_with_roles(self, text: str, intent: str) -> List[Entity]:
        """
        Extract entities with role inference based on:
        1. Explicit markers: "from X to Y" → X=source, Y=destination
        2. Position: First path often = source, second = destination
        3. Intent context: WORKFLOW_CREATE expects different roles than DATA_DOWNLOAD
        """
```

---

## Implementation Phases

### Phase 1: Training Data Format (Week 1)
1. Create YAML schema specification
2. Migrate existing patterns to YAML files
3. Implement `TrainingDataLoader` class
4. Add validation for training data files
5. Update `IntentParser` to load from YAML

### Phase 2: Slot Prompting (Week 2)
1. Define slot requirements for each intent
2. Implement `SlotValidator` class
3. Add `SlotPromptGenerator` for natural prompts
4. Integrate with `DialogueManager`
5. Add tests for slot filling flows

### Phase 3: Active Learning (Week 3)
1. Create correction storage (JSONL)
2. Implement feedback collection API
3. Add confusion matrix tracking
4. Create export tool for retraining
5. Add learning metrics dashboard

### Phase 4: Entity Roles & Balance (Week 4)
1. Extend `Entity` dataclass with role field
2. Implement role extraction patterns
3. Add `IntentBalanceChecker`
4. Create CLI tool for balance analysis
5. Add warnings to training pipeline

---

## File Changes Summary

### New Files
```
config/nlu/
├── intents/*.yaml          # Intent definitions with examples
├── entities/*.yaml         # Entity definitions with synonyms
├── synonyms.yaml           # Global synonym mappings
├── lookup_tables.yaml      # Large entity lists
└── regex_patterns.yaml     # Regex-based entity patterns

src/workflow_composer/agents/intent/
├── training_data.py        # TrainingDataLoader, schema validation
├── slot_filling.py         # SlotValidator, SlotPromptGenerator
├── active_learning.py      # ActiveLearner, correction tracking
└── balance_checker.py      # IntentBalanceChecker, metrics
```

### Modified Files
```
src/workflow_composer/agents/intent/parser.py
  - Load patterns from YAML instead of hardcoded
  - Add role field to Entity dataclass

src/workflow_composer/agents/intent/unified_parser.py
  - Integrate slot validation
  - Return clarification prompts when slots missing

src/workflow_composer/agents/unified_agent.py
  - Handle clarification responses
  - Track corrections for active learning
```

---

## Success Criteria

1. **Training Data**: All 200+ patterns migrated to YAML, validation passes
2. **Slot Prompting**: 90% of required-slot intents have prompts defined
3. **Active Learning**: Corrections stored, exportable for retraining
4. **Balance Metrics**: CLI tool reports imbalance warnings
5. **Entity Roles**: "copy from X to Y" correctly identifies source/destination

---

## Timeline

| Week | Focus | Deliverables |
|------|-------|--------------|
| 1 | Training Data Format | YAML schema, loader, migration |
| 2 | Slot Prompting | Validator, prompts, dialogue integration |
| 3 | Active Learning | Correction store, feedback API, metrics |
| 4 | Entity Roles & Balance | Role extraction, balance checker, CLI |

---

## Next Steps

1. Start with YAML schema design and training data loader
2. Migrate DATA_SCAN and DATA_SEARCH intents first as proof of concept
3. Add slot definitions for high-frequency intents
4. Implement feedback collection in web UI
