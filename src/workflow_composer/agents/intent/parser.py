"""
Intent Parser
=============

Multi-stage intent detection with confidence scoring and entity extraction.

Architecture:
1. Pattern Matching (fast, high precision for common intents)
2. Semantic Similarity (medium, handles paraphrases)  
3. LLM Classification (slow, handles complex/ambiguous cases)

Features:
- Intent hierarchy (composite intents)
- Confidence scoring
- Entity extraction
- Disambiguation suggestions
"""

import re
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Set

logger = logging.getLogger(__name__)


# =============================================================================
# INTENT TAXONOMY
# =============================================================================

class IntentType(Enum):
    """
    Hierarchical intent taxonomy for bioinformatics workflows.
    
    Categories:
    - DATA_*: Data-related operations
    - WORKFLOW_*: Pipeline/workflow operations
    - JOB_*: HPC job management
    - ANALYSIS_*: Result analysis
    - SYSTEM_*: System/infrastructure operations
    - EDUCATION_*: Learning and help
    - META_*: Conversation management
    """
    
    # Data Discovery & Management
    DATA_SCAN = auto()           # Scan local filesystem
    DATA_SEARCH = auto()         # Search public databases
    DATA_DOWNLOAD = auto()       # Download datasets
    DATA_VALIDATE = auto()       # Validate data integrity
    DATA_CLEANUP = auto()        # Clean corrupted files
    DATA_DESCRIBE = auto()       # Describe file contents
    DATA_ORGANIZE = auto()       # Organize/reorganize data
    
    # Workflow Operations
    WORKFLOW_CREATE = auto()     # Generate workflow
    WORKFLOW_LIST = auto()       # List available workflows
    WORKFLOW_CONFIGURE = auto()  # Configure workflow parameters
    WORKFLOW_VISUALIZE = auto()  # Show workflow DAG
    WORKFLOW_VALIDATE = auto()   # Validate workflow code
    WORKFLOW_MODIFY = auto()     # Modify existing workflow
    WORKFLOW_DESCRIBE = auto()   # Describe workflow details
    
    # Reference Data
    REFERENCE_CHECK = auto()     # Check reference availability
    REFERENCE_DOWNLOAD = auto()  # Download references
    REFERENCE_INDEX = auto()     # Build aligner index
    
    # Job Management
    JOB_SUBMIT = auto()          # Submit to SLURM
    JOB_STATUS = auto()          # Check job status
    JOB_LOGS = auto()            # View job logs
    JOB_CANCEL = auto()          # Cancel running job
    JOB_RESUBMIT = auto()        # Resubmit failed job
    JOB_LIST = auto()            # List all jobs
    JOB_WATCH = auto()           # Monitor job in real-time
    JOB_RESOURCES = auto()       # Check job resource usage
    
    # Analysis & Results
    ANALYSIS_RUN = auto()        # Run analysis on results
    ANALYSIS_INTERPRET = auto()  # Interpret QC/results
    ANALYSIS_COMPARE = auto()    # Compare samples/conditions
    ANALYSIS_VISUALIZE = auto()  # Generate plots
    ANALYSIS_REPORT = auto()     # Generate report
    
    # Data Operations (extended)
    DATA_COMPARE = auto()        # Compare datasets
    DATA_FILTER = auto()         # Filter datasets
    # DATA_VALIDATE already defined above
    
    # Diagnostics
    DIAGNOSE_ERROR = auto()      # Diagnose failures
    DIAGNOSE_RECOVER = auto()    # Attempt recovery
    
    # System Operations
    SYSTEM_STATUS = auto()       # Check system health
    SYSTEM_RESTART = auto()      # Restart services
    SYSTEM_COMMAND = auto()      # Run a simple shell command (not a workflow)
    
    # Education
    EDUCATION_EXPLAIN = auto()   # Explain concepts
    EDUCATION_HELP = auto()      # Show help
    EDUCATION_TUTORIAL = auto()  # Step-by-step guide
    
    # Meta/Conversational
    META_CONFIRM = auto()        # User confirmation (yes/no)
    META_CANCEL = auto()          # Cancel current operation
    META_CLARIFY = auto()         # User clarifying previous statement
    META_CORRECT = auto()         # User correcting previous statement ("actually X not Y")
    META_UNDO = auto()            # Undo last action
    META_GREETING = auto()        # Hello/greeting
    META_THANKS = auto()          # Thank you
    META_UNKNOWN = auto()         # Cannot determine intent
    
    # Context-Aware Intents
    CONTEXT_RECALL = auto()      # Recall/reference previous results
    CONTEXT_METADATA = auto()    # Get metadata about previous results
    
    # Composite Intents (multi-step)
    COMPOSITE_CHECK_THEN_SEARCH = auto()  # Check local, then search online
    COMPOSITE_SEARCH_THEN_DOWNLOAD = auto()  # Search then download
    COMPOSITE_GENERATE_THEN_RUN = auto()  # Create workflow and run


class EntityType(Enum):
    """Types of entities that can be extracted."""
    # Biological
    ORGANISM = auto()
    TISSUE = auto()
    CELL_LINE = auto()
    CELL_TYPE = auto()
    DISEASE = auto()
    GENE = auto()
    PROTEIN = auto()
    HISTONE_MARK = auto()
    
    # Data/Analysis
    ASSAY_TYPE = auto()
    DATA_TYPE = auto()
    FILE_FORMAT = auto()
    
    # Identifiers
    DATASET_ID = auto()       # GSE12345, ENCSR000AAA
    PROJECT_ID = auto()       # TCGA-GBM
    JOB_ID = auto()           # SLURM job ID
    SAMPLE_ID = auto()
    
    # Paths & Locations
    FILE_PATH = auto()
    DIRECTORY_PATH = auto()
    URL = auto()
    
    # Workflow
    WORKFLOW_TYPE = auto()
    PIPELINE_STEP = auto()
    
    # Quantitative
    COUNT = auto()
    THRESHOLD = auto()
    
    # Temporal
    TIME_REFERENCE = auto()    # "yesterday", "last job"


@dataclass
class Entity:
    """An extracted entity with metadata."""
    type: EntityType
    value: str
    canonical: str = None  # Normalized form (optional)
    span: Tuple[int, int] = (0, 0)  # Character positions (optional)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set canonical to value if not provided."""
        if self.canonical is None:
            self.canonical = self.value


@dataclass  
class IntentResult:
    """Result of intent parsing."""
    primary_intent: IntentType
    confidence: float
    entities: List[Entity]
    
    # For composite intents
    sub_intents: List[IntentType] = field(default_factory=list)
    
    # For disambiguation
    alternatives: List[Tuple[IntentType, float]] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_prompt: str = ""
    
    # Extracted slots for tool execution
    slots: Dict[str, Any] = field(default_factory=dict)
    
    # Raw pattern match info
    matched_pattern: Optional[str] = None
    
    @property
    def is_confident(self) -> bool:
        """Check if confidence is high enough for direct execution."""
        return self.confidence >= 0.7 and not self.needs_clarification
    
    def get_entity(self, entity_type: EntityType) -> Optional[Entity]:
        """Get first entity of given type."""
        for e in self.entities:
            if e.type == entity_type:
                return e
        return None
    
    def get_entities(self, entity_type: EntityType) -> List[Entity]:
        """Get all entities of given type."""
        return [e for e in self.entities if e.type == entity_type]


# =============================================================================
# INTENT PATTERNS
# =============================================================================

# High-precision patterns for common intents
# Each pattern: (regex, IntentType, slot_mappings)
# Note: Order matters - more specific patterns should come before general ones
INTENT_PATTERNS: List[Tuple[str, IntentType, Dict[str, int]]] = [
    # =========================================================================
    # COMPOSITE INTENTS (must be first - they are more specific)
    # =========================================================================
    
    # Check then search (variations)
    (r"(?:check|see|look)\s+(?:if\s+)?(?:we\s+have|there\s+is|for)\s+(.+?)[\s,]+(?:and\s+)?(?:if\s+not|else|otherwise)\s+(?:search|look|find)",
     IntentType.COMPOSITE_CHECK_THEN_SEARCH, {"query": 1}),
    (r"(?:do\s+we\s+have|is\s+there)\s+(.+?)[\s,?]+(?:if\s+not|otherwise|else)\s+(?:search|find|look|get)",
     IntentType.COMPOSITE_CHECK_THEN_SEARCH, {"query": 1}),
    (r"(?:check|scan)\s+(?:local(?:ly)?|here|this\s+folder)\s+(?:for\s+)?(.+?)[\s,]+(?:and\s+)?(?:then\s+)?(?:search|look)\s+(?:online|databases?|externally)",
     IntentType.COMPOSITE_CHECK_THEN_SEARCH, {"query": 1}),
    (r"(?:first\s+)?(?:check|look)\s+(?:for\s+)?(.+?)\s+(?:locally|here)[\s,]+(?:then\s+)?(?:search|look)\s+(?:online|remotely|databases?)",
     IntentType.COMPOSITE_CHECK_THEN_SEARCH, {"query": 1}),
    
    # Search then download
    (r"(?:search|find)\s+(?:for\s+)?(.+?)\s+(?:and\s+)?(?:then\s+)?(?:download|get)",
     IntentType.COMPOSITE_SEARCH_THEN_DOWNLOAD, {"query": 1}),
    
    # "find data, create workflow, and run it" - full pipeline flow
    (r"(?:find|search|get)\s+(?:the\s+)?data[,;]?\s+(?:and\s+)?(?:create|build|make)\s+(?:a\s+)?(?:workflow|pipeline)[,;]?\s+(?:and\s+)?(?:run|execute|submit)\s+it",
     IntentType.COMPOSITE_GENERATE_THEN_RUN, {}),
    
    # Generate then run
    (r"(?:create|generate|make)\s+(?:a\s+)?(.+?)\s+(?:workflow|pipeline)\s+(?:and\s+)?(?:then\s+)?(?:run|execute|submit)",
     IntentType.COMPOSITE_GENERATE_THEN_RUN, {"workflow_type": 1}),
    
    # =========================================================================
    # REFERENCE CHECK - must come before DATA_SCAN to prevent "check if we have X genome" from matching DATA_SCAN
    # =========================================================================
    # "check if we have the human reference genome" - explicit check for reference genome
    (r"check\s+if\s+(?:we\s+have|there\s+is)\s+(?:the\s+)?(?:a\s+)?(?:\w+\s+)?(?:reference\s+)?genome",
     IntentType.REFERENCE_CHECK, {}),
    # "do we have the human reference genome"
    (r"(?:do\s+we\s+have|have\s+we\s+got)\s+(?:the\s+)?(?:a\s+)?(?:\w+\s+)?(?:reference\s+)?genome",
     IntentType.REFERENCE_CHECK, {}),
    # "is the human reference genome available"
    (r"(?:is|are)\s+(?:the\s+)?(?:\w+\s+)?(?:reference\s+)?genome(?:s?)?\s+(?:available|ready|installed|present)",
     IntentType.REFERENCE_CHECK, {}),
    # "verify we have the reference for human"
    (r"(?:verify|confirm|make\s+sure)\s+(?:we\s+have|there\s+is)\s+(?:the\s+)?(?:a\s+)?(?:\w+\s+)?reference",
     IntentType.REFERENCE_CHECK, {}),
    
    # =========================================================================
    # EDUCATION - must come before workflow creation to catch "how does X work"
    # =========================================================================
    # Tool recommendation questions - must capture full topic after "for"
    (r"what\s+tools?\s+(?:should|would|do)\s+(?:i|we|you)?\s*(?:use|recommend|need|suggest)\s+for\s+(.+?)(?:\?|$)",
     IntentType.EDUCATION_EXPLAIN, {"topic": 1}),
    (r"what\s+tools?\s+(?:should|would|do)\s+(?:i|we|you)?\s*(?:use|recommend|need|suggest)(?:\?|$)",
     IntentType.EDUCATION_EXPLAIN, {"topic": "tools"}),  # When no topic specified
    (r"(?:recommend|suggest)\s+(?:a\s+|some\s+)?tools?\s+for\s+(.+?)(?:\?|$)",
     IntentType.EDUCATION_EXPLAIN, {"topic": 1}),
    (r"which\s+tools?\s+(?:are\s+)?(?:best|good|recommended)\s+for\s+(.+?)(?:\?|$)",
     IntentType.EDUCATION_EXPLAIN, {"topic": 1}),
    (r"what(?:'s| is)\s+(?:the\s+)?(?:best|recommended)\s+(?:tool|software|program)\s+for\s+(.+?)(?:\?|$)",
     IntentType.EDUCATION_EXPLAIN, {"topic": 1}),
    # Follow-up "how about for X specifically" - tool recommendation context
    (r"(?:how|what)\s+about\s+(?:for\s+)?(.+?)(?:\s+specifically)?(?:\?|$)",
     IntentType.EDUCATION_EXPLAIN, {"topic": 1}),
    
    # "How do/does X work" - both singular and plural subjects
    (r"how\s+do(?:es)?\s+(.+?)\s+work(?:\??|$)",
     IntentType.EDUCATION_EXPLAIN, {"concept": 1}),
    (r"how\s+do(?:es)?\s+(.+?)\s+(?:algorithms?|methods?)\s+work",
     IntentType.EDUCATION_EXPLAIN, {"concept": 1}),
    # "Can you how do X work?" (malformed but should still match)
    (r"(?:can\s+you\s+)?how\s+do(?:es)?\s+(.+?)\s+work",
     IntentType.EDUCATION_EXPLAIN, {"concept": 1}),
    (r"(?:i\s+want\s+to|help\s+me)\s+(?:learn|understand)\s+(?:about\s+)?(.+)",
     IntentType.EDUCATION_EXPLAIN, {"concept": 1}),
    # "search for a workflow to create" - wants guidance, not data
    (r"search\s+for\s+(?:a\s+)?(?:workflow|pipeline)\s+to\s+(?:create|build|make|design)",
     IntentType.EDUCATION_EXPLAIN, {"topic": "workflow_creation"}),
    # "what does X measure?" / "how does X compare to Y?"
    (r"what\s+does\s+(.+?)\s+measure(?:\??|$)",
     IntentType.EDUCATION_EXPLAIN, {"concept": 1}),
    (r"how\s+does\s+(.+?)\s+compare\s+to\s+(.+?)(?:\??|$)",
     IntentType.EDUCATION_EXPLAIN, {"concept1": 1, "concept2": 2}),
    
    # =========================================================================
    # DATA SCANNING - local filesystem
    # =========================================================================
    (r"(?:scan|find|look\s+for|discover|list|show)\s+(?:the\s+)?(?:local\s+)?(?:data|files?|samples?|fastq)(?:\s+(?:in|at|from|under)\s+(.+))?", 
     IntentType.DATA_SCAN, {"path": 1}),
    # Explicit path patterns for scan
    (r"scan\s+([/~][^\s]+)\s+(?:for\s+)?(?:data|files?|samples?|fastq)?",
     IntentType.DATA_SCAN, {"path": 1}),
    (r"scan\s+(?:for\s+)?(?:data|files?|samples?|fastq)\s+(?:in|at|from|under)\s+([/~][^\s]+)",
     IntentType.DATA_SCAN, {"path": 1}),
    # "check ~/path for data" / "please check /path for data"
    (r"(?:please\s+)?check\s+([/~][^\s]+)\s+for\s+(?:data|files?|samples?)",
     IntentType.DATA_SCAN, {"path": 1}),
    # "show me what's in /path" / "what's in /path"
    (r"(?:show\s+(?:me\s+)?)?what(?:'s|\s+is)\s+in\s+([/~][^\s]+)",
     IntentType.DATA_SCAN, {"path": 1}),
    (r"what\s+(?:data|files?|samples?)\s+(?:do\s+)?(?:i\s+have|we\s+have|is\s+available|exist)(?:\s+(?:in|at)\s+(.+))?",
     IntentType.DATA_SCAN, {"path": 1}),
    # "What samples are available locally?"
    (r"what\s+(?:data|samples?|files?)\s+(?:are|is)\s+available\s+locally",
     IntentType.DATA_SCAN, {}),
    # "What types of data do we have" / "what kinds of data are in the folder"
    (r"what\s+(?:types?|kinds?)\s+of\s+(?:data|files?|samples?)\s+(?:do\s+we\s+have|are\s+(?:there|available|in))",
     IntentType.DATA_SCAN, {}),
    # "Explain/tell me what data we have" - asking about local data
    (r"(?:can\s+you\s+)?(?:explain|tell\s+me|show\s+me)\s+(?:me\s+)?what\s+(?:types?\s+of\s+)?(?:data|files?|samples?)\s+(?:we\s+have|i\s+have|is\s+available|are\s+(?:in|available))",
     IntentType.DATA_SCAN, {}),
    
    # Data type specific scans - extract data_type slot
    # Natural language data type patterns (must come first to match longer phrases)
    # "how many single cell rna seq data files do we have"
    (r"how\s+many\s+(single[\s_-]?cell(?:\s+rna)?(?:\s+seq)?|sc[\s_-]?rna[\s_-]?seq)\s+(?:data\s+)?(?:files?|samples?)",
     IntentType.DATA_SCAN, {"data_type": 1}),
    # "do we have any single cell data" / "do we have single cell rna seq"
    (r"do\s+we\s+have\s+(?:any\s+)?(single[\s_-]?cell(?:\s+rna)?(?:\s+seq)?|sc[\s_-]?rna[\s_-]?seq)",
     IntentType.DATA_SCAN, {"data_type": 1}),
    # "how many scRNA-seq files do we have" / "do we have any ChIP-seq data"
    (r"(?:how\s+many|do\s+we\s+have(?:\s+any)?|show\s+me(?:\s+the)?|list(?:\s+all)?|find(?:\s+all)?)\s+((?:sc)?rna[_-]?seq|chip[_-]?seq|atac[_-]?seq|methylation|wgs|wes|hi[_-]?c|cut\s*(?:&|and)\s*run)\s*(?:data|files?|samples?)?",
     IntentType.DATA_SCAN, {"data_type": 1}),
    # "what scRNA-seq data do we have" / "what chip-seq samples are available"
    (r"what\s+((?:sc)?rna[_-]?seq|chip[_-]?seq|atac[_-]?seq|single[\s_-]?cell|methylation|wgs|wes|hi[_-]?c)\s+(?:data|files?|samples?)\s+(?:do\s+we\s+have|are\s+available|exist)",
     IntentType.DATA_SCAN, {"data_type": 1}),
    # "check for scRNA-seq in the data folder"
    (r"(?:check|scan|look)\s+(?:for\s+)?((?:sc)?rna[_-]?seq|chip[_-]?seq|atac[_-]?seq|single[\s_-]?cell|methylation|wgs|wes|hi[_-]?c)\s+(?:data|files?|samples?)?\s*(?:in|at|under)\s+([/~][^\s]+)?",
     IntentType.DATA_SCAN, {"data_type": 1, "path": 2}),
    
    # "Inventory my data in /path"
    (r"inventory\s+(?:my\s+)?(?:data|files?)\s+(?:in|at)\s+([/~][^\s]+)",
     IntentType.DATA_SCAN, {"path": 1}),
    # "inventory my sequencing runs" / "inventory my samples"
    (r"inventory\s+(?:my\s+)?(?:sequencing\s+runs?|samples?|data|reads?)",
     IntentType.DATA_SCAN, {}),
    # This pattern must NOT match if "if not search" follows
    # Using negative lookahead to exclude composite patterns
    (r"check\s+if\s+(?:we\s+have|there\s+is|there\s+are)\s+(?:any\s+)?(.+?)(?:\s+data|\s+locally)?$(?<!\bsearch\b)(?<!\bonline\b)",
     IntentType.DATA_SCAN, {"query": 1}),
    
    # =========================================================================
    # DATA DESCRIBE - Show details about local data (must come before DATABASE SEARCH)
    # =========================================================================
    # "show me details of methylation data" / "show details of the RNA-seq files"
    (r"(?:show|give)\s+(?:me\s+)?(?:the\s+)?details?\s+(?:of|for|about)\s+(?:the\s+)?(?:my\s+)?(.+?)(?:\s+data|\s+files?|\s+samples?)?$",
     IntentType.DATA_DESCRIBE, {"data_type": 1}),
    # "describe methylation data" / "describe my RNA-seq files"
    (r"describe\s+(?:the\s+)?(?:my\s+)?(.+?)(?:\s+data|\s+files?|\s+samples?)?$",
     IntentType.DATA_DESCRIBE, {"data_type": 1}),
    # "what's in the methylation data" / "what is in the ChIP-seq files"
    (r"what(?:'s|\s+is)\s+in\s+(?:the\s+)?(?:my\s+)?(.+?)(?:\s+data|\s+files?|\s+samples?)?$",
     IntentType.DATA_DESCRIBE, {"data_type": 1}),
    # "tell me about the methylation data" / "info about RNA-seq"
    (r"(?:tell\s+me\s+about|info(?:rmation)?\s+(?:about|on)|more\s+(?:info|details?)\s+(?:on|about))\s+(?:the\s+)?(?:my\s+)?(.+?)(?:\s+data|\s+files?)?$",
     IntentType.DATA_DESCRIBE, {"data_type": 1}),
    # "get details for GSE12345" / "show info for ENCSR000ABC"
    (r"(?:get|show|display)\s+(?:the\s+)?(?:details?|info(?:rmation)?|metadata)\s+(?:for|of|about)\s+([A-Z]+\d+[A-Z0-9]*)",
     IntentType.DATA_DESCRIBE, {"dataset_id": 1}),
    # "inspect the data" / "examine my files"
    (r"(?:inspect|examine|analyze|summarize)\s+(?:the\s+)?(?:my\s+)?(?:local\s+)?(.+?)(?:\s+data|\s+files?)?$",
     IntentType.DATA_DESCRIBE, {"data_type": 1}),
    
    # =========================================================================
    # DATABASE SEARCH
    # =========================================================================
    # Generic "Search for X" - should match before more specific patterns
    (r"(?:search|find|look)\s+for\s+(.+?)\s+data",
     IntentType.DATA_SEARCH, {"query": 1}),
    (r"(?:search|find|look)\s+for\s+(.+?)$",
     IntentType.DATA_SEARCH, {"query": 1}),
    # "looking for X in Y" / "I'm looking for X samples"
    (r"(?:i'?m\s+)?looking\s+for\s+(.+?)\s+(?:in|from)\s+(.+)",
     IntentType.DATA_SEARCH, {"query": 1, "source": 2}),
    (r"(?:i'?m\s+)?looking\s+for\s+(.+?)(?:\s+samples?|\s+data)?",
     IntentType.DATA_SEARCH, {"query": 1}),
    # "I need X data", "I want X data" patterns
    (r"(?:i\s+need|i\s+want|need|want)\s+(.+?)\s+(?:data|datasets?)(?:\s+from\s+(.+))?",
     IntentType.DATA_SEARCH, {"query": 1, "source": 2}),
    # "Find X datasets from Y" patterns  
    (r"(?:find|search\s+for|look\s+for|get)\s+(.+?)\s+(?:datasets?|data)(?:\s+from\s+(.+))?",
     IntentType.DATA_SEARCH, {"query": 1, "source": 2}),
    # "Find X ChIP-seq data" style
    (r"(?:find|search\s+for|look\s+for)\s+(.+?)\s+(?:ChIP-seq|RNA-seq|ATAC-seq|Hi-C|methylation|metagenomics)(?:\s+data)?",
     IntentType.DATA_SEARCH, {"query": 1}),
    # "Find publicly available X" 
    (r"find\s+(?:publicly\s+available|public)\s+(.+)",
     IntentType.DATA_SEARCH, {"query": 1}),
    # "Can you search X for Y"
    (r"(?:can\s+you\s+)?search\s+(\w+)\s+for\s+(.+)",
     IntentType.DATA_SEARCH, {"database": 1, "query": 2}),
    # "search ncbi sra for X data" - database in query
    (r"search\s+(?:ncbi\s+)?(?:sra|geo|encode|tcga|gdc)\s+for\s+(.+?)(?:\s+data)?",
     IntentType.DATA_SEARCH, {"query": 1}),
    
    # "Search ENCODE for X" pattern (database name first)
    (r"(?:search|query|browse)\s+(?:encode|geo|sra|tcga|gdc)\s+(?:for|database\s+for)\s+(.+)",
     IntentType.DATA_SEARCH, {"query": 1}),
    (r"(?:search|query|find|look)\s+(?:for\s+)?(.+?)\s+(?:in|on|from)\s+(?:encode|geo|sra|tcga|gdc|databases?)",
     IntentType.DATA_SEARCH, {"query": 1}),
    (r"(?:search|query|find)\s+(?:public\s+)?(?:databases?|online)\s+for\s+(.+)",
     IntentType.DATA_SEARCH, {"query": 1}),
    (r"(?:are\s+there|is\s+there)\s+(?:any\s+)?(.+?)\s+(?:datasets?|data)\s+(?:available\s+)?(?:online|in\s+(?:encode|geo|tcga))?",
     IntentType.DATA_SEARCH, {"query": 1}),
    
    # Simple search phrases: "organism tissue assay" format (no action words)
    # These are implicit search queries
    (r"^([a-z]+(?:\s+[a-z]+)?)\s+(brain|liver|heart|lung|kidney|blood|muscle|skin|bone\s+marrow|thymus|spleen|adipose|fat|pancreas|intestine|colon|ovary|testis|prostate|breast|thyroid|lymph\s+node|peripheral\s+blood)\s+(rnaseq|rna-seq|chipseq|chip-seq|atacseq|atac-seq|hic|hi-c|wgs|wes|rrbs|bisulfite|methylation|gro-seq|groseq|pro-seq|proseq|cut-?n?-?run|cutnrun)$",
     IntentType.DATA_SEARCH, {"organism": 1, "tissue": 2, "assay": 3}),
    
    # Search with exclusions - "skip X samples", "no X data", etc.
    (r"(?:skip|exclude|avoid|ignore)\s+(.+?)\s+(?:samples?|data)",
     IntentType.DATA_SEARCH, {"excluded": 1}),
    (r"(?:find|search|look\s+for)\s+(.+?)\s+data\s+(?:without|excluding|not|except)\s+(.+)",
     IntentType.DATA_SEARCH, {"query": 1, "excluded": 2}),
    (r"no\s+(.+?)\s+(?:data|samples?)",
     IntentType.DATA_SEARCH, {"excluded": 1}),
    
    # TCGA-specific search
    (r"(?:search|find|look\s+for)\s+(?:tcga|cancer)\s+(.+)",
     IntentType.DATA_SEARCH, {"query": 1, "source": "tcga"}),
    (r"(.+?)\s+(?:cancer|tumor)\s+(?:data|methylation|expression)",
     IntentType.DATA_SEARCH, {"cancer_type": 1}),
    
    # Download with exclusions
    (r"(?:download|get|fetch)\s+(?:all\s+)?(?:samples?|data|datasets?)\s+(?:except|without|excluding|but\s+not)\s+(.+)",
     IntentType.DATA_DOWNLOAD, {"excluded": 1}),
    # "fetch all X samples but not Y"
    (r"(?:download|get|fetch)\s+(?:all\s+)?(.+?)\s+(?:samples?|files?)(?:\s+(?:but\s+not|except)\s+(.+))?",
     IntentType.DATA_DOWNLOAD, {"query": 1, "excluded": 2}),
    
    # Download
    # Dataset IDs from various databases
    (r"(?:download|get|fetch)\s+(?:dataset\s+)?(GSE\d+|ENCSR[A-Z0-9]+|TCGA-[A-Z]+|PRJNA\d+|E-MTAB-\d+|SRR\d+|SRP\d+|SAMN\d+)",
     IntentType.DATA_DOWNLOAD, {"dataset_id": 1}),
    # "I want to fetch/download dataset XYZ"
    (r"i\s+want\s+to\s+(?:fetch|download|get)\s+(?:the\s+)?(?:dataset\s+)?([A-Za-z0-9\-]+)",
     IntentType.DATA_DOWNLOAD, {"dataset_id": 1}),
    (r"(?:download|get|fetch)\s+(?:this|that|the)\s+(?:dataset|data)",
     IntentType.DATA_DOWNLOAD, {}),
    # "download reference genome hg38" / "get reference genome"
    (r"(?:download|get|fetch)\s+(?:the\s+)?(?:reference\s+)?(?:genome|index|annotation)(?:\s+([a-zA-Z0-9_.-]+))?",
     IntentType.DATA_DOWNLOAD, {"reference": 1}),
    # "download the X data" (organism/species)
    (r"(?:download|get|fetch)\s+(?:the\s+)?(.+?)\s+data$",
     IntentType.DATA_DOWNLOAD, {"query": 1}),
    # "get the X files"
    (r"(?:download|get|fetch)\s+(?:the\s+)?(.+?)\s+files",
     IntentType.DATA_DOWNLOAD, {"query": 1}),
    # "save the data to ~/path"
    (r"save\s+(?:the\s+)?(?:data|files?)\s+to\s+([/~][^\s]+)",
     IntentType.DATA_DOWNLOAD, {"destination": 1}),
    (r"(?:add|queue)\s+(.+?)\s+(?:to\s+)?(?:my\s+)?(?:download|manifest)",
     IntentType.DATA_DOWNLOAD, {"dataset_id": 1}),
    # Download all / execute download commands
    (r"download\s+(?:all|both|everything)",
     IntentType.DATA_DOWNLOAD, {"download_all": True}),
    (r"(?:execute|run)\s+(?:the\s+)?download(?:s|\s+commands?)?",
     IntentType.DATA_DOWNLOAD, {"download_all": True}),
    (r"(?:please\s+)?(?:execute|run)\s+(?:the\s+)?(?:suggested\s+)?commands?",
     IntentType.DATA_DOWNLOAD, {"download_all": True}),
    (r"download\s+(?:the\s+)?(?:encode|tcga|geo)\s+(?:and\s+)?(?:encode|tcga|geo)?\s*(?:datasets?|data)?",
     IntentType.DATA_DOWNLOAD, {"download_all": True}),
    # Multi-turn context: "download the first one" / "get the second result"
    (r"(?:download|get|fetch)\s+(?:the\s+)?(?:first|second|third|last|top)\s+(?:one|result|dataset|sample)?",
     IntentType.DATA_DOWNLOAD, {"select_index": True}),

    # Multi-turn context: "only from X" / "in TCGA" / "from GSM" - refine search
    (r"^(?:only\s+)?(?:from|in)\s+(.+)$",
     IntentType.DATA_SEARCH, {"filter": 1}),
    (r"^only\s+from\s+(.+)$",
     IntentType.DATA_SEARCH, {"filter": 1}),
    
    # Multi-turn context: "show me the logs" - no job ID, assumes context
    (r"^show\s+(?:me\s+)?(?:the\s+)?logs?$",
     IntentType.JOB_LOGS, {}),

    # Workflow creation
    (r"(?:create|generate|make|build|set\s+up)\s+(?:a\s+)?(?:new\s+)?(.+?)\s+(?:workflow|pipeline|analysis)",
     IntentType.WORKFLOW_CREATE, {"workflow_type": 1}),
    # "set up X for Y" - setup pattern for organism-specific workflows
    (r"set\s+up\s+(.+?)\s+(?:for|on)\s+(.+)",
     IntentType.WORKFLOW_CREATE, {"workflow_type": 1, "organism": 2}),
    (r"(?:i\s+want\s+to|i\s+need\s+to|let's|can\s+you)\s+(?:run|do|perform)\s+(?:a\s+)?(.+?)\s+(?:analysis|workflow|pipeline)",
     IntentType.WORKFLOW_CREATE, {"workflow_type": 1}),
    # "run X analysis on Y data" - distinguish from DATA_SEARCH
    (r"(?:i\s+want\s+to\s+)?(?:run|perform|do)\s+(.+?)\s+analysis\s+(?:on|for)\s+(.+?)\s*(?:data)?$",
     IntentType.WORKFLOW_CREATE, {"workflow_type": 1, "organism": 2}),
    # "I want to analyze X data" (where X is organism/tissue) - workflow creation
    (r"i\s+want\s+to\s+analyze\s+(.+?)\s+data\s+(?:with|using)\s+(.+)",
     IntentType.WORKFLOW_CREATE, {"organism": 1, "analysis_type": 2}),
    # "I have X samples and want to find/analyze Y" - workflow creation from description
    (r"i\s+have\s+(?:.+?\s+)?(?:samples?|data)\s+(?:from\s+.+?\s+)?(?:and\s+)?(?:want|need)\s+to\s+(?:find|analyze|identify|detect)\s+(.+)",
     IntentType.WORKFLOW_CREATE, {"analysis_type": 1}),
    # "want to find differentially expressed genes" - DE analysis workflow
    (r"(?:want|need)\s+to\s+(?:find|identify|detect)\s+differentially\s+expressed\s+genes?",
     IntentType.WORKFLOW_CREATE, {"workflow_type": "differential_expression"}),
    # Simple "X analysis" / "X workflow" patterns for common bioinformatics workflows
    (r"^(cutnrun|cut\s*n\s*run|cut\s*and\s*run|cutandrun)\s+(?:analysis|workflow|pipeline)?$",
     IntentType.WORKFLOW_CREATE, {"workflow_type": "cut_and_run"}),
    (r"^(hi-?c|hic)\s+(?:analysis|workflow|pipeline)?$",
     IntentType.WORKFLOW_CREATE, {"workflow_type": "hi-c"}),
    (r"^(atac-?seq|chip-?seq|rna-?seq|wgs|wes|rrbs|bisulfite|methylation|proseq|pro-?seq)\s+(?:analysis|workflow|pipeline)?$",
     IntentType.WORKFLOW_CREATE, {"workflow_type": 1}),
    # "I need a X workflow" / "I need X workflow"
    (r"i\s+need\s+(?:a\s+)?(.+?)\s+(?:workflow|pipeline)",
     IntentType.WORKFLOW_CREATE, {"workflow_type": 1}),
    # "X workflow but use Y not Z" / "X workflow with Y"
    (r"^(.+?)\s+(?:workflow|pipeline)\s+(?:but\s+use|with)\s+(.+?)(?:\s+not\s+(.+))?$",
     IntentType.WORKFLOW_CREATE, {"workflow_type": 1, "preferred_tool": 2, "avoided_tool": 3}),
    # "X workflow for Y" / "variant calling workflow for homo sapiens"
    # CRITICAL: Use negative lookbehind to exclude submit/run/execute (those are JOB_SUBMIT)
    (r"^(?!submit\b|run\b|execute\b)(.+?)\s+(?:workflow|pipeline)\s+for\s+(.+)$",
     IntentType.WORKFLOW_CREATE, {"workflow_type": 1, "organism": 2}),
    # "first align with X, then do Y" - multi-step workflow description
    (r"first\s+(?:align|map)\s+with\s+(.+?)[,;]?\s+then\s+(?:do\s+)?(.+)",
     IntentType.WORKFLOW_CREATE, {"aligner": 1, "analysis_type": 2}),
    # "single cell analysis for X" 
    (r"single\s+cell\s+(?:analysis|workflow|pipeline)\s+for\s+(.+)",
     IntentType.WORKFLOW_CREATE, {"workflow_type": "single_cell", "organism": 1}),
    # "create workflow: X followed by Y"
    (r"(?:create\s+)?workflow:\s*(.+?)\s+followed\s+by\s+(.+)",
     IntentType.WORKFLOW_CREATE, {"step1": 1, "step2": 2}),
    # "cut&run analysis" (special characters)
    (r"^cut\s*[&n]\s*run\s+(?:analysis|workflow)?$",
     IntentType.WORKFLOW_CREATE, {"workflow_type": "cut_and_run"}),
    # "actually use X instead" - tool swap in workflow context
    (r"actually\s+use\s+(.+?)\s+instead",
     IntentType.WORKFLOW_CREATE, {"preferred_tool": 1}),
    # "add X step" - adding to workflow
    (r"add\s+(.+?)\s+step",
     IntentType.WORKFLOW_CREATE, {"add_step": 1}),
    
    # Tool preference (implies workflow context)
    (r"(?:use|prefer|choose)\s+(\w+(?:[_-]\w+)?)\s*(?:\w+)?\s*(?:instead\s+of|over|rather\s+than)\s+(\w+(?:[_-]\w+)?)",
     IntentType.WORKFLOW_CREATE, {"preferred_tool": 1, "avoided_tool": 2}),
    (r"i\s+prefer\s+(\w+(?:[_-]\w+)?)\s*(?:\w+)?\s*(?:over|to)\s+(\w+(?:[_-]\w+)?)",
     IntentType.WORKFLOW_CREATE, {"preferred_tool": 1, "avoided_tool": 2}),
    (r"(?:don't|do\s+not)\s+use\s+(\w+(?:[_-]\w+)?)\s*,?\s*(?:use|prefer)\s+(\w+(?:[_-]\w+)?)",
     IntentType.WORKFLOW_CREATE, {"avoided_tool": 1, "preferred_tool": 2}),
    # "I'd rather use X" / "I would prefer X"
    (r"i'?d\s+(?:rather|prefer\s+to)\s+use\s+(\w+(?:[_-]\w+)?)",
     IntentType.WORKFLOW_CREATE, {"preferred_tool": 1}),
    # "tool_A not tool_B" - standalone tool preference without context words
    (r"^(\w+(?:[_-]\w+)?)\s+not\s+(\w+(?:[_-]\w+)?)\s*$",
     IntentType.WORKFLOW_CREATE, {"preferred_tool": 1, "avoided_tool": 2}),
    
    # Workflow creation with exclusions (no/without/except)
    (r"(.+?)\s+(?:workflow|pipeline|analysis)\s+(?:without|excluding|but\s+(?:not|no))\s+(\w+)",
     IntentType.WORKFLOW_CREATE, {"workflow_type": 1, "excluded": 2}),
    (r"(?:variant\s+calling|peak\s+calling|alignment)\s+(?:without|not)\s+(?:using\s+)?(\w+)",
     IntentType.WORKFLOW_CREATE, {"excluded_tool": 1}),
    
    # Analyze with organism preference/exclusion
    (r"(?:analyze|process)\s+(\w+)\s+data\s*,?\s*(?:not|exclude|skip)\s+(\w+)",
     IntentType.WORKFLOW_CREATE, {"organism": 1, "excluded_organism": 2}),
    
    # Run/create with exclusion patterns
    (r"(?:run|perform|do)\s+(.+?)\s+(?:exclude|excluding|without)\s+(.+)",
     IntentType.WORKFLOW_CREATE, {"analysis_type": 1, "excluded": 2}),
    
    # Peak calling with tool preferences
    (r"(?:create\s+)?peak\s+calling\s+(?:avoid|without|skip)\s+(\w+)\s+(?:use|with)\s+(\w+)",
     IntentType.WORKFLOW_CREATE, {"workflow_type": "peak_calling", "avoided_tool": 1, "preferred_tool": 2}),
    
    # =========================================================================
    # JOB_RESUBMIT - must come before JOB_SUBMIT to avoid "submit" matching in "resubmit"
    # =========================================================================
    (r"^resubmit\b", IntentType.JOB_RESUBMIT, {}),  # Any phrase starting with resubmit
    (r"^retry\b", IntentType.JOB_RESUBMIT, {}),      # Any phrase starting with retry
    (r"^rerun\b", IntentType.JOB_RESUBMIT, {}),      # Any phrase starting with rerun
    (r"(?:resubmit|retry|rerun|restart)\s+(?:the\s+)?(?:failed\s+)?(?:job|run|analysis|workflow|pipeline)(?:\s+(\d+))?",
     IntentType.JOB_RESUBMIT, {"job_id": 1}),
    (r"(?:try\s+)?(?:the\s+)?(?:job|run)(?:\s+(\d+))?\s+again",
     IntentType.JOB_RESUBMIT, {"job_id": 1}),
    
    # Job submission - "submit my X workflow" means run existing workflow
    # CRITICAL: Full sentence patterns first (higher coverage = higher confidence)
    # "submit the X workflow for my Y data" - complete pattern with data
    (r"^submit\s+(?:the\s+)?(\S+(?:-\S+)?)\s+(?:workflow|pipeline)\s+(?:for|on|with)\s+(?:my\s+)?(?:\S+\s+)?(?:data|samples?)$",
     IntentType.JOB_SUBMIT, {"workflow_type": 1}),
    # "submit the X workflow for my data" (Y is data type)
    (r"submit\s+(?:the\s+)?(.+?)\s+(?:workflow|pipeline)\s+(?:for|on|with)\s+(?:my\s+)?(.+)$",
     IntentType.JOB_SUBMIT, {"workflow_type": 1, "data_type": 2}),
    # These patterns must be specific enough to catch "submit" + workflow type
    (r"(?:please\s+)?submit\s+(?:my\s+)?(?:the\s+)?(.+?)\s+(?:workflow|pipeline|job|analysis)",
     IntentType.JOB_SUBMIT, {"workflow_type": 1}),
    (r"(?:please\s+)?submit\s+(?:my\s+)?(?:the\s+)?(\w+(?:\s+\w+)?)\s+(?:workflow|pipeline)",
     IntentType.JOB_SUBMIT, {"workflow_type": 1}),
    # "submit the X workflow for my Y data" - Y is data type, not workflow type
    (r"submit\s+(?:the\s+)?(\w+(?:-\w+)?)\s+workflow\s+(?:for|on|with)\s+(?:my\s+)?",
     IntentType.JOB_SUBMIT, {"workflow_type": 1}),
    (r"(?:can\s+you\s+)?(?:run|execute)\s+(?:the\s+)?(?:pipeline|workflow)\s+(?:in|at|from)\s+([/~][^\s]+)",
     IntentType.JOB_SUBMIT, {"path": 1}),
    (r"(?:run|execute|submit|start)\s+(?:the\s+)?(?:workflow|pipeline|job|analysis)",
     IntentType.JOB_SUBMIT, {}),
    (r"submit\s+(?:the\s+)?workflow\s+(?:in|at|from)\s+([/~][^\s]+)",
     IntentType.JOB_SUBMIT, {"path": 1}),
    (r"(?:send|submit)\s+(?:it|this)\s+to\s+(?:slurm|cluster|hpc)",
     IntentType.JOB_SUBMIT, {}),
    (r"(?:now\s+)?submit\s+it",
     IntentType.JOB_SUBMIT, {}),
    # Short forms: "run it", "submit", "execute it", "execute" (in context of workflow)
    (r"^(?:run\s+it|submit|execute\s+it|execute)$",
     IntentType.JOB_SUBMIT, {}),
    # "execute /path/to/workflow"
    (r"execute\s+([/~][^\s]+)",
     IntentType.JOB_SUBMIT, {"path": 1}),
    
    # Job status
    (r"(?:what\s+is|check|show|get)\s+(?:the\s+)?(?:job\s+)?status(?:\s+(?:of|for)\s+(?:job\s+)?(\d+))?",
     IntentType.JOB_STATUS, {"job_id": 1}),
    (r"(?:is\s+(?:the\s+)?job|are\s+(?:the\s+)?jobs?)\s+(?:still\s+)?(?:running|done|finished|complete)",
     IntentType.JOB_STATUS, {}),
    # Short forms: "is it done?", "job status"
    (r"^(?:is\s+it\s+done|job\s+status|status)(?:\?)?$",
     IntentType.JOB_STATUS, {}),
    # "is the X analysis done?"
    (r"is\s+(?:the\s+)?(.+?)\s+(?:analysis\s+)?(?:done|finished|complete)(?:\?)?$",
     IntentType.JOB_STATUS, {"job_type": 1}),
    # "how is my job doing?"
    (r"how\s+is\s+(?:my\s+)?(?:job|analysis)\s+(?:doing|going)(?:\?)?",
     IntentType.JOB_STATUS, {}),
    # "what's happening with job X" - extended to capture job ID for better coverage
    (r"(?:how\s+is|what's\s+happening\s+with)\s+(?:the\s+)?(?:job|analysis|run)(?:\s+(\d+))?",
     IntentType.JOB_STATUS, {"job_id": 1}),
    # "what's the status of job X" must come before education patterns
    (r"what(?:'s|\s+is)\s+(?:the\s+)?status\s+(?:of|for)\s+(?:job\s+)?(\d+)",
     IntentType.JOB_STATUS, {"job_id": 1}),
    (r"(?:check|show|get)\s+(?:on\s+)?(?:job\s+)?(\d+)",
     IntentType.JOB_STATUS, {"job_id": 1}),
    (r"(?:check|see|look)\s+(?:on\s+)?(?:it|the\s+job)",
     IntentType.JOB_STATUS, {}),
    
    # Logs
    # "view job 27548 logs" - job ID before logs
    (r"(?:show|get|view|display|i\s+want\s+to\s+view)\s+(?:the\s+)?job\s+(\d+)\s+logs?",
     IntentType.JOB_LOGS, {"job_id": 1}),
    # "get output of 12345" / "show error logs for 12345"
    (r"(?:get|show)\s+(?:the\s+)?(?:output|error\s+logs?|logs?)\s+(?:of|for)\s+(?:job\s+)?(\d+)",
     IntentType.JOB_LOGS, {"job_id": 1}),
    # "get error output for job123" - alphanumeric job IDs
    (r"(?:get|show)\s+(?:the\s+)?(?:error\s+)?(?:output|logs?)\s+(?:of|for)\s+(?:job)?(\w+)",
     IntentType.JOB_LOGS, {"job_id": 1}),
    (r"(?:show|get|view|display)\s+(?:the\s+)?(?:job\s+)?logs?(?:\s+(?:for|of)\s+(?:job\s+)?(\d+))?",
     IntentType.JOB_LOGS, {"job_id": 1}),
    (r"(?:what\s+happened|what\s+went\s+wrong|show\s+me\s+(?:the\s+)?(?:error|output))",
     IntentType.JOB_LOGS, {}),
    # "what's the output of job X" - with optional "please" prefix
    (r"(?:please\s+)?what(?:'s|\s+is)\s+(?:the\s+)?output\s+(?:of|for)\s+(?:job\s+)?(\d+)",
     IntentType.JOB_LOGS, {"job_id": 1}),
    
    # Job List (must come before more generic patterns)
    (r"list\s+(?:all\s+)?(?:my\s+)?(?:running\s+)?jobs",
     IntentType.JOB_LIST, {}),
    (r"list\s+(?:my\s+)?submitted\s+jobs",
     IntentType.JOB_LIST, {}),
    (r"(?:show|display|get)\s+(?:all\s+)?(?:my\s+)?(?:running\s+)?jobs",
     IntentType.JOB_LIST, {}),
    # "show active jobs" / "Can you show active jobs"
    (r"(?:can\s+you\s+)?(?:show|display|get|list)\s+(?:the\s+)?(?:active|running|queued|pending)\s+jobs",
     IntentType.JOB_LIST, {}),
    (r"what\s+jobs\s+are\s+(?:running|active|pending|queued)",
     IntentType.JOB_LIST, {}),
    (r"what(?:'s|\s+is)\s+(?:currently\s+)?running",
     IntentType.JOB_LIST, {}),
    (r"(?:i\s+want\s+to\s+)?(?:see|view|check)\s+(?:my\s+)?(?:running\s+)?jobs",
     IntentType.JOB_LIST, {}),
    (r"(?:please\s+)?list\s+all\s+jobs",
     IntentType.JOB_LIST, {}),
    (r"(?:i\s+want\s+to\s+)?(?:list|see|check)\s+(?:all\s+)?jobs",
     IntentType.JOB_LIST, {}),
    (r"what\s+(?:all\s+)?jobs",
     IntentType.JOB_LIST, {}),
    
    # Cancel
    (r"(?:cancel|stop|kill|abort)\s+(?:the\s+)?(?:job|run|analysis)(?:\s+(\d+))?",
     IntentType.JOB_CANCEL, {"job_id": 1}),
    
    # Watch/Monitor (must come before JOB_STATUS for "monitor" to work)
    (r"(?:watch|monitor|follow|track)\s+(?:the\s+)?(?:job|run|pipeline|analysis)(?:\s+progress)?(?:\s+(\d+))?",
     IntentType.JOB_WATCH, {"job_id": 1}),
    (r"(?:watch|monitor|follow|track)\s+(?:the\s+)?progress(?:\s+(?:of\s+)?(?:job|run)?\s*(\d+))?",
     IntentType.JOB_WATCH, {"job_id": 1}),
    (r"(?:keep\s+an?\s+)?eye\s+on\s+(?:the\s+)?(?:job|run|pipeline)(?:\s+(\d+))?",
     IntentType.JOB_WATCH, {"job_id": 1}),
    (r"monitor\s+(?:my\s+)?(?:the\s+)?(?:job|run|pipeline|analysis|workflow)",
     IntentType.JOB_WATCH, {}),
    
    # Resubmit/Retry
    (r"(?:resubmit|retry|rerun|restart)\s+(?:the\s+)?(?:failed\s+)?(?:job|run|analysis)(?:\s+(\d+))?",
     IntentType.JOB_RESUBMIT, {"job_id": 1}),
    (r"(?:try\s+)?(?:the\s+)?(?:job|run)(?:\s+(\d+))?\s+again",
     IntentType.JOB_RESUBMIT, {"job_id": 1}),
    (r"(?:resubmit|retry|rerun)\s+(?:failed\s+)?(?:job|run)",
     IntentType.JOB_RESUBMIT, {}),
    
    # Diagnostics
    (r"(?:diagnose|debug|troubleshoot|analyze)\s+(?:this\s+)?(?:error|failure|problem)",
     IntentType.DIAGNOSE_ERROR, {}),
    (r"(?:what\s+went\s+wrong|why\s+did\s+it\s+fail|fix\s+this|help\s+me\s+fix)",
     IntentType.DIAGNOSE_ERROR, {}),
    # "what went wrong and how do I fix it"
    (r"what\s+went\s+wrong\s+(?:and\s+)?(?:how\s+(?:do\s+i|can\s+i|to)\s+)?(?:fix|solve|resolve)",
     IntentType.DIAGNOSE_ERROR, {}),
    # ADDED: Broader troubleshooting patterns
    (r"(?:my\s+)?(?:pipeline|workflow|job|run|script)\s+(?:failed|crashed|errored|broke|stopped|died)",
     IntentType.DIAGNOSE_ERROR, {}),
    (r"(?:got|getting|have|having|received?|seeing?)\s+(?:an?\s+)?(?:error|failure|exception|problem|issue)",
     IntentType.DIAGNOSE_ERROR, {}),
    (r"(?:it'?s?\s+)?(?:not\s+working|broken|failing|crashing|erroring)",
     IntentType.DIAGNOSE_ERROR, {}),
    (r"(?:error|failure|exception|problem|issue)\s+(?:with|in|during|when|while)",
     IntentType.DIAGNOSE_ERROR, {}),
    (r"(?:try\s+to\s+)?(?:fix|recover|resolve)\s+(?:this|the)\s+(?:error|issue|problem)",
     IntentType.DIAGNOSE_RECOVER, {}),
    
    # =========================================================================
    # REFERENCE CHECK PATTERNS - checking local reference genome availability
    # =========================================================================
    # "check if we have the human reference genome" (explicit check pattern)
    (r"check\s+if\s+(?:we\s+have|there\s+is)\s+(?:the\s+)?(?:a\s+)?(?:human|mouse|rat|zebrafish)?\s*(?:reference\s+)?(?:genome|annotation)",
     IntentType.REFERENCE_CHECK, {}),
    # "check if human reference genome is available" / "is hg38 available?"
    (r"(?:check\s+(?:if\s+)?|is\s+(?:the\s+)?|do\s+(?:we|i)\s+have\s+)(?:the\s+)?(?:human|mouse|rat|hg38|hg19|mm10|mm39|GRCh38|GRCh37|GRCm39|GRCm38)\s+(?:reference\s+)?(?:genome|assembly|index)?(?:\s+(?:available|installed|ready|present))?",
     IntentType.REFERENCE_CHECK, {}),
    # "check reference genome availability" / "check for hg38"
    (r"(?:check|verify|see|look)\s+(?:for\s+)?(?:the\s+)?(?:reference\s+)?(?:genome|index|annotation)\s+(?:availability|status)",
     IntentType.REFERENCE_CHECK, {}),
    (r"(?:check|look)\s+(?:for|if)\s+(?:the\s+)?(?:reference\s+)?(?:genome|index)(?:\s+(?:exists?|is\s+there))?",
     IntentType.REFERENCE_CHECK, {}),
    # "is reference genome available" / "do we have the reference"
    (r"(?:is\s+(?:the\s+)?|do\s+(?:we|i)\s+have\s+(?:a\s+|the\s+)?)(?:reference\s+)?(?:genome|index|annotation)(?:\s+(?:available|installed|ready|set\s+up))?",
     IntentType.REFERENCE_CHECK, {}),
    # "verify reference files" / "check reference status"
    (r"(?:verify|check|validate)\s+(?:the\s+)?(?:reference|genome|index)\s+(?:files?|status|setup)",
     IntentType.REFERENCE_CHECK, {}),
    # "which references are available" / "list available genomes"
    (r"(?:which|what|list)\s+(?:reference\s+)?(?:genomes?|references?|indices)\s+(?:are\s+)?(?:available|installed|ready)",
     IntentType.REFERENCE_CHECK, {}),
    
    # =========================================================================
    # SYSTEM COMMAND PATTERNS - run simple commands (NOT workflow submission)
    # These must come BEFORE JOB_SUBMIT patterns to take priority
    # =========================================================================
    # "run fastqc --version" / "run samtools view" - direct command execution
    (r"(?:run|execute)\s+(fastqc|samtools|bwa|bowtie|bowtie2|hisat2|star|kallisto|salmon|picard|gatk|bcftools|bedtools|deeptools|macs2|macs3|htseq|featurecounts|multiqc|trimmomatic|cutadapt|trim_galore)\s+(--?[\w-]+(?:\s+--?[\w-]+)*)",
     IntentType.SYSTEM_COMMAND, {"tool": 1, "args": 2}),
    # "run tool --version" / "run tool --help" pattern specifically  
    (r"(?:run|execute|check)\s+(\w+)\s+(--version|--help|-v|-h)\s*$",
     IntentType.SYSTEM_COMMAND, {"tool": 1, "args": 2}),
    # "fastqc --version" / "samtools --version" - direct command without "run"
    (r"^(fastqc|samtools|bwa|bowtie|bowtie2|hisat2|star|kallisto|salmon|picard|gatk|bcftools|bedtools|deeptools|macs2|macs3|htseq|featurecounts|multiqc|trimmomatic|cutadapt|trim_galore|nextflow|snakemake)\s+(--version|--help|-v|-h)\s*$",
     IntentType.SYSTEM_COMMAND, {"tool": 1, "args": 2}),
    # "what version of X" / "check X version" / "X version"
    (r"(?:what\s+(?:is\s+the\s+)?version\s+(?:of\s+)?|check\s+(?:the\s+)?(?:version\s+(?:of\s+)?)?)(fastqc|samtools|bwa|bowtie|bowtie2|hisat2|star|kallisto|salmon|picard|gatk|bcftools|bedtools|deeptools|macs|macs2|macs3|htseq|featurecounts|multiqc|trimmomatic|cutadapt|trim_galore|nextflow|snakemake)",
     IntentType.SYSTEM_COMMAND, {"tool": 1, "args": "--version"}),
    # "tool version" / "tool version?" (short form)
    (r"^(fastqc|samtools|bwa|bowtie|bowtie2|hisat2|star|kallisto|salmon|picard|gatk|bcftools|bedtools|deeptools|macs|macs2|macs3|htseq|featurecounts|multiqc|trimmomatic|cutadapt|trim_galore|nextflow|snakemake)\s+version\s*\??$",
     IntentType.SYSTEM_COMMAND, {"tool": 1, "args": "--version"}),
    # "is X installed" / "do we have X" - tool availability check
    (r"(?:is\s+|do\s+(?:we|i)\s+have\s+)(fastqc|samtools|bwa|bowtie|bowtie2|hisat2|star|kallisto|salmon|picard|gatk|bcftools|bedtools|deeptools|macs|macs2|macs3|htseq|featurecounts|multiqc|trimmomatic|cutadapt|trim_galore|nextflow|snakemake)(?:\s+installed)?",
     IntentType.SYSTEM_COMMAND, {"tool": 1, "check_installed": True}),
    # "is X available" / "can I use X" - tool availability
    (r"(?:is\s+|can\s+(?:i|we)\s+use\s+)(fastqc|samtools|bwa|bowtie|bowtie2|hisat2|star|kallisto|salmon|picard|gatk|bcftools|bedtools|deeptools|macs|macs2|macs3|htseq|featurecounts|multiqc|trimmomatic|cutadapt|trim_galore|nextflow|snakemake)(?:\s+(?:available|here))?",
     IntentType.SYSTEM_COMMAND, {"tool": 1, "check_installed": True}),
    # "where is X installed" / "path to X"
    (r"(?:where\s+is|path\s+to|locate)\s+(fastqc|samtools|bwa|bowtie|bowtie2|hisat2|star|kallisto|salmon|picard|gatk|bcftools|bedtools|deeptools|macs|macs2|macs3|htseq|featurecounts|multiqc|trimmomatic|cutadapt|trim_galore|nextflow|snakemake)",
     IntentType.SYSTEM_COMMAND, {"tool": 1, "check_installed": True}),
    # "which tools are available" / "list installed tools"
    (r"(?:which|what|list)\s+(?:bioinformatics\s+)?tools?\s+(?:are\s+)?(?:available|installed)",
     IntentType.SYSTEM_STATUS, {}),
    # "check system status" / "system health"
    (r"(?:check\s+)?(?:system|cluster|slurm)\s+(?:status|health)",
     IntentType.SYSTEM_STATUS, {}),
    
    # =========================================================================
    # WORKFLOW VALIDATION/MODIFICATION PATTERNS
    # =========================================================================
    # "validate the workflow" / "check the workflow" / "validate the complete workflow"
    (r"(?:validate|verify|check)\s+(?:the\s+)?(?:complete\s+)?(?:workflow|pipeline)(?:\s+(?:before\s+)?(?:running|submitting))?",
     IntentType.WORKFLOW_VALIDATE, {}),
    (r"(?:validate|verify|check)\s+(?:the\s+)?(?:changes|modifications)",
     IntentType.WORKFLOW_VALIDATE, {}),
    # "modify the workflow" / "change the workflow" / "update the workflow"
    (r"(?:modify|change|update|edit|fix)\s+(?:the\s+)?(?:workflow|pipeline)",
     IntentType.WORKFLOW_MODIFY, {}),
    # "apply that fix" / "apply the fix to the workflow"
    (r"(?:apply|use|implement)\s+(?:that|the)\s+(?:fix|change|modification|suggestion)",
     IntentType.WORKFLOW_MODIFY, {}),
    # "use X instead" / "switch to X"
    (r"(?:use|switch\s+to)\s+(\w+[-\w]*)\s+instead",
     IntentType.WORKFLOW_MODIFY, {"new_tool": 1}),
    # "add X step" / "add X to the workflow"  
    (r"(?:also\s+)?add\s+(.+?)(?:\s+to\s+(?:the\s+)?(?:workflow|pipeline))?$",
     IntentType.WORKFLOW_MODIFY, {"add_step": 1}),
    # "what aligner/tool did you use" - workflow describe
    (r"what\s+(?:aligner|tool|version|step)\s+(?:did\s+you|are\s+you)\s+(?:use|using)",
     IntentType.WORKFLOW_DESCRIBE, {}),
    (r"(?:describe|show|explain)\s+(?:the\s+)?(?:workflow|pipeline)(?:\s+(?:details?|steps?|configuration))?",
     IntentType.WORKFLOW_DESCRIBE, {}),
    
    # =========================================================================
    # JOB MONITORING/RESOURCE PATTERNS
    # =========================================================================
    # "how much memory/cpu is it using"
    (r"(?:how\s+much|what)\s+(?:memory|cpu|resources?|disk)\s+(?:is\s+)?(?:it|the\s+job)?\s*(?:using|usage)?",
     IntentType.JOB_RESOURCES, {}),
    (r"(?:show|check|get)\s+(?:resource|memory|cpu)\s+(?:usage|consumption|stats?)",
     IntentType.JOB_RESOURCES, {}),
    # "is it running yet" / "is it done"
    (r"^is\s+it\s+(?:running|done|finished|complete|started)(?:\s+yet)?(?:\?)?$",
     IntentType.JOB_STATUS, {}),
    # "monitor it" / "watch the job"
    (r"(?:monitor|watch|track)\s+(?:it|the\s+job|the\s+run)",
     IntentType.JOB_WATCH, {}),
    (r"(?:let\s+me\s+know|notify\s+me|alert\s+me)\s+(?:if|when)\s+(?:it|the\s+job)",
     IntentType.JOB_WATCH, {}),
    
    # =========================================================================
    # ANALYSIS/RESULTS PATTERNS
    # =========================================================================
    # "interpret the results" / "what are the results"
    (r"(?:what\s+are|show\s+me)\s+(?:the\s+)?(?:key\s+)?(?:results?|findings?|output)",
     IntentType.ANALYSIS_INTERPRET, {}),
    # "generate a report" / "create a QC report"
    (r"(?:generate|create|make|build)\s+(?:a\s+)?(?:qc\s+)?(?:report|summary)",
     IntentType.ANALYSIS_REPORT, {}),
    (r"(?:generate|create)\s+(?:a\s+)?(?:report|summary)\s+(?:i\s+can\s+)?(?:share|send)",
     IntentType.ANALYSIS_REPORT, {}),
    
    # =========================================================================
    # DATA COMPARISON/FILTERING/VALIDATION PATTERNS
    # =========================================================================
    # "compare the results" / "compare datasets"
    (r"compare\s+(?:the\s+)?(?:quality\s+of\s+)?(?:results?|datasets?|data)",
     IntentType.DATA_COMPARE, {}),
    # "filter to only" / "filter datasets"
    (r"filter\s+(?:to\s+)?(?:only\s+)?(?:datasets?|results?|those)",
     IntentType.DATA_FILTER, {}),
    (r"(?:only\s+)?(?:show|keep)\s+(?:datasets?|results?)\s+(?:with|where|that)",
     IntentType.DATA_FILTER, {}),
    # "verify the download" / "check the data"
    (r"(?:verify|validate|check)\s+(?:the\s+)?(?:downloads?|data)(?:\s+(?:completed|finished|successfully))?",
     IntentType.DATA_VALIDATE, {}),
    
    # =========================================================================
    # CONTEXTUAL SHORT PHRASES
    # =========================================================================
    # "let's run it" / "run it" / "submit it" - context-dependent
    (r"^(?:let'?s?\s+)?(?:run|submit|execute)\s+it!?$",
     IntentType.JOB_SUBMIT, {}),
    # "great, let's run it" / "ok run it" - enthusiasm + action
    (r"^(?:great|ok|okay|perfect|good|nice)[,!]?\s+(?:let'?s?\s+)?(?:run|submit)\s+it!?$",
     IntentType.JOB_SUBMIT, {}),
    # "looks good" / "that's fine" - confirmation before submit
    (r"^(?:looks?\s+good|that'?s?\s+(?:fine|good|great)|ok|okay),?\s*(?:submit|run)?\s*(?:it)?$",
     IntentType.JOB_SUBMIT, {}),
    # "show me the logs" - short form
    (r"^show\s+(?:me\s+)?(?:the\s+)?logs?(?:\s+so\s+far)?$",
     IntentType.JOB_LOGS, {}),
    
    # Analysis/Results
    (r"(?:analyze|interpret|explain)\s+(?:the\s+)?(?:results?|output|qc)",
     IntentType.ANALYSIS_INTERPRET, {}),
    (r"(?:what\s+do\s+(?:the\s+)?results?\s+(?:mean|show)|summarize\s+(?:the\s+)?results?)",
     IntentType.ANALYSIS_INTERPRET, {}),
    
    # =========================================================================
    # ADVERSARIAL PATTERNS - disambiguate tricky queries
    # =========================================================================
    # "download instructions for X" / "download information about X" - wants documentation
    (r"download\s+(?:the\s+)?(?:instructions?|information|info|details?|docs?)\s+(?:for|on|about)\s+(.+)",
     IntentType.EDUCATION_EXPLAIN, {"concept": 1}),
    
    
    # "this is not about X, it's about Y" - clarification, wants search for Y
    # FIXED: use .+? instead of \S+ to match multi-word phrases like "whole exome"
    (r"(?:this\s+is\s+not|it'?s?\s+not)\s+about\s+.+?[,;]\s+(?:it'?s?|this\s+is)\s+about\s+(.+)",
     IntentType.DATA_SEARCH, {"query": 1}),
    
    # "don't create/download/X, just search" - negation with override
    (r"(?:don'?t|do\s+not)\s+(?:create|build|make|download|get)\s+(?:a\s+)?(?:\w+)?[,;]?\s+(?:just\s+)?(?:search|find|look)",
     IntentType.DATA_SEARCH, {}),
    
    # "I don't want to X, just search" - negation with override  
    (r"i\s+don'?t\s+want\s+to\s+(?:create|download|build|make)[,;]?\s+(?:just\s+)?(?:search|find|look)",
     IntentType.DATA_SEARCH, {}),
    
    # "I don't want X, find something else" - search with exclusion
    (r"i\s+don'?t\s+want\s+(.+?)[,;]?\s+(?:find|search|show)\s+(?:something\s+else|alternatives?|others?)",
     IntentType.DATA_SEARCH, {"excluded": 1}),
    
    # "forget about searching/X, let's download" - override with second intent
    (r"forget\s+(?:about\s+)?(?:the\s+)?(?:search(?:ing)?|\w+)[,;]?\s+(?:let'?s?\s+)?(?:download|get)",
     IntentType.DATA_DOWNLOAD, {}),
    
    # "Create a search for X" - the word "create" is misleading, it's a search
    (r"(?:create|make|start)\s+a\s+search\s+(?:for|of)\s+(.+)",
     IntentType.DATA_SEARCH, {"query": 1}),
    # "Search for a way to X" - wants tutorial/explanation, not data search
    (r"search\s+for\s+(?:a\s+)?(?:way|method|approach)\s+to\s+(.+)",
     IntentType.EDUCATION_EXPLAIN, {"topic": 1}),
    # "find help about X" / "get help on X" - wants education
    (r"(?:find|get|need)\s+help\s+(?:about|on|with)\s+(.+)",
     IntentType.EDUCATION_EXPLAIN, {"topic": 1}),
    # "Forget X, just Y" - override with second intent
    (r"forget\s+(?:about\s+)?(?:the\s+)?(?:\w+)[,;]?\s+(?:just\s+)?(?:show|check|get)\s+(?:the\s+)?status",
     IntentType.JOB_STATUS, {}),
    (r"forget\s+(?:about\s+)?(?:the\s+)?(?:\w+)[,;]?\s+(?:just\s+)?(?:list|show)\s+(?:the\s+)?jobs",
     IntentType.JOB_LIST, {}),
    # "explain how to search for data" - tutorial, not search
    (r"explain\s+(?:how\s+to|me\s+how\s+to)\s+(?:search|find|look)\s+(?:for\s+)?(?:data|datasets?|samples?)",
     IntentType.EDUCATION_TUTORIAL, {"topic": "data_search"}),
    
    # =========================================================================
    # AMBIGUOUS/VAGUE PATTERNS - should trigger clarification
    # =========================================================================
    # Single word queries that are too vague
    (r"^(?:data|analysis|samples?|files?|help|workflow|pipeline)$",
     IntentType.META_UNKNOWN, {"needs_clarification": True}),
    # Very vague statements
    (r"^(?:i\s+have|i'?ve?\s+got)\s+(?:some|a\s+few)?\s*(?:data|files?|samples?)$",
     IntentType.META_UNKNOWN, {"needs_clarification": True}),
    (r"^process\s+(?:my|the)?\s*(?:data|files?|samples?)$",
     IntentType.META_UNKNOWN, {"needs_clarification": True}),
    
    # Education
    (r"(?:what\s+is|what's|explain|describe|tell\s+me\s+about)\s+(.+?)(?:\?|$)",
     IntentType.EDUCATION_EXPLAIN, {"concept": 1}),
    (r"(?:how\s+do\s+i|how\s+to|how\s+can\s+i)\s+(.+)",
     IntentType.EDUCATION_TUTORIAL, {"topic": 1}),
    (r"^(?:help|commands?|what\s+can\s+you\s+do|\?+|show\s+help|list\s+commands?)$",
     IntentType.EDUCATION_HELP, {}),
    (r"what\s+(?:can\s+(?:you|i)\s+do|are\s+(?:your|my)\s+(?:options|capabilities))",
     IntentType.EDUCATION_HELP, {}),
    # "Show me available commands" / "What features are available?" / "List capabilities"
    (r"(?:show\s+(?:me\s+)?)?(?:available|all)\s+(?:commands?|features?|options)",
     IntentType.EDUCATION_HELP, {}),
    (r"what\s+features\s+are\s+available",
     IntentType.EDUCATION_HELP, {}),
    (r"list\s+(?:all\s+)?(?:capabilities|features|commands?|options)",
     IntentType.EDUCATION_HELP, {}),
    
    # Meta/Conversational
    (r"^(?:yes|yeah|yep|ok|okay|sure|confirm|do\s+it|go\s+ahead|proceed)$",
     IntentType.META_CONFIRM, {}),
    (r"^(?:no|nope|cancel|stop|never\s+mind|abort)$",
     IntentType.META_CANCEL, {}),
    (r"^(?:hi|hello|hey|greetings)(?:\s+there)?!?$",
     IntentType.META_GREETING, {}),
    (r"^(?:thanks?|thank\s+you|thx|ty)!?$",
     IntentType.META_THANKS, {}),
    (r"(?:undo|go\s+back|revert)\s+(?:that|last|previous)",
     IntentType.META_UNDO, {}),
    
    # =========================================================================
    # META_CORRECT - User correcting previous statement (MUST be explicit corrections)
    # These patterns require context words like "actually", "wait", "meant", etc.
    # Simple "X not Y" should NOT trigger META_CORRECT - those are preferences
    # =========================================================================
    # "actually X not Y" / "actually, X not Y" (explicit correction marker)
    (r"^(?:actually|wait)[,]?\s+(?:i\s+meant\s+)?(\w+(?:[_\-]\w+)?)\s+not\s+(\w+(?:[_\-]\w+)?)\s*$",
     IntentType.META_CORRECT, {"correct_to": 1, "incorrect": 2}),
    # "wait, I meant X not Y" / "I meant X not Y"
    (r"(?:wait[,]?\s+)?i\s+meant\s+(\w+(?:[_\-]\w+)?)\s+not\s+(\w+(?:[_\-]\w+)?)",
     IntentType.META_CORRECT, {"correct_to": 1, "incorrect": 2}),
    # "no, use X instead" / "no use X" (short correction response)
    (r"^no[,]?\s+use\s+(\w+(?:[_\-]\w+)?)\s*(?:instead)?\s*$",
     IntentType.META_CORRECT, {"correct_to": 1}),
    # "change to X" / "switch to X" (short command)
    (r"^(?:change|switch)\s+to\s+(\w+(?:[_\-]\w+)?)\s*$",
     IntentType.META_CORRECT, {"correct_to": 1}),
    # "I meant X" (without negation, standalone)
    (r"^i\s+meant\s+(\w+(?:[_\-]\w+)?)\s*$",
     IntentType.META_CORRECT, {"correct_to": 1}),
]


# =============================================================================
# ENTITY EXTRACTION PATTERNS
# =============================================================================

class EntityPatterns:
    """Domain-specific entity extraction patterns."""
    
    # Dataset IDs
    DATASET_PATTERNS = [
        (r'\b(GSE\d+)\b', EntityType.DATASET_ID, "geo"),
        (r'\b(ENCSR[A-Z0-9]+)\b', EntityType.DATASET_ID, "encode"),
        (r'\b(SRR\d+|SRP\d+|SRX\d+)\b', EntityType.DATASET_ID, "sra"),
        (r'\b(TCGA-[A-Z]{2,4})\b', EntityType.PROJECT_ID, "tcga"),
        (r'\b(PRJNA\d+)\b', EntityType.PROJECT_ID, "bioproject"),
    ]
    
    # Paths
    PATH_PATTERNS = [
        (r'(?:scan|check|look|search|in|at|from|to|submit)\s+(?:\w+\s+)*?["\']?(/[^\s"\']+|~/[^\s"\']+)["\']?', EntityType.DIRECTORY_PATH),
        (r'(?:file\s+)?["\']?(/[^\s"\']+\.(?:fastq|fq|bam|bed|vcf|txt|csv|tsv)(?:\.gz)?)["\']?', EntityType.FILE_PATH),
        # Bare path at word boundary (for cases like "scan /data/raw")
        (r'\b(/[a-zA-Z0-9_/.-]+)(?:\s|$)', EntityType.DIRECTORY_PATH),
    ]
    
    # Organisms
    ORGANISMS = {
        "human": ["human", "homo sapiens", "hs", "hg38", "hg19", "grch38", "grch37"],
        "mouse": ["mouse", "mus musculus", "mm", "mm10", "mm9", "grcm39", "grcm38"],
        "rat": ["rat", "rattus norvegicus", "rn", "rn6", "rn7"],
        "zebrafish": ["zebrafish", "danio rerio", "dr", "grcz11"],
        "fly": ["fly", "drosophila", "drosophila melanogaster", "dm6"],
        "worm": ["worm", "c elegans", "caenorhabditis elegans", "ce11"],
        "yeast": ["yeast", "saccharomyces cerevisiae", "sc", "s288c"],
    }
    
    # Tissues
    TISSUES = [
        "brain", "liver", "heart", "kidney", "lung", "muscle", "spleen",
        "intestine", "colon", "skin", "blood", "bone marrow", "pancreas",
        "breast", "prostate", "ovary", "testis", "thyroid", "stomach",
        "esophagus", "bladder", "adipose", "placenta", "embryo",
    ]
    
    # Cell lines
    CELL_LINES = [
        "k562", "hela", "gm12878", "hepg2", "a549", "mcf7", "imr90",
        "h1-hesc", "h1", "h9", "jurkat", "thp1", "u2os", "hek293",
        "293t", "sh-sy5y", "nih3t3", "cho", "cos7", "raw264.7",
    ]
    
    # Diseases (focus on cancer for TCGA)
    DISEASES = {
        "glioblastoma": ["glioblastoma", "gbm", "brain cancer", "brain tumor"],
        "breast cancer": ["breast cancer", "brca", "breast tumor", "breast carcinoma"],
        "lung cancer": ["lung cancer", "lung adenocarcinoma", "luad", "lung squamous", "lusc"],
        "colon cancer": ["colon cancer", "colorectal cancer", "coad", "read"],
        "leukemia": ["leukemia", "aml", "acute myeloid", "cll", "chronic lymphocytic"],
        "lymphoma": ["lymphoma", "dlbcl", "hodgkin", "non-hodgkin"],
        "melanoma": ["melanoma", "skcm", "skin cancer"],
        "liver cancer": ["liver cancer", "hepatocellular", "lihc", "hcc"],
        "kidney cancer": ["kidney cancer", "renal cell", "kirc", "kirp"],
        "pancreatic cancer": ["pancreatic cancer", "paad", "pancreatic adenocarcinoma"],
        "prostate cancer": ["prostate cancer", "prad"],
        "ovarian cancer": ["ovarian cancer", "ov"],
    }
    
    # Assay types - expanded with informal descriptions
    ASSAY_TYPES = {
        "RNA-seq": ["rna-seq", "rnaseq", "rna seq", "mrna-seq", "transcriptome", 
                    "gene expression", "differential expression", "expression analysis",
                    "transcriptomic", "transcriptomics"],
        "ChIP-seq": ["chip-seq", "chipseq", "chip seq", "chromatin immunoprecipitation",
                     "histone", "transcription factor binding", "tf binding"],
        "ATAC-seq": ["atac-seq", "atacseq", "atac seq", "chromatin accessibility",
                     "open chromatin"],
        "scRNA-seq": ["scrna-seq", "scrnaseq", "single cell rna", "10x", "single-cell",
                      "single cell transcriptomics", "dropseq", "drop-seq"],
        "WGBS": ["wgbs", "bisulfite", "methylation", "dna methylation", "450k", "epic", 
                 "methylome", "epigenetic", "cpg methylation"],
        "Hi-C": ["hi-c", "hic", "3d genome", "chromatin conformation", 
                 "chromosome conformation", "3c", "4c", "5c"],
        "DNase-seq": ["dnase-seq", "dnaseseq", "dnase", "dnase hypersensitivity"],
        "CUT&RUN": ["cut&run", "cutnrun", "cut and run"],
        "CUT&Tag": ["cut&tag", "cutntag", "cut and tag"],
        "WGS": ["wgs", "whole genome sequencing", "whole genome", "genome sequencing",
                "dna sequencing", "genomic"],
        "WES": ["wes", "whole exome", "exome sequencing", "exome"],
        "scATAC-seq": ["scatac-seq", "scatacseq", "single cell atac"],
        "Long-read": ["long-read", "nanopore", "pacbio", "ont", "hifi", 
                      "long read sequencing", "third generation"],
        "Metagenomics": ["metagenomics", "microbiome", "16s", "shotgun metagenomics",
                         "microbial community", "metagenomic"],
        "CLIP-seq": ["clip-seq", "clipseq", "rna binding", "rna-binding", 
                     "rip-seq", "ripseq", "iclip", "eclip", "par-clip",
                     "protein-rna interaction"],
        "RRBS": ["rrbs", "reduced representation bisulfite", "targeted methylation"],
        "GRO-seq": ["gro-seq", "groseq", "nascent transcription", "run-on"],
        "PRO-seq": ["pro-seq", "proseq", "precision run-on"],
        "FAIRE-seq": ["faire-seq", "faireseq", "faire"],
        "MNase-seq": ["mnase-seq", "mnaseseq", "nucleosome positioning"],
        "Ribo-seq": ["ribo-seq", "riboseq", "ribosome profiling", "translation"],
    }
    
    # Histone marks
    HISTONE_MARKS = [
        "H3K4me1", "H3K4me2", "H3K4me3", "H3K9me3", "H3K27me3", "H3K36me3",
        "H3K27ac", "H3K9ac", "H4K20me1", "H2AZ", "H3K79me2",
    ]
    
    # Common proteins for ChIP-seq
    PROTEINS = [
        "CTCF", "p300", "Pol2", "RNAPII", "MYC", "MAX", "CEBPB", "JUND",
        "FOXA1", "GATA3", "ER", "AR", "EP300", "POLR2A", "TBP",
    ]
    
    # Job IDs
    JOB_ID_PATTERN = r'\b(?:job\s*(?:id)?\s*)?(\d{5,10})\b'


# =============================================================================
# TYPO CORRECTION
# =============================================================================

# Common terms and their variants for fuzzy matching
COMMON_TERMS = [
    # Action words
    "data", "dataset", "datasets", "download", "workflow", "search", "create",
    "analyze", "analysis", "run", "submit", "status", "help", "explain",
    "scan", "find", "check", "show", "list", "cancel", "restart",
    "resubmit", "retry", "rerun",  # Important: prevent these from being "corrected" to "submit"/"run"
    "tools", "tool", "recommend", "suggest",  # For tool recommendation queries
    # Bioinformatics terms
    "rna-seq", "rnaseq", "chip-seq", "chipseq", "atac-seq", "atacseq",
    "dna-seq", "dnaseq", "methylation", "genomics", "transcriptomics",
    "scrna-seq", "scrnaseq", "single-cell", "10x",  # Single-cell terms
    "fastq", "bam", "vcf", "bed", "reference", "genome", "annotation",
    # Organisms
    "human", "mouse", "drosophila", "zebrafish",
    # Common bioinformatics tools (prevent incorrect fuzzy matching)
    "fastqc", "samtools", "bwa", "bowtie", "bowtie2", "hisat2", "star",
    "kallisto", "salmon", "picard", "gatk", "bcftools", "bedtools",
    "deeptools", "macs2", "macs3", "htseq", "featurecounts", "multiqc",
    "trimmomatic", "cutadapt", "trim_galore", "nextflow", "snakemake",
]


# Import difflib here to avoid import inside method
from difflib import get_close_matches as _get_close_matches


def _fix_typos_impl(message: str) -> str:
    """
    Fix common typos using fuzzy matching.
    
    Uses Levenshtein-like similarity to correct typos in key terms.
    Only corrects words that are clearly typos (not in dictionary).
    
    Args:
        message: User's message with potential typos
        
    Returns:
        Message with typos corrected
    """
    words = message.split()
    corrected = []
    
    for word in words:
        word_lower = word.lower().rstrip("?.,!")
        punctuation = word[len(word_lower):] if len(word) > len(word_lower) else ""
        
        # Skip short words and words that are already valid
        if len(word_lower) < 4 or word_lower in COMMON_TERMS:
            corrected.append(word)
            continue
        
        # Try to find a close match
        matches = _get_close_matches(word_lower, COMMON_TERMS, n=1, cutoff=0.75)
        if matches:
            # Preserve original case if possible
            replacement = matches[0]
            if word[0].isupper():
                replacement = replacement.capitalize()
            corrected.append(replacement + punctuation)
            logger.debug(f"Typo corrected: '{word}' -> '{replacement}'")
        else:
            corrected.append(word)
    
    return " ".join(corrected)


# =============================================================================
# INTENT PARSER CLASS
# =============================================================================

class IntentParser:
    """
    Multi-stage intent parser with entity extraction.
    
    Stages:
    1. Pattern matching (fast, high precision)
    2. Semantic similarity (for paraphrases)
    3. LLM classification (complex cases)
    """
    
    def __init__(self, llm_client=None, use_semantic: bool = True):
        """
        Initialize the intent parser.
        
        Args:
            llm_client: Optional LLM client for complex cases
            use_semantic: Whether to use semantic similarity (requires embeddings)
        """
        self.llm_client = llm_client
        self.use_semantic = use_semantic
        self._compiled_patterns = self._compile_patterns()
        self._entity_extractor = EntityExtractor()
    
    def _fix_typos(self, message: str) -> str:
        """Instance method wrapper for typo fixing."""
        return _fix_typos_impl(message)
    
    def _compile_patterns(self) -> List[Tuple[re.Pattern, IntentType, Dict[str, int]]]:
        """Compile regex patterns for efficiency."""
        compiled = []
        for pattern, intent, slots in INTENT_PATTERNS:
            try:
                compiled.append((
                    re.compile(pattern, re.IGNORECASE),
                    intent,
                    slots
                ))
            except re.error as e:
                logger.error(f"Invalid pattern '{pattern}': {e}")
        return compiled
    
    def parse(self, message: str, context: Optional[Dict] = None) -> IntentResult:
        """
        Parse user message to determine intent.
        
        Args:
            message: User's message
            context: Optional conversation context for follow-up handling
            
        Returns:
            IntentResult with intent, confidence, and entities
        """
        message = message.strip()
        
        # Pre-stage: Fix common typos using fuzzy matching
        message = self._fix_typos(message)
        
        # Stage 0: Context-aware handling for follow-up queries
        if context and self._is_followup_query(message):
            followup_result = self._handle_followup(message, context)
            if followup_result and followup_result.confidence >= 0.6:
                return followup_result
        
        # Stage 1: Pattern matching
        pattern_result = self._match_patterns(message)
        if pattern_result and pattern_result.confidence >= 0.8:
            # Enrich with entity extraction
            pattern_result.entities = self._entity_extractor.extract(message)
            return pattern_result
        
        # Stage 2: Entity-based inference
        entities = self._entity_extractor.extract(message)
        entity_result = self._infer_from_entities(message, entities)
        
        # Combine pattern and entity results
        if pattern_result and entity_result:
            # Use pattern intent but enrich with entities
            pattern_result.entities = entities
            if entity_result.confidence > pattern_result.confidence:
                pattern_result.alternatives.append(
                    (entity_result.primary_intent, entity_result.confidence)
                )
            return pattern_result
        
        if entity_result:
            return entity_result
        
        if pattern_result:
            pattern_result.entities = entities
            return pattern_result
        
        # Stage 3: Context-based inference (for ambiguous queries)
        if context:
            context_result = self._infer_from_context(message, entities, context)
            if context_result and context_result.confidence >= 0.5:
                return context_result
        
        # Stage 4: LLM classification (if available)
        if self.llm_client:
            llm_result = self._classify_with_llm(message, context)
            if llm_result:
                llm_result.entities = entities
                return llm_result
        
        # Default: unknown intent
        return IntentResult(
            primary_intent=IntentType.META_UNKNOWN,
            confidence=0.0,
            entities=entities,
            needs_clarification=True,
            clarification_prompt="I'm not sure what you're asking. Could you rephrase or try 'help' to see what I can do?"
        )
    
    def _match_patterns(self, message: str) -> Optional[IntentResult]:
        """Match message against compiled patterns."""
        best_match = None
        best_confidence = 0.0
        alternatives = []
        
        for pattern, intent, slot_map in self._compiled_patterns:
            match = pattern.search(message)
            if match:
                # Calculate confidence based on match quality
                match_len = match.end() - match.start()
                coverage = match_len / len(message) if message else 0
                confidence = min(0.5 + coverage * 0.5, 0.95)
                
                # Extract slots from capture groups or literal values
                slots = {}
                for slot_name, group_val in slot_map.items():
                    if isinstance(group_val, int) and not isinstance(group_val, bool):
                        # It's a capture group index (but not a boolean)
                        if group_val <= len(match.groups()) and match.group(group_val):
                            slots[slot_name] = match.group(group_val).strip()
                    else:
                        # It's a literal value (including booleans)
                        slots[slot_name] = group_val
                
                if confidence > best_confidence:
                    if best_match:
                        alternatives.append((best_match.primary_intent, best_confidence))
                    best_match = IntentResult(
                        primary_intent=intent,
                        confidence=confidence,
                        entities=[],
                        slots=slots,
                        matched_pattern=pattern.pattern,
                    )
                    best_confidence = confidence
                else:
                    alternatives.append((intent, confidence))
        
        if best_match:
            best_match.alternatives = alternatives[:3]  # Top 3 alternatives
        
        return best_match
    
    def _infer_from_entities(
        self, 
        message: str, 
        entities: List[Entity]
    ) -> Optional[IntentResult]:
        """Infer intent from extracted entities."""
        entity_types = {e.type for e in entities}
        message_lower = message.lower()
        
        # If we found a dataset ID and action verb → download
        if EntityType.DATASET_ID in entity_types:
            if any(w in message_lower for w in ["download", "get", "fetch"]):
                return IntentResult(
                    primary_intent=IntentType.DATA_DOWNLOAD,
                    confidence=0.85,
                    entities=entities,
                    slots={"dataset_id": entities[0].value}
                )
        
        # If we found a path and scan-related words → scan
        if EntityType.DIRECTORY_PATH in entity_types or EntityType.FILE_PATH in entity_types:
            if any(w in message_lower for w in ["scan", "find", "look", "check", "what"]):
                return IntentResult(
                    primary_intent=IntentType.DATA_SCAN,
                    confidence=0.8,
                    entities=entities,
                )
        
        # If we found disease/tissue with data-related words → search
        if EntityType.DISEASE in entity_types or EntityType.TISSUE in entity_types:
            if any(w in message_lower for w in ["data", "dataset", "find", "search", "any"]):
                return IntentResult(
                    primary_intent=IntentType.DATA_SEARCH,
                    confidence=0.7,
                    entities=entities,
                )
        
        # If we found assay type with workflow words → create workflow
        if EntityType.ASSAY_TYPE in entity_types:
            if any(w in message_lower for w in ["workflow", "pipeline", "create", "generate", "run", "analyze"]):
                return IntentResult(
                    primary_intent=IntentType.WORKFLOW_CREATE,
                    confidence=0.75,
                    entities=entities,
                )
        
        # If we found job ID → probably status check
        if EntityType.JOB_ID in entity_types:
            return IntentResult(
                primary_intent=IntentType.JOB_STATUS,
                confidence=0.7,
                entities=entities,
            )
        
        return None
    
    def _classify_with_llm(
        self, 
        message: str, 
        context: Optional[Dict]
    ) -> Optional[IntentResult]:
        """Use LLM to classify complex intents."""
        if not self.llm_client:
            return None
        
        # Build prompt
        intent_names = [i.name for i in IntentType if not i.name.startswith("META_")]
        
        prompt = f"""Classify this bioinformatics-related user message into one intent category.

User message: "{message}"

Available intents: {', '.join(intent_names)}

Respond with JSON:
{{"intent": "INTENT_NAME", "confidence": 0.0-1.0, "reason": "brief explanation"}}
"""
        
        try:
            response = self.llm_client.chat([
                {"role": "system", "content": "You are an intent classifier for a bioinformatics assistant."},
                {"role": "user", "content": prompt}
            ])
            
            import json
            data = json.loads(response)
            intent_name = data.get("intent", "META_UNKNOWN")
            confidence = float(data.get("confidence", 0.5))
            
            try:
                intent = IntentType[intent_name]
            except KeyError:
                intent = IntentType.META_UNKNOWN
            
            return IntentResult(
                primary_intent=intent,
                confidence=confidence,
                entities=[],
            )
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return None


# =============================================================================
# ENTITY EXTRACTOR
# =============================================================================

class EntityExtractor:
    """Extract domain-specific entities from text."""
    
    def __init__(self):
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile entity extraction patterns."""
        self._dataset_patterns = [
            (re.compile(p, re.IGNORECASE), etype, source)
            for p, etype, source in EntityPatterns.DATASET_PATTERNS
        ]
        
        self._path_patterns = [
            (re.compile(p), etype)
            for p, etype in EntityPatterns.PATH_PATTERNS
        ]
        
        # Build lookup dictionaries for fast matching
        self._organism_lookup = {}
        for canonical, aliases in EntityPatterns.ORGANISMS.items():
            for alias in aliases:
                self._organism_lookup[alias.lower()] = canonical
        
        self._assay_lookup = {}
        for canonical, aliases in EntityPatterns.ASSAY_TYPES.items():
            for alias in aliases:
                self._assay_lookup[alias.lower()] = canonical
        
        self._disease_lookup = {}
        for canonical, aliases in EntityPatterns.DISEASES.items():
            for alias in aliases:
                self._disease_lookup[alias.lower()] = canonical
    
    def extract(self, text: str) -> List[Entity]:
        """Extract all entities from text."""
        entities = []
        text_lower = text.lower()
        
        # Extract dataset IDs
        for pattern, etype, source in self._dataset_patterns:
            for match in pattern.finditer(text):
                entities.append(Entity(
                    type=etype,
                    value=match.group(1),
                    canonical=match.group(1).upper(),
                    span=(match.start(1), match.end(1)),
                    metadata={"source": source}
                ))
        
        # Extract paths
        for pattern, etype in self._path_patterns:
            for match in pattern.finditer(text):
                entities.append(Entity(
                    type=etype,
                    value=match.group(1),
                    canonical=match.group(1),
                    span=(match.start(1), match.end(1)),
                ))
        
        # Extract organisms
        for alias, canonical in self._organism_lookup.items():
            if alias in text_lower:
                idx = text_lower.find(alias)
                entities.append(Entity(
                    type=EntityType.ORGANISM,
                    value=text[idx:idx+len(alias)],
                    canonical=canonical,
                    span=(idx, idx+len(alias)),
                ))
                break  # Only extract one organism
        
        # Extract tissues
        for tissue in EntityPatterns.TISSUES:
            if tissue in text_lower:
                idx = text_lower.find(tissue)
                entities.append(Entity(
                    type=EntityType.TISSUE,
                    value=tissue,
                    canonical=tissue,
                    span=(idx, idx+len(tissue)),
                ))
        
        # Extract diseases
        for alias, canonical in self._disease_lookup.items():
            if alias in text_lower:
                idx = text_lower.find(alias)
                entities.append(Entity(
                    type=EntityType.DISEASE,
                    value=text[idx:idx+len(alias)],
                    canonical=canonical,
                    span=(idx, idx+len(alias)),
                ))
                break
        
        # Extract assay types
        for alias, canonical in self._assay_lookup.items():
            if alias in text_lower:
                idx = text_lower.find(alias)
                entities.append(Entity(
                    type=EntityType.ASSAY_TYPE,
                    value=text[idx:idx+len(alias)],
                    canonical=canonical,
                    span=(idx, idx+len(alias)),
                ))
                break
        
        # Extract cell lines
        for cell_line in EntityPatterns.CELL_LINES:
            if cell_line in text_lower:
                idx = text_lower.find(cell_line)
                entities.append(Entity(
                    type=EntityType.CELL_LINE,
                    value=cell_line.upper() if len(cell_line) <= 5 else cell_line.title(),
                    canonical=cell_line.upper(),
                    span=(idx, idx+len(cell_line)),
                ))
        
        # Extract histone marks (case-sensitive)
        for mark in EntityPatterns.HISTONE_MARKS:
            if mark.lower() in text_lower:
                idx = text_lower.find(mark.lower())
                entities.append(Entity(
                    type=EntityType.HISTONE_MARK,
                    value=mark,
                    canonical=mark,
                    span=(idx, idx+len(mark)),
                ))
        
        # Extract job IDs
        job_pattern = re.compile(EntityPatterns.JOB_ID_PATTERN, re.IGNORECASE)
        for match in job_pattern.finditer(text):
            entities.append(Entity(
                type=EntityType.JOB_ID,
                value=match.group(1),
                canonical=match.group(1),
                span=(match.start(1), match.end(1)),
            ))
        
        return entities


# =============================================================================
# CONTEXT-AWARE PARSING HELPERS (added to IntentParser class)
# =============================================================================

def _add_context_methods():
    """Add context-aware methods to IntentParser."""
    
    def _is_followup_query(self, message: str) -> bool:
        """Check if this is a follow-up query (short, uses pronouns, etc.)."""
        message_lower = message.lower().strip()
        
        # Short queries are often follow-ups
        if len(message.split()) <= 6:
            # Check for pronouns and references
            followup_indicators = [
                "it", "that", "this", "those", "these",
                "what about", "how about", "and", "also",
                "compare", "versus", "vs", "or",
                "more", "other", "another", "same",
                "back to", "actually", "no,", "not"
            ]
            return any(ind in message_lower for ind in followup_indicators)
        return False
    
    def _handle_followup(self, message: str, context: Dict) -> Optional[IntentResult]:
        """Handle follow-up queries using context."""
        message_lower = message.lower().strip()
        entities = self._entity_extractor.extract(message)
        
        last_intent = context.get("last_intent")
        last_topic = context.get("last_topic")
        last_tool = context.get("last_tool")
        
        # Handle comparison queries: "How does it compare to X?"
        if any(kw in message_lower for kw in ["compare", "versus", "vs", "difference", "better"]):
            # This is an educational comparison query
            # Extract what we're comparing to
            compare_patterns = [
                r"compare.*?to\s+(.+?)[\?\.!]?$",
                r"versus\s+(.+?)[\?\.!]?$",
                r"vs\.?\s+(.+?)[\?\.!]?$",
                r"difference.*?(?:between|with)\s+(.+?)[\?\.!]?$",
                r"better.*?(?:than|for)\s+(.+?)[\?\.!]?$",
            ]
            for pattern in compare_patterns:
                match = re.search(pattern, message_lower)
                if match:
                    concept2 = match.group(1).strip()
                    concept1 = last_topic or "it"
                    return IntentResult(
                        primary_intent=IntentType.EDUCATION_EXPLAIN,
                        confidence=0.75,
                        entities=entities,
                        slots={"concept": f"comparison of {concept1} and {concept2}"},
                    )
        
        # Handle "what about X" - continues previous context
        if message_lower.startswith("what about") or message_lower.startswith("how about"):
            if last_intent == "EDUCATION_EXPLAIN":
                new_topic = message_lower.replace("what about", "").replace("how about", "").strip().rstrip("?")
                return IntentResult(
                    primary_intent=IntentType.EDUCATION_EXPLAIN,
                    confidence=0.70,
                    entities=entities,
                    slots={"concept": new_topic},
                )
            elif last_intent == "DATA_SEARCH":
                new_query = message_lower.replace("what about", "").replace("how about", "").strip().rstrip("?")
                return IntentResult(
                    primary_intent=IntentType.DATA_SEARCH,
                    confidence=0.70,
                    entities=entities,
                    slots={"query": new_query},
                )
        
        # Handle corrections: "No, I meant X" or "actually, X"
        if message_lower.startswith("no,") or message_lower.startswith("actually") or message_lower.startswith("not"):
            # Try to extract the intended meaning and re-classify
            cleaned = re.sub(r"^(no,|actually,?|not)\s*", "", message_lower).strip()
            if cleaned:
                # Re-run pattern matching on the cleaned message
                return self._match_patterns(cleaned) or IntentResult(
                    primary_intent=IntentType.META_CORRECT,
                    confidence=0.60,
                    entities=entities,
                    needs_clarification=True,
                    clarification_prompt=f"I understand you want to correct something. Could you tell me more specifically what you meant by '{cleaned}'?",
                )
        
        return None
    
    def _infer_from_context(self, message: str, entities: List[Entity], context: Dict) -> Optional[IntentResult]:
        """Infer intent from context when pattern matching fails."""
        last_intent = context.get("last_intent")
        last_topic = context.get("last_topic")
        turn_count = context.get("turn_count", 0)
        
        message_lower = message.lower().strip()
        
        # If we're in a multi-turn conversation and the message looks like a continuation
        if turn_count > 0 and len(message.split()) <= 10:
            # Look for bioinformatics entities that might suggest intent
            bio_entities = [e for e in entities if hasattr(e, 'type') and e.type in (
                EntityType.ASSAY_TYPE, EntityType.ORGANISM, EntityType.TISSUE,
                EntityType.TOOL, EntityType.FILE_FORMAT
            )]
            
            if bio_entities:
                # If entities match data types, might be search
                if any(e.type == EntityType.ASSAY_TYPE for e in bio_entities):
                    if "search" in message_lower or "find" in message_lower or "data" in message_lower:
                        return IntentResult(
                            primary_intent=IntentType.DATA_SEARCH,
                            confidence=0.55,
                            entities=entities,
                            slots={"query": " ".join(e.value for e in bio_entities)},
                        )
                
                # If continuing from educational context, likely asking about entity
                if last_intent == "EDUCATION_EXPLAIN":
                    topic = " ".join(e.value for e in bio_entities[:2])
                    return IntentResult(
                        primary_intent=IntentType.EDUCATION_EXPLAIN,
                        confidence=0.55,
                        entities=entities,
                        slots={"concept": topic},
                    )
        
        return None
    
    # Attach methods to the class
    IntentParser._is_followup_query = _is_followup_query
    IntentParser._handle_followup = _handle_followup
    IntentParser._infer_from_context = _infer_from_context

# Initialize context methods
_add_context_methods()
