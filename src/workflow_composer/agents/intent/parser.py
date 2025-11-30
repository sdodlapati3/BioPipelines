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
    
    # Analysis & Results
    ANALYSIS_RUN = auto()        # Run analysis on results
    ANALYSIS_INTERPRET = auto()  # Interpret QC/results
    ANALYSIS_COMPARE = auto()    # Compare samples/conditions
    ANALYSIS_VISUALIZE = auto()  # Generate plots
    
    # Diagnostics
    DIAGNOSE_ERROR = auto()      # Diagnose failures
    DIAGNOSE_RECOVER = auto()    # Attempt recovery
    
    # System Operations
    SYSTEM_STATUS = auto()       # Check system health
    SYSTEM_RESTART = auto()      # Restart services
    
    # Education
    EDUCATION_EXPLAIN = auto()   # Explain concepts
    EDUCATION_HELP = auto()      # Show help
    EDUCATION_TUTORIAL = auto()  # Step-by-step guide
    
    # Meta/Conversational
    META_CONFIRM = auto()        # User confirmation (yes/no)
    META_CANCEL = auto()         # Cancel current operation
    META_CLARIFY = auto()        # User clarifying previous statement
    META_UNDO = auto()           # Undo last action
    META_GREETING = auto()       # Hello/greeting
    META_THANKS = auto()         # Thank you
    META_UNKNOWN = auto()        # Cannot determine intent
    
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
    
    # Generate then run
    (r"(?:create|generate|make)\s+(?:a\s+)?(.+?)\s+(?:workflow|pipeline)\s+(?:and\s+)?(?:then\s+)?(?:run|execute|submit)",
     IntentType.COMPOSITE_GENERATE_THEN_RUN, {"workflow_type": 1}),
    
    # =========================================================================
    # EDUCATION - must come before workflow creation to catch "how does X work"
    # =========================================================================
    (r"how\s+does\s+(.+?)\s+work(?:\??|$)",
     IntentType.EDUCATION_EXPLAIN, {"concept": 1}),
    (r"how\s+do(?:es)?\s+(.+?)\s+(?:algorithms?|methods?)\s+work",
     IntentType.EDUCATION_EXPLAIN, {"concept": 1}),
    (r"(?:i\s+want\s+to|help\s+me)\s+(?:learn|understand)\s+(?:about\s+)?(.+)",
     IntentType.EDUCATION_EXPLAIN, {"concept": 1}),
    
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
    (r"what\s+(?:data|files?|samples?)\s+(?:do\s+)?(?:i\s+have|we\s+have|is\s+available|exist)(?:\s+(?:in|at)\s+(.+))?",
     IntentType.DATA_SCAN, {"path": 1}),
    # This pattern must NOT match if "if not search" follows
    # Using negative lookahead to exclude composite patterns
    (r"check\s+if\s+(?:we\s+have|there\s+is|there\s+are)\s+(?:any\s+)?(.+?)(?:\s+data|\s+locally)?$(?<!\bsearch\b)(?<!\bonline\b)",
     IntentType.DATA_SCAN, {"query": 1}),
    
    # =========================================================================
    # DATABASE SEARCH
    # =========================================================================
    # Generic "Search for X" - should match before more specific patterns
    (r"(?:search|find|look)\s+for\s+(.+?)\s+data",
     IntentType.DATA_SEARCH, {"query": 1}),
    (r"(?:search|find|look)\s+for\s+(.+?)$",
     IntentType.DATA_SEARCH, {"query": 1}),
    # "Find X ChIP-seq data" style
    (r"(?:find|search\s+for|look\s+for)\s+(.+?)\s+(?:ChIP-seq|RNA-seq|ATAC-seq|Hi-C|methylation|metagenomics)(?:\s+data)?",
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
    
    # TCGA-specific search
    (r"(?:search|find|look\s+for)\s+(?:tcga|cancer)\s+(.+)",
     IntentType.DATA_SEARCH, {"query": 1, "source": "tcga"}),
    (r"(.+?)\s+(?:cancer|tumor)\s+(?:data|methylation|expression)",
     IntentType.DATA_SEARCH, {"cancer_type": 1}),
    
    # Download with exclusions
    (r"(?:download|get|fetch)\s+(?:all\s+)?(?:samples?|data|datasets?)\s+(?:except|without|excluding|but\s+not)\s+(.+)",
     IntentType.DATA_DOWNLOAD, {"excluded": 1}),
    
    # Download
    (r"(?:download|get|fetch)\s+(?:dataset\s+)?(GSE\d+|ENCSR[A-Z0-9]+|TCGA-[A-Z]+)",
     IntentType.DATA_DOWNLOAD, {"dataset_id": 1}),
    (r"(?:download|get|fetch)\s+(?:this|that|the)\s+(?:dataset|data)",
     IntentType.DATA_DOWNLOAD, {}),
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
    
    # Workflow creation
    (r"(?:create|generate|make|build|set\s+up)\s+(?:a\s+)?(?:new\s+)?(.+?)\s+(?:workflow|pipeline|analysis)",
     IntentType.WORKFLOW_CREATE, {"workflow_type": 1}),
    (r"(?:i\s+want\s+to|i\s+need\s+to|let's|can\s+you)\s+(?:run|do|perform)\s+(?:a\s+)?(.+?)\s+(?:analysis|workflow|pipeline)",
     IntentType.WORKFLOW_CREATE, {"workflow_type": 1}),
    
    # Tool preference (implies workflow context)
    (r"(?:use|prefer|choose)\s+(\w+(?:[_-]\w+)?)\s*(?:\w+)?\s*(?:instead\s+of|over|rather\s+than)\s+(\w+(?:[_-]\w+)?)",
     IntentType.WORKFLOW_CREATE, {"preferred_tool": 1, "avoided_tool": 2}),
    (r"i\s+prefer\s+(\w+(?:[_-]\w+)?)\s*(?:\w+)?\s*(?:over|to)\s+(\w+(?:[_-]\w+)?)",
     IntentType.WORKFLOW_CREATE, {"preferred_tool": 1, "avoided_tool": 2}),
    (r"(?:don't|do\s+not)\s+use\s+(\w+(?:[_-]\w+)?)\s*,?\s*(?:use|prefer)\s+(\w+(?:[_-]\w+)?)",
     IntentType.WORKFLOW_CREATE, {"avoided_tool": 1, "preferred_tool": 2}),
    
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
    
    # Job submission - "submit my X workflow" means run existing workflow
    (r"(?:please\s+)?submit\s+(?:my\s+)?(?:the\s+)?(\w+(?:\s+\w+)?)\s+(?:workflow|pipeline)",
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
    
    # Job status
    (r"(?:what\s+is|check|show|get)\s+(?:the\s+)?(?:job\s+)?status(?:\s+(?:of|for)\s+(?:job\s+)?(\d+))?",
     IntentType.JOB_STATUS, {"job_id": 1}),
    (r"(?:is\s+(?:the\s+)?job|are\s+(?:the\s+)?jobs?)\s+(?:still\s+)?(?:running|done|finished|complete)",
     IntentType.JOB_STATUS, {}),
    (r"(?:how\s+is|what's\s+happening\s+with)\s+(?:the\s+)?(?:job|analysis|run)",
     IntentType.JOB_STATUS, {}),
    # "what's the status of job X" must come before education patterns
    (r"what(?:'s|\s+is)\s+(?:the\s+)?status\s+(?:of|for)\s+(?:job\s+)?(\d+)",
     IntentType.JOB_STATUS, {"job_id": 1}),
    (r"(?:check|show|get)\s+(?:on\s+)?(?:job\s+)?(\d+)",
     IntentType.JOB_STATUS, {"job_id": 1}),
    (r"(?:check|see|look)\s+(?:on\s+)?(?:it|the\s+job)",
     IntentType.JOB_STATUS, {}),
    
    # Logs
    (r"(?:show|get|view|display)\s+(?:the\s+)?(?:job\s+)?logs?(?:\s+(?:for|of)\s+(?:job\s+)?(\d+))?",
     IntentType.JOB_LOGS, {"job_id": 1}),
    (r"(?:what\s+happened|what\s+went\s+wrong|show\s+me\s+(?:the\s+)?(?:error|output))",
     IntentType.JOB_LOGS, {}),
    (r"what(?:'s|\s+is)\s+(?:the\s+)?output\s+(?:of|for)\s+(?:job\s+)?(\d+)",
     IntentType.JOB_LOGS, {"job_id": 1}),
    
    # Job List (must come before more generic patterns)
    (r"list\s+(?:all\s+)?(?:my\s+)?(?:running\s+)?jobs",
     IntentType.JOB_LIST, {}),
    (r"(?:show|display|get)\s+(?:all\s+)?(?:my\s+)?(?:running\s+)?jobs",
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
    
    # Diagnostics
    (r"(?:diagnose|debug|troubleshoot|analyze)\s+(?:this\s+)?(?:error|failure|problem)",
     IntentType.DIAGNOSE_ERROR, {}),
    (r"(?:what\s+went\s+wrong|why\s+did\s+it\s+fail|fix\s+this|help\s+me\s+fix)",
     IntentType.DIAGNOSE_ERROR, {}),
    (r"(?:try\s+to\s+)?(?:fix|recover|resolve)\s+(?:this|the)\s+(?:error|issue|problem)",
     IntentType.DIAGNOSE_RECOVER, {}),
    
    # Analysis/Results
    (r"(?:analyze|interpret|explain)\s+(?:the\s+)?(?:results?|output|qc)",
     IntentType.ANALYSIS_INTERPRET, {}),
    (r"(?:what\s+do\s+(?:the\s+)?results?\s+(?:mean|show)|summarize\s+(?:the\s+)?results?)",
     IntentType.ANALYSIS_INTERPRET, {}),
    
    # Education
    (r"(?:what\s+is|what's|explain|describe|tell\s+me\s+about)\s+(.+?)(?:\?|$)",
     IntentType.EDUCATION_EXPLAIN, {"concept": 1}),
    (r"(?:how\s+do\s+i|how\s+to|how\s+can\s+i)\s+(.+)",
     IntentType.EDUCATION_TUTORIAL, {"topic": 1}),
    (r"^(?:help|commands?|what\s+can\s+you\s+do|\?+|show\s+help|list\s+commands?)$",
     IntentType.EDUCATION_HELP, {}),
    (r"what\s+(?:can\s+(?:you|i)\s+do|are\s+(?:your|my)\s+(?:options|capabilities))",
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
        (r'(?:in|at|from|to)\s+["\']?(/[^\s"\']+|~/[^\s"\']+)["\']?', EntityType.DIRECTORY_PATH),
        (r'(?:file\s+)?["\']?(/[^\s"\']+\.(?:fastq|fq|bam|bed|vcf|txt|csv|tsv)(?:\.gz)?)["\']?', EntityType.FILE_PATH),
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
            context: Optional conversation context
            
        Returns:
            IntentResult with intent, confidence, and entities
        """
        message = message.strip()
        
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
        
        # Stage 3: LLM classification (if available)
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
