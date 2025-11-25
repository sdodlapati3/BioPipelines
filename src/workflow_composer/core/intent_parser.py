"""
Intent Parser
=============

Extracts structured analysis intent from natural language descriptions.

Uses LLM to understand:
- Analysis type (RNA-seq, ChIP-seq, variant calling, etc.)
- Data format and characteristics
- Organism and reference genome
- Specific parameters and requirements
- Desired outputs

Example:
    parser = IntentParser(llm)
    intent = parser.parse(
        "I have paired-end RNA-seq data from mouse liver. "
        "I want to find differentially expressed genes."
    )
    print(intent.analysis_type)  # "rna_seq_differential_expression"
    print(intent.organism)       # "mouse"
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum

from ..llm.base import LLMAdapter, Message

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Supported analysis types."""
    # RNA-seq
    RNA_SEQ_BASIC = "rna_seq_basic"
    RNA_SEQ_DE = "rna_seq_differential_expression"
    RNA_SEQ_DENOVO = "rna_seq_de_novo_assembly"
    RNA_SEQ_SPLICING = "rna_seq_alternative_splicing"
    SMALL_RNA_SEQ = "small_rna_seq"
    
    # DNA-seq
    WGS_VARIANT_CALLING = "wgs_variant_calling"
    WES_VARIANT_CALLING = "wes_variant_calling"
    SOMATIC_VARIANT_CALLING = "somatic_variant_calling"
    STRUCTURAL_VARIANT = "structural_variant_detection"
    
    # Epigenomics
    CHIP_SEQ = "chip_seq_peak_calling"
    ATAC_SEQ = "atac_seq"
    BISULFITE_SEQ = "bisulfite_seq_methylation"
    MEDIP_SEQ = "medip_seq"
    HIC = "hic_chromatin_interaction"
    
    # Single-cell
    SCRNA_SEQ = "single_cell_rna_seq"
    SCRNA_SEQ_INTEGRATION = "scrna_seq_integration"
    
    # Spatial transcriptomics
    SPATIAL_TRANSCRIPTOMICS = "spatial_transcriptomics"
    SPATIAL_VISIUM = "spatial_visium"
    SPATIAL_SLIDE_SEQ = "spatial_slide_seq"
    SPATIAL_XENIUM = "spatial_xenium"
    
    # Metagenomics
    METAGENOMICS_PROFILING = "metagenomics_profiling"
    METAGENOMICS_ASSEMBLY = "metagenomics_assembly"
    AMPLICON_16S = "amplicon_16s"
    
    # Assembly
    GENOME_ASSEMBLY = "genome_assembly"
    GENOME_ANNOTATION = "genome_annotation"
    
    # Long-read
    LONG_READ_ASSEMBLY = "long_read_assembly"
    LONG_READ_VARIANT = "long_read_variant_calling"
    LONG_READ_RNA_SEQ = "long_read_rna_seq"
    LONG_READ_ISOSEQ = "long_read_isoseq"
    LONG_READ_DIRECT_RNA = "long_read_direct_rna"
    
    # Multi-omics integration
    MULTI_OMICS = "multi_omics_integration"
    RNA_ATAC_INTEGRATION = "rna_atac_integration"
    PROTEOGENOMICS = "proteogenomics"
    MULTI_MODAL_SCRNA = "multi_modal_scrna"
    
    # Other
    CUSTOM = "custom"
    UNKNOWN = "unknown"


@dataclass
class ParsedIntent:
    """Structured representation of user's analysis intent."""
    
    # Core analysis info
    analysis_type: AnalysisType
    analysis_type_raw: str  # Original string from LLM
    confidence: float = 0.0
    
    # Data characteristics
    data_type: str = "fastq"  # fastq, bam, vcf, etc.
    paired_end: bool = True
    stranded: bool = False
    read_length: Optional[int] = None
    
    # Organism/Reference
    organism: str = ""
    genome_build: str = ""
    
    # Comparison/Design
    has_comparison: bool = False
    conditions: List[str] = field(default_factory=list)
    replicates: int = 0
    
    # Specific requirements
    tools_requested: List[str] = field(default_factory=list)
    outputs_requested: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Original query
    original_query: str = ""
    
    # LLM reasoning
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "analysis_type": self.analysis_type.value,
            "analysis_type_raw": self.analysis_type_raw,
            "confidence": self.confidence,
            "data_type": self.data_type,
            "paired_end": self.paired_end,
            "stranded": self.stranded,
            "read_length": self.read_length,
            "organism": self.organism,
            "genome_build": self.genome_build,
            "has_comparison": self.has_comparison,
            "conditions": self.conditions,
            "replicates": self.replicates,
            "tools_requested": self.tools_requested,
            "outputs_requested": self.outputs_requested,
            "parameters": self.parameters,
            "original_query": self.original_query,
            "reasoning": self.reasoning
        }


# Analysis type keywords for rule-based fallback
ANALYSIS_KEYWORDS = {
    AnalysisType.RNA_SEQ_DE: [
        "differential expression", "de analysis", "deseq2", "edger",
        "differentially expressed", "rna-seq de", "rnaseq de"
    ],
    AnalysisType.RNA_SEQ_BASIC: [
        "rna-seq", "rnaseq", "transcriptome", "gene expression",
        "quantification", "salmon", "star align"
    ],
    AnalysisType.RNA_SEQ_DENOVO: [
        "de novo assembly", "trinity", "transcriptome assembly",
        "no reference", "without reference"
    ],
    AnalysisType.CHIP_SEQ: [
        "chip-seq", "chipseq", "histone", "transcription factor",
        "peak calling", "macs2", "homer peaks", "h3k4me3", "h3k27ac"
    ],
    AnalysisType.ATAC_SEQ: [
        "atac-seq", "atacseq", "open chromatin", "chromatin accessibility"
    ],
    AnalysisType.WGS_VARIANT_CALLING: [
        "variant calling", "wgs", "whole genome", "snp", "indel",
        "gatk", "freebayes", "germline variant"
    ],
    AnalysisType.SOMATIC_VARIANT_CALLING: [
        "somatic", "tumor", "cancer", "mutation", "mutect",
        "tumor-normal", "cancer variant"
    ],
    AnalysisType.STRUCTURAL_VARIANT: [
        "structural variant", "sv", "cnv", "manta", "delly",
        "copy number", "translocation", "fusion"
    ],
    AnalysisType.SCRNA_SEQ: [
        "single cell", "single-cell", "scrna", "10x genomics",
        "cellranger", "seurat", "scanpy", "scrnaseq"
    ],
    AnalysisType.METAGENOMICS_PROFILING: [
        "metagenomics", "microbiome", "taxonomic", "kraken",
        "metaphlan", "16s", "species composition"
    ],
    AnalysisType.METAGENOMICS_ASSEMBLY: [
        "metagenome assembly", "metagenomic assembly", "megahit",
        "metaspades", "contig", "mag"
    ],
    AnalysisType.BISULFITE_SEQ: [
        "bisulfite", "methylation", "dna methylation", "bsseq",
        "bismark", "cpg", "wgbs", "rrbs"
    ],
    AnalysisType.HIC: [
        "hi-c", "hic", "chromatin interaction", "3d genome",
        "tad", "loop", "juicer"
    ],
    AnalysisType.LONG_READ_ASSEMBLY: [
        "long read", "nanopore", "pacbio", "ont", "flye",
        "canu", "assembly", "polish"
    ],
    AnalysisType.GENOME_ANNOTATION: [
        "annotation", "prokka", "augustus", "gene prediction",
        "annotate genome"
    ],
    # Spatial transcriptomics
    AnalysisType.SPATIAL_TRANSCRIPTOMICS: [
        "spatial", "spatial transcriptomics", "spatial gene expression",
        "tissue section", "spot", "spatial clustering"
    ],
    AnalysisType.SPATIAL_VISIUM: [
        "visium", "10x visium", "visium hd", "space ranger",
        "spaceranger", "visium spatial"
    ],
    AnalysisType.SPATIAL_SLIDE_SEQ: [
        "slide-seq", "slideseq", "slide seq", "puck",
        "bead array"
    ],
    AnalysisType.SPATIAL_XENIUM: [
        "xenium", "in situ", "xenium ranger", "10x xenium"
    ],
    # Long-read RNA-seq
    AnalysisType.LONG_READ_RNA_SEQ: [
        "long read rna", "full length transcript", "isoform",
        "nanopore rna", "direct rna", "isoseq", "iso-seq"
    ],
    AnalysisType.LONG_READ_ISOSEQ: [
        "isoseq", "iso-seq", "pacbio rna", "isoform sequencing",
        "fl transcript", "full-length cdna"
    ],
    AnalysisType.LONG_READ_DIRECT_RNA: [
        "direct rna", "drna", "nanopore rna", "native rna",
        "rna modification", "m6a"
    ],
    # Multi-omics integration
    AnalysisType.MULTI_OMICS: [
        "multi-omics", "multiomics", "integration", "combine",
        "integrated analysis", "multi-modal"
    ],
    AnalysisType.RNA_ATAC_INTEGRATION: [
        "rna atac", "atac rna", "joint embedding",
        "rna-seq atac-seq", "gene regulatory", "multiome"
    ],
    AnalysisType.PROTEOGENOMICS: [
        "proteogenomics", "protein genomics", "mass spec rna",
        "peptide variant", "neoantigen"
    ],
    AnalysisType.MULTI_MODAL_SCRNA: [
        "cite-seq", "citeseq", "multiome", "10x multiome",
        "scrna atacseq", "single cell multiome", "asap-seq"
    ],
}

# Organism mapping
ORGANISM_MAP = {
    "human": ("human", "hg38"),
    "homo sapiens": ("human", "hg38"),
    "mouse": ("mouse", "mm10"),
    "mus musculus": ("mouse", "mm10"),
    "rat": ("rat", "rn6"),
    "rattus norvegicus": ("rat", "rn6"),
    "zebrafish": ("zebrafish", "danRer11"),
    "danio rerio": ("zebrafish", "danRer11"),
    "fly": ("drosophila", "dm6"),
    "drosophila": ("drosophila", "dm6"),
    "worm": ("c_elegans", "ce11"),
    "c. elegans": ("c_elegans", "ce11"),
    "yeast": ("yeast", "sacCer3"),
    "arabidopsis": ("arabidopsis", "TAIR10"),
    "e. coli": ("e_coli", "K12"),
    "bacteria": ("bacteria", ""),
}


class IntentParser:
    """
    Parses natural language to extract analysis intent.
    
    Uses a hybrid approach:
    1. LLM for complex understanding and reasoning
    2. Rule-based fallback for reliability
    3. Validation against known analysis types
    """
    
    SYSTEM_PROMPT = """You are a bioinformatics expert assistant. Your task is to analyze user requests for genomics/bioinformatics analysis and extract structured information.

Given a user's description of their analysis needs, extract:
1. analysis_type: The main type of analysis (e.g., "rna_seq_differential_expression", "chip_seq_peak_calling", "wgs_variant_calling")
2. data_type: Input data format (fastq, bam, vcf, etc.)
3. paired_end: Whether data is paired-end (true/false)
4. organism: Species name
5. genome_build: Reference genome version (hg38, mm10, etc.)
6. has_comparison: Whether there's a comparison between conditions
7. conditions: List of conditions/groups being compared
8. tools_requested: Any specific tools mentioned
9. outputs_requested: What outputs the user wants
10. parameters: Any specific parameters mentioned

Respond ONLY with a valid JSON object. No explanations."""

    USER_PROMPT_TEMPLATE = """Analyze this bioinformatics request and extract structured information:

"{query}"

Respond with a JSON object containing:
{{
    "analysis_type": "string - one of: rna_seq_basic, rna_seq_differential_expression, rna_seq_de_novo_assembly, rna_seq_alternative_splicing, small_rna_seq, chip_seq_peak_calling, atac_seq, wgs_variant_calling, wes_variant_calling, somatic_variant_calling, structural_variant_detection, single_cell_rna_seq, scrna_seq_integration, spatial_transcriptomics, spatial_visium, spatial_slide_seq, spatial_xenium, metagenomics_profiling, metagenomics_assembly, amplicon_16s, bisulfite_seq_methylation, medip_seq, hic_chromatin_interaction, long_read_assembly, long_read_variant_calling, long_read_rna_seq, long_read_isoseq, long_read_direct_rna, genome_assembly, genome_annotation, multi_omics_integration, rna_atac_integration, proteogenomics, multi_modal_scrna, custom",
    "confidence": "float 0-1 indicating confidence",
    "data_type": "string - fastq, bam, vcf, h5ad, etc.",
    "paired_end": "boolean",
    "stranded": "boolean",
    "organism": "string - species name",
    "genome_build": "string - reference genome",
    "has_comparison": "boolean",
    "conditions": ["list", "of", "conditions"],
    "tools_requested": ["list", "of", "tools"],
    "outputs_requested": ["list", "of", "outputs"],
    "parameters": {{}},
    "reasoning": "brief explanation of your interpretation"
}}"""

    def __init__(self, llm: LLMAdapter):
        """
        Initialize intent parser.
        
        Args:
            llm: LLM adapter for natural language understanding
        """
        self.llm = llm
    
    def parse(self, query: str) -> ParsedIntent:
        """
        Parse a natural language query into structured intent.
        
        Args:
            query: User's natural language description
            
        Returns:
            ParsedIntent with extracted information
        """
        logger.info(f"Parsing intent from: {query[:100]}...")
        
        # Try LLM parsing first
        try:
            intent = self._parse_with_llm(query)
            logger.info(f"LLM parsed analysis type: {intent.analysis_type}")
            return intent
        except Exception as e:
            logger.warning(f"LLM parsing failed: {e}, falling back to rules")
        
        # Fall back to rule-based parsing
        return self._parse_with_rules(query)
    
    def _parse_with_llm(self, query: str) -> ParsedIntent:
        """Parse using LLM."""
        messages = [
            Message.system(self.SYSTEM_PROMPT),
            Message.user(self.USER_PROMPT_TEMPLATE.format(query=query))
        ]
        
        response = self.llm.chat(messages, temperature=0.1)
        
        # Parse JSON response
        content = response.content.strip()
        
        # Handle markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {content[:200]}")
            raise ValueError(f"Invalid JSON from LLM: {e}")
        
        # Map to AnalysisType enum
        analysis_type_str = data.get("analysis_type", "unknown")
        try:
            analysis_type = AnalysisType(analysis_type_str)
        except ValueError:
            analysis_type = AnalysisType.CUSTOM
        
        return ParsedIntent(
            analysis_type=analysis_type,
            analysis_type_raw=analysis_type_str,
            confidence=float(data.get("confidence", 0.8)),
            data_type=data.get("data_type", "fastq"),
            paired_end=data.get("paired_end", True),
            stranded=data.get("stranded", False),
            read_length=data.get("read_length"),
            organism=data.get("organism", ""),
            genome_build=data.get("genome_build", ""),
            has_comparison=data.get("has_comparison", False),
            conditions=data.get("conditions", []),
            replicates=data.get("replicates", 0),
            tools_requested=data.get("tools_requested", []),
            outputs_requested=data.get("outputs_requested", []),
            parameters=data.get("parameters", {}),
            original_query=query,
            reasoning=data.get("reasoning", "")
        )
    
    def _parse_with_rules(self, query: str) -> ParsedIntent:
        """Parse using rule-based approach."""
        query_lower = query.lower()
        
        # Detect analysis type
        analysis_type = AnalysisType.UNKNOWN
        best_match_count = 0
        
        for atype, keywords in ANALYSIS_KEYWORDS.items():
            match_count = sum(1 for kw in keywords if kw in query_lower)
            if match_count > best_match_count:
                best_match_count = match_count
                analysis_type = atype
        
        # Detect organism
        organism = ""
        genome_build = ""
        for org_key, (org_name, genome) in ORGANISM_MAP.items():
            if org_key in query_lower:
                organism = org_name
                genome_build = genome
                break
        
        # Detect paired-end
        paired_end = "paired" in query_lower or "pe" in query_lower
        if "single-end" in query_lower or "single end" in query_lower:
            paired_end = False
        
        # Detect comparison
        has_comparison = any(kw in query_lower for kw in [
            "differential", "compare", "versus", "vs", "between",
            "treatment", "control", "condition"
        ])
        
        # Extract conditions (simple heuristic)
        conditions = []
        if "treatment" in query_lower and "control" in query_lower:
            conditions = ["treatment", "control"]
        
        return ParsedIntent(
            analysis_type=analysis_type,
            analysis_type_raw=analysis_type.value,
            confidence=0.6 if analysis_type != AnalysisType.UNKNOWN else 0.3,
            data_type="fastq",
            paired_end=paired_end,
            organism=organism,
            genome_build=genome_build,
            has_comparison=has_comparison,
            conditions=conditions,
            original_query=query,
            reasoning="Parsed using rule-based approach"
        )
    
    def clarify(self, intent: ParsedIntent, question: str) -> str:
        """
        Ask LLM for clarification about the intent.
        
        Args:
            intent: Current parsed intent
            question: Specific question to clarify
            
        Returns:
            Clarification response
        """
        messages = [
            Message.system("You are a helpful bioinformatics assistant."),
            Message.user(f"""Based on this analysis request:
"{intent.original_query}"

Currently interpreted as: {intent.analysis_type.value}

Question: {question}

Provide a brief, helpful response.""")
        ]
        
        response = self.llm.chat(messages)
        return response.content
