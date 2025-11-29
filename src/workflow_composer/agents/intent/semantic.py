"""
Semantic Query Understanding
=============================

Hybrid approach combining:
1. FAISS-based semantic similarity for intent classification
2. Sentence Transformers for embedding generation
3. Domain-specific NER for bioinformatics entities
4. Pattern matching for high-confidence cases
5. LLM fallback for complex disambiguation

This module addresses the gap between raw text parsing and understanding
the semantic meaning of user queries in a bioinformatics context.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from pathlib import Path
import json
import hashlib

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Semantic search will use fallback cosine similarity.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Will use LLM embeddings as fallback.")


# =============================================================================
# INTENT EXAMPLES (Training Data for Semantic Matching)
# =============================================================================

# These examples define the semantic space for each intent
# The more diverse and representative, the better the matching
INTENT_EXAMPLES = {
    # Data Discovery
    "DATA_SCAN": [
        "scan local data",
        "what files do I have",
        "check my data folder",
        "list available samples",
        "show me the fastq files",
        "what data is available locally",
        "look for data in my directory",
        "find samples on disk",
        "discover local datasets",
        "inventory my data",
    ],
    "DATA_SEARCH": [
        "search for RNA-seq data",
        "find human brain samples in GEO",
        "look for ChIP-seq datasets",
        "search ENCODE for liver data",
        "find methylation data from TCGA",
        "are there any mouse kidney samples available",
        "search public databases for cancer data",
        "find datasets matching my criteria",
        "look up experiments in SRA",
        "query GEO for stem cell data",
        "human brain RNA-seq",  # Entity-only query
        "mouse liver ChIP-seq",  # Entity-only query
        "cancer methylation data",  # Entity-only query
        "I need transcriptome data",  # Informal
        "looking for sequencing data",  # Informal
    ],
    "DATA_DOWNLOAD": [
        "download GSE12345",
        "get the ENCODE dataset",
        "fetch this data",
        "download the selected samples",
        "retrieve the fastq files",
        "pull data from GEO",
        "get TCGA methylation data",
        "download ENCSR000AAA",
        "grab the reference genome",
        "fetch RNA-seq reads",
        "please grab that dataset for me",  # Informal
        "get it for me",  # Informal with coreference
        "download that one",  # Coreference
        "can you download this",  # Question form
        "I want to download the data",  # First person
        "download all",  # Batch download
        "download all of them",  # Batch download
        "download everything",  # Batch download
        "get all the datasets",  # Batch download
        "download all results",  # Batch download
    ],
    
    # Workflow Operations
    "WORKFLOW_CREATE": [
        "create an RNA-seq workflow",
        "generate a ChIP-seq pipeline",
        "build an analysis workflow",
        "set up a methylation analysis",
        "make a variant calling pipeline",
        "create workflow for my data",
        "generate pipeline for differential expression",
        "set up ATAC-seq analysis",
        "build a metagenomics workflow",
        "create Hi-C processing pipeline",
    ],
    "WORKFLOW_VISUALIZE": [
        "show the workflow diagram",
        "visualize the pipeline",
        "display the DAG",
        "show me the workflow steps",
        "draw the pipeline graph",
        "what does the workflow look like",
        "show workflow structure",
        "visualize analysis steps",
    ],
    
    # Job Management
    "JOB_SUBMIT": [
        "run the workflow",
        "submit to SLURM",
        "start the analysis",
        "execute the pipeline",
        "run the job",
        "submit to cluster",
        "start processing",
        "launch the workflow",
        "begin the analysis",
        "kick off the pipeline",
    ],
    "JOB_STATUS": [
        "check job status",
        "how is the job doing",
        "is it still running",
        "what's the progress",
        "show job status",
        "are my jobs complete",
        "check if analysis finished",
        "status of my runs",
        "how long until it's done",
        "monitor job progress",
    ],
    "JOB_LOGS": [
        "show the logs",
        "what went wrong",
        "display error messages",
        "show job output",
        "view the log file",
        "what happened to my job",
        "show stderr",
        "display stdout",
        "check error log",
    ],
    
    # Diagnostics
    "DIAGNOSE_ERROR": [
        "why did it fail",
        "diagnose this error",
        "what went wrong",
        "help me fix this",
        "troubleshoot the failure",
        "analyze the error",
        "debug the problem",
        "figure out what happened",
        "explain the error",
        "what caused this failure",
    ],
    
    # Analysis & Results
    "ANALYSIS_INTERPRET": [
        "interpret the results",
        "what do these results mean",
        "explain the output",
        "analyze the QC report",
        "summarize the findings",
        "what does this data show",
        "interpret the metrics",
        "explain the quality scores",
    ],
    
    # References
    "REFERENCE_CHECK": [
        "check if reference genome exists",
        "do we have the human reference",
        "is the index available",
        "check for mouse genome",
        "verify reference data",
        "is GRCh38 downloaded",
        "check reference availability",
    ],
    "REFERENCE_DOWNLOAD": [
        "download human genome",
        "get GRCh38 reference",
        "fetch mouse reference",
        "download genome annotation",
        "get reference data",
        "pull the genome files",
    ],
    
    # Education
    "EDUCATION_EXPLAIN": [
        "what is RNA-seq",
        "explain differential expression",
        "what does FDR mean",
        "tell me about ChIP-seq",
        "explain normalization",
        "what is a p-value",
        "describe the workflow steps",
        "what is peak calling",
    ],
    "EDUCATION_HELP": [
        "help",
        "what can you do",
        "show commands",
        "list capabilities",
        "how do I use this",
        "what are my options",
        "show me what's possible",
    ],
    
    # Composite
    "COMPOSITE_CHECK_THEN_SEARCH": [
        "check locally first, then search online",
        "see if we have it, otherwise search databases",
        "look here first, if not found search GEO",
        "check for local data, if missing search online",
        "try local data, fall back to database search",
    ],
    
    # Meta
    "META_CONFIRM": [
        "yes",
        "ok",
        "sure",
        "go ahead",
        "do it",
        "proceed",
        "confirm",
        "yes please",
        "that's right",
        "correct",
    ],
    "META_CANCEL": [
        "no",
        "cancel",
        "stop",
        "never mind",
        "abort",
        "don't do that",
        "forget it",
        "cancel that",
    ],
    "META_GREETING": [
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "greetings",
    ],
}


# =============================================================================
# BIOINFORMATICS NER 
# =============================================================================

@dataclass
class BioEntity:
    """A recognized bioinformatics entity."""
    text: str
    entity_type: str
    canonical: str  # Normalized form
    start: int
    end: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BioinformaticsNER:
    """
    Domain-specific Named Entity Recognition for bioinformatics.
    
    Recognizes:
    - Organisms (human, mouse, Homo sapiens, etc.)
    - Assay types (RNA-seq, ChIP-seq, etc.)
    - Tissues/Cell types (brain, liver, HeLa, etc.)
    - Diseases (cancer types, etc.)
    - Dataset IDs (GSE*, ENCSR*, TCGA-*, etc.)
    - Gene names (BRCA1, TP53, etc.)
    - File formats (FASTQ, BAM, VCF, etc.)
    """
    
    # Comprehensive entity dictionaries
    ORGANISMS = {
        # Human
        "human": "Homo sapiens", "homo sapiens": "Homo sapiens", "h. sapiens": "Homo sapiens",
        "hg38": "Homo sapiens", "hg19": "Homo sapiens", "grch38": "Homo sapiens", "grch37": "Homo sapiens",
        # Mouse
        "mouse": "Mus musculus", "mus musculus": "Mus musculus", "m. musculus": "Mus musculus",
        "mm10": "Mus musculus", "mm39": "Mus musculus", "grcm38": "Mus musculus", "grcm39": "Mus musculus",
        # Rat
        "rat": "Rattus norvegicus", "rattus norvegicus": "Rattus norvegicus",
        "rn6": "Rattus norvegicus", "rn7": "Rattus norvegicus",
        # Zebrafish
        "zebrafish": "Danio rerio", "danio rerio": "Danio rerio", "danrer": "Danio rerio",
        # Drosophila
        "fly": "Drosophila melanogaster", "drosophila": "Drosophila melanogaster",
        "drosophila melanogaster": "Drosophila melanogaster", "dm6": "Drosophila melanogaster",
        # C. elegans
        "worm": "Caenorhabditis elegans", "c. elegans": "Caenorhabditis elegans",
        "caenorhabditis elegans": "Caenorhabditis elegans", "ce11": "Caenorhabditis elegans",
        # Yeast
        "yeast": "Saccharomyces cerevisiae", "saccharomyces cerevisiae": "Saccharomyces cerevisiae",
        "s. cerevisiae": "Saccharomyces cerevisiae",
        # Arabidopsis
        "arabidopsis": "Arabidopsis thaliana", "arabidopsis thaliana": "Arabidopsis thaliana",
        # E. coli
        "e. coli": "Escherichia coli", "escherichia coli": "Escherichia coli",
    }
    
    ASSAY_TYPES = {
        # RNA
        "rna-seq": "RNA-seq", "rnaseq": "RNA-seq", "rna seq": "RNA-seq", "transcriptome": "RNA-seq",
        "mrna-seq": "mRNA-seq", "total rna": "total-RNA-seq", "poly-a": "polyA-RNA-seq",
        "single-cell rna": "scRNA-seq", "scrna-seq": "scRNA-seq", "scrnaseq": "scRNA-seq",
        "10x genomics": "scRNA-seq", "smart-seq": "scRNA-seq",
        # ChIP
        "chip-seq": "ChIP-seq", "chipseq": "ChIP-seq", "chip seq": "ChIP-seq",
        "cut&run": "CUT&RUN", "cutandrun": "CUT&RUN", "cut&tag": "CUT&Tag",
        # ATAC
        "atac-seq": "ATAC-seq", "atacseq": "ATAC-seq", "atac": "ATAC-seq",
        "dnase-seq": "DNase-seq", "dnase": "DNase-seq", "faire-seq": "FAIRE-seq",
        # Methylation
        "methylation": "Bisulfite-seq", "bisulfite": "Bisulfite-seq", "wgbs": "WGBS",
        "rrbs": "RRBS", "methylome": "Bisulfite-seq", "dna methylation": "Bisulfite-seq",
        # Hi-C
        "hi-c": "Hi-C", "hic": "Hi-C", "3c": "3C", "4c": "4C", "capture-c": "Capture-C",
        "chromatin conformation": "Hi-C", "chromosome conformation": "Hi-C",
        # DNA-seq
        "wgs": "WGS", "whole genome": "WGS", "exome": "WES", "wes": "WES",
        "targeted sequencing": "targeted-seq", "amplicon": "amplicon-seq",
        # Other
        "clip-seq": "CLIP-seq", "rip-seq": "RIP-seq", "metagenomics": "metagenomics",
        "16s": "16S-rRNA", "microbiome": "metagenomics",
    }
    
    TISSUES = {
        # Brain regions
        "brain": "brain", "cortex": "cortex", "hippocampus": "hippocampus",
        "cerebellum": "cerebellum", "frontal cortex": "frontal cortex",
        "prefrontal cortex": "prefrontal cortex",
        # Organs
        "liver": "liver", "kidney": "kidney", "heart": "heart", "lung": "lung",
        "spleen": "spleen", "pancreas": "pancreas", "intestine": "intestine",
        "colon": "colon", "stomach": "stomach", "skin": "skin", "muscle": "muscle",
        # Blood
        "blood": "blood", "pbmc": "PBMC", "peripheral blood": "peripheral blood",
        "bone marrow": "bone marrow",
        # Cell types
        "t cell": "T cell", "b cell": "B cell", "macrophage": "macrophage",
        "neutrophil": "neutrophil", "monocyte": "monocyte", "dendritic cell": "dendritic cell",
        "stem cell": "stem cell", "neuron": "neuron", "astrocyte": "astrocyte",
        "fibroblast": "fibroblast", "epithelial": "epithelial",
        # Cell lines
        "hela": "HeLa", "hek293": "HEK293", "293t": "HEK293T", "k562": "K562",
        "gm12878": "GM12878", "imr90": "IMR90", "a549": "A549", "mcf7": "MCF7",
        "jurkat": "Jurkat", "thp1": "THP-1",
    }
    
    DISEASES = {
        # Cancer types
        "cancer": "cancer", "tumor": "tumor", "carcinoma": "carcinoma",
        "leukemia": "leukemia", "lymphoma": "lymphoma", "melanoma": "melanoma",
        "glioblastoma": "glioblastoma", "gbm": "glioblastoma",
        "breast cancer": "breast cancer", "brca": "breast cancer",
        "lung cancer": "lung cancer", "luad": "lung adenocarcinoma",
        "colorectal cancer": "colorectal cancer", "coad": "colon adenocarcinoma",
        "prostate cancer": "prostate cancer", "prad": "prostate adenocarcinoma",
        "ovarian cancer": "ovarian cancer", "ov": "ovarian cancer",
        "pancreatic cancer": "pancreatic cancer", "paad": "pancreatic adenocarcinoma",
        "liver cancer": "hepatocellular carcinoma", "lihc": "hepatocellular carcinoma",
        # Other diseases
        "alzheimer": "Alzheimer's disease", "parkinson": "Parkinson's disease",
        "diabetes": "diabetes", "heart disease": "heart disease",
        "autoimmune": "autoimmune disease", "covid": "COVID-19",
    }
    
    FILE_FORMATS = {
        "fastq": "FASTQ", "fq": "FASTQ", "fastq.gz": "FASTQ",
        "bam": "BAM", "sam": "SAM", "cram": "CRAM",
        "vcf": "VCF", "bcf": "BCF", "gvcf": "gVCF",
        "bed": "BED", "bedgraph": "bedGraph", "bigwig": "BigWig", "bw": "BigWig",
        "gtf": "GTF", "gff": "GFF", "gff3": "GFF3",
        "fasta": "FASTA", "fa": "FASTA", "fna": "FASTA",
        "counts": "counts", "tpm": "TPM", "fpkm": "FPKM",
        "h5ad": "H5AD", "loom": "Loom", "rds": "RDS",
    }
    
    # Regex patterns for IDs
    DATASET_ID_PATTERNS = [
        (r'\bGSE\d{4,8}\b', 'GEO'),           # GEO series
        (r'\bGSM\d{4,8}\b', 'GEO'),           # GEO sample
        (r'\bENCSR[A-Z0-9]{6}\b', 'ENCODE'),  # ENCODE experiment
        (r'\bENCFF[A-Z0-9]{6}\b', 'ENCODE'),  # ENCODE file
        (r'\bTCGA-[A-Z]{2,4}\b', 'TCGA'),     # TCGA project
        (r'\bSRR\d{6,10}\b', 'SRA'),          # SRA run
        (r'\bSRP\d{5,8}\b', 'SRA'),           # SRA project
        (r'\bPRJNA\d{5,8}\b', 'SRA'),         # BioProject
        (r'\bSAM[NED]\d{7,10}\b', 'BioSample'), # BioSample
    ]
    
    # Common gene name patterns
    GENE_PATTERN = r'\b([A-Z][A-Z0-9]{1,10})\b'  # e.g., BRCA1, TP53, GAPDH
    
    def __init__(self):
        """Initialize NER with compiled patterns."""
        import re
        self._id_patterns = [
            (re.compile(p, re.IGNORECASE), src) 
            for p, src in self.DATASET_ID_PATTERNS
        ]
        self._gene_pattern = re.compile(self.GENE_PATTERN)
        
        # Build lowercase lookup dicts for case-insensitive matching
        self._organism_lookup = {k.lower(): v for k, v in self.ORGANISMS.items()}
        self._assay_lookup = {k.lower(): v for k, v in self.ASSAY_TYPES.items()}
        self._tissue_lookup = {k.lower(): v for k, v in self.TISSUES.items()}
        self._disease_lookup = {k.lower(): v for k, v in self.DISEASES.items()}
        self._format_lookup = {k.lower(): v for k, v in self.FILE_FORMATS.items()}
    
    def extract(self, text: str) -> List[BioEntity]:
        """
        Extract bioinformatics entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of recognized entities
        """
        import re
        entities = []
        text_lower = text.lower()
        
        # Extract dataset IDs
        for pattern, source in self._id_patterns:
            for match in pattern.finditer(text):
                entities.append(BioEntity(
                    text=match.group(),
                    entity_type="DATASET_ID",
                    canonical=match.group().upper(),
                    start=match.start(),
                    end=match.end(),
                    metadata={"source": source}
                ))
        
        # Extract from dictionaries using word boundary matching
        for word_match in re.finditer(r'\b[\w\-\.]+\b', text_lower):
            word = word_match.group()
            start, end = word_match.start(), word_match.end()
            
            # Check each entity type
            if word in self._organism_lookup:
                entities.append(BioEntity(
                    text=text[start:end],
                    entity_type="ORGANISM",
                    canonical=self._organism_lookup[word],
                    start=start, end=end
                ))
            
            if word in self._assay_lookup:
                entities.append(BioEntity(
                    text=text[start:end],
                    entity_type="ASSAY_TYPE",
                    canonical=self._assay_lookup[word],
                    start=start, end=end
                ))
            
            if word in self._tissue_lookup:
                entities.append(BioEntity(
                    text=text[start:end],
                    entity_type="TISSUE",
                    canonical=self._tissue_lookup[word],
                    start=start, end=end
                ))
            
            if word in self._disease_lookup:
                entities.append(BioEntity(
                    text=text[start:end],
                    entity_type="DISEASE",
                    canonical=self._disease_lookup[word],
                    start=start, end=end
                ))
            
            if word in self._format_lookup:
                entities.append(BioEntity(
                    text=text[start:end],
                    entity_type="FILE_FORMAT",
                    canonical=self._format_lookup[word],
                    start=start, end=end
                ))
        
        # Check multi-word phrases (e.g., "breast cancer", "RNA-seq")
        for phrase_len in [3, 2]:
            words = text_lower.split()
            for i in range(len(words) - phrase_len + 1):
                phrase = " ".join(words[i:i+phrase_len])
                
                for lookup, entity_type in [
                    (self._organism_lookup, "ORGANISM"),
                    (self._assay_lookup, "ASSAY_TYPE"),
                    (self._tissue_lookup, "TISSUE"),
                    (self._disease_lookup, "DISEASE"),
                ]:
                    if phrase in lookup:
                        # Find position in original text
                        start = text_lower.find(phrase)
                        if start >= 0:
                            end = start + len(phrase)
                            entities.append(BioEntity(
                                text=text[start:end],
                                entity_type=entity_type,
                                canonical=lookup[phrase],
                                start=start, end=end
                            ))
        
        # Remove duplicates (keep highest confidence)
        seen = set()
        unique_entities = []
        for e in sorted(entities, key=lambda x: -x.confidence):
            key = (e.start, e.end, e.entity_type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(e)
        
        return unique_entities


# =============================================================================
# SEMANTIC INTENT CLASSIFIER
# =============================================================================

class SemanticIntentClassifier:
    """
    FAISS-based semantic intent classification.
    
    Uses sentence embeddings to match user queries to known intent examples.
    Falls back to cosine similarity if FAISS is not available.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_client = None,
        index_path: Optional[Path] = None,
    ):
        """
        Initialize the classifier.
        
        Args:
            embedding_model: Sentence transformer model name
            llm_client: Optional LLM client for embeddings fallback
            index_path: Path to save/load FAISS index
        """
        self.embedding_model_name = embedding_model
        self.llm_client = llm_client
        self.index_path = index_path
        
        # Initialize embedding model
        self._embedder = None
        self._use_llm_embeddings = False
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._embedder = SentenceTransformer(embedding_model)
                self._embedding_dim = self._embedder.get_sentence_embedding_dimension()
            except Exception as e:
                logger.warning(f"Failed to load SentenceTransformer: {e}")
        
        if self._embedder is None and llm_client and hasattr(llm_client, 'embed'):
            self._use_llm_embeddings = True
            # Determine embedding dimension from a test
            try:
                test_embed = llm_client.embed("test")
                self._embedding_dim = len(test_embed)
            except Exception as e:
                logger.warning(f"LLM embeddings not available: {e}")
                self._embedding_dim = 384  # Default
        elif self._embedder is None:
            self._embedding_dim = 384  # Default for fallback
        
        # FAISS index and metadata
        self._index = None
        self._intent_labels = []  # Maps index position to intent name
        self._example_texts = []  # Original texts for debugging
        
        # Build index
        self._build_index()
    
    def _embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text(s)."""
        if isinstance(texts, str):
            texts = [texts]
        
        if self._embedder is not None:
            return self._embedder.encode(texts, convert_to_numpy=True)
        
        if self._use_llm_embeddings:
            embeddings = []
            for text in texts:
                try:
                    emb = self.llm_client.embed(text)
                    embeddings.append(emb)
                except Exception as e:
                    logger.warning(f"Embedding failed for '{text[:50]}': {e}")
                    embeddings.append([0.0] * self._embedding_dim)
            return np.array(embeddings, dtype=np.float32)
        
        # Fallback: random embeddings (not useful, just prevents crash)
        logger.warning("No embedding method available, using random vectors")
        return np.random.randn(len(texts), self._embedding_dim).astype(np.float32)
    
    def _build_index(self):
        """Build FAISS index from intent examples."""
        all_texts = []
        all_labels = []
        
        for intent, examples in INTENT_EXAMPLES.items():
            for example in examples:
                all_texts.append(example)
                all_labels.append(intent)
        
        self._example_texts = all_texts
        self._intent_labels = all_labels
        
        # Generate embeddings
        logger.info(f"Building semantic index with {len(all_texts)} examples...")
        embeddings = self._embed(all_texts)
        
        # Normalize for cosine similarity
        faiss_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        if FAISS_AVAILABLE:
            # Use FAISS for efficient similarity search
            self._index = faiss.IndexFlatIP(self._embedding_dim)  # Inner product = cosine for normalized vectors
            self._index.add(faiss_embeddings.astype(np.float32))
            logger.info(f"FAISS index built with {self._index.ntotal} vectors")
        else:
            # Store normalized embeddings for manual cosine similarity
            self._index = faiss_embeddings
            logger.info(f"Fallback index built with {len(embeddings)} vectors")
    
    def classify(
        self, 
        query: str, 
        top_k: int = 5,
        threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Classify query intent using semantic similarity.
        
        Args:
            query: User query
            top_k: Number of nearest neighbors to consider
            threshold: Minimum similarity threshold
            
        Returns:
            List of (intent, confidence) tuples, sorted by confidence
        """
        # Embed query
        query_embedding = self._embed(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        if FAISS_AVAILABLE and isinstance(self._index, faiss.Index):
            # Use FAISS search
            similarities, indices = self._index.search(
                query_embedding.astype(np.float32), 
                top_k
            )
            similarities = similarities[0]
            indices = indices[0]
        else:
            # Manual cosine similarity
            similarities = np.dot(self._index, query_embedding.T).flatten()
            indices = np.argsort(similarities)[::-1][:top_k]
            similarities = similarities[indices]
        
        # Aggregate by intent (voting)
        intent_scores = {}
        for sim, idx in zip(similarities, indices):
            if sim < threshold:
                continue
            intent = self._intent_labels[idx]
            if intent not in intent_scores:
                intent_scores[intent] = []
            intent_scores[intent].append(float(sim))
        
        # Calculate aggregate score (max + weighted average)
        results = []
        for intent, scores in intent_scores.items():
            max_score = max(scores)
            avg_score = sum(scores) / len(scores)
            # Weighted combination
            final_score = 0.7 * max_score + 0.3 * avg_score
            results.append((intent, final_score))
        
        return sorted(results, key=lambda x: -x[1])
    
    def get_similar_examples(
        self, 
        query: str, 
        top_k: int = 3
    ) -> List[Tuple[str, str, float]]:
        """
        Get most similar training examples for a query.
        
        Useful for debugging and understanding classification.
        
        Returns:
            List of (intent, example_text, similarity) tuples
        """
        query_embedding = self._embed(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        if FAISS_AVAILABLE and isinstance(self._index, faiss.Index):
            similarities, indices = self._index.search(
                query_embedding.astype(np.float32), 
                top_k
            )
            similarities = similarities[0]
            indices = indices[0]
        else:
            similarities = np.dot(self._index, query_embedding.T).flatten()
            indices = np.argsort(similarities)[::-1][:top_k]
            similarities = similarities[indices]
        
        results = []
        for sim, idx in zip(similarities, indices):
            results.append((
                self._intent_labels[idx],
                self._example_texts[idx],
                float(sim)
            ))
        
        return results


# =============================================================================
# HYBRID QUERY PARSER
# =============================================================================

@dataclass
class QueryParseResult:
    """Complete result of query parsing."""
    # Intent classification
    intent: str
    intent_confidence: float
    intent_alternatives: List[Tuple[str, float]] = field(default_factory=list)
    
    # Entity extraction
    entities: List[BioEntity] = field(default_factory=list)
    
    # Slots (key-value pairs extracted)
    slots: Dict[str, str] = field(default_factory=dict)
    
    # Parsing metadata
    parse_method: str = "unknown"  # "pattern", "semantic", "llm", "hybrid"
    similar_examples: List[Tuple[str, str, float]] = field(default_factory=list)
    
    # Flags
    needs_clarification: bool = False
    clarification_prompt: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "intent": self.intent,
            "intent_confidence": self.intent_confidence,
            "intent_alternatives": self.intent_alternatives,
            "entities": [
                {
                    "text": e.text,
                    "type": e.entity_type,
                    "canonical": e.canonical,
                }
                for e in self.entities
            ],
            "slots": self.slots,
            "parse_method": self.parse_method,
            "needs_clarification": self.needs_clarification,
        }


class HybridQueryParser:
    """
    Production-grade hybrid query parser.
    
    Combines multiple approaches for robust intent understanding:
    1. Pattern matching (fast, high precision for known patterns)
    2. Semantic similarity (handles paraphrases and variations)
    3. NER (extracts domain-specific entities)
    4. LLM fallback (complex/ambiguous cases)
    
    The parser uses a confidence-based fusion strategy to combine
    results from different methods.
    """
    
    def __init__(
        self,
        llm_client = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        pattern_weight: float = 0.4,
        semantic_weight: float = 0.4,
        entity_weight: float = 0.2,
    ):
        """
        Initialize the hybrid parser.
        
        Args:
            llm_client: LLM client for complex cases and embeddings fallback
            embedding_model: Sentence transformer model
            pattern_weight: Weight for pattern matching confidence
            semantic_weight: Weight for semantic similarity
            entity_weight: Weight for entity-based inference
        """
        self.llm_client = llm_client
        self.pattern_weight = pattern_weight
        self.semantic_weight = semantic_weight
        self.entity_weight = entity_weight
        
        # Initialize components
        self.ner = BioinformaticsNER()
        self.semantic_classifier = SemanticIntentClassifier(
            embedding_model=embedding_model,
            llm_client=llm_client
        )
        
        # Import pattern-based parser
        from .parser import IntentParser
        self.pattern_parser = IntentParser(llm_client=llm_client)
        
        logger.info("HybridQueryParser initialized")
    
    def parse(self, query: str, context: Optional[Dict] = None) -> QueryParseResult:
        """
        Parse a user query using hybrid approach.
        
        Args:
            query: User query text
            context: Optional conversation context
            
        Returns:
            QueryParseResult with intent, entities, and metadata
        """
        query = query.strip()
        if not query:
            return QueryParseResult(
                intent="META_UNKNOWN",
                intent_confidence=0.0,
                parse_method="empty",
                needs_clarification=True,
                clarification_prompt="Please enter a query."
            )
        
        # Step 1: Extract entities
        entities = self.ner.extract(query)
        
        # Step 2: Pattern matching
        pattern_result = self.pattern_parser.parse(query, context)
        pattern_intent = pattern_result.primary_intent.name
        pattern_confidence = pattern_result.confidence
        
        # Step 3: Semantic classification
        semantic_results = self.semantic_classifier.classify(query)
        semantic_intent = semantic_results[0][0] if semantic_results else None
        semantic_confidence = semantic_results[0][1] if semantic_results else 0.0
        
        # Step 4: Entity-based inference
        entity_intent, entity_confidence = self._infer_from_entities(query, entities)
        
        # Step 5: Fuse results
        intent, confidence, method = self._fuse_results(
            pattern_intent, pattern_confidence,
            semantic_intent, semantic_confidence,
            entity_intent, entity_confidence,
            entities
        )
        
        # Step 6: Build slots from entities
        slots = self._build_slots(entities, pattern_result.slots)
        
        # Step 7: Get similar examples for explainability
        similar_examples = self.semantic_classifier.get_similar_examples(query, top_k=3)
        
        # Step 8: Check if clarification needed
        needs_clarification = confidence < 0.5 or intent == "META_UNKNOWN"
        clarification_prompt = None
        if needs_clarification:
            clarification_prompt = self._generate_clarification(query, entities, similar_examples)
        
        # Build alternatives list
        alternatives = []
        if pattern_intent != intent:
            alternatives.append((pattern_intent, pattern_confidence))
        if semantic_results:
            for sem_intent, sem_conf in semantic_results[1:3]:
                if sem_intent != intent:
                    alternatives.append((sem_intent, sem_conf))
        
        return QueryParseResult(
            intent=intent,
            intent_confidence=confidence,
            intent_alternatives=alternatives,
            entities=entities,
            slots=slots,
            parse_method=method,
            similar_examples=similar_examples,
            needs_clarification=needs_clarification,
            clarification_prompt=clarification_prompt,
        )
    
    def _infer_from_entities(
        self, 
        query: str, 
        entities: List[BioEntity]
    ) -> Tuple[Optional[str], float]:
        """Infer intent from extracted entities."""
        query_lower = query.lower()
        entity_types = {e.entity_type for e in entities}
        
        # Dataset ID + action verb → download
        if "DATASET_ID" in entity_types:
            if any(w in query_lower for w in ["download", "get", "fetch", "retrieve"]):
                return "DATA_DOWNLOAD", 0.85
            if any(w in query_lower for w in ["search", "find", "look"]):
                return "DATA_SEARCH", 0.7
        
        # Assay type + create/make → workflow
        if "ASSAY_TYPE" in entity_types:
            if any(w in query_lower for w in ["create", "generate", "make", "build", "workflow", "pipeline"]):
                return "WORKFLOW_CREATE", 0.8
            if any(w in query_lower for w in ["search", "find", "data"]):
                return "DATA_SEARCH", 0.7
        
        # Disease/tissue + search terms → search
        if entity_types & {"DISEASE", "TISSUE", "ORGANISM"}:
            if any(w in query_lower for w in ["data", "dataset", "find", "search", "any"]):
                return "DATA_SEARCH", 0.7
        
        return None, 0.0
    
    def _fuse_results(
        self,
        pattern_intent: str, pattern_conf: float,
        semantic_intent: Optional[str], semantic_conf: float,
        entity_intent: Optional[str], entity_conf: float,
        entities: List[BioEntity]
    ) -> Tuple[str, float, str]:
        """
        Fuse results from different parsing methods.
        
        Uses weighted voting with confidence thresholds.
        """
        candidates = {}
        
        # Add pattern result
        if pattern_conf > 0.3:
            candidates[pattern_intent] = candidates.get(pattern_intent, 0) + \
                pattern_conf * self.pattern_weight
        
        # Add semantic result
        if semantic_intent and semantic_conf > 0.3:
            candidates[semantic_intent] = candidates.get(semantic_intent, 0) + \
                semantic_conf * self.semantic_weight
        
        # Add entity-based result
        if entity_intent and entity_conf > 0.3:
            candidates[entity_intent] = candidates.get(entity_intent, 0) + \
                entity_conf * self.entity_weight
        
        if not candidates:
            return "META_UNKNOWN", 0.0, "fallback"
        
        # Find best intent
        best_intent = max(candidates, key=candidates.get)
        best_score = candidates[best_intent]
        
        # Normalize score
        max_possible = self.pattern_weight + self.semantic_weight + self.entity_weight
        confidence = min(best_score / max_possible, 1.0)
        
        # Determine method
        if pattern_intent == best_intent and pattern_conf > 0.7:
            method = "pattern"
        elif semantic_intent == best_intent and semantic_conf > 0.7:
            method = "semantic"
        elif entity_intent == best_intent:
            method = "entity"
        else:
            method = "hybrid"
        
        return best_intent, confidence, method
    
    def _build_slots(
        self, 
        entities: List[BioEntity], 
        pattern_slots: Dict[str, Any]
    ) -> Dict[str, str]:
        """Build slot dictionary from entities and pattern matches."""
        slots = dict(pattern_slots)
        
        # Clean up query slot if present - remove trailing instructions
        if "query" in slots and slots["query"]:
            query = slots["query"]
            # Remove common trailing phrases
            stop_phrases = [
                ". if not", ".if not", ", if not", ",if not",
                ". otherwise", ".otherwise", ", otherwise",
                ". then", ".then", ", then",
                ". else", ".else", ", else",
                ". find online", ", find online",
                ". search online", ", search online",
            ]
            query_lower = query.lower()
            for phrase in stop_phrases:
                idx = query_lower.find(phrase)
                if idx > 0:
                    query = query[:idx].strip()
                    break
            slots["query"] = query
        
        for entity in entities:
            if entity.entity_type == "ORGANISM":
                slots.setdefault("organism", entity.canonical)
            elif entity.entity_type == "ASSAY_TYPE":
                slots.setdefault("assay_type", entity.canonical)
            elif entity.entity_type == "TISSUE":
                slots.setdefault("tissue", entity.canonical)
            elif entity.entity_type == "DISEASE":
                slots.setdefault("disease", entity.canonical)
            elif entity.entity_type == "DATASET_ID":
                slots.setdefault("dataset_id", entity.canonical)
            elif entity.entity_type == "FILE_FORMAT":
                slots.setdefault("file_format", entity.canonical)
        
        return slots
    
    def _generate_clarification(
        self,
        query: str,
        entities: List[BioEntity],
        similar_examples: List[Tuple[str, str, float]]
    ) -> str:
        """Generate clarification prompt for ambiguous queries."""
        if not entities and not similar_examples:
            return "I'm not sure what you're asking. Try:\n• 'search for [data type]'\n• 'create a [workflow type] workflow'\n• 'help' to see all options"
        
        if entities:
            entity_str = ", ".join(f"{e.entity_type}: {e.canonical}" for e in entities[:3])
            prompt = f"I found: {entity_str}. "
        else:
            prompt = ""
        
        if similar_examples:
            suggestions = [f"'{ex[1]}'" for ex in similar_examples[:2]]
            prompt += f"Did you mean something like: {' or '.join(suggestions)}?"
        
        return prompt


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_parser(llm_client=None) -> HybridQueryParser:
    """Create a configured HybridQueryParser instance."""
    return HybridQueryParser(llm_client=llm_client)


def quick_parse(query: str, llm_client=None) -> Dict[str, Any]:
    """
    Quick parse a query and return dictionary result.
    
    Convenience function for simple use cases.
    """
    parser = HybridQueryParser(llm_client=llm_client)
    result = parser.parse(query)
    return result.to_dict()
