#!/usr/bin/env python3
"""
Comprehensive Conversation Generator
=====================================

Generates 5000+ diverse test conversations for robust evaluation.

Design Principles:
1. Diversity over quantity - cover all edge cases
2. Realistic complexity - match real user behavior
3. Adversarial robustness - include tricky cases
4. Multi-turn context - test conversation flow
5. Domain coverage - all bioinformatics areas

Categories:
- Simple queries (1-5 words)
- Standard queries (6-15 words)
- Complex queries (16-30 words)
- Long queries (30+ words)
- Multi-turn conversations
- Negation and preferences
- Ambiguous and vague
- Typos and misspellings
- Mixed case and formatting
- Code/path mixed queries
- Multi-intent queries
- Domain-specific jargon
- Abbreviations and acronyms
- Non-English scientist names
- Real paper titles/abstracts
"""

import random
import sqlite3
import json
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import itertools

# =============================================================================
# EXPANDED DOMAIN VOCABULARY
# =============================================================================

# Organisms with common variations and misspellings
ORGANISMS = {
    "human": ["human", "homo sapiens", "h. sapiens", "hsapiens", "hg38", "hg19", "grch38", "grch37", "huamn", "humna"],
    "mouse": ["mouse", "mus musculus", "m. musculus", "mmusculus", "mm10", "mm9", "mice", "murine", "mosue"],
    "rat": ["rat", "rattus norvegicus", "r. norvegicus", "rn6", "rn7", "rats"],
    "zebrafish": ["zebrafish", "danio rerio", "d. rerio", "drerio", "danrer11", "zebra fish", "zfish"],
    "fly": ["fly", "drosophila", "drosophila melanogaster", "d. melanogaster", "dm6", "dm3", "fruit fly", "fruitfly"],
    "worm": ["worm", "c. elegans", "caenorhabditis elegans", "celegans", "ce11", "ce10", "nematode"],
    "yeast": ["yeast", "s. cerevisiae", "saccharomyces cerevisiae", "scerevisiae", "saccer3", "budding yeast"],
    "arabidopsis": ["arabidopsis", "a. thaliana", "arabidopsis thaliana", "athaliana", "tair10", "thale cress"],
    "chicken": ["chicken", "gallus gallus", "g. gallus", "galgal6"],
    "pig": ["pig", "sus scrofa", "s. scrofa", "sscrofa11", "swine", "porcine"],
    "dog": ["dog", "canis familiaris", "c. familiaris", "canfam4", "canine"],
    "cow": ["cow", "bos taurus", "b. taurus", "bostar", "bovine", "cattle"],
    "monkey": ["monkey", "macaque", "rhesus", "macaca mulatta", "rhemac10"],
    "frog": ["frog", "xenopus", "x. tropicalis", "xenopus tropicalis", "xentro10"],
    "plant": ["plant", "maize", "zea mays", "corn", "rice", "oryza sativa", "wheat", "triticum"],
    "bacteria": ["bacteria", "e. coli", "escherichia coli", "ecoli", "bacterial", "prokaryote"],
}

# Tissues with anatomical variations
TISSUES = {
    "brain": ["brain", "cerebrum", "neural", "cns", "central nervous system", "brain tissue", "whole brain"],
    "cortex": ["cortex", "cerebral cortex", "cortical", "neocortex", "prefrontal cortex", "frontal cortex", "motor cortex"],
    "hippocampus": ["hippocampus", "hippocampal", "hpc", "dentate gyrus", "ca1", "ca3"],
    "liver": ["liver", "hepatic", "hepatocyte", "liver tissue", "hepatocytes"],
    "kidney": ["kidney", "renal", "nephron", "kidney tissue", "renal cortex"],
    "heart": ["heart", "cardiac", "cardiomyocyte", "myocardium", "heart tissue", "ventricular"],
    "lung": ["lung", "pulmonary", "respiratory", "alveolar", "bronchial", "lung tissue"],
    "blood": ["blood", "peripheral blood", "whole blood", "blood cells", "circulating"],
    "pbmc": ["pbmc", "peripheral blood mononuclear cells", "peripheral blood mononuclear", "mononuclear cells"],
    "bone_marrow": ["bone marrow", "bm", "marrow", "hematopoietic"],
    "spleen": ["spleen", "splenic", "spleen tissue"],
    "thymus": ["thymus", "thymic", "thymocyte"],
    "lymph_node": ["lymph node", "lymphatic", "ln", "lymph nodes"],
    "muscle": ["muscle", "skeletal muscle", "smooth muscle", "cardiac muscle", "myocyte"],
    "skin": ["skin", "dermal", "epidermis", "dermis", "keratinocyte"],
    "intestine": ["intestine", "gut", "intestinal", "small intestine", "large intestine", "colon", "ileum"],
    "pancreas": ["pancreas", "pancreatic", "islet", "beta cell"],
    "adipose": ["adipose", "fat", "adipocyte", "white adipose", "brown adipose"],
    "placenta": ["placenta", "placental", "trophoblast"],
    "testis": ["testis", "testes", "testicular", "spermatocyte"],
    "ovary": ["ovary", "ovarian", "oocyte"],
    "embryo": ["embryo", "embryonic", "e14.5", "e11.5", "e8.5", "embryonic day"],
    "tumor": ["tumor", "tumour", "cancer tissue", "malignant", "neoplastic"],
}

# Comprehensive assay types with all variations
ASSAY_TYPES = {
    "RNA-seq": {
        "canonical": "RNA-seq",
        "aliases": ["rna-seq", "rnaseq", "rna seq", "mrna-seq", "transcriptome", "gene expression", 
                    "transcriptomic", "expression profiling", "rna sequencing", "bulk rna-seq",
                    "total rna-seq", "polya rna-seq", "stranded rna-seq"],
        "analysis": ["differential expression", "gene expression analysis", "transcriptome analysis",
                     "dge", "deseq2", "edger", "kallisto", "salmon", "htseq"],
    },
    "scRNA-seq": {
        "canonical": "scRNA-seq",
        "aliases": ["scrna-seq", "scrnaseq", "single cell rna", "single-cell rna", "10x", "10x genomics",
                    "smart-seq", "smartseq2", "drop-seq", "dropseq", "cel-seq", "single cell transcriptomics",
                    "sc rna-seq", "indrops", "single cell expression"],
        "analysis": ["clustering", "umap", "tsne", "seurat", "scanpy", "cell type annotation",
                     "trajectory analysis", "pseudotime", "velocity"],
    },
    "ChIP-seq": {
        "canonical": "ChIP-seq",
        "aliases": ["chip-seq", "chipseq", "chip seq", "chromatin immunoprecipitation", "chip",
                    "histone chip", "tf chip", "transcription factor binding"],
        "analysis": ["peak calling", "macs2", "homer", "chip-seq analysis", "motif analysis",
                     "differential binding"],
    },
    "ATAC-seq": {
        "canonical": "ATAC-seq",
        "aliases": ["atac-seq", "atacseq", "atac seq", "atac", "chromatin accessibility",
                    "open chromatin", "accessible chromatin", "transposase"],
        "analysis": ["peak calling", "chromatin accessibility analysis", "footprinting",
                     "nucleosome positioning"],
    },
    "Hi-C": {
        "canonical": "Hi-C",
        "aliases": ["hi-c", "hic", "3d genome", "chromosome conformation", "chromatin conformation",
                    "3c", "4c", "5c", "capture-c", "micro-c", "hichip", "plac-seq"],
        "analysis": ["contact matrix", "tad calling", "loop calling", "compartment analysis",
                     "juicer", "hicpro", "cooltools"],
    },
    "WGBS": {
        "canonical": "WGBS",
        "aliases": ["wgbs", "bisulfite", "methylation", "dna methylation", "bisulfite-seq",
                    "methylome", "cpg methylation", "whole genome bisulfite", "bs-seq",
                    "em-seq", "enzymatic methyl-seq", "oxbs-seq", "tab-seq"],
        "analysis": ["dmr calling", "differential methylation", "bismark", "methylkit",
                     "methylation analysis"],
    },
    "RRBS": {
        "canonical": "RRBS",
        "aliases": ["rrbs", "reduced representation bisulfite", "targeted methylation"],
        "analysis": ["methylation analysis", "dmr calling"],
    },
    "WGS": {
        "canonical": "WGS",
        "aliases": ["wgs", "whole genome sequencing", "whole genome", "genome sequencing",
                    "dna sequencing", "genomic", "resequencing", "de novo genome"],
        "analysis": ["variant calling", "snp calling", "indel calling", "structural variants",
                     "gatk", "bcftools", "freebayes", "deepvariant"],
    },
    "WES": {
        "canonical": "WES",
        "aliases": ["wes", "whole exome", "exome sequencing", "exome", "exome-seq",
                    "targeted exome", "clinical exome"],
        "analysis": ["variant calling", "mutation calling", "clinical variant analysis"],
    },
    "CLIP-seq": {
        "canonical": "CLIP-seq",
        "aliases": ["clip-seq", "clipseq", "clip", "rna binding", "rna-binding", "protein-rna",
                    "iclip", "eclip", "par-clip", "hits-clip", "rip-seq", "ripseq"],
        "analysis": ["binding site analysis", "motif discovery", "rna binding analysis"],
    },
    "metagenomics": {
        "canonical": "metagenomics",
        "aliases": ["metagenomics", "microbiome", "16s", "16s rrna", "shotgun metagenomics",
                    "metagenomic", "microbial community", "amplicon sequencing", "its sequencing"],
        "analysis": ["taxonomic profiling", "functional profiling", "qiime", "metaphlan",
                     "kraken", "diversity analysis"],
    },
    "CUT&RUN": {
        "canonical": "CUT&RUN",
        "aliases": ["cut&run", "cutnrun", "cut and run", "cutandrun", "cleavage under targets"],
        "analysis": ["peak calling", "chromatin profiling"],
    },
    "CUT&Tag": {
        "canonical": "CUT&Tag",
        "aliases": ["cut&tag", "cutntag", "cut and tag", "cutandtag"],
        "analysis": ["peak calling", "epigenetic profiling"],
    },
    "Long-read": {
        "canonical": "Long-read",
        "aliases": ["long-read", "long read", "nanopore", "pacbio", "ont", "oxford nanopore",
                    "hifi", "ccs", "clr", "smrt sequencing", "third generation"],
        "analysis": ["de novo assembly", "structural variant calling", "isoform detection",
                     "base modification detection"],
    },
    "scATAC-seq": {
        "canonical": "scATAC-seq",
        "aliases": ["scatac-seq", "scatacseq", "single cell atac", "single-cell atac",
                    "10x scatac", "sciATAC"],
        "analysis": ["chromatin accessibility", "cell type identification", "archR", "signac"],
    },
    "Spatial": {
        "canonical": "Spatial",
        "aliases": ["spatial transcriptomics", "visium", "10x visium", "slide-seq", "merfish",
                    "xenium", "cosmx", "stereo-seq", "spatial omics", "st-seq"],
        "analysis": ["spatial analysis", "tissue mapping", "cell deconvolution"],
    },
    "Ribo-seq": {
        "canonical": "Ribo-seq",
        "aliases": ["ribo-seq", "riboseq", "ribosome profiling", "rp", "translation profiling",
                    "ribosome footprinting"],
        "analysis": ["translation efficiency", "orf detection", "ribosome occupancy"],
    },
    "PRO-seq": {
        "canonical": "PRO-seq",
        "aliases": ["pro-seq", "proseq", "precision run-on", "gro-seq", "groseq",
                    "nascent transcription", "run-on sequencing"],
        "analysis": ["nascent transcription analysis", "enhancer analysis", "pause index"],
    },
}

# Histone marks
HISTONE_MARKS = [
    "H3K4me1", "H3K4me2", "H3K4me3", 
    "H3K9me1", "H3K9me2", "H3K9me3", "H3K9ac",
    "H3K27me3", "H3K27ac",
    "H3K36me3",
    "H3K79me2",
    "H4K20me1",
    "H2AZ", "H2A.Z",
    "H3K4me3+H3K27me3",  # bivalent
]

# Transcription factors and proteins
TF_PROTEINS = [
    "CTCF", "p300", "EP300", "POLR2A", "Pol2", "RNAPII", "RNA Pol II",
    "MYC", "MAX", "CEBPB", "JUND", "JUN", "FOS",
    "FOXA1", "FOXA2", "GATA1", "GATA3", "GATA4",
    "ESR1", "ER", "AR", "PGR",
    "TP53", "p53", "MYB", "ELF1", "ELK1",
    "STAT1", "STAT3", "STAT5", "NF-kB", "NFKB", "RELA",
    "SOX2", "OCT4", "POU5F1", "NANOG", "KLF4",
    "SMAD2", "SMAD3", "SMAD4",
    "YY1", "ZNF143", "RAD21", "SMC1A", "SMC3",  # cohesin
]

# Diseases with variations
DISEASES = {
    "cancer": ["cancer", "tumor", "tumour", "carcinoma", "malignancy", "neoplasm", "oncology"],
    "breast_cancer": ["breast cancer", "brca", "breast carcinoma", "mammary cancer", "breast tumor"],
    "lung_cancer": ["lung cancer", "nsclc", "sclc", "lung carcinoma", "lung adenocarcinoma", "lung squamous"],
    "leukemia": ["leukemia", "leukaemia", "aml", "cml", "all", "cll", "acute leukemia", "chronic leukemia"],
    "glioblastoma": ["glioblastoma", "gbm", "glioma", "brain tumor", "brain cancer"],
    "melanoma": ["melanoma", "skin cancer", "cutaneous melanoma"],
    "colorectal": ["colorectal cancer", "colon cancer", "rectal cancer", "crc"],
    "prostate": ["prostate cancer", "pca", "prostate adenocarcinoma"],
    "pancreatic": ["pancreatic cancer", "pdac", "pancreatic ductal adenocarcinoma"],
    "alzheimers": ["alzheimer's", "alzheimers", "ad", "alzheimer disease", "alzheimer's disease"],
    "parkinsons": ["parkinson's", "parkinsons", "pd", "parkinson disease"],
    "diabetes": ["diabetes", "t2d", "type 2 diabetes", "t1d", "type 1 diabetes", "diabetic"],
    "covid": ["covid-19", "covid", "sars-cov-2", "coronavirus", "sarscov2"],
    "autoimmune": ["autoimmune", "lupus", "sle", "rheumatoid arthritis", "ra", "ms", "multiple sclerosis"],
}

# Databases with variations
DATABASES = {
    "ENCODE": ["encode", "ENCODE", "encode project", "encode database", "encodeproject"],
    "GEO": ["geo", "GEO", "gene expression omnibus", "ncbi geo", "gse", "gsm", "gds"],
    "TCGA": ["tcga", "TCGA", "cancer genome atlas", "the cancer genome atlas"],
    "SRA": ["sra", "SRA", "sequence read archive", "ncbi sra"],
    "GDC": ["gdc", "GDC", "genomic data commons", "nci gdc"],
    "ArrayExpress": ["arrayexpress", "array express", "ebi arrayexpress"],
    "dbGaP": ["dbgap", "dbGaP", "database of genotypes and phenotypes"],
    "ENA": ["ena", "ENA", "european nucleotide archive"],
    "DDBJ": ["ddbj", "DDBJ", "dna data bank of japan"],
}

# File paths for testing
PATHS = [
    "/data/raw", "/data/samples", "/data/fastq", "/data/bam",
    "/projects/rnaseq", "/projects/chipseq", "/projects/atacseq",
    "/home/user/data", "/home/user/analysis", "/home/user/results",
    "~/experiments", "~/projects", "~/data",
    "/scratch/fastq", "/scratch/bam", "/scratch/results",
    "/shared/genomes", "/shared/references", "/shared/annotations",
    "/mnt/storage/data", "/mnt/nas/projects",
]

# Dataset IDs
DATASET_IDS = [
    "GSE12345", "GSE98765", "GSE45678", "GSE11111", "GSE22222",
    "ENCSR000ABC", "ENCSR123XYZ", "ENCSR456DEF", "ENCSR789GHI",
    "PRJNA123456", "PRJNA654321", "PRJNA111222",
    "SRR1234567", "SRR7654321", "SRR1111111",
    "TCGA-BRCA", "TCGA-LUAD", "TCGA-PRAD", "TCGA-GBM", "TCGA-COAD",
    "E-MTAB-1234", "E-MTAB-5678",
]

# Job IDs
JOB_IDS = [str(i) for i in range(10000, 99999, 1111)] + ["slurm-12345", "pbs-67890", "job_001"]


# =============================================================================
# QUERY TEMPLATES BY CATEGORY
# =============================================================================

DATA_SEARCH_TEMPLATES = [
    # Simple
    "find {organism} {assay} data",
    "search for {assay} in {organism}",
    "{organism} {tissue} {assay}",
    "get {assay} data",
    "look for {organism} data",
    # Standard
    "search {database} for {organism} {tissue} {assay} data",
    "find {assay} datasets in {organism} {tissue}",
    "I need {organism} {assay} data from {tissue}",
    "looking for {assay} in {organism} {tissue}",
    "can you find {organism} {tissue} {assay}",
    # Complex
    "search for {organism} {tissue} {assay} data with {histone} in {database}",
    "find all {assay} datasets from {organism} {tissue} related to {disease}",
    "I'm looking for {organism} {tissue} samples with {assay} and {assay2} data",
    "can you search {database} for {organism} {assay} data from {tissue} in {disease}",
    # With exclusions
    "find {organism} {assay} but not {tissue}",
    "search for {assay} data excluding {organism}",
    "{assay} data, but skip {database}",
    # With preferences
    "prefer {database} for {organism} {assay}",
    "find {assay} primarily from {database}",
]

DATA_DOWNLOAD_TEMPLATES = [
    # Simple
    "download {dataset_id}",
    "get {dataset_id}",
    "fetch {dataset_id}",
    # Standard
    "download dataset {dataset_id}",
    "download the {organism} data",
    "get the {assay} files",
    "fetch all samples from {dataset_id}",
    # Complex
    "download {dataset_id} and also {dataset_id2}",
    "get the {assay} data from {database}",
    "download samples except controls",
    "fetch all {organism} samples but not input",
    # With paths
    "download {dataset_id} to {path}",
    "save the data to {path}",
]

WORKFLOW_CREATE_TEMPLATES = [
    # Simple
    "create {assay} workflow",
    "make {assay} pipeline",
    "{assay} analysis",
    "run {assay}",
    # Standard
    "create a {assay} workflow for {organism}",
    "generate {assay} pipeline for {organism} {tissue}",
    "build {assay} analysis workflow",
    "I need a {assay} workflow",
    "set up {assay} pipeline",
    # Complex
    "create {assay} workflow for {organism} {tissue} with {tool} aligner",
    "generate a {assay} pipeline including {analysis} for {organism}",
    "build {assay} workflow for {organism} samples from {tissue} with {tool}",
    "I want to analyze {organism} {tissue} {assay} data with {analysis}",
    # With preferences
    "create {assay} workflow using {tool} instead of {tool2}",
    "generate {assay} pipeline, prefer {tool}",
    "{assay} workflow but use {tool} not {tool2}",
    # Multi-step
    "first align with {tool}, then do {analysis}",
    "create workflow: {assay} followed by {analysis}",
    # Domain specific
    "differential expression workflow for {organism}",
    "peak calling pipeline for {histone} ChIP-seq",
    "variant calling workflow for {organism} {tissue}",
    "single cell analysis for {organism} {tissue}",
]

JOB_SUBMIT_TEMPLATES = [
    # Simple
    "run it",
    "submit",
    "execute",
    "start the job",
    # Standard
    "submit the workflow",
    "run the pipeline",
    "execute the analysis",
    "start the job",
    "submit my {assay} workflow",
    # With paths
    "run the workflow in {path}",
    "submit pipeline from {path}",
    "execute {path}",
    # Complex
    "submit to slurm with 16 cores",
    "run on the cluster",
    "execute with high memory",
]

JOB_STATUS_TEMPLATES = [
    # Simple
    "job status",
    "check status",
    "is it done?",
    # Standard
    "what's the status of job {job_id}",
    "check job {job_id}",
    "is job {job_id} running?",
    "how is my job doing?",
    # Complex
    "check status of all my jobs",
    "is the {assay} analysis done?",
    "what's happening with job {job_id}",
]

JOB_LIST_TEMPLATES = [
    # Simple
    "list jobs",
    "my jobs",
    "show jobs",
    # Standard
    "list all jobs",
    "show my running jobs",
    "what jobs are running?",
    "list pending jobs",
    # Variations
    "what's currently running?",
    "show all active jobs",
    "list my queued jobs",
]

JOB_LOGS_TEMPLATES = [
    # Simple
    "show logs",
    "get logs",
    "view output",
    # Standard
    "show logs for job {job_id}",
    "get output of {job_id}",
    "what's the output?",
    "view job logs",
    # Complex
    "show error logs for {job_id}",
    "get the last 100 lines of output",
    "what went wrong with job {job_id}",
]

JOB_CANCEL_TEMPLATES = [
    "cancel job {job_id}",
    "stop job {job_id}",
    "kill job {job_id}",
    "abort the job",
    "cancel all my jobs",
    "stop the running job",
]

EDUCATION_EXPLAIN_TEMPLATES = [
    # Simple
    "what is {assay}?",
    "explain {assay}",
    "{assay}?",
    # Standard
    "what is {assay} used for?",
    "explain how {assay} works",
    "tell me about {assay}",
    "describe {assay}",
    "what does {assay} measure?",
    # Complex
    "what's the difference between {assay} and {assay2}?",
    "how does {assay} compare to {assay2}?",
    "explain the {assay} analysis workflow",
    "what are the best practices for {assay}?",
    # Domain concepts
    "what is differential expression?",
    "explain peak calling",
    "what are TADs?",
    "how does {tool} work?",
    "what is a {histone} mark?",
]

EDUCATION_HELP_TEMPLATES = [
    "help",
    "what can you do?",
    "show commands",
    "list features",
    "how do I use this?",
    "what are my options?",
    "I'm new here",
    "getting started",
]

DATA_SCAN_TEMPLATES = [
    "list files in {path}",
    "what's in {path}?",
    "show contents of {path}",
    "inventory {path}",
    "scan {path}",
    "check what data I have",
    "show my local files",
]

# =============================================================================
# MULTI-TURN CONVERSATION TEMPLATES
# =============================================================================

MULTI_TURN_SCENARIOS = [
    # Search -> Download -> Workflow
    [
        ("search for {organism} {assay} data", "DATA_SEARCH"),
        ("download the first one", "DATA_DOWNLOAD"),
        ("now create a workflow for it", "WORKFLOW_CREATE"),
        ("submit it", "JOB_SUBMIT"),
        ("check status", "JOB_STATUS"),
    ],
    # Workflow -> Corrections
    [
        ("create {assay} workflow", "WORKFLOW_CREATE"),
        ("actually use {tool} instead", "WORKFLOW_CREATE"),
        ("add {analysis} step", "WORKFLOW_CREATE"),
        ("run it", "JOB_SUBMIT"),
    ],
    # Education -> Action
    [
        ("what is {assay}?", "EDUCATION_EXPLAIN"),
        ("how do I analyze it?", "EDUCATION_TUTORIAL"),
        ("ok, find some {organism} data", "DATA_SEARCH"),
        ("create a workflow", "WORKFLOW_CREATE"),
    ],
    # Job management flow
    [
        ("submit workflow", "JOB_SUBMIT"),
        ("is it done?", "JOB_STATUS"),
        ("show me the logs", "JOB_LOGS"),
        ("looks like an error, what went wrong?", "DIAGNOSE_ERROR"),
    ],
    # Refinement flow
    [
        ("find {organism} {assay} data", "DATA_SEARCH"),
        ("only from {tissue}", "DATA_SEARCH"),
        ("exclude {disease} samples", "DATA_SEARCH"),
        ("in {database}", "DATA_SEARCH"),
    ],
]

# =============================================================================
# NEGATION AND PREFERENCE TEMPLATES
# =============================================================================

NEGATION_TEMPLATES = [
    # Exclusions
    ("create {assay} workflow but not for {organism}", "WORKFLOW_CREATE"),
    ("find {organism} data excluding {tissue}", "DATA_SEARCH"),
    ("download all except controls", "DATA_DOWNLOAD"),
    ("{assay} without {organism}", "DATA_SEARCH"),
    ("skip {organism} samples", "DATA_SEARCH"),
    ("no {tissue} data", "DATA_SEARCH"),
    # Preferences
    ("use {tool} instead of {tool2}", "WORKFLOW_CREATE"),
    ("prefer {tool} over {tool2}", "WORKFLOW_CREATE"),
    ("{tool} not {tool2}", "WORKFLOW_CREATE"),
    ("I'd rather use {tool}", "WORKFLOW_CREATE"),
    # Corrections
    ("wait, I meant {organism} not {organism2}", "META_CORRECT"),
    ("actually, {assay} not {assay2}", "META_CORRECT"),
    ("change to {tissue}", "META_CORRECT"),
    ("no, use {tool}", "META_CORRECT"),
]

# =============================================================================
# AMBIGUOUS AND EDGE CASE TEMPLATES
# =============================================================================

AMBIGUOUS_TEMPLATES = [
    # Extremely vague
    ("data", "META_UNKNOWN"),
    ("analysis", "META_UNKNOWN"),
    ("help", "EDUCATION_HELP"),
    ("?", "META_UNKNOWN"),
    ("hmm", "META_UNKNOWN"),
    # Slightly vague
    ("something with RNA", "META_UNKNOWN"),
    ("process my files", "META_UNKNOWN"),
    ("I have some samples", "META_UNKNOWN"),
    ("can you help?", "EDUCATION_HELP"),
    ("I need to analyze data", "META_UNKNOWN"),
    # Mixed signals
    ("search for a way to create a pipeline", "EDUCATION_EXPLAIN"),
    ("find help about workflows", "EDUCATION_EXPLAIN"),
    ("download information about {assay}", "EDUCATION_EXPLAIN"),
]

EDGE_CASE_TEMPLATES = [
    # ALL CAPS
    ("FIND {ORGANISM} {ASSAY} DATA", "DATA_SEARCH"),
    ("CREATE {ASSAY} WORKFLOW", "WORKFLOW_CREATE"),
    ("SUBMIT JOB", "JOB_SUBMIT"),
    # all lowercase
    ("find {organism} {assay} data", "DATA_SEARCH"),
    # Mixed case (realistic)
    ("Find {Organism} {Assay} Data", "DATA_SEARCH"),
    # Extra punctuation
    ("find {organism} {assay} data!!!", "DATA_SEARCH"),
    ("create {assay} workflow???", "WORKFLOW_CREATE"),
    ("what is {assay}????", "EDUCATION_EXPLAIN"),
    # Extra whitespace
    ("find   {organism}   {assay}   data", "DATA_SEARCH"),
    ("  create {assay} workflow  ", "WORKFLOW_CREATE"),
    # Numbers in queries
    ("find 10x genomics data", "DATA_SEARCH"),
    ("create 16S workflow", "WORKFLOW_CREATE"),
    ("search for H3K27ac ChIP-seq", "DATA_SEARCH"),
    # Special characters
    ("find CUT&RUN data", "DATA_SEARCH"),
    ("search for Hi-C", "DATA_SEARCH"),
    ("what is 3' RNA-seq?", "EDUCATION_EXPLAIN"),
    # Typos
    ("fnd {organism} data", "DATA_SEARCH"),
    ("crate {assay} workflow", "WORKFLOW_CREATE"),
    ("serach for {assay}", "DATA_SEARCH"),
    ("downlod {dataset_id}", "DATA_DOWNLOAD"),
    # Long queries
    ("I want to search for {organism} {tissue} {assay} data in {database} but only for samples that have both {assay} and {assay2} and are related to {disease}", "DATA_SEARCH"),
]

# =============================================================================
# ADVERSARIAL TEMPLATES
# =============================================================================

ADVERSARIAL_TEMPLATES = [
    # Intent confusion
    ("search for a workflow to create", "EDUCATION_EXPLAIN"),
    ("download instructions for {assay}", "EDUCATION_EXPLAIN"),
    ("explain how to search for data", "EDUCATION_TUTORIAL"),
    # Negation tricks
    ("don't create a workflow, just search", "DATA_SEARCH"),
    ("I don't want to download, just search", "DATA_SEARCH"),
    ("this is not about {assay}, it's about {assay2}", "DATA_SEARCH"),
    # Multi-intent (should pick primary)
    ("search and download {organism} data", "DATA_SEARCH"),
    ("find data, create workflow, and run it", "DATA_SEARCH"),
    # Embedded commands
    ("can you run 'ls -la' for me?", "META_UNKNOWN"),
    ("execute: print('hello')", "META_UNKNOWN"),
    # Misleading context
    ("I was searching for data but now I want to create a workflow", "WORKFLOW_CREATE"),
    ("forget about searching, let's download", "DATA_DOWNLOAD"),
]

# Tools for substitution
ALIGNERS = ["BWA", "Bowtie2", "STAR", "HISAT2", "minimap2", "salmon", "kallisto"]
PEAK_CALLERS = ["MACS2", "HOMER", "SICER", "Genrich"]
VARIANT_CALLERS = ["GATK", "bcftools", "freebayes", "DeepVariant", "strelka2"]
SC_TOOLS = ["Seurat", "Scanpy", "Monocle", "scVI", "CellRanger"]


# =============================================================================
# GENERATOR CLASS
# =============================================================================

@dataclass
class GeneratedConversation:
    """A generated conversation for testing."""
    id: str
    name: str
    category: str
    difficulty: str  # easy, medium, hard, adversarial
    turns: List[Dict[str, Any]]
    source: str = "comprehensive_generator"
    tags: List[str] = None
    description: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ComprehensiveGenerator:
    """Generate diverse test conversations."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.generated_hashes = set()
    
    def _hash_query(self, query: str) -> str:
        """Create hash of query to avoid duplicates."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()[:12]
    
    def _random_org(self) -> Tuple[str, str]:
        """Get random organism (canonical, variation)."""
        canonical = random.choice(list(ORGANISMS.keys()))
        variation = random.choice(ORGANISMS[canonical])
        return canonical, variation
    
    def _random_tissue(self) -> Tuple[str, str]:
        """Get random tissue (canonical, variation)."""
        canonical = random.choice(list(TISSUES.keys()))
        variation = random.choice(TISSUES[canonical])
        return canonical, variation
    
    def _random_assay(self) -> Tuple[str, str, List[str]]:
        """Get random assay (canonical, variation, analyses)."""
        canonical = random.choice(list(ASSAY_TYPES.keys()))
        assay_info = ASSAY_TYPES[canonical]
        variation = random.choice(assay_info["aliases"])
        analyses = assay_info.get("analysis", [])
        return canonical, variation, analyses
    
    def _random_database(self) -> Tuple[str, str]:
        """Get random database (canonical, variation)."""
        canonical = random.choice(list(DATABASES.keys()))
        variation = random.choice(DATABASES[canonical])
        return canonical, variation
    
    def _random_disease(self) -> Tuple[str, str]:
        """Get random disease (canonical, variation)."""
        canonical = random.choice(list(DISEASES.keys()))
        variation = random.choice(DISEASES[canonical])
        return canonical, variation
    
    def _fill_template(self, template: str) -> Tuple[str, Dict[str, str]]:
        """Fill a template with random values, return query and expected entities."""
        entities = {}
        query = template
        
        # Fill organisms
        if "{organism}" in query or "{ORGANISM}" in query:
            canonical, variation = self._random_org()
            query = query.replace("{organism}", variation).replace("{ORGANISM}", variation.upper())
            entities["ORGANISM"] = canonical
        
        if "{organism2}" in query:
            canonical, variation = self._random_org()
            query = query.replace("{organism2}", variation)
        
        if "{Organism}" in query:
            canonical, variation = self._random_org()
            query = query.replace("{Organism}", variation.title())
            entities["ORGANISM"] = canonical
        
        # Fill tissues
        if "{tissue}" in query:
            canonical, variation = self._random_tissue()
            query = query.replace("{tissue}", variation)
            entities["TISSUE"] = canonical
        
        # Fill assays
        if "{assay}" in query or "{ASSAY}" in query:
            canonical, variation, analyses = self._random_assay()
            query = query.replace("{assay}", variation).replace("{ASSAY}", variation.upper())
            entities["ASSAY_TYPE"] = canonical
        
        if "{assay2}" in query:
            canonical, variation, _ = self._random_assay()
            query = query.replace("{assay2}", variation)
        
        if "{Assay}" in query:
            canonical, variation, _ = self._random_assay()
            query = query.replace("{Assay}", variation.title())
            entities["ASSAY_TYPE"] = canonical
        
        # Fill analyses
        if "{analysis}" in query:
            _, _, analyses = self._random_assay()
            if analyses:
                analysis = random.choice(analyses)
                query = query.replace("{analysis}", analysis)
        
        # Fill databases
        if "{database}" in query:
            canonical, variation = self._random_database()
            query = query.replace("{database}", variation)
            entities["DATABASE"] = canonical
        
        # Fill diseases
        if "{disease}" in query:
            canonical, variation = self._random_disease()
            query = query.replace("{disease}", variation)
            entities["DISEASE"] = canonical
        
        # Fill histone marks
        if "{histone}" in query:
            histone = random.choice(HISTONE_MARKS)
            query = query.replace("{histone}", histone)
            entities["HISTONE_MARK"] = histone
        
        # Fill tools
        if "{tool}" in query:
            tool = random.choice(ALIGNERS + PEAK_CALLERS + VARIANT_CALLERS + SC_TOOLS)
            query = query.replace("{tool}", tool)
        
        if "{tool2}" in query:
            tool = random.choice(ALIGNERS + PEAK_CALLERS + VARIANT_CALLERS + SC_TOOLS)
            query = query.replace("{tool2}", tool)
        
        # Fill paths
        if "{path}" in query:
            path = random.choice(PATHS)
            query = query.replace("{path}", path)
            entities["PATH"] = path
        
        # Fill dataset IDs
        if "{dataset_id}" in query:
            dataset = random.choice(DATASET_IDS)
            query = query.replace("{dataset_id}", dataset)
            entities["DATASET_ID"] = dataset
        
        if "{dataset_id2}" in query:
            dataset = random.choice(DATASET_IDS)
            query = query.replace("{dataset_id2}", dataset)
        
        # Fill job IDs
        if "{job_id}" in query:
            job_id = random.choice(JOB_IDS)
            query = query.replace("{job_id}", job_id)
            entities["JOB_ID"] = job_id
        
        return query, entities
    
    def _get_expected_tool(self, intent: str) -> str:
        """Map intent to expected tool."""
        intent_to_tool = {
            "DATA_SEARCH": "search_databases",
            "DATA_DOWNLOAD": "download_data",
            "DATA_SCAN": "scan_local_data",
            "DATA_DESCRIBE": "describe_data",
            "WORKFLOW_CREATE": "generate_workflow",
            "JOB_SUBMIT": "submit_job",
            "JOB_STATUS": "check_job_status",
            "JOB_LIST": "list_jobs",
            "JOB_LOGS": "show_logs",
            "JOB_CANCEL": "cancel_job",
            "EDUCATION_EXPLAIN": "explain_concept",
            "EDUCATION_TUTORIAL": "show_tutorial",
            "EDUCATION_HELP": "show_help",
            "META_UNKNOWN": "clarify_intent",
            "META_CONFIRM": "confirm_action",
            "META_CANCEL": "cancel_action",
            "META_CORRECT": "update_context",
            "DIAGNOSE_ERROR": "diagnose_error",
        }
        return intent_to_tool.get(intent, "search_databases")
    
    def generate_data_search(self, count: int) -> List[GeneratedConversation]:
        """Generate data search conversations."""
        conversations = []
        max_attempts = count * 20  # Retry limit to prevent infinite loops
        attempts = 0
        for i, template in enumerate(itertools.cycle(DATA_SEARCH_TEMPLATES)):
            if len(conversations) >= count or attempts >= max_attempts:
                break
            attempts += 1
            
            query, entities = self._fill_template(template)
            query_hash = self._hash_query(query)
            
            if query_hash in self.generated_hashes:
                continue
            self.generated_hashes.add(query_hash)
            
            difficulty = "easy" if len(query.split()) < 6 else "medium" if len(query.split()) < 12 else "hard"
            
            conv = GeneratedConversation(
                id=f"DS-{len(conversations)+2000:04d}",
                name=f"Data search: {query[:40]}",
                category="data_discovery",
                difficulty=difficulty,
                turns=[{
                    "query": query,
                    "expected_intent": "DATA_SEARCH",
                    "expected_entities": entities,
                    "expected_tool": "search_databases",
                }],
                tags=["generated", "data_search"],
            )
            conversations.append(conv)
        
        return conversations
    
    def generate_data_download(self, count: int) -> List[GeneratedConversation]:
        """Generate data download conversations."""
        conversations = []
        max_attempts = count * 20
        attempts = 0
        for template in itertools.cycle(DATA_DOWNLOAD_TEMPLATES):
            if len(conversations) >= count or attempts >= max_attempts:
                break
            attempts += 1
            
            query, entities = self._fill_template(template)
            query_hash = self._hash_query(query)
            
            if query_hash in self.generated_hashes:
                continue
            self.generated_hashes.add(query_hash)
            
            conv = GeneratedConversation(
                id=f"DD-{len(conversations)+2000:04d}",
                name=f"Data download: {query[:40]}",
                category="data_discovery",
                difficulty="easy" if len(query.split()) < 5 else "medium",
                turns=[{
                    "query": query,
                    "expected_intent": "DATA_DOWNLOAD",
                    "expected_entities": entities,
                    "expected_tool": "download_data",
                }],
                tags=["generated", "data_download"],
            )
            conversations.append(conv)
        
        return conversations
    
    def generate_workflow(self, count: int) -> List[GeneratedConversation]:
        """Generate workflow creation conversations."""
        conversations = []
        max_attempts = count * 20
        attempts = 0
        for template in itertools.cycle(WORKFLOW_CREATE_TEMPLATES):
            if len(conversations) >= count or attempts >= max_attempts:
                break
            attempts += 1
            
            query, entities = self._fill_template(template)
            query_hash = self._hash_query(query)
            
            if query_hash in self.generated_hashes:
                continue
            self.generated_hashes.add(query_hash)
            
            difficulty = "easy" if len(query.split()) < 5 else "medium" if len(query.split()) < 12 else "hard"
            
            conv = GeneratedConversation(
                id=f"WC-{len(conversations)+2000:04d}",
                name=f"Workflow: {query[:40]}",
                category="workflow_generation",
                difficulty=difficulty,
                turns=[{
                    "query": query,
                    "expected_intent": "WORKFLOW_CREATE",
                    "expected_entities": entities,
                    "expected_tool": "generate_workflow",
                }],
                tags=["generated", "workflow"],
            )
            conversations.append(conv)
        
        return conversations
    
    def generate_job_management(self, count: int) -> List[GeneratedConversation]:
        """Generate job management conversations."""
        conversations = []
        
        templates = [
            (JOB_SUBMIT_TEMPLATES, "JOB_SUBMIT", "submit_job"),
            (JOB_STATUS_TEMPLATES, "JOB_STATUS", "check_job_status"),
            (JOB_LIST_TEMPLATES, "JOB_LIST", "list_jobs"),
            (JOB_LOGS_TEMPLATES, "JOB_LOGS", "show_logs"),
            (JOB_CANCEL_TEMPLATES, "JOB_CANCEL", "cancel_job"),
        ]
        
        per_type = count // len(templates)
        
        for template_list, intent, tool in templates:
            type_count = 0
            max_attempts = per_type * 20
            attempts = 0
            for template in itertools.cycle(template_list):
                if type_count >= per_type or attempts >= max_attempts:
                    break
                attempts += 1
                
                query, entities = self._fill_template(template)
                query_hash = self._hash_query(query)
                
                if query_hash in self.generated_hashes:
                    continue
                self.generated_hashes.add(query_hash)
                
                conv = GeneratedConversation(
                    id=f"JM-{len(conversations)+2000:04d}",
                    name=f"Job: {query[:40]}",
                    category="job_management",
                    difficulty="easy" if len(query.split()) < 5 else "medium",
                    turns=[{
                        "query": query,
                        "expected_intent": intent,
                        "expected_entities": entities,
                        "expected_tool": tool,
                    }],
                    tags=["generated", "job_management"],
                )
                conversations.append(conv)
                type_count += 1
        
        return conversations
    
    def generate_education(self, count: int) -> List[GeneratedConversation]:
        """Generate education conversations."""
        conversations = []
        max_attempts = count * 20
        attempts = 0
        
        for template in itertools.cycle(EDUCATION_EXPLAIN_TEMPLATES):
            if len(conversations) >= count * 0.8 or attempts >= max_attempts:
                break
            attempts += 1
            
            query, entities = self._fill_template(template)
            query_hash = self._hash_query(query)
            
            if query_hash in self.generated_hashes:
                continue
            self.generated_hashes.add(query_hash)
            
            conv = GeneratedConversation(
                id=f"ED-{len(conversations)+2000:04d}",
                name=f"Education: {query[:40]}",
                category="education",
                difficulty="easy",
                turns=[{
                    "query": query,
                    "expected_intent": "EDUCATION_EXPLAIN",
                    "expected_entities": entities,
                    "expected_tool": "explain_concept",
                }],
                tags=["generated", "education"],
            )
            conversations.append(conv)
        
        attempts = 0
        for template in itertools.cycle(EDUCATION_HELP_TEMPLATES):
            if len(conversations) >= count or attempts >= max_attempts:
                break
            attempts += 1
            
            query, _ = self._fill_template(template)
            query_hash = self._hash_query(query)
            
            if query_hash in self.generated_hashes:
                continue
            self.generated_hashes.add(query_hash)
            
            conv = GeneratedConversation(
                id=f"EH-{len(conversations)+2000:04d}",
                name=f"Help: {query[:40]}",
                category="education",
                difficulty="easy",
                turns=[{
                    "query": query,
                    "expected_intent": "EDUCATION_HELP",
                    "expected_entities": {},
                    "expected_tool": "show_help",
                }],
                tags=["generated", "help"],
            )
            conversations.append(conv)
        
        return conversations
    
    def generate_negation(self, count: int) -> List[GeneratedConversation]:
        """Generate negation and preference conversations."""
        conversations = []
        max_attempts = count * 20
        attempts = 0
        
        for template, intent in itertools.cycle(NEGATION_TEMPLATES):
            if len(conversations) >= count or attempts >= max_attempts:
                break
            attempts += 1
            
            query, entities = self._fill_template(template)
            query_hash = self._hash_query(query)
            
            if query_hash in self.generated_hashes:
                continue
            self.generated_hashes.add(query_hash)
            
            conv = GeneratedConversation(
                id=f"NG-{len(conversations)+2000:04d}",
                name=f"Negation: {query[:40]}",
                category="negation",
                difficulty="hard",
                turns=[{
                    "query": query,
                    "expected_intent": intent,
                    "expected_entities": entities,
                    "expected_tool": self._get_expected_tool(intent),
                }],
                tags=["generated", "negation", "preference"],
            )
            conversations.append(conv)
        
        return conversations
    
    def generate_ambiguous(self, count: int) -> List[GeneratedConversation]:
        """Generate ambiguous conversations."""
        conversations = []
        max_attempts = count * 20
        attempts = 0
        
        for template, intent in itertools.cycle(AMBIGUOUS_TEMPLATES):
            if len(conversations) >= count or attempts >= max_attempts:
                break
            attempts += 1
            
            query, entities = self._fill_template(template)
            query_hash = self._hash_query(query)
            
            if query_hash in self.generated_hashes:
                continue
            self.generated_hashes.add(query_hash)
            
            conv = GeneratedConversation(
                id=f"AM-{len(conversations)+2000:04d}",
                name=f"Ambiguous: {query[:40]}",
                category="ambiguous",
                difficulty="hard",
                turns=[{
                    "query": query,
                    "expected_intent": intent,
                    "expected_entities": entities,
                    "expected_tool": self._get_expected_tool(intent),
                }],
                tags=["generated", "ambiguous"],
            )
            conversations.append(conv)
        
        return conversations
    
    def generate_edge_cases(self, count: int) -> List[GeneratedConversation]:
        """Generate edge case conversations."""
        conversations = []
        max_attempts = count * 20
        attempts = 0
        
        for template, intent in itertools.cycle(EDGE_CASE_TEMPLATES):
            if len(conversations) >= count or attempts >= max_attempts:
                break
            attempts += 1
            
            query, entities = self._fill_template(template)
            query_hash = self._hash_query(query)
            
            if query_hash in self.generated_hashes:
                continue
            self.generated_hashes.add(query_hash)
            
            # Determine difficulty
            if len(query) > 100:
                difficulty = "hard"
            elif any(c in query for c in "!?&"):
                difficulty = "medium"
            else:
                difficulty = "easy"
            
            conv = GeneratedConversation(
                id=f"EC-{len(conversations)+2000:04d}",
                name=f"Edge case: {query[:40]}",
                category="edge_cases",
                difficulty=difficulty,
                turns=[{
                    "query": query,
                    "expected_intent": intent,
                    "expected_entities": entities,
                    "expected_tool": self._get_expected_tool(intent),
                }],
                tags=["generated", "edge_case"],
            )
            conversations.append(conv)
        
        return conversations
    
    def generate_adversarial(self, count: int) -> List[GeneratedConversation]:
        """Generate adversarial conversations."""
        conversations = []
        max_attempts = count * 20
        attempts = 0
        
        for template, intent in itertools.cycle(ADVERSARIAL_TEMPLATES):
            if len(conversations) >= count or attempts >= max_attempts:
                break
            attempts += 1
            
            query, entities = self._fill_template(template)
            query_hash = self._hash_query(query)
            
            if query_hash in self.generated_hashes:
                continue
            self.generated_hashes.add(query_hash)
            
            conv = GeneratedConversation(
                id=f"AV-{len(conversations)+2000:04d}",
                name=f"Adversarial: {query[:40]}",
                category="adversarial",
                difficulty="adversarial",
                turns=[{
                    "query": query,
                    "expected_intent": intent,
                    "expected_entities": entities,
                    "expected_tool": self._get_expected_tool(intent),
                }],
                tags=["generated", "adversarial"],
            )
            conversations.append(conv)
        
        return conversations
    
    def generate_multi_turn(self, count: int) -> List[GeneratedConversation]:
        """Generate multi-turn conversations."""
        conversations = []
        max_attempts = count * 20
        attempts = 0
        
        for scenario in itertools.cycle(MULTI_TURN_SCENARIOS):
            if len(conversations) >= count or attempts >= max_attempts:
                break
            attempts += 1
            
            turns = []
            for template, intent in scenario:
                query, entities = self._fill_template(template)
                turns.append({
                    "query": query,
                    "expected_intent": intent,
                    "expected_entities": entities,
                    "expected_tool": self._get_expected_tool(intent),
                })
            
            conv = GeneratedConversation(
                id=f"MT-{len(conversations)+2000:04d}",
                name=f"Multi-turn: {turns[0]['query'][:30]}",
                category="multi_turn",
                difficulty="hard",
                turns=turns,
                tags=["generated", "multi_turn", "context"],
            )
            conversations.append(conv)
        
        return conversations
    
    def generate_data_scan(self, count: int) -> List[GeneratedConversation]:
        """Generate data scan conversations."""
        conversations = []
        max_attempts = count * 20
        attempts = 0
        
        for template in itertools.cycle(DATA_SCAN_TEMPLATES):
            if len(conversations) >= count or attempts >= max_attempts:
                break
            attempts += 1
            
            query, entities = self._fill_template(template)
            query_hash = self._hash_query(query)
            
            if query_hash in self.generated_hashes:
                continue
            self.generated_hashes.add(query_hash)
            
            conv = GeneratedConversation(
                id=f"SC-{len(conversations)+2000:04d}",
                name=f"Scan: {query[:40]}",
                category="data_discovery",
                difficulty="easy",
                turns=[{
                    "query": query,
                    "expected_intent": "DATA_SCAN",
                    "expected_entities": entities,
                    "expected_tool": "scan_local_data",
                }],
                tags=["generated", "data_scan"],
            )
            conversations.append(conv)
        
        return conversations
    
    def generate_all(self, target_total: int = 5000) -> List[GeneratedConversation]:
        """Generate comprehensive test set with target total."""
        
        # Distribution based on importance and difficulty
        distribution = {
            "data_search": 0.25,      # 25% - most common
            "workflow": 0.20,          # 20% - important
            "data_download": 0.10,     # 10%
            "job_management": 0.15,    # 15%
            "education": 0.08,         # 8%
            "negation": 0.08,          # 8% - important for robustness
            "edge_cases": 0.06,        # 6%
            "multi_turn": 0.04,        # 4%
            "adversarial": 0.03,       # 3%
            "ambiguous": 0.02,         # 2%
            "data_scan": 0.02,         # 2%
        }
        
        all_conversations = []
        
        print(f"Generating {target_total} conversations...")
        
        for category, ratio in distribution.items():
            count = int(target_total * ratio)
            print(f"  {category}: {count}")
            
            if category == "data_search":
                convs = self.generate_data_search(count)
            elif category == "workflow":
                convs = self.generate_workflow(count)
            elif category == "data_download":
                convs = self.generate_data_download(count)
            elif category == "job_management":
                convs = self.generate_job_management(count)
            elif category == "education":
                convs = self.generate_education(count)
            elif category == "negation":
                convs = self.generate_negation(count)
            elif category == "edge_cases":
                convs = self.generate_edge_cases(count)
            elif category == "multi_turn":
                convs = self.generate_multi_turn(count)
            elif category == "adversarial":
                convs = self.generate_adversarial(count)
            elif category == "ambiguous":
                convs = self.generate_ambiguous(count)
            elif category == "data_scan":
                convs = self.generate_data_scan(count)
            
            all_conversations.extend(convs)
            print(f"    Generated {len(convs)}")
        
        print(f"\nTotal generated: {len(all_conversations)}")
        return all_conversations


def populate_database(db_path: str, conversations: List[GeneratedConversation]):
    """Add conversations to database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    added = 0
    skipped = 0
    
    for conv in conversations:
        # Check if already exists
        cursor.execute("SELECT id FROM conversations WHERE id = ?", (conv.id,))
        if cursor.fetchone():
            skipped += 1
            continue
        
        # Check for duplicate by hash
        turns_json = json.dumps(conv.turns)
        hash_val = hashlib.md5(turns_json.encode()).hexdigest()
        
        cursor.execute("SELECT id FROM conversations WHERE hash = ?", (hash_val,))
        if cursor.fetchone():
            skipped += 1
            continue
        
        cursor.execute("""
            INSERT INTO conversations (id, name, category, difficulty, source, turns_json, created_at, tags, description, hash)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'), ?, ?, ?)
        """, (
            conv.id,
            conv.name,
            conv.category,
            conv.difficulty,
            conv.source,
            turns_json,
            json.dumps(conv.tags) if conv.tags else "[]",
            conv.description,
            hash_val,
        ))
        added += 1
    
    conn.commit()
    conn.close()
    
    print(f"Added {added} conversations, skipped {skipped} duplicates")
    return added


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate comprehensive test conversations")
    parser.add_argument("--count", type=int, default=5000, help="Target number of conversations")
    parser.add_argument("--db", type=str, default="tests/evaluation/evaluation.db", help="Database path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    generator = ComprehensiveGenerator(seed=args.seed)
    conversations = generator.generate_all(target_total=args.count)
    
    added = populate_database(args.db, conversations)
    
    # Print statistics
    conn = sqlite3.connect(args.db)
    cursor = conn.execute("SELECT COUNT(*) FROM conversations")
    total = cursor.fetchone()[0]
    
    cursor = conn.execute("SELECT category, COUNT(*) FROM conversations GROUP BY category ORDER BY COUNT(*) DESC")
    print(f"\nDatabase now has {total} conversations:")
    for row in cursor:
        print(f"  {row[0]}: {row[1]}")
    
    conn.close()
