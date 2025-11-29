"""
LLM Query Parser
================

Parse natural language queries into structured SearchQuery objects using LLM.

This module provides intelligent query parsing that can understand:
- Complex natural language queries about genomics data
- Implicit information (e.g., "liver cells" implies tissue=liver)
- Domain-specific terminology (H3K27ac, ATAC-seq, etc.)
- Suggest appropriate data sources based on query

Example:
    parser = QueryParser()
    query = parser.parse("Find human H3K27ac ChIP-seq data from liver tissue")
    
    # Returns:
    # SearchQuery(
    #     organism="human",
    #     assay_type="ChIP-seq",
    #     target="H3K27ac",
    #     tissue="liver",
    #     source=DataSource.ENCODE
    # )
"""

import logging
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from .models import SearchQuery, DataSource, AssayType

logger = logging.getLogger(__name__)

# Default prompts for query parsing
SYSTEM_PROMPT = """You are a bioinformatics data discovery assistant. Your task is to parse natural language queries about genomics data into structured search parameters.

Given a user query, extract the following fields:
- organism: Species name (human, mouse, rat, zebrafish, fly, worm, yeast)
- assembly: Genome assembly version (GRCh38, GRCm39, etc.)
- assay_type: Experiment type (RNA-seq, ChIP-seq, ATAC-seq, scRNA-seq, Hi-C, WGBS, WGS, etc.)
- target: For ChIP-seq/CUT&RUN/CUT&Tag - the target protein or histone modification (H3K27ac, CTCF, H3K4me3, etc.)
- tissue: Tissue or organ (liver, brain, heart, etc.)
- cell_line: Cell line name (K562, HeLa, GM12878, etc.)
- cell_type: Cell type (T cell, B cell, neuron, etc.)
- treatment: Any treatment or condition
- keywords: Additional search terms
- source: Preferred data source (encode, geo, ensembl)

Respond with a JSON object containing these fields. Use null for fields not mentioned in the query.

Important notes:
- ENCODE is best for ChIP-seq, ATAC-seq, DNase-seq data
- GEO is best for RNA-seq, single-cell data, and diverse experiment types
- Ensembl is for reference genomes and annotations (not experiments)
- If no source is specified, suggest the most appropriate one based on the assay type
"""

USER_PROMPT_TEMPLATE = """Parse this query into structured search parameters:

Query: {query}

Return a JSON object with the extracted fields."""


@dataclass
class ParseResult:
    """Result of query parsing."""
    query: SearchQuery
    confidence: float
    suggested_sources: List[DataSource]
    clarification_needed: bool = False
    clarification_message: str = ""


class QueryParser:
    """
    LLM-powered query parser for natural language data discovery queries.
    
    Usage:
        parser = QueryParser(llm_client=my_llm_client)
        
        # Parse a natural language query
        result = parser.parse("human liver RNA-seq data")
        
        # Get the structured query
        query = result.query
        print(f"Organism: {query.organism}")
        print(f"Tissue: {query.tissue}")
        print(f"Assay: {query.assay_type}")
    """
    
    # Map assay types to best data sources
    ASSAY_SOURCE_MAP = {
        "ChIP-seq": [DataSource.ENCODE, DataSource.GEO],
        "ATAC-seq": [DataSource.ENCODE, DataSource.GEO],
        "DNase-seq": [DataSource.ENCODE],
        "RNA-seq": [DataSource.GEO, DataSource.ENCODE],
        "scRNA-seq": [DataSource.GEO],
        "scATAC-seq": [DataSource.GEO, DataSource.ENCODE],
        "Hi-C": [DataSource.ENCODE, DataSource.GEO],
        "WGBS": [DataSource.ENCODE, DataSource.GEO],
        "WGS": [DataSource.GEO, DataSource.SRA],
    }
    
    # Common aliases
    ORGANISM_ALIASES = {
        "human": "human",
        "homo sapiens": "human",
        "hs": "human",
        "mouse": "mouse",
        "mus musculus": "mouse",
        "mm": "mouse",
        "rat": "rat",
        "rattus norvegicus": "rat",
        "zebrafish": "zebrafish",
        "danio rerio": "zebrafish",
        "fly": "fly",
        "drosophila": "fly",
        "drosophila melanogaster": "fly",
        "worm": "worm",
        "c. elegans": "worm",
        "caenorhabditis elegans": "worm",
        "yeast": "yeast",
        "saccharomyces cerevisiae": "yeast",
    }
    
    ASSAY_ALIASES = {
        "rnaseq": "RNA-seq",
        "rna-seq": "RNA-seq",
        "rna seq": "RNA-seq",
        "chipseq": "ChIP-seq",
        "chip-seq": "ChIP-seq",
        "chip seq": "ChIP-seq",
        "atacseq": "ATAC-seq",
        "atac-seq": "ATAC-seq",
        "atac seq": "ATAC-seq",
        "single cell": "scRNA-seq",
        "single-cell": "scRNA-seq",
        "scrnaseq": "scRNA-seq",
        "scrna-seq": "scRNA-seq",
        "10x": "scRNA-seq",
        "10x genomics": "scRNA-seq",
        "hic": "Hi-C",
        "hi-c": "Hi-C",
        "wgbs": "WGBS",
        "bisulfite": "WGBS",
        "methylation": "WGBS",
        "whole genome": "WGS",
        "wgs": "WGS",
        "gene expression": "RNA-seq",
        "expression": "RNA-seq",
        "transcriptome": "RNA-seq",
        "transcriptomic": "RNA-seq",
        "mrna": "RNA-seq",
    }
    
    def __init__(self, llm_client=None):
        """
        Initialize the query parser.
        
        Args:
            llm_client: Optional LLM client for parsing. If not provided,
                       uses rule-based parsing as fallback.
        """
        self.llm_client = llm_client
    
    def parse(self, query: str) -> ParseResult:
        """
        Parse a natural language query into structured SearchQuery.
        
        Args:
            query: Natural language query
            
        Returns:
            ParseResult with structured query and metadata
        """
        logger.info(f"Parsing query: {query}")
        
        # Try LLM parsing first if available
        if self.llm_client:
            try:
                return self._parse_with_llm(query)
            except Exception as e:
                logger.warning(f"LLM parsing failed, falling back to rules: {e}")
        
        # Fall back to rule-based parsing
        return self._parse_with_rules(query)
    
    def _parse_with_llm(self, query: str) -> ParseResult:
        """Parse using LLM."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(query=query)},
        ]
        
        # Call LLM
        response = self.llm_client.chat(messages)
        
        # Parse JSON response
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("Could not parse LLM response as JSON")
        
        # Convert to SearchQuery
        search_query = self._dict_to_query(data, query)
        
        # Suggest sources
        suggested_sources = self._suggest_sources(search_query)
        
        return ParseResult(
            query=search_query,
            confidence=0.9,  # High confidence with LLM
            suggested_sources=suggested_sources,
        )
    
    def _parse_with_rules(self, query: str) -> ParseResult:
        """Parse using rule-based approach."""
        query_lower = query.lower()
        
        # Extract organism
        organism = None
        for alias, canonical in self.ORGANISM_ALIASES.items():
            if alias in query_lower:
                organism = canonical
                break
        
        # Extract assay type
        assay_type = None
        for alias, canonical in self.ASSAY_ALIASES.items():
            if alias in query_lower:
                assay_type = canonical
                break
        
        # Extract common ChIP targets
        target = None
        chip_targets = [
            "H3K27ac", "H3K4me3", "H3K4me1", "H3K27me3", "H3K36me3",
            "H3K9me3", "H3K9ac", "CTCF", "p300", "Pol2", "RNAPII",
        ]
        for t in chip_targets:
            if t.lower() in query_lower:
                target = t
                break
        
        # Extract tissue (simple keyword matching)
        tissue = None
        tissues = [
            "liver", "brain", "heart", "kidney", "lung", "muscle",
            "spleen", "intestine", "skin", "blood", "bone marrow",
        ]
        for t in tissues:
            if t in query_lower:
                tissue = t
                break
        
        # Extract cell lines
        cell_line = None
        cell_lines = ["k562", "hela", "gm12878", "hepg2", "a549", "mcf7"]
        for cl in cell_lines:
            if cl in query_lower:
                cell_line = cl.upper() if len(cl) <= 5 else cl.title()
                break
        
        # Extract keywords (remaining significant words)
        stop_words = {
            "find", "get", "download", "search", "data", "from", "in",
            "the", "a", "an", "and", "or", "for", "with", "to", "of",
        }
        keywords = [
            word for word in query_lower.split()
            if word not in stop_words
            and word not in self.ORGANISM_ALIASES
            and word not in self.ASSAY_ALIASES
            and len(word) > 2
        ]
        
        # Create SearchQuery
        search_query = SearchQuery(
            raw_query=query,
            organism=organism,
            assay_type=assay_type,
            target=target,
            tissue=tissue,
            cell_line=cell_line,
            keywords=keywords[:5],  # Limit keywords
        )
        
        # Suggest sources
        suggested_sources = self._suggest_sources(search_query)
        if suggested_sources:
            search_query.source = suggested_sources[0]
        
        # Determine confidence
        confidence = 0.5
        if organism:
            confidence += 0.1
        if assay_type:
            confidence += 0.2
        if tissue or cell_line:
            confidence += 0.1
        
        # Check if clarification needed
        clarification_needed = False
        clarification_message = ""
        if not organism:
            clarification_needed = True
            clarification_message = "Please specify an organism (human, mouse, etc.)"
        elif not assay_type:
            clarification_needed = True
            clarification_message = "What type of data are you looking for? (RNA-seq, ChIP-seq, etc.)"
        
        return ParseResult(
            query=search_query,
            confidence=confidence,
            suggested_sources=suggested_sources,
            clarification_needed=clarification_needed,
            clarification_message=clarification_message,
        )
    
    def _dict_to_query(self, data: Dict[str, Any], raw_query: str) -> SearchQuery:
        """Convert parsed dictionary to SearchQuery."""
        # Normalize source
        source = None
        if data.get("source"):
            try:
                source = DataSource(data["source"].lower())
            except ValueError:
                pass
        
        return SearchQuery(
            raw_query=raw_query,
            organism=data.get("organism"),
            assembly=data.get("assembly"),
            assay_type=data.get("assay_type"),
            target=data.get("target"),
            tissue=data.get("tissue"),
            cell_line=data.get("cell_line"),
            cell_type=data.get("cell_type"),
            treatment=data.get("treatment"),
            source=source,
            keywords=data.get("keywords", []),
        )
    
    def _suggest_sources(self, query: SearchQuery) -> List[DataSource]:
        """Suggest appropriate data sources for a query."""
        sources = []
        
        # Check if assay type maps to specific sources
        if query.assay_type:
            assay_sources = self.ASSAY_SOURCE_MAP.get(query.assay_type, [])
            sources.extend(assay_sources)
        
        # Check for reference data keywords
        if query.keywords:
            ref_keywords = {"genome", "reference", "annotation", "gtf", "fasta", "index"}
            if ref_keywords.intersection(set(query.keywords)):
                if DataSource.ENSEMBL not in sources:
                    sources.insert(0, DataSource.ENSEMBL)
        
        # Default sources if none matched
        if not sources:
            sources = [DataSource.GEO, DataSource.ENCODE]
        
        return sources


# Convenience function
def parse_query(query: str, llm_client=None) -> SearchQuery:
    """
    Parse a natural language query into structured SearchQuery.
    
    Args:
        query: Natural language query
        llm_client: Optional LLM client for enhanced parsing
        
    Returns:
        Structured SearchQuery
    """
    parser = QueryParser(llm_client)
    result = parser.parse(query)
    return result.query
