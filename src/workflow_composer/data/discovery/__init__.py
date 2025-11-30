"""
Data Discovery Module
=====================

LLM-powered data discovery for genomics databases.

This module provides intelligent search across multiple data sources:
- ENCODE Portal (ChIP-seq, ATAC-seq, RNA-seq, etc.)
- NCBI GEO/SRA (All experiment types)
- NCI GDC/TCGA (Cancer genomics - methylation, expression, mutations)
- Ensembl (Reference genomes, annotations)

Features:
- Natural language query parsing
- Multi-source federated search
- Relevance ranking
- Download management

Quick Start:
    from workflow_composer.data.discovery import DataDiscovery
    
    # Initialize discovery
    discovery = DataDiscovery()
    
    # Search with natural language
    results = discovery.search("human liver ChIP-seq H3K27ac data")
    
    # Browse results
    for dataset in results.datasets:
        print(f"{dataset.source.value}: {dataset.id} - {dataset.title}")
        print(f"  Organism: {dataset.organism}")
        print(f"  Assay: {dataset.assay_type}")
        print(f"  URL: {dataset.web_url}")
        print()
    
    # Search TCGA cancer data
    results = discovery.search("brain cancer methylation", sources=["gdc"])
    
    # Download a dataset
    discovery.download(results.datasets[0], output_dir="/data/downloads")

Advanced Usage:
    # Search specific sources
    results = discovery.search("mouse RNA-seq", sources=["geo"])
    
    # Use structured query
    from workflow_composer.data.discovery import SearchQuery, DataSource
    
    query = SearchQuery(
        organism="human",
        assay_type="ChIP-seq",
        target="H3K27ac",
        tissue="liver",
        source=DataSource.ENCODE
    )
    results = discovery.search(query)
    
    # Get dataset details
    dataset = discovery.get_dataset("ENCSR000ABC")
    print(dataset.download_urls)
    
    # Search for reference data
    refs = discovery.search("human genome annotation", sources=["ensembl"])

See Also:
    - docs/DATA_DISCOVERY_DESIGN.md for architecture details
    - docs/pipelines/ for how to use discovered data in workflows
"""

from .models import (
    # Enums
    DataSource,
    AssayType,
    FileType,
    DownloadMethod,
    
    # Data classes
    SearchQuery,
    DownloadURL,
    DatasetInfo,
    SearchResults,
    LocalReference,
    DownloadJob,
)

from .query_parser import (
    QueryParser,
    ParseResult,
    parse_query,
)

from .orchestrator import (
    DataDiscovery,
    quick_search,
    search_encode,
    search_geo,
    search_references,
)

from .adapters import (
    BaseAdapter,
    ENCODEAdapter,
    GEOAdapter,
    GDCAdapter,
    EnsemblAdapter,
    get_adapter,
    list_available_sources,
    
    # Convenience functions
    search_encode as encode_search,
    search_geo as geo_search,
    search_gdc as gdc_search,
    get_human_genome_url,
    get_human_gtf_url,
    get_mouse_genome_url,
    get_mouse_gtf_url,
)

from .parallel import (
    ParallelSearchOrchestrator,
    FederatedSearchResult,
    parallel_search,
)

__all__ = [
    # Main orchestrator
    "DataDiscovery",
    
    # Query parsing
    "QueryParser",
    "ParseResult",
    "parse_query",
    
    # Data models
    "DataSource",
    "AssayType",
    "FileType",
    "DownloadMethod",
    "SearchQuery",
    "DownloadURL",
    "DatasetInfo",
    "SearchResults",
    "LocalReference",
    "DownloadJob",
    
    # Adapters
    "BaseAdapter",
    "ENCODEAdapter",
    "GEOAdapter",
    "GDCAdapter",
    "EnsemblAdapter",
    "get_adapter",
    "list_available_sources",
    
    # Parallel search
    "ParallelSearchOrchestrator",
    "FederatedSearchResult",
    "parallel_search",
    
    # Convenience functions
    "quick_search",
    "search_encode",
    "search_geo",
    "search_references",
    "encode_search",
    "geo_search",
    "gdc_search",
    "get_human_genome_url",
    "get_human_gtf_url",
    "get_mouse_genome_url",
    "get_mouse_gtf_url",
]
