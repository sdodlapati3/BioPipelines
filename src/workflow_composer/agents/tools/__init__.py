"""
BioPipelines Agent Tools
========================

Modular tool system for the BioPipelines AI agent.
All tools are organized by category and unified through the AgentTools class.

Usage:
    from tools import get_agent_tools
    tools = get_agent_tools()
    result = tools.execute_tool("scan_data", path="/data")
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Import base types
from .base import ToolResult, ToolName, TOOL_PATTERNS

# Import registry
from .registry import ToolRegistry, get_registry

# Import tool implementations and patterns
from .data_discovery import (
    SCAN_DATA_PATTERNS,
    SEARCH_DATABASES_PATTERNS,
    SEARCH_TCGA_PATTERNS,
    DESCRIBE_FILES_PATTERNS,
    VALIDATE_DATASET_PATTERNS,
    GET_DATASET_DETAILS_PATTERNS,
    scan_data_impl,
    search_databases_impl,
    search_tcga_impl,
    describe_files_impl,
    validate_dataset_impl,
    get_dataset_details_impl,
)

from .data_management import (
    DOWNLOAD_DATASET_PATTERNS,
    DOWNLOAD_REFERENCE_PATTERNS,
    BUILD_INDEX_PATTERNS,
    CLEANUP_DATA_PATTERNS,
    CONFIRM_CLEANUP_PATTERNS,
    download_dataset_impl,
    download_reference_impl,
    build_index_impl,
    cleanup_data_impl,
    confirm_cleanup_impl,
)

from .workflow import (
    GENERATE_WORKFLOW_PATTERNS,
    LIST_WORKFLOWS_PATTERNS,
    CHECK_REFERENCES_PATTERNS,
    VISUALIZE_WORKFLOW_PATTERNS,
    generate_workflow_impl,
    list_workflows_impl,
    check_references_impl,
    visualize_workflow_impl,
)

from .execution import (
    SUBMIT_JOB_PATTERNS,
    GET_JOB_STATUS_PATTERNS,
    GET_LOGS_PATTERNS,
    CANCEL_JOB_PATTERNS,
    CHECK_SYSTEM_HEALTH_PATTERNS,
    RESTART_VLLM_PATTERNS,
    RESUBMIT_JOB_PATTERNS,
    WATCH_JOB_PATTERNS,
    LIST_JOBS_PATTERNS,
    submit_job_impl,
    get_job_status_impl,
    get_logs_impl,
    cancel_job_impl,
    check_system_health_impl,
    restart_vllm_impl,
    resubmit_job_impl,
    watch_job_impl,
    list_jobs_impl,
)

from .diagnostics import (
    DIAGNOSE_ERROR_PATTERNS,
    ANALYZE_RESULTS_PATTERNS,
    RECOVER_ERROR_PATTERNS,
    diagnose_error_impl,
    analyze_results_impl,
    recover_error_impl,
)

from .education import (
    EXPLAIN_CONCEPT_PATTERNS,
    COMPARE_SAMPLES_PATTERNS,
    GET_HELP_PATTERNS,
    explain_concept_impl,
    compare_samples_impl,
    get_help_impl,
    CONCEPT_KNOWLEDGE,
)

# Database Clients (Phase 2)
from .databases import (
    get_uniprot_client,
    get_string_client,
    get_kegg_client,
    get_reactome_client,
    get_pubmed_client,
    get_clinvar_client,
    DatabaseResult,
)


# =============================================================================
# DATABASE TOOL PATTERNS (Phase 2)
# =============================================================================

# UniProt patterns
SEARCH_UNIPROT_PATTERNS = [
    r"search\s+(?:uniprot|protein\s+database)\s+(?:for\s+)?(.+)",
    r"find\s+protein\s+(?:for\s+|named\s+)?(.+)\s+(?:in\s+)?uniprot",
    r"look\s+up\s+(.+)\s+(?:in\s+)?uniprot",
    r"get\s+(?:protein\s+)?info(?:rmation)?\s+(?:for|about)\s+(.+)",
]

GET_PROTEIN_PATTERNS = [
    r"get\s+protein\s+(.+)",
    r"fetch\s+protein\s+(.+)",
    r"uniprot\s+(?:id\s+)?([A-Z0-9_]+)",
]

# STRING patterns
SEARCH_STRING_PATTERNS = [
    r"search\s+string(?:\s+db)?\s+(?:for\s+)?(.+)",
    r"find\s+interactions?\s+(?:for\s+)?(.+)\s+in\s+string",
    r"protein\s+interactions?\s+(?:for\s+)?(.+)",
]

GET_INTERACTIONS_PATTERNS = [
    r"get\s+(?:protein\s+)?interactions?\s+(?:for\s+)?(.+)",
    r"what\s+(?:proteins?\s+)?interacts?\s+with\s+(.+)",
    r"interaction\s+partners?\s+(?:for|of)\s+(.+)",
]

GET_ENRICHMENT_PATTERNS = [
    r"(?:get\s+)?enrichment\s+(?:analysis\s+)?(?:for\s+)?(.+)",
    r"pathway\s+enrichment\s+(?:for\s+)?(.+)",
    r"go\s+enrichment\s+(?:for\s+)?(.+)",
]

# KEGG patterns
SEARCH_KEGG_PATTERNS = [
    r"search\s+kegg\s+(?:for\s+)?(.+)",
    r"find\s+(?:kegg\s+)?pathway\s+(?:for\s+)?(.+)",
    r"kegg\s+pathway\s+(.+)",
]

GET_PATHWAY_PATTERNS = [
    r"get\s+pathway\s+(.+)",
    r"show\s+pathway\s+(.+)",
    r"pathway\s+(?:info|details?)\s+(?:for\s+)?(.+)",
]

# Reactome patterns
SEARCH_REACTOME_PATTERNS = [
    r"search\s+reactome\s+(?:for\s+)?(.+)",
    r"find\s+(?:in\s+)?reactome\s+(.+)",
    r"reactome\s+pathway\s+(.+)",
]

ANALYZE_GENES_PATTERNS = [
    r"analyze\s+genes?\s+(.+)",
    r"gene\s+analysis\s+(?:for\s+)?(.+)",
    r"pathway\s+analysis\s+(?:for\s+)?(.+)",
]

# PubMed patterns
SEARCH_PUBMED_PATTERNS = [
    r"search\s+pubmed\s+(?:for\s+)?(.+)",
    r"find\s+(?:papers?|articles?|publications?)\s+(?:about|on|for)\s+(.+)",
    r"pubmed\s+search\s+(.+)",
    r"literature\s+(?:search|review)\s+(?:for|on)\s+(.+)",
]

GET_ARTICLE_PATTERNS = [
    r"get\s+(?:pubmed\s+)?article\s+(.+)",
    r"fetch\s+(?:pubmed\s+)?paper\s+(\d+)",
    r"pmid\s+(\d+)",
]

# ClinVar patterns
SEARCH_CLINVAR_PATTERNS = [
    r"search\s+clinvar\s+(?:for\s+)?(.+)",
    r"find\s+(?:clinical\s+)?variants?\s+(?:for|in)\s+(.+)",
    r"clinvar\s+(?:search\s+)?(.+)",
    r"pathogenic\s+variants?\s+(?:for|in)\s+(.+)",
]

GET_VARIANTS_PATTERNS = [
    r"get\s+variants?\s+(?:for|in)\s+(.+)",
    r"show\s+variants?\s+(?:for|in)\s+(.+)",
    r"variant\s+info\s+(.+)",
]


# =============================================================================
# UNIFIED TOOL PATTERNS
# =============================================================================

# Combined pattern list for regex-based tool detection
# Order matters - more specific patterns should come first
ALL_TOOL_PATTERNS = [
    # Data Discovery
    (ToolName.SCAN_DATA, SCAN_DATA_PATTERNS),
    (ToolName.SEARCH_DATABASES, SEARCH_DATABASES_PATTERNS),
    (ToolName.SEARCH_TCGA, SEARCH_TCGA_PATTERNS),
    (ToolName.GET_DATASET_DETAILS, GET_DATASET_DETAILS_PATTERNS),
    (ToolName.DESCRIBE_FILES, DESCRIBE_FILES_PATTERNS),
    (ToolName.VALIDATE_DATASET, VALIDATE_DATASET_PATTERNS),
    
    # Data Management
    (ToolName.DOWNLOAD_DATASET, DOWNLOAD_DATASET_PATTERNS),
    (ToolName.DOWNLOAD_REFERENCE, DOWNLOAD_REFERENCE_PATTERNS),
    (ToolName.BUILD_INDEX, BUILD_INDEX_PATTERNS),
    (ToolName.CLEANUP_DATA, CLEANUP_DATA_PATTERNS),
    (ToolName.CONFIRM_CLEANUP, CONFIRM_CLEANUP_PATTERNS),
    
    # Workflow
    (ToolName.GENERATE_WORKFLOW, GENERATE_WORKFLOW_PATTERNS),
    (ToolName.LIST_WORKFLOWS, LIST_WORKFLOWS_PATTERNS),
    (ToolName.CHECK_REFERENCES, CHECK_REFERENCES_PATTERNS),
    (ToolName.VISUALIZE_WORKFLOW, VISUALIZE_WORKFLOW_PATTERNS),
    
    # Execution - more specific patterns first
    (ToolName.RESTART_VLLM, RESTART_VLLM_PATTERNS),
    (ToolName.RESUBMIT_JOB, RESUBMIT_JOB_PATTERNS),
    (ToolName.WATCH_JOB, WATCH_JOB_PATTERNS),
    (ToolName.LIST_JOBS, LIST_JOBS_PATTERNS),
    (ToolName.SUBMIT_JOB, SUBMIT_JOB_PATTERNS),
    (ToolName.GET_JOB_STATUS, GET_JOB_STATUS_PATTERNS),
    (ToolName.GET_LOGS, GET_LOGS_PATTERNS),
    (ToolName.CANCEL_JOB, CANCEL_JOB_PATTERNS),
    (ToolName.CHECK_SYSTEM_HEALTH, CHECK_SYSTEM_HEALTH_PATTERNS),
    
    # Diagnostics - recover_error before diagnose_error
    (ToolName.RECOVER_ERROR, RECOVER_ERROR_PATTERNS),
    (ToolName.DIAGNOSE_ERROR, DIAGNOSE_ERROR_PATTERNS),
    (ToolName.ANALYZE_RESULTS, ANALYZE_RESULTS_PATTERNS),
    
    # Education
    (ToolName.EXPLAIN_CONCEPT, EXPLAIN_CONCEPT_PATTERNS),
    (ToolName.COMPARE_SAMPLES, COMPARE_SAMPLES_PATTERNS),
    (ToolName.SHOW_HELP, GET_HELP_PATTERNS),
    
    # Database Clients (Phase 2)
    (ToolName.SEARCH_UNIPROT, SEARCH_UNIPROT_PATTERNS),
    (ToolName.GET_PROTEIN, GET_PROTEIN_PATTERNS),
    (ToolName.SEARCH_STRING, SEARCH_STRING_PATTERNS),
    (ToolName.GET_INTERACTIONS, GET_INTERACTIONS_PATTERNS),
    (ToolName.GET_ENRICHMENT, GET_ENRICHMENT_PATTERNS),
    (ToolName.SEARCH_KEGG, SEARCH_KEGG_PATTERNS),
    (ToolName.GET_PATHWAY, GET_PATHWAY_PATTERNS),
    (ToolName.SEARCH_REACTOME, SEARCH_REACTOME_PATTERNS),
    (ToolName.ANALYZE_GENES, ANALYZE_GENES_PATTERNS),
    (ToolName.SEARCH_PUBMED, SEARCH_PUBMED_PATTERNS),
    (ToolName.GET_ARTICLE, GET_ARTICLE_PATTERNS),
    (ToolName.SEARCH_CLINVAR, SEARCH_CLINVAR_PATTERNS),
    (ToolName.GET_VARIANTS, GET_VARIANTS_PATTERNS),
]


# =============================================================================
# UNIFIED AGENT TOOLS CLASS
# =============================================================================

class AgentTools:
    """
    Unified interface for all BioPipelines agent tools.
    
    This class provides:
    - Centralized tool execution
    - Pattern-based tool detection
    - OpenAI function definitions
    
    Usage:
        tools = AgentTools()
        result = tools.execute_tool("scan_data", path="/data")
    """
    
    def __init__(self, base_path: str = None):
        """
        Initialize agent tools.
        
        Args:
            base_path: Base directory for data operations
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        
        # Initialize scanner for data discovery
        self._scanner = None
        self._manifest = None
        self._init_data_components()
        
        # Build tool dispatch table
        self._tool_dispatch = self._build_dispatch_table()
        
        # Cache for OpenAI function definitions
        self._function_definitions = None
        
        logger.info(f"AgentTools initialized with {len(self._tool_dispatch)} tools")
    
    def _init_data_components(self):
        """Initialize data scanner and manifest."""
        try:
            from workflow_composer.data import LocalSampleScanner, DataManifest
            self._scanner = LocalSampleScanner()
            self._manifest = DataManifest()
            logger.info("Data scanner initialized successfully")
        except ImportError as e:
            logger.warning(f"Data components not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize data components: {e}")
    
    def _build_dispatch_table(self) -> Dict[str, callable]:
        """Build mapping from tool name to implementation."""
        dispatch = {
            # Data Discovery - pass scanner and manifest
            "scan_data": lambda **kw: scan_data_impl(scanner=self._scanner, manifest=self._manifest, **kw),
            "search_databases": lambda **kw: search_databases_impl(**kw),
            "search_tcga": lambda **kw: search_tcga_impl(**kw),
            "get_dataset_details": lambda **kw: get_dataset_details_impl(**kw),
            "describe_files": lambda **kw: describe_files_impl(**kw),
            "validate_dataset": lambda **kw: validate_dataset_impl(**kw),
            
            # Data Management
            "download_dataset": lambda **kw: download_dataset_impl(**kw),
            "download_reference": lambda **kw: download_reference_impl(**kw),
            "build_index": lambda **kw: build_index_impl(**kw),
            "cleanup_data": lambda **kw: cleanup_data_impl(**kw),
            "confirm_cleanup": lambda **kw: confirm_cleanup_impl(**kw),
            
            # Workflow
            "generate_workflow": lambda **kw: generate_workflow_impl(**kw),
            "list_workflows": lambda **kw: list_workflows_impl(**kw),
            "check_references": lambda **kw: check_references_impl(**kw),
            "visualize_workflow": lambda **kw: visualize_workflow_impl(**kw),
            
            # Execution
            "submit_job": lambda **kw: submit_job_impl(**kw),
            "get_job_status": lambda **kw: get_job_status_impl(**kw),
            "get_logs": lambda **kw: get_logs_impl(**kw),
            "cancel_job": lambda **kw: cancel_job_impl(**kw),
            "check_system_health": lambda **kw: check_system_health_impl(**kw),
            "restart_vllm": lambda **kw: restart_vllm_impl(**kw),
            "resubmit_job": lambda **kw: resubmit_job_impl(**kw),
            "watch_job": lambda **kw: watch_job_impl(**kw),
            "list_jobs": lambda **kw: list_jobs_impl(**kw),
            
            # Diagnostics
            "diagnose_error": lambda **kw: diagnose_error_impl(**kw),
            "analyze_results": lambda **kw: analyze_results_impl(**kw),
            "recover_error": lambda **kw: recover_error_impl(**kw),
            
            # Education
            "explain_concept": lambda **kw: explain_concept_impl(**kw),
            "compare_samples": lambda **kw: compare_samples_impl(**kw),
            "get_help": lambda **kw: get_help_impl(),
            "show_help": lambda **kw: get_help_impl(),  # Alias for get_help (matches ToolName.SHOW_HELP)
            
            # Database Clients (Phase 2)
            "search_uniprot": lambda **kw: self._search_uniprot(**kw),
            "get_protein": lambda **kw: self._get_protein(**kw),
            "search_string": lambda **kw: self._search_string(**kw),
            "get_interactions": lambda **kw: self._get_interactions(**kw),
            "get_enrichment": lambda **kw: self._get_enrichment(**kw),
            "search_kegg": lambda **kw: self._search_kegg(**kw),
            "get_pathway": lambda **kw: self._get_pathway(**kw),
            "search_reactome": lambda **kw: self._search_reactome(**kw),
            "analyze_genes": lambda **kw: self._analyze_genes(**kw),
            "search_pubmed": lambda **kw: self._search_pubmed(**kw),
            "get_article": lambda **kw: self._get_article(**kw),
            "search_clinvar": lambda **kw: self._search_clinvar(**kw),
            "get_variants": lambda **kw: self._get_variants(**kw),
        }
        return dispatch
    
    # =========================================================================
    # DATABASE CLIENT IMPLEMENTATIONS (Phase 2)
    # =========================================================================
    
    def _search_uniprot(self, query: str, organism: str = "human", limit: int = 10, **kwargs) -> ToolResult:
        """Search UniProt for proteins."""
        try:
            client = get_uniprot_client()
            result = client.search(query, organism=organism, limit=limit)
            if result.success:
                return ToolResult(
                    success=True,
                    tool_name="search_uniprot",
                    data=result.data,
                    message=f"✅ Found {result.count} proteins in UniProt for '{query}'"
                )
            return ToolResult(
                success=False,
                tool_name="search_uniprot",
                error=result.message,
                message=f"❌ UniProt search failed: {result.message}"
            )
        except Exception as e:
            return ToolResult(success=False, tool_name="search_uniprot", error=str(e))
    
    def _get_protein(self, accession: str, **kwargs) -> ToolResult:
        """Get protein details from UniProt."""
        try:
            client = get_uniprot_client()
            result = client.get_by_id(accession)
            if result.success:
                return ToolResult(
                    success=True,
                    tool_name="get_protein",
                    data=result.data,
                    message=f"✅ Retrieved protein {accession} from UniProt"
                )
            return ToolResult(
                success=False,
                tool_name="get_protein",
                error=result.message,
                message=f"❌ Failed to get protein: {result.message}"
            )
        except Exception as e:
            return ToolResult(success=False, tool_name="get_protein", error=str(e))
    
    def _search_string(self, query: str, organism: str = "human", limit: int = 10, **kwargs) -> ToolResult:
        """Search STRING for proteins."""
        try:
            client = get_string_client()
            result = client.search(query, organism=organism, limit=limit)
            if result.success:
                return ToolResult(
                    success=True,
                    tool_name="search_string",
                    data=result.data,
                    message=f"✅ Found {result.count} proteins in STRING for '{query}'"
                )
            return ToolResult(
                success=False,
                tool_name="search_string",
                error=result.message,
                message=f"❌ STRING search failed: {result.message}"
            )
        except Exception as e:
            return ToolResult(success=False, tool_name="search_string", error=str(e))
    
    def _get_interactions(self, proteins: str, organism: str = "human", score_threshold: float = 0.4, **kwargs) -> ToolResult:
        """Get protein-protein interactions from STRING."""
        try:
            client = get_string_client()
            # proteins can be comma-separated or a single protein
            protein_list = [p.strip() for p in proteins.split(",")]
            result = client.get_interactions(protein_list, organism=organism, score_threshold=score_threshold)
            if result.success:
                return ToolResult(
                    success=True,
                    tool_name="get_interactions",
                    data=result.data,
                    message=f"✅ Found {result.count} interactions from STRING"
                )
            return ToolResult(
                success=False,
                tool_name="get_interactions",
                error=result.message,
                message=f"❌ Failed to get interactions: {result.message}"
            )
        except Exception as e:
            return ToolResult(success=False, tool_name="get_interactions", error=str(e))
    
    def _get_enrichment(self, genes: str, organism: str = "human", **kwargs) -> ToolResult:
        """Get functional enrichment from STRING."""
        try:
            client = get_string_client()
            gene_list = [g.strip() for g in genes.split(",")]
            result = client.get_enrichment(gene_list, organism=organism)
            if result.success:
                return ToolResult(
                    success=True,
                    tool_name="get_enrichment",
                    data=result.data,
                    message=f"✅ Found {result.count} enriched terms"
                )
            return ToolResult(
                success=False,
                tool_name="get_enrichment",
                error=result.message,
                message=f"❌ Enrichment analysis failed: {result.message}"
            )
        except Exception as e:
            return ToolResult(success=False, tool_name="get_enrichment", error=str(e))
    
    def _search_kegg(self, query: str, database: str = "pathway", organism: str = "hsa", **kwargs) -> ToolResult:
        """Search KEGG for pathways or genes."""
        try:
            client = get_kegg_client()
            result = client.search(query, database=database, organism=organism)
            if result.success:
                return ToolResult(
                    success=True,
                    tool_name="search_kegg",
                    data=result.data,
                    message=f"✅ Found {result.count} results in KEGG"
                )
            return ToolResult(
                success=False,
                tool_name="search_kegg",
                error=result.message,
                message=f"❌ KEGG search failed: {result.message}"
            )
        except Exception as e:
            return ToolResult(success=False, tool_name="search_kegg", error=str(e))
    
    def _get_pathway(self, pathway_id: str, **kwargs) -> ToolResult:
        """Get pathway details from KEGG."""
        try:
            client = get_kegg_client()
            result = client.get_pathway(pathway_id)
            if result.success:
                return ToolResult(
                    success=True,
                    tool_name="get_pathway",
                    data=result.data,
                    message=f"✅ Retrieved pathway {pathway_id} from KEGG"
                )
            return ToolResult(
                success=False,
                tool_name="get_pathway",
                error=result.message,
                message=f"❌ Failed to get pathway: {result.message}"
            )
        except Exception as e:
            return ToolResult(success=False, tool_name="get_pathway", error=str(e))
    
    def _search_reactome(self, query: str, species: str = "Homo sapiens", **kwargs) -> ToolResult:
        """Search Reactome for pathways."""
        try:
            client = get_reactome_client()
            result = client.search(query, species=species)
            if result.success:
                return ToolResult(
                    success=True,
                    tool_name="search_reactome",
                    data=result.data,
                    message=f"✅ Found {result.count} pathways in Reactome"
                )
            return ToolResult(
                success=False,
                tool_name="search_reactome",
                error=result.message,
                message=f"❌ Reactome search failed: {result.message}"
            )
        except Exception as e:
            return ToolResult(success=False, tool_name="search_reactome", error=str(e))
    
    def _analyze_genes(self, genes: str, species: str = "Homo sapiens", **kwargs) -> ToolResult:
        """Analyze gene list with Reactome pathway analysis."""
        try:
            client = get_reactome_client()
            gene_list = [g.strip() for g in genes.split(",")]
            result = client.analyze_genes(gene_list, species=species)
            if result.success:
                return ToolResult(
                    success=True,
                    tool_name="analyze_genes",
                    data=result.data,
                    message=f"✅ Pathway analysis complete: {result.count} enriched pathways"
                )
            return ToolResult(
                success=False,
                tool_name="analyze_genes",
                error=result.message,
                message=f"❌ Gene analysis failed: {result.message}"
            )
        except Exception as e:
            return ToolResult(success=False, tool_name="analyze_genes", error=str(e))
    
    def _search_pubmed(self, query: str, limit: int = 10, sort: str = "relevance", **kwargs) -> ToolResult:
        """Search PubMed for articles."""
        try:
            client = get_pubmed_client()
            result = client.search(query, max_results=limit, sort=sort)
            if result.success:
                return ToolResult(
                    success=True,
                    tool_name="search_pubmed",
                    data=result.data,
                    message=f"✅ Found {result.count} articles in PubMed"
                )
            return ToolResult(
                success=False,
                tool_name="search_pubmed",
                error=result.message,
                message=f"❌ PubMed search failed: {result.message}"
            )
        except Exception as e:
            return ToolResult(success=False, tool_name="search_pubmed", error=str(e))
    
    def _get_article(self, pmid: str, **kwargs) -> ToolResult:
        """Get article details from PubMed."""
        try:
            client = get_pubmed_client()
            result = client.get_by_id(pmid)
            if result.success:
                return ToolResult(
                    success=True,
                    tool_name="get_article",
                    data=result.data,
                    message=f"✅ Retrieved article PMID {pmid}"
                )
            return ToolResult(
                success=False,
                tool_name="get_article",
                error=result.message,
                message=f"❌ Failed to get article: {result.message}"
            )
        except Exception as e:
            return ToolResult(success=False, tool_name="get_article", error=str(e))
    
    def _search_clinvar(self, query: str, limit: int = 20, **kwargs) -> ToolResult:
        """Search ClinVar for variants."""
        try:
            client = get_clinvar_client()
            result = client.search(query, max_results=limit)
            if result.success:
                return ToolResult(
                    success=True,
                    tool_name="search_clinvar",
                    data=result.data,
                    message=f"✅ Found {result.count} variants in ClinVar"
                )
            return ToolResult(
                success=False,
                tool_name="search_clinvar",
                error=result.message,
                message=f"❌ ClinVar search failed: {result.message}"
            )
        except Exception as e:
            return ToolResult(success=False, tool_name="search_clinvar", error=str(e))
    
    def _get_variants(self, gene: str, significance: str = None, limit: int = 20, **kwargs) -> ToolResult:
        """Get variants for a gene from ClinVar."""
        try:
            client = get_clinvar_client()
            if significance:
                result = client.get_pathogenic_variants(gene, max_results=limit)
            else:
                result = client.search_gene(gene, max_results=limit)
            if result.success:
                return ToolResult(
                    success=True,
                    tool_name="get_variants",
                    data=result.data,
                    message=f"✅ Found {result.count} variants for {gene}"
                )
            return ToolResult(
                success=False,
                tool_name="get_variants",
                error=result.message,
                message=f"❌ Failed to get variants: {result.message}"
            )
        except Exception as e:
            return ToolResult(success=False, tool_name="get_variants", error=str(e))
    
    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Execute a tool by name.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Arguments to pass to the tool
            
        Returns:
            ToolResult with execution results
        """
        if tool_name not in self._tool_dispatch:
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=f"Unknown tool: {tool_name}",
                message=f"❌ Unknown tool: `{tool_name}`. Use 'help' to see available tools."
            )
        
        try:
            tool_func = self._tool_dispatch[tool_name]
            result = tool_func(**kwargs)
            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}", exc_info=True)
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=str(e),
                message=f"❌ Error executing {tool_name}: {e}"
            )
    
    # Mapping of tool names to their primary argument names
    _TOOL_ARG_MAPPING = {
        ToolName.SCAN_DATA: ["path"],
        ToolName.SEARCH_DATABASES: ["query"],
        ToolName.SEARCH_TCGA: ["query", "cancer_type"],
        ToolName.GET_DATASET_DETAILS: ["dataset_id"],
        ToolName.DESCRIBE_FILES: ["path"],
        ToolName.VALIDATE_DATASET: ["path"],
        ToolName.DOWNLOAD_DATASET: ["dataset_id", "destination"],
        ToolName.DOWNLOAD_REFERENCE: ["genome"],
        ToolName.BUILD_INDEX: ["reference", "tool"],
        ToolName.GENERATE_WORKFLOW: ["pipeline_type", "input_dir"],
        ToolName.CHECK_REFERENCES: ["genome"],
        ToolName.SUBMIT_JOB: ["workflow_dir"],
        ToolName.GET_JOB_STATUS: ["job_id"],
        ToolName.GET_LOGS: ["job_id"],
        ToolName.CANCEL_JOB: ["job_id"],
        ToolName.DIAGNOSE_ERROR: ["job_id", "log_content"],
        # Database Clients (Phase 2)
        ToolName.SEARCH_UNIPROT: ["query", "organism"],
        ToolName.GET_PROTEIN: ["accession"],
        ToolName.SEARCH_STRING: ["query", "organism"],
        ToolName.GET_INTERACTIONS: ["proteins", "organism"],
        ToolName.GET_ENRICHMENT: ["genes", "organism"],
        ToolName.SEARCH_KEGG: ["query", "database"],
        ToolName.GET_PATHWAY: ["pathway_id"],
        ToolName.SEARCH_REACTOME: ["query", "species"],
        ToolName.ANALYZE_GENES: ["genes", "species"],
        ToolName.SEARCH_PUBMED: ["query", "limit"],
        ToolName.GET_ARTICLE: ["pmid"],
        ToolName.SEARCH_CLINVAR: ["query", "limit"],
        ToolName.GET_VARIANTS: ["gene", "significance"],
    }
    
    def execute(self, tool_name_or_enum, args: List[str] = None) -> ToolResult:
        """
        Execute a tool, converting positional args to kwargs.
        
        This method handles the conversion from pattern-captured args to
        proper keyword arguments for tool implementations.
        
        Args:
            tool_name_or_enum: Tool name (str or ToolName enum)
            args: List of captured arguments from pattern matching
            
        Returns:
            ToolResult with execution results
        """
        # Normalize tool name
        if isinstance(tool_name_or_enum, ToolName):
            tool_name = tool_name_or_enum.value
            tool_enum = tool_name_or_enum
        else:
            tool_name = str(tool_name_or_enum)
            try:
                tool_enum = ToolName(tool_name)
            except ValueError:
                tool_enum = None
        
        # Build kwargs from positional args
        kwargs = {}
        if args and tool_enum and tool_enum in self._TOOL_ARG_MAPPING:
            arg_names = self._TOOL_ARG_MAPPING[tool_enum]
            for i, arg in enumerate(args):
                if arg and i < len(arg_names):
                    kwargs[arg_names[i]] = arg
        
        return self.execute_tool(tool_name, **kwargs)
    
    def detect_tool(self, message: str) -> Optional[Tuple[ToolName, List[str]]]:
        """
        Detect which tool a user message is requesting.
        
        Args:
            message: User message to analyze
            
        Returns:
            Tuple of (ToolName, captured_args) if detected, None otherwise
        """
        message_lower = message.lower().strip()
        
        for tool_name, patterns in ALL_TOOL_PATTERNS:
            for pattern in patterns:
                match = re.search(pattern, message_lower, re.IGNORECASE)
                if match:
                    # Extract captured groups as args
                    args = list(match.groups()) if match.groups() else []
                    return (tool_name, args)
        
        return None
    
    def show_help(self) -> ToolResult:
        """
        Show help information about available tools.
        
        Returns:
            ToolResult with help content
        """
        categories = {
            "Data Discovery": [
                "scan_data - Scan directories for data files",
                "search_databases - Search ENCODE, GEO, SRA for datasets",
                "describe_files - Get metadata about files",
            ],
            "Workflow Generation": [
                "generate_workflow - Create analysis pipelines",
                "check_references - Verify reference genomes",
                "list_workflows - Show available workflow types",
            ],
            "Job Management": [
                "submit_job - Submit jobs to SLURM",
                "get_job_status - Check job status",
                "get_logs - View job logs",
                "cancel_job - Cancel running jobs",
            ],
            "Diagnostics": [
                "diagnose_error - Analyze job failures",
                "check_system_health - Check system status",
            ],
        }
        
        help_text = "# Available Tools\n\n"
        for category, tools in categories.items():
            help_text += f"## {category}\n"
            for tool in tools:
                help_text += f"- {tool}\n"
            help_text += "\n"
        
        return ToolResult(
            success=True,
            tool_name="help",
            data=categories,
            message=help_text
        )
    
    def get_openai_functions(self) -> List[Dict[str, Any]]:
        """
        Get OpenAI function definitions for all tools.
        
        Returns:
            List of function definitions in OpenAI format
        """
        if self._function_definitions:
            return self._function_definitions
        
        self._function_definitions = [
            {
                "name": "scan_data",
                "description": "Scan the workspace for data files (FASTQ, BAM, VCF, etc.)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory to scan"},
                    },
                    "required": []
                }
            },
            {
                "name": "search_databases",
                "description": "Comprehensive parallel search across ALL public databases (ENCODE, GEO, SRA, Ensembl) plus TCGA for cancer queries. Automatically deduplicates results. Use this for broad data discovery.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language search query (e.g., 'human brain methylation', 'mouse RNA-seq liver', 'cancer DNA methylation')"},
                        "include_tcga": {"type": "boolean", "description": "Include TCGA search for cancer-related queries (default: true)"},
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "search_tcga",
                "description": "Search TCGA/GDC for cancer data. Cancer types: GBM/glioblastoma (BRAIN), BRCA (BREAST), LUAD (LUNG), COAD (COLON), PRAD (PROSTATE), KIRC (KIDNEY), LIHC (LIVER), SKCM (MELANOMA).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cancer_type": {
                            "type": "string", 
                            "description": "Cancer type: GBM/brain (glioblastoma), BRCA (breast), LUAD (lung), COAD (colon), etc."
                        },
                        "data_type": {
                            "type": "string", 
                            "description": "Data type: methylation, RNA-seq, WXS, WGS, clinical"
                        },
                    },
                    "required": []
                }
            },
            {
                "name": "download_dataset",
                "description": "Download a dataset from GEO, ENCODE, or TCGA/GDC. For TCGA, use project codes like TCGA-GBM (brain cancer), TCGA-BRCA (breast), TCGA-LUAD (lung).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {
                            "type": "string", 
                            "description": "Dataset ID: GSE* (GEO), ENCSR* (ENCODE), or TCGA-* (GDC). For brain cancer use TCGA-GBM, for breast use TCGA-BRCA."
                        },
                        "data_type": {
                            "type": "string",
                            "description": "Type of data: methylation, rnaseq, wgs, wes, clinical"
                        },
                    },
                    "required": ["dataset_id"]
                }
            },
            {
                "name": "generate_workflow",
                "description": "Generate a Nextflow/Snakemake workflow",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workflow_type": {"type": "string", "description": "Type (rnaseq, chipseq, etc.)"},
                        "input_files": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["workflow_type"]
                }
            },
            {
                "name": "list_workflows",
                "description": "List available workflow templates",
                "parameters": {"type": "object", "properties": {}, "required": []}
            },
            {
                "name": "check_references",
                "description": "Check availability of reference genomes, annotations, and aligner indexes. Uses ReferenceManager for comprehensive status.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "organism": {"type": "string", "description": "Organism (human, mouse, rat, zebrafish)"},
                        "assembly": {"type": "string", "description": "Genome assembly (GRCh38, GRCm39, etc.)"},
                    },
                    "required": []
                }
            },
            {
                "name": "download_reference",
                "description": "Download reference genome, GTF annotation, or transcriptome from Ensembl. Supports human, mouse, rat, zebrafish.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "organism": {"type": "string", "description": "Organism: human, mouse, rat, zebrafish"},
                        "assembly": {"type": "string", "description": "Assembly: GRCh38, GRCh37, GRCm39, GRCm38, mRatBN7.2, GRCz11"},
                        "resource": {"type": "string", "description": "Resource type: genome, gtf, transcriptome"},
                    },
                    "required": ["organism", "assembly", "resource"]
                }
            },
            {
                "name": "build_index",
                "description": "Build an aligner index (STAR, Salmon, BWA, HISAT2, Kallisto) for a genome.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "aligner": {"type": "string", "description": "Aligner: star, salmon, bwa, hisat2, kallisto"},
                        "organism": {"type": "string", "description": "Organism (used if genome_path not provided)"},
                        "assembly": {"type": "string", "description": "Assembly (used if genome_path not provided)"},
                        "genome_path": {"type": "string", "description": "Path to genome FASTA (optional)"},
                        "gtf_path": {"type": "string", "description": "Path to GTF annotation (optional, recommended for STAR)"},
                    },
                    "required": ["aligner"]
                }
            },
            {
                "name": "visualize_workflow",
                "description": "Generate a DAG diagram visualization of a Nextflow workflow.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workflow_dir": {"type": "string", "description": "Path to workflow directory"},
                        "output_format": {"type": "string", "description": "Output format: png, svg, pdf, txt"},
                    },
                    "required": []
                }
            },
            {
                "name": "submit_job",
                "description": "Submit a workflow job to SLURM",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workflow_path": {"type": "string", "description": "Path to workflow"},
                        "profile": {"type": "string", "enum": ["slurm", "local", "docker"]},
                    },
                    "required": []
                }
            },
            {
                "name": "check_system_health",
                "description": "Check system health including vLLM server, GPU status, disk space, memory, and SLURM availability.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "restart_vllm",
                "description": "Restart the vLLM LLM server. Use when the server is unresponsive or crashed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "force": {"type": "boolean", "description": "Force kill existing processes"},
                        "wait_healthy": {"type": "boolean", "description": "Wait for server to become healthy"},
                        "timeout": {"type": "integer", "description": "Timeout in seconds (default: 60)"},
                    },
                    "required": []
                }
            },
            {
                "name": "resubmit_job",
                "description": "Resubmit a failed SLURM job, optionally with modified resources.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string", "description": "Original SLURM job ID to resubmit"},
                        "script_path": {"type": "string", "description": "Path to submission script (auto-detected if job_id provided)"},
                        "modify_resources": {"type": "object", "description": "Resource modifications (e.g., {\"mem\": \"32G\", \"time\": \"4:00:00\"})"},
                    },
                    "required": []
                }
            },
            {
                "name": "watch_job",
                "description": "Get detailed SLURM job information including metadata, times, and optional logs. More comprehensive than get_job_status.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string", "description": "SLURM job ID to monitor"},
                        "include_logs": {"type": "boolean", "description": "Include recent log output for failed jobs"},
                        "tail_lines": {"type": "integer", "description": "Number of log lines to include (default: 50)"},
                    },
                    "required": []
                }
            },
            {
                "name": "list_jobs",
                "description": "List all SLURM jobs for the current user. Shows job IDs, names, states, and runtimes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user": {"type": "string", "description": "Filter by username (default: current user)"},
                        "state": {"type": "string", "description": "Filter by state: pending, running, completed, failed"},
                        "partition": {"type": "string", "description": "Filter by partition name"},
                    },
                    "required": []
                }
            },
            {
                "name": "recover_error",
                "description": "Execute recovery actions for diagnosed errors. Supports restart_server, resubmit_job, clear_cache, install_module.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "description": "Recovery action: restart_server, resubmit_job, clear_cache, install_module"},
                        "job_id": {"type": "string", "description": "SLURM job ID (for resubmit_job)"},
                        "error_log": {"type": "string", "description": "Error log content for diagnosis"},
                        "confirm": {"type": "boolean", "description": "Execute without confirmation prompt"},
                    },
                    "required": []
                }
            },
            {
                "name": "get_job_status",
                "description": "Get status of SLURM jobs and/or Nextflow workflow execution. Shows progress bars and process counts for Nextflow.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string", "description": "SLURM job ID to check"},
                        "workflow_dir": {"type": "string", "description": "Nextflow workflow directory to monitor"},
                    },
                    "required": []
                }
            },
            {
                "name": "diagnose_error",
                "description": "Diagnose pipeline errors using 50+ patterns with AI-powered analysis. Supports OutOfMemory, Permission, DiskSpace, Network, SLURM, and more.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "error_text": {"type": "string", "description": "Error text to diagnose"},
                        "log_file": {"type": "string", "description": "Path to log file to analyze"},
                        "job_id": {"type": "string", "description": "SLURM job ID to collect logs from"},
                        "work_dir": {"type": "string", "description": "Nextflow work directory to scan"},
                        "auto_fix": {"type": "boolean", "description": "Attempt automatic fixes for safe operations"},
                    },
                    "required": []
                }
            },
            {
                "name": "analyze_results",
                "description": "Analyze workflow results and provide interpretation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "results_path": {"type": "string", "description": "Path to results"},
                    },
                    "required": []
                }
            },
            {
                "name": "explain_concept",
                "description": "Explain a bioinformatics concept or tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "concept": {"type": "string", "description": "Concept to explain"},
                    },
                    "required": ["concept"]
                }
            },
            {
                "name": "compare_samples",
                "description": "Compare two samples or conditions with statistical analysis and visualization suggestions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sample1": {"type": "string", "description": "First sample or condition"},
                        "sample2": {"type": "string", "description": "Second sample or condition"},
                        "comparison_type": {"type": "string", "description": "Type: expression, methylation, variants, peaks"},
                    },
                    "required": ["sample1", "sample2"]
                }
            },
            {
                "name": "describe_files",
                "description": "Get detailed description and statistics for a data file (FASTQ, BAM, VCF, etc.)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to the file to describe"},
                    },
                    "required": ["file_path"]
                }
            },
            {
                "name": "validate_dataset",
                "description": "Validate a dataset for completeness, format, and quality issues",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to dataset directory or file"},
                        "expected_samples": {"type": "integer", "description": "Expected number of samples"},
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "cleanup_data",
                "description": "Identify and optionally remove unnecessary files (temp files, logs, intermediate outputs)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory to scan for cleanup"},
                        "dry_run": {"type": "boolean", "description": "If true, only report what would be deleted"},
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "confirm_cleanup",
                "description": "Confirm and execute a previously proposed cleanup operation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cleanup_id": {"type": "string", "description": "Cleanup operation ID from previous dry run"},
                        "confirmed": {"type": "boolean", "description": "Set to true to confirm deletion"},
                    },
                    "required": ["cleanup_id", "confirmed"]
                }
            },
            {
                "name": "get_logs",
                "description": "Get workflow execution logs from Nextflow or SLURM",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workflow_dir": {"type": "string", "description": "Workflow directory path"},
                        "job_id": {"type": "string", "description": "SLURM job ID"},
                        "tail_lines": {"type": "integer", "description": "Number of lines from end (default: 50)"},
                    },
                    "required": []
                }
            },
            {
                "name": "cancel_job",
                "description": "Cancel a running SLURM job",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string", "description": "SLURM job ID to cancel"},
                    },
                    "required": ["job_id"]
                }
            },
            {
                "name": "get_help",
                "description": "Show help information about available commands",
                "parameters": {"type": "object", "properties": {}, "required": []}
            },
            # Database Clients (Phase 2)
            {
                "name": "search_uniprot",
                "description": "Search UniProt protein database for proteins by name, gene, function, or organism. Returns protein accessions, names, and annotations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query (protein name, gene symbol, function keyword)"},
                        "organism": {"type": "string", "description": "Organism filter: human, mouse, rat, yeast, etc. (default: human)"},
                        "limit": {"type": "integer", "description": "Maximum results to return (default: 10)"},
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_protein",
                "description": "Get detailed protein information from UniProt by accession ID. Includes sequence, function, GO terms, and more.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "accession": {"type": "string", "description": "UniProt accession ID (e.g., P53_HUMAN, P04637)"},
                    },
                    "required": ["accession"]
                }
            },
            {
                "name": "search_string",
                "description": "Search STRING protein interaction database for proteins. STRING provides protein-protein interaction networks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Protein or gene name to search"},
                        "organism": {"type": "string", "description": "Organism: human, mouse, yeast, etc. (default: human)"},
                        "limit": {"type": "integer", "description": "Maximum results (default: 10)"},
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_interactions",
                "description": "Get protein-protein interactions from STRING database. Returns interaction partners with confidence scores.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "proteins": {"type": "string", "description": "Comma-separated protein names (e.g., 'TP53,MDM2,BRCA1')"},
                        "organism": {"type": "string", "description": "Organism: human, mouse, etc. (default: human)"},
                        "score_threshold": {"type": "number", "description": "Minimum interaction score 0-1 (default: 0.4)"},
                    },
                    "required": ["proteins"]
                }
            },
            {
                "name": "get_enrichment",
                "description": "Perform functional enrichment analysis on a gene list using STRING. Returns enriched GO terms, KEGG pathways, and more.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "genes": {"type": "string", "description": "Comma-separated gene/protein names (e.g., 'TP53,BRCA1,EGFR')"},
                        "organism": {"type": "string", "description": "Organism: human, mouse, etc. (default: human)"},
                    },
                    "required": ["genes"]
                }
            },
            {
                "name": "search_kegg",
                "description": "Search KEGG pathway database. KEGG provides metabolic and signaling pathway information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query (pathway name, gene, compound)"},
                        "database": {"type": "string", "description": "Database to search: pathway, genes, compound (default: pathway)"},
                        "organism": {"type": "string", "description": "KEGG organism code: hsa (human), mmu (mouse), etc. (default: hsa)"},
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_pathway",
                "description": "Get detailed KEGG pathway information including genes, reactions, and connections.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pathway_id": {"type": "string", "description": "KEGG pathway ID (e.g., hsa04110, map04110)"},
                    },
                    "required": ["pathway_id"]
                }
            },
            {
                "name": "search_reactome",
                "description": "Search Reactome pathway database. Reactome provides curated pathway and reaction information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query (pathway name, gene, reaction)"},
                        "species": {"type": "string", "description": "Species filter (default: Homo sapiens)"},
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "analyze_genes",
                "description": "Perform Reactome pathway enrichment analysis on a gene list. Identifies over-represented biological pathways.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "genes": {"type": "string", "description": "Comma-separated gene symbols (e.g., 'TP53,BRCA1,EGFR')"},
                        "species": {"type": "string", "description": "Species (default: Homo sapiens)"},
                    },
                    "required": ["genes"]
                }
            },
            {
                "name": "search_pubmed",
                "description": "Search PubMed for scientific articles and publications. Returns article titles, abstracts, and PMIDs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query (keywords, authors, MeSH terms)"},
                        "limit": {"type": "integer", "description": "Maximum results (default: 10)"},
                        "sort": {"type": "string", "description": "Sort by: relevance, pub_date, first_author (default: relevance)"},
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_article",
                "description": "Get detailed article information from PubMed by PMID. Returns title, abstract, authors, and citations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pmid": {"type": "string", "description": "PubMed ID (e.g., 12345678)"},
                    },
                    "required": ["pmid"]
                }
            },
            {
                "name": "search_clinvar",
                "description": "Search ClinVar for clinical variant annotations. Returns variant pathogenicity and clinical significance.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query (gene, variant, condition)"},
                        "limit": {"type": "integer", "description": "Maximum results (default: 20)"},
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_variants",
                "description": "Get ClinVar variants for a specific gene, optionally filtered by clinical significance.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gene": {"type": "string", "description": "Gene symbol (e.g., BRCA1, TP53)"},
                        "significance": {"type": "string", "description": "Filter by significance: pathogenic, likely_pathogenic, benign (optional)"},
                        "limit": {"type": "integer", "description": "Maximum results (default: 20)"},
                    },
                    "required": ["gene"]
                }
            },
        ]
        
        return self._function_definitions
    
    def get_tool_names(self) -> List[str]:
        """Get list of all available tool names."""
        return list(self._tool_dispatch.keys())
    
    def get_tool_count(self) -> int:
        """Get count of available tools."""
        return len(self._tool_dispatch)
    
    @property
    def tools(self) -> Dict[str, callable]:
        """Get the tool dispatch table (for introspection)."""
        return self._tool_dispatch


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

_agent_tools: Optional[AgentTools] = None


def get_agent_tools(base_path: str = None) -> AgentTools:
    """
    Get the singleton AgentTools instance.
    
    Args:
        base_path: Base path for data operations
        
    Returns:
        AgentTools instance
    """
    global _agent_tools
    if _agent_tools is None:
        _agent_tools = AgentTools(base_path=base_path)
    return _agent_tools


def process_tool_request(tool_name_or_message: str, args_or_state=None) -> Optional[ToolResult]:
    """
    Process a tool request.
    
    Supports two calling conventions:
    1. process_tool_request(tool_name, args_dict) - Execute specific tool
    2. process_tool_request(message, app_state) - Detect and execute from message
    
    Args:
        tool_name_or_message: Tool name or user's chat message
        args_or_state: Tool arguments dict or application state
        
    Returns:
        ToolResult if a tool was executed, None otherwise
    """
    tools = get_agent_tools()
    
    # Check if first arg is a known tool name (new convention)
    if tool_name_or_message in tools._tool_dispatch:
        args = args_or_state if isinstance(args_or_state, dict) else {}
        return tools.execute_tool(tool_name_or_message, **args)
    
    # Otherwise, try to detect tool from message (old convention)
    detected = tools.detect_tool(tool_name_or_message)
    
    if detected:
        return tools.execute_tool(detected)
    
    return None


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

# Proactive Prefetching (Phase 5)
from .prefetch import (
    PrefetchManager,
    PrefetchTask,
    PrefetchConfig,
    PrefetchStrategy,
    BackgroundExecutor,
    get_prefetch_manager,
)

__all__ = [
    # Base types
    "ToolResult",
    "ToolName",
    "TOOL_PATTERNS",
    
    # Registry
    "ToolRegistry",
    "get_registry",
    
    # Main interface
    "AgentTools",
    "get_agent_tools",
    "process_tool_request",
    
    # Pattern lists
    "ALL_TOOL_PATTERNS",
    
    # Knowledge base
    "CONCEPT_KNOWLEDGE",
    
    # Proactive Prefetching (Phase 5)
    "PrefetchManager",
    "PrefetchTask",
    "PrefetchConfig",
    "PrefetchStrategy",
    "BackgroundExecutor",
    "get_prefetch_manager",
    
    # Database Clients (Phase 2)
    "get_uniprot_client",
    "get_string_client",
    "get_kegg_client",
    "get_reactome_client",
    "get_pubmed_client",
    "get_clinvar_client",
    "DatabaseResult",
]
