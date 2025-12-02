"""
BioPipelines MCP Server
========================

Model Context Protocol server exposing BioPipelines capabilities
for integration with Claude Code, Cursor, and other MCP clients.

The server provides:
- Data discovery tools (ENCODE, GEO, TCGA search)
- Workflow generation (RNA-seq, ChIP-seq, etc.)
- Database queries (UniProt, STRING, KEGG, Reactome, PubMed, ClinVar)
- Job management (submit, status, cancel)
- Educational tools (concept explanation)

Usage:
    # Run via stdio (default for Claude Code integration)
    python -m workflow_composer.mcp.server
    
    # Run via HTTP (for development/testing)
    python -m workflow_composer.mcp.server --transport http --port 8080
"""

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """Definition of an MCP tool."""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable
    

@dataclass
class ResourceDefinition:
    """Definition of an MCP resource."""
    uri: str
    name: str
    description: str
    handler: Callable


class BioPipelinesMCPServer:
    """
    MCP Server exposing BioPipelines capabilities.
    
    This server implements the Model Context Protocol to expose
    BioPipelines tools to MCP-compatible clients like Claude Code.
    """
    
    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
        self.resources: Dict[str, ResourceDefinition] = {}
        self._setup_tools()
        self._setup_resources()
    
    def _setup_tools(self):
        """Register all BioPipelines tools."""
        
        # Data Discovery Tools
        self._register_tool(
            name="search_encode",
            description="Search ENCODE database for chromatin accessibility, histone modifications, transcription factor binding, and gene expression data.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search terms (cell type, target, assay, etc.)"
                    },
                    "assay_type": {
                        "type": "string",
                        "enum": ["ChIP-seq", "ATAC-seq", "RNA-seq", "WGBS", "Hi-C"],
                        "description": "Filter by assay type"
                    },
                    "organism": {
                        "type": "string",
                        "enum": ["human", "mouse"],
                        "default": "human",
                        "description": "Filter by organism"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum results to return"
                    }
                },
                "required": ["query"]
            },
            handler=self._handle_search_encode
        )
        
        self._register_tool(
            name="search_geo",
            description="Search NCBI GEO database for gene expression datasets and experiments.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search terms"
                    },
                    "organism": {
                        "type": "string",
                        "description": "Filter by organism (e.g., 'Homo sapiens', 'Mus musculus')"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum results"
                    }
                },
                "required": ["query"]
            },
            handler=self._handle_search_geo
        )
        
        # Workflow Generation Tools
        self._register_tool(
            name="create_workflow",
            description="Generate a bioinformatics analysis workflow (Nextflow or Snakemake) for RNA-seq, ChIP-seq, variant calling, etc.",
            parameters={
                "type": "object",
                "properties": {
                    "analysis_type": {
                        "type": "string",
                        "enum": ["rnaseq", "chipseq", "methylation", "variant", "atacseq"],
                        "description": "Type of analysis"
                    },
                    "organism": {
                        "type": "string",
                        "default": "human",
                        "description": "Reference organism"
                    },
                    "input_dir": {
                        "type": "string",
                        "description": "Input data directory"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory"
                    },
                    "workflow_engine": {
                        "type": "string",
                        "enum": ["nextflow", "snakemake"],
                        "default": "nextflow",
                        "description": "Workflow engine"
                    }
                },
                "required": ["analysis_type"]
            },
            handler=self._handle_create_workflow
        )
        
        self._register_tool(
            name="use_workflow_template",
            description="Generate a workflow from a pre-built template with customizable parameters.",
            parameters={
                "type": "object",
                "properties": {
                    "template_name": {
                        "type": "string",
                        "description": "Name of the template (e.g., 'basic_de', 'full_analysis', 'peak_calling')"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Template parameters"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory for generated workflow"
                    }
                },
                "required": ["template_name"]
            },
            handler=self._handle_use_template
        )
        
        # Database Query Tools
        self._register_tool(
            name="search_uniprot",
            description="Search UniProt database for protein sequences, annotations, and functions.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Gene name, protein name, or keywords"
                    },
                    "organism": {
                        "type": "string",
                        "default": "human",
                        "description": "Filter by organism"
                    },
                    "reviewed": {
                        "type": "boolean",
                        "default": True,
                        "description": "Only return reviewed (Swiss-Prot) entries"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 25,
                        "description": "Maximum results"
                    }
                },
                "required": ["query"]
            },
            handler=self._handle_search_uniprot
        )
        
        self._register_tool(
            name="get_protein_interactions",
            description="Get protein-protein interactions from STRING database.",
            parameters={
                "type": "object",
                "properties": {
                    "genes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of gene names"
                    },
                    "organism": {
                        "type": "string",
                        "default": "human",
                        "description": "Organism"
                    },
                    "score_threshold": {
                        "type": "integer",
                        "default": 400,
                        "description": "Minimum interaction score (0-1000)"
                    }
                },
                "required": ["genes"]
            },
            handler=self._handle_get_interactions
        )
        
        self._register_tool(
            name="get_functional_enrichment",
            description="Get Gene Ontology and pathway enrichment analysis for a gene list.",
            parameters={
                "type": "object",
                "properties": {
                    "genes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of gene names"
                    },
                    "organism": {
                        "type": "string",
                        "default": "human",
                        "description": "Organism"
                    }
                },
                "required": ["genes"]
            },
            handler=self._handle_get_enrichment
        )
        
        self._register_tool(
            name="search_kegg_pathways",
            description="Search KEGG database for metabolic and signaling pathways.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Pathway name or related terms"
                    },
                    "organism": {
                        "type": "string",
                        "default": "hsa",
                        "description": "KEGG organism code (hsa=human, mmu=mouse)"
                    }
                },
                "required": ["query"]
            },
            handler=self._handle_search_kegg
        )
        
        self._register_tool(
            name="search_pubmed",
            description="Search PubMed for scientific literature and publications.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search terms"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum results"
                    },
                    "sort": {
                        "type": "string",
                        "enum": ["relevance", "date"],
                        "default": "relevance",
                        "description": "Sort order"
                    }
                },
                "required": ["query"]
            },
            handler=self._handle_search_pubmed
        )
        
        self._register_tool(
            name="search_variants",
            description="Search ClinVar for variant pathogenicity information.",
            parameters={
                "type": "object",
                "properties": {
                    "gene": {
                        "type": "string",
                        "description": "Gene symbol"
                    },
                    "significance": {
                        "type": "string",
                        "enum": ["pathogenic", "likely_pathogenic", "uncertain_significance", "benign"],
                        "description": "Filter by clinical significance"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 25,
                        "description": "Maximum results"
                    }
                },
                "required": ["gene"]
            },
            handler=self._handle_search_variants
        )
        
        # Educational Tools
        self._register_tool(
            name="explain_concept",
            description="Explain a bioinformatics concept, tool, or method in detail.",
            parameters={
                "type": "object",
                "properties": {
                    "concept": {
                        "type": "string",
                        "description": "The concept to explain (e.g., 'DESeq2', 'peak calling', 'GWAS')"
                    },
                    "level": {
                        "type": "string",
                        "enum": ["beginner", "intermediate", "advanced"],
                        "default": "intermediate",
                        "description": "Explanation level"
                    }
                },
                "required": ["concept"]
            },
            handler=self._handle_explain_concept
        )
        
        # Job Management Tools
        self._register_tool(
            name="check_job_status",
            description="Check the status of a submitted workflow job.",
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Job identifier"
                    }
                },
                "required": ["job_id"]
            },
            handler=self._handle_check_job
        )
    
    def _setup_resources(self):
        """Register MCP resources."""
        
        self._register_resource(
            uri="biopipelines://skills",
            name="Available Skills",
            description="List of all available BioPipelines skills and capabilities",
            handler=self._handle_get_skills
        )
        
        self._register_resource(
            uri="biopipelines://templates",
            name="Workflow Templates",
            description="List of available pre-built workflow templates",
            handler=self._handle_get_templates
        )
        
        self._register_resource(
            uri="biopipelines://databases",
            name="Database Integrations",
            description="List of integrated biological databases",
            handler=self._handle_get_databases
        )
    
    def _register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable
    ):
        """Register a tool with the server."""
        self.tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler
        )
    
    def _register_resource(
        self,
        uri: str,
        name: str,
        description: str,
        handler: Callable
    ):
        """Register a resource with the server."""
        self.resources[uri] = ResourceDefinition(
            uri=uri,
            name=name,
            description=description,
            handler=handler
        )
    
    # Tool Handlers
    async def _handle_search_encode(self, **kwargs) -> Dict[str, Any]:
        """Handle ENCODE search."""
        try:
            # Import BioPipelines tools
            from workflow_composer.agents.tools import search_data
            
            result = search_data(
                source="ENCODE",
                query=kwargs.get("query", ""),
                assay_type=kwargs.get("assay_type"),
                organism=kwargs.get("organism", "human"),
                limit=kwargs.get("limit", 10)
            )
            
            return {
                "success": True,
                "content": self._format_search_results(result)
            }
        except Exception as e:
            logger.error(f"ENCODE search failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_search_geo(self, **kwargs) -> Dict[str, Any]:
        """Handle GEO search."""
        try:
            from workflow_composer.agents.tools import search_data
            
            result = search_data(
                source="GEO",
                query=kwargs.get("query", ""),
                organism=kwargs.get("organism"),
                limit=kwargs.get("limit", 10)
            )
            
            return {
                "success": True,
                "content": self._format_search_results(result)
            }
        except Exception as e:
            logger.error(f"GEO search failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_create_workflow(self, **kwargs) -> Dict[str, Any]:
        """Handle workflow creation."""
        try:
            from workflow_composer import BioPipelines
            
            bp = BioPipelines()
            workflow = bp.compose(
                analysis_type=kwargs.get("analysis_type"),
                organism=kwargs.get("organism", "human"),
                input_dir=kwargs.get("input_dir"),
                output_dir=kwargs.get("output_dir"),
                workflow_engine=kwargs.get("workflow_engine", "nextflow")
            )
            
            return {
                "success": True,
                "content": f"Generated {kwargs.get('analysis_type')} workflow:\n\n```{kwargs.get('workflow_engine', 'nextflow')}\n{workflow}\n```"
            }
        except Exception as e:
            logger.error(f"Workflow creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_use_template(self, **kwargs) -> Dict[str, Any]:
        """Handle workflow template usage."""
        try:
            # Add config to path
            sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "config"))
            from workflow_templates import get_template_engine
            
            engine = get_template_engine()
            template = engine.get_template(kwargs.get("template_name"))
            
            if not template:
                return {
                    "success": False,
                    "error": f"Template not found: {kwargs.get('template_name')}"
                }
            
            result = engine.generate(
                kwargs.get("template_name"),
                output_dir=kwargs.get("output_dir"),
                **kwargs.get("parameters", {})
            )
            
            return {
                "success": result.get("success", False),
                "content": f"Generated workflow from template '{kwargs.get('template_name')}'\n\nFiles:\n" + 
                          "\n".join(f"- {k}: {v}" for k, v in result.get("files", {}).items())
            }
        except Exception as e:
            logger.error(f"Template usage failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_search_uniprot(self, **kwargs) -> Dict[str, Any]:
        """Handle UniProt search."""
        try:
            from workflow_composer.agents.tools.databases import get_uniprot_client
            
            client = get_uniprot_client()
            result = client.search(
                query=kwargs.get("query", ""),
                organism=kwargs.get("organism", "human"),
                reviewed=kwargs.get("reviewed", True),
                limit=kwargs.get("limit", 25)
            )
            
            return {
                "success": result.success,
                "content": self._format_protein_results(result)
            }
        except Exception as e:
            logger.error(f"UniProt search failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_get_interactions(self, **kwargs) -> Dict[str, Any]:
        """Handle STRING interaction query."""
        try:
            from workflow_composer.agents.tools.databases import get_string_client
            
            client = get_string_client()
            species_map = {"human": 9606, "mouse": 10090, "rat": 10116}
            organism = kwargs.get("organism", "human")
            species = species_map.get(organism, 9606)
            
            result = client.search(
                identifiers=kwargs.get("genes", []),
                species=species,
                required_score=kwargs.get("score_threshold", 400)
            )
            
            return {
                "success": result.success,
                "content": self._format_interaction_results(result)
            }
        except Exception as e:
            logger.error(f"STRING search failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_get_enrichment(self, **kwargs) -> Dict[str, Any]:
        """Handle functional enrichment analysis."""
        try:
            from workflow_composer.agents.tools.databases import get_string_client
            
            client = get_string_client()
            species_map = {"human": 9606, "mouse": 10090, "rat": 10116}
            organism = kwargs.get("organism", "human")
            species = species_map.get(organism, 9606)
            
            result = client.get_enrichment(
                identifiers=kwargs.get("genes", []),
                species=species
            )
            
            return {
                "success": result.success,
                "content": self._format_enrichment_results(result)
            }
        except Exception as e:
            logger.error(f"Enrichment analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_search_kegg(self, **kwargs) -> Dict[str, Any]:
        """Handle KEGG pathway search."""
        try:
            from workflow_composer.agents.tools.databases import get_kegg_client
            
            client = get_kegg_client()
            result = client.search(
                query=kwargs.get("query", ""),
                organism=kwargs.get("organism", "hsa")
            )
            
            return {
                "success": result.success,
                "content": self._format_pathway_results(result)
            }
        except Exception as e:
            logger.error(f"KEGG search failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_search_pubmed(self, **kwargs) -> Dict[str, Any]:
        """Handle PubMed search."""
        try:
            from workflow_composer.agents.tools.databases import get_pubmed_client
            
            client = get_pubmed_client()
            result = client.search(
                query=kwargs.get("query", ""),
                max_results=kwargs.get("limit", 10),
                sort=kwargs.get("sort", "relevance")
            )
            
            return {
                "success": result.success,
                "content": self._format_pubmed_results(result)
            }
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_search_variants(self, **kwargs) -> Dict[str, Any]:
        """Handle ClinVar variant search."""
        try:
            from workflow_composer.agents.tools.databases import get_clinvar_client
            
            client = get_clinvar_client()
            result = client.search_by_gene(
                gene=kwargs.get("gene", ""),
                significance=kwargs.get("significance"),
                limit=kwargs.get("limit", 25)
            )
            
            return {
                "success": result.success,
                "content": self._format_variant_results(result)
            }
        except Exception as e:
            logger.error(f"ClinVar search failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_explain_concept(self, **kwargs) -> Dict[str, Any]:
        """Handle concept explanation."""
        try:
            from workflow_composer.agents.tools import explain_concept
            
            result = explain_concept(
                concept=kwargs.get("concept", ""),
                level=kwargs.get("level", "intermediate")
            )
            
            return {
                "success": True,
                "content": result.get("explanation", "No explanation available")
            }
        except Exception as e:
            logger.error(f"Concept explanation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_check_job(self, **kwargs) -> Dict[str, Any]:
        """Handle job status check."""
        try:
            from workflow_composer.agents.tools import check_job_status
            
            result = check_job_status(job_id=kwargs.get("job_id", ""))
            
            return {
                "success": True,
                "content": f"Job {kwargs.get('job_id')}:\n" +
                          f"Status: {result.get('status', 'unknown')}\n" +
                          f"Progress: {result.get('progress', 'N/A')}"
            }
        except Exception as e:
            logger.error(f"Job status check failed: {e}")
            return {"success": False, "error": str(e)}
    
    # Resource Handlers
    async def _handle_get_skills(self) -> str:
        """Get available skills."""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "config"))
            from skills import get_skill_registry
            
            registry = get_skill_registry()
            
            text = "# Available BioPipelines Skills\n\n"
            
            categories = ["data_discovery", "workflow_generation", "job_management", "education"]
            for category in categories:
                skills = registry.get_skills_by_category(category)
                if skills:
                    text += f"## {category.replace('_', ' ').title()}\n\n"
                    for skill in skills:
                        text += f"- **{skill.name}**: {skill.description[:100]}...\n"
                    text += "\n"
            
            return text
        except Exception as e:
            return f"Failed to load skills: {e}"
    
    async def _handle_get_templates(self) -> str:
        """Get available workflow templates."""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "config"))
            from workflow_templates import get_template_engine
            
            engine = get_template_engine()
            templates = engine.list_templates()
            
            text = "# Available Workflow Templates\n\n"
            
            for template in templates:
                text += f"## {template.display_name}\n"
                text += f"{template.description}\n\n"
                text += f"**Category**: {template.category}\n"
                text += f"**Tags**: {', '.join(template.tags)}\n"
                text += f"**Engine**: {template.engine}\n\n"
            
            return text
        except Exception as e:
            return f"Failed to load templates: {e}"
    
    async def _handle_get_databases(self) -> str:
        """Get available database integrations."""
        databases = """# Integrated Biological Databases

## UniProt
- Protein sequences, annotations, and functions
- Swiss-Prot (reviewed) and TrEMBL entries
- Gene Ontology annotations

## STRING
- Protein-protein interactions
- Functional enrichment analysis
- Network visualization

## KEGG
- Metabolic pathways
- Signaling pathways
- Disease pathways

## Reactome
- Biological pathways
- Reaction networks
- Gene set analysis

## PubMed
- Scientific literature search
- Citation information
- Abstract retrieval

## ClinVar
- Variant pathogenicity
- Clinical significance
- Disease associations
"""
        return databases
    
    # Formatting helpers
    def _format_search_results(self, result) -> str:
        """Format search results."""
        if not hasattr(result, 'success') or not result.success:
            return f"Search failed: {getattr(result, 'message', 'Unknown error')}"
        
        text = f"Found {result.count} results:\n\n"
        for item in result.data[:10]:
            if isinstance(item, dict):
                text += f"- **{item.get('id', 'N/A')}**: {item.get('title', item.get('name', 'N/A'))}\n"
            else:
                text += f"- {item}\n"
        
        return text
    
    def _format_protein_results(self, result) -> str:
        """Format UniProt results."""
        if not result.success:
            return f"Search failed: {result.message}"
        
        text = f"Found {result.count} proteins:\n\n"
        for protein in result.data[:10]:
            accession = protein.get("primaryAccession", "N/A")
            name = "N/A"
            if "proteinDescription" in protein:
                rec_name = protein["proteinDescription"].get("recommendedName", {})
                if "fullName" in rec_name:
                    name = rec_name["fullName"].get("value", "N/A")
            
            gene = "N/A"
            if protein.get("genes"):
                gene_data = protein["genes"][0].get("geneName", {})
                gene = gene_data.get("value", "N/A")
            
            text += f"- **{accession}** ({gene}): {name}\n"
        
        return text
    
    def _format_interaction_results(self, result) -> str:
        """Format STRING interaction results."""
        if not result.success:
            return f"Search failed: {result.message}"
        
        text = f"Found {result.count} interactions:\n\n"
        for interaction in result.data[:20]:
            p1 = interaction.get("preferredName_A", interaction.get("stringId_A", "?"))
            p2 = interaction.get("preferredName_B", interaction.get("stringId_B", "?"))
            score = interaction.get("score", 0)
            text += f"- {p1} ↔ {p2} (score: {score})\n"
        
        return text
    
    def _format_enrichment_results(self, result) -> str:
        """Format enrichment results."""
        if not result.success:
            return f"Enrichment failed: {result.message}"
        
        text = f"Found {result.count} enriched terms:\n\n"
        
        # Group by category
        categories = {}
        for term in result.data[:50]:
            cat = term.get("category", "Other")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(term)
        
        for cat, terms in categories.items():
            text += f"## {cat}\n"
            for term in terms[:5]:
                desc = term.get("description", term.get("term", "N/A"))
                pvalue = term.get("p_value", term.get("fdr", 1.0))
                try:
                    text += f"- {desc} (p={float(pvalue):.2e})\n"
                except:
                    text += f"- {desc}\n"
            text += "\n"
        
        return text
    
    def _format_pathway_results(self, result) -> str:
        """Format KEGG pathway results."""
        if not result.success:
            return f"Search failed: {result.message}"
        
        text = f"Found {result.count} pathways:\n\n"
        for pathway in result.data[:15]:
            pid = pathway.get("id", "N/A")
            name = pathway.get("name", "N/A")
            text += f"- **{pid}**: {name}\n"
        
        return text
    
    def _format_pubmed_results(self, result) -> str:
        """Format PubMed results."""
        if not result.success:
            return f"Search failed: {result.message}"
        
        text = f"Found {result.count} articles:\n\n"
        for article in result.data[:10]:
            pmid = article.get("pmid", article.get("id", "N/A"))
            title = article.get("title", "N/A")
            authors = article.get("authors", [])
            author_str = authors[0] if authors else "Unknown"
            if len(authors) > 1:
                author_str += " et al."
            year = article.get("year", "")
            
            text += f"- **PMID:{pmid}** ({year}) {author_str}: {title[:100]}...\n"
        
        return text
    
    def _format_variant_results(self, result) -> str:
        """Format ClinVar variant results."""
        if not result.success:
            return f"Search failed: {result.message}"
        
        text = f"Found {result.count} variants:\n\n"
        for variant in result.data[:15]:
            vid = variant.get("id", variant.get("variation_id", "N/A"))
            name = variant.get("name", variant.get("title", "N/A"))
            sig = variant.get("clinical_significance", variant.get("significance", "N/A"))
            text += f"- **{vid}**: {name} [{sig}]\n"
        
        return text
    
    # Protocol methods
    def get_tools_list(self) -> List[Dict[str, Any]]:
        """Get list of tools in MCP format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.parameters
            }
            for tool in self.tools.values()
        ]
    
    def get_resources_list(self) -> List[Dict[str, Any]]:
        """Get list of resources in MCP format."""
        return [
            {
                "uri": res.uri,
                "name": res.name,
                "description": res.description
            }
            for res in self.resources.values()
        ]
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool by name."""
        if name not in self.tools:
            return {"success": False, "error": f"Unknown tool: {name}"}
        
        tool = self.tools[name]
        return await tool.handler(**arguments)
    
    async def read_resource(self, uri: str) -> str:
        """Read a resource by URI."""
        if uri not in self.resources:
            return f"Unknown resource: {uri}"
        
        resource = self.resources[uri]
        return await resource.handler()
    
    # Server run methods
    async def run_stdio(self):
        """Run server using stdio transport (for Claude Code integration)."""
        logger.info("Starting BioPipelines MCP Server (stdio)")
        
        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                
                request = json.loads(line.strip())
                response = await self._handle_request(request)
                
                print(json.dumps(response), flush=True)
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
            except Exception as e:
                logger.error(f"Error handling request: {e}")
    
    async def run_http(self, host: str = "0.0.0.0", port: int = 8080):
        """Run server using HTTP transport (for development/testing)."""
        try:
            from aiohttp import web
        except ImportError:
            logger.error("aiohttp required for HTTP transport. Install with: pip install aiohttp")
            return
        
        async def handle_request(request):
            data = await request.json()
            response = await self._handle_request(data)
            return web.json_response(response)
        
        async def handle_tools(request):
            return web.json_response({"tools": self.get_tools_list()})
        
        async def handle_resources(request):
            return web.json_response({"resources": self.get_resources_list()})
        
        app = web.Application()
        app.router.add_post("/", handle_request)
        app.router.add_get("/tools", handle_tools)
        app.router.add_get("/resources", handle_resources)
        
        logger.info(f"Starting BioPipelines MCP Server (HTTP) on {host}:{port}")
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        # Keep running
        while True:
            await asyncio.sleep(3600)
    
    async def _handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an MCP request."""
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")
        
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": "biopipelines",
                        "version": "1.0.0"
                    },
                    "capabilities": {
                        "tools": {},
                        "resources": {}
                    }
                }
            }
        
        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": self.get_tools_list()}
            }
        
        elif method == "tools/call":
            name = params.get("name", "")
            arguments = params.get("arguments", {})
            result = await self.call_tool(name, arguments)
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": result.get("content", result.get("error", "No result"))
                        }
                    ],
                    "isError": not result.get("success", False)
                }
            }
        
        elif method == "resources/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"resources": self.get_resources_list()}
            }
        
        elif method == "resources/read":
            uri = params.get("uri", "")
            content = await self.read_resource(uri)
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "text/markdown",
                            "text": content
                        }
                    ]
                }
            }
        
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }


def create_server() -> BioPipelinesMCPServer:
    """Create a new MCP server instance."""
    return BioPipelinesMCPServer()


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BioPipelines MCP Server")
    parser.add_argument(
        "--transport", 
        choices=["stdio", "http"], 
        default="stdio",
        help="Transport method"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="HTTP port (for http transport)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="HTTP host (for http transport)"
    )
    
    args = parser.parse_args()
    
    server = create_server()
    
    if args.transport == "stdio":
        await server.run_stdio()
    else:
        await server.run_http(host=args.host, port=args.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
