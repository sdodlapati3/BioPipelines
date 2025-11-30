"""
Workflow Routes
==============

Endpoints for workflow generation and management.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Header, Query
from pydantic import BaseModel, Field

from workflow_composer.auth.models import AuthResult
from workflow_composer.auth.dependencies import require_api_key, optional_api_key

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class WorkflowGenerateRequest(BaseModel):
    """Request to generate a workflow from natural language."""
    
    query: str = Field(..., description="Natural language query describing the workflow")
    organism: Optional[str] = Field(None, description="Target organism (e.g., 'human', 'mouse')")
    data_types: Optional[List[str]] = Field(None, description="Data types to include")
    output_format: str = Field("nextflow", description="Output format: 'nextflow' or 'snakemake'")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Perform differential expression analysis on RNA-seq data comparing treated vs control samples",
                "organism": "human",
                "output_format": "nextflow"
            }
        }


class WorkflowGenerateResponse(BaseModel):
    """Response from workflow generation."""
    
    workflow_id: str = Field(..., description="Generated workflow ID")
    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow description")
    pipeline_type: str = Field(..., description="Detected pipeline type")
    tools: List[str] = Field(..., description="Tools included in workflow")
    code: str = Field(..., description="Generated workflow code")
    format: str = Field(..., description="Output format used")


class WorkflowListResponse(BaseModel):
    """Response listing workflows."""
    
    workflows: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/workflows/generate", response_model=WorkflowGenerateResponse)
async def generate_workflow(
    request: WorkflowGenerateRequest,
    auth: AuthResult = Depends(require_api_key),
    authorization: Optional[str] = Header(None),
):
    """
    Generate a workflow from a natural language query.
    
    Uses the AI agent to:
    1. Parse the query and identify pipeline type
    2. Select appropriate tools
    3. Generate workflow code
    """
    if not auth.success:
        raise HTTPException(status_code=401, detail=auth.error)
    
    # Store query in request state for RAG middleware
    from starlette.requests import Request
    
    try:
        # Try to use existing composer components
        from workflow_composer.core.query_parser import IntentParser, ParsedIntent
        from workflow_composer.core.tool_selector import ToolSelector
        from workflow_composer.core.workflow_generator import WorkflowGenerator
        from workflow_composer.core.module_mapper import ModuleMapper
        from pathlib import Path
        import uuid
        
        # Get paths
        project_root = Path(__file__).parent.parent.parent.parent.parent
        catalog_path = project_root / "data" / "tool_catalog"
        patterns_path = project_root / "config"
        
        # For LLM-based parsing, we need an LLM adapter
        # For now, use rule-based fallback via static intent creation
        intent = _create_intent_from_request(request)
        
        # Select tools
        if catalog_path.exists():
            selector = ToolSelector(str(catalog_path))
            tool_matches = selector.find_tools(intent.analysis_type.value)
            tools = [m.tool for m in tool_matches]
        else:
            tools = []
        
        # Map to modules
        mapper = ModuleMapper()
        modules = mapper.map_tools_to_modules(intent, tools) if tools else []
        
        # Generate workflow
        generator = WorkflowGenerator(str(patterns_path))
        workflow = generator.generate(intent, modules)
        
        workflow_id = str(uuid.uuid4())[:8]
        
        # Record for RAG learning
        try:
            from workflow_composer.agents.rag import get_rag_orchestrator
            rag = get_rag_orchestrator()
            rag.record_execution(
                query=request.query,
                tool_name="workflow_generate",
                tool_args={
                    "organism": request.organism,
                    "format": request.output_format,
                    "analysis_type": intent.analysis_type.value,
                },
                success=True,
                user_id=str(auth.user.id) if auth.user else None,
            )
        except Exception as e:
            logger.debug(f"RAG recording failed: {e}")
        
        return WorkflowGenerateResponse(
            workflow_id=workflow_id,
            name=workflow.name,
            description=f"Generated workflow for {intent.analysis_type.value}",
            pipeline_type=intent.analysis_type.value,
            tools=[t.name for t in tools] if tools else [],
            code=workflow.main_nf,
            format=request.output_format,
        )
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        raise HTTPException(
            status_code=503,
            detail="Workflow generation service unavailable"
        )
    except Exception as e:
        logger.exception(f"Workflow generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Workflow generation failed: {str(e)}"
        )


def _create_intent_from_request(request: WorkflowGenerateRequest):
    """Create a ParsedIntent from the API request using rule-based matching."""
    from workflow_composer.core.query_parser import ParsedIntent, AnalysisType
    
    query_lower = request.query.lower()
    
    # Rule-based analysis type detection
    if "rna-seq" in query_lower or "rnaseq" in query_lower:
        if "differential" in query_lower or "de " in query_lower or "deseq" in query_lower:
            analysis_type = AnalysisType.RNA_SEQ_DE
        elif "single" in query_lower or "scrna" in query_lower:
            analysis_type = AnalysisType.SCRNA_SEQ
        else:
            analysis_type = AnalysisType.RNA_SEQ_BASIC
    elif "chip-seq" in query_lower or "chipseq" in query_lower:
        analysis_type = AnalysisType.CHIP_SEQ
    elif "atac-seq" in query_lower or "atacseq" in query_lower:
        analysis_type = AnalysisType.ATAC_SEQ
    elif "variant" in query_lower:
        if "somatic" in query_lower:
            analysis_type = AnalysisType.SOMATIC_VARIANT_CALLING
        elif "structural" in query_lower:
            analysis_type = AnalysisType.STRUCTURAL_VARIANT
        else:
            analysis_type = AnalysisType.WGS_VARIANT_CALLING
    elif "methylat" in query_lower or "bisulfite" in query_lower:
        analysis_type = AnalysisType.BISULFITE_SEQ
    elif "metagenomic" in query_lower or "16s" in query_lower:
        if "16s" in query_lower:
            analysis_type = AnalysisType.AMPLICON_16S
        else:
            analysis_type = AnalysisType.METAGENOMICS_PROFILING
    elif "long read" in query_lower or "nanopore" in query_lower or "pacbio" in query_lower:
        analysis_type = AnalysisType.LONG_READ_ASSEMBLY
    else:
        analysis_type = AnalysisType.CUSTOM
    
    # Detect organism
    organism = request.organism or "human"  # Default to human
    if "mouse" in query_lower or "mus musculus" in query_lower:
        organism = "mouse"
    elif "zebrafish" in query_lower or "danio" in query_lower:
        organism = "zebrafish"
    elif "drosophila" in query_lower or "fruit fly" in query_lower:
        organism = "drosophila"
    
    # Detect if comparison study
    has_comparison = any(word in query_lower for word in [
        "vs", "versus", "compare", "differential", "between", "control"
    ])
    
    return ParsedIntent(
        analysis_type=analysis_type,
        analysis_type_raw=analysis_type.value,
        confidence=0.8,
        data_type="fastq",
        paired_end=True,
        organism=organism,
        genome_build="hg38" if organism == "human" else "mm10" if organism == "mouse" else "",
        has_comparison=has_comparison,
        conditions=["treated", "control"] if has_comparison else [],
        original_query=request.query,
    )


@router.get("/workflows", response_model=WorkflowListResponse)
async def list_workflows(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    auth: AuthResult = Depends(optional_api_key),
):
    """
    List generated workflows.
    
    If authenticated, shows user's workflows.
    If anonymous, shows public/example workflows only.
    """
    # For now, return empty list until database is integrated
    return WorkflowListResponse(
        workflows=[],
        total=0,
        page=page,
        page_size=page_size,
    )


@router.get("/workflows/{workflow_id}")
async def get_workflow(
    workflow_id: str,
    auth: AuthResult = Depends(optional_api_key),
):
    """
    Get a specific workflow by ID.
    """
    # Placeholder - will be implemented with database
    raise HTTPException(status_code=404, detail="Workflow not found")


@router.delete("/workflows/{workflow_id}")
async def delete_workflow(
    workflow_id: str,
    auth: AuthResult = Depends(require_api_key),
):
    """
    Delete a workflow.
    
    Only the owner can delete their workflows.
    """
    if not auth.success:
        raise HTTPException(status_code=401, detail=auth.error)
    
    # Placeholder - will be implemented with database
    raise HTTPException(status_code=404, detail="Workflow not found")
