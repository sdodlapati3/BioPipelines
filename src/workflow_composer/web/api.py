#!/usr/bin/env python3
"""
BioPipelines - FastAPI REST Backend
===================================

RESTful API for the BioPipelines Workflow Composer.
Provides endpoints for:
- Workflow generation
- Tool/module search
- LLM chat
- Status monitoring

Usage:
    uvicorn workflow_composer.web.api:app --reload
    # or
    python -m workflow_composer.web.api
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

# Import workflow composer components
try:
    from workflow_composer import Composer
    from workflow_composer.core import ToolSelector, ModuleMapper, AnalysisType
    from workflow_composer.llm import get_llm, check_providers, Message
    COMPOSER_AVAILABLE = True
except ImportError:
    COMPOSER_AVAILABLE = False


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="BioPipelines API",
    description="REST API for AI-powered bioinformatics workflow generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_DIR = Path(__file__).parent.parent.parent.parent
GENERATED_DIR = BASE_DIR / "generated_workflows"
GENERATED_DIR.mkdir(exist_ok=True)


# ============================================================================
# Pydantic Models
# ============================================================================

class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Chat history")
    provider: str = Field(default="openai", description="LLM provider")
    model: Optional[str] = Field(default=None, description="Model name")
    stream: bool = Field(default=False, description="Enable streaming")


class ChatResponse(BaseModel):
    content: str
    provider: str
    model: str
    tokens_used: Optional[int] = None


class GenerateRequest(BaseModel):
    description: str = Field(..., description="Natural language workflow description")
    provider: str = Field(default="openai", description="LLM provider")
    model: Optional[str] = Field(default=None, description="Model name")
    name: Optional[str] = Field(default=None, description="Workflow name")
    output_format: str = Field(default="nextflow", description="Output format")


class GenerateResponse(BaseModel):
    workflow_id: str
    name: str
    analysis_type: str
    tools_used: List[str]
    modules_used: List[str]
    output_dir: str
    files: List[str]


class ToolSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    container: Optional[str] = Field(default=None, description="Filter by container")
    limit: int = Field(default=20, description="Max results")


class ToolInfo(BaseModel):
    name: str
    container: str
    category: Optional[str]
    description: Optional[str]
    score: Optional[float] = None


class IntentParseRequest(BaseModel):
    description: str = Field(..., description="Natural language description")
    provider: str = Field(default="openai", description="LLM provider")


class IntentResponse(BaseModel):
    analysis_type: str
    organism: Optional[str]
    genome_build: Optional[str]
    data_type: Optional[str]
    has_comparison: bool
    confidence: float


class ProviderStatus(BaseModel):
    name: str
    available: bool
    default_model: Optional[str]


class StatsResponse(BaseModel):
    tools: int
    modules: int
    containers: int
    analysis_types: int
    providers: List[ProviderStatus]


# ============================================================================
# App State
# ============================================================================

class AppState:
    """Application state manager."""
    
    def __init__(self):
        self._composers: Dict[str, Composer] = {}
        self._tool_selector: Optional[ToolSelector] = None
        self._module_mapper: Optional[ModuleMapper] = None
    
    def get_composer(self, provider: str, model: Optional[str] = None) -> Composer:
        """Get or create a composer for a specific provider."""
        key = f"{provider}:{model or 'default'}"
        
        if key not in self._composers:
            llm = get_llm(provider, model=model) if model else get_llm(provider)
            self._composers[key] = Composer(llm=llm)
        
        return self._composers[key]
    
    @property
    def tool_selector(self) -> Optional[ToolSelector]:
        if self._tool_selector is None and COMPOSER_AVAILABLE:
            try:
                # Try to get from any composer
                for composer in self._composers.values():
                    if composer.tool_selector:
                        self._tool_selector = composer.tool_selector
                        break
            except:
                pass
        return self._tool_selector
    
    @property
    def module_mapper(self) -> Optional[ModuleMapper]:
        if self._module_mapper is None and COMPOSER_AVAILABLE:
            try:
                for composer in self._composers.values():
                    if composer.module_mapper:
                        self._module_mapper = composer.module_mapper
                        break
            except:
                pass
        return self._module_mapper


state = AppState()


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """API root - returns basic info."""
    return {
        "name": "BioPipelines API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "composer_available": COMPOSER_AVAILABLE,
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/stats", response_model=StatsResponse, tags=["General"])
async def get_stats():
    """Get system statistics."""
    available_providers = check_providers() if COMPOSER_AVAILABLE else {}
    
    providers = [
        ProviderStatus(
            name=name,
            available=is_available,
            default_model=get_default_model(name) if is_available else None
        )
        for name, is_available in available_providers.items()
    ]
    
    tools_count = len(state.tool_selector.tools) if state.tool_selector else 0
    modules_count = len(state.module_mapper.modules) if state.module_mapper else 0
    
    return StatsResponse(
        tools=tools_count,
        modules=modules_count,
        containers=12,
        analysis_types=len(AnalysisType) if COMPOSER_AVAILABLE else 38,
        providers=providers,
    )


def get_default_model(provider: str) -> str:
    """Get default model for a provider."""
    defaults = {
        "openai": "gpt-4o",
        "anthropic": "claude-3-sonnet",
        "ollama": "llama3.1",
        "vllm": "mistral-7b",
    }
    return defaults.get(provider, "unknown")


# ============================================================================
# Chat Endpoints
# ============================================================================

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Chat with the AI assistant.
    
    Send a conversation history and get a response.
    """
    if not COMPOSER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Workflow composer not available")
    
    available = check_providers()
    if not available.get(request.provider):
        raise HTTPException(
            status_code=400, 
            detail=f"Provider '{request.provider}' not available. Available: {[k for k, v in available.items() if v]}"
        )
    
    try:
        composer = state.get_composer(request.provider, request.model)
        
        # Convert messages
        messages = [
            Message(role=m.role, content=m.content)
            for m in request.messages
        ]
        
        # Get response
        response = composer.llm.chat(messages)
        
        return ChatResponse(
            content=response.content,
            provider=request.provider,
            model=request.model or get_default_model(request.provider),
            tokens_used=response.usage.get("total_tokens") if hasattr(response, "usage") else None,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest):
    """
    Chat with streaming response.
    
    Returns a Server-Sent Events stream.
    """
    if not COMPOSER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Workflow composer not available")
    
    async def generate():
        try:
            composer = state.get_composer(request.provider, request.model)
            messages = [Message(role=m.role, content=m.content) for m in request.messages]
            
            # Stream if supported
            if hasattr(composer.llm, 'chat_stream'):
                for chunk in composer.llm.chat_stream(messages):
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
            else:
                response = composer.llm.chat(messages)
                yield f"data: {json.dumps({'content': response.content})}\n\n"
            
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


# ============================================================================
# Workflow Generation Endpoints
# ============================================================================

@app.post("/generate", response_model=GenerateResponse, tags=["Workflow"])
async def generate_workflow(request: GenerateRequest):
    """
    Generate a bioinformatics workflow from natural language description.
    """
    if not COMPOSER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Workflow composer not available")
    
    available = check_providers()
    if not available.get(request.provider):
        raise HTTPException(status_code=400, detail=f"Provider '{request.provider}' not available")
    
    try:
        composer = state.get_composer(request.provider, request.model)
        
        # Generate workflow ID
        workflow_name = request.name or f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        workflow_id = workflow_name.replace(" ", "_").lower()
        output_dir = GENERATED_DIR / workflow_id
        
        # Generate
        workflow = composer.generate(request.description, output_dir=str(output_dir))
        
        # Get list of generated files
        files = [f.name for f in output_dir.iterdir()] if output_dir.exists() else []
        
        return GenerateResponse(
            workflow_id=workflow_id,
            name=workflow.name if hasattr(workflow, "name") else workflow_name,
            analysis_type=workflow.analysis_type.value if hasattr(workflow, "analysis_type") else "unknown",
            tools_used=workflow.tools_used if hasattr(workflow, "tools_used") else [],
            modules_used=workflow.modules_used if hasattr(workflow, "modules_used") else [],
            output_dir=str(output_dir),
            files=files,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/parse-intent", response_model=IntentResponse, tags=["Workflow"])
async def parse_intent(request: IntentParseRequest):
    """
    Parse a natural language description into structured analysis intent.
    """
    if not COMPOSER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Workflow composer not available")
    
    try:
        composer = state.get_composer(request.provider)
        intent = composer.parse_intent(request.description)
        
        return IntentResponse(
            analysis_type=intent.analysis_type.value,
            organism=intent.organism,
            genome_build=intent.genome_build,
            data_type=intent.data_type,
            has_comparison=intent.has_comparison,
            confidence=intent.confidence,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workflows", tags=["Workflow"])
async def list_workflows():
    """List all generated workflows."""
    workflows = []
    
    for d in GENERATED_DIR.iterdir():
        if d.is_dir():
            workflows.append({
                "id": d.name,
                "created": datetime.fromtimestamp(d.stat().st_ctime).isoformat(),
                "files": [f.name for f in d.iterdir()],
            })
    
    return {"workflows": sorted(workflows, key=lambda x: x["created"], reverse=True)}


@app.get("/workflows/{workflow_id}", tags=["Workflow"])
async def get_workflow(workflow_id: str):
    """Get details of a specific workflow."""
    workflow_dir = GENERATED_DIR / workflow_id
    
    if not workflow_dir.exists():
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    files = {}
    for f in workflow_dir.iterdir():
        if f.is_file() and f.suffix in [".nf", ".config", ".json", ".yaml", ".md"]:
            files[f.name] = f.read_text()[:5000]  # Limit content size
    
    return {
        "id": workflow_id,
        "created": datetime.fromtimestamp(workflow_dir.stat().st_ctime).isoformat(),
        "files": files,
    }


@app.get("/workflows/{workflow_id}/download", tags=["Workflow"])
async def download_workflow(workflow_id: str):
    """Download a workflow as a zip file."""
    workflow_dir = GENERATED_DIR / workflow_id
    
    if not workflow_dir.exists():
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    zip_path = GENERATED_DIR / f"{workflow_id}.zip"
    shutil.make_archive(str(zip_path.with_suffix('')), 'zip', workflow_dir)
    
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{workflow_id}.zip",
    )


# ============================================================================
# Tool/Module Endpoints
# ============================================================================

@app.post("/tools/search", response_model=List[ToolInfo], tags=["Tools"])
async def search_tools(request: ToolSearchRequest):
    """Search for bioinformatics tools."""
    if not state.tool_selector:
        # Return demo data
        return [
            ToolInfo(name="fastqc", container="base", category="QC", description="Quality control"),
            ToolInfo(name="multiqc", container="base", category="QC", description="Report aggregation"),
        ]
    
    try:
        results = state.tool_selector.fuzzy_search(
            request.query,
            limit=request.limit,
            container=request.container,
        )
        
        return [
            ToolInfo(
                name=m.tool.name,
                container=m.tool.container,
                category=m.tool.category,
                description=m.tool.description[:200] if m.tool.description else None,
                score=m.score,
            )
            for m in results
        ]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools/containers", tags=["Tools"])
async def list_containers():
    """List all available containers."""
    if not state.tool_selector:
        return {"containers": ["base", "rna-seq", "dna-seq", "chip-seq", "atac-seq"]}
    
    containers = list(set(t.container for t in state.tool_selector.tools.values()))
    return {"containers": sorted(containers)}


@app.get("/modules", tags=["Modules"])
async def list_modules():
    """List all available Nextflow modules."""
    if not state.module_mapper:
        return {
            "modules": ["fastqc", "multiqc", "star", "bwa"],
            "by_category": {"qc": ["fastqc", "multiqc"], "alignment": ["star", "bwa"]},
        }
    
    try:
        modules = state.module_mapper.list_modules()
        by_category = state.module_mapper.list_by_category()
        
        return {
            "modules": modules,
            "by_category": by_category,
            "total": len(modules),
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/modules/{module_name}", tags=["Modules"])
async def get_module(module_name: str):
    """Get details of a specific module."""
    if not state.module_mapper:
        raise HTTPException(status_code=503, detail="Module mapper not available")
    
    module = state.module_mapper.get_module(module_name)
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    
    return {
        "name": module.name,
        "path": str(module.path),
        "inputs": module.inputs,
        "outputs": module.outputs,
        "parameters": module.parameters,
    }


# ============================================================================
# Analysis Types
# ============================================================================

@app.get("/analysis-types", tags=["Analysis"])
async def list_analysis_types():
    """List all supported analysis types."""
    if not COMPOSER_AVAILABLE:
        return {"types": ["rnaseq", "chipseq", "dnaseq", "scrnaseq"]}
    
    return {
        "types": [at.value for at in AnalysisType],
        "total": len(AnalysisType),
    }


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the FastAPI server."""
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="BioPipelines REST API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║        🧬 BioPipelines - REST API                                ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  API Docs:  http://{args.host}:{args.port}/docs                            ║
║  ReDoc:     http://{args.host}:{args.port}/redoc                           ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "workflow_composer.web.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
