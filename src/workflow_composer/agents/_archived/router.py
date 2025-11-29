"""
Agentic Router for BioPipelines
================================

A tool-calling LLM router that replaces regex-based intent detection
with a small, fast language model.

Supports:
- Local vLLM inference (T4 GPU compatible)
- Cloud fallback (Lightning.ai, OpenAI)
- OpenAI-compatible function calling
- Graceful degradation to regex

Usage:
    router = AgentRouter()
    result = await router.route("scan data in /scratch/mydata")
    # Returns: {"tool": "scan_data", "arguments": {"path": "/scratch/mydata"}}
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Definitions (OpenAI Function Calling Format)
# =============================================================================

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "scan_data",
            "description": "Scan a directory for FASTQ, BAM, or other sequencing files. Use this when the user wants to find or discover data files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The absolute path to the directory to scan"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to scan subdirectories",
                        "default": True
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_databases",
            "description": "Search remote databases like ENCODE, GEO, or SRA for public datasets. Use this when the user wants to find external data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query terms"
                    },
                    "organism": {
                        "type": "string",
                        "description": "Organism name (e.g., 'human', 'mouse')"
                    },
                    "assay_type": {
                        "type": "string",
                        "description": "Type of assay (e.g., 'RNA-seq', 'ChIP-seq', 'ATAC-seq')"
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Databases to search",
                        "default": ["ENCODE", "GEO"]
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_references",
            "description": "Check if reference genome files are available for an organism",
            "parameters": {
                "type": "object",
                "properties": {
                    "organism": {
                        "type": "string",
                        "description": "Organism name (e.g., 'human', 'mouse')"
                    },
                    "assembly": {
                        "type": "string",
                        "description": "Genome assembly (e.g., 'GRCh38', 'mm10')"
                    }
                },
                "required": ["organism"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_workflow",
            "description": "Generate a bioinformatics workflow/pipeline. Use this when the user wants to create, build, or generate a pipeline.",
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis_type": {
                        "type": "string",
                        "enum": ["rna_seq", "chip_seq", "atac_seq", "methylation", "variant_calling", "metagenomics", "single_cell", "long_read", "hi_c"],
                        "description": "Type of analysis"
                    },
                    "organism": {
                        "type": "string",
                        "description": "Organism name"
                    },
                    "data_path": {
                        "type": "string",
                        "description": "Path to input data"
                    },
                    "comparison": {
                        "type": "string",
                        "description": "Groups to compare for differential analysis"
                    },
                    "tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific tools to use"
                    }
                },
                "required": ["analysis_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_job",
            "description": "Submit a workflow to run on SLURM cluster or locally",
            "parameters": {
                "type": "object",
                "properties": {
                    "workflow_name": {
                        "type": "string",
                        "description": "Name of the workflow to submit"
                    },
                    "profile": {
                        "type": "string",
                        "enum": ["slurm", "local", "docker"],
                        "description": "Execution profile",
                        "default": "slurm"
                    },
                    "resume": {
                        "type": "boolean",
                        "description": "Resume from previous run",
                        "default": False
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_job_status",
            "description": "Check the status of running or completed jobs",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Job ID to check (optional, shows all if not provided)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_logs",
            "description": "Get logs from a job for debugging",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Job ID to get logs for"
                    },
                    "tail_lines": {
                        "type": "integer",
                        "description": "Number of lines to show",
                        "default": 50
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "diagnose_error",
            "description": "Diagnose and explain errors from failed jobs",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Job ID to diagnose"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_job",
            "description": "Cancel a running job",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Job ID to cancel"
                    }
                },
                "required": ["job_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_workflows",
            "description": "List available or generated workflows",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


# =============================================================================
# Router Result Types
# =============================================================================

class RoutingStrategy(Enum):
    """How the routing was determined."""
    LOCAL_LLM = "local_llm"
    CLOUD_LLM = "cloud_llm"
    REGEX_FALLBACK = "regex_fallback"
    DIRECT_RESPONSE = "direct_response"


@dataclass
class RouteResult:
    """Result from routing a user query."""
    tool: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    requires_generation: bool = False
    clarification: Optional[str] = None
    response: Optional[str] = None
    strategy: RoutingStrategy = RoutingStrategy.REGEX_FALLBACK
    confidence: float = 0.0
    raw_response: Optional[str] = None


# =============================================================================
# Agent Router
# =============================================================================

class AgentRouter:
    """
    Routes user queries to appropriate tools using LLM with function calling.
    
    Hierarchy:
    1. Try local vLLM (fastest, no cost)
    2. Fall back to cloud API (Lightning.ai, OpenAI)
    3. Fall back to regex patterns (always works)
    """
    
    def __init__(
        self,
        local_url: str = "http://localhost:8000/v1",
        local_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        cloud_provider: str = "lightning",
        use_local: bool = True,
        use_cloud: bool = True,
        use_regex_fallback: bool = True,
    ):
        """
        Initialize the router.
        
        Args:
            local_url: URL of local vLLM server
            local_model: Model name for local inference
            cloud_provider: Cloud provider ("lightning", "openai", "anthropic")
            use_local: Whether to try local inference
            use_cloud: Whether to try cloud inference
            use_regex_fallback: Whether to fall back to regex
        """
        self.local_url = local_url
        self.local_model = local_model
        self.cloud_provider = cloud_provider
        self.use_local = use_local
        self.use_cloud = use_cloud
        self.use_regex_fallback = use_regex_fallback
        
        self._local_client = None
        self._cloud_client = None
        self._local_available = None
        
    def _get_local_client(self):
        """Get or create local vLLM client."""
        if self._local_client is not None:
            return self._local_client
        
        if not self.use_local:
            return None
            
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url=self.local_url,
                api_key="not-needed"
            )
            # Quick health check
            client.models.list()
            self._local_client = client
            self._local_available = True
            logger.info(f"Connected to local vLLM at {self.local_url}")
            return client
        except Exception as e:
            logger.debug(f"Local vLLM not available: {e}")
            self._local_available = False
            return None
    
    def _get_cloud_client(self):
        """Get or create cloud client."""
        if self._cloud_client is not None:
            return self._cloud_client
            
        if not self.use_cloud:
            return None
        
        try:
            from openai import OpenAI
            
            if self.cloud_provider == "lightning":
                api_key = os.environ.get("LIGHTNING_API_KEY")
                if api_key:
                    self._cloud_client = OpenAI(
                        base_url="https://api.lightning.ai/v1",
                        api_key=api_key
                    )
                    logger.info("Using Lightning.ai for cloud routing")
                    return self._cloud_client
                    
            elif self.cloud_provider == "openai":
                api_key = os.environ.get("OPENAI_API_KEY")
                if api_key:
                    self._cloud_client = OpenAI(api_key=api_key)
                    logger.info("Using OpenAI for cloud routing")
                    return self._cloud_client
                    
        except Exception as e:
            logger.debug(f"Cloud client not available: {e}")
            
        return None
    
    def is_local_available(self) -> bool:
        """Check if local inference is available."""
        if self._local_available is None:
            self._get_local_client()
        return self._local_available or False
    
    async def route(
        self, 
        message: str, 
        context: Dict[str, Any] = None
    ) -> RouteResult:
        """
        Route a user message to the appropriate tool.
        
        Args:
            message: User's natural language message
            context: Current conversation context
            
        Returns:
            RouteResult with tool name and arguments
        """
        # Try local LLM first
        local_client = self._get_local_client()
        if local_client:
            try:
                result = await self._route_with_llm(
                    local_client, 
                    self.local_model, 
                    message, 
                    context,
                    RoutingStrategy.LOCAL_LLM
                )
                if result.tool or result.response:
                    return result
            except Exception as e:
                logger.warning(f"Local routing failed: {e}")
        
        # Try cloud LLM
        cloud_client = self._get_cloud_client()
        if cloud_client:
            try:
                cloud_model = self._get_cloud_model()
                result = await self._route_with_llm(
                    cloud_client,
                    cloud_model,
                    message,
                    context,
                    RoutingStrategy.CLOUD_LLM
                )
                if result.tool or result.response:
                    return result
            except Exception as e:
                logger.warning(f"Cloud routing failed: {e}")
        
        # Fall back to regex
        if self.use_regex_fallback:
            return self._route_with_regex(message, context)
        
        # Nothing worked
        return RouteResult(
            response="I'm having trouble understanding your request. Could you try rephrasing it?",
            strategy=RoutingStrategy.DIRECT_RESPONSE
        )
    
    async def _route_with_llm(
        self,
        client,
        model: str,
        message: str,
        context: Dict[str, Any],
        strategy: RoutingStrategy
    ) -> RouteResult:
        """Route using LLM with function calling."""
        
        system_prompt = self._build_system_prompt(context)
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                tools=AGENT_TOOLS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=512
            )
            
            choice = response.choices[0]
            
            # Check for tool calls
            if choice.message.tool_calls:
                tool_call = choice.message.tool_calls[0]
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                
                return RouteResult(
                    tool=tool_call.function.name,
                    arguments=arguments,
                    requires_generation=tool_call.function.name == "generate_workflow",
                    strategy=strategy,
                    confidence=0.9,
                    raw_response=str(response)
                )
            
            # No tool call - return direct response
            if choice.message.content:
                return RouteResult(
                    response=choice.message.content,
                    strategy=strategy,
                    confidence=0.8,
                    raw_response=str(response)
                )
                
        except Exception as e:
            logger.error(f"LLM routing error: {e}")
            raise
        
        return RouteResult(strategy=strategy)
    
    def _route_with_regex(
        self, 
        message: str, 
        context: Dict[str, Any]
    ) -> RouteResult:
        """
        Fall back to regex-based routing.
        Uses existing patterns from tools.py.
        """
        import re
        
        message_lower = message.lower()
        
        # Data scanning patterns
        path_match = re.search(
            r'(?:scan|find|look for|check|discover|list|show)\s+(?:the\s+)?(?:local\s+)?(?:data|files?|samples?|fastq|folders?|directories?|datasets?)\s+(?:in|at|from|under|within)\s+[\'"]?([\/~][^\s\'"?\]]+)',
            message_lower
        )
        if path_match:
            return RouteResult(
                tool="scan_data",
                arguments={"path": path_match.group(1), "recursive": True},
                strategy=RoutingStrategy.REGEX_FALLBACK,
                confidence=0.7
            )
        
        # Simple path scan
        simple_path = re.search(r'scan\s+[\'"]?([\/~][^\s\'"?\]]+)', message_lower)
        if simple_path:
            return RouteResult(
                tool="scan_data",
                arguments={"path": simple_path.group(1), "recursive": True},
                strategy=RoutingStrategy.REGEX_FALLBACK,
                confidence=0.6
            )
        
        # Search databases
        if any(kw in message_lower for kw in ["search", "find in encode", "find in geo", "database"]):
            return RouteResult(
                tool="search_databases",
                arguments={"query": message},
                strategy=RoutingStrategy.REGEX_FALLBACK,
                confidence=0.5
            )
        
        # Workflow generation
        if any(kw in message_lower for kw in ["generate workflow", "create workflow", "build pipeline", "create pipeline"]):
            # Try to extract analysis type
            analysis_type = "rna_seq"  # default
            if "chip" in message_lower:
                analysis_type = "chip_seq"
            elif "atac" in message_lower:
                analysis_type = "atac_seq"
            elif "methyl" in message_lower:
                analysis_type = "methylation"
            elif "variant" in message_lower:
                analysis_type = "variant_calling"
            elif "metagen" in message_lower:
                analysis_type = "metagenomics"
            elif "single" in message_lower or "scrna" in message_lower:
                analysis_type = "single_cell"
            
            return RouteResult(
                tool="generate_workflow",
                arguments={"analysis_type": analysis_type},
                requires_generation=True,
                strategy=RoutingStrategy.REGEX_FALLBACK,
                confidence=0.6
            )
        
        # Job status
        if any(kw in message_lower for kw in ["status", "how is", "progress"]):
            return RouteResult(
                tool="get_job_status",
                arguments={},
                strategy=RoutingStrategy.REGEX_FALLBACK,
                confidence=0.5
            )
        
        # Logs
        if any(kw in message_lower for kw in ["logs", "log output", "show log"]):
            return RouteResult(
                tool="get_logs",
                arguments={},
                strategy=RoutingStrategy.REGEX_FALLBACK,
                confidence=0.5
            )
        
        # Cancel
        if any(kw in message_lower for kw in ["cancel", "stop", "abort"]):
            return RouteResult(
                tool="cancel_job",
                arguments={},
                strategy=RoutingStrategy.REGEX_FALLBACK,
                confidence=0.5
            )
        
        # Diagnose
        if any(kw in message_lower for kw in ["diagnose", "what went wrong", "why failed", "error"]):
            return RouteResult(
                tool="diagnose_error",
                arguments={},
                strategy=RoutingStrategy.REGEX_FALLBACK,
                confidence=0.5
            )
        
        # Submit job
        if any(kw in message_lower for kw in ["run it", "execute", "submit", "start"]):
            return RouteResult(
                tool="submit_job",
                arguments={},
                strategy=RoutingStrategy.REGEX_FALLBACK,
                confidence=0.5
            )
        
        # No tool matched
        return RouteResult(
            strategy=RoutingStrategy.REGEX_FALLBACK,
            confidence=0.0
        )
    
    def _build_system_prompt(self, context: Dict[str, Any] = None) -> str:
        """Build context-aware system prompt."""
        prompt = """You are BioPipelines AI, an expert assistant for bioinformatics workflow design and execution.

Your capabilities:
1. Scan directories for sequencing data (FASTQ, BAM files)
2. Search public databases (ENCODE, GEO, SRA)
3. Generate Nextflow workflows for various analyses
4. Submit and monitor jobs on SLURM clusters
5. Diagnose pipeline errors

IMPORTANT INSTRUCTIONS:
- Use tools to help the user. Don't just describe what you could do.
- If the user mentions a path starting with / or ~, they want to scan data.
- If the user wants a workflow, use generate_workflow.
- If the user mentions analysis types (RNA-seq, ChIP-seq, etc.), they probably want a workflow.
- Be concise in your responses.

"""
        if context:
            prompt += "\nCURRENT CONTEXT:\n"
            if context.get("data_loaded"):
                prompt += f"- Loaded data: {context.get('sample_count', 0)} samples from {context.get('data_path', 'unknown')}\n"
            if context.get("last_workflow"):
                prompt += f"- Last workflow: {context.get('last_workflow')}\n"
            if context.get("active_job"):
                prompt += f"- Active job: {context.get('active_job')} ({context.get('job_status', 'unknown')})\n"
        else:
            prompt += "\nNo data or workflows loaded yet.\n"
        
        return prompt
    
    def _get_cloud_model(self) -> str:
        """Get the model name for cloud provider."""
        if self.cloud_provider == "lightning":
            return "meta-llama/Llama-3.3-70B-Instruct"
        elif self.cloud_provider == "openai":
            return "gpt-4o"
        else:
            return "gpt-3.5-turbo"
    
    def get_status(self) -> Dict[str, Any]:
        """Get router status."""
        return {
            "local_available": self.is_local_available(),
            "local_url": self.local_url if self.is_local_available() else None,
            "local_model": self.local_model if self.is_local_available() else None,
            "cloud_available": self._get_cloud_client() is not None,
            "cloud_provider": self.cloud_provider if self._cloud_client else None,
            "fallback_enabled": self.use_regex_fallback,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_router_instance = None

def get_router() -> AgentRouter:
    """Get or create the global router instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = AgentRouter()
    return _router_instance


async def route_message(message: str, context: Dict[str, Any] = None) -> RouteResult:
    """Route a message using the global router."""
    router = get_router()
    return await router.route(message, context)


# =============================================================================
# CLI Testing
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        router = AgentRouter()
        
        print("\n🧪 Testing AgentRouter")
        print("=" * 60)
        print(f"Status: {router.get_status()}")
        print()
        
        test_queries = [
            "scan data in /scratch/sdodl001/BioPipelines/data/raw",
            "search for H3K27ac ChIP-seq data in ENCODE",
            "create an RNA-seq workflow for mouse differential expression",
            "what's the status of my job?",
            "show me the logs",
            "the job failed, can you diagnose it?",
            "run it on SLURM",
            "hello, how are you?",
        ]
        
        for query in test_queries:
            print(f"📝 Query: \"{query}\"")
            result = await router.route(query)
            print(f"   Tool: {result.tool}")
            print(f"   Args: {result.arguments}")
            print(f"   Strategy: {result.strategy.value}")
            print(f"   Confidence: {result.confidence:.2f}")
            if result.response:
                print(f"   Response: {result.response[:100]}...")
            print()
    
    asyncio.run(test())
