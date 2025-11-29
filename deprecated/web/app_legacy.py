"""
BioPipelines - Chat-First Web Interface (Gradio 6.x)

A minimal, focused UI where chat is the primary interface.
All features are accessible through natural conversation.

Refactored to use unified chat handler for clean separation of concerns.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Generator
from datetime import datetime

import gradio as gr

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import unified chat handler
from workflow_composer.web.chat_handler import get_chat_handler, UnifiedChatHandler

# ============================================================================
# Configuration
# ============================================================================

def detect_vllm_endpoint() -> str:
    """Dynamically detect vLLM endpoint from running SLURM jobs."""
    import subprocess
    try:
        # Check for running vLLM job
        result = subprocess.run(
            ["squeue", "--me", "-h", "-o", "%i %j %T %N"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.strip().split('\n'):
            if line and ('vllm' in line.lower() or 'biopipelines' in line.lower()) and 'RUNNING' in line:
                parts = line.split()
                if len(parts) >= 4:
                    node = parts[3]
                    return f"http://{node}:8000/v1"
    except Exception:
        pass
    # Fallback to env var or localhost
    return os.environ.get("VLLM_URL", "http://localhost:8000/v1")

VLLM_URL = detect_vllm_endpoint()
USE_LOCAL_LLM = os.environ.get("USE_LOCAL_LLM", "false").lower() == "true"
DEFAULT_PORT = int(os.environ.get("GRADIO_PORT", "7860"))

# ============================================================================
# LLM Provider Management - Fallback Chain
# ============================================================================

class LLMProvider:
    """Manages LLM providers with automatic fallback.
    
    Priority order:
    1. Local vLLM (Llama-70B on H100) - Free, unlimited, best for coding
    2. GitHub Models API (free tier with Pro+) - gpt-4o-mini, DeepSeek-R1
    3. Google Gemini (free tier) - gemini-2.0-flash
    4. OpenAI (paid, last resort) - gpt-4o
    """
    
    PROVIDERS = [
        ("local", "Local vLLM", None),
        ("github", "GitHub Models", "GITHUB_TOKEN"),
        ("google", "Google Gemini", "GOOGLE_API_KEY"),
        ("openai", "OpenAI", "OPENAI_API_KEY"),
    ]
    
    def __init__(self):
        self.clients: Dict[str, Any] = {}
        self.models: Dict[str, str] = {}
        self.active_provider: Optional[str] = None
        self._setup_providers()
    
    def _setup_providers(self):
        """Initialize all available providers."""
        try:
            from openai import OpenAI
        except ImportError:
            print("⚠ OpenAI package not installed")
            return
        
        # 1. Local vLLM (always set up, may not be ready)
        # With 2x H100: Qwen3-Coder-30B-A3B (best for coding, 3B active MoE)
        # With 4x H100: MiniMax-M2 (456B MoE, best reasoning)
        self.clients["local"] = OpenAI(base_url=VLLM_URL, api_key="not-needed")
        self.models["local"] = "Qwen/Qwen3-Coder-30B-A3B-Instruct"  # Updated dynamically
        
        # 2. GitHub Models API (free tier with Pro+ account!)
        # Uses GitHub PAT with models:read scope
        # Endpoint: https://models.inference.ai.azure.com
        github_token = os.environ.get("GITHUB_TOKEN", "")
        if github_token:
            self.clients["github"] = OpenAI(
                base_url="https://models.inference.ai.azure.com",
                api_key=github_token
            )
            # gpt-4o-mini is free and good for coding tasks
            # Can also use: DeepSeek-R1, Llama-3.1-405B, etc.
            self.models["github"] = "gpt-4o-mini"
            print("✓ GitHub Models configured (gpt-4o-mini)")
        
        # 3. Google Gemini (via OpenAI-compatible API) - free tier
        google_key = os.environ.get("GOOGLE_API_KEY", "")
        if google_key:
            self.clients["google"] = OpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=google_key
            )
            self.models["google"] = "gemini-2.0-flash"
            print("✓ Google Gemini configured")
        
        # 4. OpenAI (paid, use as last resort)
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if openai_key:
            self.clients["openai"] = OpenAI(api_key=openai_key)
            self.models["openai"] = "gpt-4o"
            print("✓ OpenAI configured (gpt-4o) - paid fallback")
        
        # Check local vLLM availability
        if self._check_local_vllm():
            self.active_provider = "local"
            print(f"✓ Local vLLM ready: {VLLM_URL}")
        else:
            # Find first working cloud provider
            self.active_provider = self._find_working_provider()
            if self.active_provider:
                print(f"✓ Using {self.active_provider} as primary LLM")
            else:
                print("⚠ No working LLM provider found")
    
    def _check_local_vllm(self) -> bool:
        """Check if local vLLM is ready (with cooldown to avoid spam)."""
        import time
        
        # Cooldown: don't check more than once every 30 seconds
        now = time.time()
        if hasattr(self, '_last_vllm_check'):
            if now - self._last_vllm_check < 30:
                return getattr(self, '_last_vllm_result', False)
        
        self._last_vllm_check = now
        
        try:
            import requests
            health_url = VLLM_URL.replace('/v1', '') + '/health'
            resp = requests.get(health_url, timeout=3)
            if resp.status_code == 200:
                # Also verify models endpoint works
                models_resp = requests.get(f"{VLLM_URL}/models", timeout=3)
                self._last_vllm_result = models_resp.status_code == 200
                if self._last_vllm_result and not getattr(self, '_vllm_ready_logged', False):
                    print("✓ Local vLLM is now ready")
                    self._vllm_ready_logged = True
                return self._last_vllm_result
            self._last_vllm_result = False
            return False
        except Exception:
            # Silently fail - don't spam logs
            self._last_vllm_result = False
            return False
    
    def _find_working_provider(self) -> Optional[str]:
        """Find the first working cloud provider.
        
        Priority: github (free) > google (free) > openai (paid)
        """
        for name in ["github", "google", "openai"]:
            if name in self.clients:
                if self._test_provider(name):
                    return name
        return None
    
    def _test_provider(self, name: str) -> bool:
        """Test if a provider is working with a simple API call."""
        if name not in self.clients:
            return False
        try:
            client = self.clients[name]
            model = self.models[name]
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            print(f"  ⚠ {name} failed: {e}")
            return False
    
    def get_client_and_model(self) -> tuple:
        """Get the active client and model, with fallback."""
        # Always check if local vLLM became available
        if self._check_local_vllm():
            self.active_provider = "local"
            return self.clients["local"], self.models["local"]
        
        # Use active provider
        if self.active_provider and self.active_provider in self.clients:
            return self.clients[self.active_provider], self.models[self.active_provider]
        
        # Try to find a working provider
        self.active_provider = self._find_working_provider()
        if self.active_provider:
            return self.clients[self.active_provider], self.models[self.active_provider]
        
        return None, None
    
    def status(self) -> str:
        """Get human-readable status."""
        if self._check_local_vllm():
            return "🟢 Local LLM"
        if self.active_provider:
            names = {"openai": "OpenAI", "google": "Gemini", "lightning": "Lightning"}
            return f"🟢 {names.get(self.active_provider, self.active_provider)}"
        return "🔴 No LLM"
    
    @property
    def available(self) -> bool:
        """Check if any LLM is available."""
        return self.active_provider is not None or self._check_local_vllm()

# Initialize provider
llm_provider = LLMProvider()
LLM_AVAILABLE = llm_provider.available

# ============================================================================
# Unified Chat Handler 
# ============================================================================

# Initialize the unified chat handler (handles tools, validation, LLM fallback)
chat_handler: Optional[UnifiedChatHandler] = None
try:
    chat_handler = get_chat_handler()
    print("✓ Unified chat handler initialized")
except Exception as e:
    print(f"⚠ Chat handler not available: {e}")

# ============================================================================
# Application State
# ============================================================================

class AppState:
    """Simple application state."""
    
    def __init__(self):
        self.data_path: Optional[str] = None
        self.samples: List[Dict] = []
        self.current_workflow: Optional[str] = None
        self.workflow_path: Optional[str] = None
        self.jobs: Dict[str, Dict] = {}
    
    def to_context_string(self) -> str:
        """Return context for LLM."""
        parts = []
        if self.data_path:
            parts.append(f"Data: {self.data_path} ({len(self.samples)} samples)")
        if self.current_workflow:
            parts.append(f"Workflow: {self.current_workflow}")
        running = len([j for j in self.jobs.values() if j.get("status") == "running"])
        if running:
            parts.append(f"Jobs: {running} running")
        return "; ".join(parts) if parts else "No data loaded yet"
    
    def summary(self) -> str:
        """One-line status."""
        parts = []
        if self.samples:
            parts.append(f"📁 {len(self.samples)} samples")
        if self.current_workflow:
            parts.append(f"📋 {self.current_workflow}")
        running = len([j for j in self.jobs.values() if j.get("status") == "running"])
        if running:
            parts.append(f"🔄 {running} jobs")
        return " | ".join(parts) if parts else "Ready"

# Global state
state = AppState()

# ============================================================================
# Chat Handler
# ============================================================================

def get_system_prompt() -> str:
    """Generate system prompt with current context."""
    return f"""You are BioPipelines Assistant, an intelligent AI that helps researchers run bioinformatics analyses.

## Current Context:
{state.to_context_string()}

## Your Capabilities:
1. **Data Discovery**: Find local data, search remote databases (ENCODE, GEO, TCGA)
2. **Data Validation**: Check if downloaded data is real or just metadata
3. **Workflow Generation**: Create analysis pipelines (RNA-seq, ChIP-seq, methylation, etc.)
4. **Job Execution**: Submit and monitor SLURM jobs
5. **Error Diagnosis**: Analyze failed jobs and suggest fixes
6. **Results Analysis**: Interpret QC reports and outputs
7. **Education**: Explain bioinformatics concepts

## Available Tools:
You MUST use these tools when appropriate. Don't just give instructions - execute the tools!

**Data Tools:**
- **scan_data(path?)** - Find FASTQ/BAM files in a directory
- **describe_files(path?)** - Get file sizes, rows, columns 
- **validate_dataset(id?)** - Check if data is real vs metadata (IMPORTANT!)
- **cleanup_data(path?)** - Remove corrupted files

**Search Tools:**
- **search_databases(query)** - Search ENCODE, GEO for public datasets
- **search_tcga(query)** - Search TCGA/GDC for cancer data (USE FOR CANCER!)
- **check_references(organism)** - Verify reference genomes

**Workflow Tools:**  
- **generate_workflow(description)** - Create a new workflow from description
- **list_workflows()** - Show available workflows
- **submit_job(workflow?)** - Run on SLURM

**Monitoring Tools:**
- **get_job_status(job_id?)** - Check job progress
- **monitor_jobs()** - Check if downloads completed
- **get_logs(job_id?)** - View job output
- **diagnose_error()** - AI-powered error analysis

**Learning Tools:**
- **analyze_results(path?)** - Interpret QC reports
- **explain_concept(term)** - Learn about bioinformatics terms

## Critical Rules:
1. When user asks about data - USE scan_data or describe_files
2. When user downloads data - SUGGEST validate_dataset afterward
3. When user asks about CANCER data - USE search_tcga (not search_databases)
4. When user says "download all" - USE download_dataset for each ID
5. After errors - USE diagnose_error to help
6. Be proactive - suggest next steps after each action

## Response Style:
- Be concise but thorough
- Use markdown formatting
- Proactively suggest next steps
- Warn about potential issues (like metadata-only downloads) 
When they ask to clean up corrupted files - USE the cleanup_data tool.
Do NOT just give instructions - actually execute the tools!
"""

# Define tools for function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "scan_data",
            "description": "Scan a directory for FASTQ, BAM, and other sequencing data files. Use this when user asks about their data, what files they have, wants to see samples, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to scan. If not provided, scans default data directory (/scratch/sdodl001/BioPipelines/data)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "describe_files",
            "description": "Get detailed information about files in a directory - file sizes, row counts, column names, file types. Use when user wants to inspect or understand their data files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to inspect"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "validate_dataset",
            "description": "CRITICAL: Check if a downloaded dataset contains actual sequencing data (FASTQ) or just metadata files. Always suggest this after downloading! Many GEO datasets are metadata-only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path_or_id": {
                        "type": "string",
                        "description": "Path to dataset directory or dataset ID like GSE200839"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cleanup_data",
            "description": "Scan for corrupted data files (HTML error pages masquerading as FASTQ files, invalid gzip files). Shows preview first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to clean"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_cleanup",
            "description": "Confirm and execute a pending cleanup operation after user says yes.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_databases",
            "description": "Search ENCODE and GEO for public datasets. For CANCER data, use search_tcga instead!",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, e.g. 'human RNA-seq liver'"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_tcga",
            "description": "Search TCGA/GDC Cancer Genome Atlas for tumor datasets. USE THIS FOR CANCER DATA - not search_databases! Covers GBM, BRCA, LUAD, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Cancer search query, e.g. 'GBM methylation', 'breast cancer RNA-seq', 'brain tumor'"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "download_dataset",
            "description": "Download a dataset from GEO or ENCODE. Always suggest validate_dataset afterward!",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "Dataset ID like GSE200839 or ENCSR856UND"
                    }
                },
                "required": ["dataset_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_workflow",
            "description": "Generate a new bioinformatics workflow from a description. Creates Nextflow pipeline code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "What kind of analysis, e.g. 'RNA-seq differential expression', 'ChIP-seq peak calling'"
                    }
                },
                "required": ["description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_workflows",
            "description": "List available bioinformatics workflow templates",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_results",
            "description": "Analyze and interpret workflow results - QC reports, logs, output files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to results directory"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "explain_concept",
            "description": "Explain a bioinformatics concept, term, or method. Use for educational questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "concept": {
                        "type": "string",
                        "description": "The term to explain, e.g. 'DNA methylation', 'WGBS', 'differential expression'"
                    }
                },
                "required": ["concept"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "diagnose_error",
            "description": "Diagnose errors in a failed job using AI analysis. Use when user asks about failures.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Optional SLURM job ID"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "monitor_jobs",
            "description": "Check if SLURM jobs completed and data was saved. Use when user asks about downloads.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


def execute_tool_call(tool_name: str, arguments: dict, return_full: bool = False):
    """
    Execute a tool and return the result.
    
    Args:
        tool_name: Name of the tool to execute
        arguments: Tool arguments
        return_full: If True, returns (message, result_data) for validation
    
    Returns:
        If return_full: tuple of (message, result_data)
        Otherwise: just the message string
    """
    if not agent_tools:
        result = ("⚠️ Tools not available", {})
        return result if return_full else result[0]
    
    try:
        if tool_name == "scan_data":
            result = agent_tools.scan_data(arguments.get("path"))
            if result.success and result.data:
                state.samples = result.data.get("samples", [])
                state.data_path = result.data.get("path")
            return (result.message, result.data or {}) if return_full else result.message
        
        elif tool_name == "describe_files":
            result = agent_tools.describe_files(arguments.get("path"))
            return (result.message, result.data or {}) if return_full else result.message
        
        elif tool_name == "validate_dataset":
            result = agent_tools.validate_dataset(arguments.get("path_or_id"))
            return (result.message, result.data or {}) if return_full else result.message
        
        elif tool_name == "cleanup_data":
            result = agent_tools.cleanup_data(arguments.get("path"), confirm=False)
            return (result.message, result.data or {}) if return_full else result.message
        
        elif tool_name == "confirm_cleanup":
            result = agent_tools.confirm_cleanup()
            return (result.message, result.data or {}) if return_full else result.message
        
        elif tool_name == "search_databases":
            result = agent_tools.search_databases(arguments.get("query", ""))
            return (result.message, result.data or {}) if return_full else result.message
        
        elif tool_name == "search_tcga":
            result = agent_tools.search_tcga(arguments.get("query", ""))
            return (result.message, result.data or {}) if return_full else result.message
        
        elif tool_name == "download_dataset":
            result = agent_tools.download_dataset(arguments.get("dataset_id", ""))
            return (result.message, result.data or {}) if return_full else result.message
        
        elif tool_name == "generate_workflow":
            result = agent_tools.generate_workflow(arguments.get("description", ""))
            return (result.message, result.data or {}) if return_full else result.message
        
        elif tool_name == "list_workflows":
            result = agent_tools.list_workflows()
            return (result.message, result.data or {}) if return_full else result.message
        
        elif tool_name == "analyze_results":
            result = agent_tools.analyze_results(arguments.get("path"))
            return (result.message, result.data or {}) if return_full else result.message
        
        elif tool_name == "explain_concept":
            result = agent_tools.explain_concept(arguments.get("concept", ""))
            return (result.message, result.data or {}) if return_full else result.message
        
        elif tool_name == "diagnose_error":
            result = agent_tools.diagnose_error(arguments.get("job_id"))
            return (result.message, result.data or {}) if return_full else result.message
        
        elif tool_name == "monitor_jobs":
            result = agent_tools.monitor_jobs()
            return (result.message, result.data or {}) if return_full else result.message
        
        else:
            msg = f"⚠️ Unknown tool: {tool_name}"
            return (msg, {}) if return_full else msg
    
    except Exception as e:
        import traceback
        logger_msg = traceback.format_exc()
        print(f"Tool {tool_name} error: {logger_msg}")
        msg = f"❌ Tool error: {e}"
        return (msg, {"error": str(e)}) if return_full else msg


def try_pattern_match(message: str) -> Optional[str]:
    """
    Try to match common patterns first (fast, no LLM cost).
    Now includes validation to ensure results match user intent.
    
    Returns:
        Validated tool result if matched, None otherwise.
    """
    if not agent_tools:
        return None
    
    try:
        # Use existing pattern detection from tools.py
        tool_info = agent_tools.detect_tool(message)
        if not tool_info:
            return None
        
        tool_name, params = tool_info
        
        # Map ToolName enum to execute_tool_call
        from workflow_composer.agents.tools import ToolName
        
        tool_map = {
            ToolName.SCAN_DATA: ("scan_data", {"path": params[0] if params else None}),
            ToolName.CLEANUP_DATA: ("cleanup_data", {"path": params[0] if params else None}),
            ToolName.CONFIRM_CLEANUP: ("confirm_cleanup", {}),
            ToolName.SEARCH_DATABASES: ("search_databases", {"query": params[0] if params else ""}),
            ToolName.DOWNLOAD_DATASET: ("download_dataset", {"dataset_id": params[0] if params else ""}),
            ToolName.LIST_WORKFLOWS: ("list_workflows", {}),
            ToolName.CHECK_REFERENCES: ("check_references", {"organism": params[0] if params else "human"}),
        }
        
        if tool_name in tool_map:
            func_name, args = tool_map[tool_name]
            
            # Execute tool with full result for validation
            result_message, result_data = execute_tool_call(func_name, args, return_full=True)
            
            # VALIDATION: Check if result matches user intent
            if response_validator:
                validation = validate_tool_response(
                    tool_name=func_name,
                    tool_result=result_data,
                    response_message=result_message,
                    user_message=message,
                )
                
                # Return validated response (may include caveats/warnings)
                return validation.validated_response
            
            return result_message
        
    except Exception as e:
        # Pattern matching failed, will fall through to LLM
        import logging
        logging.getLogger(__name__).debug(f"Pattern match failed: {e}")
    
    return None


def chat_response(message: str, history: List[Dict]) -> Generator[List[Dict], None, None]:
    """
    Generate chat response with hybrid approach:
    1. Pattern matching (fast, no cost)
    2. Validation layer (ensures accuracy)
    3. LLM with function calling (for complex queries)
    """
    
    # Add user message to history immediately
    new_history = history + [{"role": "user", "content": message}]
    
    # Update validation context with conversation history
    if response_validator:
        for msg in history[-5:]:  # Last 5 messages for context
            content = msg.get("content", "")
            # Handle Gradio's list format
            if isinstance(content, list):
                content = ' '.join(str(c) if isinstance(c, str) else c.get('text', '') for c in content if c)
            response_validator.context.add_message(msg["role"], content)
        response_validator.context.add_message("user", message)
    
    # STEP 1: Try fast pattern matching with validation
    if agent_tools:
        pattern_result = try_pattern_match(message)
        if pattern_result:
            yield new_history + [{"role": "assistant", "content": pattern_result}]
            return
    
    # STEP 2: Fall back to LLM with function calling
    client, model = llm_provider.get_client_and_model()
    
    if not client:
        yield new_history + [
            {"role": "assistant", "content": "⚠️ No LLM available. Please set OPENAI_API_KEY or GOOGLE_API_KEY."}
        ]
        return
    
    # Build messages for API
    messages = [{"role": "system", "content": get_system_prompt()}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})
    
    try:
        # Let LLM decide if it needs tools
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=2048,
        )
        
        assistant_message = response.choices[0].message
        
        # Check if LLM wants to call a tool
        if assistant_message.tool_calls:
            # Execute all tool calls with validation
            validated_results = []
            for tool_call in assistant_message.tool_calls:
                func_name = tool_call.function.name
                try:
                    func_args = json.loads(tool_call.function.arguments)
                except:
                    func_args = {}
                
                # Execute with full result for validation
                result_message, result_data = execute_tool_call(func_name, func_args, return_full=True)
                
                # Validate the result against user intent
                if response_validator:
                    validation = validate_tool_response(
                        tool_name=func_name,
                        tool_result=result_data,
                        response_message=result_message,
                        user_message=message,
                    )
                    validated_results.append(validation.validated_response)
                else:
                    validated_results.append(result_message)
            
            # Combine validated tool results
            combined_result = "\n\n".join(validated_results)
            yield new_history + [{"role": "assistant", "content": combined_result}]
        
        else:
            # No tool call - just return the response
            content = assistant_message.content or ""
            yield new_history + [{"role": "assistant", "content": content}]
        
    except Exception as e:
        error_msg = str(e)
        
        # If tool calling not supported, fall back to streaming without tools
        if "tool" in error_msg.lower() or "function" in error_msg.lower():
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    max_tokens=2048,
                )
                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        yield new_history + [{"role": "assistant", "content": full_response}]
                return
            except Exception as e2:
                error_msg = str(e2)
        
        # Try fallback provider
        if "401" in error_msg or "403" in error_msg:
            llm_provider.active_provider = None
            alt_client, alt_model = llm_provider.get_client_and_model()
            if alt_client and alt_client != client:
                try:
                    response = alt_client.chat.completions.create(
                        model=alt_model,
                        messages=messages,
                        stream=True,
                        max_tokens=2048,
                    )
                    full_response = ""
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            yield new_history + [{"role": "assistant", "content": full_response}]
                    return
                except Exception as e2:
                    error_msg = f"{error_msg}\nFallback: {e2}"
        
        yield new_history + [{"role": "assistant", "content": f"❌ Error: {error_msg}"}]


# ============================================================================
# Status Functions
# ============================================================================

def get_status() -> str:
    """Get current status as HTML."""
    llm_status = llm_provider.status()
    return f"{llm_status} | {state.summary()}"


# ============================================================================
# Example Messages
# ============================================================================

EXAMPLES = [
    {"text": "Help me get started with BioPipelines", "display_text": "🚀 Get Started"},
    {"text": "Scan my data in ~/data/fastq", "display_text": "📁 Scan Data"},
    {"text": "Create an RNA-seq differential expression workflow", "display_text": "🧬 RNA-seq"},
    {"text": "What workflows are available?", "display_text": "📋 List Workflows"},
    {"text": "Show me my running jobs", "display_text": "📊 Check Jobs"},
]


# ============================================================================
# Main UI
# ============================================================================

def create_app() -> gr.Blocks:
    """Create the Gradio app."""
    
    with gr.Blocks(title="🧬 BioPipelines") as app:
        
        # Header
        gr.Markdown("# 🧬 BioPipelines\n*AI-powered bioinformatics workflow composer*")
        
        # Status bar (auto-refresh every 30s)
        status = gr.Markdown(value=get_status, every=30)
        
        # Main Chat
        chatbot = gr.Chatbot(
            label="Chat",
            height=500,
            placeholder="Ask me to scan data, create workflows, or run analyses...",
            examples=EXAMPLES,
            buttons=["copy"],
            avatar_images=(None, "🧬"),
            layout="bubble",
        )
        
        # Input
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Type your message here...",
                show_label=False,
                scale=9,
                lines=1,
            )
            send = gr.Button("Send", variant="primary", scale=1)
        
        # Settings (collapsed)
        with gr.Accordion("⚙️ Settings", open=False):
            gr.Markdown(f"""
**LLM:** {"Local vLLM at " + VLLM_URL if USE_LOCAL_LLM else "Lightning AI Cloud"}  
**Agent:** {"Available ✓" if AGENT_AVAILABLE else "Not available"}
            """)
            clear = gr.Button("🗑️ Clear Chat")
        
        # Event handlers
        def submit(message, history):
            if not message.strip():
                return history, ""
            for response in chat_response(message, history):
                yield response, ""
        
        def clear_chat():
            return [], ""
        
        # Wire up events
        msg.submit(submit, [msg, chatbot], [chatbot, msg])
        send.click(submit, [msg, chatbot], [chatbot, msg])
        clear.click(clear_chat, outputs=[chatbot, msg])
    
    return app


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BioPipelines Web Interface")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()
    
    print()
    print("╔════════════════════════════════════════════╗")
    print("║          🧬 BioPipelines                   ║")
    print("╠════════════════════════════════════════════╣")
    print(f"║  Server: http://localhost:{args.port:<5}            ║")
    print(f"║  LLM: {'Local vLLM' if USE_LOCAL_LLM else 'Cloud API':<15}               ║")
    print("╚════════════════════════════════════════════╝")
    print()
    
    app = create_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
