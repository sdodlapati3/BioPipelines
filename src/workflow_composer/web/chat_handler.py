"""
Unified Chat Handler
====================

A robust chat handler that integrates:
- Modular tools system
- LLM fallback chain
- Session management
- Error handling with graceful degradation
- Retry logic for transient failures
"""

import logging
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Generator, Tuple
from pathlib import Path
from functools import wraps

logger = logging.getLogger(__name__)


# =============================================================================
# RETRY DECORATOR
# =============================================================================

def with_retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator for retrying functions on transient failures.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    error_str = str(e).lower()
                    
                    # Don't retry on authentication errors
                    if "401" in str(e) or "403" in str(e) or "invalid" in error_str:
                        raise
                    
                    # Don't retry on the last attempt
                    if attempt == max_retries:
                        raise
                    
                    # Retry on transient errors
                    if any(x in error_str for x in ["timeout", "rate", "429", "500", "502", "503"]):
                        logger.warning(f"Retry {attempt + 1}/{max_retries} for {func.__name__}: {e}")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        raise
            
            raise last_error
        return wrapper
    return decorator


# =============================================================================
# LLM PROVIDER WITH FALLBACK CHAIN
# =============================================================================

import os

# Import shared utility
from workflow_composer.web.utils import detect_vllm_endpoint


class LLMProvider:
    """
    Manages LLM providers with automatic fallback.
    
    Priority order:
    1. Local vLLM (on HPC GPU nodes) - Free, unlimited
    2. GitHub Models API (free with Pro+) - gpt-4o-mini, DeepSeek-R1
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
        self._load_secrets()
        self._setup_providers()
    
    def _load_secrets(self):
        """Load API keys from .secrets/ directory if not in environment."""
        secrets_dir = Path(__file__).parent.parent.parent.parent / ".secrets"
        
        secrets_map = {
            "github_token": "GITHUB_TOKEN",
            "google_api_key": "GOOGLE_API_KEY",
            "openai_key": "OPENAI_API_KEY",
            "hf_token": "HF_TOKEN",
        }
        
        for filename, env_var in secrets_map.items():
            if not os.environ.get(env_var):
                secret_file = secrets_dir / filename
                if secret_file.exists():
                    value = secret_file.read_text().strip()
                    if value:
                        os.environ[env_var] = value
                        logger.debug(f"Loaded {env_var} from .secrets/{filename}")
    
    def _setup_providers(self):
        """Initialize all available providers."""
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("OpenAI package not installed")
            return
        
        # 1. Local vLLM (auto-detect from SLURM)
        vllm_url = detect_vllm_endpoint()
        self.clients["local"] = OpenAI(base_url=vllm_url, api_key="not-needed")
        self.models["local"] = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
        
        # 2. GitHub Models API (FREE with GitHub Pro+!)
        # Uses your regular GITHUB_TOKEN with models:read scope
        # Endpoint: https://models.inference.ai.azure.com
        # Models: gpt-4o-mini (150 req/day), gpt-4o (50 req/day), DeepSeek-R1, Llama
        github_token = os.environ.get("GITHUB_TOKEN", "")
        if github_token:
            self.clients["github"] = OpenAI(
                base_url="https://models.inference.ai.azure.com",
                api_key=github_token
            )
            self.models["github"] = "gpt-4o-mini"  # Best for coding, 150 req/day free
            print("✓ GitHub Models configured (gpt-4o-mini) - FREE with Pro+")
        
        # 3. Google Gemini (free tier available)
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
        
        # Determine active provider
        if self._check_local_vllm():
            self.active_provider = "local"
            print(f"✓ Using local vLLM at {vllm_url}")
        else:
            self.active_provider = self._find_working_provider()
            if self.active_provider:
                print(f"✓ Using {self.active_provider} as primary LLM")
    
    def _check_local_vllm(self) -> bool:
        """Check if local vLLM is accessible."""
        try:
            import requests
            client = self.clients.get("local")
            if client:
                # Try to list models
                response = requests.get(
                    f"{client.base_url.rstrip('/')}/models",
                    timeout=2
                )
                return response.status_code == 200
        except Exception:
            pass
        return False
    
    def _find_working_provider(self) -> Optional[str]:
        """Find the first working cloud provider."""
        for provider, name, env_var in self.PROVIDERS:
            if provider == "local":
                continue
            if provider in self.clients:
                return provider
        return None
    
    def get_client_and_model(self) -> Tuple[Any, str]:
        """Get active client and model name."""
        if self.active_provider and self.active_provider in self.clients:
            return self.clients[self.active_provider], self.models[self.active_provider]
        return None, ""
    
    def status(self) -> str:
        """Get status string."""
        names = {"local": "Local vLLM", "github": "GitHub Models", "google": "Gemini", "openai": "OpenAI"}
        if self.active_provider:
            return f"🟢 {names.get(self.active_provider, self.active_provider)}"
        return "🔴 No LLM"
    
    @property
    def available(self) -> bool:
        """Check if any LLM is available."""
        return self.active_provider is not None


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

@dataclass
class ConversationMessage:
    """A single message in a conversation."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tool_calls: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class SessionState:
    """State for a single chat session."""
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    messages: List[ConversationMessage] = field(default_factory=list)
    
    # Application state
    data_path: Optional[str] = None
    samples: List[Dict] = field(default_factory=list)
    current_workflow: Optional[str] = None
    workflow_path: Optional[str] = None
    jobs: Dict[str, Dict] = field(default_factory=dict)
    
    # Pending operations (for confirmations)
    pending_cleanup: Optional[Dict] = None
    pending_download: Optional[Dict] = None
    
    def add_message(self, role: str, content: str, **kwargs):
        """Add a message to history."""
        self.messages.append(ConversationMessage(
            role=role,
            content=content,
            **kwargs
        ))
    
    def get_history(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get recent messages as dict format."""
        return [
            {"role": m.role, "content": m.content}
            for m in self.messages[-limit:]
        ]
    
    def context_string(self) -> str:
        """Get context summary for LLM."""
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


class SessionManager:
    """Manages multiple chat sessions."""
    
    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}
        self._default_session_id = "default"
    
    def get_session(self, session_id: str = None) -> SessionState:
        """Get or create a session."""
        sid = session_id or self._default_session_id
        if sid not in self.sessions:
            self.sessions[sid] = SessionState(session_id=sid)
        return self.sessions[sid]
    
    def clear_session(self, session_id: str = None):
        """Clear a session's history."""
        sid = session_id or self._default_session_id
        if sid in self.sessions:
            self.sessions[sid] = SessionState(session_id=sid)


# =============================================================================
# UNIFIED CHAT HANDLER
# =============================================================================

class UnifiedChatHandler:
    """
    Unified chat handler with:
    - Pattern-based tool detection (fast, no LLM cost)
    - LLM function calling (for complex queries)
    - Graceful fallbacks
    - Session management
    - Retry logic for transient failures
    """
    
    def __init__(self, llm_provider=None):
        """
        Initialize the chat handler.
        
        Args:
            llm_provider: LLMProvider instance for API calls
        """
        self.llm_provider = llm_provider
        self.session_manager = SessionManager()
        self._tools = None
        self._tools_available = False
        self._init_tools()
    
    def _init_tools(self):
        """Initialize the tools system."""
        try:
            from workflow_composer.agents.tools import get_agent_tools
            self._tools = get_agent_tools()
            self._tools_available = True
            logger.info(f"Chat handler initialized with {self._tools.get_tool_count()} tools")
        except ImportError as e:
            logger.warning(f"Tools not available: {e}")
            self._tools_available = False
    
    @property
    def agent_tools(self):
        """Get the agent tools instance."""
        return self._tools
    
    @property
    def tools_available(self) -> bool:
        """Check if tools are available."""
        return self._tools_available
    
    def _call_llm_with_retry(
        self, 
        client, 
        model: str, 
        messages: List[Dict],
        tools: List[Dict] = None,
        max_retries: int = 3
    ):
        """
        Call LLM with retry logic for transient failures.
        
        Args:
            client: OpenAI-compatible client
            model: Model name
            messages: Chat messages
            tools: Optional tools for function calling
            max_retries: Maximum retry attempts
            
        Returns:
            Response from LLM
        """
        last_error = None
        delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                if tools:
                    return client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        max_tokens=2048,
                    )
                else:
                    return client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=2048,
                    )
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Don't retry on auth or permanent errors
                if any(x in error_str for x in ["401", "403", "invalid", "not found"]):
                    raise
                
                # Last attempt - raise
                if attempt == max_retries:
                    raise
                
                # Retry on transient errors
                if any(x in error_str for x in ["timeout", "rate", "429", "500", "502", "503", "connection"]):
                    logger.warning(f"LLM retry {attempt + 1}/{max_retries}: {e}")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    raise
        
        raise last_error
    
    def get_system_prompt(self, session: SessionState) -> str:
        """Generate system prompt with current context."""
        return f"""You are BioPipelines Assistant, an intelligent AI that helps researchers run bioinformatics analyses.

## Current Context:
{session.context_string()}

## Tool Usage Rules (CRITICAL - Follow Exactly):

### For Finding Data:
- **Local data**: Use `scan_data` to find files on disk
- **Cancer data**: Use `search_tcga` with cancer_type (GBM=brain, BRCA=breast, LUAD=lung)
- **Other public data**: Use `search_databases` with query (searches GEO, ENCODE)

### For Downloading Data:
- **ALWAYS use `download_dataset`** when user says "download", "get", "fetch"
- For TCGA: `download_dataset(dataset_id="TCGA-GBM", data_type="methylation")`
- For GEO: `download_dataset(dataset_id="GSE12345")`
- For ENCODE: `download_dataset(dataset_id="ENCSR000AAA")`

### Multi-Step Queries:
When user asks to "check if we have X, if not find online":
1. First call `scan_data` to check local files
2. Then call `search_databases` AND/OR `search_tcga` to find online sources
3. You can call MULTIPLE tools in one response!

## Available Tools:
- **scan_data(path?)** - Find FASTQ/BAM files locally
- **describe_files(path?)** - Get file details
- **validate_dataset(path?)** - Check if data is real
- **cleanup_data(path?)** - Remove corrupted files
- **search_databases(query)** - Search GEO, ENCODE, SRA
- **search_tcga(cancer_type, data_type?)** - Search TCGA/GDC (cancer_type: GBM, BRCA, LUAD, etc.)
- **download_dataset(dataset_id, data_type?)** - Download from GEO/ENCODE/TCGA
- **generate_workflow(workflow_type)** - Create workflow
- **list_workflows()** - Show available workflows
- **check_references(organism)** - Verify reference genomes
- **submit_job(workflow?)** - Run on SLURM cluster
- **get_job_status(job_id?)** - Check job progress
- **get_logs(job_id?)** - View job output
- **diagnose_error()** - AI-powered error analysis
- **analyze_results(path?)** - Interpret QC reports
- **explain_concept(concept)** - Explain bioinformatics terms

## Response Style:
- Execute tools, don't just describe them
- Use markdown formatting
- Suggest next steps after each action
"""
    
    def get_openai_tools(self) -> List[Dict]:
        """Get tool definitions for OpenAI function calling."""
        if self._tools:
            return self._tools.get_openai_functions()
        return []
    
    def execute_tool(self, tool_name: str, arguments: Dict, session: SessionState) -> Tuple[str, Dict]:
        """
        Execute a tool and return (message, data).
        
        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            session: Current session for state updates
            
        Returns:
            Tuple of (result_message, result_data)
        """
        if not self._tools:
            return ("⚠️ Tools not available", {"error": "Tools not initialized"})
        
        try:
            # Map argument names for compatibility
            mapped_args = self._map_arguments(tool_name, arguments)
            
            # Execute tool
            result = self._tools.execute_tool(tool_name, **mapped_args)
            
            # Update session state based on results
            self._update_session_from_result(tool_name, result, session)
            
            return (result.message, result.data or {})
            
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}", exc_info=True)
            return (f"❌ Tool error: {e}", {"error": str(e)})
    
    def _map_arguments(self, tool_name: str, arguments: Dict) -> Dict:
        """Map OpenAI function arguments to tool parameters."""
        # Most arguments pass through directly
        mapped = dict(arguments)
        
        # Handle common argument name differences
        if tool_name == "scan_data" and "path" in mapped:
            # scan_data uses 'path' parameter
            pass
        elif tool_name == "download_dataset" and "dataset_id" in mapped:
            # Map dataset_id to the expected parameter
            pass
        elif tool_name == "explain_concept" and "concept" in mapped:
            pass
        elif tool_name == "search_databases" and "query" in mapped:
            pass
        elif tool_name == "search_tcga":
            # Map query to cancer_type/data_type
            if "query" in mapped:
                mapped["cancer_type"] = mapped.pop("query", "")
        
        return mapped
    
    def _update_session_from_result(self, tool_name: str, result, session: SessionState):
        """Update session state based on tool results."""
        if not result.success:
            return
        
        data = result.data or {}
        
        if tool_name == "scan_data":
            session.samples = data.get("samples", [])
            session.data_path = data.get("path")
        
        elif tool_name == "generate_workflow":
            session.current_workflow = data.get("workflow_type")
            session.workflow_path = data.get("output_dir")
        
        elif tool_name == "cleanup_data":
            # Store pending cleanup for confirmation
            session.pending_cleanup = data
        
        elif tool_name == "submit_job":
            job_id = data.get("job_id")
            if job_id:
                session.jobs[job_id] = {"status": "submitted", "submitted_at": datetime.now().isoformat()}
    
    def _get_model_attribution(self, provider: str, model: str) -> str:
        """
        Generate a subtle model attribution footer.
        
        Args:
            provider: Provider name (github, google, openai, local)
            model: Model name
            
        Returns:
            Attribution string (subtle, gray text)
        """
        # Map providers to display names and icons
        provider_info = {
            "github": ("GitHub Models", "🐙", "free"),
            "google": ("Google Gemini", "✨", "free"),
            "openai": ("OpenAI", "🔑", "paid"),
            "local": ("Local vLLM", "🖥️", "free"),
        }
        
        info = provider_info.get(provider, (provider, "🤖", ""))
        display_name, icon, cost = info
        
        # Short model name
        short_model = model.split("/")[-1] if "/" in model else model
        
        # Cost indicator
        cost_badge = " 💰" if cost == "paid" else ""
        
        return f"<sub>*{icon} {display_name}: {short_model}{cost_badge}*</sub>"
    
    def detect_and_execute_tool(self, message: str, session: SessionState) -> Optional[str]:
        """
        Try to detect and execute a tool from the message.
        
        Args:
            message: User's message
            session: Current session
            
        Returns:
            Tool result message if detected, None otherwise
        """
        if not self._tools:
            return None
        
        # Use the modular tool detection
        detected = self._tools.detect_tool(message)
        if not detected:
            return None
        
        # Extract any arguments from the message
        arguments = self._extract_arguments(detected, message)
        
        # Execute the tool
        result_msg, result_data = self.execute_tool(detected, arguments, session)
        
        return result_msg
    
    def _extract_arguments(self, tool_name: str, message: str) -> Dict:
        """Extract arguments from a message for a specific tool."""
        args = {}
        message_lower = message.lower()
        
        # Path extraction
        path_match = re.search(r'(?:in|at|from)\s+["\']?([/~][^\s"\']+)["\']?', message)
        if path_match:
            args["path"] = path_match.group(1)
        
        # Dataset ID extraction
        dataset_match = re.search(r'(GSE\d+|ENCSR[A-Z0-9]+)', message, re.IGNORECASE)
        if dataset_match:
            args["dataset_id"] = dataset_match.group(1)
        
        # Query extraction for search
        if tool_name in ("search_databases", "search_tcga"):
            # Remove command words to get the query
            query = re.sub(r'^(?:search|query|find|look for)\s+(?:for\s+)?', '', message_lower)
            query = re.sub(r'\s+(?:in|on|from)\s+(?:encode|geo|tcga|gdc).*$', '', query)
            args["query"] = query.strip()
        
        # Concept extraction for explain
        if tool_name == "explain_concept":
            concept_match = re.search(r'(?:what\s+is|explain|describe|tell\s+me\s+about)\s+(.+?)(?:\?|$)', message_lower)
            if concept_match:
                args["concept"] = concept_match.group(1).strip()
        
        # Workflow type extraction
        if tool_name == "generate_workflow":
            desc_match = re.search(r'(?:create|generate|make|build)\s+(?:a\s+)?(.+?)(?:\s+workflow|\s+pipeline)?$', message_lower)
            if desc_match:
                args["workflow_type"] = desc_match.group(1).strip()
        
        return args
    
    def chat(
        self, 
        message: str, 
        history: List[Dict] = None,
        session_id: str = None,
    ) -> Generator[List[Dict], None, None]:
        """
        Process a chat message and generate responses.
        
        Args:
            message: User's message
            history: Previous messages (Gradio format)
            session_id: Optional session identifier
            
        Yields:
            Updated history with assistant responses
        """
        session = self.session_manager.get_session(session_id)
        history = history or []
        
        # Add user message
        new_history = history + [{"role": "user", "content": message}]
        session.add_message("user", message)
        
        # STEP 1: Try pattern-based tool execution (fast, free)
        if self._tools_available:
            tool_result = self.detect_and_execute_tool(message, session)
            if tool_result:
                session.add_message("assistant", tool_result)
                yield new_history + [{"role": "assistant", "content": tool_result}]
                return
        
        # STEP 2: Use LLM with function calling
        if not self.llm_provider or not self.llm_provider.available:
            error_msg = "⚠️ No LLM available. Please configure GITHUB_TOKEN, GOOGLE_API_KEY, or OPENAI_API_KEY."
            session.add_message("assistant", error_msg)
            yield new_history + [{"role": "assistant", "content": error_msg}]
            return
        
        client, model = self.llm_provider.get_client_and_model()
        if not client:
            error_msg = "⚠️ LLM client not available."
            yield new_history + [{"role": "assistant", "content": error_msg}]
            return
        
        # Build messages
        messages = [{"role": "system", "content": self.get_system_prompt(session)}]
        messages.extend(session.get_history(limit=10))
        
        try:
            # Try with function calling first (with retry)
            tools = self.get_openai_tools()
            if tools:
                formatted_tools = [{"type": "function", "function": t} for t in tools]
                response = self._call_llm_with_retry(
                    client, model, messages, tools=formatted_tools
                )
            else:
                response = self._call_llm_with_retry(client, model, messages)
            
            assistant_message = response.choices[0].message
            
            # Get model attribution
            provider_name = self.llm_provider.active_provider or "unknown"
            model_info = self._get_model_attribution(provider_name, model)
            
            # Handle tool calls
            if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                results = []
                for tool_call in assistant_message.tool_calls:
                    func_name = tool_call.function.name
                    try:
                        func_args = json.loads(tool_call.function.arguments)
                    except:
                        func_args = {}
                    
                    # Log tool call for debugging
                    logger.info(f"LLM called tool: {func_name} with args: {func_args}")
                    print(f"🔧 Tool called: {func_name}({func_args})")
                    
                    result_msg, result_data = self.execute_tool(func_name, func_args, session)
                    results.append(result_msg)
                
                combined = "\n\n".join(results)
                combined_with_attr = f"{combined}\n\n{model_info}"
                session.add_message("assistant", combined_with_attr)
                yield new_history + [{"role": "assistant", "content": combined_with_attr}]
            
            else:
                # No tool call - return text response
                content = assistant_message.content or ""
                content_with_attr = f"{content}\n\n{model_info}"
                session.add_message("assistant", content_with_attr)
                yield new_history + [{"role": "assistant", "content": content_with_attr}]
        
        except Exception as e:
            error_str = str(e)
            logger.error(f"LLM error: {error_str}")
            
            # Try streaming without tools if function calling failed
            if "tool" in error_str.lower() or "function" in error_str.lower():
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
                    session.add_message("assistant", full_response)
                    return
                except Exception as e2:
                    error_str = f"{error_str}\nStreaming fallback: {e2}"
            
            error_msg = f"❌ Error: {error_str}"
            session.add_message("assistant", error_msg)
            yield new_history + [{"role": "assistant", "content": error_msg}]
    
    def clear_history(self, session_id: str = None):
        """Clear chat history for a session."""
        self.session_manager.clear_session(session_id)
    
    def get_status(self, session_id: str = None) -> str:
        """Get status string for display."""
        session = self.session_manager.get_session(session_id)
        
        llm_status = "🔴 No LLM"
        if self.llm_provider and self.llm_provider.available:
            llm_status = self.llm_provider.status()
        
        return f"{llm_status} | {session.summary()}"


# =============================================================================
# MODULE SINGLETON
# =============================================================================

_chat_handler: Optional[UnifiedChatHandler] = None


def get_chat_handler(llm_provider=None) -> UnifiedChatHandler:
    """Get or create the unified chat handler."""
    global _chat_handler
    if _chat_handler is None:
        # Auto-create LLM provider if not provided
        if llm_provider is None:
            llm_provider = LLMProvider()
        _chat_handler = UnifiedChatHandler(llm_provider=llm_provider)
    return _chat_handler


def chat(message: str, history: List[Dict] = None) -> Generator[List[Dict], None, None]:
    """Convenience function for chat."""
    handler = get_chat_handler()
    yield from handler.chat(message, history)


__all__ = [
    "UnifiedChatHandler",
    "SessionState",
    "SessionManager",
    "get_chat_handler",
    "chat",
]
