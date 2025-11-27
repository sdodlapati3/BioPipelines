"""
BioPipelines - Chat-First Web Interface (Gradio 6.x)

A minimal, focused UI where chat is the primary interface.
All features are accessible through natural conversation.
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator
from datetime import datetime

import gradio as gr

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# Configuration
# ============================================================================

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1")
USE_LOCAL_LLM = os.environ.get("USE_LOCAL_LLM", "false").lower() == "true"
DEFAULT_PORT = int(os.environ.get("GRADIO_PORT", "7860"))

# ============================================================================
# LLM Provider Management - Fallback Chain
# ============================================================================

class LLMProvider:
    """Manages LLM providers with automatic fallback."""
    
    PROVIDERS = [
        ("local", "Local vLLM", None),
        ("openai", "OpenAI", "OPENAI_API_KEY"),
        ("google", "Google Gemini", "GOOGLE_API_KEY"),
        ("lightning", "Lightning AI", "LIGHTNING_API_KEY"),
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
        self.clients["local"] = OpenAI(base_url=VLLM_URL, api_key="not-needed")
        self.models["local"] = "MiniMaxAI/MiniMax-M2"
        
        # 2. OpenAI
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if openai_key:
            self.clients["openai"] = OpenAI(api_key=openai_key)
            self.models["openai"] = "gpt-4o-mini"  # Cost-effective default
            print("✓ OpenAI configured")
        
        # 3. Google Gemini (via OpenAI-compatible API)
        google_key = os.environ.get("GOOGLE_API_KEY", "")
        if google_key:
            self.clients["google"] = OpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=google_key
            )
            self.models["google"] = "gemini-2.0-flash"
            print("✓ Google Gemini configured")
        
        # 4. Lightning AI
        lightning_key = os.environ.get("LIGHTNING_API_KEY", "")
        if lightning_key:
            self.clients["lightning"] = OpenAI(
                base_url="https://api.lightning.ai/v1",
                api_key=lightning_key
            )
            self.models["lightning"] = "nvidia/Llama-3.1-Nemotron-Ultra-253B-v1"
            print("✓ Lightning AI configured")
        
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
        """Check if local vLLM is ready."""
        if not USE_LOCAL_LLM:
            return False
        try:
            import requests
            resp = requests.get(f"{VLLM_URL.replace('/v1', '')}/health", timeout=2)
            return resp.status_code == 200
        except:
            return False
    
    def _find_working_provider(self) -> Optional[str]:
        """Find the first working cloud provider."""
        for name in ["openai", "google", "lightning"]:
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
# Agent Setup  
# ============================================================================

agent = None
AGENT_AVAILABLE = False
agent_tools = None

try:
    from workflow_composer.agents.autonomous.agent import AutonomousAgent
    agent = AutonomousAgent()
    AGENT_AVAILABLE = True
    print("✓ Autonomous agent available")
except ImportError as e:
    print(f"⚠ Agent not available: {e}")

# Import tools for direct execution
try:
    from workflow_composer.agents.tools import AgentTools, ToolName
    agent_tools = AgentTools()
    print("✓ Agent tools available")
except ImportError as e:
    print(f"⚠ Agent tools not available: {e}")

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
    return f"""You are BioPipelines Assistant, an AI that helps researchers run bioinformatics analyses.

## Current Context:
{state.to_context_string()}

## Your Capabilities:
1. **Scan Data**: Analyze FASTQ/BAM files in a directory
2. **Create Workflows**: Generate Nextflow pipelines (RNA-seq, ChIP-seq, WGS, etc.)
3. **Execute Pipelines**: Submit jobs to the cluster
4. **Monitor Jobs**: Check status, view logs
5. **Analyze Results**: View QC reports, statistics

## Available Tools:
You have access to these tools. Call them when appropriate:

1. **scan_data(path?)** - Scan a directory for FASTQ/BAM files. If no path given, scans default data directory.
2. **cleanup_data(path?)** - Remove corrupted data files (HTML error pages, invalid gzips). 
3. **search_databases(query)** - Search ENCODE, GEO for public datasets.
4. **check_references(organism)** - Check if reference genome is available.
5. **list_workflows()** - Show available workflow templates.

## Response Style:
- Be concise and helpful
- Use markdown for formatting  
- When showing code, use code blocks
- Proactively suggest next steps

IMPORTANT: When the user asks about their data, files, samples - USE the scan_data tool. 
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
            "name": "cleanup_data",
            "description": "Scan for corrupted data files (HTML error pages masquerading as FASTQ files, invalid gzip files, broken symlinks). Shows a preview first and requires confirmation before deleting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to clean. If not provided, cleans default data directory."
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
            "description": "Confirm and execute a pending cleanup operation. Call this when the user confirms they want to delete corrupted files after seeing the preview.",
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
            "description": "Search public databases (ENCODE, GEO, SRA) for datasets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, e.g. 'human RNA-seq liver cancer'"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_workflows",
            "description": "List available bioinformatics workflow templates (RNA-seq, ChIP-seq, methylation, etc.)",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


def execute_tool_call(tool_name: str, arguments: dict) -> str:
    """Execute a tool and return the result message."""
    if not agent_tools:
        return "⚠️ Tools not available"
    
    try:
        if tool_name == "scan_data":
            result = agent_tools.scan_data(arguments.get("path"))
            if result.success and result.data:
                state.samples = result.data.get("samples", [])
                state.data_path = result.data.get("path")
            return result.message
        
        elif tool_name == "cleanup_data":
            # Always show preview first (confirm=False)
            result = agent_tools.cleanup_data(arguments.get("path"), confirm=False)
            return result.message
        
        elif tool_name == "confirm_cleanup":
            # User confirmed deletion
            result = agent_tools.confirm_cleanup()
            return result.message
        
        elif tool_name == "search_databases":
            result = agent_tools.search_databases(arguments.get("query", ""))
            return result.message
        
        elif tool_name == "list_workflows":
            result = agent_tools.list_workflows()
            return result.message
        
        else:
            return f"⚠️ Unknown tool: {tool_name}"
    
    except Exception as e:
        return f"❌ Tool error: {e}"


def try_pattern_match(message: str) -> Optional[str]:
    """
    Try to match common patterns first (fast, no LLM cost).
    Returns tool result if matched, None otherwise.
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
            ToolName.LIST_WORKFLOWS: ("list_workflows", {}),
            ToolName.CHECK_REFERENCES: ("check_references", {"organism": params[0] if params else "human"}),
        }
        
        if tool_name in tool_map:
            func_name, args = tool_map[tool_name]
            return execute_tool_call(func_name, args)
        
    except Exception as e:
        # Pattern matching failed, will fall through to LLM
        pass
    
    return None


def chat_response(message: str, history: List[Dict]) -> Generator[List[Dict], None, None]:
    """Generate chat response with hybrid approach: patterns first, then LLM."""
    
    # Add user message to history immediately
    new_history = history + [{"role": "user", "content": message}]
    
    # STEP 1: Try fast pattern matching first (no LLM cost)
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
            # Execute all tool calls
            tool_results = []
            for tool_call in assistant_message.tool_calls:
                func_name = tool_call.function.name
                try:
                    func_args = json.loads(tool_call.function.arguments)
                except:
                    func_args = {}
                
                result = execute_tool_call(func_name, func_args)
                tool_results.append(result)
            
            # Combine tool results
            combined_result = "\n\n".join(tool_results)
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
