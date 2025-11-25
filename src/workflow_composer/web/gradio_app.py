#!/usr/bin/env python3
"""
BioPipelines - Modern Gradio Web Interface
===========================================

A beautiful, interactive web UI for AI-powered bioinformatics workflow generation.

Features:
- Chat-based workflow generation with LLM
- Real-time streaming responses
- Tool and module browser with search
- Workflow visualization
- Multiple LLM provider support (OpenAI, vLLM, Ollama)
- File upload for sample sheets
- Download generated workflows

Usage:
    python -m workflow_composer.web.gradio_app
    # or
    biocomposer ui
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Generator, Optional, List, Dict, Any, Tuple

import gradio as gr

# Import workflow composer components
try:
    from workflow_composer import Composer
    from workflow_composer.core import ToolSelector, ModuleMapper, AnalysisType
    from workflow_composer.llm import get_llm, check_providers, Message
    COMPOSER_AVAILABLE = True
except ImportError:
    COMPOSER_AVAILABLE = False
    print("Warning: workflow_composer not fully installed. Running in demo mode.")


# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).parent.parent.parent.parent
GENERATED_DIR = BASE_DIR / "generated_workflows"
GENERATED_DIR.mkdir(exist_ok=True)

# Theme colors
THEME = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="blue",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    button_primary_background_fill="linear-gradient(135deg, #059669, #10b981)",
    button_primary_background_fill_hover="linear-gradient(135deg, #047857, #059669)",
    block_title_text_color="#1e293b",
    block_label_text_color="#475569",
)

# Analysis type descriptions for better UX
ANALYSIS_EXAMPLES = {
    "RNA-seq": "RNA-seq differential expression analysis for mouse samples comparing treatment vs control using STAR and DESeq2",
    "ChIP-seq": "ChIP-seq peak calling for human H3K27ac samples with input controls using Bowtie2 and MACS2",
    "ATAC-seq": "ATAC-seq chromatin accessibility analysis for human cells with Bowtie2 alignment and MACS2 peak calling",
    "Variant Calling": "Whole exome sequencing variant calling for human samples using BWA-MEM2 and GATK HaplotypeCaller",
    "Single-cell RNA": "10x Genomics single-cell RNA-seq analysis with STARsolo and Seurat clustering",
    "Metagenomics": "Shotgun metagenomics analysis with Kraken2 taxonomic classification and MetaPhlAn profiling",
    "Long-read": "Oxford Nanopore long-read sequencing analysis with minimap2 alignment and structural variant calling",
    "Methylation": "Bisulfite sequencing methylation analysis with Bismark alignment and methylation calling",
}


# ============================================================================
# Global State
# ============================================================================

class AppState:
    """Application state manager."""
    
    def __init__(self):
        self.composer: Optional[Composer] = None
        self.tool_selector: Optional[ToolSelector] = None
        self.module_mapper: Optional[ModuleMapper] = None
        self.current_provider = "openai"
        self.chat_history: List[Tuple[str, str]] = []
        
    def initialize(self, provider: str = "openai", model: str = None):
        """Initialize or reinitialize with a specific provider."""
        try:
            available = check_providers() if COMPOSER_AVAILABLE else {}
            
            if provider not in available or not available.get(provider):
                # Fall back to first available
                for p, is_available in available.items():
                    if is_available:
                        provider = p
                        break
                else:
                    return False, "No LLM providers available"
            
            llm = get_llm(provider, model=model) if model else get_llm(provider)
            self.composer = Composer(llm=llm)
            self.current_provider = provider
            
            # Initialize tool selector and module mapper
            self.tool_selector = self.composer.tool_selector
            self.module_mapper = self.composer.module_mapper
            
            return True, f"Initialized with {provider}"
        except Exception as e:
            return False, str(e)
    
    def get_stats(self) -> Dict[str, int]:
        """Get tool and module statistics."""
        stats = {
            "tools": 0,
            "modules": 0,
            "containers": 12,
            "analysis_types": len(AnalysisType) if COMPOSER_AVAILABLE else 38
        }
        
        if self.tool_selector:
            stats["tools"] = len(self.tool_selector.tools)
        if self.module_mapper:
            stats["modules"] = len(self.module_mapper.modules)
            
        return stats


# Global app state
app_state = AppState()


# ============================================================================
# Core Functions
# ============================================================================

def check_available_providers() -> Dict[str, bool]:
    """Check which LLM providers are available."""
    if not COMPOSER_AVAILABLE:
        return {"openai": False, "vllm": False, "ollama": False, "anthropic": False}
    
    try:
        return check_providers()
    except:
        return {"openai": False, "vllm": False, "ollama": False, "anthropic": False}


def get_provider_choices() -> List[str]:
    """Get list of available provider choices for dropdown."""
    available = check_available_providers()
    choices = []
    
    provider_labels = {
        "openai": "🟢 OpenAI (GPT-4o)",
        "vllm": "🟣 vLLM (Local GPU)",
        "ollama": "🟠 Ollama (Local)",
        "anthropic": "🔵 Anthropic (Claude)",
    }
    
    for provider, is_available in available.items():
        if is_available:
            choices.append(provider_labels.get(provider, provider))
        else:
            # Show unavailable options grayed out
            choices.append(f"⚫ {provider.title()} (not configured)")
    
    return choices if choices else ["⚫ No providers available"]


def extract_provider_key(choice: str) -> str:
    """Extract provider key from dropdown choice."""
    mapping = {
        "openai": "openai",
        "vllm": "vllm",
        "ollama": "ollama",
        "anthropic": "anthropic",
    }
    
    choice_lower = choice.lower()
    for key in mapping:
        if key in choice_lower:
            return key
    return "openai"


def chat_with_composer(
    message: str,
    history: List[Tuple[str, str]],
    provider: str,
) -> Generator[Tuple[List[Tuple[str, str]], str], None, None]:
    """
    Chat with the AI workflow composer.
    Streams responses for better UX.
    """
    if not message.strip():
        yield history, ""
        return
    
    # Extract actual provider
    provider_key = extract_provider_key(provider)
    
    # Initialize if needed
    if app_state.composer is None or app_state.current_provider != provider_key:
        success, msg = app_state.initialize(provider_key)
        if not success:
            history.append((message, f"❌ Error: {msg}"))
            yield history, ""
            return
    
    # Add user message to history
    history.append((message, ""))
    yield history, ""
    
    # Check if this is a workflow generation request
    is_generation_request = any(kw in message.lower() for kw in [
        "generate", "create", "build", "make", "workflow", "pipeline",
        "analyze", "analysis", "process", "run"
    ])
    
    try:
        if is_generation_request and app_state.composer:
            # Generate workflow
            response_parts = []
            
            # First, parse the intent
            yield history[:-1] + [(message, "🔍 Parsing your request...")], ""
            
            intent = app_state.composer.parse_intent(message)
            intent_info = f"""
📋 **Detected Analysis:**
- Type: `{intent.analysis_type.value}`
- Organism: `{intent.organism or 'Not specified'}`
- Genome: `{intent.genome_build or 'Auto-detect'}`
- Confidence: `{intent.confidence:.0%}`

"""
            response_parts.append(intent_info)
            history[-1] = (message, "".join(response_parts) + "🔧 Checking tool availability...")
            yield history, ""
            
            # Check readiness
            readiness = app_state.composer.check_readiness(message)
            
            if readiness.get("ready"):
                tools_found = readiness.get("tools_found", 0)
                modules_found = readiness.get("modules_found", 0)
                
                readiness_info = f"""✅ **Ready to generate!**
- Tools available: `{tools_found}`
- Modules available: `{modules_found}`

"""
                response_parts.append(readiness_info)
                history[-1] = (message, "".join(response_parts) + "⚙️ Generating workflow...")
                yield history, ""
                
                # Generate the workflow
                workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                output_dir = GENERATED_DIR / workflow_id
                
                workflow = app_state.composer.generate(message, output_dir=str(output_dir))
                
                # Format response
                workflow_info = f"""🎉 **Workflow Generated!**

**Name:** `{workflow.name if hasattr(workflow, 'name') else workflow_id}`
**Output:** `{output_dir}`

**Tools used:**
{', '.join(f'`{t}`' for t in (workflow.tools_used if hasattr(workflow, 'tools_used') else []))}

**Modules:**
{', '.join(f'`{m}`' for m in (workflow.modules_used if hasattr(workflow, 'modules_used') else []))}

📥 Use the **Download** tab to get your workflow files.
"""
                response_parts.append(workflow_info)
                
            else:
                issues = readiness.get("issues", ["Unknown issue"])
                response_parts.append(f"""⚠️ **Cannot generate workflow:**
{chr(10).join(f'- {issue}' for issue in issues)}

Please provide more details or check tool availability.
""")
            
            history[-1] = (message, "".join(response_parts))
            yield history, ""
            
        else:
            # Regular chat - use LLM for conversational response
            if app_state.composer and app_state.composer.llm:
                # Create system context
                system_msg = """You are BioPipelines AI Assistant, an expert in bioinformatics workflow design.
You help users:
1. Design and generate bioinformatics pipelines (RNA-seq, ChIP-seq, variant calling, etc.)
2. Explain bioinformatics tools and methods
3. Troubleshoot pipeline issues
4. Recommend best practices

Be concise but helpful. Use markdown formatting."""
                
                messages = [Message.system(system_msg)]
                
                # Add chat history
                for user_msg, assistant_msg in history[:-1]:
                    messages.append(Message.user(user_msg))
                    if assistant_msg:
                        messages.append(Message.assistant(assistant_msg))
                
                messages.append(Message.user(message))
                
                # Get response (streaming if supported)
                response = app_state.composer.llm.chat(messages)
                history[-1] = (message, response.content)
                yield history, ""
            else:
                history[-1] = (message, "I'm not fully initialized. Please select an LLM provider.")
                yield history, ""
                
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        history[-1] = (message, error_msg)
        yield history, ""


def search_tools(query: str, container_filter: str = "") -> str:
    """Search for bioinformatics tools."""
    if not query or len(query) < 2:
        return "Enter at least 2 characters to search..."
    
    if not app_state.tool_selector:
        # Return demo data
        return """
| Tool | Container | Category |
|------|-----------|----------|
| fastqc | base | QC |
| multiqc | base | QC |
| star | rna-seq | Alignment |
| bwa | dna-seq | Alignment |
| gatk | dna-seq | Variant Calling |
        """
    
    try:
        results = app_state.tool_selector.fuzzy_search(
            query, 
            limit=20,
            container=container_filter if container_filter else None
        )
        
        if not results:
            return "No tools found matching your query."
        
        # Format as markdown table
        table = "| Tool | Container | Category | Score |\n|------|-----------|----------|-------|\n"
        for match in results[:15]:
            tool = match.tool
            table += f"| `{tool.name}` | {tool.container} | {tool.category or '-'} | {match.score:.2f} |\n"
        
        return table
    except Exception as e:
        return f"Search error: {e}"


def get_modules_by_category() -> str:
    """Get all modules organized by category."""
    if not app_state.module_mapper:
        # Return demo data
        return """
## 📦 Available Modules

### QC
- fastqc
- multiqc
- fastp

### Alignment
- star
- bwa
- bowtie2
- minimap2

### Quantification
- featurecounts
- salmon
- kallisto

### Variant Calling
- gatk_haplotypecaller
- bcftools
- deepvariant
        """
    
    try:
        modules = app_state.module_mapper.list_by_category()
        
        output = "## 📦 Available Modules\n\n"
        for category, mods in sorted(modules.items()):
            output += f"### {category.title()}\n"
            for mod in sorted(mods)[:10]:
                output += f"- `{mod}`\n"
            if len(mods) > 10:
                output += f"- *... and {len(mods) - 10} more*\n"
            output += "\n"
        
        return output
    except Exception as e:
        return f"Error loading modules: {e}"


def get_example_prompts() -> List[List[str]]:
    """Get example prompts for the UI."""
    return [
        [example] for example in ANALYSIS_EXAMPLES.values()
    ]


def download_latest_workflow() -> Optional[str]:
    """Get path to latest generated workflow for download."""
    try:
        workflows = sorted(GENERATED_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
        if workflows:
            latest = workflows[0]
            zip_path = GENERATED_DIR / f"{latest.name}.zip"
            shutil.make_archive(str(zip_path.with_suffix('')), 'zip', latest)
            return str(zip_path)
    except Exception as e:
        print(f"Download error: {e}")
    return None


def refresh_stats() -> Tuple[str, str, str, str]:
    """Refresh and return statistics."""
    stats = app_state.get_stats()
    return (
        f"🔧 {stats['tools']}",
        f"📦 {stats['modules']}",
        f"🐳 {stats['containers']}",
        f"🧬 {stats['analysis_types']}"
    )


# ============================================================================
# Gradio Interface
# ============================================================================

def create_interface() -> gr.Blocks:
    """Create the Gradio interface."""
    
    with gr.Blocks(
        title="BioPipelines - AI Workflow Composer",
        theme=THEME,
        css="""
        .main-header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #059669 0%, #3b82f6 100%);
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .main-header h1 {
            color: white !important;
            margin: 0;
            font-size: 2.2em;
        }
        .main-header p {
            color: rgba(255,255,255,0.9) !important;
            margin: 5px 0 0 0;
        }
        .stat-box {
            text-align: center;
            padding: 15px;
            background: linear-gradient(135deg, #f8fafc, #f1f5f9);
            border-radius: 10px;
            border: 1px solid #e2e8f0;
        }
        .stat-box .stat-number {
            font-size: 1.8em;
            font-weight: bold;
            color: #059669;
        }
        .example-btn {
            font-size: 0.9em !important;
        }
        footer {
            display: none !important;
        }
        """
    ) as demo:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🧬 BioPipelines</h1>
            <p>AI-Powered Bioinformatics Workflow Composer</p>
        </div>
        """)
        
        # Stats Row
        with gr.Row():
            tools_stat = gr.Markdown("🔧 Loading...", elem_classes=["stat-box"])
            modules_stat = gr.Markdown("📦 Loading...", elem_classes=["stat-box"])
            containers_stat = gr.Markdown("🐳 12", elem_classes=["stat-box"])
            analyses_stat = gr.Markdown("🧬 38", elem_classes=["stat-box"])
        
        # Main Tabs
        with gr.Tabs():
            
            # ========== Chat Tab ==========
            with gr.TabItem("💬 Chat & Generate", id="chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="BioPipelines AI",
                            height=450,
                            show_copy_button=True,
                            avatar_images=(None, "https://raw.githubusercontent.com/gradio-app/gradio/main/gradio/icons/logo.png"),
                        )
                        
                        with gr.Row():
                            msg_input = gr.Textbox(
                                label="Your message",
                                placeholder="Describe the bioinformatics analysis you want to perform...",
                                lines=2,
                                scale=4,
                            )
                            send_btn = gr.Button("Send 🚀", variant="primary", scale=1)
                        
                        with gr.Accordion("📝 Example Prompts", open=False):
                            gr.Examples(
                                examples=get_example_prompts(),
                                inputs=msg_input,
                                label="Click an example to use it:",
                            )
                    
                    with gr.Column(scale=1):
                        provider_dropdown = gr.Dropdown(
                            choices=get_provider_choices(),
                            value=get_provider_choices()[0] if get_provider_choices() else None,
                            label="🤖 LLM Provider",
                            interactive=True,
                        )
                        
                        gr.Markdown("### Quick Actions")
                        
                        clear_btn = gr.Button("🗑️ Clear Chat", size="sm")
                        
                        gr.Markdown("""
                        ### Tips
                        - Be specific about organism and genome
                        - Mention specific tools if preferred
                        - Include sample type (paired-end, etc.)
                        - Describe comparison groups
                        """)
            
            # ========== Tools Tab ==========
            with gr.TabItem("🔧 Tool Browser", id="tools"):
                with gr.Row():
                    tool_search = gr.Textbox(
                        label="Search Tools",
                        placeholder="e.g., alignment, variant, fastq...",
                        scale=3,
                    )
                    container_filter = gr.Dropdown(
                        choices=["", "base", "rna-seq", "dna-seq", "chip-seq", "atac-seq", 
                                 "scrna-seq", "metagenomics", "methylation", "long-read"],
                        label="Filter by Container",
                        scale=1,
                    )
                
                tool_results = gr.Markdown("Enter a search term to find tools...")
                
                tool_search.change(
                    fn=search_tools,
                    inputs=[tool_search, container_filter],
                    outputs=tool_results,
                )
                container_filter.change(
                    fn=search_tools,
                    inputs=[tool_search, container_filter],
                    outputs=tool_results,
                )
            
            # ========== Modules Tab ==========
            with gr.TabItem("📦 Modules", id="modules"):
                modules_display = gr.Markdown(get_modules_by_category())
                refresh_modules_btn = gr.Button("🔄 Refresh Modules")
                refresh_modules_btn.click(
                    fn=get_modules_by_category,
                    outputs=modules_display,
                )
            
            # ========== Download Tab ==========
            with gr.TabItem("📥 Download", id="download"):
                gr.Markdown("""
                ## Download Generated Workflows
                
                After generating a workflow in the Chat tab, you can download it here.
                The download includes:
                - `main.nf` - Main Nextflow workflow
                - `nextflow.config` - Configuration file
                - `modules/` - Required modules
                - `README.md` - Usage instructions
                """)
                
                download_btn = gr.Button("📥 Download Latest Workflow", variant="primary")
                download_file = gr.File(label="Download")
                
                download_btn.click(
                    fn=download_latest_workflow,
                    outputs=download_file,
                )
            
            # ========== Settings Tab ==========
            with gr.TabItem("⚙️ Settings", id="settings"):
                gr.Markdown("## LLM Configuration")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### OpenAI")
                        openai_status = gr.Markdown(
                            "✅ Configured" if os.getenv("OPENAI_API_KEY") else "❌ Not configured"
                        )
                        gr.Markdown("Set `OPENAI_API_KEY` environment variable")
                    
                    with gr.Column():
                        gr.Markdown("### vLLM")
                        vllm_url = gr.Textbox(
                            label="vLLM Server URL",
                            value=os.getenv("VLLM_API_BASE", "http://localhost:8000/v1"),
                            interactive=True,
                        )
                        vllm_model = gr.Dropdown(
                            choices=["llama3.1-8b", "mistral-7b", "qwen2.5-7b", "codellama-34b"],
                            label="Model",
                            value="mistral-7b",
                        )
                
                gr.Markdown("---")
                gr.Markdown("### System Info")
                
                system_info = gr.Markdown(f"""
                - **Workflow Composer:** {'✅ Available' if COMPOSER_AVAILABLE else '❌ Not installed'}
                - **Generated Workflows Dir:** `{GENERATED_DIR}`
                - **Python Path:** `{Path(__file__).parent}`
                """)
        
        # ========== Event Handlers ==========
        
        # Chat submission
        msg_input.submit(
            fn=chat_with_composer,
            inputs=[msg_input, chatbot, provider_dropdown],
            outputs=[chatbot, msg_input],
        )
        
        send_btn.click(
            fn=chat_with_composer,
            inputs=[msg_input, chatbot, provider_dropdown],
            outputs=[chatbot, msg_input],
        )
        
        # Clear chat
        clear_btn.click(
            fn=lambda: ([], ""),
            outputs=[chatbot, msg_input],
        )
        
        # Load stats on start
        demo.load(
            fn=refresh_stats,
            outputs=[tools_stat, modules_stat, containers_stat, analyses_stat],
        )
        
        # Initialize app state on provider change
        provider_dropdown.change(
            fn=lambda p: app_state.initialize(extract_provider_key(p)),
            inputs=[provider_dropdown],
        )
    
    return demo


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Launch the Gradio web interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BioPipelines Web UI")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║        🧬 BioPipelines - AI Workflow Composer                    ║
║                      Web Interface                               ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Starting Gradio server...                                       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize app state
    available = check_available_providers()
    if any(available.values()):
        provider = next(k for k, v in available.items() if v)
        app_state.initialize(provider)
        print(f"  ✅ Initialized with {provider} provider")
    else:
        print("  ⚠️  No LLM providers available - running in demo mode")
    
    # Create and launch interface
    demo = create_interface()
    
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True,
    )


if __name__ == "__main__":
    main()
