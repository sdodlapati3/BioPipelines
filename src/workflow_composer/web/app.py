"""
BioPipelines - Chat-First Web Interface (Gradio 6.x)
====================================================

A minimal, focused UI where chat is the primary interface.
All features are accessible through natural conversation.

This version uses:
- BioPipelines facade (the single entry point)
- ModelOrchestrator for LLM routing
- Session management built into the facade
- Job Status Panel for monitoring SLURM jobs
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Generator

import gradio as gr

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# Configuration
# ============================================================================

from workflow_composer.web.utils import detect_vllm_endpoint, get_default_port, use_local_llm
from workflow_composer.web.components.job_panel import (
    get_user_jobs, format_jobs_table, get_recent_jobs, get_job_log, cancel_job
)

VLLM_URL = detect_vllm_endpoint()
USE_LOCAL_LLM = use_local_llm()
DEFAULT_PORT = get_default_port()

# ============================================================================
# Import BioPipelines Facade
# ============================================================================

bp = None
BP_AVAILABLE = False

try:
    from workflow_composer import BioPipelines
    bp = BioPipelines()
    BP_AVAILABLE = True
    print("✓ BioPipelines facade initialized")
except Exception as e:
    print(f"⚠ BioPipelines facade not available: {e}")
    import traceback
    traceback.print_exc()


# ============================================================================
# Chat Response Generator
# ============================================================================

# Session ID per web client (simplified - in production, use cookies/auth)
_session_ids = {}


def get_or_create_session(request: gr.Request = None) -> str:
    """Get or create a session for the web client."""
    if not bp:
        return None
    
    # Use request client ID if available, otherwise default
    client_id = "web_default"
    if request and hasattr(request, 'client'):
        client_id = f"web_{request.client.host}_{request.client.port}"
    
    if client_id not in _session_ids:
        try:
            _session_ids[client_id] = bp.create_session(client_id)
        except Exception:
            return None
    
    return _session_ids.get(client_id)


def chat_response(message: str, history: List[Dict], request: gr.Request = None) -> Generator[List[Dict], None, None]:
    """
    Generate chat response using the BioPipelines facade with streaming.
    
    Supports session management for multi-turn conversations.
    Intelligently routes to multi-agent system for workflow generation queries.
    
    Args:
        message: User's input message
        history: Chat history as list of dicts with "role" and "content"
        request: Gradio request for session identification
    
    Yields:
        Updated history with assistant response (progressively)
    """
    if not message.strip():
        yield history
        return
    
    if not bp:
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "⚠️ BioPipelines not available."}
        ]
        return
    
    # Get session for this client
    session_id = get_or_create_session(request)
    
    # Add user message to history first
    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""}
    ]
    
    # Check if this is a workflow generation request that should use multi-agent
    query_lower = message.lower()
    is_workflow_generation = any(phrase in query_lower for phrase in [
        'generate workflow', 'create workflow', 'build workflow', 'make workflow',
        'generate pipeline', 'create pipeline', 'build pipeline',
        'create a', 'generate a', 'build a'  # + "workflow/pipeline" context
    ]) and any(word in query_lower for word in ['workflow', 'pipeline', 'analysis'])
    
    try:
        if is_workflow_generation:
            # Use multi-agent system for high-quality workflow generation
            yield from _stream_multiagent_response(message, new_history, session_id)
        elif hasattr(bp, 'chat_stream'):
            # Use standard chat stream for other queries
            full_response = ""
            for chunk in bp.chat_stream(message, history=history, session_id=session_id):
                full_response += chunk
                new_history[-1]["content"] = full_response
                yield new_history
        else:
            # Fallback to non-streaming
            response = bp.chat(message, history=history, session_id=session_id)
            new_history[-1]["content"] = response.message
            yield new_history
    except Exception as e:
        new_history[-1]["content"] = f"❌ Error: {e}"
        yield new_history


def _stream_multiagent_response(
    message: str, 
    history: List[Dict], 
    session_id: str
) -> Generator[List[Dict], None, None]:
    """
    Stream multi-agent workflow generation responses into chat history.
    
    Args:
        message: User's workflow generation request
        history: Current chat history (with empty assistant message at end)
        session_id: Session ID for tracking
    
    Yields:
        Updated history with progressive multi-agent updates
    """
    import asyncio
    
    try:
        output_parts = ["🤖 **Starting Multi-Agent Workflow Generation**\n\n"]
        history[-1]["content"] = "".join(output_parts)
        yield history
        
        async def stream_updates():
            async for update in bp.generate_with_agents_streaming(message):
                phase = update.get("phase", "")
                status = update.get("status", "")
                
                if phase == "planning":
                    if status == "started":
                        output_parts.append("📋 **Planning**: Analyzing requirements...\n")
                    elif status == "complete":
                        plan_info = update.get("plan", {})
                        output_parts.append(f"📋 **Planning**: Complete - {plan_info.get('steps', 0)} steps planned\n")
                
                elif phase == "codegen":
                    if status == "started":
                        output_parts.append("💻 **Generating Code**: Writing Nextflow DSL2...\n")
                    elif status == "complete":
                        lines = update.get("lines", 0)
                        output_parts.append(f"💻 **Code Generation**: Complete - {lines} lines\n")
                
                elif phase == "validation":
                    if status == "started":
                        output_parts.append("🔍 **Validating**: Running static analysis + LLM review...\n")
                    elif status == "complete":
                        issues = update.get("issues", 0)
                        valid = update.get("valid", False)
                        status_text = "✓ Passed" if valid else f"⚠️ {issues} issues"
                        output_parts.append(f"🔍 **Validation**: {status_text}\n")
                
                elif phase == "documentation":
                    if status == "started":
                        output_parts.append("📝 **Documentation**: Generating README...\n")
                    elif status == "complete":
                        output_parts.append("📝 **Documentation**: Complete\n")
                
                elif phase == "complete":
                    result = update.get("result", {})
                    output_parts.append("\n---\n")
                    output_parts.append("### ✅ Workflow Generated Successfully!\n\n")
                    output_parts.append(f"**Name:** {result.get('name', 'workflow')}\n")
                    output_parts.append(f"**Lines of code:** {result.get('code_lines', 0)}\n")
                    output_parts.append(f"**Validation:** {'Passed ✓' if result.get('validation_passed') else 'Has issues'}\n")
                    output_parts.append("\n*Tip: Say \"show workflow\" or \"run workflow\" to see or execute it.*")
                
                elif phase == "error":
                    output_parts.append(f"\n❌ **Error**: {update.get('error', 'Unknown error')}\n")
                
                yield "".join(output_parts)
        
        # Run async generator
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            gen = stream_updates()
            while True:
                try:
                    result = loop.run_until_complete(gen.__anext__())
                    history[-1]["content"] = result
                    yield history
                except StopAsyncIteration:
                    break
        finally:
            loop.close()
            
    except ImportError:
        # Multi-agent not available, fallback to regular chat
        history[-1]["content"] = "⚠️ Multi-agent system not available, using standard generation...\n\n"
        yield history
        
        # Use regular chat_stream as fallback
        full_response = history[-1]["content"]
        for chunk in bp.chat_stream(message, session_id=session_id):
            full_response += chunk
            history[-1]["content"] = full_response
            yield history
    except Exception as e:
        history[-1]["content"] += f"\n❌ Error: {e}"
        yield history


# ============================================================================
# Status Functions
# ============================================================================

def get_status() -> str:
    """Get current status as HTML."""
    if bp:
        health = bp.health_check()
        status_parts = []
        if health.get("llm_available"):
            status_parts.append(f"🟢 LLM: {health.get('llm_provider', 'unknown')}")
        else:
            status_parts.append("🔴 No LLM")
        if health.get("tools_available"):
            status_parts.append(f"🛠️ {health.get('tool_count', 0)} tools")
        return " | ".join(status_parts) if status_parts else "🟢 Ready"
    return "🔴 BioPipelines not available"


# ============================================================================
# Example Messages
# ============================================================================

EXAMPLES = [
    {"text": "Help me get started with BioPipelines", "display_text": "🚀 Get Started"},
    {"text": "Scan my data in ~/data/fastq", "display_text": "📁 Scan Data"},
    {"text": "Create an RNA-seq differential expression workflow", "display_text": "🧬 RNA-seq"},
    {"text": "What workflows are available?", "display_text": "📋 List Workflows"},
    {"text": "Show me my running jobs", "display_text": "📊 Check Jobs"},
    {"text": "Search ENCODE for H3K27ac ChIP-seq in liver", "display_text": "🔬 Search ENCODE"},
    {"text": "Search TCGA for lung cancer RNA-seq data", "display_text": "🏥 Search TCGA"},
]


# ============================================================================
# Multi-Agent Workflow Generation
# ============================================================================

def generate_workflow_with_agents(description: str) -> Generator[str, None, None]:
    """
    Generate workflow using multi-agent system with streaming progress.
    
    Args:
        description: Natural language workflow description
    
    Yields:
        Progress updates as markdown strings
    """
    if not bp:
        yield "❌ BioPipelines not available"
        return
    
    try:
        from workflow_composer.agents.specialists import WorkflowState
        
        state_emojis = {
            WorkflowState.IDLE: "⏳",
            WorkflowState.PLANNING: "📋",
            WorkflowState.GENERATING: "🔧",
            WorkflowState.VALIDATING: "🔍",
            WorkflowState.FIXING: "🔨",
            WorkflowState.DOCUMENTING: "📝",
            WorkflowState.COMPLETE: "✅",
            WorkflowState.FAILED: "❌"
        }
        
        output_lines = ["## 🤖 Multi-Agent Workflow Generation\n"]
        output_lines.append(f"**Query:** {description}\n")
        output_lines.append("---\n")
        
        import asyncio
        
        async def stream_updates():
            async for update in bp.generate_with_agents_streaming(description):
                state = update.get("state", WorkflowState.IDLE)
                emoji = state_emojis.get(state, "🔄")
                
                if "message" in update:
                    output_lines.append(f"{emoji} **{state.name}**: {update['message']}\n")
                
                if state == WorkflowState.COMPLETE and "result" in update:
                    result = update["result"]
                    output_lines.append("\n---\n")
                    output_lines.append("### ✅ Workflow Generated Successfully!\n")
                    output_lines.append(f"- **Validation:** {'Passed ✓' if result.validation_passed else 'Issues found ⚠️'}\n")
                    if result.output_files:
                        output_lines.append("- **Output files:**\n")
                        for f in result.output_files:
                            output_lines.append(f"  - `{f}`\n")
                
                yield "\n".join(output_lines)
        
        # Run async generator synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            gen = stream_updates()
            while True:
                try:
                    result = loop.run_until_complete(gen.__anext__())
                    yield result
                except StopAsyncIteration:
                    break
        finally:
            loop.close()
            
    except ImportError as e:
        yield f"❌ Multi-agent system not available: {e}"
    except Exception as e:
        yield f"❌ Error during generation: {e}"


# ============================================================================
# Main UI
# ============================================================================

def create_app() -> gr.Blocks:
    """Create the Gradio app."""
    
    with gr.Blocks(title="🧬 BioPipelines") as app:
        
        # Header
        gr.Markdown("# 🧬 BioPipelines\n*AI-powered bioinformatics workflow composer*")
        
        # Main layout: Chat + Job Panel side by side
        with gr.Row():
            # Main Chat Column (75%)
            with gr.Column(scale=3):
                # Main Chat
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=450,
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
            
            # Job Status Panel (25%)
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### 📊 Jobs")
                
                # Active jobs with auto-refresh
                with gr.Group():
                    jobs_refresh_btn = gr.Button("🔄 Refresh", size="sm")
                    active_jobs_html = gr.HTML(value="<p>Loading...</p>")
                
                # Recent jobs (collapsed)
                with gr.Accordion("📋 Recent (24h)", open=False):
                    recent_jobs_html = gr.HTML(value="<p>Click refresh...</p>")
                
                # Quick job actions
                with gr.Accordion("⚡ Quick Actions", open=False):
                    quick_job_id = gr.Textbox(
                        label="Job ID",
                        placeholder="Enter job ID...",
                        scale=1,
                    )
                    with gr.Row():
                        view_log_btn = gr.Button("📄 Log", size="sm")
                        cancel_job_btn = gr.Button("❌ Cancel", size="sm", variant="stop")
                    job_action_output = gr.Markdown("")
        
        # Feedback section (for intent corrections)
        with gr.Accordion("📝 Feedback & Learning", open=False):
            gr.Markdown("**Help improve the assistant by correcting intent classifications:**")
            with gr.Row():
                feedback_query = gr.Textbox(
                    label="Query",
                    placeholder="The query that was misclassified...",
                    scale=3,
                )
                feedback_intent = gr.Dropdown(
                    label="Correct Intent",
                    choices=[
                        "DATA_SEARCH", "DATA_DOWNLOAD", "DATA_SCAN", "DATA_VALIDATE",
                        "WORKFLOW_CREATE", "WORKFLOW_LIST", "WORKFLOW_CONFIGURE",
                        "JOB_SUBMIT", "JOB_STATUS", "JOB_CANCEL", "JOB_LOGS",
                        "ANALYSIS_QC", "ANALYSIS_RESULTS", "ANALYSIS_COMPARE",
                        "META_HELP", "META_EXPLAIN", "META_STATUS",
                    ],
                    scale=2,
                )
            feedback_text = gr.Textbox(
                label="Additional Feedback (optional)",
                placeholder="Any context that might help...",
                lines=1,
            )
            feedback_btn = gr.Button("Submit Feedback", variant="secondary")
            feedback_result = gr.Markdown("")
            
            # Learning stats
            with gr.Row():
                stats_btn = gr.Button("📊 Show Stats")
                stats_output = gr.JSON(label="Learning Statistics")
        
        # Advanced Multi-Agent Generation (now integrated into chat - this is for manual override)
        with gr.Accordion("🤖 Manual Multi-Agent Generation", open=False):
            gr.Markdown("""
            **Note:** Multi-agent generation is now **automatically triggered** when you ask to create workflows in chat!
            
            This panel is for manual override when you want explicit control over the generation process.
            
            **Specialist Agents:**
            - 📋 Planner → 💻 CodeGen → 🔍 Validator → 📝 DocAgent → ✅ QCAgent
            """)
            
            with gr.Row():
                agent_query = gr.Textbox(
                    label="Workflow Description",
                    placeholder="Describe your bioinformatics workflow...",
                    scale=4,
                )
                agent_output_dir = gr.Textbox(
                    label="Output Directory (optional)",
                    placeholder="e.g., ./my_workflow",
                    scale=2,
                )
            
            agent_generate_btn = gr.Button("🚀 Generate with Multi-Agent System", variant="primary")
            agent_progress = gr.Markdown("*Click 'Generate' to start...*")
        
        # Settings (collapsed)
        with gr.Accordion("⚙️ Settings", open=False):
            settings_info = "**BioPipelines:** " + ("Available ✓" if BP_AVAILABLE else "Not available")
            if bp:
                health = bp.health_check()
                settings_info += f"\n**LLM:** {health.get('llm_provider', 'Not configured')}"
                settings_info += f"\n**Tools:** {health.get('tool_count', 0)}"
            gr.Markdown(settings_info)
            clear = gr.Button("🗑️ Clear Chat")
        
        # Event handlers
        def submit(message, history):
            if not message.strip():
                return history, ""
            for response in chat_response(message, history):
                yield response, ""
        
        def clear_chat():
            return [], ""
        
        def submit_feedback(query, intent, text):
            if not bp:
                return "❌ BioPipelines not available"
            if not query or not intent:
                return "⚠️ Please provide both query and correct intent"
            # Feedback through the agent if available
            try:
                if hasattr(bp, 'agent') and bp.agent:
                    bp.agent.submit_feedback(query, intent, text)
                    return "✅ Feedback recorded!"
                return "⚠️ Feedback system not available"
            except Exception as e:
                return f"❌ {e}"
        
        def get_learning_stats():
            if not bp:
                return {}
            try:
                if hasattr(bp, 'agent') and bp.agent and hasattr(bp.agent, 'get_learning_stats'):
                    return bp.agent.get_learning_stats()
            except Exception:
                pass
            return {"message": "Stats not available"}
        
        # Wire up events
        msg.submit(submit, [msg, chatbot], [chatbot, msg])
        send.click(submit, [msg, chatbot], [chatbot, msg])
        clear.click(clear_chat, outputs=[chatbot, msg])
        feedback_btn.click(submit_feedback, [feedback_query, feedback_intent, feedback_text], feedback_result)
        stats_btn.click(get_learning_stats, outputs=stats_output)
        
        # Multi-agent generation event
        def run_agent_generation(query, output_dir):
            if not query or not query.strip():
                yield "*Please enter a workflow description*"
                return
            for update in generate_workflow_with_agents(query):
                yield update
        
        agent_generate_btn.click(
            run_agent_generation,
            inputs=[agent_query, agent_output_dir],
            outputs=agent_progress
        )
        
        # Job panel events
        def refresh_all_jobs():
            """Refresh both active and recent job panels."""
            return format_jobs_table(get_user_jobs()), format_jobs_table(get_recent_jobs())
        
        def view_job_log(job_id):
            if not job_id or not job_id.strip():
                return "Enter a job ID"
            return get_job_log(job_id.strip())
        
        def cancel_slurm_job(job_id):
            if not job_id or not job_id.strip():
                return "Enter a job ID"
            return cancel_job(job_id.strip())
        
        jobs_refresh_btn.click(refresh_all_jobs, outputs=[active_jobs_html, recent_jobs_html])
        view_log_btn.click(view_job_log, inputs=[quick_job_id], outputs=[job_action_output])
        cancel_job_btn.click(cancel_slurm_job, inputs=[quick_job_id], outputs=[job_action_output])
        
        # Auto-refresh both panels every 30 seconds
        job_timer = gr.Timer(value=30)
        job_timer.tick(refresh_all_jobs, outputs=[active_jobs_html, recent_jobs_html])
        
        # Load jobs on page open
        app.load(refresh_all_jobs, outputs=[active_jobs_html, recent_jobs_html])
    
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
    print("╔════════════════════════════════════════════════════════════╗")
    print("║                    🧬 BioPipelines                         ║")
    print("╠════════════════════════════════════════════════════════════╣")
    print(f"║  Server: http://localhost:{args.port:<5}                          ║")
    print(f"║  Facade: {'BioPipelines ✓' if BP_AVAILABLE else 'Not available ✗':<30}    ║")
    if bp:
        health = bp.health_check()
        llm_provider = health.get('llm_provider') or 'None'
        tool_count = health.get('tool_count') or 0
        print(f"║  LLM: {llm_provider:<10}                                   ║")
        print(f"║  Tools: {tool_count:<3} available                                   ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    
    app = create_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
