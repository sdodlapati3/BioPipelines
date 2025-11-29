"""
BioPipelines - Chat-First Web Interface (Gradio 6.x)
====================================================

A minimal, focused UI where chat is the primary interface.
All features are accessible through natural conversation.

This refactored version uses:
- Unified chat handler (handles tools, LLM fallback, validation)
- Session management for persistent state
- Clean separation of concerns
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

VLLM_URL = detect_vllm_endpoint()
USE_LOCAL_LLM = use_local_llm()
DEFAULT_PORT = get_default_port()

# ============================================================================
# Import Unified Chat Handler
# ============================================================================

chat_handler = None
HANDLER_AVAILABLE = False

try:
    from workflow_composer.web.chat_handler import get_chat_handler, UnifiedChatHandler
    chat_handler = get_chat_handler()
    HANDLER_AVAILABLE = True
    print("✓ Unified chat handler initialized")
except Exception as e:
    print(f"⚠ Chat handler not available: {e}")
    import traceback
    traceback.print_exc()


# ============================================================================
# Chat Response Generator
# ============================================================================

def chat_response(message: str, history: List[Dict]) -> Generator[List[Dict], None, None]:
    """
    Generate chat response using the unified handler.
    
    Args:
        message: User's input message
        history: Chat history as list of dicts with "role" and "content"
    
    Yields:
        Updated history with assistant response
    """
    if not message.strip():
        yield history
        return
    
    if not chat_handler:
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "⚠️ Chat handler not available."}
        ]
        return
    
    # Use the unified handler's chat method
    try:
        for response in chat_handler.chat(message, history):
            yield response
    except Exception as e:
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"❌ Error: {e}"}
        ]


# ============================================================================
# Status Functions
# ============================================================================

def get_status() -> str:
    """Get current status as HTML."""
    if chat_handler:
        session = chat_handler.session_manager.get_session()
        handler_status = "🟢 Ready" if chat_handler._tools_available else "🟡 Limited"
        session_status = session.summary()
        return f"{handler_status} | {session_status}"
    return "🔴 Handler not available"


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
            settings_info = "**Handler:** " + ("Available ✓" if HANDLER_AVAILABLE else "Not available")
            if chat_handler:
                settings_info += f"\n**Tools:** {len(chat_handler.agent_tools.tools) if chat_handler.agent_tools else 0}"
            gr.Markdown(settings_info)
            clear = gr.Button("🗑️ Clear Chat")
        
        # Event handlers
        def submit(message, history):
            if not message.strip():
                return history, ""
            for response in chat_response(message, history):
                yield response, ""
        
        def clear_chat():
            if chat_handler:
                # Clear session state
                chat_handler.session_manager.clear_session()
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
    print("╔════════════════════════════════════════════════════════════╗")
    print("║                    🧬 BioPipelines                         ║")
    print("╠════════════════════════════════════════════════════════════╣")
    print(f"║  Server: http://localhost:{args.port:<5}                          ║")
    print(f"║  Handler: {'Unified Chat Handler ✓' if HANDLER_AVAILABLE else 'Not available ✗':<30}    ║")
    if chat_handler and chat_handler.agent_tools:
        print(f"║  Tools: {len(chat_handler.agent_tools.tools):<3} available                                   ║")
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
