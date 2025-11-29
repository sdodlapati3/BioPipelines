"""
Web UI module for BioPipelines Workflow Composer.

Provides a chat-first Gradio interface.

The frontend uses the BioPipelines facade directly:
    from workflow_composer import BioPipelines
    bp = BioPipelines()
    response = bp.chat("analyze my RNA-seq data")

Usage:
    # Start the web UI
    python -m workflow_composer.web.app --share
    
    # Or via start_server.sh
    ./scripts/start_server.sh --cloud   # Cloud LLM only
    ./scripts/start_server.sh --gpu     # Local vLLM on GPU
"""

# Import Gradio app (main interface)
try:
    from .app import create_app, main
except ImportError:
    create_app = None
    main = None

# Import utilities
try:
    from .utils import detect_vllm_endpoint, get_default_port, use_local_llm
except ImportError:
    detect_vllm_endpoint = None
    get_default_port = None
    use_local_llm = None

__all__ = [
    'create_app',
    'main',
    'detect_vllm_endpoint',
    'get_default_port',
    'use_local_llm',
]
