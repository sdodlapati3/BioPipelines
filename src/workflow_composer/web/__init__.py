"""
Web UI module for BioPipelines Workflow Composer.

Provides multiple interfaces:
- Gradio UI: Modern chat-based interface (recommended)
- Flask App: Original web interface
- FastAPI: REST API for programmatic access

Usage:
    # Gradio UI (recommended)
    python -m workflow_composer.web.gradio_app
    
    # Flask App
    python -m workflow_composer.web.app
    
    # FastAPI
    uvicorn workflow_composer.web.api:app --reload
"""

# Import Flask app (original)
try:
    from .app import app as flask_app, main as flask_main
except ImportError:
    flask_app = None
    flask_main = None

# Import Gradio app (recommended)
try:
    from .gradio_app import create_interface, main as gradio_main
except ImportError:
    create_interface = None
    gradio_main = None

# Import FastAPI app
try:
    from .api import app as fastapi_app, main as api_main
except ImportError:
    fastapi_app = None
    api_main = None

# Default to Gradio
main = gradio_main or flask_main

__all__ = [
    'flask_app',
    'flask_main',
    'create_interface', 
    'gradio_main',
    'fastapi_app',
    'api_main',
    'main',
]
