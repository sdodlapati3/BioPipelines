"""
Web components for the BioPipelines Gradio UI.

This package provides reusable utility components for the main Gradio app.
Note: result_browser.py and unified_workspace.py have been archived as their
functionality is now integrated directly into gradio_app.py.
"""

# Import data tab utility components (may fail if gradio not installed)
try:
    from .data_tab import (
        DataTabState,
        create_local_scanner_ui,
        create_remote_search_ui,
        create_reference_manager_ui,
        create_data_summary_panel,
    )
    _DATA_TAB_AVAILABLE = True
except ImportError:
    _DATA_TAB_AVAILABLE = False
    DataTabState = None
    create_local_scanner_ui = None
    create_remote_search_ui = None
    create_reference_manager_ui = None
    create_data_summary_panel = None

# Import job panel components
try:
    from .job_panel import (
        get_user_jobs,
        get_recent_jobs,
        format_jobs_table,
        get_job_log,
        cancel_job,
        create_job_panel,
        create_job_tab,
    )
    _JOB_PANEL_AVAILABLE = True
except ImportError:
    _JOB_PANEL_AVAILABLE = False

__all__ = [
    # Data tab utilities
    "DataTabState",
    "create_local_scanner_ui",
    "create_remote_search_ui",
    "create_reference_manager_ui",
    "create_data_summary_panel",
    # Job panel utilities
    "get_user_jobs",
    "get_recent_jobs",
    "format_jobs_table",
    "get_job_log",
    "cancel_job",
    "create_job_panel",
    "create_job_tab",
]
