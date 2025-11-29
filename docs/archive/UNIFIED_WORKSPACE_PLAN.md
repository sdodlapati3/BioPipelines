# BioPipelines: Unified Workspace Implementation Plan

**Created:** November 27, 2025  
**Version:** 1.0  
**Status:** 🚧 IMPLEMENTING  
**Priority:** 🔴 HIGH  

---

## Executive Summary

Transform BioPipelines from a 5-tab interface to a **Unified Workspace** with 3 tabs:

| Before | After |
|--------|-------|
| Data → Workspace → Execute → Results → Advanced | **Workspace → Results → Advanced** |

The unified Workspace combines:
- 💬 **AI Chat** - Natural language workflow generation
- 📁 **Data Discovery** - Scan files, search databases (as AI tools)
- 🚀 **Execution** - Submit jobs, monitor progress (in sidebar)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  💬 UNIFIED WORKSPACE                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────┐  ┌────────────────────────────┐  │
│  │  CHAT AREA (70%)                      │  │  CONTEXT SIDEBAR (30%)     │  │
│  │  ────────────────                     │  │  ────────────────────────  │  │
│  │                                       │  │                            │  │
│  │  User: Analyze RNA-seq in /data/p1    │  │  📁 DATA MANIFEST          │  │
│  │                                       │  │  ├─ 12 samples (paired)    │  │
│  │  AI: Found 12 paired-end samples:     │  │  ├─ Organism: human        │  │
│  │      ┌─────────────────────────────┐  │  │  └─ Reference: GRCh38 ✓    │  │
│  │      │ ☑ sample1_R1/R2.fastq.gz   │  │  │                            │  │
│  │      │ ☑ sample2_R1/R2.fastq.gz   │  │  │  ─────────────────────────  │  │
│  │      └─────────────────────────────┘  │  │                            │  │
│  │                                       │  │  🚀 ACTIVE JOBS            │  │
│  │      Ready to generate workflow.      │  │  ├─ Job 12345: ████░ 67%   │  │
│  │      [Generate DESeq2 Pipeline]       │  │  └─ Job 12340: ✓ Done      │  │
│  │                                       │  │                            │  │
│  │  User: Run it on SLURM                │  │  [Cancel] [Logs] [Refresh] │  │
│  │                                       │  │                            │  │
│  │  AI: Submitted job 12346.             │  │  ─────────────────────────  │  │
│  │      Monitoring in sidebar →          │  │                            │  │
│  │                                       │  │  ⚡ QUICK ACTIONS          │  │
│  │  ─────────────────────────────────    │  │  [📂 Scan Data]            │  │
│  │  [Type your message...]    [Send 🚀]  │  │  [🔍 Search Databases]     │  │
│  │                                       │  │  [📊 View Results]         │  │
│  └───────────────────────────────────────┘  └────────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Create Agent Tools (NEW)

Create tools that the AI can invoke during chat:

```python
# src/workflow_composer/agents/tools.py

class AgentTools:
    """Tools available to the AI agent during chat."""
    
    @tool("scan_data")
    def scan_local_data(self, path: str, recursive: bool = True) -> ScanResult:
        """Scan a directory for FASTQ files."""
        scanner = LocalSampleScanner()
        return scanner.scan_directory(Path(path), recursive=recursive)
    
    @tool("search_databases")
    def search_remote_databases(self, query: str, sources: List[str] = ["ENCODE", "GEO"]) -> SearchResults:
        """Search ENCODE, GEO, Ensembl for datasets."""
        # Uses existing discovery adapters
        pass
    
    @tool("submit_job")
    def submit_workflow(self, workflow_name: str, profile: str = "slurm") -> str:
        """Submit a workflow to run."""
        # Uses existing execution infrastructure
        pass
    
    @tool("get_job_status")
    def get_job_status(self, job_id: str) -> JobStatus:
        """Get status of a running job."""
        pass
    
    @tool("get_logs")
    def get_job_logs(self, job_id: str, lines: int = 50) -> str:
        """Get logs from a job."""
        pass
    
    @tool("cancel_job")
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        pass
    
    @tool("diagnose_error")
    def diagnose_failure(self, job_id: str) -> DiagnosisResult:
        """AI-powered error diagnosis for failed jobs."""
        pass
```

### Phase 2: Create Unified Workspace Component

```python
# src/workflow_composer/web/components/unified_workspace.py

def create_unified_workspace():
    """Create the unified workspace with chat + sidebar."""
    
    with gr.Row():
        # ===== MAIN CHAT AREA (70%) =====
        with gr.Column(scale=7):
            # Stats bar
            with gr.Row():
                gr.Markdown("📊 **9,909** Tools")
                gr.Markdown("📦 **71** Modules")
                gr.Markdown("🐳 **10** Containers")
            
            # Chatbot
            chatbot = gr.Chatbot(height=550, show_label=False)
            
            # Input area
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Describe your analysis or ask questions...",
                    lines=2, scale=5, show_label=False
                )
                send_btn = gr.Button("Send 🚀", variant="primary", scale=1)
            
            # Examples
            with gr.Accordion("📝 Examples", open=False):
                gr.Examples(examples=[...], inputs=msg_input)
        
        # ===== CONTEXT SIDEBAR (30%) =====
        with gr.Column(scale=3):
            # LLM Provider selector
            provider_dropdown = gr.Dropdown(...)
            
            gr.Markdown("---")
            
            # Data Manifest Panel
            with gr.Accordion("📁 Data Manifest", open=True):
                sample_count = gr.Markdown("**0** samples")
                reference_status = gr.Markdown("No reference configured")
                scan_btn = gr.Button("📂 Scan Directory", size="sm")
                search_btn = gr.Button("🔍 Search Databases", size="sm")
            
            gr.Markdown("---")
            
            # Active Jobs Panel
            gr.Markdown("### 🚀 Active Jobs")
            jobs_html = gr.HTML(render_jobs_panel())
            
            with gr.Row():
                refresh_jobs_btn = gr.Button("🔄", size="sm")
                cancel_job_btn = gr.Button("🛑", size="sm")
                logs_btn = gr.Button("📄", size="sm")
            
            job_selector = gr.Dropdown(label="Select Job", interactive=True)
            
            # Auto-refresh timer for jobs
            job_timer = gr.Timer(10, active=True)
            
            gr.Markdown("---")
            
            # Recent Workflows
            with gr.Accordion("📋 Recent Workflows", open=False):
                recent_workflows = gr.Markdown()
            
            # Quick Actions
            gr.Markdown("### ⚡ Quick Actions")
            with gr.Row():
                clear_chat_btn = gr.Button("🗑️ Clear", size="sm")
                view_results_btn = gr.Button("📊 Results", size="sm")
    
    return components
```

### Phase 3: Modify Chat Handler to Use Tools

```python
def chat_with_composer(message, history, provider, app_state):
    """Enhanced chat handler with tool execution."""
    
    # Detect tool invocation patterns
    tool_patterns = {
        r"scan\s+(?:data|files?|directory)\s+(?:in|at|from)?\s*(.+)": "scan_data",
        r"search\s+(?:for|databases?)\s+(.+)": "search_databases",
        r"run\s+(?:it|workflow|pipeline)\s+(?:on|with)?\s*(\w+)?": "submit_job",
        r"show\s+(?:me\s+)?(?:the\s+)?logs?\s*(?:for)?\s*(\d+)?": "get_logs",
        r"cancel\s+(?:job\s+)?(\d+)?": "cancel_job",
        r"what(?:'s|\s+is)\s+(?:the\s+)?status": "get_job_status",
    }
    
    # Execute tool if pattern matches
    for pattern, tool_name in tool_patterns.items():
        if match := re.search(pattern, message, re.IGNORECASE):
            result = execute_tool(tool_name, match.groups(), app_state)
            yield format_tool_result(result, history)
            return
    
    # Otherwise, normal AI workflow generation
    ...
```

### Phase 4: Refactor gradio_app.py

1. **Remove separate Data tab** (merge into sidebar)
2. **Remove separate Execute tab** (merge into sidebar)
3. **Keep Results tab** (needs full width for file browsing)
4. **Keep Advanced tab** (power user tool browser)

---

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `web/components/unified_workspace.py` | **CREATE** | New unified workspace component |
| `agents/tools.py` | **CREATE** | AI agent tools for data/execution |
| `web/gradio_app.py` | **MODIFY** | Replace 3 tabs with 1 unified tab |
| `web/components/data_tab.py` | **KEEP** | Reuse functions as tool backends |
| `execution/monitor.py` | **MODIFY** | Add job status API for sidebar |

---

## UI Components Mapping

### From Data Tab → Sidebar + Tools

| Original (Data Tab) | New Location |
|---------------------|--------------|
| Scan directory UI | Sidebar button + AI tool |
| Remote search UI | Sidebar button + AI tool |
| Reference status | Sidebar display |
| Sample table | Chat response widget |
| Manifest summary | Sidebar panel |

### From Execute Tab → Sidebar + Tools

| Original (Execute Tab) | New Location |
|------------------------|--------------|
| Workflow dropdown | Sidebar "Recent Workflows" |
| Submit button | AI tool + Chat button |
| Job monitor | Sidebar "Active Jobs" panel |
| Logs viewer | Chat response / Modal |
| Cancel button | Sidebar button |
| AI Diagnosis | AI tool |

---

## Event Handler Flow

```
User types: "Scan data in /data/project1"
    │
    ▼
chat_handler() detects "scan" pattern
    │
    ▼
Calls AgentTools.scan_local_data("/data/project1")
    │
    ▼
Returns ScanResult with 12 samples
    │
    ▼
Updates sidebar manifest panel
    │
    ▼
Responds in chat with interactive sample table
    │
    ▼
User types: "Generate DESeq2 workflow"
    │
    ▼
AI generates workflow (manifest auto-injected)
    │
    ▼
User types: "Run it on SLURM"
    │
    ▼
Calls AgentTools.submit_workflow("workflow_name", "slurm")
    │
    ▼
Job appears in sidebar panel
    │
    ▼
Timer auto-refreshes job status every 10 seconds
```

---

## Implementation Order

1. ✅ Create `agents/tools.py` with all agent tools
2. ✅ Create `unified_workspace.py` component
3. ✅ Modify chat handler to detect and execute tools
4. ✅ Wire sidebar to job monitor
5. ✅ Refactor `gradio_app.py` to use unified workspace
6. ✅ Test all functionality
7. ✅ Update documentation

---

## Success Criteria

- [ ] Single Workspace tab handles data + chat + execution
- [ ] Sidebar shows live data manifest
- [ ] Sidebar shows live job status with auto-refresh
- [ ] User can scan data via chat or sidebar button
- [ ] User can submit jobs via chat or button
- [ ] User can view logs via chat or button
- [ ] All 148+ tests still pass
- [ ] Professional, modern UI appearance
