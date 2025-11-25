"""
BioPipelines Web UI
===================

Simple Flask-based web interface for the AI Workflow Composer.

Features:
- Natural language workflow generation
- Tool/module browser
- Workflow visualization
- Download generated workflows
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from workflow_composer import Composer
    from workflow_composer.core import ToolSelector, ModuleMapper
    COMPOSER_AVAILABLE = True
except ImportError:
    COMPOSER_AVAILABLE = False
    Composer = None
    ToolSelector = None
    ModuleMapper = None

app = Flask(__name__)
CORS(app)

# Configuration
BASE_DIR = Path(__file__).parent.parent.parent
GENERATED_DIR = BASE_DIR / "generated_workflows"
GENERATED_DIR.mkdir(exist_ok=True)

# Initialize components (lazy loading)
_composer = None
_tool_selector = None
_module_mapper = None


def get_composer():
    global _composer
    if _composer is None:
        try:
            _composer = Composer()
        except Exception as e:
            print(f"Warning: Could not initialize Composer: {e}")
            _composer = False
    return _composer if _composer else None


def get_tool_selector():
    global _tool_selector
    if _tool_selector is None:
        try:
            _tool_selector = ToolSelector(str(BASE_DIR / "data/tool_catalog"))
        except Exception as e:
            print(f"Warning: Could not initialize ToolSelector: {e}")
            _tool_selector = False
    return _tool_selector if _tool_selector else None


def get_module_mapper():
    global _module_mapper
    if _module_mapper is None:
        try:
            _module_mapper = ModuleMapper(str(BASE_DIR / "nextflow-pipelines/modules"))
        except Exception as e:
            print(f"Warning: Could not initialize ModuleMapper: {e}")
            _module_mapper = False
    return _module_mapper if _module_mapper else None


# HTML Templates (inline for simplicity)
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BioPipelines - AI Workflow Composer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --accent-color: #27ae60;
        }
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .navbar {
            background: rgba(44, 62, 80, 0.95) !important;
        }
        .card {
            border: none;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        .card-header {
            background: var(--primary-color);
            color: white;
            border-radius: 15px 15px 0 0 !important;
        }
        .btn-generate {
            background: linear-gradient(45deg, #27ae60, #2ecc71);
            border: none;
            padding: 12px 30px;
            font-size: 1.1em;
        }
        .btn-generate:hover {
            background: linear-gradient(45deg, #219a52, #27ae60);
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(39, 174, 96, 0.4);
        }
        #prompt-input {
            min-height: 150px;
            border-radius: 10px;
        }
        .workflow-output {
            background: #1e1e1e;
            color: #d4d4d4;
            border-radius: 10px;
            font-family: 'Fira Code', monospace;
            font-size: 0.9em;
        }
        .tool-badge {
            background: var(--secondary-color);
            margin: 2px;
        }
        .module-badge {
            background: var(--accent-color);
            margin: 2px;
        }
        .stats-card {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 20px;
            color: white;
            text-align: center;
        }
        .stats-number {
            font-size: 2.5em;
            font-weight: bold;
        }
        .loading {
            display: none;
        }
        .loading.active {
            display: block;
        }
        .spinner-border {
            width: 3rem;
            height: 3rem;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark">
        <div class="container">
            <a class="navbar-brand" href="#">
                <i class="bi bi-diagram-3"></i> BioPipelines
            </a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link active" href="#"><i class="bi bi-house"></i> Home</a>
                <a class="nav-link" href="#tools"><i class="bi bi-tools"></i> Tools</a>
                <a class="nav-link" href="#modules"><i class="bi bi-box"></i> Modules</a>
                <a class="nav-link" href="https://github.com/sdodlapa/BioPipelines" target="_blank">
                    <i class="bi bi-github"></i> GitHub
                </a>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        <!-- Stats Row -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="stats-card">
                    <div class="stats-number" id="tool-count">-</div>
                    <div>Tools Available</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stats-card">
                    <div class="stats-number" id="module-count">-</div>
                    <div>Nextflow Modules</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stats-card">
                    <div class="stats-number">12</div>
                    <div>Containers</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stats-card">
                    <div class="stats-number">10+</div>
                    <div>Analysis Types</div>
                </div>
            </div>
        </div>

        <!-- Main Generation Card -->
        <div class="row">
            <div class="col-lg-8 mx-auto">
                <div class="card mb-4">
                    <div class="card-header">
                        <h4 class="mb-0"><i class="bi bi-magic"></i> AI Workflow Composer</h4>
                    </div>
                    <div class="card-body">
                        <div class="mb-3">
                            <label class="form-label">Describe your bioinformatics analysis:</label>
                            <textarea id="prompt-input" class="form-control" 
                                placeholder="Example: RNA-seq differential expression analysis for mouse samples comparing treatment vs control with STAR alignment and DESeq2"></textarea>
                        </div>
                        
                        <div class="row mb-3">
                            <div class="col-md-6">
                                <label class="form-label">LLM Provider:</label>
                                <select id="provider-select" class="form-select">
                                    <option value="openai">OpenAI (GPT-4o)</option>
                                    <option value="vllm">vLLM (Local GPU)</option>
                                    <option value="ollama">Ollama (Local)</option>
                                    <option value="anthropic">Anthropic</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Workflow Name:</label>
                                <input type="text" id="workflow-name" class="form-control" 
                                    placeholder="my_workflow">
                            </div>
                        </div>
                        
                        <button id="generate-btn" class="btn btn-generate btn-lg w-100" onclick="generateWorkflow()">
                            <i class="bi bi-lightning-charge"></i> Generate Workflow
                        </button>
                        
                        <div id="loading" class="loading text-center mt-4">
                            <div class="spinner-border text-primary" role="status"></div>
                            <p class="mt-2">Generating workflow with AI...</p>
                        </div>
                    </div>
                </div>

                <!-- Results Card -->
                <div id="results-card" class="card mb-4" style="display: none;">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0"><i class="bi bi-check-circle"></i> Generated Workflow</h5>
                        <button class="btn btn-sm btn-light" onclick="downloadWorkflow()">
                            <i class="bi bi-download"></i> Download
                        </button>
                    </div>
                    <div class="card-body">
                        <div class="mb-3">
                            <strong>Tools:</strong>
                            <div id="tools-used"></div>
                        </div>
                        <div class="mb-3">
                            <strong>Modules:</strong>
                            <div id="modules-used"></div>
                        </div>
                        <div>
                            <strong>main.nf:</strong>
                            <pre class="workflow-output p-3 mt-2"><code id="workflow-code"></code></pre>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Tools Section -->
        <div class="row mt-4" id="tools">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0"><i class="bi bi-search"></i> Tool Browser</h5>
                    </div>
                    <div class="card-body">
                        <div class="row mb-3">
                            <div class="col-md-8">
                                <input type="text" id="tool-search" class="form-control" 
                                    placeholder="Search tools..." onkeyup="searchTools()">
                            </div>
                            <div class="col-md-4">
                                <select id="container-filter" class="form-select" onchange="searchTools()">
                                    <option value="">All Containers</option>
                                </select>
                            </div>
                        </div>
                        <div id="tool-results" class="row"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Modules Section -->
        <div class="row mt-4 mb-4" id="modules">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0"><i class="bi bi-box"></i> Module Library</h5>
                    </div>
                    <div class="card-body">
                        <div id="module-list" class="row"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        let currentWorkflowId = null;

        // Load initial data
        document.addEventListener('DOMContentLoaded', function() {
            loadStats();
            loadContainers();
            loadModules();
        });

        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                document.getElementById('tool-count').textContent = data.tools || '-';
                document.getElementById('module-count').textContent = data.modules || '-';
            } catch (e) {
                console.error('Failed to load stats:', e);
            }
        }

        async function loadContainers() {
            try {
                const response = await fetch('/api/containers');
                const data = await response.json();
                const select = document.getElementById('container-filter');
                data.containers.forEach(c => {
                    const option = document.createElement('option');
                    option.value = c;
                    option.textContent = c;
                    select.appendChild(option);
                });
            } catch (e) {
                console.error('Failed to load containers:', e);
            }
        }

        async function loadModules() {
            try {
                const response = await fetch('/api/modules');
                const data = await response.json();
                const container = document.getElementById('module-list');
                container.innerHTML = '';
                
                Object.entries(data.by_category || {}).forEach(([category, modules]) => {
                    const col = document.createElement('div');
                    col.className = 'col-md-4 mb-3';
                    col.innerHTML = `
                        <div class="p-3 bg-light rounded">
                            <h6 class="text-primary">${category}</h6>
                            <div>${modules.map(m => 
                                `<span class="badge module-badge">${m}</span>`
                            ).join('')}</div>
                        </div>
                    `;
                    container.appendChild(col);
                });
            } catch (e) {
                console.error('Failed to load modules:', e);
            }
        }

        async function searchTools() {
            const query = document.getElementById('tool-search').value;
            const container = document.getElementById('container-filter').value;
            
            if (query.length < 2 && !container) {
                document.getElementById('tool-results').innerHTML = 
                    '<p class="text-muted">Enter at least 2 characters to search</p>';
                return;
            }

            try {
                const params = new URLSearchParams();
                if (query) params.append('q', query);
                if (container) params.append('container', container);
                params.append('limit', '20');
                
                const response = await fetch(`/api/tools/search?${params}`);
                const data = await response.json();
                
                const resultsDiv = document.getElementById('tool-results');
                resultsDiv.innerHTML = '';
                
                if (data.tools && data.tools.length > 0) {
                    data.tools.forEach(tool => {
                        const col = document.createElement('div');
                        col.className = 'col-md-4 mb-2';
                        col.innerHTML = `
                            <div class="p-2 border rounded">
                                <strong>${tool.name}</strong>
                                <span class="badge bg-secondary float-end">${tool.container}</span>
                                <br><small class="text-muted">${tool.description || ''}</small>
                            </div>
                        `;
                        resultsDiv.appendChild(col);
                    });
                } else {
                    resultsDiv.innerHTML = '<p class="text-muted">No tools found</p>';
                }
            } catch (e) {
                console.error('Search failed:', e);
            }
        }

        async function generateWorkflow() {
            const prompt = document.getElementById('prompt-input').value;
            const provider = document.getElementById('provider-select').value;
            const name = document.getElementById('workflow-name').value || 'my_workflow';
            
            if (!prompt.trim()) {
                alert('Please enter a workflow description');
                return;
            }

            document.getElementById('loading').classList.add('active');
            document.getElementById('generate-btn').disabled = true;
            document.getElementById('results-card').style.display = 'none';

            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({prompt, provider, name})
                });
                
                const data = await response.json();
                
                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                currentWorkflowId = data.workflow_id;
                
                // Display results
                document.getElementById('tools-used').innerHTML = 
                    (data.tools || []).map(t => 
                        `<span class="badge tool-badge">${t}</span>`
                    ).join('');
                
                document.getElementById('modules-used').innerHTML = 
                    (data.modules || []).map(m => 
                        `<span class="badge module-badge">${m}</span>`
                    ).join('');
                
                document.getElementById('workflow-code').textContent = 
                    data.main_nf || 'No workflow generated';
                
                document.getElementById('results-card').style.display = 'block';
                
            } catch (e) {
                alert('Failed to generate workflow: ' + e.message);
            } finally {
                document.getElementById('loading').classList.remove('active');
                document.getElementById('generate-btn').disabled = false;
            }
        }

        function downloadWorkflow() {
            if (currentWorkflowId) {
                window.location.href = `/api/download/${currentWorkflowId}`;
            }
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return INDEX_HTML


@app.route('/api/stats')
def get_stats():
    """Get tool and module counts."""
    stats = {'tools': 0, 'modules': 0, 'containers': 12}
    
    selector = get_tool_selector()
    if selector:
        stats['tools'] = len(selector.tools)
    
    mapper = get_module_mapper()
    if mapper:
        stats['modules'] = len(mapper.modules)
    
    return jsonify(stats)


@app.route('/api/containers')
def get_containers():
    """List available containers."""
    selector = get_tool_selector()
    if selector:
        containers = list(set(t.container for t in selector.tools.values()))
        return jsonify({'containers': sorted(containers)})
    return jsonify({'containers': []})


@app.route('/api/tools/search')
def search_tools():
    """Search tools."""
    query = request.args.get('q', '')
    container = request.args.get('container', '')
    limit = int(request.args.get('limit', 20))
    
    selector = get_tool_selector()
    if not selector:
        return jsonify({'tools': [], 'error': 'Tool selector not available'})
    
    results = selector.search(query, limit=limit, container=container if container else None)
    
    tools = [
        {
            'name': t.name,
            'container': t.container,
            'category': t.category,
            'description': t.description[:100] if t.description else ''
        }
        for t in results
    ]
    
    return jsonify({'tools': tools})


@app.route('/api/modules')
def get_modules():
    """List all modules."""
    mapper = get_module_mapper()
    if not mapper:
        return jsonify({'modules': [], 'by_category': {}})
    
    modules = mapper.list_modules()
    by_category = {}
    
    for name, module in mapper.modules.items():
        category = module.path.parent.parent.name
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(name)
    
    return jsonify({
        'modules': modules,
        'by_category': by_category
    })


@app.route('/api/generate', methods=['POST'])
def generate_workflow():
    """Generate a workflow from natural language."""
    data = request.json
    prompt = data.get('prompt', '')
    provider = data.get('provider', 'ollama')
    name = data.get('name', 'workflow')
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    composer = get_composer()
    if not composer:
        # Fallback: generate a mock workflow without LLM
        return generate_mock_workflow(prompt, name)
    
    try:
        # Generate workflow
        workflow_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = GENERATED_DIR / workflow_id
        
        workflow = composer.generate(prompt, output_dir=str(output_dir))
        
        return jsonify({
            'workflow_id': workflow_id,
            'tools': workflow.tools if hasattr(workflow, 'tools') else [],
            'modules': workflow.modules if hasattr(workflow, 'modules') else [],
            'main_nf': workflow.main_nf if hasattr(workflow, 'main_nf') else '',
            'config': workflow.config if hasattr(workflow, 'config') else ''
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def generate_mock_workflow(prompt: str, name: str):
    """Generate a mock workflow when LLM is not available."""
    # Parse prompt for keywords
    prompt_lower = prompt.lower()
    
    tools = []
    modules = []
    
    # Detect analysis type and add appropriate tools
    if 'rna' in prompt_lower or 'rnaseq' in prompt_lower:
        tools = ['fastqc', 'fastp', 'star', 'featurecounts', 'deseq2', 'multiqc']
        modules = ['fastqc', 'fastp', 'star', 'featurecounts', 'deseq2', 'multiqc']
    elif 'chip' in prompt_lower:
        tools = ['fastqc', 'trim_galore', 'bowtie2', 'macs2', 'homer', 'multiqc']
        modules = ['fastqc', 'trim_galore', 'bowtie2', 'macs2', 'homer', 'multiqc']
    elif 'variant' in prompt_lower or 'wgs' in prompt_lower or 'wes' in prompt_lower:
        tools = ['fastqc', 'fastp', 'bwa', 'samtools', 'gatk', 'bcftools', 'multiqc']
        modules = ['fastqc', 'fastp', 'bwamem', 'samtools', 'gatk_haplotypecaller', 'bcftools', 'multiqc']
    elif 'single' in prompt_lower or 'scrna' in prompt_lower or '10x' in prompt_lower:
        tools = ['starsolo', 'seurat', 'scanpy']
        modules = ['starsolo', 'seurat', 'scanpy']
    else:
        tools = ['fastqc', 'multiqc']
        modules = ['fastqc', 'multiqc']
    
    # Generate mock main.nf
    main_nf = f'''#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// {name} - Generated by BioPipelines
// Analysis: {prompt[:100]}...

// Include modules
{chr(10).join(f"include {{ {m.upper()} }} from './modules/{m}/main.nf'" for m in modules)}

// Main workflow
workflow {{
    // Input channel
    reads_ch = Channel.fromFilePairs(params.input)
    
    // Run pipeline
{chr(10).join(f"    {m.upper()}(reads_ch)" for m in modules)}
}}
'''
    
    # Save mock workflow
    workflow_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = GENERATED_DIR / workflow_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    (output_dir / 'main.nf').write_text(main_nf)
    
    return jsonify({
        'workflow_id': workflow_id,
        'tools': tools,
        'modules': modules,
        'main_nf': main_nf,
        'note': 'Generated without LLM (mock workflow)'
    })


@app.route('/api/download/<workflow_id>')
def download_workflow(workflow_id):
    """Download generated workflow as zip."""
    workflow_dir = GENERATED_DIR / workflow_id
    
    if not workflow_dir.exists():
        return jsonify({'error': 'Workflow not found'}), 404
    
    # Create zip file
    zip_path = GENERATED_DIR / f"{workflow_id}.zip"
    shutil.make_archive(str(zip_path.with_suffix('')), 'zip', workflow_dir)
    
    return send_file(
        zip_path,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f"{workflow_id}.zip"
    )


def main():
    """Run the web server."""
    import argparse
    parser = argparse.ArgumentParser(description='BioPipelines Web UI')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           BioPipelines - AI Workflow Composer                ║
║                       Web Interface                          ║
╠══════════════════════════════════════════════════════════════╣
║  Open in browser: http://{args.host}:{args.port}                       ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
