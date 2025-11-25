"""
Visualization Module
====================

Generate visualizations of workflows, results, and analysis summaries.

Supports:
- Workflow DAG diagrams
- Pipeline execution reports
- QC summary plots
- Interactive HTML reports

Example:
    from workflow_composer.viz import WorkflowVisualizer
    
    viz = WorkflowVisualizer()
    viz.render_dag(workflow, "workflow_dag.png")
    viz.generate_report(workflow, "report.html")
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Node:
    """DAG node representing a process."""
    id: str
    label: str
    process_type: str  # process, channel, input, output
    module: Optional[str] = None
    container: Optional[str] = None


@dataclass
class Edge:
    """DAG edge representing data flow."""
    source: str
    target: str
    label: Optional[str] = None


class WorkflowVisualizer:
    """
    Generate visualizations for workflows.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Default output directory
        """
        self.output_dir = Path(output_dir) if output_dir else Path(".")
    
    def render_dag(
        self,
        workflow: Any,
        output_path: Optional[str] = None,
        format: str = "png",
        show_containers: bool = True
    ) -> Path:
        """
        Render workflow DAG diagram.
        
        Args:
            workflow: Workflow object
            output_path: Output file path
            format: Output format (png, svg, pdf)
            show_containers: Show container info on nodes
            
        Returns:
            Path to generated diagram
        """
        try:
            import graphviz
        except ImportError:
            logger.warning("graphviz not installed, generating text representation")
            return self._render_dag_text(workflow, output_path)
        
        dot = graphviz.Digraph(comment=workflow.name)
        dot.attr(rankdir='TB', splines='ortho')
        
        # Style settings
        dot.attr('node', shape='box', style='rounded,filled')
        
        # Add nodes
        nodes = self._extract_nodes(workflow)
        for node in nodes:
            color = self._get_node_color(node.process_type)
            label = node.label
            if show_containers and node.container:
                label = f"{label}\\n[{node.container}]"
            dot.node(node.id, label, fillcolor=color)
        
        # Add edges
        edges = self._extract_edges(workflow)
        for edge in edges:
            dot.edge(edge.source, edge.target, label=edge.label or "")
        
        # Render
        if output_path:
            out = Path(output_path)
        else:
            out = self.output_dir / f"{workflow.name}_dag"
        
        dot.render(str(out.with_suffix('')), format=format, cleanup=True)
        return out.with_suffix(f'.{format}')
    
    def _render_dag_text(self, workflow: Any, output_path: Optional[str]) -> Path:
        """Render DAG as ASCII art."""
        lines = [
            f"Workflow: {workflow.name}",
            "=" * 40,
            ""
        ]
        
        nodes = self._extract_nodes(workflow)
        edges = self._extract_edges(workflow)
        
        lines.append("Processes:")
        for node in nodes:
            if node.process_type == "process":
                container = f" [{node.container}]" if node.container else ""
                lines.append(f"  - {node.label}{container}")
        
        lines.append("")
        lines.append("Data Flow:")
        for edge in edges:
            lines.append(f"  {edge.source} --> {edge.target}")
        
        content = "\n".join(lines)
        
        if output_path:
            out = Path(output_path)
        else:
            out = self.output_dir / f"{workflow.name}_dag.txt"
        
        out.write_text(content)
        return out
    
    def _extract_nodes(self, workflow: Any) -> List[Node]:
        """Extract nodes from workflow."""
        nodes = []
        
        # Parse modules used
        if hasattr(workflow, 'modules'):
            for i, module in enumerate(workflow.modules):
                nodes.append(Node(
                    id=f"proc_{i}",
                    label=module.name if hasattr(module, 'name') else str(module),
                    process_type="process",
                    module=str(module),
                    container=getattr(module, 'container', None)
                ))
        
        # Add input/output nodes
        nodes.insert(0, Node(id="input", label="Input Files", process_type="input"))
        nodes.append(Node(id="output", label="Results", process_type="output"))
        
        return nodes
    
    def _extract_edges(self, workflow: Any) -> List[Edge]:
        """Extract edges from workflow."""
        edges = []
        
        # Create linear flow for now
        nodes = self._extract_nodes(workflow)
        for i in range(len(nodes) - 1):
            edges.append(Edge(
                source=nodes[i].id,
                target=nodes[i + 1].id
            ))
        
        return edges
    
    def _get_node_color(self, process_type: str) -> str:
        """Get node color based on type."""
        colors = {
            "input": "#98FB98",     # Pale green
            "output": "#FFB6C1",    # Light pink
            "process": "#87CEEB",   # Sky blue
            "channel": "#FFFACD"    # Lemon chiffon
        }
        return colors.get(process_type, "#FFFFFF")
    
    def generate_report(
        self,
        workflow: Any,
        output_path: Optional[str] = None,
        include_code: bool = True,
        include_dag: bool = True
    ) -> Path:
        """
        Generate HTML report for workflow.
        
        Args:
            workflow: Workflow object
            output_path: Output file path
            include_code: Include workflow code
            include_dag: Include DAG diagram
            
        Returns:
            Path to generated report
        """
        if output_path:
            out = Path(output_path)
        else:
            out = self.output_dir / f"{workflow.name}_report.html"
        
        html = self._generate_html_report(workflow, include_code, include_dag)
        out.write_text(html)
        
        logger.info(f"Report generated: {out}")
        return out
    
    def _generate_html_report(
        self,
        workflow: Any,
        include_code: bool,
        include_dag: bool
    ) -> str:
        """Generate HTML content."""
        
        # Extract workflow info
        name = workflow.name if hasattr(workflow, 'name') else "Workflow"
        description = getattr(workflow, 'description', "")
        main_nf = getattr(workflow, 'main_nf', "")
        config = getattr(workflow, 'config', "")
        
        # Modules info
        modules_html = ""
        if hasattr(workflow, 'modules'):
            modules_html = "<ul>"
            for m in workflow.modules:
                mod_name = m.name if hasattr(m, 'name') else str(m)
                container = getattr(m, 'container', 'N/A')
                modules_html += f"<li><strong>{mod_name}</strong> (container: {container})</li>"
            modules_html += "</ul>"
        
        # Code section
        code_html = ""
        if include_code and main_nf:
            code_html = f"""
            <h2>Workflow Code</h2>
            <h3>main.nf</h3>
            <pre><code class="language-groovy">{self._escape_html(main_nf)}</code></pre>
            """
            if config:
                code_html += f"""
                <h3>nextflow.config</h3>
                <pre><code class="language-groovy">{self._escape_html(config)}</code></pre>
                """
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{name} - Workflow Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        pre {{
            background: #f8f8f8;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            border: 1px solid #e0e0e0;
        }}
        code {{
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 13px;
        }}
        .meta {{
            color: #666;
            font-size: 14px;
            margin-bottom: 20px;
        }}
        .module-list {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
        }}
        ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        li {{
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }}
        li:last-child {{
            border-bottom: none;
        }}
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 12px;
            margin-left: 5px;
        }}
        .badge-blue {{
            background: #3498db;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 {name}</h1>
        <p class="meta">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h2>Description</h2>
        <p>{description or "No description provided."}</p>
        
        <h2>Modules Used</h2>
        <div class="module-list">
            {modules_html or "<p>No modules specified.</p>"}
        </div>
        
        {code_html}
        
        <h2>Usage</h2>
        <pre><code>nextflow run main.nf -profile singularity --input samples.csv</code></pre>
        
        <hr style="margin-top: 40px;">
        <p style="color: #999; font-size: 12px;">
            Generated by BioPipelines Workflow Composer
        </p>
    </div>
</body>
</html>"""
        
        return html
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (
            text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
    
    def generate_summary(
        self,
        workflow: Any,
        format: str = "markdown"
    ) -> str:
        """
        Generate text summary of workflow.
        
        Args:
            workflow: Workflow object
            format: Output format (markdown, text, json)
            
        Returns:
            Summary text
        """
        if format == "json":
            return self._summary_json(workflow)
        elif format == "markdown":
            return self._summary_markdown(workflow)
        else:
            return self._summary_text(workflow)
    
    def _summary_markdown(self, workflow: Any) -> str:
        """Generate markdown summary."""
        name = getattr(workflow, 'name', 'Workflow')
        desc = getattr(workflow, 'description', '')
        
        lines = [
            f"# {name}",
            "",
            desc or "*No description*",
            "",
            "## Modules",
            ""
        ]
        
        if hasattr(workflow, 'modules'):
            for m in workflow.modules:
                mod_name = m.name if hasattr(m, 'name') else str(m)
                container = getattr(m, 'container', 'N/A')
                lines.append(f"- **{mod_name}** (container: `{container}`)")
        
        lines.extend([
            "",
            "## Quick Start",
            "",
            "```bash",
            "nextflow run main.nf -profile singularity --input samples.csv",
            "```"
        ])
        
        return "\n".join(lines)
    
    def _summary_text(self, workflow: Any) -> str:
        """Generate plain text summary."""
        name = getattr(workflow, 'name', 'Workflow')
        
        lines = [
            name,
            "=" * len(name),
            ""
        ]
        
        if hasattr(workflow, 'modules'):
            lines.append("Modules:")
            for m in workflow.modules:
                mod_name = m.name if hasattr(m, 'name') else str(m)
                lines.append(f"  - {mod_name}")
        
        return "\n".join(lines)
    
    def _summary_json(self, workflow: Any) -> str:
        """Generate JSON summary."""
        data = {
            "name": getattr(workflow, 'name', 'workflow'),
            "description": getattr(workflow, 'description', ''),
            "modules": []
        }
        
        if hasattr(workflow, 'modules'):
            for m in workflow.modules:
                data["modules"].append({
                    "name": m.name if hasattr(m, 'name') else str(m),
                    "container": getattr(m, 'container', None)
                })
        
        return json.dumps(data, indent=2)
