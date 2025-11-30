"""Report generation for BioPipelines evaluation.

This module provides report generators for evaluation results
in various formats (HTML, JSON, Markdown).
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .evaluator import EvaluationRun
    from .metrics import EvaluationMetrics


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    
    # Output directory
    output_dir: str = "reports/evaluation"
    
    # Include detailed results
    include_details: bool = True
    
    # Include raw scores
    include_scores: bool = True
    
    # Include query responses
    include_responses: bool = True
    
    # Maximum response length to include
    max_response_length: int = 500
    
    # Report title prefix
    title_prefix: str = "BioPipelines Evaluation"


class ReportGenerator(ABC):
    """Base class for report generators."""
    
    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
    
    @abstractmethod
    def generate(
        self,
        run: "EvaluationRun",
        metrics: "EvaluationMetrics",
        output_path: Optional[str] = None,
    ) -> str:
        """Generate a report.
        
        Args:
            run: The evaluation run
            metrics: Aggregated metrics
            output_path: Optional output path
            
        Returns:
            Path to the generated report
        """
        pass
    
    def _ensure_output_dir(self, path: str) -> None:
        """Ensure output directory exists."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)


class JSONReportGenerator(ReportGenerator):
    """Generate JSON reports."""
    
    def generate(
        self,
        run: "EvaluationRun",
        metrics: "EvaluationMetrics",
        output_path: Optional[str] = None,
    ) -> str:
        """Generate JSON report."""
        if output_path is None:
            output_path = os.path.join(
                self.config.output_dir,
                f"{run.run_id}.json"
            )
        
        self._ensure_output_dir(output_path)
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "run": run.to_dict(),
            "metrics": metrics.to_dict(),
        }
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        return output_path


class HTMLReportGenerator(ReportGenerator):
    """Generate HTML reports."""
    
    def generate(
        self,
        run: "EvaluationRun",
        metrics: "EvaluationMetrics",
        output_path: Optional[str] = None,
    ) -> str:
        """Generate HTML report."""
        if output_path is None:
            output_path = os.path.join(
                self.config.output_dir,
                f"{run.run_id}.html"
            )
        
        self._ensure_output_dir(output_path)
        
        html = self._build_html(run, metrics)
        
        with open(output_path, "w") as f:
            f.write(html)
        
        return output_path
    
    def _build_html(
        self,
        run: "EvaluationRun",
        metrics: "EvaluationMetrics",
    ) -> str:
        """Build HTML content."""
        title = f"{self.config.title_prefix} - {run.run_id}"
        
        # Build summary cards
        summary_html = self._build_summary_section(run, metrics)
        
        # Build dimension chart data
        dimension_data = self._build_dimension_chart(metrics)
        
        # Build category breakdown
        category_html = self._build_category_section(metrics)
        
        # Build results table
        results_html = self._build_results_table(run) if self.config.include_details else ""
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        header {{
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            padding: 30px 20px;
            margin-bottom: 30px;
        }}
        header h1 {{
            font-size: 2rem;
            margin-bottom: 10px;
        }}
        header p {{
            opacity: 0.8;
        }}
        .cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .card h3 {{
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 10px;
            text-transform: uppercase;
        }}
        .card .value {{
            font-size: 2rem;
            font-weight: bold;
            color: #2c3e50;
        }}
        .card .value.success {{
            color: #27ae60;
        }}
        .card .value.warning {{
            color: #f39c12;
        }}
        .card .value.danger {{
            color: #e74c3c;
        }}
        section {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        section h2 {{
            font-size: 1.3rem;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }}
        .dimension-bars {{
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        .dimension-bar {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        .dimension-bar label {{
            width: 120px;
            font-weight: 500;
        }}
        .dimension-bar .bar-container {{
            flex: 1;
            height: 24px;
            background: #eee;
            border-radius: 12px;
            overflow: hidden;
        }}
        .dimension-bar .bar {{
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2ecc71);
            border-radius: 12px;
            transition: width 0.5s ease;
        }}
        .dimension-bar .score {{
            width: 60px;
            text-align: right;
            font-weight: bold;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .status {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }}
        .status.completed {{
            background: #d4edda;
            color: #155724;
        }}
        .status.failed {{
            background: #f8d7da;
            color: #721c24;
        }}
        .status.timeout {{
            background: #fff3cd;
            color: #856404;
        }}
        .score-pill {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.85rem;
            background: #e3f2fd;
            color: #1976d2;
        }}
        footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>{title}</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Benchmark: {run.benchmark_name} | Duration: {run.duration_seconds:.1f}s</p>
        </div>
    </header>
    
    <div class="container">
        {summary_html}
        
        <section>
            <h2>Score Dimensions</h2>
            {dimension_data}
        </section>
        
        {category_html}
        
        {results_html}
    </div>
    
    <footer>
        BioPipelines Evaluation Framework v1.0
    </footer>
</body>
</html>
"""
    
    def _build_summary_section(
        self,
        run: "EvaluationRun",
        metrics: "EvaluationMetrics",
    ) -> str:
        """Build summary cards HTML."""
        success_class = "success" if metrics.success_rate >= 80 else ("warning" if metrics.success_rate >= 50 else "danger")
        score_class = "success" if metrics.score_stats.mean >= 0.8 else ("warning" if metrics.score_stats.mean >= 0.5 else "danger")
        
        return f"""
        <div class="cards">
            <div class="card">
                <h3>Total Queries</h3>
                <div class="value">{metrics.total_queries}</div>
            </div>
            <div class="card">
                <h3>Success Rate</h3>
                <div class="value {success_class}">{metrics.success_rate:.1f}%</div>
            </div>
            <div class="card">
                <h3>Average Score</h3>
                <div class="value {score_class}">{metrics.score_stats.mean:.3f}</div>
            </div>
            <div class="card">
                <h3>Avg Latency</h3>
                <div class="value">{metrics.latency_stats.mean:.0f}ms</div>
            </div>
        </div>
        """
    
    def _build_dimension_chart(self, metrics: "EvaluationMetrics") -> str:
        """Build dimension score bars."""
        bars = []
        for dim, stats in sorted(metrics.dimension_stats.items()):
            width = stats.mean * 100
            bars.append(f"""
            <div class="dimension-bar">
                <label>{dim.replace('_', ' ').title()}</label>
                <div class="bar-container">
                    <div class="bar" style="width: {width}%"></div>
                </div>
                <span class="score">{stats.mean:.3f}</span>
            </div>
            """)
        
        return f"""
        <div class="dimension-bars">
            {''.join(bars)}
        </div>
        """
    
    def _build_category_section(self, metrics: "EvaluationMetrics") -> str:
        """Build category breakdown section."""
        if not metrics.category_metrics:
            return ""
        
        rows = []
        for cat, cat_metrics in sorted(metrics.category_metrics.items()):
            rows.append(f"""
            <tr>
                <td>{cat.replace('_', ' ').title()}</td>
                <td>{cat_metrics.total_queries}</td>
                <td>{cat_metrics.success_rate:.1f}%</td>
                <td><span class="score-pill">{cat_metrics.avg_score:.3f}</span></td>
                <td>{cat_metrics.avg_latency_ms:.0f}ms</td>
            </tr>
            """)
        
        return f"""
        <section>
            <h2>By Category</h2>
            <table>
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Queries</th>
                        <th>Success Rate</th>
                        <th>Avg Score</th>
                        <th>Avg Latency</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </section>
        """
    
    def _build_results_table(self, run: "EvaluationRun") -> str:
        """Build detailed results table."""
        rows = []
        for result in run.results:
            status_class = result.status.value
            score = f"{result.overall_score:.3f}" if result.scores else "N/A"
            
            # Truncate query if too long
            query = result.query_text
            if len(query) > 80:
                query = query[:77] + "..."
            
            rows.append(f"""
            <tr>
                <td>{result.query_id}</td>
                <td>{query}</td>
                <td><span class="status {status_class}">{result.status.value}</span></td>
                <td><span class="score-pill">{score}</span></td>
                <td>{result.latency_ms:.0f}ms</td>
                <td>{', '.join(result.tools_called) or 'None'}</td>
            </tr>
            """)
        
        return f"""
        <section>
            <h2>Detailed Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Query</th>
                        <th>Status</th>
                        <th>Score</th>
                        <th>Latency</th>
                        <th>Tools Called</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </section>
        """


class MarkdownReportGenerator(ReportGenerator):
    """Generate Markdown reports."""
    
    def generate(
        self,
        run: "EvaluationRun",
        metrics: "EvaluationMetrics",
        output_path: Optional[str] = None,
    ) -> str:
        """Generate Markdown report."""
        if output_path is None:
            output_path = os.path.join(
                self.config.output_dir,
                f"{run.run_id}.md"
            )
        
        self._ensure_output_dir(output_path)
        
        md = self._build_markdown(run, metrics)
        
        with open(output_path, "w") as f:
            f.write(md)
        
        return output_path
    
    def _build_markdown(
        self,
        run: "EvaluationRun",
        metrics: "EvaluationMetrics",
    ) -> str:
        """Build Markdown content."""
        lines = [
            f"# {self.config.title_prefix} - {run.run_id}",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Benchmark:** {run.benchmark_name}",
            f"**Duration:** {run.duration_seconds:.1f}s",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Queries | {metrics.total_queries} |",
            f"| Successful | {metrics.successful} |",
            f"| Failed | {metrics.failed} |",
            f"| Timeouts | {metrics.timeouts} |",
            f"| Success Rate | {metrics.success_rate:.1f}% |",
            f"| Average Score | {metrics.score_stats.mean:.3f} |",
            f"| Average Latency | {metrics.latency_stats.mean:.0f}ms |",
            "",
            "## Score Dimensions",
            "",
            "| Dimension | Mean | Std Dev | Min | Max |",
            "|-----------|------|---------|-----|-----|",
        ]
        
        for dim, stats in sorted(metrics.dimension_stats.items()):
            lines.append(
                f"| {dim} | {stats.mean:.3f} | {stats.std_dev:.3f} | "
                f"{stats.min_value:.3f} | {stats.max_value:.3f} |"
            )
        
        if metrics.category_metrics:
            lines.extend([
                "",
                "## By Category",
                "",
                "| Category | Queries | Success Rate | Avg Score |",
                "|----------|---------|--------------|-----------|",
            ])
            
            for cat, cat_metrics in sorted(metrics.category_metrics.items()):
                lines.append(
                    f"| {cat} | {cat_metrics.total_queries} | "
                    f"{cat_metrics.success_rate:.1f}% | {cat_metrics.avg_score:.3f} |"
                )
        
        if self.config.include_details:
            lines.extend([
                "",
                "## Detailed Results",
                "",
                "| ID | Status | Score | Latency |",
                "|----|--------|-------|---------|",
            ])
            
            for result in run.results:
                score = f"{result.overall_score:.3f}" if result.scores else "N/A"
                lines.append(
                    f"| {result.query_id} | {result.status.value} | "
                    f"{score} | {result.latency_ms:.0f}ms |"
                )
        
        return "\n".join(lines)


def create_report_generator(
    format: str = "html",
    config: Optional[ReportConfig] = None,
) -> ReportGenerator:
    """Create a report generator.
    
    Args:
        format: Report format ("html", "json", "markdown")
        config: Optional report configuration
        
    Returns:
        ReportGenerator instance
    """
    generators = {
        "html": HTMLReportGenerator,
        "json": JSONReportGenerator,
        "markdown": MarkdownReportGenerator,
        "md": MarkdownReportGenerator,
    }
    
    generator_class = generators.get(format.lower())
    if generator_class is None:
        raise ValueError(f"Unknown format: {format}. Choose from: {list(generators.keys())}")
    
    return generator_class(config)


__all__ = [
    "ReportConfig",
    "ReportGenerator",
    "JSONReportGenerator",
    "HTMLReportGenerator",
    "MarkdownReportGenerator",
    "create_report_generator",
]
