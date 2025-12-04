#!/usr/bin/env python3
"""
Metrics Dashboard for BioPipelines Chat Agent Evaluation

Provides visualization capabilities for evaluation metrics:
1. Terminal-based dashboard (no dependencies)
2. HTML report generation
3. Trend visualization
4. Category heatmaps
5. Failure analysis views

Usage:
    python scripts/metrics_dashboard.py --last-run
    python scripts/metrics_dashboard.py --trends
    python scripts/metrics_dashboard.py --html-report output.html
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# TERMINAL DASHBOARD
# ============================================================================

class TerminalDashboard:
    """Terminal-based dashboard with ASCII visualizations."""
    
    def __init__(self, width: int = 80):
        self.width = width
    
    def header(self, title: str) -> str:
        """Create a header."""
        return f"\n{'=' * self.width}\n{title.center(self.width)}\n{'=' * self.width}"
    
    def section(self, title: str) -> str:
        """Create a section header."""
        return f"\n{'─' * self.width}\n{title}\n{'─' * self.width}"
    
    def progress_bar(self, value: float, max_value: float = 1.0, length: int = 40) -> str:
        """Create a progress bar."""
        if max_value == 0:
            return "░" * length
        
        filled = int((value / max_value) * length)
        empty = length - filled
        
        # Color indicators
        if value / max_value >= 0.8:
            indicator = "█"
        elif value / max_value >= 0.6:
            indicator = "▓"
        else:
            indicator = "▒"
        
        return indicator * filled + "░" * empty
    
    def metric_line(self, name: str, value: float, target: float = 0.8) -> str:
        """Create a metric display line."""
        bar = self.progress_bar(value)
        percent = value * 100
        status = "✓" if value >= target else "✗"
        return f"  {status} {name:25} [{bar}] {percent:5.1f}%"
    
    def table(self, headers: list, rows: list, col_widths: list = None) -> str:
        """Create a simple table."""
        if not col_widths:
            col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 
                          for i in range(len(headers))]
        
        lines = []
        
        # Header
        header_line = "│".join(h.center(w) for h, w in zip(headers, col_widths))
        lines.append(f"┌{'┬'.join('─' * w for w in col_widths)}┐")
        lines.append(f"│{header_line}│")
        lines.append(f"├{'┼'.join('─' * w for w in col_widths)}┤")
        
        # Rows
        for row in rows:
            row_line = "│".join(str(v).center(w) for v, w in zip(row, col_widths))
            lines.append(f"│{row_line}│")
        
        lines.append(f"└{'┴'.join('─' * w for w in col_widths)}┘")
        
        return "\n".join(lines)
    
    def heatmap(self, data: dict, title: str = "Heatmap") -> str:
        """Create a simple ASCII heatmap."""
        lines = [f"\n{title}:"]
        
        max_val = max(data.values()) if data else 1
        
        for key, value in sorted(data.items()):
            normalized = value / max_val if max_val > 0 else 0
            
            if normalized >= 0.8:
                block = "██"
            elif normalized >= 0.6:
                block = "▓▓"
            elif normalized >= 0.4:
                block = "▒▒"
            elif normalized >= 0.2:
                block = "░░"
            else:
                block = "  "
            
            lines.append(f"  {key:20} {block} {value:.1%}")
        
        return "\n".join(lines)
    
    def trend_chart(self, values: list, labels: list = None, title: str = "Trend") -> str:
        """Create a simple ASCII trend chart."""
        if not values:
            return f"\n{title}: No data"
        
        height = 10
        width = min(len(values), 40)
        
        # Normalize values
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val != min_val else 1
        
        normalized = [(v - min_val) / range_val for v in values[-width:]]
        
        lines = [f"\n{title} (last {len(normalized)} runs):"]
        lines.append(f"  {max_val*100:5.1f}% ┌{'─' * (width * 2)}┐")
        
        for row in range(height, -1, -1):
            threshold = row / height
            line = "  " + " " * 7 + "│"
            
            for i, val in enumerate(normalized):
                if val >= threshold:
                    line += "█ "
                elif val >= threshold - 0.1:
                    line += "▄ "
                else:
                    line += "  "
            
            line += "│"
            lines.append(line)
        
        lines.append(f"  {min_val*100:5.1f}% └{'─' * (width * 2)}┘")
        
        # X-axis labels
        if labels:
            label_line = "        "
            step = max(1, len(labels[-width:]) // 5)
            for i, label in enumerate(labels[-width:]):
                if i % step == 0:
                    label_line += label[:2]
                else:
                    label_line += "  "
            lines.append(label_line)
        
        return "\n".join(lines)
    
    def render_summary(self, summary: dict) -> str:
        """Render a complete evaluation summary."""
        output = []
        
        # Header
        output.append(self.header("EVALUATION DASHBOARD"))
        output.append(f"\nRun ID: {summary.get('run_id', 'Unknown')}")
        output.append(f"Timestamp: {summary.get('timestamp', 'Unknown')}")
        
        # Overall stats
        output.append(self.section("📊 OVERALL STATISTICS"))
        total = summary.get("total_tests", 0)
        passed = summary.get("passed_tests", 0)
        failed = summary.get("failed_tests", 0)
        
        output.append(f"\n  Total Tests:  {total}")
        output.append(f"  Passed:       {passed} ({passed/total*100:.1f}%)" if total > 0 else "  Passed: 0")
        output.append(f"  Failed:       {failed} ({failed/total*100:.1f}%)" if total > 0 else "  Failed: 0")
        
        # Metrics
        output.append(self.section("📈 METRICS"))
        output.append(self.metric_line("Overall Accuracy", summary.get("overall_accuracy", 0)))
        output.append(self.metric_line("Intent Accuracy", summary.get("intent_accuracy", 0)))
        output.append(self.metric_line("Entity F1", summary.get("entity_f1", 0)))
        output.append(self.metric_line("Tool Accuracy", summary.get("tool_accuracy", 0)))
        
        if summary.get("semantic_similarity", 0) > 0:
            output.append(self.metric_line("Semantic Similarity", summary.get("semantic_similarity", 0)))
        
        # Category breakdown
        if summary.get("by_category"):
            output.append(self.section("📁 CATEGORY BREAKDOWN"))
            
            cat_data = {}
            for cat, data in summary["by_category"].items():
                total = data.get("passed", 0) + data.get("failed", 0)
                if total > 0:
                    cat_data[cat] = data.get("passed", 0) / total
            
            output.append(self.heatmap(cat_data, "Pass Rates by Category"))
        
        # Difficulty breakdown
        if summary.get("by_difficulty"):
            output.append(self.section("📊 DIFFICULTY BREAKDOWN"))
            
            headers = ["Tier", "Passed", "Failed", "Rate"]
            rows = []
            for tier in sorted(summary["by_difficulty"].keys()):
                data = summary["by_difficulty"][tier]
                total = data.get("passed", 0) + data.get("failed", 0)
                rate = data.get("passed", 0) / total * 100 if total > 0 else 0
                rows.append([tier, data.get("passed", 0), data.get("failed", 0), f"{rate:.0f}%"])
            
            output.append("\n" + self.table(headers, rows))
        
        # Timing
        output.append(self.section("⏱️  TIMING"))
        output.append(f"\n  Total Time:     {summary.get('total_time_seconds', 0):.2f}s")
        output.append(f"  Avg per Test:   {summary.get('avg_time_per_test_ms', 0):.2f}ms")
        
        output.append("\n" + "=" * self.width)
        
        return "\n".join(output)
    
    def render_trends(self, trend_data: dict) -> str:
        """Render trend analysis."""
        output = []
        output.append(self.header("HISTORICAL TRENDS"))
        
        if "error" in trend_data:
            output.append(f"\nError: {trend_data['error']}")
            return "\n".join(output)
        
        # Metric trends
        if "metrics" in trend_data:
            output.append(self.section("📈 METRIC TRENDS"))
            
            for metric, data in trend_data["metrics"].items():
                current = data.get("current", 0)
                average = data.get("average", 0)
                trend = data.get("trend", "stable")
                
                trend_indicator = "↑" if trend == "improving" else "↓" if trend == "declining" else "→"
                
                output.append(f"\n  {metric}:")
                output.append(f"    Current: {current*100:.1f}%")
                output.append(f"    Average: {average*100:.1f}%")
                output.append(f"    Trend:   {trend_indicator} {trend}")
            
            # Chart for overall accuracy if available
            if "history" in trend_data:
                values = [h.get("overall_accuracy", 0) for h in trend_data["history"]]
                dates = [h.get("date", "")[:5] for h in trend_data["history"]]
                output.append(self.trend_chart(values, dates, "Overall Accuracy Over Time"))
        
        output.append("\n" + "=" * self.width)
        
        return "\n".join(output)


# ============================================================================
# HTML REPORT GENERATOR
# ============================================================================

class HTMLReportGenerator:
    """Generate HTML reports for evaluation results."""
    
    def __init__(self):
        self.template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BioPipelines Evaluation Report</title>
    <style>
        :root {
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --text-primary: #eee;
            --text-secondary: #aaa;
            --accent: #0f3460;
            --success: #00d26a;
            --warning: #ffc107;
            --danger: #ff4757;
            --info: #00b4d8;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
        }
        
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        
        header {
            background: var(--bg-secondary);
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        header h1 { color: var(--info); margin-bottom: 10px; }
        header .meta { color: var(--text-secondary); font-size: 0.9em; }
        
        .dashboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        
        .card {
            background: var(--bg-secondary);
            border-radius: 10px;
            padding: 20px;
            border-left: 4px solid var(--info);
        }
        
        .card h3 {
            color: var(--text-secondary);
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        
        .card .value {
            font-size: 2.5em;
            font-weight: bold;
        }
        
        .card.success { border-left-color: var(--success); }
        .card.success .value { color: var(--success); }
        
        .card.warning { border-left-color: var(--warning); }
        .card.warning .value { color: var(--warning); }
        
        .card.danger { border-left-color: var(--danger); }
        .card.danger .value { color: var(--danger); }
        
        .metrics-grid { display: grid; grid-template-columns: 1fr; gap: 15px; margin-top: 20px; }
        
        .metric-bar {
            background: var(--bg-secondary);
            border-radius: 10px;
            padding: 15px;
        }
        
        .metric-bar .label { display: flex; justify-content: space-between; margin-bottom: 8px; }
        .metric-bar .bar { height: 20px; background: var(--accent); border-radius: 10px; overflow: hidden; }
        .metric-bar .fill { height: 100%; border-radius: 10px; transition: width 0.3s; }
        
        .fill.high { background: linear-gradient(90deg, var(--success), #00ff88); }
        .fill.medium { background: linear-gradient(90deg, var(--warning), #ffdd00); }
        .fill.low { background: linear-gradient(90deg, var(--danger), #ff6b6b); }
        
        .section { margin-top: 30px; }
        .section h2 { color: var(--info); margin-bottom: 15px; }
        
        table {
            width: 100%;
            border-collapse: collapse;
            background: var(--bg-secondary);
            border-radius: 10px;
            overflow: hidden;
        }
        
        th, td { padding: 12px 15px; text-align: left; }
        th { background: var(--accent); color: var(--text-primary); }
        tr:nth-child(even) { background: rgba(255,255,255,0.05); }
        
        .status-pass { color: var(--success); }
        .status-fail { color: var(--danger); }
        
        .trend-up { color: var(--success); }
        .trend-down { color: var(--danger); }
        .trend-stable { color: var(--text-secondary); }
        
        footer {
            margin-top: 40px;
            padding: 20px;
            text-align: center;
            color: var(--text-secondary);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧬 BioPipelines Evaluation Report</h1>
            <div class="meta">
                <span>Run ID: {run_id}</span> | 
                <span>Generated: {timestamp}</span>
            </div>
        </header>
        
        <div class="dashboard">
            {summary_cards}
        </div>
        
        <div class="section">
            <h2>📈 Metrics Overview</h2>
            <div class="metrics-grid">
                {metric_bars}
            </div>
        </div>
        
        {category_section}
        
        {difficulty_section}
        
        {failures_section}
        
        <footer>
            <p>BioPipelines Chat Agent Evaluation System v2.0</p>
        </footer>
    </div>
</body>
</html>
"""
    
    def _get_color_class(self, value: float) -> str:
        """Get color class based on value."""
        if value >= 0.8:
            return "high"
        elif value >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _get_card_class(self, value: float) -> str:
        """Get card class based on value."""
        if value >= 0.8:
            return "success"
        elif value >= 0.6:
            return "warning"
        else:
            return "danger"
    
    def _summary_card(self, title: str, value: str, card_class: str = "") -> str:
        """Generate a summary card."""
        return f"""
        <div class="card {card_class}">
            <h3>{title}</h3>
            <div class="value">{value}</div>
        </div>
        """
    
    def _metric_bar(self, name: str, value: float) -> str:
        """Generate a metric bar."""
        color_class = self._get_color_class(value)
        percent = value * 100
        return f"""
        <div class="metric-bar">
            <div class="label">
                <span>{name}</span>
                <span>{percent:.1f}%</span>
            </div>
            <div class="bar">
                <div class="fill {color_class}" style="width: {percent}%"></div>
            </div>
        </div>
        """
    
    def generate(self, summary: dict, failures: list = None) -> str:
        """Generate the HTML report."""
        # Summary cards
        total = summary.get("total_tests", 0)
        passed = summary.get("passed_tests", 0)
        pass_rate = passed / total if total > 0 else 0
        
        summary_cards = [
            self._summary_card("Total Tests", str(total)),
            self._summary_card("Passed", str(passed), self._get_card_class(pass_rate)),
            self._summary_card("Failed", str(summary.get("failed_tests", 0)), 
                              "danger" if summary.get("failed_tests", 0) > 0 else "success"),
            self._summary_card("Pass Rate", f"{pass_rate*100:.1f}%", self._get_card_class(pass_rate)),
        ]
        
        # Metric bars
        metric_bars = [
            self._metric_bar("Overall Accuracy", summary.get("overall_accuracy", 0)),
            self._metric_bar("Intent Accuracy", summary.get("intent_accuracy", 0)),
            self._metric_bar("Entity F1 Score", summary.get("entity_f1", 0)),
            self._metric_bar("Tool Accuracy", summary.get("tool_accuracy", 0)),
        ]
        
        if summary.get("semantic_similarity", 0) > 0:
            metric_bars.append(
                self._metric_bar("Semantic Similarity", summary.get("semantic_similarity", 0))
            )
        
        # Category section
        category_section = ""
        if summary.get("by_category"):
            rows = []
            for cat, data in sorted(summary["by_category"].items()):
                total = data.get("passed", 0) + data.get("failed", 0)
                rate = data.get("passed", 0) / total if total > 0 else 0
                status_class = "status-pass" if rate >= 0.8 else "status-fail"
                rows.append(f"""
                <tr>
                    <td>{cat}</td>
                    <td>{data.get('passed', 0)}</td>
                    <td>{data.get('failed', 0)}</td>
                    <td class="{status_class}">{rate*100:.1f}%</td>
                </tr>
                """)
            
            category_section = f"""
            <div class="section">
                <h2>📁 Results by Category</h2>
                <table>
                    <thead>
                        <tr><th>Category</th><th>Passed</th><th>Failed</th><th>Rate</th></tr>
                    </thead>
                    <tbody>
                        {"".join(rows)}
                    </tbody>
                </table>
            </div>
            """
        
        # Difficulty section
        difficulty_section = ""
        if summary.get("by_difficulty"):
            rows = []
            for tier in sorted(summary["by_difficulty"].keys()):
                data = summary["by_difficulty"][tier]
                total = data.get("passed", 0) + data.get("failed", 0)
                rate = data.get("passed", 0) / total if total > 0 else 0
                status_class = "status-pass" if rate >= 0.8 else "status-fail"
                rows.append(f"""
                <tr>
                    <td>Tier {tier}</td>
                    <td>{data.get('passed', 0)}</td>
                    <td>{data.get('failed', 0)}</td>
                    <td class="{status_class}">{rate*100:.1f}%</td>
                </tr>
                """)
            
            difficulty_section = f"""
            <div class="section">
                <h2>📊 Results by Difficulty</h2>
                <table>
                    <thead>
                        <tr><th>Difficulty</th><th>Passed</th><th>Failed</th><th>Rate</th></tr>
                    </thead>
                    <tbody>
                        {"".join(rows)}
                    </tbody>
                </table>
            </div>
            """
        
        # Failures section
        failures_section = ""
        if failures:
            rows = []
            for f in failures[:20]:  # Show top 20 failures
                query = f.get("query", "")[:50] + "..." if len(f.get("query", "")) > 50 else f.get("query", "")
                rows.append(f"""
                <tr>
                    <td>{f.get('test_id', 'Unknown')}</td>
                    <td>{query}</td>
                    <td>{f.get('expected_intent', 'N/A')}</td>
                    <td class="status-fail">{f.get('actual_intent', 'N/A')}</td>
                </tr>
                """)
            
            failures_section = f"""
            <div class="section">
                <h2>❌ Top Failures</h2>
                <table>
                    <thead>
                        <tr><th>Test ID</th><th>Query</th><th>Expected</th><th>Actual</th></tr>
                    </thead>
                    <tbody>
                        {"".join(rows)}
                    </tbody>
                </table>
            </div>
            """
        
        # Generate final HTML - use Template for safe substitution
        from string import Template
        
        # Replace format placeholders with Template style
        template_str = self.template.replace("{run_id}", "$run_id")
        template_str = template_str.replace("{timestamp}", "$timestamp")
        template_str = template_str.replace("{summary_cards}", "$summary_cards")
        template_str = template_str.replace("{metric_bars}", "$metric_bars")
        template_str = template_str.replace("{category_section}", "$category_section")
        template_str = template_str.replace("{difficulty_section}", "$difficulty_section")
        template_str = template_str.replace("{failures_section}", "$failures_section")
        
        template = Template(template_str)
        return template.safe_substitute(
            run_id=summary.get("run_id", "Unknown"),
            timestamp=summary.get("timestamp", datetime.now().isoformat()),
            summary_cards="\n".join(summary_cards),
            metric_bars="\n".join(metric_bars),
            category_section=category_section,
            difficulty_section=difficulty_section,
            failures_section=failures_section
        )


# ============================================================================
# DASHBOARD MANAGER
# ============================================================================

class DashboardManager:
    """Manage dashboard display and report generation."""
    
    def __init__(self, reports_dir: str = None):
        self.reports_dir = Path(reports_dir) if reports_dir else project_root / "reports" / "evaluations"
        self.terminal = TerminalDashboard()
        self.html_generator = HTMLReportGenerator()
    
    def get_latest_summary(self) -> Optional[dict]:
        """Get the most recent evaluation summary."""
        if not self.reports_dir.exists():
            return None
        
        summaries = list(self.reports_dir.glob("*_summary.json"))
        if not summaries:
            return None
        
        # Sort by modification time
        latest = max(summaries, key=lambda p: p.stat().st_mtime)
        
        with open(latest) as f:
            return json.load(f)
    
    def get_latest_failures(self) -> list:
        """Get failures from the most recent evaluation."""
        if not self.reports_dir.exists():
            return []
        
        failures_files = list(self.reports_dir.glob("*_failures.json"))
        if not failures_files:
            return []
        
        latest = max(failures_files, key=lambda p: p.stat().st_mtime)
        
        with open(latest) as f:
            return json.load(f)
    
    def show_terminal_dashboard(self, summary: dict = None):
        """Display terminal dashboard."""
        if summary is None:
            summary = self.get_latest_summary()
        
        if summary is None:
            print("No evaluation data found. Run an evaluation first.")
            return
        
        print(self.terminal.render_summary(summary))
    
    def show_trends(self, days: int = 30):
        """Display historical trends."""
        try:
            from evaluation.historical_tracker import HistoricalTracker
            
            db_path = str(project_root / "data" / "evaluation_history.db")
            tracker = HistoricalTracker(db_path)
            
            trend_data = tracker.get_trend_analysis(days=days)
            print(self.terminal.render_trends(trend_data))
            
        except ImportError:
            print("Historical tracker not available.")
        except Exception as e:
            print(f"Error loading trends: {e}")
    
    def generate_html_report(self, output_path: str, summary: dict = None):
        """Generate HTML report."""
        if summary is None:
            summary = self.get_latest_summary()
        
        if summary is None:
            print("No evaluation data found.")
            return
        
        failures = self.get_latest_failures()
        
        html = self.html_generator.generate(summary, failures)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            f.write(html)
        
        print(f"HTML report generated: {output_path}")
    
    def compare_runs(self, run_id1: str, run_id2: str):
        """Compare two evaluation runs."""
        summary1_path = self.reports_dir / f"{run_id1}_summary.json"
        summary2_path = self.reports_dir / f"{run_id2}_summary.json"
        
        if not summary1_path.exists() or not summary2_path.exists():
            print("One or both run summaries not found.")
            return
        
        with open(summary1_path) as f:
            summary1 = json.load(f)
        with open(summary2_path) as f:
            summary2 = json.load(f)
        
        print(self.terminal.header("RUN COMPARISON"))
        print(f"\nRun 1: {run_id1}")
        print(f"Run 2: {run_id2}")
        
        print(self.terminal.section("METRIC COMPARISON"))
        
        metrics = ["overall_accuracy", "intent_accuracy", "entity_f1", "tool_accuracy"]
        
        for metric in metrics:
            v1 = summary1.get(metric, 0)
            v2 = summary2.get(metric, 0)
            diff = v2 - v1
            
            indicator = "↑" if diff > 0 else "↓" if diff < 0 else "→"
            
            print(f"\n  {metric}:")
            print(f"    Run 1: {v1*100:.1f}%")
            print(f"    Run 2: {v2*100:.1f}%")
            print(f"    Change: {indicator} {abs(diff)*100:.1f}%")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Metrics dashboard for BioPipelines evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--last-run",
        action="store_true",
        help="Show dashboard for the last evaluation run"
    )
    parser.add_argument(
        "--trends",
        action="store_true",
        help="Show historical trends"
    )
    parser.add_argument(
        "--html-report",
        metavar="PATH",
        help="Generate HTML report to specified path"
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("RUN_ID_1", "RUN_ID_2"),
        help="Compare two evaluation runs"
    )
    parser.add_argument(
        "--reports-dir",
        help="Directory containing evaluation reports"
    )
    
    args = parser.parse_args()
    
    dashboard = DashboardManager(args.reports_dir)
    
    if args.trends:
        dashboard.show_trends()
    elif args.html_report:
        dashboard.generate_html_report(args.html_report)
    elif args.compare:
        dashboard.compare_runs(args.compare[0], args.compare[1])
    else:
        # Default: show last run
        dashboard.show_terminal_dashboard()


if __name__ == "__main__":
    main()
