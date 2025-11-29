#!/usr/bin/env python3
"""
Code Quality Report Generator
==============================
Uses professional open-source tools to identify code issues:
- Vulture: Dead code detection
- Pylint: Duplicate code detection  
- Radon: Code complexity analysis
- Flake8: Style and bug detection

This script REPORTS only - no automatic fixes.
Review the output and fix manually.

Usage:
    python scripts/code_quality_report.py                    # Full report
    python scripts/code_quality_report.py --duplicates       # Duplicates only
    python scripts/code_quality_report.py --dead-code        # Dead code only
    python scripts/code_quality_report.py --complexity       # Complexity only
    python scripts/code_quality_report.py --output report.md # Save to file
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_command(cmd: list, timeout: int = 120) -> str:
    """Run a command and return output."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent.parent
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT after {timeout}s]"
    except FileNotFoundError:
        return f"[TOOL NOT FOUND: {cmd[0]}]"


def check_tool(name: str) -> bool:
    """Check if a tool is installed."""
    result = subprocess.run(
        ["which", name],
        capture_output=True
    )
    return result.returncode == 0


def vulture_report(src_dir: str = "src/workflow_composer") -> str:
    """Generate dead code report using Vulture."""
    lines = [
        "## 💀 Dead Code Detection (Vulture)",
        "",
        "Vulture finds unused code with confidence scores.",
        "Higher confidence = more likely truly unused.",
        "",
        "```"
    ]
    
    output = run_command([
        "vulture", src_dir, 
        "--min-confidence", "80"
    ])
    
    # Parse and format
    issues = output.strip().split('\n')
    if issues and issues[0]:
        lines.extend(issues[:50])  # Limit to 50 issues
        if len(issues) > 50:
            lines.append(f"... and {len(issues) - 50} more issues")
    else:
        lines.append("No dead code found with >80% confidence")
    
    lines.append("```")
    lines.append("")
    lines.append(f"**Total issues:** {len([i for i in issues if i.strip()])}")
    
    return "\n".join(lines)


def pylint_duplicates_report(src_dir: str = "src/workflow_composer") -> str:
    """Generate duplicate code report using Pylint."""
    lines = [
        "## 🔄 Duplicate Code Detection (Pylint)",
        "",
        "Pylint identifies similar code blocks (min 10 lines).",
        "Review each case - some duplication may be intentional.",
        "",
        "```"
    ]
    
    output = run_command([
        "pylint", src_dir,
        "--disable=all",
        "--enable=duplicate-code",
        "--min-similarity-lines=10"
    ], timeout=300)
    
    # Parse output
    current_block = []
    blocks = []
    
    for line in output.split('\n'):
        if 'R0801' in line or 'Similar lines' in line:
            if current_block:
                blocks.append('\n'.join(current_block))
            current_block = [line]
        elif current_block and line.strip():
            current_block.append(line)
    
    if current_block:
        blocks.append('\n'.join(current_block))
    
    if blocks:
        for block in blocks[:10]:  # Show first 10 duplicate groups
            lines.append(block)
            lines.append("---")
        if len(blocks) > 10:
            lines.append(f"... and {len(blocks) - 10} more duplicate groups")
    else:
        lines.append("No significant duplicate code found")
    
    lines.append("```")
    lines.append("")
    lines.append(f"**Total duplicate groups:** {len(blocks)}")
    
    return "\n".join(lines)


def radon_complexity_report(src_dir: str = "src/workflow_composer") -> str:
    """Generate code complexity report using Radon."""
    lines = [
        "## 📊 Code Complexity Analysis (Radon)",
        "",
        "Complexity grades: A (simple) → F (very complex)",
        "Focus on C, D, E, F rated functions for refactoring.",
        "",
        "### High Complexity Functions (C or worse)",
        "```"
    ]
    
    output = run_command([
        "radon", "cc", src_dir,
        "-a", "-s",
        "--min", "C",  # Only show C or worse
        "--total-average"
    ])
    
    complex_funcs = [l for l in output.split('\n') if l.strip() and not l.startswith('src/')]
    
    # Group by file
    current_file = None
    for line in output.split('\n'):
        if line.startswith('src/'):
            current_file = line.strip()
        elif line.strip() and '- C' in line or '- D' in line or '- E' in line or '- F' in line:
            if current_file:
                lines.append(f"{current_file}")
                current_file = None
            lines.append(f"  {line.strip()}")
    
    if len([l for l in lines if l.strip()]) <= 7:  # Just headers
        lines.append("No high-complexity functions found (all A or B rated)")
    
    lines.append("```")
    
    # Get average
    avg_line = [l for l in output.split('\n') if 'Average complexity' in l]
    if avg_line:
        lines.append("")
        lines.append(f"**{avg_line[0].strip()}**")
    
    return "\n".join(lines)


def radon_maintainability_report(src_dir: str = "src/workflow_composer") -> str:
    """Generate maintainability index report."""
    lines = [
        "## 🔧 Maintainability Index (Radon)",
        "",
        "MI grades: A (highly maintainable) → C (difficult to maintain)",
        "",
        "### Files with Low Maintainability (C rated)",
        "```"
    ]
    
    output = run_command([
        "radon", "mi", src_dir,
        "-s",
        "--min", "C"  # Only show C or worse
    ])
    
    bad_files = [l for l in output.split('\n') if l.strip() and ' - C' in l]
    
    if bad_files:
        lines.extend(bad_files[:20])
        if len(bad_files) > 20:
            lines.append(f"... and {len(bad_files) - 20} more files")
    else:
        lines.append("All files have good maintainability (A or B rated)")
    
    lines.append("```")
    
    return "\n".join(lines)


def generate_summary(src_dir: str = "src/workflow_composer") -> str:
    """Generate a quick summary."""
    
    # Count issues
    vulture_out = run_command(["vulture", src_dir, "--min-confidence", "80"])
    dead_code_count = len([l for l in vulture_out.split('\n') if l.strip()])
    
    pylint_out = run_command([
        "pylint", src_dir, "--disable=all", "--enable=duplicate-code",
        "--min-similarity-lines=10"
    ], timeout=300)
    dup_count = pylint_out.count('R0801')
    
    radon_out = run_command(["radon", "cc", src_dir, "-a", "-s", "--min", "C"])
    complex_count = radon_out.count('- C') + radon_out.count('- D') + radon_out.count('- E') + radon_out.count('- F')
    
    lines = [
        "## 📋 Summary",
        "",
        "| Category | Count | Priority |",
        "|----------|-------|----------|",
        f"| 💀 Dead Code (>80% confidence) | {dead_code_count} | Medium |",
        f"| 🔄 Duplicate Code Groups | {dup_count} | High |",
        f"| 📊 High Complexity Functions | {complex_count} | Medium |",
        "",
        "### Recommended Actions",
        "",
        "1. **Duplicate Code**: Review each duplicate group. Consolidate if truly identical.",
        "2. **Dead Code**: Verify with `git log` - if unused for >6 months, consider removal.",
        "3. **Complexity**: Break down C/D/E/F functions into smaller units.",
        ""
    ]
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate code quality report")
    parser.add_argument("--duplicates", action="store_true", help="Show duplicate code only")
    parser.add_argument("--dead-code", action="store_true", help="Show dead code only")
    parser.add_argument("--complexity", action="store_true", help="Show complexity only")
    parser.add_argument("--maintainability", action="store_true", help="Show maintainability only")
    parser.add_argument("--output", "-o", type=str, help="Save report to file")
    parser.add_argument("--src", type=str, default="src/workflow_composer", help="Source directory")
    
    args = parser.parse_args()
    
    # Check tools
    tools = ["vulture", "pylint", "radon"]
    missing = [t for t in tools if not check_tool(t)]
    if missing:
        print(f"❌ Missing tools: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)
    
    # Generate report
    report_parts = [
        "# 📊 Code Quality Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Source: `{args.src}`",
        "",
        "---",
        ""
    ]
    
    show_all = not any([args.duplicates, args.dead_code, args.complexity, args.maintainability])
    
    if show_all:
        print("Generating full code quality report...")
        report_parts.append(generate_summary(args.src))
    
    if args.duplicates or show_all:
        print("  Checking for duplicate code...")
        report_parts.append(pylint_duplicates_report(args.src))
        report_parts.append("")
    
    if args.dead_code or show_all:
        print("  Checking for dead code...")
        report_parts.append(vulture_report(args.src))
        report_parts.append("")
    
    if args.complexity or show_all:
        print("  Analyzing complexity...")
        report_parts.append(radon_complexity_report(args.src))
        report_parts.append("")
    
    if args.maintainability or show_all:
        print("  Checking maintainability...")
        report_parts.append(radon_maintainability_report(args.src))
    
    report = "\n".join(report_parts)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\n✅ Report saved to: {args.output}")
    else:
        print("\n" + report)


if __name__ == "__main__":
    main()
