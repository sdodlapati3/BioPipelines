#!/usr/bin/env python3
"""
BioPipelines Environment Validation
====================================

Pre-testing validation to ensure all components are ready.
Run this before starting the full test suite tomorrow.

Usage:
    python validate_environment.py              # Full validation
    python validate_environment.py --llm        # Check LLM only
    python validate_environment.py --slurm      # Check SLURM only
    python validate_environment.py --quick      # Quick checks only
"""

import sys
import subprocess
import os
from pathlib import Path
from typing import Dict, Tuple
import json
import socket
import urllib.request
import urllib.error

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class EnvironmentValidator:
    """Validates BioPipelines environment setup."""
    
    def __init__(self):
        self.workspace_root = Path(__file__).parent
        self.results = {}
        self.warnings = []
        self.errors = []
    
    def print_header(self, text: str):
        """Print section header."""
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")
    
    def check(self, name: str, passed: bool, message: str = ""):
        """Record check result."""
        status = f"{Colors.GREEN}✓{Colors.RESET}" if passed else f"{Colors.RED}✗{Colors.RESET}"
        detail = f" - {message}" if message else ""
        print(f"{status} {name:<40}{detail}")
        
        if not passed:
            self.errors.append(name)
        
        self.results[name] = passed
    
    def warn(self, name: str, message: str):
        """Record warning."""
        print(f"{Colors.YELLOW}⚠{Colors.RESET} {name:<40} - {message}")
        self.warnings.append((name, message))
    
    def check_python(self):
        """Check Python installation and version."""
        self.print_header("PYTHON ENVIRONMENT")
        
        # Python version
        version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        required = (3, 8)
        if sys.version_info >= required:
            self.check("Python version", True, f"{version} (required: ≥3.8)")
        else:
            self.check("Python version", False, f"{version} (required: ≥3.8)")
        
        # Executable path
        exec_path = sys.executable
        self.check("Python executable", True, exec_path)
        
        # Virtual environment
        in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        if in_venv:
            self.check("Virtual environment", True, sys.prefix)
        else:
            self.warn("Virtual environment", "Not in venv (recommended but not required)")
    
    def check_imports(self):
        """Check critical Python imports."""
        self.print_header("REQUIRED PACKAGES")
        
        packages = {
            'workflow_composer': 'Main package',
            'pytest': 'Testing framework',
            'nextflow': 'Nextflow integration',
            'yaml': 'YAML parsing',
            'requests': 'HTTP requests',
            'pandas': 'Data processing',
            'matplotlib': 'Visualization',
            'gradio': 'Web UI framework',
        }
        
        for package, description in packages.items():
            try:
                __import__(package)
                self.check(f"Import {package}", True, description)
            except ImportError as e:
                self.check(f"Import {package}", False, description)
    
    def check_nextflow(self):
        """Check Nextflow installation."""
        self.print_header("NEXTFLOW")
        
        # Check command exists
        try:
            result = subprocess.run(
                ['nextflow', '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                output = result.stdout.strip()
                self.check("Nextflow installed", True, output.split('\n')[0])
            else:
                self.check("Nextflow installed", False, "Command failed")
        except FileNotFoundError:
            self.check("Nextflow installed", False, "Command not found")
        except Exception as e:
            self.check("Nextflow installed", False, str(e))
    
    def check_slurm(self):
        """Check SLURM availability."""
        self.print_header("SLURM CLUSTER")
        
        # Check sinfo
        try:
            result = subprocess.run(
                ['sinfo', '--partition=all', '--Format=partition,nodes,available'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self.check("SLURM cluster", True, "Responsive")
                # Show available partitions
                lines = result.stdout.strip().split('\n')[:5]
                for line in lines:
                    print(f"    {line}")
            else:
                self.check("SLURM cluster", False, "sinfo failed")
        except FileNotFoundError:
            self.check("SLURM cluster", False, "sinfo not found")
        except Exception as e:
            self.check("SLURM cluster", False, str(e))
        
        # Check sacctmgr
        try:
            result = subprocess.run(
                ['sacctmgr', 'list', 'user', os.environ.get('USER', 'unknown'), '--noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                user_accounts = len(result.stdout.strip().split('\n'))
                self.check("User accounts", True, f"{user_accounts} account(s)")
            else:
                self.warn("User accounts", "Could not list accounts")
        except FileNotFoundError:
            self.warn("User accounts", "sacctmgr not found")
    
    def check_llm(self):
        """Check LLM endpoints."""
        self.print_header("LLM ENDPOINTS")
        
        # Check Ollama
        try:
            urllib.request.urlopen('http://localhost:11434/api/tags', timeout=2)
            self.check("Ollama server", True, "http://localhost:11434")
        except (urllib.error.URLError, socket.timeout):
            self.warn("Ollama server", "Not responding at localhost:11434 (start with 'ollama serve')")
        except Exception as e:
            self.warn("Ollama server", str(e))
        
        # Check OpenAI API key
        if os.environ.get('OPENAI_API_KEY'):
            key_preview = os.environ['OPENAI_API_KEY'][:20] + "..."
            self.check("OpenAI API key", True, key_preview)
        else:
            self.warn("OpenAI API key", "Not set (optional if using Ollama)")
        
        # Check Anthropic API key
        if os.environ.get('ANTHROPIC_API_KEY'):
            key_preview = os.environ['ANTHROPIC_API_KEY'][:20] + "..."
            self.check("Anthropic API key", True, key_preview)
        else:
            self.warn("Anthropic API key", "Not set (optional)")
    
    def check_data(self):
        """Check data directories and test data."""
        self.print_header("DATA DIRECTORIES")
        
        directories = {
            'tests': 'Test suite',
            'tests/test_data': 'Test data',
            'data/raw': 'Raw data',
            'data/processed': 'Processed data',
            'data/results': 'Results output',
            'generated_workflows': 'Generated workflows',
            'config': 'Configuration',
        }
        
        for path, description in directories.items():
            full_path = self.workspace_root / path
            if full_path.exists():
                file_count = len(list(full_path.glob('*')))
                self.check(f"Directory {path}", True, f"{file_count} items")
            else:
                self.check(f"Directory {path}", False, description)
    
    def check_config(self):
        """Check configuration files."""
        self.print_header("CONFIGURATION")
        
        configs = {
            'config/defaults.yaml': 'Default configuration',
            'config/slurm.yaml': 'SLURM profile',
            'config/composer.yaml': 'Composer settings',
            'src/workflow_composer/config.py': 'Config module',
        }
        
        for path, description in configs.items():
            full_path = self.workspace_root / path
            if full_path.exists():
                size_kb = full_path.stat().st_size / 1024
                self.check(f"Config {path}", True, f"{size_kb:.1f}KB")
            else:
                self.check(f"Config {path}", False, description)
    
    def check_workspace(self):
        """Check workspace structure."""
        self.print_header("WORKSPACE STRUCTURE")
        
        # Check main directories
        main_dirs = ['src', 'tests', 'config', 'data', 'docs', 'generated_workflows']
        present = sum(1 for d in main_dirs if (self.workspace_root / d).exists())
        self.check(f"Main directories", present >= 6, f"{present}/{len(main_dirs)} present")
        
        # Check test files
        test_files = [
            'tests/test_workflow_composer.py',
            'tests/test_diagnosis.py',
            'tests/test_results.py',
            'tests/test_data_discovery.py',
            'tests/test_integration.py',
        ]
        test_present = sum(1 for f in test_files if (self.workspace_root / f).exists())
        self.check(f"Test files", test_present >= 4, f"{test_present}/{len(test_files)} present")
    
    def check_disk_space(self):
        """Check available disk space."""
        self.print_header("SYSTEM RESOURCES")
        
        import shutil
        stat = shutil.disk_usage(self.workspace_root)
        
        free_gb = stat.free / (1024**3)
        total_gb = stat.total / (1024**3)
        used_pct = (stat.used / stat.total * 100) if stat.total > 0 else 0
        
        if free_gb > 50:
            self.check("Disk space", True, f"{free_gb:.1f}GB free ({used_pct:.0f}% used)")
        elif free_gb > 10:
            self.warn("Disk space", f"Only {free_gb:.1f}GB free (consider cleanup)")
        else:
            self.check("Disk space", False, f"Only {free_gb:.1f}GB free (critical)")
        
        # Check memory
        try:
            result = subprocess.run(
                ['free', '-h'],
                capture_output=True,
                text=True,
                timeout=2
            )
            lines = result.stdout.strip().split('\n')
            mem_line = lines[1]  # Mem: line
            self.check("System memory", True, mem_line.split()[1:4])
        except Exception as e:
            self.warn("System memory", str(e))
    
    def validate_all(self):
        """Run all validations."""
        self.print_header("BIOPIPELINES ENVIRONMENT VALIDATION")
        
        self.check_python()
        self.check_imports()
        self.check_workspace()
        self.check_config()
        self.check_disk_space()
        self.check_nextflow()
        self.check_slurm()
        self.check_llm()
        self.check_data()
        
        self.print_summary()
    
    def validate_quick(self):
        """Run quick validation."""
        self.print_header("QUICK VALIDATION")
        
        self.check_python()
        self.check_imports()
        self.check_workspace()
        self.check_disk_space()
        
        self.print_summary()
    
    def validate_llm_only(self):
        """Validate LLM endpoints only."""
        self.print_header("LLM VALIDATION")
        self.check_llm()
        self.print_summary()
    
    def validate_slurm_only(self):
        """Validate SLURM only."""
        self.print_header("SLURM VALIDATION")
        self.check_slurm()
        self.print_summary()
    
    def print_summary(self):
        """Print validation summary."""
        self.print_header("VALIDATION SUMMARY")
        
        total = len(self.results)
        passed = sum(1 for v in self.results.values() if v)
        failed = total - passed
        
        print(f"Results: {Colors.GREEN}{passed} passed{Colors.RESET}, " 
              f"{Colors.RED}{failed} failed{Colors.RESET}, "
              f"{len(self.warnings)} warnings")
        
        if self.errors:
            print(f"\n{Colors.RED}{Colors.BOLD}Errors:{Colors.RESET}")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}Warnings:{Colors.RESET}")
            for name, msg in self.warnings:
                print(f"  - {name}: {msg}")
        
        # Overall status
        if not self.errors and len(self.warnings) <= 2:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✓ ENVIRONMENT READY FOR TESTING{Colors.RESET}")
            return 0
        elif not self.errors:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠ ENVIRONMENT MOSTLY READY (minor issues){Colors.RESET}")
            return 1
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}✗ ENVIRONMENT NOT READY (fix errors above){Colors.RESET}")
            return 2
    
    def save_report(self, filename: str = None):
        """Save validation report."""
        if filename is None:
            from datetime import datetime
            filename = f"env_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'results': self.results,
            'errors': self.errors,
            'warnings': self.warnings,
        }
        
        report_path = self.workspace_root / filename
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport saved: {report_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='BioPipelines Environment Validator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_environment.py              # Full validation
  python validate_environment.py --quick      # Quick checks
  python validate_environment.py --llm        # LLM endpoints only
  python validate_environment.py --slurm      # SLURM only
  python validate_environment.py --save       # Save report to JSON
        """
    )
    
    parser.add_argument('--quick', action='store_true', help='Quick validation only')
    parser.add_argument('--llm', action='store_true', help='Check LLM endpoints only')
    parser.add_argument('--slurm', action='store_true', help='Check SLURM only')
    parser.add_argument('--save', action='store_true', help='Save report to JSON')
    
    args = parser.parse_args()
    
    validator = EnvironmentValidator()
    
    if args.quick:
        validator.validate_quick()
    elif args.llm:
        validator.validate_llm_only()
    elif args.slurm:
        validator.validate_slurm_only()
    else:
        validator.validate_all()
    
    if args.save:
        validator.save_report()
    
    return validator.print_summary()


if __name__ == '__main__':
    sys.exit(main())
