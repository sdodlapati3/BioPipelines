#!/usr/bin/env python3
"""
Test Runner for BioPipelines Query Flow Validation
===================================================

Comprehensive test execution suite for tomorrow's validation.
Runs all component tests in sequence with reporting.

Usage:
    python test_runner.py                    # Run all tests
    python test_runner.py --components frontend,intent  # Run specific components
    python test_runner.py --suite integration          # Run specific suite
    python test_runner.py --manual                     # Run manual checklist
"""

import subprocess
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import time
from collections import defaultdict

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class TestRunner:
    """Main test runner for BioPipelines validation."""
    
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.test_dir = workspace_root / "tests"
        self.results = defaultdict(list)
        self.timings = {}
        self.start_time = None
        
        # Test configurations
        self.components = {
            'frontend': {
                'description': 'Gradio UI Tab Tests',
                'tests': [],  # Manual or special handling
                'required': False
            },
            'intent': {
                'description': 'Intent Parsing Tests',
                'file': 'test_workflow_composer.py::TestImports',
                'required': True
            },
            'tool_selection': {
                'description': 'Tool Selector Tests',
                'file': 'test_integration.py::TestComposerIntegration',
                'required': True
            },
            'module_mapping': {
                'description': 'Module Mapping Tests',
                'file': 'test_workflow_composer.py::TestConfig',
                'required': True
            },
            'workflow_gen': {
                'description': 'Workflow Generation Tests',
                'file': 'test_workflow_composer.py::TestLLMFactory',
                'required': True
            },
            'execution': {
                'description': 'SLURM Execution Tests',
                'file': 'test_integration.py::TestComposerIntegration',
                'required': False
            },
            'monitoring': {
                'description': 'Job Monitoring Tests',
                'file': 'test_integration.py::TestPerformance',
                'required': False
            },
            'diagnosis': {
                'description': 'Error Diagnosis Tests',
                'file': 'test_diagnosis.py::TestErrorDiagnosisAgent',
                'required': True
            },
            'results': {
                'description': 'Results Processing Tests',
                'file': 'test_results.py',
                'required': True
            },
            'discovery': {
                'description': 'Data Discovery Tests',
                'file': 'test_data_discovery.py::TestDataDiscovery',
                'required': True
            }
        }
    
    def print_header(self, text: str):
        """Print formatted section header."""
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")
    
    def print_status(self, text: str, status: str):
        """Print status with color."""
        if status == "PASS":
            color = Colors.GREEN
            symbol = "✓"
        elif status == "FAIL":
            color = Colors.RED
            symbol = "✗"
        elif status == "SKIP":
            color = Colors.YELLOW
            symbol = "⊘"
        else:  # INFO
            color = Colors.CYAN
            symbol = "ℹ"
        
        print(f"{color}{symbol} {text:<60} [{status}]{Colors.RESET}")
    
    def run_pytest(self, test_path: str, verbose: bool = False) -> Tuple[bool, str, float]:
        """Run pytest on a specific test path."""
        cmd = [
            sys.executable, "-m", "pytest",
            test_path,
            "-v" if verbose else "-q",
            "--tb=short",
            "-x",  # Stop on first failure
            "--color=yes"
        ]
        
        start = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=self.test_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            elapsed = time.time() - start
            
            passed = result.returncode == 0
            output = result.stdout + result.stderr
            
            return passed, output, elapsed
        except subprocess.TimeoutExpired:
            return False, "Test timed out after 5 minutes", 300.0
        except Exception as e:
            return False, f"Error running test: {str(e)}", 0.0
    
    def run_component_tests(self, components: List[str] = None, verbose: bool = False):
        """Run tests for specified components."""
        if components is None:
            components = list(self.components.keys())
        
        self.print_header("COMPONENT TESTING")
        
        passed_count = 0
        failed_count = 0
        
        for component in components:
            if component not in self.components:
                self.print_status(f"Unknown component: {component}", "SKIP")
                continue
            
            config = self.components[component]
            self.print_status(f"Running: {config['description']}", "INFO")
            
            if not config.get('file'):
                self.print_status(f"{config['description']}", "SKIP")
                continue
            
            test_file = self.test_dir / config['file']
            
            # Check if file exists
            test_file_base = str(test_file).split('::')[0]
            if not Path(test_file_base).exists():
                self.print_status(f"{config['description']}", "SKIP")
                continue
            
            passed, output, elapsed = self.run_pytest(str(test_file), verbose)
            
            self.timings[component] = elapsed
            
            if passed:
                self.print_status(f"{config['description']} ({elapsed:.2f}s)", "PASS")
                passed_count += 1
            else:
                self.print_status(f"{config['description']} ({elapsed:.2f}s)", "FAIL")
                failed_count += 1
                if verbose:
                    print(f"{Colors.YELLOW}Output:{Colors.RESET}")
                    print(output[-500:])  # Last 500 chars
        
        return passed_count, failed_count
    
    def run_integration_tests(self, verbose: bool = False):
        """Run integration tests."""
        self.print_header("INTEGRATION TESTING")
        
        test_suites = [
            ('ComposerIntegration', 'test_integration.py::TestComposerIntegration'),
            ('DiagnosisIntegration', 'test_integration.py::TestDiagnosisIntegration'),
            ('ResultsIntegration', 'test_integration.py::TestResultsIntegration'),
            ('DiscoveryIntegration', 'test_integration.py::TestDiscoveryIntegration'),
        ]
        
        passed_count = 0
        failed_count = 0
        
        for suite_name, test_path in test_suites:
            self.print_status(f"Integration: {suite_name}", "INFO")
            
            test_file = self.test_dir / test_path
            test_file_base = str(test_file).split('::')[0]
            
            if not Path(test_file_base).exists():
                self.print_status(f"Integration: {suite_name}", "SKIP")
                continue
            
            passed, output, elapsed = self.run_pytest(str(test_file), verbose)
            
            if passed:
                self.print_status(f"Integration: {suite_name} ({elapsed:.2f}s)", "PASS")
                passed_count += 1
            else:
                self.print_status(f"Integration: {suite_name} ({elapsed:.2f}s)", "FAIL")
                failed_count += 1
        
        return passed_count, failed_count
    
    def run_performance_tests(self, verbose: bool = False):
        """Run performance benchmarks."""
        self.print_header("PERFORMANCE TESTING")
        
        test_path = 'test_integration.py::TestPerformance'
        self.print_status("Running performance benchmarks", "INFO")
        
        test_file = self.test_dir / test_path
        if not Path(str(test_file).split('::')[0]).exists():
            self.print_status("Performance tests", "SKIP")
            return 0, 1
        
        passed, output, elapsed = self.run_pytest(str(test_file), verbose)
        
        if passed:
            self.print_status(f"Performance benchmarks ({elapsed:.2f}s)", "PASS")
            return 1, 0
        else:
            self.print_status(f"Performance benchmarks ({elapsed:.2f}s)", "FAIL")
            return 0, 1
    
    def print_manual_checklist(self):
        """Print manual testing checklist."""
        self.print_header("MANUAL TESTING CHECKLIST")
        
        checklist = {
            "FRONTEND": [
                "Launch Gradio UI: python -m workflow_composer.web",
                "Test Workflow tab: Submit 'RNA-seq human' query",
                "Test Diagnosis tab: Upload test .nextflow.log file",
                "Test Results tab: Browse sample results directory",
                "Test Data Discovery tab: Search 'human ChIP-seq H3K27ac'",
                "Verify tab switching preserves state",
            ],
            "INTENT_PARSING": [
                "Verify 'RNA-seq human samples' → RNA_SEQ_DE intent",
                "Verify 'ChIP-seq H3K27ac' → CHIP_SEQ with target",
                "Verify confidence scores reasonable (0.7-1.0)",
                "Test edge cases: vague queries, typos",
            ],
            "TOOL_SELECTION": [
                "Verify RNA-seq → nf-core/rnaseq selected",
                "Verify ChIP-seq → nf-core/chipseq selected",
                "Verify ATAC-seq → nf-core/atacseq selected",
                "Test fallback for ambiguous queries",
            ],
            "MODULE_MAPPING": [
                "Verify STAR/Salmon modules available",
                "Check parameter validation works",
                "Verify resource defaults reasonable",
            ],
            "WORKFLOW_GENERATION": [
                "Run: nextflow lint generated_workflow/main.nf",
                "Verify main.nf contains all expected processes",
                "Check params.yaml properly formatted",
                "Verify workflow compiles without errors",
            ],
            "EXECUTION": [
                "Test SLURM submission: sbatch ...",
                "Verify job ID returned",
                "Check job appears in squeue",
                "Monitor job progress with squeue/sacct",
            ],
            "MONITORING": [
                "Observe real-time status updates (30s intervals)",
                "Check CPU/memory tracking from SLURM",
                "Verify log streaming works",
                "Test timeout handling",
            ],
            "DIAGNOSIS": [
                "Upload OOM error log → verify OUT_OF_MEMORY detected",
                "Upload file-not-found error → verify FILE_NOT_FOUND detected",
                "Upload container error → verify CONTAINER_ERROR detected",
                "Check suggested fixes are appropriate",
            ],
            "RESULTS": [
                "Verify RNA-seq outputs detected:",
                "  - MultiQC report viewable",
                "  - BAM files indexed",
                "  - Gene count tables displayed",
                "  - QC metrics highlighted",
                "Test PDF export functionality",
            ],
            "DATA_DISCOVERY": [
                "Search 'human ChIP-seq H3K27ac' → ENCODE results",
                "Search 'mouse brain RNA-seq' → GEO results",
                "Search 'human genome GRCh38' → Ensembl results",
                "Verify download links working",
                "Test multi-source parallel search timing",
            ],
        }
        
        for category, items in checklist.items():
            print(f"\n{Colors.BOLD}{Colors.CYAN}{category}{Colors.RESET}")
            for i, item in enumerate(items, 1):
                print(f"  {i:2d}. {Colors.YELLOW}[ ]{Colors.RESET} {item}")
    
    def print_summary(self, total_passed: int, total_failed: int):
        """Print test summary."""
        self.print_header("TEST SUMMARY")
        
        total = total_passed + total_failed
        success_rate = (total_passed / total * 100) if total > 0 else 0
        
        print(f"Total Tests: {total}")
        print(f"{Colors.GREEN}Passed: {total_passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {total_failed}{Colors.RESET}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        print(f"\n{Colors.BOLD}Timing Breakdown:{Colors.RESET}")
        for component, timing in sorted(self.timings.items(), key=lambda x: x[1], reverse=True):
            print(f"  {component:20s}: {timing:7.2f}s")
        
        total_time = sum(self.timings.values())
        print(f"{'Total':20s}: {total_time:7.2f}s")
        
        # Recommendation
        if total_failed == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✓ READY TO DEPLOY{Colors.RESET}")
        elif success_rate >= 95:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠ MOSTLY READY (minor issues){Colors.RESET}")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}✗ NOT READY (critical issues){Colors.RESET}")
    
    def save_results(self, filename: str = None):
        """Save test results to JSON."""
        if filename is None:
            filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results_file = self.workspace_root / filename
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'components': self.components,
            'timings': self.timings,
            'results': dict(self.results)
        }
        
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='BioPipelines Test Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_runner.py                                    # Run all tests
  python test_runner.py --components frontend,intent      # Run specific components
  python test_runner.py --suite integration                # Run integration suite
  python test_runner.py --manual                           # Show manual checklist
  python test_runner.py --all -v                           # All tests, verbose
        """
    )
    
    parser.add_argument(
        '--components',
        help='Comma-separated list of components to test',
        default=None
    )
    parser.add_argument(
        '--suite',
        choices=['unit', 'integration', 'performance'],
        help='Run specific test suite',
        default=None
    )
    parser.add_argument(
        '--manual',
        action='store_true',
        help='Show manual testing checklist'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all tests (component + integration + performance)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save results to JSON file'
    )
    
    args = parser.parse_args()
    
    # Find workspace root
    workspace_root = Path(__file__).parent
    if not (workspace_root / 'tests').exists():
        workspace_root = Path.cwd()
    
    runner = TestRunner(workspace_root)
    
    total_passed = 0
    total_failed = 0
    
    # Show manual checklist
    if args.manual:
        runner.print_manual_checklist()
        return
    
    # Run tests
    if args.all or not any([args.components, args.suite]):
        # Run all components
        components_to_run = list(runner.components.keys())
        p, f = runner.run_component_tests(components_to_run, args.verbose)
        total_passed += p
        total_failed += f
        
        # Run integration tests
        p, f = runner.run_integration_tests(args.verbose)
        total_passed += p
        total_failed += f
        
        # Run performance tests
        p, f = runner.run_performance_tests(args.verbose)
        total_passed += p
        total_failed += f
    else:
        # Run specified components
        if args.components:
            components = args.components.split(',')
            p, f = runner.run_component_tests(components, args.verbose)
            total_passed += p
            total_failed += f
        
        # Run specified suite
        if args.suite == 'integration':
            p, f = runner.run_integration_tests(args.verbose)
            total_passed += p
            total_failed += f
        elif args.suite == 'performance':
            p, f = runner.run_performance_tests(args.verbose)
            total_passed += p
            total_failed += f
    
    # Print summary
    runner.print_summary(total_passed, total_failed)
    
    # Save results if requested
    if args.save:
        runner.save_results()
    
    # Exit with appropriate code
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == '__main__':
    main()
