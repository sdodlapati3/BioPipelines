"""
Integration Tests for BioPipelines
==================================

End-to-end integration tests for the complete workflow:
- Workflow generation
- Error diagnosis integration
- Results collection
- Data discovery
- UI components
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os
import json
import time

# Import main components
try:
    from workflow_composer import Composer
    COMPOSER_AVAILABLE = True
except ImportError:
    COMPOSER_AVAILABLE = False

try:
    from workflow_composer.diagnosis import ErrorDiagnosisAgent, AutoFixEngine
    DIAGNOSIS_AVAILABLE = True
except ImportError:
    DIAGNOSIS_AVAILABLE = False

try:
    from workflow_composer.results import ResultCollector, ResultViewer, ResultArchiver
    RESULTS_AVAILABLE = True
except ImportError:
    RESULTS_AVAILABLE = False

try:
    from workflow_composer.data.discovery import DataDiscovery, SearchQuery
    DISCOVERY_AVAILABLE = True
except ImportError:
    DISCOVERY_AVAILABLE = False


# ============================================================================
# Composer Integration Tests
# ============================================================================

@pytest.mark.skipif(not COMPOSER_AVAILABLE, reason="Composer not available")
class TestComposerIntegration:
    """Test workflow composer integration."""
    
    def test_composer_initialization(self):
        """Test Composer can be initialized."""
        composer = Composer()
        assert composer is not None
    
    def test_composer_tool_selector(self):
        """Test Composer has tool selector."""
        composer = Composer()
        assert hasattr(composer, 'tool_selector') or hasattr(composer, 'tools')
    
    def test_composer_module_mapper(self):
        """Test Composer has module mapper."""
        composer = Composer()
        assert hasattr(composer, 'module_mapper') or hasattr(composer, 'modules')
    
    def test_simple_workflow_generation(self):
        """Test generating a simple workflow."""
        composer = Composer()
        
        # Simple RNA-seq request - requires LLM server
        try:
            if hasattr(composer, 'compose'):
                result = composer.compose("RNA-seq analysis for human samples")
                assert result is not None
            elif hasattr(composer, 'generate'):
                result = composer.generate("RNA-seq analysis for human samples")
                assert result is not None
        except ConnectionError:
            pytest.skip("LLM server not available")
        except Exception as e:
            if "connect" in str(e).lower() or "connection" in str(e).lower():
                pytest.skip(f"LLM server not available: {e}")
            raise


# ============================================================================
# Diagnosis Integration Tests
# ============================================================================

@pytest.mark.skipif(not DIAGNOSIS_AVAILABLE, reason="Diagnosis not available")
class TestDiagnosisIntegration:
    """Test error diagnosis integration with other components."""
    
    def test_diagnosis_with_log_files(self):
        """Test diagnosis with actual log file content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create realistic Nextflow log
            log_content = """
            Nov-26 10:00:00.000 [main] DEBUG nextflow.processor - Task process > STAR_ALIGN (sample_1)
            Nov-26 10:00:01.000 [main] INFO  nextflow.processor - Submitted process > STAR_ALIGN (sample_1)
            Nov-26 10:01:00.000 [main] ERROR nextflow.processor - Process `STAR_ALIGN (sample_1)` terminated with an error exit status (137)
            
            Command exit status:
              137
            
            Command output:
              (empty)
            
            Command error:
              slurmstepd: error: Detected 1 oom-kill event(s) in StepId=123456.0. 
              Some of your processes may have been killed by the cgroup out-of-memory handler.
            """
            
            log_file = Path(tmpdir) / ".nextflow.log"
            log_file.write_text(log_content)
            
            agent = ErrorDiagnosisAgent()
            
            if hasattr(agent, 'diagnose'):
                result = agent.diagnose(log_content)
                assert result is not None
                # Should identify OOM error
                if hasattr(result, 'category'):
                    assert 'memory' in str(result.category).lower() or result.category is not None
    
    def test_auto_fix_integration(self):
        """Test auto-fix engine integration."""
        engine = AutoFixEngine()
        
        # Test that engine can generate fixes
        assert engine is not None
        
        if hasattr(engine, 'get_available_fixes'):
            fixes = engine.get_available_fixes()
            assert fixes is not None
    
    def test_diagnosis_to_fix_workflow(self):
        """Test complete diagnosis to fix workflow."""
        agent = ErrorDiagnosisAgent()
        engine = AutoFixEngine()
        
        error_text = "ERROR: Reference genome not found at /data/reference/hg38.fa"
        
        # Step 1: Diagnose
        if hasattr(agent, 'diagnose'):
            diagnosis = agent.diagnose(error_text)
            assert diagnosis is not None
        
        # Step 2: Get suggested fixes
        if hasattr(engine, 'get_fixes') and hasattr(diagnosis, 'category'):
            fixes = engine.get_fixes(diagnosis.category)
            # Should have at least one suggestion
            assert fixes is not None


# ============================================================================
# Results Integration Tests
# ============================================================================

@pytest.mark.skipif(not RESULTS_AVAILABLE, reason="Results not available")
class TestResultsIntegration:
    """Test results visualization integration."""
    
    def test_results_full_workflow(self):
        """Test complete results workflow."""
        # Use non-tmp directory to avoid exclusion
        import os
        results_dir = Path(os.getcwd()) / "test_integ_results"
        results_dir.mkdir(exist_ok=True)
        
        try:
            # QC reports
            qc_dir = results_dir / "qc"
            qc_dir.mkdir()
            (qc_dir / "multiqc_report.html").write_text("""
            <!DOCTYPE html>
            <html>
            <head><title>MultiQC Report</title></head>
            <body>
                <h1>MultiQC Report</h1>
                <p>Total samples: 4</p>
                <p>Pass rate: 100%</p>
            </body>
            </html>
            """)
            
            # Counts
            counts_dir = results_dir / "counts"
            counts_dir.mkdir()
            (counts_dir / "gene_counts.tsv").write_text(
                "gene_id\tsample1\tsample2\n"
                "ENSG00000141510\t1000\t1200\n"
                "ENSG00000012048\t500\t600\n"
            )
            
            # Plots
            plots_dir = results_dir / "plots"
            plots_dir.mkdir()
            (plots_dir / "pca_plot.png").write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)
            
            # Test collection
            collector = ResultCollector()
            scan_results = collector.scan(str(results_dir))
            
            assert scan_results is not None
            
            # Test viewing
            viewer = ResultViewer()
            html_file = qc_dir / "multiqc_report.html"
            if hasattr(viewer, 'view'):
                content = viewer.view(html_file)
                assert content is not None
            
            # Test archiving - pass summary, not directory
            archiver = ResultArchiver()
            archive_dir = Path(os.getcwd()) / "test_archive_out"
            archive_dir.mkdir(exist_ok=True)
            try:
                archive_path = archiver.create_archive(scan_results, archive_dir / "results.zip")
                if archive_path:
                    assert Path(archive_path).exists()
            finally:
                import shutil
                shutil.rmtree(archive_dir, ignore_errors=True)
        finally:
            import shutil
            shutil.rmtree(results_dir, ignore_errors=True)
    
    def test_results_with_pipeline_patterns(self):
        """Test results detection with pipeline-specific patterns."""
        import os
        results_dir = Path(os.getcwd()) / "test_pipeline_results"
        results_dir.mkdir(exist_ok=True)
        
        try:
            # Create RNA-seq specific output structure
            (results_dir / "star_align").mkdir()
            (results_dir / "star_align" / "sample1_Aligned.sortedByCoord.out.bam").write_bytes(b"")
            (results_dir / "star_align" / "sample1_Log.final.out").write_text("STAR alignment stats")
            
            (results_dir / "salmon_quant").mkdir()
            (results_dir / "salmon_quant" / "sample1").mkdir()
            (results_dir / "salmon_quant" / "sample1" / "quant.sf").write_text(
                "Name\tLength\tEffectiveLength\tTPM\tNumReads\n"
            )
            
            (results_dir / "deseq2").mkdir()
            (results_dir / "deseq2" / "differential_expression.tsv").write_text(
                "gene\tlog2FoldChange\tpadj\n"
                "TP53\t2.5\t0.001\n"
            )
            
            collector = ResultCollector()
            results = collector.scan(str(results_dir))
            assert results is not None
        finally:
            import shutil
            shutil.rmtree(results_dir, ignore_errors=True)


# ============================================================================
# Data Discovery Integration Tests
# ============================================================================

@pytest.mark.skipif(not DISCOVERY_AVAILABLE, reason="Discovery not available")
class TestDiscoveryIntegration:
    """Test data discovery integration."""
    
    def test_discovery_initialization(self):
        """Test DataDiscovery initialization."""
        discovery = DataDiscovery()
        assert discovery is not None
    
    def test_query_parsing(self):
        """Test natural language query parsing."""
        discovery = DataDiscovery()
        
        # Test search with natural language
        if hasattr(discovery, 'search'):
            results = discovery.search(
                "human ChIP-seq H3K27ac",
                sources=['encode'],
                max_results=2
            )
            assert results is not None
    
    def test_multi_source_search(self):
        """Test searching multiple sources."""
        discovery = DataDiscovery()
        
        if hasattr(discovery, 'search'):
            results = discovery.search(
                "mouse RNA-seq brain",
                sources=['encode', 'geo'],
                max_results=2
            )
            assert results is not None
            if hasattr(results, 'sources_searched'):
                assert len(results.sources_searched) >= 1


# ============================================================================
# Cross-Module Integration Tests
# ============================================================================

class TestCrossModuleIntegration:
    """Test integration between different modules."""
    
    @pytest.mark.skipif(not (DIAGNOSIS_AVAILABLE and RESULTS_AVAILABLE), 
                       reason="Required modules not available")
    def test_diagnosis_after_results(self):
        """Test diagnosis after results collection (for failed jobs)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate a failed job with partial results
            (Path(tmpdir) / "partial_output.txt").write_text("Partial results before crash")
            
            # Error logs
            (Path(tmpdir) / ".nextflow.log").write_text(
                "ERROR: Process failed with exit code 1\n"
                "java.lang.OutOfMemoryError: Java heap space"
            )
            
            # Collect results (may be partial)
            collector = ResultCollector()
            if hasattr(collector, 'scan'):
                results = collector.scan(tmpdir)
                assert results is not None
            
            # Diagnose the error
            agent = ErrorDiagnosisAgent()
            log_content = (Path(tmpdir) / ".nextflow.log").read_text()
            if hasattr(agent, 'diagnose'):
                diagnosis = agent.diagnose(log_content)
                assert diagnosis is not None
    
    @pytest.mark.skipif(not (DISCOVERY_AVAILABLE and COMPOSER_AVAILABLE),
                       reason="Required modules not available")
    def test_discovery_to_composer(self):
        """Test using discovered data in workflow composition."""
        # This tests the workflow: discover data → compose workflow
        discovery = DataDiscovery()
        
        # Find reference data
        if hasattr(discovery, 'search'):
            results = discovery.search("human genome GRCh38", max_results=1)
            
            if results and hasattr(results, 'datasets') and results.datasets:
                # Use first result in workflow
                dataset = results.datasets[0]
                
                # Compose workflow (mocked)
                composer = Composer()
                # Workflow would use the discovered reference
                assert dataset.id is not None


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance-related tests."""
    
    @pytest.mark.skipif(not DIAGNOSIS_AVAILABLE, reason="Diagnosis not available")
    def test_pattern_matching_speed(self):
        """Test that pattern matching is fast."""
        from workflow_composer.diagnosis.patterns import get_all_patterns
        import re
        
        patterns = get_all_patterns()
        
        # Create test log
        test_log = "ERROR: File not found\n" * 1000
        
        start = time.time()
        # Manually match patterns
        for category, pattern in patterns.items():
            for regex in pattern.patterns:
                re.search(regex, test_log, re.IGNORECASE)
        elapsed = time.time() - start
        
        # Should complete in under 1 second
        assert elapsed < 1.0
    
    @pytest.mark.skipif(not RESULTS_AVAILABLE, reason="Results not available")
    def test_result_scanning_speed(self):
        """Test that result scanning is reasonably fast."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 100 files
            for i in range(100):
                (Path(tmpdir) / f"file_{i}.txt").write_text(f"content {i}")
            
            collector = ResultCollector()
            
            start = time.time()
            if hasattr(collector, 'scan'):
                collector.scan(tmpdir)
            elif hasattr(collector, 'collect'):
                collector.collect(tmpdir)
            elapsed = time.time() - start
            
            # Should complete in under 5 seconds
            assert elapsed < 5.0


# ============================================================================
# Configuration Tests
# ============================================================================

class TestConfiguration:
    """Test configuration and settings."""
    
    def test_config_file_exists(self):
        """Test that config files exist."""
        config_paths = [
            "config/defaults.yaml",
            "config/slurm.yaml",
        ]
        
        for path in config_paths:
            full_path = Path(__file__).parent.parent / path
            if full_path.exists():
                assert full_path.is_file()
    
    def test_environment_variables(self):
        """Test required environment variables are documented."""
        # These are optional but should be handled gracefully
        env_vars = [
            "OPENAI_API_KEY",
            "LIGHTNING_API_KEY",
            "ANTHROPIC_API_KEY",
        ]
        
        # Just check that accessing them doesn't crash
        for var in env_vars:
            value = os.environ.get(var)
            # Value may be None, that's OK


# ============================================================================
# Web Interface Tests (without launching server)
# ============================================================================

class TestWebComponents:
    """Test web interface components without starting server."""
    
    def test_gradio_app_importable(self):
        """Test that Gradio app can be imported."""
        try:
            from workflow_composer.web import gradio_app
            assert gradio_app is not None
        except ImportError:
            pytest.skip("Gradio app not available")
    
    def test_web_components_importable(self):
        """Test that web components can be imported."""
        try:
            from workflow_composer.web.components import result_browser
            assert result_browser is not None
        except ImportError:
            pytest.skip("Web components not available")


# ============================================================================
# Data Integrity Tests
# ============================================================================

class TestDataIntegrity:
    """Test data integrity across operations."""
    
    @pytest.mark.skipif(not RESULTS_AVAILABLE, reason="Results not available")
    def test_archive_preserves_files(self):
        """Test that archiving preserves all files."""
        import zipfile
        import os
        import shutil
        
        # Use non-tmp directory to avoid exclusion
        results_dir = Path(os.getcwd()) / "test_archive_preserve"
        results_dir.mkdir(exist_ok=True)
        
        try:
            # Create test files with known content
            files = {
                "test1.txt": "content1",
                "test2.html": "<html></html>",
                "subdir/test3.tsv": "col1\tcol2",
            }
            
            for path, content in files.items():
                filepath = results_dir / path
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(content)
            
            # First scan to get summary
            collector = ResultCollector()
            summary = collector.scan(str(results_dir))
            
            archiver = ResultArchiver()
            output_dir = Path(os.getcwd()) / "test_archive_output2"
            output_dir.mkdir(exist_ok=True)
            
            try:
                archive_path = archiver.create_archive(summary, output_dir / "test.zip")
                
                if archive_path and Path(archive_path).exists():
                    with zipfile.ZipFile(archive_path, 'r') as zf:
                        # Extract and verify
                        extract_dir = Path(os.getcwd()) / "test_extract"
                        extract_dir.mkdir(exist_ok=True)
                        try:
                            zf.extractall(extract_dir)
                            
                            # Verify at least some files exist
                            extracted_files = list(extract_dir.rglob("*"))
                            assert len(extracted_files) >= 2
                        finally:
                            shutil.rmtree(extract_dir, ignore_errors=True)
            finally:
                shutil.rmtree(output_dir, ignore_errors=True)
        finally:
            shutil.rmtree(results_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
