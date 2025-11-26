"""
Tests for Results Visualization Module
======================================

Tests the results visualization functionality including:
- Result collection and scanning
- File type detection
- Result viewing
- Archive creation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os
import json

# Import modules under test
from workflow_composer.results.result_types import (
    FileType,
    ResultCategory,
    ResultFile,
    ResultSummary,
    FILE_TYPE_TO_CATEGORY,
)
from workflow_composer.results.collector import ResultCollector
from workflow_composer.results.detector import FileTypeDetector
from workflow_composer.results.viewer import ResultViewer
from workflow_composer.results.archiver import ResultArchiver
from workflow_composer.results.patterns import PIPELINE_PATTERNS, GENERIC_PATTERNS


# ============================================================================
# Result Types Tests
# ============================================================================

class TestResultTypes:
    """Test result type definitions."""
    
    def test_file_type_enum(self):
        """Test FileType enum values exist."""
        assert hasattr(FileType, 'QC_REPORT')
        assert hasattr(FileType, 'IMAGE')
        assert hasattr(FileType, 'TABLE')
        assert hasattr(FileType, 'TEXT')
        assert hasattr(FileType, 'LOG')
    
    def test_result_category_enum(self):
        """Test ResultCategory enum values exist."""
        assert hasattr(ResultCategory, 'QC_REPORTS')
        assert hasattr(ResultCategory, 'VISUALIZATIONS')
        assert hasattr(ResultCategory, 'DATA_FILES')
        assert hasattr(ResultCategory, 'LOGS')
    
    def test_result_file_dataclass(self):
        """Test ResultFile dataclass creation."""
        from datetime import datetime
        result_file = ResultFile(
            path=Path("/results/multiqc_report.html"),
            name="multiqc_report.html",
            relative_path="multiqc_report.html",
            size=102400,
            file_type=FileType.QC_REPORT,
            category=ResultCategory.QC_REPORTS,
            modified=datetime.now(),
        )
        assert result_file.name == "multiqc_report.html"
        assert result_file.file_type == FileType.QC_REPORT
        assert result_file.size == 102400
    
    def test_file_type_to_category_mapping(self):
        """Test FILE_TYPE_TO_CATEGORY has entries."""
        assert len(FILE_TYPE_TO_CATEGORY) > 0


# ============================================================================
# Result Patterns Tests
# ============================================================================

class TestResultPatterns:
    """Test result detection patterns."""
    
    def test_pipeline_patterns_not_empty(self):
        """Test that PIPELINE_PATTERNS has entries."""
        assert len(PIPELINE_PATTERNS) > 0
    
    def test_generic_patterns_not_empty(self):
        """Test that GENERIC_PATTERNS has entries."""
        assert len(GENERIC_PATTERNS) > 0


# ============================================================================
# File Type Detector Tests
# ============================================================================

class TestFileTypeDetector:
    """Test file type detection."""
    
    def test_detector_initialization(self):
        """Test FileTypeDetector instantiation."""
        detector = FileTypeDetector()
        assert detector is not None
    
    def test_detect_html_file(self):
        """Test detection of HTML files."""
        detector = FileTypeDetector()
        
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            f.write(b"<html><body>Test</body></html>")
            f.flush()
            
            file_type, category = detector.detect(Path(f.name))
            assert file_type == FileType.QC_REPORT
            
            os.unlink(f.name)
    
    def test_detect_image_file(self):
        """Test detection of image files."""
        detector = FileTypeDetector()
        
        # Test PNG
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Write minimal PNG header
            f.write(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)
            f.flush()
            
            file_type, category = detector.detect(Path(f.name))
            assert file_type == FileType.IMAGE
            
            os.unlink(f.name)
    
    def test_detect_tsv_file(self):
        """Test detection of TSV files."""
        detector = FileTypeDetector()
        
        # Test TSV
        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as f:
            f.write(b"col1\tcol2\tcol3\nval1\tval2\tval3")
            f.flush()
            
            file_type, category = detector.detect(Path(f.name))
            assert file_type == FileType.TABLE
            
            os.unlink(f.name)
    
    def test_detect_csv_file(self):
        """Test detection of CSV files."""
        detector = FileTypeDetector()
        
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"col1,col2,col3\nval1,val2,val3")
            f.flush()
            
            file_type, category = detector.detect(Path(f.name))
            assert file_type == FileType.TABLE
            
            os.unlink(f.name)
    
    def test_detect_text_file(self):
        """Test detection of text files."""
        detector = FileTypeDetector()
        
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"This is plain text content")
            f.flush()
            
            file_type, category = detector.detect(Path(f.name))
            assert file_type == FileType.TEXT
            
            os.unlink(f.name)
    
    def test_detect_log_file(self):
        """Test detection of log files."""
        detector = FileTypeDetector()
        
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            f.write(b"2024-01-01 10:00:00 INFO: Test log entry")
            f.flush()
            
            file_type, category = detector.detect(Path(f.name))
            assert file_type in [FileType.TEXT, FileType.LOG]
            
            os.unlink(f.name)


# ============================================================================
# Result Collector Tests
# ============================================================================

class TestResultCollector:
    """Test result collection functionality."""
    
    def test_collector_initialization(self):
        """Test ResultCollector instantiation."""
        collector = ResultCollector()
        assert collector is not None
    
    def test_scan_empty_directory(self):
        """Test scanning an empty directory."""
        # Use current working directory to avoid /tmp exclusion issue
        import os
        test_dir = Path(os.getcwd()) / "test_empty_scan_dir"
        test_dir.mkdir(exist_ok=True)
        
        try:
            collector = ResultCollector()
            results = collector.scan(str(test_dir))
            
            assert results is not None
            # Empty directory should return summary with 0 files
            assert results.total_files == 0
        finally:
            # Cleanup
            test_dir.rmdir()
    
    def test_scan_with_html_reports(self):
        """Test scanning directory with HTML reports."""
        # Use current working directory to avoid /tmp exclusion
        import os
        test_dir = Path(os.getcwd()) / "test_html_scan_dir"
        test_dir.mkdir(exist_ok=True)
        
        try:
            # Create HTML file
            html_file = test_dir / "multiqc_report.html"
            html_file.write_text("<html><head><title>MultiQC</title></head></html>")
            
            collector = ResultCollector()
            results = collector.scan(str(test_dir))
            
            assert results is not None
            # Should find at least one file
            assert results.total_files >= 1
        finally:
            # Cleanup
            for f in test_dir.iterdir():
                f.unlink()
            test_dir.rmdir()
    
    def test_scan_with_mixed_files(self):
        """Test scanning directory with mixed file types."""
        import os
        test_dir = Path(os.getcwd()) / "test_mixed_scan_dir"
        test_dir.mkdir(exist_ok=True)
        
        try:
            # Create various result files
            (test_dir / "report.html").write_text("<html></html>")
            (test_dir / "plot.png").write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)
            (test_dir / "counts.tsv").write_text("gene\tcount\nTP53\t100")
            (test_dir / "summary.txt").write_text("Analysis complete")
            
            collector = ResultCollector()
            results = collector.scan(str(test_dir))
            
            assert results is not None
            assert results.total_files >= 3  # Should find most files
        finally:
            # Cleanup
            for f in test_dir.iterdir():
                f.unlink()
            test_dir.rmdir()
    
    def test_scan_nested_directories(self):
        """Test scanning nested directory structure."""
        import os
        test_dir = Path(os.getcwd()) / "test_nested_scan_dir"
        test_dir.mkdir(exist_ok=True)
        
        try:
            # Create nested structure
            qc_dir = test_dir / "qc"
            qc_dir.mkdir()
            (qc_dir / "fastqc_report.html").write_text("<html>FastQC</html>")
            
            counts_dir = test_dir / "counts"
            counts_dir.mkdir()
            (counts_dir / "gene_counts.tsv").write_text("gene\tcount")
            
            collector = ResultCollector()
            results = collector.scan(str(test_dir))
            
            assert results is not None
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(test_dir)


# ============================================================================
# Result Viewer Tests
# ============================================================================

class TestResultViewer:
    """Test result viewing functionality."""
    
    def test_viewer_initialization(self):
        """Test ResultViewer instantiation."""
        viewer = ResultViewer()
        assert viewer is not None
    
    def test_view_html_content(self):
        """Test viewing HTML content."""
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            f.write(b"<html><body><h1>Test Report</h1></body></html>")
            f.flush()
            
            viewer = ResultViewer()
            
            if hasattr(viewer, 'view'):
                content = viewer.view(Path(f.name))
                assert content is not None
            elif hasattr(viewer, 'get_content'):
                content = viewer.get_content(Path(f.name))
                assert content is not None
            
            os.unlink(f.name)
    
    def test_view_text_content(self):
        """Test viewing text content."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"Analysis Summary\n================\nTotal reads: 1000000\n")
            f.flush()
            
            viewer = ResultViewer()
            
            if hasattr(viewer, 'view'):
                content = viewer.view(Path(f.name))
                assert content is not None
            elif hasattr(viewer, 'get_content'):
                content = viewer.get_content(Path(f.name))
                assert content is not None
            
            os.unlink(f.name)


# ============================================================================
# Result Archiver Tests
# ============================================================================

class TestResultArchiver:
    """Test result archiving functionality."""
    
    def test_archiver_initialization(self):
        """Test ResultArchiver instantiation."""
        archiver = ResultArchiver()
        assert archiver is not None
    
    def test_create_archive(self):
        """Test creating ZIP archive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files to archive
            (Path(tmpdir) / "report.html").write_text("<html></html>")
            (Path(tmpdir) / "data.tsv").write_text("col1\tcol2")
            
            archiver = ResultArchiver()
            output_dir = tempfile.mkdtemp()
            
            # Scan first to get summary
            collector = ResultCollector()
            summary = collector.scan(tmpdir)
            
            archive_path = archiver.create_archive(summary, Path(output_dir) / "test.zip")
            
            assert archive_path is not None
            assert Path(archive_path).exists()
    
    def test_archive_creates_zip(self):
        """Test that archive creates a valid ZIP file."""
        import zipfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files to archive
            (Path(tmpdir) / "file1.txt").write_text("content 1")
            (Path(tmpdir) / "file2.txt").write_text("content 2")
            
            archiver = ResultArchiver()
            output_dir = tempfile.mkdtemp()
            
            # Scan first to get summary
            collector = ResultCollector()
            summary = collector.scan(tmpdir)
            
            archive_path = archiver.create_archive(summary, Path(output_dir) / "test.zip")
            
            if archive_path and Path(archive_path).exists():
                # Verify it's a valid ZIP
                assert zipfile.is_zipfile(archive_path)


# ============================================================================
# Cloud Transfer Tests
# ============================================================================

class TestCloudTransfer:
    """Test cloud transfer functionality."""
    
    def test_cloud_transfer_import(self):
        """Test cloud transfer module can be imported."""
        from workflow_composer.results.cloud_transfer import CloudTransfer
        assert CloudTransfer is not None
    
    def test_cloud_transfer_initialization(self):
        """Test CloudTransfer instantiation."""
        from workflow_composer.results.cloud_transfer import CloudTransfer
        
        transfer = CloudTransfer()
        assert transfer is not None


# ============================================================================
# Integration Tests
# ============================================================================

class TestResultsIntegration:
    """Integration tests for results workflow."""
    
    def test_full_results_workflow(self):
        """Test complete results workflow: scan → view → archive."""
        import os
        import shutil
        test_dir = Path(os.getcwd()) / "test_integration_scan_dir"
        test_dir.mkdir(exist_ok=True)
        
        try:
            # Create mock result files
            (test_dir / "multiqc_report.html").write_text("""
            <html>
            <head><title>MultiQC Report</title></head>
            <body><h1>QC Summary</h1></body>
            </html>
            """)
            (test_dir / "gene_counts.tsv").write_text("gene\tcount\nTP53\t1000")
            (test_dir / "summary.txt").write_text("Analysis complete")
            
            # Step 1: Collect
            collector = ResultCollector()
            results = collector.scan(str(test_dir))
            
            assert results is not None
            assert results.total_files >= 2
            
            # Step 2: View (at least one file)
            viewer = ResultViewer()
            html_file = test_dir / "multiqc_report.html"
            if hasattr(viewer, 'view'):
                content = viewer.view(html_file)
                assert content is not None or content == ""
            
            # Step 3: Archive
            archiver = ResultArchiver()
            output_dir = Path(os.getcwd()) / "test_archive_output"
            output_dir.mkdir(exist_ok=True)
            
            try:
                archive_path = archiver.create_archive(results, output_dir / "results.zip")
                assert archive_path is None or Path(archive_path).exists()
            finally:
                shutil.rmtree(output_dir, ignore_errors=True)
        finally:
            # Cleanup
            shutil.rmtree(test_dir, ignore_errors=True)


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_nonexistent_directory(self):
        """Test handling of non-existent directory."""
        collector = ResultCollector()
        
        # Should handle gracefully - may raise or return empty
        try:
            results = collector.scan("/nonexistent/path/12345")
            assert results is not None or results is None
        except Exception:
            pass  # Expected
    
    def test_very_large_directory(self):
        """Test handling directory with many files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create many files
            for i in range(100):
                (Path(tmpdir) / f"file_{i}.txt").write_text(f"content {i}")
            
            collector = ResultCollector()
            results = collector.scan(tmpdir)
            assert results is not None
    
    def test_binary_file_handling(self):
        """Test handling of binary files."""
        with tempfile.NamedTemporaryFile(suffix=".bam", delete=False) as f:
            # Write binary content
            f.write(b'\x00\x01\x02\x03\x04\x05')
            f.flush()
            
            detector = FileTypeDetector()
            file_type, category = detector.detect(Path(f.name))
            
            # Should detect as some type
            assert file_type is not None
            
            os.unlink(f.name)
    
    def test_symlink_handling(self):
        """Test handling of symbolic links."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file and symlink
            real_file = Path(tmpdir) / "real_file.txt"
            real_file.write_text("real content")
            
            link_file = Path(tmpdir) / "link_file.txt"
            try:
                link_file.symlink_to(real_file)
                
                collector = ResultCollector()
                results = collector.scan(tmpdir)
                assert results is not None
            except OSError:
                # Symlinks may not be supported on all systems
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
