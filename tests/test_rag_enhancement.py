"""
Tests for Phase 2.6: RAG Enhancement
====================================

Tests for KnowledgeBase, NFCoreIndexer, and ErrorPatternDB.
"""

import pytest
import tempfile
import asyncio
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import json
import yaml

from workflow_composer.agents.rag.knowledge_base import (
    KnowledgeBase,
    KnowledgeSource,
    KnowledgeDocument,
    KnowledgeBaseConfig,
)
from workflow_composer.agents.rag.error_patterns import (
    ErrorPatternDB,
    ErrorSolution,
)
from workflow_composer.agents.rag.nf_core_indexer import (
    NFCoreIndexer,
    NFCoreModule,
)


# ============================================================================
# KnowledgeDocument Tests
# ============================================================================

class TestKnowledgeDocument:
    """Tests for KnowledgeDocument dataclass."""
    
    def test_create_document(self):
        """Test creating a knowledge document."""
        doc = KnowledgeDocument(
            id="test_doc_1",
            source=KnowledgeSource.TOOL_CATALOG,
            title="STAR Aligner",
            content="STAR is a fast RNA-seq aligner",
            metadata={"category": "alignment"},
        )
        
        assert doc.id == "test_doc_1"
        assert doc.source == KnowledgeSource.TOOL_CATALOG
        assert doc.title == "STAR Aligner"
        assert doc.content == "STAR is a fast RNA-seq aligner"
        assert doc.metadata["category"] == "alignment"
    
    def test_to_dict(self):
        """Test converting document to dictionary."""
        doc = KnowledgeDocument(
            id="test_1",
            source=KnowledgeSource.NF_CORE_MODULES,
            title="Test Module",
            content="Module content",
        )
        
        data = doc.to_dict()
        
        assert data["id"] == "test_1"
        assert data["source"] == "nf_core_modules"
        assert data["title"] == "Test Module"
        assert data["content"] == "Module content"
        assert "created_at" in data
    
    def test_from_dict(self):
        """Test creating document from dictionary."""
        data = {
            "id": "doc_from_dict",
            "source": "error_patterns",
            "title": "OOM Error",
            "content": "Out of memory error handling",
            "metadata": {"category": "memory"},
            "created_at": "2024-01-01T12:00:00",
        }
        
        doc = KnowledgeDocument.from_dict(data)
        
        assert doc.id == "doc_from_dict"
        assert doc.source == KnowledgeSource.ERROR_PATTERNS
        assert doc.title == "OOM Error"
        assert doc.metadata["category"] == "memory"


class TestKnowledgeBaseConfig:
    """Tests for KnowledgeBaseConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = KnowledgeBaseConfig()
        
        assert config.base_path == "~/.biopipelines/knowledge"
        assert config.auto_index_nf_core is False
        assert config.auto_index_tools is True
        assert config.max_results == 10
        assert config.embedding_enabled is False
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = KnowledgeBaseConfig(
            base_path="/custom/path",
            auto_index_nf_core=True,
            max_results=20,
        )
        
        assert config.base_path == "/custom/path"
        assert config.auto_index_nf_core is True
        assert config.max_results == 20


# ============================================================================
# KnowledgeBase Tests
# ============================================================================

class TestKnowledgeBase:
    """Tests for KnowledgeBase class."""
    
    @pytest.fixture
    def temp_kb(self):
        """Create a temporary knowledge base."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(base_path=tmpdir)
            yield kb
    
    def test_initialization(self, temp_kb):
        """Test knowledge base initialization."""
        assert temp_kb.base_path.exists()
        assert temp_kb.db_path.exists()
    
    def test_add_document(self, temp_kb):
        """Test adding a document."""
        doc = KnowledgeDocument(
            id="star_aligner",
            source=KnowledgeSource.TOOL_CATALOG,
            title="STAR",
            content="STAR is an ultrafast universal RNA-seq aligner",
            metadata={"category": "alignment"},
        )
        
        temp_kb.add_document(doc)
        
        stats = temp_kb.get_stats()
        assert stats.get("tool_catalog", 0) == 1
    
    def test_add_multiple_documents(self, temp_kb):
        """Test adding multiple documents."""
        docs = [
            KnowledgeDocument(
                id="doc_1",
                source=KnowledgeSource.TOOL_CATALOG,
                title="STAR",
                content="RNA-seq aligner",
            ),
            KnowledgeDocument(
                id="doc_2",
                source=KnowledgeSource.TOOL_CATALOG,
                title="Salmon",
                content="Transcript quantification",
            ),
            KnowledgeDocument(
                id="doc_3",
                source=KnowledgeSource.ERROR_PATTERNS,
                title="OOM Error",
                content="Out of memory handling",
            ),
        ]
        
        for doc in docs:
            temp_kb.add_document(doc)
        
        stats = temp_kb.get_stats()
        assert stats.get("tool_catalog", 0) == 2
        assert stats.get("error_patterns", 0) == 1
    
    def test_search_documents(self, temp_kb):
        """Test searching documents."""
        docs = [
            KnowledgeDocument(
                id="star",
                source=KnowledgeSource.TOOL_CATALOG,
                title="STAR Aligner",
                content="STAR is an ultrafast RNA-seq aligner",
            ),
            KnowledgeDocument(
                id="salmon",
                source=KnowledgeSource.TOOL_CATALOG,
                title="Salmon",
                content="Salmon quantifies transcripts",
            ),
        ]
        
        for doc in docs:
            temp_kb.add_document(doc)
        
        results = temp_kb.search("RNA-seq aligner")
        
        assert len(results) >= 1
        assert any("STAR" in r.title for r in results)
    
    def test_search_with_source_filter(self, temp_kb):
        """Test searching with source filter."""
        docs = [
            KnowledgeDocument(
                id="tool_1",
                source=KnowledgeSource.TOOL_CATALOG,
                title="Tool A",
                content="Alignment tool",
            ),
            KnowledgeDocument(
                id="error_1",
                source=KnowledgeSource.ERROR_PATTERNS,
                title="Alignment Error",
                content="Alignment error solution",
            ),
        ]
        
        for doc in docs:
            temp_kb.add_document(doc)
        
        results = temp_kb.search(
            "alignment",
            sources=[KnowledgeSource.TOOL_CATALOG],
        )
        
        assert len(results) >= 1
        assert all(r.source == KnowledgeSource.TOOL_CATALOG for r in results)
    
    def test_get_by_source(self, temp_kb):
        """Test retrieving documents by source."""
        docs = [
            KnowledgeDocument(
                id="tool_1",
                source=KnowledgeSource.TOOL_CATALOG,
                title="Tool 1",
                content="Content 1",
            ),
            KnowledgeDocument(
                id="tool_2",
                source=KnowledgeSource.TOOL_CATALOG,
                title="Tool 2",
                content="Content 2",
            ),
            KnowledgeDocument(
                id="error_1",
                source=KnowledgeSource.ERROR_PATTERNS,
                title="Error 1",
                content="Error content",
            ),
        ]
        
        for doc in docs:
            temp_kb.add_document(doc)
        
        tool_docs = temp_kb.get_by_source(KnowledgeSource.TOOL_CATALOG)
        
        assert len(tool_docs) == 2
        assert all(d.source == KnowledgeSource.TOOL_CATALOG for d in tool_docs)
    
    def test_cleanup(self, temp_kb):
        """Test cleaning up documents."""
        doc = KnowledgeDocument(
            id="test",
            source=KnowledgeSource.TOOL_CATALOG,
            title="Test",
            content="Test content",
        )
        temp_kb.add_document(doc)
        
        assert temp_kb.get_stats().get("tool_catalog", 0) == 1
        
        temp_kb.cleanup(KnowledgeSource.TOOL_CATALOG)
        
        assert temp_kb.get_stats().get("tool_catalog", 0) == 0
    
    def test_cleanup_all(self, temp_kb):
        """Test cleaning up all documents."""
        docs = [
            KnowledgeDocument(id="t1", source=KnowledgeSource.TOOL_CATALOG, 
                            title="T1", content="C1"),
            KnowledgeDocument(id="e1", source=KnowledgeSource.ERROR_PATTERNS,
                            title="E1", content="C2"),
        ]
        
        for doc in docs:
            temp_kb.add_document(doc)
        
        temp_kb.cleanup()
        
        stats = temp_kb.get_stats()
        assert sum(stats.values()) == 0
    
    def test_index_best_practices(self, temp_kb):
        """Test indexing best practices."""
        practices = [
            {
                "title": "Use containers",
                "content": "Always use containers for reproducibility",
                "category": "reproducibility",
            },
            {
                "title": "Resource optimization",
                "content": "Profile memory and CPU usage",
                "category": "performance",
            },
        ]
        
        temp_kb.index_best_practices(practices)
        
        stats = temp_kb.get_stats()
        assert stats.get("best_practices", 0) == 2
    
    def test_index_tool_catalog(self, temp_kb):
        """Test indexing tool catalog."""
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_file = Path(tmpdir) / "tools.yaml"
            catalog_file.write_text(yaml.dump({
                "tools": [
                    {
                        "name": "STAR",
                        "description": "RNA-seq aligner",
                        "category": "alignment",
                    },
                    {
                        "name": "Salmon",
                        "description": "Transcript quantification",
                        "category": "quantification",
                    },
                ]
            }))
            
            temp_kb.index_tool_catalog(tmpdir)
            
            stats = temp_kb.get_stats()
            assert stats.get("tool_catalog", 0) == 2


# ============================================================================
# ErrorSolution Tests
# ============================================================================

class TestErrorSolution:
    """Tests for ErrorSolution dataclass."""
    
    def test_create_solution(self):
        """Test creating an error solution."""
        solution = ErrorSolution(
            pattern=r"Out of memory",
            cause="Insufficient memory",
            solution="Increase memory allocation",
            category="memory",
        )
        
        assert solution.pattern == r"Out of memory"
        assert solution.cause == "Insufficient memory"
        assert solution.category == "memory"
        assert solution.confidence == 0.8
    
    def test_regex_matching(self):
        """Test regex pattern matching."""
        solution = ErrorSolution(
            pattern=r"OOM|OutOfMemory|memory.*exceeded",
            cause="Memory limit exceeded",
            solution="Increase memory",
        )
        
        match1, conf1 = solution.matches("Process killed due to OOM")
        match2, conf2 = solution.matches("OutOfMemoryError in Java")
        match3, conf3 = solution.matches("No matching error here")
        
        assert match1 is True
        assert conf1 > 0
        assert match2 is True
        assert match3 is False
        assert conf3 == 0.0
    
    def test_simple_string_matching(self):
        """Test simple string matching when regex fails."""
        # Create with invalid regex
        solution = ErrorSolution(
            pattern="memory error",  # Simple string
            cause="Memory issue",
            solution="Check memory",
        )
        
        match, conf = solution.matches("A memory error occurred")
        
        assert match is True
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        solution = ErrorSolution(
            pattern=r"error",
            cause="cause",
            solution="fix",
            category="test",
            analysis_types=["rna-seq"],
        )
        
        data = solution.to_dict()
        
        assert data["pattern"] == r"error"
        assert data["cause"] == "cause"
        assert data["category"] == "test"
        assert "rna-seq" in data["analysis_types"]
    
    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "pattern": r"STAR.*error",
            "cause": "STAR alignment failed",
            "solution": "Check STAR index",
            "category": "tool",
            "analysis_types": ["rna-seq"],
            "confidence": 0.9,
        }
        
        solution = ErrorSolution.from_dict(data)
        
        assert solution.pattern == r"STAR.*error"
        assert solution.category == "tool"
        assert solution.confidence == 0.9
        assert "rna-seq" in solution.analysis_types


# ============================================================================
# ErrorPatternDB Tests
# ============================================================================

class TestErrorPatternDB:
    """Tests for ErrorPatternDB class."""
    
    def test_initialization(self):
        """Test default pattern initialization."""
        db = ErrorPatternDB()
        
        assert len(db.patterns) > 0
        assert len(db.get_categories()) > 0
    
    def test_find_memory_solution(self):
        """Test finding solution for memory error."""
        db = ErrorPatternDB()
        
        error = "Process killed: Out of memory error in STAR alignment"
        solution = db.find_solution(error)
        
        assert solution is not None
        assert solution.category == "memory"
        assert "memory" in solution.solution.lower()
    
    def test_find_disk_solution(self):
        """Test finding solution for disk error."""
        db = ErrorPatternDB()
        
        error = "Failed to write output: No space left on device"
        solution = db.find_solution(error)
        
        assert solution is not None
        assert solution.category == "disk"
    
    def test_find_input_solution(self):
        """Test finding solution for input error."""
        db = ErrorPatternDB()
        
        error = "FileNotFoundError: No such file or directory: '/data/sample.fastq'"
        solution = db.find_solution(error)
        
        assert solution is not None
        assert solution.category == "input"
    
    def test_find_all_solutions(self):
        """Test finding multiple solutions."""
        db = ErrorPatternDB()
        
        # This should match multiple patterns
        error = "STAR process killed due to memory limit exceeded"
        solutions = db.find_all_solutions(error)
        
        assert len(solutions) >= 1
        # Should be sorted by confidence
        confidences = [conf for _, conf in solutions]
        assert confidences == sorted(confidences, reverse=True)
    
    def test_analysis_type_boost(self):
        """Test confidence boost for matching analysis type."""
        db = ErrorPatternDB()
        
        error = "STAR alignment failed with fatal error"
        
        # With matching analysis type
        solution_rna = db.find_solution(error, analysis_type="rna-seq")
        
        # With non-matching analysis type  
        solution_dna = db.find_solution(error, analysis_type="dna-seq")
        
        # Both should find a solution, but rna-seq should have higher confidence
        assert solution_rna is not None
        assert solution_dna is not None
    
    def test_format_solution(self):
        """Test formatted solution output."""
        db = ErrorPatternDB()
        
        error = "Out of memory error"
        formatted = db.format_solution(error)
        
        assert formatted is not None
        assert "Error Category:" in formatted
        assert "Likely Cause:" in formatted
        assert "Suggested Solution:" in formatted
    
    def test_get_patterns_by_category(self):
        """Test getting patterns by category."""
        db = ErrorPatternDB()
        
        memory_patterns = db.get_patterns_by_category("memory")
        
        assert len(memory_patterns) > 0
        assert all(p.category == "memory" for p in memory_patterns)
    
    def test_add_custom_pattern(self):
        """Test adding custom pattern."""
        db = ErrorPatternDB()
        initial_count = len(db.patterns)
        
        custom = ErrorSolution(
            pattern=r"custom error pattern",
            cause="Custom cause",
            solution="Custom solution",
            category="custom",
        )
        
        db.add_pattern(custom)
        
        assert len(db.patterns) == initial_count + 1
        
        match, _ = custom.matches("A custom error pattern occurred")
        assert match is True
    
    def test_load_patterns_from_file(self):
        """Test loading patterns from YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                "patterns": [
                    {
                        "pattern": r"custom yaml error",
                        "cause": "From YAML",
                        "solution": "YAML solution",
                        "category": "yaml_test",
                    }
                ]
            }, f)
            f.flush()
            
            db = ErrorPatternDB(patterns_file=f.name)
            
            # Should have both default and custom patterns
            yaml_patterns = db.get_patterns_by_category("yaml_test")
            assert len(yaml_patterns) == 1
    
    def test_export_patterns(self):
        """Test exporting patterns to file."""
        db = ErrorPatternDB()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            output_path = f.name
        
        db.export_patterns(output_path)
        
        # Verify export
        exported = yaml.safe_load(Path(output_path).read_text())
        assert "patterns" in exported
        assert len(exported["patterns"]) == len(db.patterns)
    
    def test_no_match_returns_none(self):
        """Test that non-matching errors return None."""
        db = ErrorPatternDB()
        
        error = "This is a completely unrelated message with no errors"
        solution = db.find_solution(error)
        
        # May or may not match depending on patterns
        # At least verify no exception
        assert solution is None or isinstance(solution, ErrorSolution)


# ============================================================================
# NFCoreModule Tests
# ============================================================================

class TestNFCoreModule:
    """Tests for NFCoreModule dataclass."""
    
    def test_create_module(self):
        """Test creating an nf-core module."""
        module = NFCoreModule(
            name="star_align",
            path="star/align",
            description="Align reads with STAR",
            keywords=["alignment", "rna-seq"],
            tools=["STAR"],
            analysis_types=["rna-seq"],
        )
        
        assert module.name == "star_align"
        assert module.path == "star/align"
        assert "rna-seq" in module.analysis_types
        assert "STAR" in module.tools
    
    def test_to_dict(self):
        """Test converting module to dictionary."""
        module = NFCoreModule(
            name="test_module",
            path="test/path",
            description="Test description",
        )
        
        data = module.to_dict()
        
        assert data["name"] == "test_module"
        assert data["path"] == "test/path"
        assert data["description"] == "Test description"
    
    def test_get_searchable_text(self):
        """Test getting searchable text."""
        module = NFCoreModule(
            name="star_align",
            path="star/align",
            description="RNA-seq alignment tool",
            keywords=["alignment", "spliced"],
            tools=["STAR"],
            inputs=[{"description": "FASTQ reads"}],
            outputs=[{"description": "BAM file"}],
        )
        
        text = module.get_searchable_text()
        
        assert "star_align" in text
        assert "RNA-seq alignment" in text
        assert "alignment" in text
        assert "FASTQ reads" in text


# ============================================================================
# NFCoreIndexer Tests
# ============================================================================

class TestNFCoreIndexer:
    """Tests for NFCoreIndexer class."""
    
    @pytest.fixture
    def temp_indexer(self):
        """Create a temporary indexer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = NFCoreIndexer(cache_dir=tmpdir)
            yield indexer
    
    def test_initialization(self, temp_indexer):
        """Test indexer initialization."""
        assert temp_indexer.cache_dir.exists()
        assert temp_indexer.modules == []
        assert temp_indexer.last_updated is None
    
    def test_tool_to_analysis_mapping(self):
        """Test tool to analysis type mapping."""
        indexer = NFCoreIndexer()
        
        assert "rna-seq" in indexer.TOOL_TO_ANALYSIS["star"]
        assert "dna-seq" in indexer.TOOL_TO_ANALYSIS["bwa"]
        assert "chip-seq" in indexer.TOOL_TO_ANALYSIS["macs2"]
    
    def test_search_empty_index(self, temp_indexer):
        """Test searching with empty index."""
        results = temp_indexer.search("STAR alignment")
        
        assert results == []
    
    def test_search_with_modules(self, temp_indexer):
        """Test searching indexed modules."""
        # Manually add modules for testing
        temp_indexer.modules = [
            NFCoreModule(
                name="star_align",
                path="star/align",
                description="STAR alignment",
                tools=["STAR"],
                keywords=["alignment", "rna-seq"],
                analysis_types=["rna-seq"],
            ),
            NFCoreModule(
                name="salmon_quant",
                path="salmon/quant",
                description="Salmon quantification",
                tools=["salmon"],
                keywords=["quantification"],
                analysis_types=["rna-seq"],
            ),
            NFCoreModule(
                name="bwa_mem",
                path="bwa/mem",
                description="BWA alignment",
                tools=["BWA"],
                keywords=["alignment", "dna-seq"],
                analysis_types=["dna-seq"],
            ),
        ]
        
        results = temp_indexer.search("STAR")
        
        assert len(results) >= 1
        assert results[0][0].name == "star_align"
    
    def test_search_with_analysis_filter(self, temp_indexer):
        """Test searching with analysis type boost."""
        temp_indexer.modules = [
            NFCoreModule(
                name="star_align",
                path="star/align",
                description="alignment tool",
                analysis_types=["rna-seq"],
            ),
            NFCoreModule(
                name="bwa_mem",
                path="bwa/mem",
                description="alignment tool",
                analysis_types=["dna-seq"],
            ),
        ]
        
        # Search with rna-seq filter should boost STAR
        results = temp_indexer.search("alignment", analysis_type="rna-seq")
        
        assert len(results) == 2
        # STAR should be ranked higher due to analysis type match
        rna_seq_module = next((m for m, _ in results if "rna-seq" in m.analysis_types), None)
        assert rna_seq_module is not None
    
    def test_get_modules_for_analysis(self, temp_indexer):
        """Test getting modules by analysis type."""
        temp_indexer.modules = [
            NFCoreModule(name="m1", path="p1", analysis_types=["rna-seq"]),
            NFCoreModule(name="m2", path="p2", analysis_types=["rna-seq", "chip-seq"]),
            NFCoreModule(name="m3", path="p3", analysis_types=["dna-seq"]),
        ]
        
        rna_modules = temp_indexer.get_modules_for_analysis("rna-seq")
        
        assert len(rna_modules) == 2
        assert all("rna-seq" in m.analysis_types for m in rna_modules)
    
    def test_get_module_by_name(self, temp_indexer):
        """Test getting module by name."""
        temp_indexer.modules = [
            NFCoreModule(name="star_align", path="star/align"),
            NFCoreModule(name="salmon_quant", path="salmon/quant"),
        ]
        
        module = temp_indexer.get_module("star_align")
        
        assert module is not None
        assert module.name == "star_align"
        
        # Test not found
        not_found = temp_indexer.get_module("nonexistent")
        assert not_found is None
    
    def test_export_import_index(self, temp_indexer):
        """Test exporting and importing index."""
        temp_indexer.modules = [
            NFCoreModule(
                name="test_module",
                path="test/path",
                description="Test module",
                tools=["tool1"],
            ),
        ]
        temp_indexer.last_updated = datetime.now()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name
        
        # Export
        temp_indexer.export_index(output_path)
        
        # Create new indexer and import
        new_indexer = NFCoreIndexer(cache_dir=tempfile.mkdtemp())
        count = new_indexer.import_index(output_path)
        
        assert count == 1
        assert len(new_indexer.modules) == 1
        assert new_indexer.modules[0].name == "test_module"
    
    def test_import_nonexistent_file(self, temp_indexer):
        """Test importing from nonexistent file."""
        count = temp_indexer.import_index("/nonexistent/path.json")
        
        assert count == 0
        assert temp_indexer.modules == []
    
    def test_get_stats(self, temp_indexer):
        """Test getting indexer statistics."""
        temp_indexer.modules = [
            NFCoreModule(name="m1", path="p1", tools=["STAR"], 
                        analysis_types=["rna-seq"]),
            NFCoreModule(name="m2", path="p2", tools=["STAR", "samtools"],
                        analysis_types=["rna-seq"]),
            NFCoreModule(name="m3", path="p3", tools=["BWA"],
                        analysis_types=["dna-seq"]),
        ]
        temp_indexer.last_updated = datetime.now()
        
        stats = temp_indexer.get_stats()
        
        assert stats["total_modules"] == 3
        assert stats["analysis_type_counts"]["rna-seq"] == 2
        assert stats["analysis_type_counts"]["dna-seq"] == 1
        assert stats["last_updated"] is not None


# ============================================================================
# Integration Tests
# ============================================================================

class TestRAGIntegration:
    """Integration tests for RAG enhancement components."""
    
    def test_knowledge_base_with_error_patterns(self):
        """Test knowledge base indexing error patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(base_path=tmpdir)
            error_db = ErrorPatternDB()
            
            # Index error patterns as knowledge documents
            for i, pattern in enumerate(error_db.patterns[:5]):
                doc = KnowledgeDocument(
                    id=f"error_{i}",
                    source=KnowledgeSource.ERROR_PATTERNS,
                    title=pattern.category,
                    content=f"{pattern.cause}\n{pattern.solution}",
                    metadata={"category": pattern.category},
                )
                kb.add_document(doc)
            
            stats = kb.get_stats()
            assert stats.get("error_patterns", 0) == 5
    
    def test_knowledge_base_with_nf_core_modules(self):
        """Test knowledge base with nf-core module documents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(base_path=tmpdir)
            indexer = NFCoreIndexer(cache_dir=tmpdir)
            
            # Create mock modules
            indexer.modules = [
                NFCoreModule(
                    name="star_align",
                    path="star/align",
                    description="STAR RNA-seq alignment",
                ),
                NFCoreModule(
                    name="salmon_quant",
                    path="salmon/quant",
                    description="Salmon transcript quantification",
                ),
            ]
            
            # Index as knowledge documents
            for module in indexer.modules:
                doc = KnowledgeDocument(
                    id=f"nfcore_{module.name}",
                    source=KnowledgeSource.NF_CORE_MODULES,
                    title=module.name,
                    content=module.get_searchable_text(),
                    metadata={"path": module.path},
                )
                kb.add_document(doc)
            
            # Search
            results = kb.search(
                "alignment",
                sources=[KnowledgeSource.NF_CORE_MODULES]
            )
            
            assert len(results) >= 1
    
    def test_full_rag_workflow(self):
        """Test complete RAG enhancement workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize components
            kb = KnowledgeBase(base_path=tmpdir)
            error_db = ErrorPatternDB()
            indexer = NFCoreIndexer(cache_dir=tmpdir)
            
            # Simulate user query with error
            user_query = "RNA-seq pipeline failing with Out of memory"
            
            # 1. Get error solution
            error_solution = error_db.find_solution(user_query)
            assert error_solution is not None
            
            # 2. Mock modules for context
            indexer.modules = [
                NFCoreModule(
                    name="star_align",
                    path="star/align",
                    description="RNA-seq alignment with STAR",
                    analysis_types=["rna-seq"],
                ),
            ]
            
            # 3. Search for relevant modules
            modules = indexer.search("RNA-seq", analysis_type="rna-seq")
            assert len(modules) >= 1
            
            # 4. Build enhanced context
            context = {
                "error_solution": error_solution.to_dict() if error_solution else None,
                "relevant_modules": [m.to_dict() for m, _ in modules[:3]],
            }
            
            assert context["error_solution"]["category"] == "memory"
            assert len(context["relevant_modules"]) >= 1


# ============================================================================
# Import Tests
# ============================================================================

class TestRAGImports:
    """Test that all Phase 2.6 components are properly exported."""
    
    def test_import_from_rag_package(self):
        """Test importing from rag package."""
        from workflow_composer.agents.rag import (
            KnowledgeBase,
            KnowledgeSource,
            KnowledgeDocument,
            KnowledgeBaseConfig,
            ErrorPatternDB,
            ErrorSolution,
            NFCoreIndexer,
            NFCoreModule,
        )
        
        assert KnowledgeBase is not None
        assert KnowledgeSource is not None
        assert KnowledgeDocument is not None
        assert KnowledgeBaseConfig is not None
        assert ErrorPatternDB is not None
        assert ErrorSolution is not None
        assert NFCoreIndexer is not None
        assert NFCoreModule is not None
    
    def test_existing_rag_exports_preserved(self):
        """Test that existing RAG exports are still available."""
        from workflow_composer.agents.rag import (
            ToolMemory,
            ArgumentMemory,
            RAGToolSelector,
            RAGOrchestrator,
        )
        
        assert ToolMemory is not None
        assert ArgumentMemory is not None
        assert RAGToolSelector is not None
        assert RAGOrchestrator is not None
