"""Tests for Phase 7: Evaluation Framework."""

import pytest
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from workflow_composer.evaluation.benchmarks import (
    Benchmark,
    BenchmarkCategory,
    BenchmarkQuery,
    ExpectedBehavior,
    load_benchmarks,
    DISCOVERY_BENCHMARKS,
    WORKFLOW_BENCHMARKS,
    EDUCATION_BENCHMARKS,
)
from workflow_composer.evaluation.evaluator import (
    Evaluator,
    EvaluatorConfig,
    EvaluationResult,
    EvaluationRun,
    EvaluationStatus,
    ToolCall,
)
from workflow_composer.evaluation.scorer import (
    Score,
    ScorerConfig,
    RuleBasedScorer,
    LLMJudgeScorer,
    CompositeScorer,
    create_scorer,
)
from workflow_composer.evaluation.metrics import (
    MetricStats,
    CategoryMetrics,
    EvaluationMetrics,
    MetricAggregator,
)
from workflow_composer.evaluation.report import (
    ReportConfig,
    JSONReportGenerator,
    HTMLReportGenerator,
    MarkdownReportGenerator,
    create_report_generator,
)


class TestBenchmarkQuery:
    """Tests for BenchmarkQuery."""
    
    def test_query_creation(self):
        """Test creating a benchmark query."""
        query = BenchmarkQuery(
            id="test_001",
            query="Find RNA-seq data",
            category=BenchmarkCategory.DATA_DISCOVERY,
            expected_behavior=ExpectedBehavior.TOOL_CALL,
            expected_tools=["search_datasets"],
            expected_keywords=["RNA-seq", "data"],
            difficulty=2,
        )
        
        assert query.id == "test_001"
        assert query.query == "Find RNA-seq data"
        assert query.category == BenchmarkCategory.DATA_DISCOVERY
        assert len(query.expected_tools) == 1
    
    def test_query_to_dict(self):
        """Test converting query to dict."""
        query = BenchmarkQuery(
            id="test_001",
            query="Test query",
            category=BenchmarkCategory.DATA_DISCOVERY,
            expected_behavior=ExpectedBehavior.TOOL_CALL,
        )
        
        data = query.to_dict()
        
        assert data["id"] == "test_001"
        assert data["category"] == "data_discovery"
        assert data["expected_behavior"] == "tool_call"
    
    def test_query_from_dict(self):
        """Test creating query from dict."""
        data = {
            "id": "test_001",
            "query": "Test query",
            "category": "data_discovery",
            "expected_behavior": "tool_call",
            "expected_tools": ["tool1"],
            "difficulty": 3,
        }
        
        query = BenchmarkQuery.from_dict(data)
        
        assert query.id == "test_001"
        assert query.category == BenchmarkCategory.DATA_DISCOVERY
        assert query.difficulty == 3


class TestBenchmark:
    """Tests for Benchmark."""
    
    def test_benchmark_creation(self):
        """Test creating a benchmark."""
        queries = [
            BenchmarkQuery(
                id=f"test_{i}",
                query=f"Query {i}",
                category=BenchmarkCategory.DATA_DISCOVERY,
                expected_behavior=ExpectedBehavior.TOOL_CALL,
                difficulty=i,
            )
            for i in range(1, 4)
        ]
        
        benchmark = Benchmark(
            name="Test Benchmark",
            description="A test benchmark",
            version="1.0.0",
            queries=queries,
        )
        
        assert benchmark.name == "Test Benchmark"
        assert len(benchmark.queries) == 3
    
    def test_filter_by_category(self):
        """Test filtering by category."""
        queries = [
            BenchmarkQuery(
                id="disc_001",
                query="Search data",
                category=BenchmarkCategory.DATA_DISCOVERY,
                expected_behavior=ExpectedBehavior.TOOL_CALL,
            ),
            BenchmarkQuery(
                id="wf_001",
                query="Create workflow",
                category=BenchmarkCategory.WORKFLOW_GENERATION,
                expected_behavior=ExpectedBehavior.TOOL_CALL,
            ),
        ]
        
        benchmark = Benchmark(
            name="Test",
            description="Test",
            version="1.0.0",
            queries=queries,
        )
        
        discovery = benchmark.filter_by_category(BenchmarkCategory.DATA_DISCOVERY)
        
        assert len(discovery) == 1
        assert discovery[0].id == "disc_001"
    
    def test_filter_by_difficulty(self):
        """Test filtering by difficulty."""
        queries = [
            BenchmarkQuery(
                id=f"test_{i}",
                query=f"Query {i}",
                category=BenchmarkCategory.DATA_DISCOVERY,
                expected_behavior=ExpectedBehavior.TOOL_CALL,
                difficulty=i,
            )
            for i in range(1, 6)
        ]
        
        benchmark = Benchmark(
            name="Test",
            description="Test",
            version="1.0.0",
            queries=queries,
        )
        
        easy = benchmark.filter_by_difficulty(2)
        
        assert len(easy) == 2
        assert all(q.difficulty <= 2 for q in easy)
    
    def test_save_and_load(self):
        """Test saving and loading benchmark."""
        queries = [
            BenchmarkQuery(
                id="test_001",
                query="Test query",
                category=BenchmarkCategory.DATA_DISCOVERY,
                expected_behavior=ExpectedBehavior.TOOL_CALL,
            ),
        ]
        
        benchmark = Benchmark(
            name="Test",
            description="Test",
            version="1.0.0",
            queries=queries,
        )
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        
        try:
            benchmark.save(path)
            loaded = Benchmark.load(path)
            
            assert loaded.name == "Test"
            assert len(loaded.queries) == 1
        finally:
            Path(path).unlink(missing_ok=True)


class TestLoadBenchmarks:
    """Tests for load_benchmarks function."""
    
    def test_load_all_benchmarks(self):
        """Test loading all benchmarks."""
        benchmark = load_benchmarks()
        
        assert len(benchmark.queries) > 0
        assert benchmark.name == "BioPipelines Standard Benchmark"
    
    def test_load_filtered_benchmarks(self):
        """Test loading filtered benchmarks."""
        benchmark = load_benchmarks(
            categories=[BenchmarkCategory.DATA_DISCOVERY],
            max_difficulty=2,
        )
        
        for query in benchmark.queries:
            assert query.category == BenchmarkCategory.DATA_DISCOVERY
            assert query.difficulty <= 2
    
    def test_builtin_benchmarks_exist(self):
        """Test that builtin benchmarks are defined."""
        assert len(DISCOVERY_BENCHMARKS) > 0
        assert len(WORKFLOW_BENCHMARKS) > 0
        assert len(EDUCATION_BENCHMARKS) > 0


class TestEvaluatorConfig:
    """Tests for EvaluatorConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = EvaluatorConfig()
        
        assert config.query_timeout == 60.0
        assert config.max_concurrent == 5
        assert config.continue_on_failure is True


class TestToolCall:
    """Tests for ToolCall."""
    
    def test_tool_call_creation(self):
        """Test creating a tool call record."""
        call = ToolCall(
            tool_name="search_datasets",
            arguments={"query": "RNA-seq"},
            result={"datasets": []},
            duration_ms=100.0,
            success=True,
        )
        
        assert call.tool_name == "search_datasets"
        assert call.success is True


class TestEvaluationResult:
    """Tests for EvaluationResult."""
    
    def test_result_creation(self):
        """Test creating an evaluation result."""
        result = EvaluationResult(
            query_id="test_001",
            query_text="Find RNA-seq data",
            status=EvaluationStatus.COMPLETED,
            response="Found 10 datasets",
            latency_ms=150.0,
        )
        
        assert result.query_id == "test_001"
        assert result.is_successful is True
    
    def test_tools_called(self):
        """Test getting list of tools called."""
        result = EvaluationResult(
            query_id="test_001",
            query_text="Test",
            status=EvaluationStatus.COMPLETED,
            tool_calls=[
                ToolCall("tool1", {}, None, 50, True),
                ToolCall("tool2", {}, None, 50, True),
            ],
        )
        
        assert result.tools_called == ["tool1", "tool2"]
    
    def test_overall_score(self):
        """Test overall score calculation."""
        result = EvaluationResult(
            query_id="test_001",
            query_text="Test",
            status=EvaluationStatus.COMPLETED,
            scores={"accuracy": 0.8, "relevance": 0.9},
        )
        
        assert abs(result.overall_score - 0.85) < 0.001
    
    def test_to_dict(self):
        """Test converting to dict."""
        result = EvaluationResult(
            query_id="test_001",
            query_text="Test",
            status=EvaluationStatus.COMPLETED,
        )
        
        data = result.to_dict()
        
        assert data["query_id"] == "test_001"
        assert data["status"] == "completed"


class TestEvaluationRun:
    """Tests for EvaluationRun."""
    
    def test_run_creation(self):
        """Test creating an evaluation run."""
        run = EvaluationRun(
            run_id="run_001",
            benchmark_name="Test Benchmark",
            config=EvaluatorConfig(),
        )
        
        assert run.run_id == "run_001"
        assert len(run.results) == 0
    
    def test_success_rate(self):
        """Test success rate calculation."""
        run = EvaluationRun(
            run_id="run_001",
            benchmark_name="Test",
            config=EvaluatorConfig(),
            results=[
                EvaluationResult("q1", "Q1", EvaluationStatus.COMPLETED),
                EvaluationResult("q2", "Q2", EvaluationStatus.COMPLETED),
                EvaluationResult("q3", "Q3", EvaluationStatus.FAILED),
            ],
        )
        
        assert run.success_rate == pytest.approx(66.67, rel=0.1)
    
    def test_summary(self):
        """Test generating summary."""
        run = EvaluationRun(
            run_id="run_001",
            benchmark_name="Test",
            config=EvaluatorConfig(),
        )
        
        summary = run.summary()
        
        assert "run_001" in summary
        assert "Test" in summary


class TestEvaluator:
    """Tests for Evaluator."""
    
    @pytest.mark.asyncio
    async def test_run_benchmarks(self):
        """Test running benchmarks."""
        benchmark = Benchmark(
            name="Test",
            description="Test",
            version="1.0.0",
            queries=[
                BenchmarkQuery(
                    id="test_001",
                    query="Find data",
                    category=BenchmarkCategory.DATA_DISCOVERY,
                    expected_behavior=ExpectedBehavior.TOOL_CALL,
                    expected_tools=["search_datasets"],
                ),
            ],
        )
        
        evaluator = Evaluator()
        run = await evaluator.run_benchmarks(benchmark)
        
        assert run.run_id is not None
        assert len(run.results) == 1
    
    @pytest.mark.asyncio
    async def test_evaluate_single(self):
        """Test evaluating single query."""
        evaluator = Evaluator()
        
        result = await evaluator.evaluate_single(
            query_text="Find RNA-seq data",
            expected_tools=["search_datasets"],
        )
        
        assert result.query_id == "adhoc"
        assert result.status is not None


class TestRuleBasedScorer:
    """Tests for RuleBasedScorer."""
    
    @pytest.mark.asyncio
    async def test_score_response(self):
        """Test scoring a response."""
        scorer = RuleBasedScorer()
        
        scores = await scorer.score(
            query="Find RNA-seq data",
            response="I found several RNA-seq datasets for your query.",
        )
        
        assert "relevance" in scores
        assert "length" in scores
        assert all(0 <= s <= 1 for s in scores.values())
    
    @pytest.mark.asyncio
    async def test_score_with_ground_truth(self):
        """Test scoring with ground truth."""
        scorer = RuleBasedScorer()
        
        scores = await scorer.score(
            query="What is RNA-seq?",
            response="RNA-seq is a sequencing technique used for transcriptomics.",
            ground_truth="RNA-seq is a technique for transcriptome analysis.",
        )
        
        assert "accuracy" in scores


class TestMetricStats:
    """Tests for MetricStats."""
    
    def test_from_values(self):
        """Test creating stats from values."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = MetricStats.from_values("test_metric", values)
        
        assert stats.count == 5
        assert stats.mean == 3.0
        assert stats.median == 3.0
        assert stats.min_value == 1.0
        assert stats.max_value == 5.0
    
    def test_from_empty_values(self):
        """Test creating stats from empty values."""
        stats = MetricStats.from_values("test_metric", [])
        
        assert stats.count == 0
        assert stats.mean == 0.0


class TestMetricAggregator:
    """Tests for MetricAggregator."""
    
    def test_aggregate(self):
        """Test aggregating metrics from a run."""
        run = EvaluationRun(
            run_id="run_001",
            benchmark_name="Test",
            config=EvaluatorConfig(),
            results=[
                EvaluationResult(
                    "q1", "Q1", EvaluationStatus.COMPLETED,
                    latency_ms=100, scores={"accuracy": 0.8}
                ),
                EvaluationResult(
                    "q2", "Q2", EvaluationStatus.COMPLETED,
                    latency_ms=150, scores={"accuracy": 0.9}
                ),
            ],
        )
        run.end_time = datetime.now()
        
        aggregator = MetricAggregator()
        metrics = aggregator.aggregate(run)
        
        assert metrics.total_queries == 2
        assert metrics.successful == 2
        assert metrics.success_rate == 100.0


class TestReportGenerators:
    """Tests for report generators."""
    
    def _create_test_run(self):
        """Create a test run for report generation."""
        return EvaluationRun(
            run_id="test_run",
            benchmark_name="Test Benchmark",
            config=EvaluatorConfig(),
            results=[
                EvaluationResult(
                    "q1", "Test query 1", EvaluationStatus.COMPLETED,
                    latency_ms=100, scores={"accuracy": 0.8},
                    metadata={"category": "data_discovery"}
                ),
            ],
            end_time=datetime.now(),
        )
    
    def _create_test_metrics(self, run):
        """Create test metrics."""
        aggregator = MetricAggregator()
        return aggregator.aggregate(run)
    
    def test_json_report(self):
        """Test JSON report generation."""
        run = self._create_test_run()
        metrics = self._create_test_metrics(run)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ReportConfig(output_dir=tmpdir)
            generator = JSONReportGenerator(config)
            
            path = generator.generate(run, metrics)
            
            assert Path(path).exists()
            with open(path) as f:
                data = json.load(f)
            assert "run" in data
            assert "metrics" in data
    
    def test_html_report(self):
        """Test HTML report generation."""
        run = self._create_test_run()
        metrics = self._create_test_metrics(run)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ReportConfig(output_dir=tmpdir)
            generator = HTMLReportGenerator(config)
            
            path = generator.generate(run, metrics)
            
            assert Path(path).exists()
            content = Path(path).read_text()
            assert "<html" in content
            assert "Test Benchmark" in content
    
    def test_markdown_report(self):
        """Test Markdown report generation."""
        run = self._create_test_run()
        metrics = self._create_test_metrics(run)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ReportConfig(output_dir=tmpdir)
            generator = MarkdownReportGenerator(config)
            
            path = generator.generate(run, metrics)
            
            assert Path(path).exists()
            content = Path(path).read_text()
            assert "# " in content
            assert "Test Benchmark" in content
    
    def test_create_report_generator(self):
        """Test create_report_generator factory."""
        html = create_report_generator("html")
        json_gen = create_report_generator("json")
        md = create_report_generator("markdown")
        
        assert isinstance(html, HTMLReportGenerator)
        assert isinstance(json_gen, JSONReportGenerator)
        assert isinstance(md, MarkdownReportGenerator)
        
        with pytest.raises(ValueError):
            create_report_generator("unknown")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
