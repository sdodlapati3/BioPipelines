"""Evaluator for running benchmarks against BioPipelines.

This module provides the core evaluation logic for testing the
agentic system against benchmark queries.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .benchmarks import Benchmark, BenchmarkQuery

logger = logging.getLogger(__name__)


class EvaluationStatus(Enum):
    """Status of an evaluation."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class EvaluatorConfig:
    """Configuration for the evaluator."""
    
    # Timeout for each query (seconds)
    query_timeout: float = 60.0
    
    # Maximum concurrent evaluations
    max_concurrent: int = 5
    
    # Whether to continue on failure
    continue_on_failure: bool = True
    
    # Whether to collect detailed traces
    collect_traces: bool = True
    
    # Whether to use LLM-as-judge scoring
    use_llm_judge: bool = True
    
    # LLM model for judging (if enabled)
    judge_model: str = "gpt-4"
    
    # Retry failed queries
    retry_count: int = 1
    
    # Delay between retries (seconds)
    retry_delay: float = 2.0


@dataclass
class ToolCall:
    """Record of a tool call during evaluation."""
    
    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    duration_ms: float
    success: bool
    error: Optional[str] = None


@dataclass
class EvaluationResult:
    """Result of evaluating a single query.
    
    Attributes:
        query_id: ID of the benchmark query
        query_text: The query text
        status: Evaluation status
        response: The agent's response
        tool_calls: List of tool calls made
        latency_ms: Total latency in milliseconds
        scores: Dictionary of score dimensions
        metadata: Additional metadata
    """
    
    query_id: str
    query_text: str
    status: EvaluationStatus
    response: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    latency_ms: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def is_successful(self) -> bool:
        """Check if evaluation completed successfully."""
        return self.status == EvaluationStatus.COMPLETED
    
    @property
    def tools_called(self) -> List[str]:
        """Get list of tools called."""
        return [tc.tool_name for tc in self.tool_calls]
    
    @property
    def overall_score(self) -> float:
        """Calculate overall score (average of all dimensions)."""
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "status": self.status.value,
            "response": self.response,
            "tool_calls": [
                {
                    "tool_name": tc.tool_name,
                    "arguments": tc.arguments,
                    "result": str(tc.result)[:500] if tc.result else None,
                    "duration_ms": tc.duration_ms,
                    "success": tc.success,
                    "error": tc.error,
                }
                for tc in self.tool_calls
            ],
            "latency_ms": self.latency_ms,
            "scores": self.scores,
            "overall_score": self.overall_score,
            "metadata": self.metadata,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class EvaluationRun:
    """Complete evaluation run with all results.
    
    Attributes:
        run_id: Unique identifier for this run
        benchmark_name: Name of the benchmark used
        config: Evaluator configuration
        results: List of evaluation results
        start_time: When the run started
        end_time: When the run ended
    """
    
    run_id: str
    benchmark_name: str
    config: EvaluatorConfig
    results: List[EvaluationResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    @property
    def duration_seconds(self) -> float:
        """Total duration of the run."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def success_rate(self) -> float:
        """Percentage of successful evaluations."""
        if not self.results:
            return 0.0
        successful = sum(1 for r in self.results if r.is_successful)
        return successful / len(self.results) * 100
    
    @property
    def average_latency_ms(self) -> float:
        """Average latency across all queries."""
        successful = [r for r in self.results if r.is_successful]
        if not successful:
            return 0.0
        return sum(r.latency_ms for r in successful) / len(successful)
    
    @property
    def average_score(self) -> float:
        """Average overall score."""
        scored = [r for r in self.results if r.scores]
        if not scored:
            return 0.0
        return sum(r.overall_score for r in scored) / len(scored)
    
    def get_results_by_category(self) -> Dict[str, List[EvaluationResult]]:
        """Group results by query category."""
        grouped: Dict[str, List[EvaluationResult]] = {}
        for result in self.results:
            category = result.metadata.get("category", "unknown")
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(result)
        return grouped
    
    def summary(self) -> str:
        """Generate a text summary of the run."""
        lines = [
            f"Evaluation Run: {self.run_id}",
            f"Benchmark: {self.benchmark_name}",
            f"Duration: {self.duration_seconds:.1f}s",
            f"Queries: {len(self.results)}",
            f"Success Rate: {self.success_rate:.1f}%",
            f"Average Latency: {self.average_latency_ms:.1f}ms",
            f"Average Score: {self.average_score:.2f}",
        ]
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "benchmark_name": self.benchmark_name,
            "config": {
                "query_timeout": self.config.query_timeout,
                "max_concurrent": self.config.max_concurrent,
                "use_llm_judge": self.config.use_llm_judge,
            },
            "results": [r.to_dict() for r in self.results],
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "summary": {
                "duration_seconds": self.duration_seconds,
                "success_rate": self.success_rate,
                "average_latency_ms": self.average_latency_ms,
                "average_score": self.average_score,
                "total_queries": len(self.results),
            },
        }


class Evaluator:
    """Evaluator for running benchmarks.
    
    This class coordinates the evaluation of benchmark queries against
    the BioPipelines agentic system.
    
    Example:
        >>> evaluator = Evaluator()
        >>> benchmark = load_benchmarks()
        >>> run = await evaluator.run_benchmarks(benchmark)
        >>> print(run.summary())
    """
    
    def __init__(
        self,
        config: Optional[EvaluatorConfig] = None,
        agent: Any = None,
        scorer: Any = None,
    ):
        """Initialize evaluator.
        
        Args:
            config: Evaluator configuration
            agent: Agent to evaluate (uses default if None)
            scorer: Scorer for evaluating responses
        """
        self.config = config or EvaluatorConfig()
        self._agent = agent
        self._scorer = scorer
        self._run_counter = 0
    
    async def run_benchmarks(
        self,
        benchmark: "Benchmark",
        queries: Optional[List["BenchmarkQuery"]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> EvaluationRun:
        """Run benchmark evaluation.
        
        Args:
            benchmark: Benchmark to run
            queries: Optional subset of queries (all if None)
            progress_callback: Optional callback for progress updates
            
        Returns:
            EvaluationRun with all results
        """
        self._run_counter += 1
        run_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._run_counter}"
        
        run = EvaluationRun(
            run_id=run_id,
            benchmark_name=benchmark.name,
            config=self.config,
        )
        
        queries_to_run = queries or benchmark.queries
        total = len(queries_to_run)
        
        logger.info(f"Starting evaluation run {run_id} with {total} queries")
        
        # Run evaluations
        if self.config.max_concurrent > 1:
            # Parallel execution
            semaphore = asyncio.Semaphore(self.config.max_concurrent)
            
            async def eval_with_semaphore(query: "BenchmarkQuery", idx: int):
                async with semaphore:
                    result = await self._evaluate_query(query)
                    if progress_callback:
                        progress_callback(idx + 1, total)
                    return result
            
            tasks = [
                eval_with_semaphore(q, i)
                for i, q in enumerate(queries_to_run)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Query evaluation failed: {result}")
                    if not self.config.continue_on_failure:
                        raise result
                else:
                    run.results.append(result)
        else:
            # Sequential execution
            for i, query in enumerate(queries_to_run):
                try:
                    result = await self._evaluate_query(query)
                    run.results.append(result)
                    
                    if progress_callback:
                        progress_callback(i + 1, total)
                        
                except Exception as e:
                    logger.error(f"Query {query.id} failed: {e}")
                    if not self.config.continue_on_failure:
                        raise
                    
                    # Record failure
                    run.results.append(EvaluationResult(
                        query_id=query.id,
                        query_text=query.query,
                        status=EvaluationStatus.FAILED,
                        error=str(e),
                        metadata={"category": query.category.value},
                    ))
        
        run.end_time = datetime.now()
        logger.info(f"Evaluation run {run_id} completed: {run.summary()}")
        
        return run
    
    async def _evaluate_query(
        self,
        query: "BenchmarkQuery",
    ) -> EvaluationResult:
        """Evaluate a single query.
        
        Args:
            query: The benchmark query to evaluate
            
        Returns:
            EvaluationResult
        """
        result = EvaluationResult(
            query_id=query.id,
            query_text=query.query,
            status=EvaluationStatus.RUNNING,
            metadata={
                "category": query.category.value,
                "expected_behavior": query.expected_behavior.value,
                "expected_tools": query.expected_tools,
                "difficulty": query.difficulty,
            },
        )
        
        start_time = time.perf_counter()
        
        try:
            # Execute query against agent
            response, tool_calls = await self._execute_query(query.query)
            
            result.response = response
            result.tool_calls = tool_calls
            result.latency_ms = (time.perf_counter() - start_time) * 1000
            result.status = EvaluationStatus.COMPLETED
            
            # Score the result
            await self._score_result(result, query)
            
        except asyncio.TimeoutError:
            result.status = EvaluationStatus.TIMEOUT
            result.error = f"Query timed out after {self.config.query_timeout}s"
            result.latency_ms = (time.perf_counter() - start_time) * 1000
            
        except Exception as e:
            result.status = EvaluationStatus.FAILED
            result.error = str(e)
            result.latency_ms = (time.perf_counter() - start_time) * 1000
        
        return result
    
    async def _execute_query(
        self,
        query_text: str,
    ) -> tuple[str, List[ToolCall]]:
        """Execute a query against the agent.
        
        Args:
            query_text: The query text
            
        Returns:
            Tuple of (response_text, tool_calls)
        """
        tool_calls = []
        
        if self._agent is None:
            # Mock execution for testing
            return await self._mock_execute(query_text)
        
        # Execute with timeout
        async def execute():
            # Hook into agent's tool execution to capture calls
            captured_calls = []
            
            # Call agent
            response = await self._agent.process_query(query_text)
            
            return response, captured_calls
        
        response, calls = await asyncio.wait_for(
            execute(),
            timeout=self.config.query_timeout
        )
        
        return response, calls
    
    async def _mock_execute(
        self,
        query_text: str,
    ) -> tuple[str, List[ToolCall]]:
        """Mock execution for testing without real agent."""
        # Simulate some processing time
        await asyncio.sleep(0.1)
        
        # Generate mock response based on query
        query_lower = query_text.lower()
        
        tool_calls = []
        response = "I can help you with that."
        
        if "search" in query_lower or "find" in query_lower:
            tool_calls.append(ToolCall(
                tool_name="search_datasets",
                arguments={"query": query_text},
                result={"datasets": []},
                duration_ms=50,
                success=True,
            ))
            response = "I found several relevant datasets."
            
        elif "workflow" in query_lower or "pipeline" in query_lower:
            tool_calls.append(ToolCall(
                tool_name="generate_workflow",
                arguments={"analysis_type": "rna-seq"},
                result={"workflow": "nextflow"},
                duration_ms=100,
                success=True,
            ))
            response = "I've generated a workflow for your analysis."
            
        elif "job" in query_lower or "status" in query_lower:
            tool_calls.append(ToolCall(
                tool_name="list_jobs",
                arguments={},
                result={"jobs": []},
                duration_ms=30,
                success=True,
            ))
            response = "Here are your jobs."
        
        return response, tool_calls
    
    async def _score_result(
        self,
        result: EvaluationResult,
        query: "BenchmarkQuery",
    ) -> None:
        """Score an evaluation result.
        
        Args:
            result: The result to score
            query: The benchmark query
        """
        scores = {}
        
        # Tool accuracy: Did it call the expected tools?
        if query.expected_tools:
            called = set(result.tools_called)
            expected = set(query.expected_tools)
            
            if expected:
                precision = len(called & expected) / len(called) if called else 0
                recall = len(called & expected) / len(expected)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                scores["tool_accuracy"] = f1
            else:
                scores["tool_accuracy"] = 1.0 if not called else 0.0
        
        # Keyword coverage: Are expected keywords in response?
        if query.expected_keywords and result.response:
            response_lower = result.response.lower()
            found = sum(1 for kw in query.expected_keywords if kw.lower() in response_lower)
            scores["keyword_coverage"] = found / len(query.expected_keywords)
        
        # Latency score: Faster is better (normalize to 0-1)
        # Target: < 1000ms = 1.0, > 5000ms = 0.0
        latency_score = max(0, 1 - (result.latency_ms - 1000) / 4000)
        scores["latency"] = min(1.0, max(0.0, latency_score))
        
        # Success score
        scores["success"] = 1.0 if result.status == EvaluationStatus.COMPLETED else 0.0
        
        # Use LLM judge if enabled
        if self.config.use_llm_judge and self._scorer and result.response:
            try:
                llm_scores = await self._scorer.score(
                    query=query.query,
                    response=result.response,
                    ground_truth=query.ground_truth,
                )
                scores.update(llm_scores)
            except Exception as e:
                logger.warning(f"LLM scoring failed: {e}")
        
        result.scores = scores
    
    async def evaluate_single(
        self,
        query_text: str,
        expected_tools: Optional[List[str]] = None,
        expected_keywords: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """Evaluate a single ad-hoc query.
        
        Args:
            query_text: The query to evaluate
            expected_tools: Optional expected tool calls
            expected_keywords: Optional expected keywords
            
        Returns:
            EvaluationResult
        """
        from .benchmarks import BenchmarkQuery, BenchmarkCategory, ExpectedBehavior
        
        query = BenchmarkQuery(
            id="adhoc",
            query=query_text,
            category=BenchmarkCategory.DATA_DISCOVERY,
            expected_behavior=ExpectedBehavior.TOOL_CALL if expected_tools else ExpectedBehavior.TEXT_RESPONSE,
            expected_tools=expected_tools or [],
            expected_keywords=expected_keywords or [],
        )
        
        return await self._evaluate_query(query)


__all__ = [
    "Evaluator",
    "EvaluatorConfig",
    "EvaluationResult",
    "EvaluationRun",
    "EvaluationStatus",
    "ToolCall",
]
