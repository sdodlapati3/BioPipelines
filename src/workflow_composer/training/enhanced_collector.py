"""
Enhanced Agent Data Collector
=============================

Captures comprehensive trace data from the UnifiedAgent including:
- Intent parsing details (method, confidence, LLM usage)
- Tool selection and execution traces
- LLM provider routing info
- Error traces and failure analysis
- Timing breakdowns for each stage

This data enables:
- Training data generation with rich metadata
- System performance analysis
- Intent parser accuracy measurement
- Tool selection effectiveness tracking
- Error pattern identification
"""

import asyncio
import json
import logging
import os
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _serialize_value(val: Any) -> Any:
    """Recursively convert non-JSON-serializable types."""
    if isinstance(val, Path):
        return str(val)
    elif isinstance(val, dict):
        return {k: _serialize_value(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [_serialize_value(v) for v in val]
    elif hasattr(val, '__dict__') and not isinstance(val, type):
        # For objects with __dict__ that aren't classes, convert to string
        try:
            return str(val)
        except:
            return "<non-serializable>"
    return val
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Set environment variable before any agent imports
os.environ.setdefault('BIOPIPELINES_LLM_ENABLED', 'false')

# Use absolute imports to ensure we use the same module as the agent
# (avoids src.workflow_composer vs workflow_composer module duplication)
from workflow_composer.agents.unified_agent import UnifiedAgent, AgentResponse, ResponseType
from workflow_composer.infrastructure.observability import (
    get_tracer, 
    InMemoryExporter,
    Span,
    SpanStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENHANCED DATA STRUCTURES
# =============================================================================

@dataclass
class IntentParseTrace:
    """Detailed trace of intent parsing."""
    
    intent: str = ""
    confidence: float = 0.0
    method: str = ""  # pattern, semantic, llm_arbiter, unanimous
    llm_invoked: bool = False
    llm_provider: str = ""  # which LLM was used (if any)
    llm_latency_ms: float = 0.0
    entities: List[Dict[str, Any]] = field(default_factory=list)
    slots: Dict[str, Any] = field(default_factory=dict)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    needs_clarification: bool = False
    fallback_used: bool = False  # Did we fall back to hybrid parser?
    
    def to_dict(self) -> Dict[str, Any]:
        return _serialize_value(asdict(self))


@dataclass 
class ToolExecutionTrace:
    """Detailed trace of tool execution."""
    
    tool_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    error_type: str = ""
    duration_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return _serialize_value(asdict(self))


@dataclass
class QueryTrace:
    """Complete trace of processing a single query."""
    
    # Input
    query: str
    category: str = ""
    
    # Task Classification
    task_type: str = ""
    task_classification_ms: float = 0.0
    
    # Intent Parsing
    intent_parse: IntentParseTrace = field(default_factory=IntentParseTrace)
    intent_parse_ms: float = 0.0
    
    # Tool Selection & Execution
    tool_selected: str = ""
    tool_selection_method: str = ""  # regex, intent_mapping, llm
    tool_executions: List[ToolExecutionTrace] = field(default_factory=list)
    tool_execution_ms: float = 0.0
    
    # RAG (if used)
    rag_used: bool = False
    rag_contexts_retrieved: int = 0
    rag_latency_ms: float = 0.0
    
    # LLM Usage Summary
    total_llm_calls: int = 0
    llm_providers_used: List[str] = field(default_factory=list)
    total_llm_latency_ms: float = 0.0
    
    # Response
    response: str = ""
    response_type: str = ""
    success: bool = False
    suggestions: List[str] = field(default_factory=list)
    
    # Errors
    error_occurred: bool = False
    error_message: str = ""
    error_type: str = ""
    error_traceback: str = ""
    
    # Timing
    total_duration_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Raw spans for debugging
    spans: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        result = _serialize_value(asdict(self))
        result['intent_parse'] = self.intent_parse.to_dict()
        result['tool_executions'] = [t.to_dict() for t in self.tool_executions]
        result['spans'] = _serialize_value(self.spans)
        return result
    
    def to_training_example(self) -> Dict[str, Any]:
        """Convert to training format with metadata."""
        return {
            "instruction": self.query,
            "input": "",
            "output": self.response,
            "metadata": {
                "category": self.category,
                "task_type": self.task_type,
                "intent": self.intent_parse.intent,
                "intent_confidence": self.intent_parse.confidence,
                "intent_method": self.intent_parse.method,
                "llm_invoked": self.intent_parse.llm_invoked,
                "tool_selected": self.tool_selected,
                "tools_executed": [t.tool_name for t in self.tool_executions],
                "success": self.success,
                "total_duration_ms": self.total_duration_ms,
                "error": self.error_message if self.error_occurred else None,
            }
        }
    
    def to_analysis_record(self) -> Dict[str, Any]:
        """Convert to format optimized for analysis."""
        return {
            "query": self.query,
            "category": self.category,
            "success": self.success,
            
            # Intent parsing metrics
            "intent": self.intent_parse.intent,
            "intent_confidence": self.intent_parse.confidence,
            "intent_method": self.intent_parse.method,
            "intent_llm_used": self.intent_parse.llm_invoked,
            "intent_fallback": self.intent_parse.fallback_used,
            
            # Tool metrics
            "tool_selected": self.tool_selected,
            "tool_success": all(t.success for t in self.tool_executions) if self.tool_executions else None,
            "num_tools_executed": len(self.tool_executions),
            
            # LLM metrics
            "llm_calls": self.total_llm_calls,
            "llm_providers": self.llm_providers_used,
            "llm_latency_ms": self.total_llm_latency_ms,
            
            # Performance
            "total_ms": self.total_duration_ms,
            "parse_ms": self.intent_parse_ms,
            "tool_ms": self.tool_execution_ms,
            
            # Error tracking
            "error": self.error_occurred,
            "error_type": self.error_type if self.error_occurred else None,
            "error_message": self.error_message if self.error_occurred else None,
            
            "timestamp": self.timestamp,
        }


# =============================================================================
# ENHANCED COLLECTOR
# =============================================================================

class EnhancedAgentCollector:
    """
    Collects comprehensive trace data from the UnifiedAgent.
    
    Unlike the basic AgentDataCollector, this captures:
    - Full intent parsing traces (method, confidence, LLM usage)
    - Tool execution details
    - LLM routing information
    - Error traces
    - Timing breakdowns
    """
    
    def __init__(
        self,
        output_dir: Path = None,
        capture_spans: bool = True,
    ):
        self.output_dir = output_dir or Path("training_data/traces")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.capture_spans = capture_spans
        self.agent: Optional[UnifiedAgent] = None
        self.span_exporter: Optional[InMemoryExporter] = None
        
        self.traces: List[QueryTrace] = []
        
        # Statistics
        self.stats = {
            "total_queries": 0,
            "successful": 0,
            "failed": 0,
            "by_intent_method": {},
            "by_tool": {},
            "llm_invocations": 0,
            "avg_latency_ms": 0.0,
            "errors_by_type": {},
        }
    
    async def initialize(self):
        """Initialize the agent with span capture."""
        if self.agent is None:
            logger.info("Initializing UnifiedAgent with enhanced tracing...")
            
            # Set up in-memory span exporter
            if self.capture_spans:
                self.span_exporter = InMemoryExporter(max_spans=1000)
                tracer = get_tracer()
                tracer.add_exporter(self.span_exporter)
            
            self.agent = UnifiedAgent()
            logger.info("Agent initialized with enhanced tracing")
    
    def _extract_spans(self) -> List[Dict[str, Any]]:
        """Extract and clear captured spans."""
        if not self.span_exporter:
            return []
        
        spans = []
        for span in self.span_exporter.get_spans():
            spans.append(span.to_dict())
        
        self.span_exporter.clear()
        return spans
    
    def _parse_spans_for_trace(
        self, 
        spans: List[Dict[str, Any]], 
        trace: QueryTrace
    ):
        """Parse span data to populate trace fields."""
        for span in spans:
            name = span.get("name", "")
            tags = span.get("tags", {})
            duration = span.get("duration_ms", 0)
            
            if name == "classify_task":
                trace.task_type = tags.get("task_type", "")
                trace.task_classification_ms = duration
            
            elif name == "intent_parse":
                trace.intent_parse.intent = tags.get("intent", "")
                trace.intent_parse.confidence = tags.get("confidence", 0.0)
                trace.intent_parse.method = tags.get("method", "")
                trace.intent_parse.llm_invoked = tags.get("llm_invoked", False)
                trace.intent_parse.needs_clarification = tags.get("needs_clarification", False)
                trace.intent_parse_ms = duration
            
            elif name == "hybrid_parse":
                trace.intent_parse.fallback_used = True
                if not trace.intent_parse.intent:
                    trace.intent_parse.intent = tags.get("intent", "")
                    trace.intent_parse.confidence = tags.get("confidence", 0.0)
            
            elif name == "rag_enhance":
                trace.rag_used = True
                trace.rag_contexts_retrieved = tags.get("context_count", 0)
                trace.rag_latency_ms = duration
            
            elif name.startswith("tool_"):
                tool_exec = ToolExecutionTrace(
                    tool_name=tags.get("tool_name", name),
                    success=span.get("status") == "ok",
                    duration_ms=duration,
                )
                trace.tool_executions.append(tool_exec)
                trace.tool_execution_ms += duration
    
    async def run_query(
        self, 
        query: str, 
        category: str = ""
    ) -> QueryTrace:
        """
        Run a query and capture comprehensive trace.
        
        Args:
            query: The query to run
            category: Category label
            
        Returns:
            QueryTrace with full execution details
        """
        await self.initialize()
        
        trace = QueryTrace(query=query, category=category)
        start_time = datetime.now()
        
        # Clear any previous spans
        if self.span_exporter:
            self.span_exporter.clear()
        
        try:
            # Run through agent
            response = await self.agent.process_query(query)
            
            # Extract response data
            trace.response = response.message or ""
            trace.response_type = response.response_type.value if response.response_type else "unknown"
            trace.success = response.success
            trace.task_type = response.task_type.value if response.task_type else ""
            trace.suggestions = response.suggestions or []
            
            # Extract tool executions from response
            if response.tool_executions:
                for exec_info in response.tool_executions:
                    tool_trace = ToolExecutionTrace(
                        tool_name=exec_info.tool_name,
                        parameters=exec_info.parameters,
                        success=exec_info.result.success if exec_info.result else False,
                        duration_ms=exec_info.duration_ms,
                    )
                    if exec_info.result:
                        tool_trace.result_data = exec_info.result.data or {}
                        if not exec_info.result.success:
                            tool_trace.error_message = exec_info.result.error or ""
                    
                    trace.tool_executions.append(tool_trace)
                    
                    if not trace.tool_selected:
                        trace.tool_selected = exec_info.tool_name
            
        except Exception as e:
            trace.error_occurred = True
            trace.error_message = str(e)
            trace.error_type = type(e).__name__
            trace.error_traceback = traceback.format_exc()
            trace.success = False
            
            # Update error stats
            self.stats["errors_by_type"][type(e).__name__] = \
                self.stats["errors_by_type"].get(type(e).__name__, 0) + 1
        
        # Calculate total duration
        trace.total_duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Extract and parse spans
        if self.capture_spans:
            trace.spans = self._extract_spans()
            self._parse_spans_for_trace(trace.spans, trace)
        
        # Update statistics
        self.stats["total_queries"] += 1
        if trace.success:
            self.stats["successful"] += 1
        else:
            self.stats["failed"] += 1
        
        method = trace.intent_parse.method or "unknown"
        self.stats["by_intent_method"][method] = \
            self.stats["by_intent_method"].get(method, 0) + 1
        
        if trace.tool_selected:
            self.stats["by_tool"][trace.tool_selected] = \
                self.stats["by_tool"].get(trace.tool_selected, 0) + 1
        
        if trace.intent_parse.llm_invoked:
            self.stats["llm_invocations"] += 1
        
        # Update running average latency
        n = self.stats["total_queries"]
        self.stats["avg_latency_ms"] = (
            (self.stats["avg_latency_ms"] * (n - 1) + trace.total_duration_ms) / n
        )
        
        self.traces.append(trace)
        return trace
    
    async def run_batch(
        self,
        queries: List[Tuple[str, str]],  # (query, category) pairs
        progress_interval: int = 10,
    ) -> List[QueryTrace]:
        """Run a batch of queries."""
        logger.info(f"Running batch of {len(queries)} queries")
        
        traces = []
        for i, (query, category) in enumerate(queries):
            trace = await self.run_query(query, category)
            traces.append(trace)
            
            if (i + 1) % progress_interval == 0:
                logger.info(f"Progress: {i+1}/{len(queries)}")
        
        return traces
    
    def export_traces(self, prefix: str = "traces") -> Dict[str, Path]:
        """Export all traces in multiple formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_files = {}
        
        # Full traces (for debugging)
        full_path = self.output_dir / f"{prefix}_full_{timestamp}.jsonl"
        with open(full_path, 'w') as f:
            for trace in self.traces:
                f.write(json.dumps(trace.to_dict()) + "\n")
        output_files["full_traces"] = full_path
        
        # Training format
        training_path = self.output_dir / f"{prefix}_training_{timestamp}.jsonl"
        with open(training_path, 'w') as f:
            for trace in self.traces:
                if trace.success:  # Only successful for training
                    f.write(json.dumps(trace.to_training_example()) + "\n")
        output_files["training"] = training_path
        
        # Analysis format
        analysis_path = self.output_dir / f"{prefix}_analysis_{timestamp}.jsonl"
        with open(analysis_path, 'w') as f:
            for trace in self.traces:
                f.write(json.dumps(trace.to_analysis_record()) + "\n")
        output_files["analysis"] = analysis_path
        
        # Statistics
        stats_path = self.output_dir / f"{prefix}_stats_{timestamp}.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        output_files["stats"] = stats_path
        
        logger.info(f"Exported {len(self.traces)} traces to {self.output_dir}")
        return output_files
    
    def print_summary(self):
        """Print comprehensive summary."""
        print("\n" + "=" * 70)
        print("ENHANCED TRACE COLLECTION SUMMARY")
        print("=" * 70)
        
        print(f"\n📊 QUERIES")
        print(f"  Total:      {self.stats['total_queries']}")
        print(f"  Successful: {self.stats['successful']}")
        print(f"  Failed:     {self.stats['failed']}")
        
        if self.stats['total_queries'] > 0:
            success_rate = self.stats['successful'] / self.stats['total_queries'] * 100
            print(f"  Success Rate: {success_rate:.1f}%")
        
        print(f"\n🧠 INTENT PARSING METHODS")
        for method, count in sorted(self.stats['by_intent_method'].items(), key=lambda x: -x[1]):
            pct = count / self.stats['total_queries'] * 100 if self.stats['total_queries'] > 0 else 0
            print(f"  {method}: {count} ({pct:.1f}%)")
        
        print(f"\n🤖 LLM USAGE")
        llm_rate = self.stats['llm_invocations'] / self.stats['total_queries'] * 100 if self.stats['total_queries'] > 0 else 0
        print(f"  LLM Invocations: {self.stats['llm_invocations']} ({llm_rate:.1f}% of queries)")
        
        print(f"\n🔧 TOOLS USED")
        for tool, count in sorted(self.stats['by_tool'].items(), key=lambda x: -x[1])[:10]:
            print(f"  {tool}: {count}")
        
        if self.stats['errors_by_type']:
            print(f"\n❌ ERRORS BY TYPE")
            for error_type, count in sorted(self.stats['errors_by_type'].items(), key=lambda x: -x[1]):
                print(f"  {error_type}: {count}")
        
        print(f"\n⏱️ PERFORMANCE")
        print(f"  Avg Latency: {self.stats['avg_latency_ms']:.1f}ms")
        
        print("=" * 70)


# =============================================================================
# QUERY TEMPLATES FOR COMPREHENSIVE TESTING
# =============================================================================

COMPREHENSIVE_TEST_QUERIES = [
    # Data Discovery (should use pattern matching)
    ("scan my data folder", "data_discovery"),
    ("what files do I have?", "data_discovery"),
    ("list FASTQ files in data/raw", "data_discovery"),
    
    # Workflow Generation (pattern + slot filling)
    ("create an RNA-seq workflow", "workflow_generation"),
    ("build a ChIP-seq pipeline for H3K27ac", "workflow_generation"),
    ("generate methylation analysis for mouse", "workflow_generation"),
    
    # Education (semantic matching likely)
    ("explain what DESeq2 does", "education"),
    ("what is differential expression?", "education"),
    ("how does peak calling work?", "education"),
    
    # Job Management (pattern matching)
    ("show my jobs", "job_management"),
    ("check job status", "job_management"),
    ("list running jobs", "job_management"),
    
    # Database Search (may need LLM for complex queries)
    ("search for TP53 in databases", "database_search"),
    ("find BRCA1 expression data", "database_search"),
    
    # Ambiguous queries (should trigger LLM arbiter or clarification)
    ("analyze this data", "ambiguous"),
    ("help me with my project", "ambiguous"),
    ("what should I do next?", "ambiguous"),
    
    # Error scenarios (for testing error handling)
    ("download dataset XYZ123", "error_test"),  # Non-existent dataset
    ("run workflow on /nonexistent/path", "error_test"),
]


async def run_comprehensive_test(output_dir: Path = None) -> EnhancedAgentCollector:
    """Run comprehensive test with enhanced collection."""
    collector = EnhancedAgentCollector(output_dir=output_dir)
    
    await collector.run_batch(COMPREHENSIVE_TEST_QUERIES)
    
    output_files = collector.export_traces(prefix="comprehensive_test")
    collector.print_summary()
    
    print("\n📁 Exported Files:")
    for name, path in output_files.items():
        print(f"  {name}: {path}")
    
    return collector


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_comprehensive_test())
