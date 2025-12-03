"""
Unified Training Data Pipeline
==============================

Integrates real agent data collection with synthetic conversation generation
to produce comprehensive training datasets for fine-tuning bioinformatics AI.

Usage:
    python -m src.workflow_composer.training.pipeline --real 100 --synthetic 500 --output training_data
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .agent_data_collector import AgentDataCollector, AgentInteraction
from .enhanced_collector import EnhancedAgentCollector, QueryTrace
from .conversation_runner import ConversationRunner, ConversationResult, RunnerConfig
from .conversation_generator import ConversationGenerator, GeneratedConversation

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the training data pipeline."""
    
    # Real agent collection settings
    real_single_turn: int = 100
    real_multi_turn_sessions: int = 20
    
    # Enhanced tracing (captures intent parsing, LLM usage, etc.)
    use_enhanced_collector: bool = True
    capture_traces: bool = True  # Full trace capture for analysis
    
    # Synthetic collection settings
    synthetic_conversations: int = 500
    synthetic_multi_turn_ratio: float = 0.3
    
    # Output settings
    output_dir: Path = Path("training_data")
    prefix: str = "biopipelines"
    export_formats: List[str] = field(default_factory=lambda: ["openai", "alpaca", "sharegpt"])
    
    # Quality settings
    min_response_length: int = 50
    max_response_length: int = 4000
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineStats:
    """Statistics from the pipeline run."""
    
    # Collection stats
    real_queries: int = 0
    real_successful: int = 0
    synthetic_conversations: int = 0
    synthetic_examples: int = 0
    
    # Combined stats
    total_examples: int = 0
    filtered_examples: int = 0
    
    # Enhanced trace stats
    trace_files: Dict[str, str] = field(default_factory=dict)
    intent_method_distribution: Dict[str, int] = field(default_factory=dict)
    llm_invocation_count: int = 0
    
    # By category
    category_counts: Dict[str, int] = field(default_factory=dict)
    tool_counts: Dict[str, int] = field(default_factory=dict)
    
    # Timing
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0


class TrainingDataPipeline:
    """
    Unified pipeline for collecting training data from multiple sources.
    
    Combines:
    1. Real agent interactions (AgentDataCollector or EnhancedAgentCollector)
    2. Synthetic conversations (ConversationRunner)
    
    With enhanced tracing (use_enhanced_collector=True), captures:
    - Intent parsing method, confidence, LLM usage
    - Tool selection decisions
    - Execution timing breakdowns
    - Error traces for analysis
    
    Produces training-ready datasets in multiple formats.
    """
    
    def __init__(self, config: PipelineConfig = None):
        """Initialize the pipeline."""
        self.config = config or PipelineConfig()
        
        self.real_collector: Optional[AgentDataCollector] = None
        self.enhanced_collector: Optional[EnhancedAgentCollector] = None
        self.synthetic_runner: Optional[ConversationRunner] = None
        
        self.all_examples: List[Dict[str, Any]] = []
        self.enhanced_traces: List[QueryTrace] = []
        self.stats = PipelineStats()
        
        self.system_prompt = """You are BioPipelines AI, an expert assistant for bioinformatics workflows.

You help researchers with:
- Discovering and organizing sequencing data (FASTQ, BAM, etc.)
- Generating analysis workflows (RNA-seq, ChIP-seq, ATAC-seq, methylation, etc.)
- Running and monitoring jobs on HPC clusters (SLURM)
- Troubleshooting errors and diagnosing issues
- Explaining bioinformatics concepts

Be helpful, accurate, and proactive. Provide actionable suggestions."""

    async def run(self) -> PipelineStats:
        """Run the complete pipeline."""
        logger.info("Starting Training Data Pipeline")
        self.stats.start_time = datetime.now().isoformat()
        
        try:
            # Phase 1: Real agent collection
            if self.config.real_single_turn > 0 or self.config.real_multi_turn_sessions > 0:
                if self.config.use_enhanced_collector:
                    await self._collect_with_enhanced_tracing()
                else:
                    await self._collect_real_data()
            
            # Phase 2: Synthetic conversation generation
            if self.config.synthetic_conversations > 0:
                await self._generate_synthetic_data()
            
            # Phase 3: Combine and filter
            self._combine_and_filter()
            
            # Phase 4: Export
            output_files = self._export_all()
            
            self.stats.end_time = datetime.now().isoformat()
            start = datetime.fromisoformat(self.stats.start_time)
            end = datetime.fromisoformat(self.stats.end_time)
            self.stats.duration_seconds = (end - start).total_seconds()
            
            # Save stats
            stats_file = self.config.output_dir / f"{self.config.prefix}_pipeline_stats.json"
            with open(stats_file, 'w') as f:
                stats_dict = {
                    "real_queries": self.stats.real_queries,
                    "real_successful": self.stats.real_successful,
                    "synthetic_conversations": self.stats.synthetic_conversations,
                    "synthetic_examples": self.stats.synthetic_examples,
                    "total_examples": self.stats.total_examples,
                    "filtered_examples": self.stats.filtered_examples,
                    "category_counts": self.stats.category_counts,
                    "tool_counts": self.stats.tool_counts,
                    "start_time": self.stats.start_time,
                    "end_time": self.stats.end_time,
                    "duration_seconds": self.stats.duration_seconds,
                }
                
                # Add enhanced trace info if available
                if self.config.use_enhanced_collector:
                    stats_dict["enhanced_tracing"] = {
                        "intent_method_distribution": self.stats.intent_method_distribution,
                        "llm_invocation_count": self.stats.llm_invocation_count,
                        "trace_files": self.stats.trace_files,
                    }
                
                json.dump(stats_dict, f, indent=2)
            
            logger.info(f"Pipeline complete. Stats saved to {stats_file}")
            
            return self.stats
            
        except Exception as e:
            logger.exception("Pipeline failed")
            raise
    
    async def _collect_with_enhanced_tracing(self):
        """Collect data with enhanced tracing (intent parsing, LLM usage, etc.)."""
        logger.info("Phase 1: Enhanced Agent Collection with Tracing")
        
        # Initialize enhanced collector
        self.enhanced_collector = EnhancedAgentCollector(
            output_dir=self.config.output_dir / "traces",
            capture_spans=self.config.capture_traces,
        )
        
        # Build query list from sample queries
        queries = self._build_query_set(
            self.config.real_single_turn + self.config.real_multi_turn_sessions * 3
        )
        
        logger.info(f"Running {len(queries)} queries with enhanced tracing")
        
        # Run with enhanced tracing
        traces = await self.enhanced_collector.run_batch(queries)
        self.enhanced_traces = traces
        
        # Convert to standard training examples
        for trace in traces:
            if trace.success:
                example = trace.to_training_example()
                example["source"] = "real_agent_enhanced"
                self.all_examples.append(example)
                
                # Update stats
                category = trace.category
                self.stats.category_counts[category] = \
                    self.stats.category_counts.get(category, 0) + 1
                
                if trace.tool_selected:
                    self.stats.tool_counts[trace.tool_selected] = \
                        self.stats.tool_counts.get(trace.tool_selected, 0) + 1
                
                # Track intent parsing methods
                method = trace.intent_parse.method or "unknown"
                self.stats.intent_method_distribution[method] = \
                    self.stats.intent_method_distribution.get(method, 0) + 1
                
                if trace.intent_parse.llm_invoked:
                    self.stats.llm_invocation_count += 1
        
        # Export traces for analysis
        if self.config.capture_traces:
            trace_files = self.enhanced_collector.export_traces(
                prefix=f"{self.config.prefix}_traces"
            )
            self.stats.trace_files = {k: str(v) for k, v in trace_files.items()}
        
        self.stats.real_queries = len(traces)
        self.stats.real_successful = sum(1 for t in traces if t.success)
        
        # Print summary
        self.enhanced_collector.print_summary()
        
        logger.info(f"Enhanced collection complete: {self.stats.real_successful}/{self.stats.real_queries} successful")
    
    def _build_query_set(self, target_count: int) -> List[Tuple[str, str]]:
        """Build a diverse set of queries for collection."""
        # Use the query generator from agent_data_collector
        from .agent_data_collector import QUERY_TEMPLATES
        
        queries = []
        for category, category_queries in QUERY_TEMPLATES.items():
            for query in category_queries:
                # Skip template queries with placeholders
                if '{' not in query:
                    queries.append((query, category))
        
        # Duplicate if needed to reach target
        while len(queries) < target_count:
            queries.extend(queries[:target_count - len(queries)])
        
        return queries[:target_count]
    
    async def _collect_real_data(self):
        """Collect data from real agent interactions (basic collector)."""
        logger.info("Phase 1: Real Agent Collection (Basic)")
        
        self.real_collector = AgentDataCollector(
            output_dir=self.config.output_dir,
            system_prompt=self.system_prompt,
        )
        
        # Single-turn collection
        if self.config.real_single_turn > 0:
            logger.info(f"Collecting {self.config.real_single_turn} single-turn queries")
            await self.real_collector.run_single_turn_collection(
                num_queries=self.config.real_single_turn
            )
        
        # Multi-turn collection
        if self.config.real_multi_turn_sessions > 0:
            logger.info(f"Collecting {self.config.real_multi_turn_sessions} multi-turn sessions")
            await self.real_collector.run_multi_turn_collection(
                num_sessions=self.config.real_multi_turn_sessions
            )
        
        # Convert to standard format
        for interaction in self.real_collector.all_interactions:
            if interaction.success:
                example = {
                    "source": "real_agent",
                    "instruction": interaction.query,
                    "input": "",
                    "output": interaction.response,
                    "category": interaction.category,
                    "tools_used": [t["tool_name"] for t in interaction.tools_used],
                    "response_type": interaction.response_type,
                }
                self.all_examples.append(example)
                
                # Update stats
                self.stats.category_counts[interaction.category] = \
                    self.stats.category_counts.get(interaction.category, 0) + 1
                
                for tool in interaction.tools_used:
                    tool_name = tool["tool_name"]
                    self.stats.tool_counts[tool_name] = \
                        self.stats.tool_counts.get(tool_name, 0) + 1
        
        self.stats.real_queries = self.real_collector.stats["total_queries"]
        self.stats.real_successful = self.real_collector.stats["successful_queries"]
        
        logger.info(f"Real collection complete: {self.stats.real_successful}/{self.stats.real_queries} successful")
    
    async def _generate_synthetic_data(self):
        """Generate synthetic conversation data."""
        logger.info("Phase 2: Synthetic Conversation Generation")
        
        # Generate conversations using the generator
        generator = ConversationGenerator()
        conversations = await generator.generate_dataset(num_conversations=self.config.synthetic_conversations)
        
        logger.info(f"Generated {len(conversations)} synthetic conversations, running through system...")
        
        # Run conversations through the runner
        runner = ConversationRunner()
        results = await runner.run_all(conversations, save_results=False)
        
        # Convert to standard format
        for result in results:
            # Each TurnResult has user_message, system_response, etc.
            for turn_result in result.turn_results:
                example = {
                    "source": "synthetic",
                    "instruction": turn_result.user_message,
                    "input": "",
                    "output": turn_result.system_response,
                    "category": result.category,
                    "tools_used": turn_result.tools_selected,  # TurnResult uses tools_selected
                    "response_type": "success" if turn_result.success else "error",
                }
                self.all_examples.append(example)
                self.stats.synthetic_examples += 1
                
                # Update category stats
                category = example["category"]
                if category:
                    self.stats.category_counts[category] = \
                        self.stats.category_counts.get(category, 0) + 1
        
        self.stats.synthetic_conversations = len(results)
        
        logger.info(f"Synthetic generation complete: {self.stats.synthetic_examples} examples from {len(results)} conversations")
    
    def _combine_and_filter(self):
        """Combine and filter examples for quality."""
        logger.info("Phase 3: Combining and Filtering")
        
        initial_count = len(self.all_examples)
        
        # Filter by quality
        filtered = []
        for example in self.all_examples:
            output = example.get("output", "")
            instruction = example.get("instruction", "")
            
            # Skip empty or too short
            if len(output) < self.config.min_response_length:
                continue
            
            # Skip too long
            if len(output) > self.config.max_response_length:
                # Truncate if slightly over
                if len(output) < self.config.max_response_length * 1.5:
                    example["output"] = output[:self.config.max_response_length] + "..."
                else:
                    continue
            
            # Skip empty instructions
            if len(instruction.strip()) < 5:
                continue
            
            filtered.append(example)
        
        self.all_examples = filtered
        self.stats.total_examples = len(filtered)
        self.stats.filtered_examples = initial_count - len(filtered)
        
        logger.info(f"Filtering complete: {len(filtered)} examples ({self.stats.filtered_examples} filtered)")
    
    def _export_all(self) -> Dict[str, Path]:
        """Export all examples in configured formats."""
        logger.info("Phase 4: Exporting")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_files = {}
        
        if "alpaca" in self.config.export_formats:
            # Alpaca format
            alpaca_data = []
            for example in self.all_examples:
                alpaca_data.append({
                    "instruction": example["instruction"],
                    "input": example.get("input", ""),
                    "output": example["output"],
                })
            
            path = self.config.output_dir / f"{self.config.prefix}_alpaca_{timestamp}.json"
            with open(path, 'w') as f:
                json.dump(alpaca_data, f, indent=2)
            output_files["alpaca"] = path
            logger.info(f"Exported Alpaca: {path} ({len(alpaca_data)} examples)")
        
        if "openai" in self.config.export_formats:
            # OpenAI chat format (JSONL)
            path = self.config.output_dir / f"{self.config.prefix}_openai_{timestamp}.jsonl"
            with open(path, 'w') as f:
                for example in self.all_examples:
                    chat = {
                        "messages": [
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": example["instruction"]},
                            {"role": "assistant", "content": example["output"]},
                        ]
                    }
                    f.write(json.dumps(chat) + "\n")
            output_files["openai"] = path
            logger.info(f"Exported OpenAI: {path}")
        
        if "sharegpt" in self.config.export_formats:
            # ShareGPT format
            sharegpt_data = []
            for example in self.all_examples:
                sharegpt_data.append({
                    "conversations": [
                        {"from": "system", "value": self.system_prompt},
                        {"from": "human", "value": example["instruction"]},
                        {"from": "gpt", "value": example["output"]},
                    ],
                    "metadata": {
                        "source": example.get("source", "unknown"),
                        "category": example.get("category", ""),
                        "tools_used": example.get("tools_used", []),
                    }
                })
            
            path = self.config.output_dir / f"{self.config.prefix}_sharegpt_{timestamp}.json"
            with open(path, 'w') as f:
                json.dump(sharegpt_data, f, indent=2)
            output_files["sharegpt"] = path
            logger.info(f"Exported ShareGPT: {path}")
        
        # Always export raw combined data
        raw_path = self.config.output_dir / f"{self.config.prefix}_raw_{timestamp}.json"
        with open(raw_path, 'w') as f:
            json.dump(self.all_examples, f, indent=2)
        output_files["raw"] = raw_path
        
        return output_files
    
    def print_summary(self):
        """Print a summary of the pipeline run."""
        print("\n" + "=" * 70)
        print("TRAINING DATA PIPELINE SUMMARY")
        print("=" * 70)
        
        print(f"\n📊 COLLECTION STATS")
        print(f"  Real Agent Queries:   {self.stats.real_queries}")
        print(f"  Real Successful:      {self.stats.real_successful}")
        print(f"  Synthetic Convos:     {self.stats.synthetic_conversations}")
        print(f"  Synthetic Examples:   {self.stats.synthetic_examples}")
        
        print(f"\n📦 FINAL DATASET")
        print(f"  Total Examples:       {self.stats.total_examples}")
        print(f"  Filtered Out:         {self.stats.filtered_examples}")
        
        if self.stats.category_counts:
            print(f"\n📁 BY CATEGORY")
            for cat, count in sorted(self.stats.category_counts.items(), key=lambda x: -x[1]):
                print(f"  {cat}: {count}")
        
        if self.stats.tool_counts:
            print(f"\n🔧 TOOLS USED (top 10)")
            for tool, count in sorted(self.stats.tool_counts.items(), key=lambda x: -x[1])[:10]:
                print(f"  {tool}: {count}")
        
        print(f"\n⏱️ DURATION: {self.stats.duration_seconds:.1f} seconds")
        print("=" * 70)


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BioPipelines Training Data Pipeline")
    parser.add_argument("--real", type=int, default=50, help="Number of real agent queries")
    parser.add_argument("--real-sessions", type=int, default=10, help="Number of multi-turn sessions")
    parser.add_argument("--synthetic", type=int, default=100, help="Number of synthetic conversations")
    parser.add_argument("--output", type=str, default="training_data", help="Output directory")
    parser.add_argument("--prefix", type=str, default="biopipelines", help="Output file prefix")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    config = PipelineConfig(
        real_single_turn=args.real,
        real_multi_turn_sessions=args.real_sessions,
        synthetic_conversations=args.synthetic,
        output_dir=Path(args.output),
        prefix=args.prefix,
    )
    
    pipeline = TrainingDataPipeline(config)
    stats = await pipeline.run()
    pipeline.print_summary()
    
    print(f"\n✅ Training data saved to: {config.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
