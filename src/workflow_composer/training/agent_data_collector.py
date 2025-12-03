"""
Real Agent Data Collector
=========================

Collects training data by running actual queries through the UnifiedAgent.
Unlike the synthetic conversation runner, this uses real agent responses
and actual tool executions.

Usage:
    collector = AgentDataCollector()
    await collector.run_collection(num_queries=100)
    collector.export_training_data()
"""

import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Set environment variable before any agent imports
os.environ.setdefault('BIOPIPELINES_LLM_ENABLED', 'false')

from ..agents.unified_agent import UnifiedAgent, AgentResponse, ResponseType

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class AgentInteraction:
    """Record of a single agent interaction."""
    
    query: str
    response: str
    response_type: str
    success: bool
    tools_used: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    category: str = ""
    duration_ms: float = 0.0
    
    def to_training_example(self) -> Dict[str, Any]:
        """Convert to training example format."""
        return {
            "instruction": self.query,
            "input": "",
            "output": self.response,
            "category": self.category,
            "tools_used": [t["tool_name"] for t in self.tools_used],
            "success": self.success,
        }
    
    def to_chat_format(self, system_prompt: str = None) -> Dict[str, Any]:
        """Convert to OpenAI chat format."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": self.query})
        messages.append({"role": "assistant", "content": self.response})
        
        return {"messages": messages}


@dataclass
class ConversationSession:
    """A multi-turn conversation session."""
    
    session_id: str
    interactions: List[AgentInteraction] = field(default_factory=list)
    category: str = ""
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: str = ""
    
    def add_interaction(self, interaction: AgentInteraction):
        """Add an interaction to the session."""
        self.interactions.append(interaction)
        if not self.category and interaction.category:
            self.category = interaction.category
    
    def to_training_examples(self) -> List[Dict[str, Any]]:
        """Convert all interactions to training examples."""
        return [i.to_training_example() for i in self.interactions]
    
    def to_multi_turn_chat(self, system_prompt: str = None) -> Dict[str, Any]:
        """Convert session to multi-turn chat format."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        for interaction in self.interactions:
            messages.append({"role": "user", "content": interaction.query})
            messages.append({"role": "assistant", "content": interaction.response})
        
        return {"messages": messages}


# =============================================================================
# QUERY TEMPLATES
# =============================================================================

# Categories of queries that the agent handles
QUERY_TEMPLATES = {
    "data_discovery": [
        "scan my data folder",
        "scan the data directory",
        "what files do I have in data/raw?",
        "show me my FASTQ files",
        "find RNA-seq data in the workspace",
        "list all sequencing data",
        "describe the files in data/processed",
        "what data formats do I have?",
        "count my samples",
        "scan {path} for sequencing files",
    ],
    
    "workflow_generation": [
        "create an RNA-seq workflow",
        "generate a ChIP-seq pipeline",
        "build a methylation analysis workflow",
        "create an ATAC-seq pipeline",
        "generate a single-cell RNA-seq workflow",
        "make a variant calling pipeline for WGS data",
        "create differential expression workflow",
        "generate a Hi-C workflow",
        "build a metagenomics pipeline",
        "I need a workflow for {analysis_type} analysis",
    ],
    
    "workflow_listing": [
        "list my workflows",
        "show available workflows",
        "what workflows have I created?",
        "list generated pipelines",
        "show me the workflows",
    ],
    
    "job_management": [
        "show my jobs",
        "what jobs are running?",
        "check job status",
        "list running jobs",
        "any jobs pending?",
        "show recent job history",
        "check if my jobs completed",
    ],
    
    "education": [
        "explain what DESeq2 does",
        "what is differential expression analysis?",
        "explain ChIP-seq",
        "how does STAR alignment work?",
        "what are normalized counts?",
        "explain the difference between TPM and FPKM",
        "what is peak calling?",
        "how does bisulfite sequencing work?",
        "what are DMRs?",
        "explain single-cell clustering",
        "what is {concept}?",
    ],
    
    "database_search": [
        "search for TP53 in databases",
        "find information about BRCA1",
        "search ENCODE for K562 data",
        "look up EGFR pathway",
        "find protein interactions for MYC",
        "search for {gene} data",
    ],
    
    "references": [
        "check my reference genomes",
        "what references do I have?",
        "check if I have GRCh38 reference",
        "list available genome indices",
        "do I have STAR index for human?",
    ],
    
    "help": [
        "what can you help me with?",
        "show help",
        "list your capabilities",
        "what commands are available?",
        "how do I use this system?",
        "what tools do you have?",
    ],
    
    "diagnostics": [
        "diagnose the last error",
        "what went wrong with my job?",
        "explain this error",
        "troubleshoot my failed workflow",
        "why did my job fail?",
    ],
}

# Variable substitutions for templates
TEMPLATE_VARIABLES = {
    "{path}": [
        "data/raw",
        "data/processed",
        ".",
        "data",
        "generated_workflows",
    ],
    "{analysis_type}": [
        "RNA-seq",
        "ChIP-seq",
        "ATAC-seq",
        "methylation",
        "variant calling",
        "single-cell",
    ],
    "{concept}": [
        "DESeq2",
        "STAR alignment",
        "peak calling",
        "differential expression",
        "normalization",
        "clustering",
        "batch correction",
    ],
    "{gene}": [
        "TP53",
        "BRCA1",
        "EGFR",
        "MYC",
        "KRAS",
        "PTEN",
    ],
}


# =============================================================================
# AGENT DATA COLLECTOR
# =============================================================================

class AgentDataCollector:
    """
    Collects training data by running queries through the real UnifiedAgent.
    
    This collector:
    1. Generates queries from templates
    2. Runs them through the actual agent
    3. Captures responses and tool executions
    4. Exports in training-ready formats
    """
    
    def __init__(
        self,
        output_dir: Path = None,
        system_prompt: str = None,
    ):
        """
        Initialize the collector.
        
        Args:
            output_dir: Where to save collected data
            system_prompt: System prompt to include in training examples
        """
        self.output_dir = output_dir or Path("training_data")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.system_prompt = system_prompt or (
            "You are BioPipelines AI, an expert assistant for bioinformatics workflows. "
            "You help users analyze sequencing data, generate workflows, manage jobs, "
            "and troubleshoot issues. Be helpful, accurate, and proactive."
        )
        
        self.agent: Optional[UnifiedAgent] = None
        self.sessions: List[ConversationSession] = []
        self.all_interactions: List[AgentInteraction] = []
        
        # Statistics
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "by_category": {},
            "tools_used": {},
        }
    
    async def initialize(self):
        """Initialize the agent."""
        if self.agent is None:
            logger.info("Initializing UnifiedAgent...")
            self.agent = UnifiedAgent()
            logger.info("Agent initialized successfully")
    
    def _generate_query(self, category: str = None) -> Tuple[str, str]:
        """
        Generate a query from templates.
        
        Args:
            category: Specific category to generate from
            
        Returns:
            Tuple of (query, category)
        """
        if category is None:
            category = random.choice(list(QUERY_TEMPLATES.keys()))
        
        template = random.choice(QUERY_TEMPLATES[category])
        
        # Substitute variables
        query = template
        for var, values in TEMPLATE_VARIABLES.items():
            if var in query:
                query = query.replace(var, random.choice(values))
        
        return query, category
    
    async def run_query(self, query: str, category: str = "") -> AgentInteraction:
        """
        Run a single query through the agent.
        
        Args:
            query: The query to run
            category: Category label for the query
            
        Returns:
            AgentInteraction with the results
        """
        await self.initialize()
        
        start_time = datetime.now()
        
        try:
            response = await self.agent.process_query(query)
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            # Extract tool executions
            tools_used = []
            if response.tool_executions:
                for exec_info in response.tool_executions:
                    tools_used.append({
                        "tool_name": exec_info.tool_name,
                        "parameters": exec_info.parameters,
                        "success": exec_info.result.success if exec_info.result else False,
                    })
            
            interaction = AgentInteraction(
                query=query,
                response=response.message or "",
                response_type=response.response_type.value if response.response_type else "unknown",
                success=response.success,
                tools_used=tools_used,
                category=category,
                duration_ms=duration,
            )
            
            # Update statistics
            self.stats["total_queries"] += 1
            if response.success:
                self.stats["successful_queries"] += 1
            else:
                self.stats["failed_queries"] += 1
            
            self.stats["by_category"][category] = self.stats["by_category"].get(category, 0) + 1
            
            for tool in tools_used:
                tool_name = tool["tool_name"]
                self.stats["tools_used"][tool_name] = self.stats["tools_used"].get(tool_name, 0) + 1
            
            return interaction
            
        except Exception as e:
            logger.error(f"Error running query '{query}': {e}")
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            interaction = AgentInteraction(
                query=query,
                response=f"Error: {str(e)}",
                response_type="error",
                success=False,
                category=category,
                duration_ms=duration,
            )
            
            self.stats["total_queries"] += 1
            self.stats["failed_queries"] += 1
            
            return interaction
    
    async def run_single_turn_collection(
        self,
        num_queries: int = 100,
        categories: List[str] = None,
    ) -> List[AgentInteraction]:
        """
        Run a collection of single-turn queries.
        
        Args:
            num_queries: Number of queries to run
            categories: Specific categories to include (None = all)
            
        Returns:
            List of AgentInteractions
        """
        logger.info(f"Starting single-turn collection with {num_queries} queries")
        
        if categories is None:
            categories = list(QUERY_TEMPLATES.keys())
        
        interactions = []
        
        for i in range(num_queries):
            # Rotate through categories
            category = categories[i % len(categories)]
            query, cat = self._generate_query(category)
            
            logger.debug(f"Query {i+1}/{num_queries} [{category}]: {query}")
            
            interaction = await self.run_query(query, category=cat)
            interactions.append(interaction)
            self.all_interactions.append(interaction)
            
            # Progress update
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{num_queries} queries completed")
        
        logger.info(f"Completed single-turn collection: {len(interactions)} interactions")
        return interactions
    
    async def run_multi_turn_collection(
        self,
        num_sessions: int = 20,
        turns_per_session: Tuple[int, int] = (3, 7),
    ) -> List[ConversationSession]:
        """
        Run multi-turn conversation collection.
        
        Args:
            num_sessions: Number of conversation sessions
            turns_per_session: Min and max turns per session
            
        Returns:
            List of ConversationSessions
        """
        logger.info(f"Starting multi-turn collection with {num_sessions} sessions")
        
        # Conversation flow patterns
        flow_patterns = [
            # Data exploration flow
            ["data_discovery", "workflow_generation", "job_management"],
            # Learning flow
            ["help", "education", "education", "workflow_generation"],
            # Workflow creation flow
            ["workflow_generation", "workflow_listing", "job_management"],
            # Troubleshooting flow
            ["job_management", "diagnostics", "references"],
            # Research flow
            ["database_search", "education", "workflow_generation"],
        ]
        
        for i in range(num_sessions):
            session = ConversationSession(
                session_id=f"session_{i+1}_{datetime.now().strftime('%H%M%S')}"
            )
            
            # Choose a flow pattern
            pattern = random.choice(flow_patterns)
            num_turns = random.randint(turns_per_session[0], turns_per_session[1])
            
            # Extend pattern if needed
            while len(pattern) < num_turns:
                pattern = pattern + random.choice(flow_patterns)
            
            for turn in range(num_turns):
                category = pattern[turn % len(pattern)]
                query, cat = self._generate_query(category)
                
                interaction = await self.run_query(query, category=cat)
                session.add_interaction(interaction)
                self.all_interactions.append(interaction)
            
            session.end_time = datetime.now().isoformat()
            self.sessions.append(session)
            
            if (i + 1) % 5 == 0:
                logger.info(f"Progress: {i+1}/{num_sessions} sessions completed")
        
        logger.info(f"Completed multi-turn collection: {len(self.sessions)} sessions")
        return self.sessions
    
    def export_training_data(
        self,
        formats: List[str] = None,
        prefix: str = "agent_training",
    ) -> Dict[str, Path]:
        """
        Export collected data in various formats.
        
        Args:
            formats: List of formats to export (alpaca, openai, sharegpt)
            prefix: Prefix for output files
            
        Returns:
            Dict mapping format names to output paths
        """
        if formats is None:
            formats = ["alpaca", "openai", "sharegpt"]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_files = {}
        
        # Filter to successful interactions only for training
        successful = [i for i in self.all_interactions if i.success]
        
        logger.info(f"Exporting {len(successful)} successful interactions")
        
        if "alpaca" in formats:
            # Alpaca format: instruction, input, output
            alpaca_data = []
            for interaction in successful:
                alpaca_data.append({
                    "instruction": interaction.query,
                    "input": "",
                    "output": interaction.response,
                })
            
            path = self.output_dir / f"{prefix}_alpaca_{timestamp}.json"
            with open(path, 'w') as f:
                json.dump(alpaca_data, f, indent=2)
            output_files["alpaca"] = path
            logger.info(f"Exported Alpaca format: {path}")
        
        if "openai" in formats:
            # OpenAI chat format
            openai_data = []
            for interaction in successful:
                openai_data.append({
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": interaction.query},
                        {"role": "assistant", "content": interaction.response},
                    ]
                })
            
            path = self.output_dir / f"{prefix}_openai_{timestamp}.jsonl"
            with open(path, 'w') as f:
                for item in openai_data:
                    f.write(json.dumps(item) + "\n")
            output_files["openai"] = path
            logger.info(f"Exported OpenAI format: {path}")
        
        if "sharegpt" in formats:
            # ShareGPT format
            sharegpt_data = []
            for interaction in successful:
                sharegpt_data.append({
                    "conversations": [
                        {"from": "system", "value": self.system_prompt},
                        {"from": "human", "value": interaction.query},
                        {"from": "gpt", "value": interaction.response},
                    ],
                    "metadata": {
                        "category": interaction.category,
                        "tools_used": [t["tool_name"] for t in interaction.tools_used],
                    }
                })
            
            path = self.output_dir / f"{prefix}_sharegpt_{timestamp}.json"
            with open(path, 'w') as f:
                json.dump(sharegpt_data, f, indent=2)
            output_files["sharegpt"] = path
            logger.info(f"Exported ShareGPT format: {path}")
        
        # Export multi-turn conversations
        if self.sessions:
            multi_turn_data = []
            for session in self.sessions:
                if any(i.success for i in session.interactions):
                    multi_turn_data.append(session.to_multi_turn_chat(self.system_prompt))
            
            path = self.output_dir / f"{prefix}_multiturn_{timestamp}.jsonl"
            with open(path, 'w') as f:
                for item in multi_turn_data:
                    f.write(json.dumps(item) + "\n")
            output_files["multiturn"] = path
            logger.info(f"Exported multi-turn: {path}")
        
        # Export statistics
        stats_path = self.output_dir / f"{prefix}_stats_{timestamp}.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        output_files["stats"] = stats_path
        
        return output_files
    
    def print_summary(self):
        """Print collection summary."""
        print("\n" + "=" * 60)
        print("AGENT DATA COLLECTION SUMMARY")
        print("=" * 60)
        
        print(f"\n📊 Total Queries: {self.stats['total_queries']}")
        print(f"✅ Successful: {self.stats['successful_queries']}")
        print(f"❌ Failed: {self.stats['failed_queries']}")
        
        if self.stats['total_queries'] > 0:
            success_rate = self.stats['successful_queries'] / self.stats['total_queries'] * 100
            print(f"📈 Success Rate: {success_rate:.1f}%")
        
        print("\n📁 By Category:")
        for cat, count in sorted(self.stats["by_category"].items(), key=lambda x: -x[1]):
            print(f"  - {cat}: {count}")
        
        print("\n🔧 Tools Used:")
        for tool, count in sorted(self.stats["tools_used"].items(), key=lambda x: -x[1])[:10]:
            print(f"  - {tool}: {count}")
        
        print("\n💬 Sessions: {}".format(len(self.sessions)))
        print("=" * 60)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def main():
    """Main entry point for running data collection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect training data from the BioPipelines agent")
    parser.add_argument("--single", type=int, default=50, help="Number of single-turn queries")
    parser.add_argument("--sessions", type=int, default=10, help="Number of multi-turn sessions")
    parser.add_argument("--output", type=str, default="training_data", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    collector = AgentDataCollector(output_dir=Path(args.output))
    
    # Run single-turn collection
    if args.single > 0:
        await collector.run_single_turn_collection(num_queries=args.single)
    
    # Run multi-turn collection
    if args.sessions > 0:
        await collector.run_multi_turn_collection(num_sessions=args.sessions)
    
    # Export data
    output_files = collector.export_training_data()
    
    # Print summary
    collector.print_summary()
    
    print("\n📁 Exported Files:")
    for format_name, path in output_files.items():
        print(f"  - {format_name}: {path}")


if __name__ == "__main__":
    asyncio.run(main())
