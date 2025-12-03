"""
Training Data Generator
=======================

Generates synthetic training data from golden queries, tool catalog,
and analysis definitions.
"""

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from .config import GeneratorConfig, VariationType

logger = logging.getLogger(__name__)


@dataclass
class QueryVariation:
    """A variation of a base query."""
    
    text: str
    variation_type: VariationType
    base_query: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "variation_type": self.variation_type.value,
            "base_query": self.base_query,
        }


@dataclass
class TrainingExample:
    """A complete training example."""
    
    id: str
    query: str
    intent: Dict[str, Any]
    tools: List[str] = field(default_factory=list)
    workflow: Optional[str] = None
    explanation: Optional[str] = None
    
    # Metadata
    source: str = "synthetic"
    category: str = "unknown"
    analysis_type: str = "unknown"
    difficulty: int = 1
    validated: bool = False
    
    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_training_format(self, system_prompt: str = "") -> Dict[str, Any]:
        """Convert to instruction-following format."""
        
        # Build assistant response
        response_parts = []
        
        if self.intent:
            response_parts.append("## Analysis Intent")
            response_parts.append(f"- **Type**: {self.intent.get('analysis_type', 'Unknown')}")
            if self.intent.get('organism'):
                response_parts.append(f"- **Organism**: {self.intent['organism']}")
            if self.intent.get('data_type'):
                response_parts.append(f"- **Data**: {self.intent['data_type']}")
        
        if self.tools:
            response_parts.append("\n## Recommended Tools")
            for i, tool in enumerate(self.tools, 1):
                response_parts.append(f"{i}. **{tool}**")
        
        if self.workflow:
            response_parts.append("\n## Workflow")
            response_parts.append(f"```nextflow\n{self.workflow}\n```")
        
        if self.explanation:
            response_parts.append(f"\n## Explanation\n{self.explanation}")
        
        assistant_response = "\n".join(response_parts)
        
        return {
            "id": self.id,
            "messages": [
                {"role": "system", "content": system_prompt} if system_prompt else None,
                {"role": "user", "content": self.query},
                {"role": "assistant", "content": assistant_response},
            ],
            "metadata": {
                "source": self.source,
                "category": self.category,
                "analysis_type": self.analysis_type,
                "tools": self.tools,
                "difficulty": self.difficulty,
                "validated": self.validated,
            }
        }


class TrainingDataGenerator:
    """Generate synthetic training data from golden queries."""
    
    def __init__(self, config: GeneratorConfig = None):
        self.config = config or GeneratorConfig()
        self._llm = None
        self._intent_parser = None
        self._workflow_generator = None
        self._tool_selector = None
        
    def _get_llm(self):
        """Lazy load LLM adapter."""
        if self._llm is None:
            from ..llm import get_llm
            self._llm = get_llm()
        return self._llm
    
    def _get_intent_parser(self):
        """Lazy load intent parser."""
        if self._intent_parser is None:
            from ..core.query_parser import IntentParser
            self._intent_parser = IntentParser(self._get_llm())
        return self._intent_parser
    
    def _get_workflow_generator(self):
        """Lazy load workflow generator."""
        if self._workflow_generator is None:
            from ..core.workflow_generator import WorkflowGenerator
            self._workflow_generator = WorkflowGenerator()
        return self._workflow_generator
    
    def _get_tool_selector(self):
        """Lazy load tool selector."""
        if self._tool_selector is None:
            from ..core.tool_selector import ToolSelector
            self._tool_selector = ToolSelector()
        return self._tool_selector
    
    def _generate_id(self, content: str) -> str:
        """Generate unique ID from content hash."""
        hash_val = hashlib.md5(content.encode()).hexdigest()[:12]
        return f"train_{hash_val}"
    
    async def generate_query_variations(
        self, 
        base_query: str,
        count: int = None
    ) -> List[QueryVariation]:
        """Generate variations of a query using LLM."""
        
        count = count or self.config.variations_per_query
        variations = []
        
        llm = self._get_llm()
        
        # Generate variations for each type
        variations_per_type = count // len(VariationType)
        
        for var_type in VariationType:
            prompt = self._build_variation_prompt(base_query, var_type, variations_per_type)
            
            try:
                from ..llm.base import Message
                response = await llm.agenerate([
                    Message(role="user", content=prompt)
                ])
                
                # Parse variations from response
                parsed = self._parse_variations(response.content, var_type, base_query)
                variations.extend(parsed)
                
            except Exception as e:
                logger.warning(f"Failed to generate {var_type.value} variations: {e}")
                continue
        
        return variations
    
    def _build_variation_prompt(
        self, 
        base_query: str, 
        var_type: VariationType,
        count: int
    ) -> str:
        """Build prompt for generating query variations."""
        
        type_instructions = {
            VariationType.FORMAL: "Rewrite in formal, technical language as a researcher would ask.",
            VariationType.CASUAL: "Rewrite in casual, conversational language as a student might ask.",
            VariationType.MINIMAL: "Create short, abbreviated versions (5-10 words).",
            VariationType.DETAILED: "Create detailed versions with specific parameters and context.",
            VariationType.EDGE_CASE: "Create challenging versions with typos, incomplete sentences, or ambiguous phrasing.",
        }
        
        return f"""Generate {count} variations of this bioinformatics query.

Original query: "{base_query}"

Instructions: {type_instructions[var_type]}

IMPORTANT: 
- Each variation must convey the same analysis intent
- Variations should be realistic queries a user might ask
- Return only the variations, one per line
- Do not include numbers or bullet points

Output {count} variations:"""
    
    def _parse_variations(
        self, 
        response: str, 
        var_type: VariationType,
        base_query: str
    ) -> List[QueryVariation]:
        """Parse variations from LLM response."""
        
        variations = []
        lines = response.strip().split('\n')
        
        for line in lines:
            # Clean up the line
            line = line.strip()
            line = re.sub(r'^[\d\.\)\-\*]+\s*', '', line)  # Remove numbering
            line = line.strip('"\'')  # Remove quotes
            
            if line and len(line) > 5:  # Minimum reasonable query length
                variations.append(QueryVariation(
                    text=line,
                    variation_type=var_type,
                    base_query=base_query,
                ))
        
        return variations
    
    async def generate_full_example(
        self, 
        query: str,
        expected_analysis_type: str = None,
        expected_tools: List[str] = None
    ) -> TrainingExample:
        """Generate a complete training example with intent and workflow."""
        
        # Parse intent
        intent_parser = self._get_intent_parser()
        try:
            intent = await intent_parser.parse_async(query)
            intent_dict = intent.to_dict() if hasattr(intent, 'to_dict') else vars(intent)
        except Exception as e:
            logger.warning(f"Intent parsing failed for '{query[:50]}...': {e}")
            intent_dict = {"analysis_type": expected_analysis_type or "unknown"}
        
        # Get tools
        tool_selector = self._get_tool_selector()
        analysis_type = intent_dict.get('analysis_type', expected_analysis_type)
        
        try:
            tools = tool_selector.find_tools_for_analysis(analysis_type)
            tool_names = [t.name if hasattr(t, 'name') else str(t) for t in tools]
        except Exception as e:
            logger.warning(f"Tool selection failed: {e}")
            tool_names = expected_tools or []
        
        # Generate workflow
        workflow = None
        if self.config.validate_workflows:
            try:
                workflow_gen = self._get_workflow_generator()
                wf = workflow_gen.generate(intent_dict, tool_names)
                workflow = wf.code if hasattr(wf, 'code') else str(wf)
            except Exception as e:
                logger.warning(f"Workflow generation failed: {e}")
        
        # Create example
        example = TrainingExample(
            id=self._generate_id(query),
            query=query,
            intent=intent_dict,
            tools=tool_names,
            workflow=workflow,
            source="synthetic",
            category=self._categorize_query(query),
            analysis_type=analysis_type or "unknown",
            validated=workflow is not None,
        )
        
        return example
    
    def _categorize_query(self, query: str) -> str:
        """Categorize query into a training category."""
        
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ['what is', 'explain', 'how does', 'describe']):
            return "education"
        elif any(kw in query_lower for kw in ['create', 'generate', 'build', 'make']):
            return "workflow_generation"
        elif any(kw in query_lower for kw in ['search', 'find', 'download', 'get']):
            return "data_discovery"
        elif any(kw in query_lower for kw in ['run', 'submit', 'execute', 'start']):
            return "job_management"
        elif any(kw in query_lower for kw in ['error', 'fail', 'fix', 'debug']):
            return "troubleshooting"
        else:
            return "workflow_generation"
    
    async def generate_from_golden_queries(
        self, 
        golden_queries: List[Any] = None
    ) -> List[TrainingExample]:
        """Generate training data from all golden queries."""
        
        # Load golden queries if not provided
        if golden_queries is None:
            golden_queries = self._load_golden_queries()
        
        examples = []
        total = len(golden_queries)
        
        logger.info(f"Generating training data from {total} golden queries")
        
        for i, gq in enumerate(golden_queries):
            if i % 10 == 0:
                logger.info(f"Processing query {i+1}/{total}")
            
            # Get query text
            query = gq.query if hasattr(gq, 'query') else str(gq)
            expected_type = gq.expected_analysis_type.value if hasattr(gq, 'expected_analysis_type') else None
            expected_tools = gq.expected_tools if hasattr(gq, 'expected_tools') else []
            
            try:
                # Generate variations
                variations = await self.generate_query_variations(query, count=10)
                
                # Generate full example for base query
                example = await self.generate_full_example(
                    query, expected_type, expected_tools
                )
                examples.append(example)
                
                # Generate examples for variations (subset)
                for var in variations[:5]:  # Limit to 5 per query
                    var_example = await self.generate_full_example(
                        var.text, expected_type, expected_tools
                    )
                    var_example.source = f"synthetic_variation_{var.variation_type.value}"
                    examples.append(var_example)
                    
            except Exception as e:
                logger.error(f"Failed to process query '{query[:50]}...': {e}")
                continue
        
        logger.info(f"Generated {len(examples)} training examples")
        return examples
    
    def _load_golden_queries(self) -> List[Any]:
        """Load golden queries from test file."""
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "tests"))
            from test_golden_queries import ALL_GOLDEN_QUERIES
            return ALL_GOLDEN_QUERIES
        except ImportError as e:
            logger.error(f"Could not load golden queries: {e}")
            return []
    
    def generate_tool_examples(self) -> List[TrainingExample]:
        """Generate tool selection training examples."""
        
        examples = []
        
        # Load tool catalog
        try:
            from ..agents.rag.tool_catalog_indexer import TOOL_DESCRIPTIONS
        except ImportError:
            logger.warning("Could not load tool descriptions")
            return examples
        
        # Generate examples for each tool
        for tool_name, info in TOOL_DESCRIPTIONS.items():
            # Tool explanation example
            query = f"What is {tool_name} and when should I use it?"
            
            example = TrainingExample(
                id=self._generate_id(f"tool_{tool_name}"),
                query=query,
                intent={"type": "education", "topic": tool_name},
                tools=[tool_name],
                explanation=f"{info.get('full_name', tool_name)}: {info.get('description', '')}",
                source="tool_catalog",
                category="education",
                analysis_type="education",
                difficulty=1,
                validated=True,
            )
            examples.append(example)
            
            # Use cases example
            if info.get('use_cases'):
                use_cases = info['use_cases']
                for use_case in use_cases[:2]:
                    query = f"I need to do {use_case}, what tool should I use?"
                    
                    example = TrainingExample(
                        id=self._generate_id(f"usecase_{tool_name}_{use_case[:20]}"),
                        query=query,
                        intent={"type": "tool_selection", "task": use_case},
                        tools=[tool_name],
                        explanation=f"For {use_case}, I recommend using {info.get('full_name', tool_name)}.",
                        source="tool_catalog",
                        category="tool_selection",
                        analysis_type="tool_selection",
                        difficulty=1,
                        validated=True,
                    )
                    examples.append(example)
        
        logger.info(f"Generated {len(examples)} tool examples")
        return examples
    
    def save_examples(
        self, 
        examples: List[TrainingExample],
        output_path: Path = None
    ) -> Path:
        """Save examples to JSONL file."""
        
        output_path = output_path or self.config.output_dir / "synthetic_data.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for example in examples:
                f.write(json.dumps(example.to_dict()) + '\n')
        
        logger.info(f"Saved {len(examples)} examples to {output_path}")
        return output_path


async def generate_training_data(
    output_dir: Path = None,
    include_variations: bool = True,
    include_tools: bool = True,
) -> Path:
    """Convenience function to generate all training data."""
    
    config = GeneratorConfig()
    if output_dir:
        config.output_dir = output_dir
    
    generator = TrainingDataGenerator(config)
    
    all_examples = []
    
    # Generate from golden queries
    if include_variations:
        golden_examples = await generator.generate_from_golden_queries()
        all_examples.extend(golden_examples)
    
    # Generate tool examples
    if include_tools:
        tool_examples = generator.generate_tool_examples()
        all_examples.extend(tool_examples)
    
    # Save all examples
    output_path = generator.save_examples(all_examples)
    
    return output_path


if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    async def main():
        output = await generate_training_data()
        print(f"Training data saved to: {output}")
    
    asyncio.run(main())
