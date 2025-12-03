"""
Data Pipeline
=============

Processes, validates, and prepares training data for fine-tuning.
"""

import json
import logging
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
import math

from .config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics for a dataset."""
    
    total_examples: int = 0
    valid_examples: int = 0
    
    # Intent metrics
    intent_parse_rate: float = 0.0
    intent_confidence_mean: float = 0.0
    
    # Tool metrics
    tool_coverage: float = 0.0
    unique_tools: int = 0
    
    # Workflow metrics
    workflow_valid_rate: float = 0.0
    
    # Category distribution
    category_distribution: Dict[str, int] = field(default_factory=dict)
    category_entropy: float = 0.0
    
    # Difficulty distribution
    difficulty_distribution: Dict[int, int] = field(default_factory=dict)
    
    # Source distribution
    source_distribution: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_examples": self.total_examples,
            "valid_examples": self.valid_examples,
            "validity_rate": self.valid_examples / self.total_examples if self.total_examples > 0 else 0,
            "intent_parse_rate": self.intent_parse_rate,
            "intent_confidence_mean": self.intent_confidence_mean,
            "tool_coverage": self.tool_coverage,
            "unique_tools": self.unique_tools,
            "workflow_valid_rate": self.workflow_valid_rate,
            "category_distribution": self.category_distribution,
            "category_entropy": self.category_entropy,
            "difficulty_distribution": self.difficulty_distribution,
            "source_distribution": self.source_distribution,
        }


class DataValidator:
    """Validates training examples."""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self._known_tools = self._load_known_tools()
        self._known_analysis_types = self._load_known_analysis_types()
    
    def _load_known_tools(self) -> Set[str]:
        """Load list of known tools."""
        try:
            from ..agents.rag.tool_catalog_indexer import TOOL_DESCRIPTIONS
            return set(TOOL_DESCRIPTIONS.keys())
        except ImportError:
            logger.warning("Could not load tool descriptions")
            return set()
    
    def _load_known_analysis_types(self) -> Set[str]:
        """Load list of known analysis types."""
        try:
            from ..core.query_parser import AnalysisType
            return {at.value for at in AnalysisType}
        except ImportError:
            logger.warning("Could not load analysis types")
            return set()
    
    def validate_example(self, example: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a single training example."""
        
        errors = []
        
        # Required fields
        if not example.get('query'):
            errors.append("Missing query field")
        
        if not example.get('id'):
            errors.append("Missing id field")
        
        # Intent validation
        intent = example.get('intent', {})
        if self.config.require_tool_coverage:
            if not intent:
                errors.append("Missing intent")
            elif not intent.get('analysis_type'):
                errors.append("Missing analysis_type in intent")
        
        # Workflow validation
        if self.config.require_validated_workflow:
            workflow = example.get('workflow')
            if not workflow:
                errors.append("Missing workflow")
            elif not self._validate_workflow_syntax(workflow):
                errors.append("Invalid workflow syntax")
        
        # Tool validation
        tools = example.get('tools', [])
        if self.config.require_tool_coverage and tools:
            unknown_tools = [t for t in tools if t.lower() not in {k.lower() for k in self._known_tools}]
            if unknown_tools:
                errors.append(f"Unknown tools: {unknown_tools}")
        
        # Quality check
        if example.get('quality_score', 1.0) < self.config.min_quality_score:
            errors.append(f"Quality score too low: {example.get('quality_score')}")
        
        return len(errors) == 0, errors
    
    def _validate_workflow_syntax(self, workflow: str) -> bool:
        """Basic validation of Nextflow workflow syntax."""
        
        if not workflow:
            return False
        
        # Check for basic Nextflow constructs
        has_process_or_workflow = 'process' in workflow or 'workflow' in workflow
        has_include = 'include' in workflow
        
        # At least one of these should be present
        if not (has_process_or_workflow or has_include):
            return False
        
        # Check for balanced braces
        open_braces = workflow.count('{')
        close_braces = workflow.count('}')
        
        if open_braces != close_braces:
            return False
        
        return True


class QualityScorer:
    """Scores training example quality."""
    
    def score_example(self, example: Dict[str, Any]) -> float:
        """Calculate quality score for an example."""
        
        score = 0.5  # Base score
        
        # Intent quality
        intent = example.get('intent', {})
        if intent:
            score += 0.1
            if intent.get('confidence', 0) > 0.8:
                score += 0.1
        
        # Tool coverage
        tools = example.get('tools', [])
        if tools:
            score += 0.1
            if len(tools) >= 2:
                score += 0.05
        
        # Workflow present and valid
        workflow = example.get('workflow')
        if workflow:
            score += 0.1
            if example.get('validated', False):
                score += 0.1
        
        # Explanation present
        if example.get('explanation'):
            score += 0.05
        
        return min(1.0, score)


class TrainingDataPipeline:
    """Process and prepare training data."""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.validator = DataValidator(config)
        self.scorer = QualityScorer()
        
        # Ensure directories exist
        self.config.raw_dir.mkdir(parents=True, exist_ok=True)
        self.config.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_raw_data(self) -> List[Dict[str, Any]]:
        """Load all raw training data."""
        
        examples = []
        
        # Load from synthetic directory
        synthetic_dir = self.config.raw_dir / "synthetic"
        if synthetic_dir.exists():
            for file_path in synthetic_dir.glob("*.jsonl"):
                examples.extend(self._load_jsonl(file_path))
        
        # Load from interactions directory
        interactions_dir = self.config.raw_dir / "interactions"
        if interactions_dir.exists():
            for file_path in interactions_dir.glob("*.jsonl"):
                examples.extend(self._load_jsonl(file_path))
        
        # Load any files directly in raw_dir
        for file_path in self.config.raw_dir.glob("*.jsonl"):
            examples.extend(self._load_jsonl(file_path))
        
        logger.info(f"Loaded {len(examples)} raw examples")
        return examples
    
    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        """Load examples from JSONL file."""
        
        examples = []
        
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            examples.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON in {path}: {e}")
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
        
        return examples
    
    def validate_examples(
        self, 
        examples: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Validate examples and separate valid from invalid."""
        
        valid = []
        invalid = []
        
        for example in examples:
            is_valid, errors = self.validator.validate_example(example)
            
            if is_valid:
                valid.append(example)
            else:
                example['validation_errors'] = errors
                invalid.append(example)
        
        logger.info(f"Validation: {len(valid)} valid, {len(invalid)} invalid")
        return valid, invalid
    
    def deduplicate(
        self, 
        examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate examples based on query similarity."""
        
        if not self.config.deduplicate:
            return examples
        
        seen_hashes = set()
        unique = []
        
        for example in examples:
            # Create hash from normalized query
            query = example.get('query', '').lower().strip()
            query_hash = hashlib.md5(query.encode()).hexdigest()
            
            if query_hash not in seen_hashes:
                seen_hashes.add(query_hash)
                unique.append(example)
        
        logger.info(f"Deduplication: {len(examples)} -> {len(unique)} examples")
        return unique
    
    def score_and_filter(
        self, 
        examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Score examples and filter by quality threshold."""
        
        scored = []
        
        for example in examples:
            # Use existing score or calculate new one
            if 'quality_score' not in example:
                example['quality_score'] = self.scorer.score_example(example)
            
            if example['quality_score'] >= self.config.min_quality_score:
                scored.append(example)
        
        logger.info(f"Quality filter: {len(examples)} -> {len(scored)} examples")
        return scored
    
    def create_splits(
        self, 
        examples: List[Dict[str, Any]]
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Create train/val/test splits with stratification."""
        
        # Group by analysis type for stratification
        by_type = {}
        for example in examples:
            analysis_type = example.get('analysis_type', 'unknown')
            if analysis_type not in by_type:
                by_type[analysis_type] = []
            by_type[analysis_type].append(example)
        
        train, val, test = [], [], []
        
        for analysis_type, type_examples in by_type.items():
            n = len(type_examples)
            
            # Calculate split sizes
            n_train = int(n * self.config.train_ratio)
            n_val = int(n * self.config.val_ratio)
            
            # Shuffle deterministically
            type_examples.sort(key=lambda x: x.get('id', ''))
            
            # Split
            train.extend(type_examples[:n_train])
            val.extend(type_examples[n_train:n_train + n_val])
            test.extend(type_examples[n_train + n_val:])
        
        logger.info(f"Splits: train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test
    
    def calculate_metrics(
        self, 
        examples: List[Dict[str, Any]]
    ) -> QualityMetrics:
        """Calculate quality metrics for a dataset."""
        
        metrics = QualityMetrics()
        metrics.total_examples = len(examples)
        
        # Count various attributes
        has_intent = 0
        confidence_sum = 0.0
        has_workflow = 0
        valid_workflow = 0
        all_tools = set()
        category_counts = Counter()
        difficulty_counts = Counter()
        source_counts = Counter()
        
        for example in examples:
            # Intent
            intent = example.get('intent', {})
            if intent:
                has_intent += 1
                confidence_sum += intent.get('confidence', 0.5)
            
            # Workflow
            if example.get('workflow'):
                has_workflow += 1
                if example.get('validated', False):
                    valid_workflow += 1
            
            # Tools
            for tool in example.get('tools', []):
                all_tools.add(tool.lower())
            
            # Category
            category_counts[example.get('category', 'unknown')] += 1
            
            # Difficulty
            difficulty_counts[example.get('difficulty', 1)] += 1
            
            # Source
            source_counts[example.get('source', 'unknown')] += 1
        
        # Calculate final metrics
        if examples:
            metrics.valid_examples = has_workflow
            metrics.intent_parse_rate = has_intent / len(examples)
            metrics.intent_confidence_mean = confidence_sum / max(has_intent, 1)
            metrics.workflow_valid_rate = valid_workflow / max(has_workflow, 1)
        
        metrics.unique_tools = len(all_tools)
        
        # Load known tools for coverage calculation
        try:
            from ..agents.rag.tool_catalog_indexer import TOOL_DESCRIPTIONS
            known_tools = set(k.lower() for k in TOOL_DESCRIPTIONS.keys())
            covered = len(all_tools & known_tools)
            metrics.tool_coverage = covered / len(known_tools) if known_tools else 0
        except ImportError:
            metrics.tool_coverage = 0
        
        metrics.category_distribution = dict(category_counts)
        metrics.difficulty_distribution = dict(difficulty_counts)
        metrics.source_distribution = dict(source_counts)
        
        # Calculate category entropy
        if category_counts:
            total = sum(category_counts.values())
            probs = [c / total for c in category_counts.values()]
            metrics.category_entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        
        return metrics
    
    def save_split(
        self, 
        examples: List[Dict[str, Any]], 
        split_name: str
    ) -> Path:
        """Save a data split to file."""
        
        output_path = self.config.processed_dir / f"{split_name}.jsonl"
        
        with open(output_path, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        
        logger.info(f"Saved {len(examples)} examples to {output_path}")
        return output_path
    
    def process_all(self) -> Dict[str, Any]:
        """Run the complete processing pipeline."""
        
        # Load raw data
        raw = self.load_raw_data()
        
        if not raw:
            logger.warning("No raw data found")
            return {"status": "no_data"}
        
        # Validate
        valid, invalid = self.validate_examples(raw)
        
        # Deduplicate
        deduplicated = self.deduplicate(valid)
        
        # Score and filter
        filtered = self.score_and_filter(deduplicated)
        
        # Create splits
        train, val, test = self.create_splits(filtered)
        
        # Save splits
        train_path = self.save_split(train, "train")
        val_path = self.save_split(val, "val")
        test_path = self.save_split(test, "test")
        
        # Save invalid for review
        if invalid:
            invalid_path = self.config.processed_dir / "invalid.jsonl"
            with open(invalid_path, 'w') as f:
                for example in invalid:
                    f.write(json.dumps(example) + '\n')
        
        # Calculate metrics
        metrics = self.calculate_metrics(filtered)
        
        # Save metrics
        metrics_path = self.config.processed_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        
        return {
            "status": "success",
            "raw_count": len(raw),
            "valid_count": len(valid),
            "final_count": len(filtered),
            "train_count": len(train),
            "val_count": len(val),
            "test_count": len(test),
            "train_path": str(train_path),
            "val_path": str(val_path),
            "test_path": str(test_path),
            "metrics": metrics.to_dict(),
        }


def process_training_data(
    raw_dir: Path = None,
    output_dir: Path = None,
) -> Dict[str, Any]:
    """Convenience function to process all training data."""
    
    config = PipelineConfig()
    if raw_dir:
        config.raw_dir = raw_dir
    if output_dir:
        config.processed_dir = output_dir
    
    pipeline = TrainingDataPipeline(config)
    return pipeline.process_all()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    result = process_training_data()
    print(json.dumps(result, indent=2))
