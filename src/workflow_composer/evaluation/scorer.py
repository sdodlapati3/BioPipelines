"""Scoring module for BioPipelines evaluation.

This module provides different scoring strategies including:
- Rule-based scoring for objective metrics
- LLM-as-judge scoring for subjective quality
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Score:
    """A single score dimension.
    
    Attributes:
        dimension: Name of the score dimension
        value: Score value (0.0 to 1.0)
        confidence: Confidence in the score
        explanation: Explanation for the score
    """
    
    dimension: str
    value: float
    confidence: float = 1.0
    explanation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dimension": self.dimension,
            "value": self.value,
            "confidence": self.confidence,
            "explanation": self.explanation,
        }


@dataclass
class ScorerConfig:
    """Configuration for scorers."""
    
    # LLM model for judging
    model: str = "gpt-4"
    
    # Temperature for LLM
    temperature: float = 0.0
    
    # Maximum tokens for response
    max_tokens: int = 500
    
    # Scoring dimensions
    dimensions: List[str] = field(default_factory=lambda: [
        "relevance",
        "accuracy",
        "completeness",
        "helpfulness",
    ])
    
    # Weight for each dimension (must sum to 1)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "relevance": 0.3,
        "accuracy": 0.3,
        "completeness": 0.2,
        "helpfulness": 0.2,
    })


class Scorer(ABC):
    """Base class for scorers."""
    
    @abstractmethod
    async def score(
        self,
        query: str,
        response: str,
        ground_truth: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Score a response.
        
        Args:
            query: The user query
            response: The agent's response
            ground_truth: Optional ground truth answer
            context: Additional context for scoring
            
        Returns:
            Dictionary of score dimensions and values
        """
        pass


class RuleBasedScorer(Scorer):
    """Rule-based scorer for objective metrics.
    
    This scorer uses heuristics and rules to evaluate responses
    without requiring an LLM.
    """
    
    def __init__(self, config: Optional[ScorerConfig] = None):
        self.config = config or ScorerConfig()
    
    async def score(
        self,
        query: str,
        response: str,
        ground_truth: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Score using rule-based heuristics."""
        scores = {}
        context = context or {}
        
        # Length score: Not too short, not too long
        length = len(response)
        if length < 50:
            length_score = length / 50
        elif length > 2000:
            length_score = max(0.5, 1 - (length - 2000) / 5000)
        else:
            length_score = 1.0
        scores["length"] = length_score
        
        # Keyword overlap with query
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        if query_words:
            overlap = len(query_words & response_words) / len(query_words)
            scores["relevance"] = min(1.0, overlap * 2)  # Scale up
        
        # Ground truth comparison
        if ground_truth:
            gt_words = set(ground_truth.lower().split())
            if gt_words:
                overlap = len(gt_words & response_words) / len(gt_words)
                scores["accuracy"] = overlap
        
        # Structure score (has sentences, paragraphs)
        has_punctuation = any(c in response for c in ".!?")
        has_structure = "\n" in response or len(response.split(". ")) > 1
        scores["structure"] = 0.5 + (0.25 if has_punctuation else 0) + (0.25 if has_structure else 0)
        
        # Check for error indicators
        error_phrases = ["error", "failed", "unable to", "cannot", "don't know"]
        has_error = any(phrase in response.lower() for phrase in error_phrases)
        scores["success"] = 0.5 if has_error else 1.0
        
        # Expected tools check
        expected_tools = context.get("expected_tools", [])
        called_tools = context.get("called_tools", [])
        if expected_tools:
            called_set = set(called_tools)
            expected_set = set(expected_tools)
            if expected_set:
                recall = len(called_set & expected_set) / len(expected_set)
                scores["tool_accuracy"] = recall
        
        return scores


class LLMJudgeScorer(Scorer):
    """LLM-as-judge scorer for subjective quality.
    
    Uses an LLM to evaluate response quality across multiple dimensions.
    """
    
    def __init__(
        self,
        config: Optional[ScorerConfig] = None,
        llm_client: Any = None,
    ):
        self.config = config or ScorerConfig()
        self._llm_client = llm_client
    
    async def score(
        self,
        query: str,
        response: str,
        ground_truth: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Score using LLM-as-judge."""
        if self._llm_client is None:
            logger.warning("No LLM client configured, falling back to rule-based scoring")
            fallback = RuleBasedScorer(self.config)
            return await fallback.score(query, response, ground_truth, context)
        
        # Build prompt for LLM judge
        prompt = self._build_judge_prompt(query, response, ground_truth, context)
        
        try:
            # Call LLM
            llm_response = await self._call_llm(prompt)
            
            # Parse scores from response
            scores = self._parse_scores(llm_response)
            
            return scores
            
        except Exception as e:
            logger.error(f"LLM scoring failed: {e}")
            fallback = RuleBasedScorer(self.config)
            return await fallback.score(query, response, ground_truth, context)
    
    def _build_judge_prompt(
        self,
        query: str,
        response: str,
        ground_truth: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Build the prompt for the LLM judge."""
        dimensions_text = "\n".join(
            f"- {dim}: Rate 0-10" for dim in self.config.dimensions
        )
        
        ground_truth_section = ""
        if ground_truth:
            ground_truth_section = f"""
Reference Answer (ground truth):
{ground_truth}

"""

        return f"""You are an expert evaluator for a bioinformatics assistant.
Rate the following response on these dimensions:

{dimensions_text}

User Query:
{query}

{ground_truth_section}Assistant Response:
{response}

Provide your evaluation as JSON with this format:
{{
    "relevance": {{"score": <0-10>, "explanation": "<brief explanation>"}},
    "accuracy": {{"score": <0-10>, "explanation": "<brief explanation>"}},
    "completeness": {{"score": <0-10>, "explanation": "<brief explanation>"}},
    "helpfulness": {{"score": <0-10>, "explanation": "<brief explanation>"}}
}}

Be critical but fair. Focus on:
- Relevance: Does the response address the query?
- Accuracy: Is the information correct?
- Completeness: Does it cover all aspects?
- Helpfulness: Would this actually help the user?
"""
    
    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the prompt."""
        if hasattr(self._llm_client, 'generate'):
            return await self._llm_client.generate(
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        elif hasattr(self._llm_client, 'complete'):
            return await self._llm_client.complete(prompt)
        else:
            raise ValueError("LLM client does not have generate or complete method")
    
    def _parse_scores(self, llm_response: str) -> Dict[str, float]:
        """Parse scores from LLM response."""
        import json
        import re
        
        scores = {}
        
        # Try to extract JSON from response
        json_match = re.search(r'\{[^}]+\}', llm_response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                
                for dim in self.config.dimensions:
                    if dim in data:
                        if isinstance(data[dim], dict):
                            score = data[dim].get("score", 5) / 10.0
                        else:
                            score = float(data[dim]) / 10.0
                        scores[dim] = min(1.0, max(0.0, score))
                        
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to find scores in text
        if not scores:
            for dim in self.config.dimensions:
                pattern = rf'{dim}[:\s]+(\d+(?:\.\d+)?)'
                match = re.search(pattern, llm_response, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    if score > 1:
                        score = score / 10.0
                    scores[dim] = min(1.0, max(0.0, score))
        
        # Fill missing dimensions with 0.5
        for dim in self.config.dimensions:
            if dim not in scores:
                scores[dim] = 0.5
        
        return scores
    
    def compute_weighted_score(self, scores: Dict[str, float]) -> float:
        """Compute weighted overall score."""
        total = 0.0
        weight_sum = 0.0
        
        for dim, weight in self.config.weights.items():
            if dim in scores:
                total += scores[dim] * weight
                weight_sum += weight
        
        if weight_sum == 0:
            return 0.0
        
        return total / weight_sum


class CompositeScorer(Scorer):
    """Combines multiple scorers with configurable weights."""
    
    def __init__(
        self,
        scorers: List[tuple[Scorer, float]],  # (scorer, weight)
    ):
        """Initialize with list of (scorer, weight) tuples."""
        self.scorers = scorers
        
        # Normalize weights
        total_weight = sum(w for _, w in scorers)
        self.scorers = [(s, w / total_weight) for s, w in scorers]
    
    async def score(
        self,
        query: str,
        response: str,
        ground_truth: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Score using all scorers and combine."""
        combined_scores: Dict[str, List[tuple[float, float]]] = {}
        
        for scorer, weight in self.scorers:
            try:
                scores = await scorer.score(query, response, ground_truth, context)
                
                for dim, value in scores.items():
                    if dim not in combined_scores:
                        combined_scores[dim] = []
                    combined_scores[dim].append((value, weight))
                    
            except Exception as e:
                logger.warning(f"Scorer failed: {e}")
        
        # Compute weighted averages
        final_scores = {}
        for dim, values in combined_scores.items():
            total = sum(v * w for v, w in values)
            weight_sum = sum(w for _, w in values)
            final_scores[dim] = total / weight_sum if weight_sum > 0 else 0.5
        
        return final_scores


# Factory function
def create_scorer(
    config: Optional[ScorerConfig] = None,
    llm_client: Any = None,
    use_llm: bool = True,
) -> Scorer:
    """Create an appropriate scorer.
    
    Args:
        config: Scorer configuration
        llm_client: LLM client for LLM-based scoring
        use_llm: Whether to use LLM scoring
        
    Returns:
        Appropriate Scorer instance
    """
    config = config or ScorerConfig()
    
    if use_llm and llm_client:
        # Use composite with both rule-based and LLM
        return CompositeScorer([
            (RuleBasedScorer(config), 0.3),
            (LLMJudgeScorer(config, llm_client), 0.7),
        ])
    else:
        return RuleBasedScorer(config)


__all__ = [
    "Score",
    "ScorerConfig",
    "Scorer",
    "RuleBasedScorer",
    "LLMJudgeScorer",
    "CompositeScorer",
    "create_scorer",
]
