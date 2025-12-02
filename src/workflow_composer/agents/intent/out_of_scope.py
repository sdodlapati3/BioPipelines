"""
Out-of-Scope Detection for Chat Agent.

Phase 7 of Professional Chat Agent implementation.

Features:
- Topic boundary detection
- Graceful deflection responses
- Redirect suggestions
- Confidence-based handling
- Domain knowledge integration
"""

import logging
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Pattern,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


class ScopeCategory(Enum):
    """Category of scope classification."""
    IN_SCOPE = "in_scope"
    PARTIALLY_IN_SCOPE = "partially_in_scope"
    OUT_OF_SCOPE = "out_of_scope"
    AMBIGUOUS = "ambiguous"
    SENSITIVE = "sensitive"
    HARMFUL = "harmful"


class DeflectionStrategy(Enum):
    """Strategy for handling out-of-scope queries."""
    POLITE_DECLINE = "polite_decline"
    REDIRECT = "redirect"
    PARTIAL_ANSWER = "partial_answer"
    REQUEST_CLARIFICATION = "request_clarification"
    SUGGEST_ALTERNATIVE = "suggest_alternative"
    ESCALATE = "escalate"


@dataclass
class Topic:
    """Represents a topic domain."""
    name: str
    description: str = ""
    keywords: Set[str] = field(default_factory=set)
    patterns: List[str] = field(default_factory=list)
    subtopics: List["Topic"] = field(default_factory=list)
    parent: Optional["Topic"] = None
    is_sensitive: bool = False
    
    def matches(self, text: str) -> Tuple[bool, float]:
        """
        Check if text matches this topic.
        
        Returns:
            Tuple of (matches, confidence)
        """
        text_lower = text.lower()
        
        # Check keywords
        keyword_matches = sum(1 for kw in self.keywords if kw.lower() in text_lower)
        keyword_score = keyword_matches / len(self.keywords) if self.keywords else 0
        
        # Check patterns
        pattern_matches = 0
        for pattern in self.patterns:
            try:
                if re.search(pattern, text_lower):
                    pattern_matches += 1
            except re.error:
                continue
        pattern_score = pattern_matches / len(self.patterns) if self.patterns else 0
        
        # Combined score
        if self.keywords and self.patterns:
            score = (keyword_score + pattern_score) / 2
        else:
            score = keyword_score or pattern_score
        
        return score > 0.1, score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "keywords": list(self.keywords),
            "patterns": self.patterns,
            "subtopics": [st.to_dict() for st in self.subtopics],
            "is_sensitive": self.is_sensitive,
        }


@dataclass
class ScopeResult:
    """Result of scope classification."""
    category: ScopeCategory
    confidence: float
    matched_topics: List[Topic] = field(default_factory=list)
    reason: str = ""
    suggested_strategy: DeflectionStrategy = DeflectionStrategy.POLITE_DECLINE
    redirect_topic: Optional[Topic] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_in_scope(self) -> bool:
        """Check if query is in scope."""
        return self.category in [ScopeCategory.IN_SCOPE, ScopeCategory.PARTIALLY_IN_SCOPE]
    
    @property
    def needs_deflection(self) -> bool:
        """Check if query needs deflection."""
        return self.category in [
            ScopeCategory.OUT_OF_SCOPE,
            ScopeCategory.SENSITIVE,
            ScopeCategory.HARMFUL
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "confidence": self.confidence,
            "matched_topics": [t.to_dict() for t in self.matched_topics],
            "reason": self.reason,
            "suggested_strategy": self.suggested_strategy.value,
            "redirect_topic": self.redirect_topic.to_dict() if self.redirect_topic else None,
            "is_in_scope": self.is_in_scope,
            "needs_deflection": self.needs_deflection,
            "metadata": self.metadata,
        }


@dataclass
class DeflectionResponse:
    """Response for deflection."""
    message: str
    suggestions: List[str] = field(default_factory=list)
    redirect_url: str = ""
    escalate: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message": self.message,
            "suggestions": self.suggestions,
            "redirect_url": self.redirect_url,
            "escalate": self.escalate,
        }


class ScopeClassifier(ABC):
    """Abstract base class for scope classifiers."""
    
    @abstractmethod
    def classify(self, query: str, context: Optional[Dict[str, Any]] = None) -> ScopeResult:
        """Classify a query's scope."""
        pass


class KeywordScopeClassifier(ScopeClassifier):
    """Keyword-based scope classifier."""
    
    def __init__(self, in_scope_topics: List[Topic]):
        self.in_scope_topics = in_scope_topics
    
    def classify(self, query: str, context: Optional[Dict[str, Any]] = None) -> ScopeResult:
        """Classify based on keyword matching."""
        matched_topics = []
        max_confidence = 0.0
        
        for topic in self.in_scope_topics:
            matches, confidence = topic.matches(query)
            if matches:
                matched_topics.append(topic)
                max_confidence = max(max_confidence, confidence)
        
        if matched_topics:
            # Check for sensitive topics
            if any(t.is_sensitive for t in matched_topics):
                return ScopeResult(
                    category=ScopeCategory.SENSITIVE,
                    confidence=max_confidence,
                    matched_topics=matched_topics,
                    reason="Query touches on sensitive topic",
                    suggested_strategy=DeflectionStrategy.ESCALATE
                )
            
            if max_confidence > 0.5:
                return ScopeResult(
                    category=ScopeCategory.IN_SCOPE,
                    confidence=max_confidence,
                    matched_topics=matched_topics,
                    reason="Query matches in-scope topics"
                )
            else:
                return ScopeResult(
                    category=ScopeCategory.PARTIALLY_IN_SCOPE,
                    confidence=max_confidence,
                    matched_topics=matched_topics,
                    reason="Query partially matches in-scope topics",
                    suggested_strategy=DeflectionStrategy.REQUEST_CLARIFICATION
                )
        
        return ScopeResult(
            category=ScopeCategory.OUT_OF_SCOPE,
            confidence=1.0 - max_confidence,
            reason="Query does not match any in-scope topics",
            suggested_strategy=DeflectionStrategy.POLITE_DECLINE
        )


class PatternScopeClassifier(ScopeClassifier):
    """Pattern-based scope classifier using regex."""
    
    def __init__(self):
        self._in_scope_patterns: List[Tuple[Pattern, float]] = []
        self._out_scope_patterns: List[Tuple[Pattern, str]] = []
        self._harmful_patterns: List[Tuple[Pattern, str]] = []
    
    def add_in_scope_pattern(self, pattern: str, weight: float = 1.0) -> None:
        """Add an in-scope pattern."""
        self._in_scope_patterns.append((re.compile(pattern, re.IGNORECASE), weight))
    
    def add_out_scope_pattern(self, pattern: str, reason: str = "") -> None:
        """Add an out-of-scope pattern."""
        self._out_scope_patterns.append((re.compile(pattern, re.IGNORECASE), reason))
    
    def add_harmful_pattern(self, pattern: str, reason: str = "") -> None:
        """Add a harmful content pattern."""
        self._harmful_patterns.append((re.compile(pattern, re.IGNORECASE), reason))
    
    def classify(self, query: str, context: Optional[Dict[str, Any]] = None) -> ScopeResult:
        """Classify based on pattern matching."""
        # Check harmful patterns first
        for pattern, reason in self._harmful_patterns:
            if pattern.search(query):
                return ScopeResult(
                    category=ScopeCategory.HARMFUL,
                    confidence=1.0,
                    reason=reason or "Query contains potentially harmful content",
                    suggested_strategy=DeflectionStrategy.POLITE_DECLINE
                )
        
        # Check out-of-scope patterns
        for pattern, reason in self._out_scope_patterns:
            if pattern.search(query):
                return ScopeResult(
                    category=ScopeCategory.OUT_OF_SCOPE,
                    confidence=0.9,
                    reason=reason or "Query matches out-of-scope pattern",
                    suggested_strategy=DeflectionStrategy.REDIRECT
                )
        
        # Check in-scope patterns
        total_weight = 0.0
        matched_weight = 0.0
        
        for pattern, weight in self._in_scope_patterns:
            total_weight += weight
            if pattern.search(query):
                matched_weight += weight
        
        if total_weight == 0:
            return ScopeResult(
                category=ScopeCategory.AMBIGUOUS,
                confidence=0.5,
                reason="No patterns configured"
            )
        
        confidence = matched_weight / total_weight
        
        if confidence > 0.5:
            return ScopeResult(
                category=ScopeCategory.IN_SCOPE,
                confidence=confidence,
                reason="Query matches in-scope patterns"
            )
        elif confidence > 0.2:
            return ScopeResult(
                category=ScopeCategory.PARTIALLY_IN_SCOPE,
                confidence=confidence,
                reason="Query partially matches in-scope patterns",
                suggested_strategy=DeflectionStrategy.PARTIAL_ANSWER
            )
        else:
            return ScopeResult(
                category=ScopeCategory.OUT_OF_SCOPE,
                confidence=1 - confidence,
                reason="Query does not match in-scope patterns",
                suggested_strategy=DeflectionStrategy.SUGGEST_ALTERNATIVE
            )


class EnsembleScopeClassifier(ScopeClassifier):
    """Ensemble of multiple classifiers."""
    
    def __init__(self):
        self._classifiers: List[Tuple[ScopeClassifier, float]] = []
    
    def add_classifier(self, classifier: ScopeClassifier, weight: float = 1.0) -> None:
        """Add a classifier with a weight."""
        self._classifiers.append((classifier, weight))
    
    def classify(self, query: str, context: Optional[Dict[str, Any]] = None) -> ScopeResult:
        """Combine results from all classifiers."""
        if not self._classifiers:
            return ScopeResult(
                category=ScopeCategory.AMBIGUOUS,
                confidence=0.0,
                reason="No classifiers configured"
            )
        
        results: List[Tuple[ScopeResult, float]] = []
        total_weight = 0.0
        
        for classifier, weight in self._classifiers:
            result = classifier.classify(query, context)
            results.append((result, weight))
            total_weight += weight
        
        # Vote on category
        category_scores: Dict[ScopeCategory, float] = {}
        for result, weight in results:
            normalized_weight = weight / total_weight
            category_scores[result.category] = (
                category_scores.get(result.category, 0) + 
                result.confidence * normalized_weight
            )
        
        # Find winning category
        best_category = max(category_scores, key=category_scores.get)
        
        # Calculate aggregate confidence
        avg_confidence = sum(r.confidence * w for r, w in results) / total_weight
        
        # Collect all matched topics
        all_topics = []
        for result, _ in results:
            all_topics.extend(result.matched_topics)
        
        # Determine strategy
        strategies = [r.suggested_strategy for r, _ in results]
        strategy = max(set(strategies), key=strategies.count) if strategies else DeflectionStrategy.POLITE_DECLINE
        
        return ScopeResult(
            category=best_category,
            confidence=avg_confidence,
            matched_topics=list(set(all_topics)),
            reason=f"Ensemble decision with {len(self._classifiers)} classifiers",
            suggested_strategy=strategy
        )


class DeflectionResponseGenerator:
    """Generates deflection responses based on scope results."""
    
    def __init__(self):
        self._templates: Dict[ScopeCategory, List[str]] = {
            ScopeCategory.OUT_OF_SCOPE: [
                "I specialize in {domain}, and this query seems outside my area of expertise.",
                "I'm not able to help with that particular topic. I focus on {domain}.",
                "That's outside my wheelhouse. I'm here to help with {domain}.",
            ],
            ScopeCategory.SENSITIVE: [
                "This topic requires careful handling. Let me connect you with a specialist.",
                "I want to make sure you get the best help. Let me escalate this.",
                "This is a sensitive area. I'll connect you with someone who can help.",
            ],
            ScopeCategory.HARMFUL: [
                "I'm not able to assist with that request.",
                "I cannot help with that type of request.",
                "This isn't something I can assist with.",
            ],
            ScopeCategory.AMBIGUOUS: [
                "I'm not sure I understand. Could you tell me more about what you need?",
                "Could you clarify what you're looking for? I want to make sure I can help.",
                "I'd like to help, but I need a bit more context. What specifically are you trying to do?",
            ],
        }
        
        self._strategy_templates: Dict[DeflectionStrategy, str] = {
            DeflectionStrategy.POLITE_DECLINE: "I appreciate you reaching out, but {reason}",
            DeflectionStrategy.REDIRECT: "While I can't help with that directly, you might try {redirect}.",
            DeflectionStrategy.PARTIAL_ANSWER: "I can partially help with this. {partial}",
            DeflectionStrategy.REQUEST_CLARIFICATION: "Could you help me understand better? {clarification}",
            DeflectionStrategy.SUGGEST_ALTERNATIVE: "Instead, I could help you with {alternative}.",
            DeflectionStrategy.ESCALATE: "Let me connect you with someone who can better assist.",
        }
        
        self._domain = "bioinformatics workflows and analyses"
        self._suggestions_by_topic: Dict[str, List[str]] = {}
    
    def set_domain(self, domain: str) -> None:
        """Set the domain description."""
        self._domain = domain
    
    def add_suggestions(self, topic: str, suggestions: List[str]) -> None:
        """Add suggestions for a topic."""
        self._suggestions_by_topic[topic] = suggestions
    
    def add_template(self, category: ScopeCategory, template: str) -> None:
        """Add a custom template for a category."""
        if category not in self._templates:
            self._templates[category] = []
        self._templates[category].append(template)
    
    def generate(self, scope_result: ScopeResult) -> DeflectionResponse:
        """Generate a deflection response."""
        import random
        
        templates = self._templates.get(scope_result.category, [])
        if not templates:
            templates = ["I'm not able to help with that request."]
        
        template = random.choice(templates)
        message = template.format(domain=self._domain)
        
        # Add strategy-specific content
        strategy_template = self._strategy_templates.get(scope_result.suggested_strategy, "")
        if strategy_template:
            if scope_result.suggested_strategy == DeflectionStrategy.REDIRECT:
                redirect = scope_result.redirect_topic.name if scope_result.redirect_topic else "our documentation"
                message += " " + strategy_template.format(redirect=redirect)
        
        # Generate suggestions
        suggestions = []
        for topic in scope_result.matched_topics:
            topic_suggestions = self._suggestions_by_topic.get(topic.name, [])
            suggestions.extend(topic_suggestions[:2])
        
        # Add default suggestions if none found
        if not suggestions:
            suggestions = [
                "Try asking about creating a workflow",
                "Ask about available analysis types",
                "Get help with data processing",
            ]
        
        return DeflectionResponse(
            message=message,
            suggestions=suggestions[:5],
            escalate=scope_result.suggested_strategy == DeflectionStrategy.ESCALATE
        )


class OutOfScopeHandler:
    """
    Main handler for out-of-scope detection and response.
    
    Features:
    - Multiple classification strategies
    - Topic management
    - Response generation
    - Logging and analytics
    """
    
    def __init__(
        self,
        classifier: Optional[ScopeClassifier] = None,
        response_generator: Optional[DeflectionResponseGenerator] = None
    ):
        self.classifier = classifier or self._create_default_classifier()
        self.response_generator = response_generator or DeflectionResponseGenerator()
        
        # Statistics
        self._stats: Dict[str, int] = {
            "total_queries": 0,
            "in_scope": 0,
            "out_of_scope": 0,
            "deflected": 0,
        }
        
        # History
        self._history: List[Tuple[str, ScopeResult]] = []
        self._max_history = 1000
        
        # Callbacks
        self._on_out_of_scope: List[Callable[[str, ScopeResult], None]] = []
        
        self._lock = threading.Lock()
    
    def _create_default_classifier(self) -> ScopeClassifier:
        """Create default classifier with bioinformatics topics."""
        bioinformatics_topic = Topic(
            name="bioinformatics",
            description="Bioinformatics analysis and workflows",
            keywords={
                "rna-seq", "dna-seq", "chip-seq", "atac-seq", "methylation",
                "genome", "transcriptome", "proteome", "sequence", "alignment",
                "variant", "mutation", "gene", "expression", "differential",
                "fastq", "bam", "vcf", "gtf", "gff", "fasta",
                "workflow", "pipeline", "analysis", "processing",
                "nf-core", "nextflow", "snakemake", "bioconductor",
            },
            patterns=[
                r"rna[- ]?seq",
                r"dna[- ]?seq",
                r"chip[- ]?seq",
                r"variant\s+call",
                r"gene\s+expression",
                r"differential\s+expression",
                r"sequence\s+align",
                r"genome\s+assembl",
                r"(create|run|build)\s+(a\s+)?(workflow|pipeline)",
            ]
        )
        
        return KeywordScopeClassifier([bioinformatics_topic])
    
    def add_in_scope_topic(self, topic: Topic) -> None:
        """Add an in-scope topic."""
        if isinstance(self.classifier, KeywordScopeClassifier):
            self.classifier.in_scope_topics.append(topic)
    
    def check_scope(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ScopeResult:
        """
        Check if a query is in scope.
        
        Args:
            query: The user query
            context: Optional context information
        
        Returns:
            ScopeResult with classification
        """
        result = self.classifier.classify(query, context)
        
        with self._lock:
            self._stats["total_queries"] += 1
            if result.is_in_scope:
                self._stats["in_scope"] += 1
            else:
                self._stats["out_of_scope"] += 1
            
            if result.needs_deflection:
                self._stats["deflected"] += 1
                for callback in self._on_out_of_scope:
                    try:
                        callback(query, result)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
            
            # Record history
            self._history.append((query, result))
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
        
        return result
    
    def handle_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[ScopeResult, Optional[DeflectionResponse]]:
        """
        Check scope and generate response if needed.
        
        Args:
            query: The user query
            context: Optional context
        
        Returns:
            Tuple of (ScopeResult, DeflectionResponse or None)
        """
        result = self.check_scope(query, context)
        
        if result.needs_deflection:
            response = self.response_generator.generate(result)
            return result, response
        
        return result, None
    
    def on_out_of_scope(self, callback: Callable[[str, ScopeResult], None]) -> None:
        """Register callback for out-of-scope queries."""
        self._on_out_of_scope.append(callback)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics."""
        with self._lock:
            total = self._stats["total_queries"]
            return {
                **self._stats,
                "in_scope_rate": self._stats["in_scope"] / total if total else 0,
                "out_of_scope_rate": self._stats["out_of_scope"] / total if total else 0,
                "deflection_rate": self._stats["deflected"] / total if total else 0,
            }
    
    def get_history(self, limit: int = 100) -> List[Tuple[str, ScopeResult]]:
        """Get recent classification history."""
        with self._lock:
            return self._history[-limit:]
    
    def get_common_out_of_scope(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get most common out-of-scope query patterns."""
        out_of_scope = [
            query for query, result in self._history
            if result.category == ScopeCategory.OUT_OF_SCOPE
        ]
        
        # Simple frequency count
        from collections import Counter
        return Counter(out_of_scope).most_common(limit)


class DomainKnowledge:
    """
    Domain knowledge base for scope detection.
    
    Helps determine what's in-scope based on domain expertise.
    """
    
    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self._topics: Dict[str, Topic] = {}
        self._capabilities: Set[str] = set()
        self._limitations: Set[str] = set()
        self._related_domains: Dict[str, str] = {}  # domain -> redirect info
    
    def add_topic(self, topic: Topic) -> None:
        """Add a topic to domain knowledge."""
        self._topics[topic.name] = topic
    
    def get_topic(self, name: str) -> Optional[Topic]:
        """Get a topic by name."""
        return self._topics.get(name)
    
    def get_all_topics(self) -> List[Topic]:
        """Get all topics."""
        return list(self._topics.values())
    
    def add_capability(self, capability: str) -> None:
        """Add a capability."""
        self._capabilities.add(capability)
    
    def add_limitation(self, limitation: str) -> None:
        """Add a limitation."""
        self._limitations.add(limitation)
    
    def add_related_domain(self, domain: str, redirect_info: str) -> None:
        """Add a related domain with redirect info."""
        self._related_domains[domain] = redirect_info
    
    def can_handle(self, capability: str) -> bool:
        """Check if capability is supported."""
        return capability.lower() in {c.lower() for c in self._capabilities}
    
    def get_redirect_suggestion(self, domain: str) -> Optional[str]:
        """Get redirect suggestion for a related domain."""
        return self._related_domains.get(domain)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "domain_name": self.domain_name,
            "topics": {name: topic.to_dict() for name, topic in self._topics.items()},
            "capabilities": list(self._capabilities),
            "limitations": list(self._limitations),
            "related_domains": self._related_domains,
        }


class BioinformaticsDomainKnowledge(DomainKnowledge):
    """Pre-configured domain knowledge for bioinformatics."""
    
    def __init__(self):
        super().__init__("Bioinformatics")
        self._initialize()
    
    def _initialize(self):
        """Initialize with bioinformatics knowledge."""
        # Add topics
        self.add_topic(Topic(
            name="rna_seq",
            description="RNA sequencing analysis",
            keywords={"rna-seq", "rnaseq", "transcriptome", "differential expression", "deseq2", "edger"},
            patterns=[r"rna[- ]?seq", r"transcriptom", r"gene\s+expression"]
        ))
        
        self.add_topic(Topic(
            name="dna_seq",
            description="DNA sequencing and variant analysis",
            keywords={"dna-seq", "whole genome", "exome", "variant calling", "gatk", "mutation"},
            patterns=[r"dna[- ]?seq", r"variant\s+call", r"whole\s+genome"]
        ))
        
        self.add_topic(Topic(
            name="chip_seq",
            description="ChIP-seq analysis for protein-DNA interactions",
            keywords={"chip-seq", "chipseq", "peak calling", "histone", "transcription factor"},
            patterns=[r"chip[- ]?seq", r"peak\s+call"]
        ))
        
        self.add_topic(Topic(
            name="methylation",
            description="DNA methylation analysis",
            keywords={"methylation", "bisulfite", "wgbs", "rrbs", "methylome"},
            patterns=[r"methyl", r"bisulfite", r"cpg"]
        ))
        
        self.add_topic(Topic(
            name="metagenomics",
            description="Metagenomic analysis",
            keywords={"metagenomics", "microbiome", "16s", "taxonomy", "kraken"},
            patterns=[r"metagenom", r"microbiome", r"16s"]
        ))
        
        self.add_topic(Topic(
            name="workflows",
            description="Workflow management",
            keywords={"workflow", "pipeline", "nextflow", "snakemake", "nf-core"},
            patterns=[r"workflow", r"pipeline", r"nextflow", r"snakemake"]
        ))
        
        # Add capabilities
        capabilities = [
            "create workflows",
            "run analyses",
            "process sequencing data",
            "generate reports",
            "quality control",
            "alignment",
            "variant calling",
            "differential expression",
            "peak calling",
        ]
        for cap in capabilities:
            self.add_capability(cap)
        
        # Add limitations
        limitations = [
            "clinical diagnosis",
            "medical advice",
            "financial analysis",
            "general programming",
            "non-biological data",
        ]
        for lim in limitations:
            self.add_limitation(lim)
        
        # Add related domains
        self.add_related_domain("medical", "Please consult a healthcare professional")
        self.add_related_domain("programming", "For general programming, try Stack Overflow or documentation")


# Singleton pattern
_handler: Optional[OutOfScopeHandler] = None
_handler_lock = threading.Lock()


def get_out_of_scope_handler() -> OutOfScopeHandler:
    """Get the singleton out-of-scope handler."""
    global _handler
    if _handler is None:
        with _handler_lock:
            if _handler is None:
                _handler = OutOfScopeHandler()
    return _handler


def reset_out_of_scope_handler() -> None:
    """Reset the singleton handler."""
    global _handler
    with _handler_lock:
        _handler = None


# Convenience function
def check_query_scope(
    query: str,
    context: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Optional[str]]:
    """
    Quick check if a query is in scope.
    
    Args:
        query: The user query
        context: Optional context
    
    Returns:
        Tuple of (is_in_scope, deflection_message if out of scope)
    """
    handler = get_out_of_scope_handler()
    result, response = handler.handle_query(query, context)
    
    return result.is_in_scope, response.message if response else None


__all__ = [
    # Enums
    "ScopeCategory",
    "DeflectionStrategy",
    
    # Data classes
    "Topic",
    "ScopeResult",
    "DeflectionResponse",
    
    # Classifiers
    "ScopeClassifier",
    "KeywordScopeClassifier",
    "PatternScopeClassifier",
    "EnsembleScopeClassifier",
    
    # Response generation
    "DeflectionResponseGenerator",
    
    # Main handler
    "OutOfScopeHandler",
    
    # Domain knowledge
    "DomainKnowledge",
    "BioinformaticsDomainKnowledge",
    
    # Singleton
    "get_out_of_scope_handler",
    "reset_out_of_scope_handler",
    
    # Convenience
    "check_query_scope",
]
