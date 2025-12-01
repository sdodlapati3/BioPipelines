"""
Unified Ensemble Query Parser (DEPRECATED)
==========================================

⚠️ DEPRECATED: Use UnifiedIntentParser instead.

This module is kept for backward compatibility only.
For new code, use:

    from workflow_composer.agents.intent import UnifiedIntentParser
    parser = UnifiedIntentParser()
    result = parser.parse("your query")

UnifiedIntentParser provides:
- Better accuracy (87%+ vs 80%)
- Hierarchical parsing with LLM arbiter
- Decision caching for performance
- Rate-limit resistant provider cascade

---
Original Description:

A robust query parsing system that combines multiple methods with
confidence-weighted voting for the most accurate intent detection.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    UnifiedEnsembleParser                        │
    │  ┌──────────┬──────────┬──────────┬──────────┬──────────────┐  │
    │  │ Rule     │ Semantic │ NER      │ LLM      │ RAG          │  │
    │  │ Patterns │ FAISS    │ BioBERT  │ Fallback │ History      │  │
    │  │ (0.25)   │ (0.30)   │ (0.20)   │ (0.15)   │ (0.10)       │  │
    │  └──────────┴──────────┴──────────┴──────────┴──────────────┘  │
    │                          ↓                                      │
    │              Confidence-Weighted Fusion                         │
    │              + Agreement Boosting                               │
    │                          ↓                                      │
    │               Final Intent + Confidence                         │
    └─────────────────────────────────────────────────────────────────┘

Robustness Features:
1. Graceful degradation - works even if components fail
2. Agreement boosting - confidence increases when methods agree
3. Disambiguation - handles ambiguous queries with clarification
4. Learning - RAG system learns from past successful queries
5. Multi-language - patterns handle common variations

Usage:
    from workflow_composer.agents.intent.unified_ensemble import UnifiedEnsembleParser
    
    parser = UnifiedEnsembleParser()
    result = parser.parse("search ENCODE for human liver ChIP-seq H3K27ac")
    
    print(f"Intent: {result.intent}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Method: {result.winning_method}")
    print(f"Entities: {result.entities}")
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import time

# Import Phase 1 components
try:
    from .negation_handler import NegationHandler, NegationResult, NegationType
    NEGATION_AVAILABLE = True
except ImportError:
    NEGATION_AVAILABLE = False
    NegationResult = None

logger = logging.getLogger(__name__)


class ParsingMethod(Enum):
    """Available parsing methods."""
    RULE_PATTERN = "rule_pattern"
    SEMANTIC_FAISS = "semantic_faiss"
    NER_ENTITY = "ner_entity"
    LLM_GENERATION = "llm_generation"
    RAG_HISTORY = "rag_history"
    ENSEMBLE_VOTE = "ensemble_vote"


@dataclass
class MethodVote:
    """A vote from a single parsing method."""
    method: ParsingMethod
    intent: str
    confidence: float
    weight: float  # Method's base weight in ensemble
    entities: List[Any] = field(default_factory=list)
    slots: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: Optional[str] = None
    
    @property
    def weighted_score(self) -> float:
        """Calculate weighted confidence score."""
        return self.confidence * self.weight


@dataclass
class EnsembleParseResult:
    """Result from ensemble parsing with full voting details."""
    # Primary result
    intent: str
    confidence: float
    
    # Alternative interpretations
    alternatives: List[Tuple[str, float]] = field(default_factory=list)
    
    # Entity extraction (combined from all methods)
    entities: List[Any] = field(default_factory=list)
    slots: Dict[str, Any] = field(default_factory=dict)
    
    # Voting details
    votes: List[MethodVote] = field(default_factory=list)
    winning_method: ParsingMethod = ParsingMethod.ENSEMBLE_VOTE
    agreement_level: float = 0.0  # 0-1, how much methods agreed
    
    # Query understanding
    original_query: str = ""
    normalized_query: str = ""
    
    # Negation handling (Phase 1 improvement)
    negation_result: Optional[Any] = None  # NegationResult when available
    excluded_terms: List[str] = field(default_factory=list)
    preferred_terms: List[str] = field(default_factory=list)
    
    # Clarification
    needs_clarification: bool = False
    clarification_prompt: Optional[str] = None
    
    # Performance
    total_latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "alternatives": self.alternatives,
            "entities": [
                {"text": e.text, "type": e.entity_type, "canonical": getattr(e, 'canonical', e.text)}
                for e in self.entities
            ],
            "slots": self.slots,
            "winning_method": self.winning_method.value,
            "agreement_level": self.agreement_level,
            "needs_clarification": self.needs_clarification,
            "latency_ms": self.total_latency_ms,
            "excluded_terms": self.excluded_terms,
            "preferred_terms": self.preferred_terms,
        }


class UnifiedEnsembleParser:
    """
    Production-grade ensemble parser combining multiple NLU methods.
    
    ⚠️ DEPRECATED: Use UnifiedIntentParser instead for better accuracy.
    
    Features:
    - Weighted voting across 5 parsing methods
    - Agreement boosting when methods concur
    - Graceful degradation if methods fail
    - Confidence calibration based on query complexity
    - Learning from RAG history
    
    Migration:
        # Old:
        parser = UnifiedEnsembleParser()
        result = parser.parse(query)
        
        # New (RECOMMENDED):
        from workflow_composer.agents.intent import UnifiedIntentParser
        parser = UnifiedIntentParser()
        result = parser.parse(query)
    """
    
    # Default weights for each method (must sum to 1.0)
    DEFAULT_WEIGHTS = {
        ParsingMethod.RULE_PATTERN: 0.25,    # Fast, precise for known patterns
        ParsingMethod.SEMANTIC_FAISS: 0.30,  # Handles paraphrases
        ParsingMethod.NER_ENTITY: 0.20,      # Domain-specific entities
        ParsingMethod.LLM_GENERATION: 0.15,  # Complex disambiguation
        ParsingMethod.RAG_HISTORY: 0.10,     # Learn from past queries
    }
    
    # Agreement bonus (multiplied by agreement level)
    AGREEMENT_BONUS = 0.15
    
    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.75
    MEDIUM_CONFIDENCE_THRESHOLD = 0.50
    CLARIFICATION_THRESHOLD = 0.40
    
    def __init__(
        self,
        weights: Optional[Dict[ParsingMethod, float]] = None,
        llm_client=None,
        enable_rag: bool = True,
        enable_llm_fallback: bool = True,
        enable_negation_handling: bool = True,
    ):
        """
        Initialize the ensemble parser.
        
        Args:
            weights: Custom weights for each method (must sum to 1.0)
            llm_client: LLM client for generation fallback
            enable_rag: Enable RAG-based history learning
            enable_llm_fallback: Enable LLM for complex queries
            enable_negation_handling: Enable negation detection (Phase 1)
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.llm_client = llm_client
        self.enable_rag = enable_rag
        self.enable_llm_fallback = enable_llm_fallback
        self.enable_negation_handling = enable_negation_handling
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        # Initialize components (lazy loading)
        self._pattern_parser = None
        self._semantic_classifier = None
        self._ner = None
        self._rag_orchestrator = None
        self._negation_handler = None
        
        import warnings
        warnings.warn(
            "UnifiedEnsembleParser is deprecated. Use UnifiedIntentParser instead.",
            DeprecationWarning,
            stacklevel=2
        )
        logger.info("UnifiedEnsembleParser initialized (DEPRECATED - use UnifiedIntentParser)")
        logger.info(f"  Weights: {self.weights}")
        logger.info(f"  Negation handling: {enable_negation_handling and NEGATION_AVAILABLE}")
    
    @property
    def negation_handler(self):
        """Lazy load negation handler."""
        if self._negation_handler is None and NEGATION_AVAILABLE:
            try:
                self._negation_handler = NegationHandler()
            except Exception as e:
                logger.warning(f"Failed to load negation handler: {e}")
        return self._negation_handler
    
    @property
    def pattern_parser(self):
        """Lazy load pattern parser."""
        if self._pattern_parser is None:
            try:
                from .parser import IntentParser
                self._pattern_parser = IntentParser(llm_client=self.llm_client)
            except Exception as e:
                logger.warning(f"Failed to load pattern parser: {e}")
        return self._pattern_parser
    
    @property
    def semantic_classifier(self):
        """Lazy load semantic classifier."""
        if self._semantic_classifier is None:
            try:
                from .semantic import SemanticIntentClassifier
                self._semantic_classifier = SemanticIntentClassifier(
                    llm_client=self.llm_client
                )
            except Exception as e:
                logger.warning(f"Failed to load semantic classifier: {e}")
        return self._semantic_classifier
    
    @property
    def ner(self):
        """Lazy load NER."""
        if self._ner is None:
            try:
                from .semantic import BioinformaticsNER
                self._ner = BioinformaticsNER()
            except Exception as e:
                logger.warning(f"Failed to load NER: {e}")
        return self._ner
    
    @property
    def rag_orchestrator(self):
        """Lazy load RAG orchestrator."""
        if self._rag_orchestrator is None and self.enable_rag:
            try:
                from ..rag import get_rag_orchestrator
                self._rag_orchestrator = get_rag_orchestrator()
            except Exception as e:
                logger.debug(f"RAG orchestrator not available: {e}")
        return self._rag_orchestrator
    
    def parse(
        self,
        query: str,
        context: Optional[Dict] = None,
        parallel: bool = True,
    ) -> EnsembleParseResult:
        """
        Parse a query using ensemble of methods.
        
        Args:
            query: User query
            context: Optional conversation context
            parallel: Run methods in parallel (faster but more resources)
            
        Returns:
            EnsembleParseResult with voting details
        """
        start_time = time.time()
        query = query.strip()
        
        if not query:
            return EnsembleParseResult(
                intent="META_UNKNOWN",
                confidence=0.0,
                needs_clarification=True,
                clarification_prompt="Please enter a query.",
            )
        
        # Phase 1: Negation detection (preprocess before parsing)
        negation_result = None
        excluded_terms = []
        preferred_terms = []
        
        if self.enable_negation_handling and self.negation_handler:
            try:
                negation_result = self.negation_handler.detect(query)
                if negation_result.has_negation:
                    excluded_terms = negation_result.negated_terms
                    preferred_terms = negation_result.preferred_terms
                    logger.debug(
                        f"Negation detected: type={negation_result.negation_type.value}, "
                        f"excluded={excluded_terms}, preferred={preferred_terms}"
                    )
            except Exception as e:
                logger.warning(f"Negation detection failed: {e}")
        
        # Normalize query
        normalized = self._normalize_query(query)
        
        # Collect votes from all methods
        votes = []
        
        # Method 1: Rule-based pattern matching
        vote = self._run_pattern_parser(query, context)
        if vote:
            votes.append(vote)
        
        # Method 2: Semantic similarity (FAISS)
        vote = self._run_semantic_classifier(query)
        if vote:
            votes.append(vote)
        
        # Method 3: NER-based inference
        vote = self._run_ner_inference(query)
        if vote:
            votes.append(vote)
        
        # Method 4: RAG history (learn from past queries)
        if self.enable_rag:
            vote = self._run_rag_lookup(query)
            if vote:
                votes.append(vote)
        
        # Method 5: LLM fallback (only for low-confidence cases)
        # Defer until we know other methods' confidence
        
        # Combine votes
        result = self._combine_votes(votes, query, normalized)
        
        # LLM fallback if confidence is too low
        if result.confidence < self.CLARIFICATION_THRESHOLD and self.enable_llm_fallback:
            llm_vote = self._run_llm_fallback(query, result)
            if llm_vote:
                votes.append(llm_vote)
                # Recombine with LLM vote
                result = self._combine_votes(votes, query, normalized)
        
        # Apply negation to filter entities (Phase 1 improvement)
        if negation_result and negation_result.has_negation and self.negation_handler:
            # Filter out negated entities from slots
            filtered_slots = self.negation_handler.apply_to_entities(
                result.slots, negation_result
            )
            result.slots = filtered_slots
            result.negation_result = negation_result
            result.excluded_terms = excluded_terms
            result.preferred_terms = preferred_terms
        
        # Calculate total latency
        result.total_latency_ms = (time.time() - start_time) * 1000
        result.original_query = query
        result.normalized_query = normalized
        result.votes = votes
        
        # Check if clarification needed
        if result.confidence < self.CLARIFICATION_THRESHOLD:
            result.needs_clarification = True
            result.clarification_prompt = self._generate_clarification(
                query, result.entities, result.alternatives
            )
        
        # Log with negation info if detected
        negation_info = ""
        if excluded_terms:
            negation_info = f", excluded={excluded_terms}"
        
        logger.info(
            f"Ensemble parse: intent={result.intent}, "
            f"confidence={result.confidence:.2f}, "
            f"method={result.winning_method.value}, "
            f"agreement={result.agreement_level:.2f}, "
            f"latency={result.total_latency_ms:.0f}ms{negation_info}"
        )
        
        return result
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for better matching."""
        # Lowercase
        normalized = query.lower()
        # Normalize whitespace
        normalized = " ".join(normalized.split())
        # Common substitutions
        substitutions = {
            "rna-seq": "rnaseq",
            "chip-seq": "chipseq",
            "atac-seq": "atacseq",
            "hi-c": "hic",
        }
        for old, new in substitutions.items():
            normalized = normalized.replace(old, new)
        return normalized
    
    def _run_pattern_parser(
        self, 
        query: str, 
        context: Optional[Dict]
    ) -> Optional[MethodVote]:
        """Run rule-based pattern parsing."""
        if not self.pattern_parser:
            return None
        
        start = time.time()
        try:
            result = self.pattern_parser.parse(query, context)
            return MethodVote(
                method=ParsingMethod.RULE_PATTERN,
                intent=result.primary_intent.name,
                confidence=result.confidence,
                weight=self.weights[ParsingMethod.RULE_PATTERN],
                slots=result.slots,
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            logger.debug(f"Pattern parser failed: {e}")
            return MethodVote(
                method=ParsingMethod.RULE_PATTERN,
                intent="META_UNKNOWN",
                confidence=0.0,
                weight=self.weights[ParsingMethod.RULE_PATTERN],
                error=str(e),
                latency_ms=(time.time() - start) * 1000,
            )
    
    def _run_semantic_classifier(self, query: str) -> Optional[MethodVote]:
        """Run FAISS semantic similarity classification."""
        if not self.semantic_classifier:
            return None
        
        start = time.time()
        try:
            results = self.semantic_classifier.classify(query, top_k=3)
            if not results:
                return None
            
            intent, confidence = results[0]
            return MethodVote(
                method=ParsingMethod.SEMANTIC_FAISS,
                intent=intent,
                confidence=confidence,
                weight=self.weights[ParsingMethod.SEMANTIC_FAISS],
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            logger.debug(f"Semantic classifier failed: {e}")
            return None
    
    def _run_ner_inference(self, query: str) -> Optional[MethodVote]:
        """Run NER-based intent inference."""
        if not self.ner:
            return None
        
        start = time.time()
        try:
            entities = self.ner.extract(query)
            if not entities:
                return None
            
            # Infer intent from entities
            intent, confidence = self._infer_intent_from_entities(query, entities)
            
            # Even if we can't infer intent, return a low-confidence vote 
            # to contribute entities to the final result
            if not intent:
                intent = "META_UNKNOWN"
                confidence = 0.1  # Low confidence, just for entity contribution
            
            return MethodVote(
                method=ParsingMethod.NER_ENTITY,
                intent=intent,
                confidence=confidence,
                weight=self.weights[ParsingMethod.NER_ENTITY],
                entities=entities,
                slots=self._entities_to_slots(entities),
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            logger.debug(f"NER inference failed: {e}")
            return None
    
    def _run_rag_lookup(self, query: str) -> Optional[MethodVote]:
        """Look up similar past queries from RAG history."""
        if not self.rag_orchestrator:
            return None
        
        start = time.time()
        try:
            # Find similar successful queries
            enhancement = self.rag_orchestrator.enhance(
                query=query,
                candidate_tools=[],  # We want intent, not tool
                base_args={}
            )
            
            if enhancement and enhancement.suggested_tool:
                # Map tool name back to intent
                tool_to_intent = {
                    "search_databases": "DATA_SEARCH",
                    "scan_data": "DATA_SCAN",
                    "download_dataset": "DATA_DOWNLOAD",
                    "generate_workflow": "WORKFLOW_CREATE",
                    "get_job_status": "JOB_STATUS",
                    "explain_concept": "EDUCATION_EXPLAIN",
                }
                
                intent = tool_to_intent.get(enhancement.suggested_tool)
                if intent:
                    return MethodVote(
                        method=ParsingMethod.RAG_HISTORY,
                        intent=intent,
                        confidence=enhancement.confidence * 0.9,  # Slight discount
                        weight=self.weights[ParsingMethod.RAG_HISTORY],
                        slots=enhancement.suggested_args or {},
                        latency_ms=(time.time() - start) * 1000,
                    )
        except Exception as e:
            logger.debug(f"RAG lookup failed: {e}")
        
        return None
    
    def _run_llm_fallback(
        self, 
        query: str, 
        current_result: EnsembleParseResult
    ) -> Optional[MethodVote]:
        """Use LLM for complex/ambiguous queries."""
        if not self.llm_client:
            return None
        
        start = time.time()
        try:
            # Build prompt with context about what we found so far
            entity_desc = ", ".join(
                f"{e.entity_type}:{e.text}" 
                for e in current_result.entities[:5]
            ) if current_result.entities else "none"
            
            prompt = f"""Analyze this bioinformatics query and determine the user's intent.

Query: "{query}"

Detected entities: {entity_desc}
Current best guess: {current_result.intent} (confidence: {current_result.confidence:.2f})

Choose the most likely intent from:
- DATA_SEARCH: User wants to find/search for datasets
- DATA_DOWNLOAD: User wants to download specific data
- DATA_SCAN: User wants to check local data
- WORKFLOW_CREATE: User wants to create/generate a pipeline
- JOB_STATUS: User wants to check job status
- EDUCATION_EXPLAIN: User wants to learn about a concept
- META_CONFIRM: User is confirming a previous action
- META_CANCEL: User is canceling

Respond with JSON: {{"intent": "INTENT_NAME", "confidence": 0.0-1.0, "reason": "brief explanation"}}"""

            response = self.llm_client.complete(prompt, max_tokens=100)
            
            # Parse response
            import json
            data = json.loads(response.content.strip())
            
            return MethodVote(
                method=ParsingMethod.LLM_GENERATION,
                intent=data["intent"],
                confidence=data["confidence"],
                weight=self.weights[ParsingMethod.LLM_GENERATION],
                latency_ms=(time.time() - start) * 1000,
            )
            
        except Exception as e:
            logger.debug(f"LLM fallback failed: {e}")
            return None
    
    def _infer_intent_from_entities(
        self, 
        query: str, 
        entities: List[Any]
    ) -> Tuple[Optional[str], float]:
        """Infer intent from extracted entities."""
        query_lower = query.lower()
        entity_types = {e.entity_type for e in entities}
        
        # Strong signals
        if "DATASET_ID" in entity_types:
            if any(w in query_lower for w in ["download", "get", "fetch", "retrieve"]):
                return "DATA_DOWNLOAD", 0.9
            if any(w in query_lower for w in ["details", "info", "about", "show", "describe"]):
                return "DATA_DESCRIBE", 0.85
        
        # Path-based signals - DATA_SCAN
        if "PATH" in entity_types:
            if any(w in query_lower for w in ["scan", "check", "list", "what files", "inventory"]):
                return "DATA_SCAN", 0.85
            if any(w in query_lower for w in ["submit", "run", "execute"]):
                return "JOB_SUBMIT", 0.85
        
        # File format signals
        if "FILE_FORMAT" in entity_types:
            if any(w in query_lower for w in ["scan", "find", "list", "look for"]):
                return "DATA_SCAN", 0.75
        
        # Job ID signals
        if "JOB_ID" in entity_types:
            if any(w in query_lower for w in ["status", "check", "how is"]):
                return "JOB_STATUS", 0.9
            if any(w in query_lower for w in ["cancel", "stop", "kill"]):
                return "JOB_CANCEL", 0.9
            if any(w in query_lower for w in ["log", "output", "error"]):
                return "JOB_LOGS", 0.85
        
        # Workflow creation signals
        if "ASSAY_TYPE" in entity_types:
            if any(w in query_lower for w in ["create", "generate", "make", "build", "workflow", "pipeline"]):
                return "WORKFLOW_CREATE", 0.85
        
        # Search signals
        if entity_types & {"ASSAY_TYPE", "ORGANISM", "TISSUE", "DISEASE"}:
            if any(w in query_lower for w in ["search", "find", "look", "data", "dataset"]):
                return "DATA_SEARCH", 0.8
            # Even without explicit search words, entities suggest search
            if len(entity_types) >= 2:
                return "DATA_SEARCH", 0.6
        
        return None, 0.0
    
    def _entities_to_slots(self, entities: List[Any]) -> Dict[str, Any]:
        """Convert entities to slot dictionary."""
        slots = {}
        for e in entities:
            if e.entity_type == "ORGANISM":
                slots.setdefault("organism", getattr(e, 'canonical', e.text))
            elif e.entity_type == "ASSAY_TYPE":
                slots.setdefault("assay_type", getattr(e, 'canonical', e.text))
            elif e.entity_type == "TISSUE":
                slots.setdefault("tissue", getattr(e, 'canonical', e.text))
            elif e.entity_type == "DISEASE":
                slots.setdefault("disease", getattr(e, 'canonical', e.text))
            elif e.entity_type == "DATASET_ID":
                slots.setdefault("dataset_id", e.text)
            elif e.entity_type == "PATH":
                slots.setdefault("path", e.text)
            elif e.entity_type == "JOB_ID":
                slots.setdefault("job_id", e.text)
            elif e.entity_type == "FILE_FORMAT":
                slots.setdefault("file_format", getattr(e, 'canonical', e.text))
        return slots
    
    def _combine_votes(
        self, 
        votes: List[MethodVote],
        query: str,
        normalized: str,
    ) -> EnsembleParseResult:
        """Combine votes from all methods using weighted voting."""
        
        if not votes:
            return EnsembleParseResult(
                intent="META_UNKNOWN",
                confidence=0.0,
                agreement_level=0.0,
            )
        
        # Aggregate scores by intent
        intent_scores: Dict[str, float] = {}
        intent_sources: Dict[str, List[ParsingMethod]] = {}
        
        # Special handling: if rule_pattern has very high confidence (>0.90),
        # trust it strongly for job/data operations where patterns are precise
        rule_pattern_vote = next(
            (v for v in votes if v.method == ParsingMethod.RULE_PATTERN and v.confidence >= 0.90),
            None
        )
        if rule_pattern_vote and rule_pattern_vote.intent in (
            'JOB_SUBMIT', 'JOB_STATUS', 'JOB_LIST', 'JOB_CANCEL', 'JOB_LOGS',
            'DATA_DOWNLOAD', 'DATA_SEARCH', 'DATA_SCAN',
        ):
            # Give rule_pattern a significant boost for these precise operations
            intent_scores[rule_pattern_vote.intent] = rule_pattern_vote.confidence * 1.5
            intent_sources[rule_pattern_vote.intent] = [ParsingMethod.RULE_PATTERN]
        
        for vote in votes:
            if vote.confidence > 0.1:  # Ignore very low confidence votes
                score = vote.weighted_score
                intent_scores[vote.intent] = intent_scores.get(vote.intent, 0) + score
                
                if vote.intent not in intent_sources:
                    intent_sources[vote.intent] = []
                intent_sources[vote.intent].append(vote.method)
        
        if not intent_scores:
            return EnsembleParseResult(
                intent="META_UNKNOWN",
                confidence=0.0,
                agreement_level=0.0,
                votes=votes,
            )
        
        # Find winning intent
        winning_intent = max(intent_scores, key=intent_scores.get)
        base_score = intent_scores[winning_intent]
        
        # Calculate agreement level
        agreeing_methods = len(intent_sources.get(winning_intent, []))
        total_voting_methods = len([v for v in votes if v.confidence > 0.1])
        agreement_level = agreeing_methods / total_voting_methods if total_voting_methods > 0 else 0
        
        # Apply agreement bonus
        agreement_bonus = agreement_level * self.AGREEMENT_BONUS
        
        # Calculate final confidence
        total_weight = sum(v.weight for v in votes if v.confidence > 0.1)
        confidence = (base_score / total_weight + agreement_bonus) if total_weight > 0 else 0
        confidence = min(confidence, 1.0)
        
        # Determine winning method
        if agreeing_methods >= 3:
            winning_method = ParsingMethod.ENSEMBLE_VOTE
        else:
            # Find highest confidence vote for winning intent
            best_vote = max(
                (v for v in votes if v.intent == winning_intent),
                key=lambda v: v.confidence,
                default=None
            )
            winning_method = best_vote.method if best_vote else ParsingMethod.ENSEMBLE_VOTE
        
        # Collect all entities and slots
        all_entities = []
        all_slots = {}
        seen_entities = set()
        
        for vote in votes:
            for e in vote.entities:
                key = (e.entity_type, e.text.lower())
                if key not in seen_entities:
                    seen_entities.add(key)
                    all_entities.append(e)
            for k, v in vote.slots.items():
                all_slots.setdefault(k, v)
        
        # Build alternatives
        alternatives = [
            (intent, score / total_weight if total_weight > 0 else 0)
            for intent, score in sorted(intent_scores.items(), key=lambda x: -x[1])
            if intent != winning_intent
        ][:3]
        
        return EnsembleParseResult(
            intent=winning_intent,
            confidence=confidence,
            alternatives=alternatives,
            entities=all_entities,
            slots=all_slots,
            winning_method=winning_method,
            agreement_level=agreement_level,
            votes=votes,
        )
    
    def _generate_clarification(
        self,
        query: str,
        entities: List[Any],
        alternatives: List[Tuple[str, float]],
    ) -> str:
        """Generate a helpful clarification prompt."""
        if not entities and not alternatives:
            return (
                "I'm not sure what you're asking. Try:\n"
                "• 'search for [organism] [assay type] data'\n"
                "• 'create a [pipeline type] workflow'\n"
                "• 'help' to see all options"
            )
        
        parts = []
        
        if entities:
            entity_str = ", ".join(
                f"**{e.entity_type}**: {getattr(e, 'canonical', e.text)}"
                for e in entities[:3]
            )
            parts.append(f"I found: {entity_str}")
        
        if alternatives:
            suggestions = [
                f"- {alt[0].replace('_', ' ').lower().title()}"
                for alt, _ in alternatives[:2]
            ]
            parts.append(f"Did you mean:\n" + "\n".join(suggestions))
        
        return " ".join(parts) + "\n\nPlease clarify what you'd like to do."


# Factory function
def create_ensemble_parser(
    llm_client=None,
    enable_rag: bool = True,
) -> UnifiedEnsembleParser:
    """Create a configured ensemble parser."""
    return UnifiedEnsembleParser(
        llm_client=llm_client,
        enable_rag=enable_rag,
    )
