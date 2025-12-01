"""
Unified Intent Parser
=====================

A unified parser that combines the arbiter with fallback to simpler methods.
Provides IntentResult-compatible output for integration with DialogueManager.

This bridges the gap between:
- IntentArbiter (returns ArbiterResult)
- DialogueManager (expects IntentResult)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .parser import IntentParser, IntentResult, IntentType, Entity, EntityType
from .arbiter import IntentArbiter, ArbiterResult, ParserVote
from .semantic import SemanticIntentClassifier

logger = logging.getLogger(__name__)


@dataclass
class UnifiedParseResult:
    """
    Intent parsing result that wraps IntentResult with arbiter information.
    
    Provides IntentResult-like interface for DialogueManager compatibility,
    while adding arbiter-specific fields.
    """
    # Core fields (matching IntentResult)
    primary_intent: IntentType
    confidence: float
    entities: List[Entity] = field(default_factory=list)
    sub_intents: List[IntentType] = field(default_factory=list)
    alternatives: List[tuple] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_prompt: str = ""
    slots: Dict[str, Any] = field(default_factory=dict)
    matched_pattern: Optional[str] = None
    raw_query: str = ""
    
    # Arbiter-specific fields
    arbiter_result: Optional[ArbiterResult] = None
    llm_invoked: bool = False
    reasoning: str = ""
    method: str = "pattern"
    
    @property
    def is_confident(self) -> bool:
        """Check if confidence is high enough for direct execution."""
        return self.confidence >= 0.7 and not self.needs_clarification
    
    def get_entity(self, entity_type: EntityType) -> Optional[Entity]:
        """Get first entity of given type."""
        for e in self.entities:
            if e.type == entity_type:
                return e
        return None
    
    def get_entities(self, entity_type: EntityType) -> List[Entity]:
        """Get all entities of given type."""
        return [e for e in self.entities if e.type == entity_type]
    
    @classmethod
    def from_arbiter_result(
        cls, 
        arbiter_result: ArbiterResult,
        entities: List[Entity] = None,
        slots: Dict[str, Any] = None,
    ) -> "UnifiedParseResult":
        """Create UnifiedParseResult from ArbiterResult."""
        # Convert intent string to IntentType
        try:
            primary_intent = IntentType[arbiter_result.final_intent]
        except KeyError:
            logger.warning(f"Unknown intent type: {arbiter_result.final_intent}")
            primary_intent = IntentType.META_UNKNOWN
        
        return cls(
            primary_intent=primary_intent,
            confidence=arbiter_result.confidence,
            entities=entities or [],
            slots=slots or {},
            arbiter_result=arbiter_result,
            llm_invoked=arbiter_result.llm_invoked,
            reasoning=arbiter_result.reasoning,
            method=arbiter_result.method,
        )
    
    @classmethod
    def from_intent_result(cls, result: IntentResult) -> "UnifiedParseResult":
        """Create UnifiedParseResult from regular IntentResult."""
        return cls(
            primary_intent=result.primary_intent,
            sub_intents=result.sub_intents,
            confidence=result.confidence,
            entities=result.entities,
            slots=result.slots,
            raw_query=getattr(result, 'raw_query', ''),
            alternatives=result.alternatives,
            needs_clarification=result.needs_clarification,
            clarification_prompt=result.clarification_prompt,
            matched_pattern=result.matched_pattern,
            method="pattern",
        )


class UnifiedIntentParser:
    """
    Unified parser that uses IntentArbiter with fallback.
    
    Features:
    - Uses arbiter for intelligent intent resolution
    - Cascading provider router for LLM calls (rate-limit resistant)
    - Falls back to pattern parser if arbiter unavailable
    - Returns DialogueManager-compatible IntentResult
    
    Usage:
        parser = UnifiedIntentParser(use_cascade=True)
        result = parser.parse("search for human ChIP-seq data")
        print(result.primary_intent)  # IntentType.DATA_SEARCH
        print(result.llm_invoked)     # True/False
    """
    
    # LRU cache for arbiter decisions (Phase 3a: Decision Caching)
    # Key: normalized query, Value: UnifiedParseResult
    MAX_CACHE_SIZE = 1000
    
    def __init__(
        self,
        use_cascade: bool = True,
        arbiter_strategy: str = "smart",
        llm_provider: str = None,
        enable_cache: bool = True,
    ):
        """
        Initialize the unified parser.
        
        Args:
            use_cascade: Use cascading provider router (rate-limit resistant)
            arbiter_strategy: Arbiter strategy ("smart", "always", "disagreement")
            llm_provider: Specific provider to use (None = use cascade/default)
            enable_cache: Enable LRU cache for arbiter decisions (default: True)
        """
        self.use_cascade = use_cascade
        self.arbiter_strategy = arbiter_strategy
        self.llm_provider = llm_provider
        self.enable_cache = enable_cache
        
        # Lazy-load components
        self._pattern_parser = None
        self._arbiter = None
        self._semantic_classifier = None
        
        # Decision cache (LRU)
        self._cache: Dict[str, UnifiedParseResult] = {}
        self._cache_order: List[str] = []  # LRU order tracking
        self._cache_stats = {"hits": 0, "misses": 0}
        
        # Monitoring metrics (Phase 4a)
        self._metrics = {
            "total_queries": 0,
            "pattern_only": 0,
            "llm_invoked": 0,
            "unanimous": 0,
            "cache_hits": 0,
            "errors": 0,
            "total_latency_ms": 0.0,
            "llm_latency_ms": 0.0,
            "by_intent": {},  # intent -> count
            "by_method": {},  # method -> count
        }
        
        logger.info(
            f"UnifiedIntentParser initialized: "
            f"cascade={use_cascade}, strategy={arbiter_strategy}, cache={enable_cache}"
        )
    
    @property
    def pattern_parser(self) -> IntentParser:
        """Lazy-load pattern parser."""
        if self._pattern_parser is None:
            self._pattern_parser = IntentParser()
        return self._pattern_parser
    
    @property
    def arbiter(self) -> Optional[IntentArbiter]:
        """Lazy-load arbiter with cascade support."""
        if self._arbiter is None:
            try:
                self._arbiter = IntentArbiter(use_cascade=self.use_cascade)
                logger.debug("IntentArbiter loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load IntentArbiter: {e}")
        return self._arbiter
    
    @property
    def semantic_classifier(self) -> Optional[SemanticIntentClassifier]:
        """Lazy-load semantic classifier."""
        if self._semantic_classifier is None:
            try:
                self._semantic_classifier = SemanticIntentClassifier()
            except Exception as e:
                logger.debug(f"SemanticIntentClassifier not available: {e}")
        return self._semantic_classifier
    
    def _normalize_for_cache(self, query: str) -> str:
        """Normalize query for cache lookup."""
        # Lowercase, strip, collapse whitespace
        normalized = " ".join(query.lower().strip().split())
        return normalized
    
    def _cache_get(self, query: str) -> Optional[UnifiedParseResult]:
        """Check cache for a query result."""
        if not self.enable_cache:
            return None
        
        key = self._normalize_for_cache(query)
        if key in self._cache:
            # Move to end of LRU list
            self._cache_order.remove(key)
            self._cache_order.append(key)
            self._cache_stats["hits"] += 1
            logger.debug(f"Cache hit for: {query[:30]}...")
            return self._cache[key]
        
        self._cache_stats["misses"] += 1
        return None
    
    def _cache_put(self, query: str, result: UnifiedParseResult) -> None:
        """Store a result in cache."""
        if not self.enable_cache:
            return
        
        key = self._normalize_for_cache(query)
        
        # Evict oldest if at capacity
        if len(self._cache) >= self.MAX_CACHE_SIZE:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]
        
        self._cache[key] = result
        self._cache_order.append(key)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = self._cache_stats["hits"] / total if total > 0 else 0
        return {
            "hits": self._cache_stats["hits"],
            "misses": self._cache_stats["misses"],
            "hit_rate": hit_rate,
            "size": len(self._cache),
            "max_size": self.MAX_CACHE_SIZE,
        }
    
    def clear_cache(self) -> None:
        """Clear the decision cache."""
        self._cache.clear()
        self._cache_order.clear()
        self._cache_stats = {"hits": 0, "misses": 0}
    
    def _record_metrics(self, result: UnifiedParseResult, latency_ms: float) -> None:
        """Record metrics for monitoring."""
        self._metrics["total_queries"] += 1
        self._metrics["total_latency_ms"] += latency_ms
        
        method = getattr(result, 'method', 'pattern')
        intent = result.primary_intent.name
        
        # By method
        self._metrics["by_method"][method] = self._metrics["by_method"].get(method, 0) + 1
        
        # By intent
        self._metrics["by_intent"][intent] = self._metrics["by_intent"].get(intent, 0) + 1
        
        # Specific counters
        if method == "pattern":
            self._metrics["pattern_only"] += 1
        elif method == "llm_arbiter":
            self._metrics["llm_invoked"] += 1
            self._metrics["llm_latency_ms"] += latency_ms
        elif method == "unanimous":
            self._metrics["unanimous"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get monitoring metrics."""
        total = self._metrics["total_queries"]
        if total == 0:
            return {"total_queries": 0, "llm_rate": 0, "avg_latency_ms": 0}
        
        llm_count = self._metrics["llm_invoked"]
        return {
            "total_queries": total,
            "pattern_only": self._metrics["pattern_only"],
            "llm_invoked": llm_count,
            "unanimous": self._metrics["unanimous"],
            "cache_hits": self._cache_stats["hits"],
            "errors": self._metrics["errors"],
            "llm_rate": llm_count / total * 100 if total > 0 else 0,
            "avg_latency_ms": self._metrics["total_latency_ms"] / total,
            "avg_llm_latency_ms": self._metrics["llm_latency_ms"] / llm_count if llm_count > 0 else 0,
            "by_method": self._metrics["by_method"],
            "by_intent": dict(sorted(
                self._metrics["by_intent"].items(), 
                key=lambda x: -x[1]
            )[:10]),  # Top 10 intents
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self._metrics = {
            "total_queries": 0,
            "pattern_only": 0,
            "llm_invoked": 0,
            "unanimous": 0,
            "cache_hits": 0,
            "errors": 0,
            "total_latency_ms": 0.0,
            "llm_latency_ms": 0.0,
            "by_intent": {},
            "by_method": {},
        }
    
    def parse(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> UnifiedParseResult:
        """
        Parse a query and return unified result.
        
        Uses cache lookup first, then arbiter when available, with fallback to pattern parser.
        
        Args:
            query: User query
            context: Optional conversation context
            
        Returns:
            UnifiedParseResult compatible with DialogueManager
        """
        start_time = time.time()
        query = query.strip()
        
        if not query:
            return UnifiedParseResult(
                primary_intent=IntentType.META_UNKNOWN,
                confidence=0.0,
                method="empty_query",
            )
        
        try:
            # Step 0: Check cache (only for queries that would trigger LLM)
            cached = self._cache_get(query)
            if cached is not None:
                self._metrics["cache_hits"] += 1
                latency_ms = (time.time() - start_time) * 1000
                self._record_metrics(cached, latency_ms)
                return cached
            
            # Step 1: Pattern parsing (fast, always available)
            pattern_result = self.pattern_parser.parse(query, context)
            
            # Step 2: Semantic classification if available
            semantic_result = None
            if self.semantic_classifier:
                try:
                    results = self.semantic_classifier.classify(query, top_k=1)
                    if results:
                        intent_name, conf = results[0]
                        semantic_result = (intent_name, conf)
                except Exception as e:
                    logger.debug(f"Semantic classification failed: {e}")
            
            # Step 3: Determine if arbiter is needed
            if self._should_use_arbiter(pattern_result, semantic_result, query):
                arbiter_result = self._invoke_arbiter(
                    query, pattern_result, semantic_result
                )
                if arbiter_result:
                    # If arbiter changed the intent to a different category,
                    # clear slots from the original pattern (they may not be valid)
                    slots_to_use = pattern_result.slots
                    original_intent = pattern_result.primary_intent.name
                    final_intent = arbiter_result.final_intent
                    
                    if original_intent != final_intent:
                        original_category = self._get_intent_category(original_intent)
                        final_category = self._get_intent_category(final_intent)
                        
                        if original_category != final_category:
                            # Different category - slots from original pattern likely invalid
                            # Clear them to prevent type mismatches (e.g., 'concept' for DATA_SCAN)
                            logger.debug(
                                f"Clearing slots due to category change: "
                                f"{original_category} -> {final_category}"
                            )
                            slots_to_use = {}
                    
                    result = UnifiedParseResult.from_arbiter_result(
                        arbiter_result,
                        entities=pattern_result.entities,
                        slots=slots_to_use,
                    )
                    # Cache LLM results for future lookups
                    if result.llm_invoked:
                        self._cache_put(query, result)
                    
                    latency_ms = (time.time() - start_time) * 1000
                    self._record_metrics(result, latency_ms)
                    return result
            
            # Step 4: Fallback to pattern result
            result = UnifiedParseResult.from_intent_result(pattern_result)
            latency_ms = (time.time() - start_time) * 1000
            self._record_metrics(result, latency_ms)
            return result
            
        except Exception as e:
            self._metrics["errors"] += 1
            logger.error(f"Parse error: {e}")
            raise
    
    def _should_use_arbiter(
        self,
        pattern_result: IntentResult,
        semantic_result: Optional[tuple],
        query: str,
    ) -> bool:
        """
        Determine if the arbiter should be invoked.
        
        We use a tiered approach:
        1. High confidence pattern (>= 0.85): Trust it, UNLESS there are competing alternatives
        2. Medium confidence (0.6 - 0.85): Check for complexity indicators
        3. Low confidence (< 0.6): Use LLM for disambiguation
        4. Complexity indicators: Always use LLM for negation, ambiguity
        5. Competing alternatives: If multiple patterns matched with similar confidence
           from different categories, use LLM to disambiguate
        """
        if not self.arbiter:
            return False
        
        if self.arbiter_strategy == "always":
            return True
        
        confidence = pattern_result.confidence
        intent = pattern_result.primary_intent.name
        query_lower = query.lower()
        
        # Check for competing high-confidence alternatives from different categories
        # This catches cases like DATA_SCAN vs EDUCATION_EXPLAIN both matching
        if pattern_result.alternatives:
            for alt_intent, alt_conf in pattern_result.alternatives:
                # If alternative is close in confidence (within 0.15) and from different category
                conf_diff = abs(confidence - alt_conf)
                if conf_diff < 0.15:
                    primary_category = self._get_intent_category(intent)
                    alt_category = self._get_intent_category(alt_intent.name if hasattr(alt_intent, 'name') else str(alt_intent))
                    
                    if primary_category != alt_category:
                        logger.debug(
                            f"Competing alternatives from different categories: "
                            f"{intent}({confidence:.2f}) vs {alt_intent}({alt_conf:.2f})"
                        )
                        return True
        
        # Tier 1: High confidence pattern matching - trust it
        if confidence >= 0.85:
            # Exception: always check negation even with high confidence
            negation_words = ["not", "don't", "doesn't", "no ", "never", "forget", "ignore", "skip"]
            if any(word in query_lower for word in negation_words):
                logger.debug(f"High confidence but negation detected: {confidence:.2f}")
                return True
            return False
        
        # Tier 2: Check for disagreement with semantic classifier
        if semantic_result:
            semantic_intent, semantic_conf = semantic_result
            # Normalize intent names for comparison
            pattern_normalized = intent.replace("_", "").lower()
            semantic_normalized = semantic_intent.replace("_", "").lower()
            
            if pattern_normalized != semantic_normalized and semantic_conf > 0.5:
                logger.debug(
                    f"Disagreement: pattern={intent}({confidence:.2f}), "
                    f"semantic={semantic_intent}({semantic_conf:.2f})"
                )
                return True
        
        # Tier 3: Low confidence - need LLM help
        if confidence < 0.6:
            logger.debug(f"Low confidence: {confidence:.2f}")
            return True
        
        # Tier 4: Complexity indicators (negation, conditionals, etc.)
        complexity_indicators = {
            "negation": ["not", "don't", "doesn't", "instead", "rather", "but", "forget", "ignore"],
            "conditional": ["if ", "when ", "unless", "otherwise"],
            "comparative": ["better", "rather", "prefer", "instead of"],
            "change": ["actually", "wait", "no,", "change", "switch"],
        }
        
        for indicator_type, words in complexity_indicators.items():
            if any(word in query_lower for word in words):
                logger.debug(f"Complexity indicator ({indicator_type}) detected")
                return True
        
        # Medium confidence (0.6-0.85) without complexity: trust pattern
        return False
    
    def _get_intent_category(self, intent_name: str) -> str:
        """
        Get the high-level category for an intent.
        
        This helps detect when competing matches are from fundamentally
        different domains (e.g., DATA vs EDUCATION vs WORKFLOW).
        """
        intent_upper = intent_name.upper()
        
        # Data operations
        if intent_upper.startswith("DATA_"):
            return "DATA"
        
        # Workflow operations  
        if intent_upper.startswith("WORKFLOW_") or "PIPELINE" in intent_upper:
            return "WORKFLOW"
        
        # Education/Help
        if intent_upper.startswith("EDUCATION_") or intent_upper.startswith("HELP_"):
            return "EDUCATION"
        
        # Meta/System
        if intent_upper.startswith("META_"):
            return "META"
        
        # Tool operations
        if intent_upper.startswith("TOOL_"):
            return "TOOL"
        
        # Session management
        if intent_upper.startswith("SESSION_"):
            return "SESSION"
        
        return "OTHER"
    
    def _invoke_arbiter(
        self,
        query: str,
        pattern_result: IntentResult,
        semantic_result: Optional[tuple],
    ) -> Optional[ArbiterResult]:
        """Invoke the arbiter for final decision."""
        if not self.arbiter:
            return None
        
        try:
            # Build votes from available results
            votes = [
                ParserVote(
                    parser_name="pattern",
                    intent=pattern_result.primary_intent.name,
                    confidence=pattern_result.confidence,
                    evidence=f"slots={list(pattern_result.slots.keys())}",
                ),
            ]
            
            if semantic_result:
                intent_name, conf = semantic_result
                votes.append(ParserVote(
                    parser_name="semantic",
                    intent=intent_name,
                    confidence=conf,
                ))
            
            # Call arbiter
            result = self.arbiter.arbitrate(
                query=query,
                votes=votes,
            )
            
            return result
            
        except Exception as e:
            logger.warning(f"Arbiter invocation failed: {e}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get parser status information including cache stats."""
        status = {
            "use_cascade": self.use_cascade,
            "arbiter_strategy": self.arbiter_strategy,
            "enable_cache": self.enable_cache,
            "pattern_parser": self._pattern_parser is not None,
            "arbiter": self._arbiter is not None,
            "semantic_classifier": self._semantic_classifier is not None,
            "cache": self.get_cache_stats(),
        }
        
        if self._arbiter and hasattr(self._arbiter, '_router'):
            router = self._arbiter._router
            if router:
                status["router_status"] = router.get_status()
        
        return status


# Convenience factory
def create_unified_parser(
    use_cascade: bool = True,
    strategy: str = "smart",
) -> UnifiedIntentParser:
    """Create a configured unified parser."""
    return UnifiedIntentParser(
        use_cascade=use_cascade,
        arbiter_strategy=strategy,
    )
