"""
Unified Intent Parser
=====================

A unified parser that combines the arbiter with fallback to simpler methods.
Provides IntentResult-compatible output for integration with DialogueManager.

This bridges the gap between:
- IntentArbiter (returns ArbiterResult)
- DialogueManager (expects IntentResult)

Architecture:
- Parallel execution of Pattern + Semantic + Entity extraction (CPU-bound)
- Normalized confidence scores for fair comparison
- Weighted ensemble voting before LLM escalation
- Entity-aware intent boosting
"""

import logging
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
    
    # =========================================================================
    # Ensemble Methods: Parallel Execution, Normalization, Entity-Aware Voting
    # =========================================================================
    
    def _run_parsers_parallel(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[IntentResult], Optional[List[Tuple[str, float]]], List[Entity]]:
        """
        Run pattern parser, semantic classifier, and entity extraction in parallel.
        
        Returns:
            Tuple of (pattern_result, semantic_results, entities)
        """
        pattern_result = None
        semantic_results = None
        entities = []
        
        def run_pattern():
            return self.pattern_parser.parse(query, context)
        
        def run_semantic():
            if self.semantic_classifier:
                try:
                    return self.semantic_classifier.classify(query, top_k=5, threshold=0.3)
                except Exception as e:
                    logger.debug(f"Semantic classification failed: {e}")
            return None
        
        def run_entity():
            try:
                return self.pattern_parser._entity_extractor.extract(query)
            except Exception as e:
                logger.debug(f"Entity extraction failed: {e}")
                return []
        
        # Run all three in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(run_pattern): "pattern",
                executor.submit(run_semantic): "semantic",
                executor.submit(run_entity): "entity",
            }
            
            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    result = future.result(timeout=2.0)  # 2 second timeout
                    if task_name == "pattern":
                        pattern_result = result
                    elif task_name == "semantic":
                        semantic_results = result
                    elif task_name == "entity":
                        entities = result or []
                except Exception as e:
                    logger.warning(f"Parallel task {task_name} failed: {e}")
        
        return pattern_result, semantic_results, entities
    
    def _normalize_confidence(
        self, 
        confidence: float, 
        source: str,
        score_distribution: Optional[List[float]] = None
    ) -> float:
        """
        Normalize confidence scores to a comparable scale [0, 1].
        
        Different sources use different scales:
        - Pattern: Linear coverage-based (0.5 to 0.95)
        - Semantic: Cosine similarity (typically 0.3 to 0.9)
        - Arbiter: LLM confidence (0.0 to 1.0)
        
        We use min-max normalization with source-specific bounds.
        """
        if source == "pattern":
            # Pattern uses 0.5 + coverage * 0.5, so range is [0.5, 0.95]
            # Normalize to [0, 1]
            min_conf, max_conf = 0.5, 0.95
            normalized = (confidence - min_conf) / (max_conf - min_conf)
            return max(0.0, min(1.0, normalized))
        
        elif source == "semantic":
            # Semantic uses cosine similarity, typically in [0.3, 0.9]
            # Apply softmax if we have multiple scores for proper probability
            if score_distribution and len(score_distribution) > 1:
                scores = np.array(score_distribution)
                # Temperature-scaled softmax
                temp = 0.3  # Lower = sharper distribution
                exp_scores = np.exp(scores / temp)
                softmax = exp_scores / np.sum(exp_scores)
                # Return the first score (top intent) as probability
                return float(softmax[0])
            else:
                # Fallback: simple min-max normalization
                min_conf, max_conf = 0.3, 0.9
                normalized = (confidence - min_conf) / (max_conf - min_conf)
                return max(0.0, min(1.0, normalized))
        
        else:
            # Arbiter or other sources: assume already [0, 1]
            return max(0.0, min(1.0, confidence))
    
    def _run_parsers_parallel(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[IntentResult, Optional[List[Tuple[str, float]]], List[Entity]]:
        """
        Run all local parsers in parallel for efficiency.
        
        Executes Pattern + Semantic + Entity extraction concurrently,
        reducing total latency compared to sequential execution.
        
        Returns:
            Tuple of (pattern_result, semantic_results, entities)
        """
        pattern_result = None
        semantic_results = None
        entities = []
        
        def run_pattern():
            return self.pattern_parser.parse(query, context)
        
        def run_semantic():
            if self.semantic_classifier:
                try:
                    return self.semantic_classifier.classify(query, top_k=5)
                except Exception as e:
                    logger.debug(f"Semantic classification failed: {e}")
                    return None
            return None
        
        def run_entity_extraction():
            # Entity extraction is done by the pattern parser as part of parsing
            # We'll get entities from the pattern result
            return []
        
        # For short queries, parallel overhead may not be worth it
        # Only parallelize if semantic classifier is available
        if self.semantic_classifier:
            try:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    pattern_future = executor.submit(run_pattern)
                    semantic_future = executor.submit(run_semantic)
                    
                    # Wait for both to complete
                    pattern_result = pattern_future.result(timeout=5.0)
                    semantic_results = semantic_future.result(timeout=5.0)
            except Exception as e:
                logger.warning(f"Parallel parsing failed, falling back to sequential: {e}")
                pattern_result = run_pattern()
                semantic_results = run_semantic()
        else:
            # Sequential fallback if no semantic classifier
            pattern_result = run_pattern()
        
        # Entities come from pattern parser
        if pattern_result:
            entities = pattern_result.entities
        
        return pattern_result, semantic_results, entities
    
    def _entity_intent_boost(
        self, 
        entities: List[Entity], 
        query: str
    ) -> Dict[str, float]:
        """
        Calculate intent boost scores based on extracted entities.
        
        If entities suggest a particular intent, we boost its score.
        This provides a third signal beyond pattern and semantic matching.
        
        Returns:
            Dict mapping intent names to boost values (can be negative for penalties)
        """
        boosts = {}
        entity_types = {e.type for e in entities}
        query_lower = query.lower()
        
        # Path entities → DATA_SCAN
        if EntityType.DIRECTORY_PATH in entity_types or EntityType.FILE_PATH in entity_types:
            boosts["DATA_SCAN"] = 0.15
            boosts["DATA_DESCRIBE"] = 0.10
            # Penalize EDUCATION if path is present (user has local context)
            boosts["EDUCATION_EXPLAIN"] = -0.10
        
        # Dataset ID → DATA_DOWNLOAD
        if EntityType.DATASET_ID in entity_types:
            if any(w in query_lower for w in ["download", "get", "fetch"]):
                boosts["DATA_DOWNLOAD"] = 0.20
            else:
                boosts["DATA_SEARCH"] = 0.10
        
        # Disease/Tissue → DATA_SEARCH (looking for datasets)
        if EntityType.DISEASE in entity_types or EntityType.TISSUE in entity_types:
            boosts["DATA_SEARCH"] = 0.15
            # Unless "explain" is in query
            if "explain" in query_lower or "what is" in query_lower:
                boosts["EDUCATION_EXPLAIN"] = 0.10
        
        # Workflow-related entities → WORKFLOW_*
        if EntityType.WORKFLOW_TYPE in entity_types:
            boosts["WORKFLOW_CREATE"] = 0.15
        
        # Assay type or Data type → different intents based on context
        if EntityType.ASSAY_TYPE in entity_types or EntityType.DATA_TYPE in entity_types:
            if any(w in query_lower for w in ["run", "execute", "start", "perform"]):
                boosts["JOB_SUBMIT"] = 0.15
            elif any(w in query_lower for w in ["explain", "what is", "how does"]):
                boosts["EDUCATION_EXPLAIN"] = 0.10
            else:
                boosts["WORKFLOW_CREATE"] = 0.10
        
        # "Local" or "folder" or "directory" keywords without entities still suggest DATA_SCAN
        local_keywords = ["local", "folder", "directory", "my data", "our data", "we have"]
        if any(kw in query_lower for kw in local_keywords):
            boosts["DATA_SCAN"] = boosts.get("DATA_SCAN", 0) + 0.10
        
        return boosts
    
    def _weighted_ensemble_vote(
        self,
        pattern_result: IntentResult,
        semantic_results: Optional[List[Tuple[str, float]]],
        entities: List[Entity],
        query: str,
    ) -> Tuple[str, float, str, Dict[str, float]]:
        """
        Combine all signals using weighted voting.
        
        Weights:
        - Pattern: 0.35 (fast, good for exact matches)
        - Semantic: 0.40 (better semantic understanding)
        - Entity boost: 0.25 (domain knowledge)
        
        Returns:
            Tuple of (winning_intent, confidence, method, all_scores)
        """
        WEIGHT_PATTERN = 0.35
        WEIGHT_SEMANTIC = 0.40
        WEIGHT_ENTITY = 0.25
        
        # Initialize scores for all possible intents
        intent_scores: Dict[str, float] = {}
        
        # 1. Add pattern contribution
        if pattern_result:
            pattern_intent = pattern_result.primary_intent.name
            pattern_conf = self._normalize_confidence(pattern_result.confidence, "pattern")
            intent_scores[pattern_intent] = intent_scores.get(pattern_intent, 0) + WEIGHT_PATTERN * pattern_conf
            
            # Add alternatives with lower weight
            for alt_intent, alt_conf in pattern_result.alternatives[:3]:
                alt_name = alt_intent.name if hasattr(alt_intent, 'name') else str(alt_intent)
                alt_norm = self._normalize_confidence(alt_conf, "pattern")
                intent_scores[alt_name] = intent_scores.get(alt_name, 0) + WEIGHT_PATTERN * alt_norm * 0.5
        
        # 2. Add semantic contribution
        if semantic_results:
            # Get score distribution for softmax normalization
            score_dist = [score for _, score in semantic_results]
            
            for intent_name, conf in semantic_results[:5]:
                semantic_conf = self._normalize_confidence(conf, "semantic", score_dist)
                # Weight decreases for lower-ranked results
                rank_weight = 1.0 if intent_name == semantic_results[0][0] else 0.5
                intent_scores[intent_name] = intent_scores.get(intent_name, 0) + WEIGHT_SEMANTIC * semantic_conf * rank_weight
        
        # 3. Add entity boost contribution
        entity_boosts = self._entity_intent_boost(entities, query)
        for intent_name, boost in entity_boosts.items():
            intent_scores[intent_name] = intent_scores.get(intent_name, 0) + WEIGHT_ENTITY * boost
        
        # Find winner
        if not intent_scores:
            return "META_UNKNOWN", 0.0, "no_votes", {}
        
        # Sort by score
        sorted_intents = sorted(intent_scores.items(), key=lambda x: -x[1])
        winner, winner_score = sorted_intents[0]
        
        # Determine if unanimous (top score is much higher than second)
        if len(sorted_intents) > 1:
            second_score = sorted_intents[1][1]
            margin = winner_score - second_score
            method = "unanimous" if margin > 0.15 else "ensemble"
        else:
            method = "unanimous"
        
        # Log the voting
        logger.debug(
            f"Ensemble vote: winner={winner}({winner_score:.3f}), "
            f"scores={[(k, f'{v:.3f}') for k, v in sorted_intents[:3]]}"
        )
        
        return winner, winner_score, method, dict(sorted_intents)

    def parse(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> UnifiedParseResult:
        """
        Parse a query and return unified result.
        
        Architecture:
        1. Check cache for previously processed queries
        2. Run Pattern + Semantic + Entity extraction in PARALLEL (CPU-bound)
        3. Use weighted ensemble voting to combine all signals
        4. Only invoke LLM arbiter if ensemble confidence is low or disagreement exists
        
        This ensures efficient use of all local CPU methods before cloud escalation.
        
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
            # Step 0: Check cache
            cached = self._cache_get(query)
            if cached is not None:
                self._metrics["cache_hits"] += 1
                latency_ms = (time.time() - start_time) * 1000
                self._record_metrics(cached, latency_ms)
                return cached
            
            # Step 1: Run all local parsers in PARALLEL
            pattern_result, semantic_results, entities = self._run_parsers_parallel(query, context)
            
            # Step 2: Weighted ensemble voting
            winner_intent, ensemble_conf, vote_method, all_scores = self._weighted_ensemble_vote(
                pattern_result, semantic_results, entities, query
            )
            
            # Extract top semantic result for arbiter decision
            semantic_result = None
            if semantic_results:
                semantic_result = semantic_results[0]  # (intent_name, confidence)
            
            # Step 3: Determine if we need LLM arbiter
            # Use arbiter if:
            # - Ensemble confidence is low (<0.5)
            # - Ensemble method is not "unanimous" (close race)
            # - Cross-category disagreement between pattern and semantic
            needs_arbiter = False
            
            if ensemble_conf < 0.5:
                logger.debug(f"Low ensemble confidence: {ensemble_conf:.3f}")
                needs_arbiter = True
            elif vote_method != "unanimous":
                # Check if top 2 are from different categories
                sorted_scores = sorted(all_scores.items(), key=lambda x: -x[1])
                if len(sorted_scores) >= 2:
                    first_cat = self._get_intent_category(sorted_scores[0][0])
                    second_cat = self._get_intent_category(sorted_scores[1][0])
                    if first_cat != second_cat and sorted_scores[1][1] > 0.3:
                        logger.debug(
                            f"Close cross-category race: {sorted_scores[0]} vs {sorted_scores[1]}"
                        )
                        needs_arbiter = True
            
            # Also check for complexity indicators
            if not needs_arbiter:
                needs_arbiter = self._has_complexity_indicators(query)
            
            # Step 4: Invoke arbiter if needed
            if needs_arbiter and self.arbiter:
                arbiter_result = self._invoke_arbiter(query, pattern_result, semantic_result)
                if arbiter_result:
                    # Determine appropriate slots
                    slots_to_use = pattern_result.slots if pattern_result else {}
                    original_intent = pattern_result.primary_intent.name if pattern_result else "META_UNKNOWN"
                    final_intent = arbiter_result.final_intent
                    
                    if original_intent != final_intent:
                        original_category = self._get_intent_category(original_intent)
                        final_category = self._get_intent_category(final_intent)
                        if original_category != final_category:
                            logger.debug(f"Clearing slots: {original_category} -> {final_category}")
                            slots_to_use = {}
                    
                    result = UnifiedParseResult.from_arbiter_result(
                        arbiter_result,
                        entities=entities,
                        slots=slots_to_use,
                    )
                    if result.llm_invoked:
                        self._cache_put(query, result)
                    
                    latency_ms = (time.time() - start_time) * 1000
                    self._record_metrics(result, latency_ms)
                    return result
            
            # Step 5: Return ensemble result (no LLM needed)
            try:
                final_intent = IntentType[winner_intent]
            except KeyError:
                logger.warning(f"Unknown intent: {winner_intent}")
                final_intent = IntentType.META_UNKNOWN
            
            # Use slots from pattern if intent matches, else empty
            slots_to_use = {}
            if pattern_result and pattern_result.primary_intent.name == winner_intent:
                slots_to_use = pattern_result.slots
            
            result = UnifiedParseResult(
                primary_intent=final_intent,
                confidence=ensemble_conf,
                entities=entities,
                slots=slots_to_use,
                alternatives=[(IntentType[k], v) for k, v in list(all_scores.items())[:3] 
                             if k != winner_intent and k in IntentType.__members__],
                method=vote_method,
                matched_pattern=pattern_result.matched_pattern if pattern_result else None,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            self._record_metrics(result, latency_ms)
            return result
            
        except Exception as e:
            self._metrics["errors"] += 1
            logger.error(f"Parse error: {e}")
            raise
    
    def _has_complexity_indicators(self, query: str) -> bool:
        """Check for complexity indicators that warrant LLM involvement."""
        query_lower = query.lower()
        
        complexity_indicators = {
            "negation": ["not", "don't", "doesn't", "instead", "rather", "but not", "forget", "ignore"],
            "conditional": ["if ", "when ", "unless", "otherwise"],
            "comparative": ["better", "rather than", "prefer", "instead of"],
            "change": ["actually", "wait", "no,", "change", "switch to"],
        }
        
        for indicator_type, words in complexity_indicators.items():
            if any(word in query_lower for word in words):
                logger.debug(f"Complexity indicator ({indicator_type}) detected")
                return True
        
        return False
    
    def _should_use_arbiter(
        self,
        pattern_result: IntentResult,
        semantic_result: Optional[tuple],
        query: str,
    ) -> bool:
        """
        Determine if the arbiter should be invoked.
        
        We use a tiered approach:
        1. Always check for cross-parser disagreement first (pattern vs semantic)
        2. Check for competing pattern alternatives from different categories
        3. High confidence pattern (>= 0.85): Trust ONLY if no disagreement
        4. Medium confidence (0.6 - 0.85): Check for complexity indicators
        5. Low confidence (< 0.6): Use LLM for disambiguation
        6. Complexity indicators: Always use LLM for negation, ambiguity
        """
        if not self.arbiter:
            return False
        
        if self.arbiter_strategy == "always":
            return True
        
        confidence = pattern_result.confidence
        intent = pattern_result.primary_intent.name
        query_lower = query.lower()
        
        # =====================================================================
        # CRITICAL: Check semantic disagreement BEFORE high-confidence short-circuit
        # This ensures we don't blindly trust pattern matching when semantic
        # analysis suggests a different intent
        # =====================================================================
        if semantic_result:
            semantic_intent, semantic_conf = semantic_result
            # Normalize intent names for comparison
            pattern_normalized = intent.replace("_", "").lower()
            semantic_normalized = semantic_intent.replace("_", "").lower()
            
            # Check if different intents from different categories
            if pattern_normalized != semantic_normalized and semantic_conf > 0.5:
                pattern_category = self._get_intent_category(intent)
                semantic_category = self._get_intent_category(semantic_intent)
                
                # If from different categories, definitely invoke arbiter
                if pattern_category != semantic_category:
                    logger.debug(
                        f"Cross-parser disagreement (different categories): "
                        f"pattern={intent}({confidence:.2f}, {pattern_category}), "
                        f"semantic={semantic_intent}({semantic_conf:.2f}, {semantic_category})"
                    )
                    return True
                
                # Same category but different intent - invoke if confidence gap is small
                if semantic_conf > confidence - 0.2:
                    logger.debug(
                        f"Cross-parser disagreement (same category): "
                        f"pattern={intent}({confidence:.2f}), "
                        f"semantic={semantic_intent}({semantic_conf:.2f})"
                    )
                    return True
        
        # Check for competing high-confidence alternatives from pattern parser
        # This catches cases where multiple patterns matched similarly
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
        
        # Tier 1: High confidence pattern matching - trust it IF no disagreement detected above
        if confidence >= 0.85:
            # Exception: always check negation even with high confidence
            negation_words = ["not", "don't", "doesn't", "no ", "never", "forget", "ignore", "skip"]
            if any(word in query_lower for word in negation_words):
                logger.debug(f"High confidence but negation detected: {confidence:.2f}")
                return True
            return False
        
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
