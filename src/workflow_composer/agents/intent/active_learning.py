"""
Active Learning Infrastructure
==============================

Track user corrections and feedback to improve intent classification over time.

Features:
- Correction storage (JSONL format)
- Confirmation tracking (positive signal)
- Confusion matrix analysis
- Export for retraining
- Learning metrics

Usage:
    learner = ActiveLearner()
    
    # Record correction
    learner.record_correction(
        query="search for liver data",
        predicted="DATA_SCAN",
        corrected="DATA_SEARCH"
    )
    
    # Get metrics
    metrics = learner.get_metrics()
    print(f"Correction rate: {metrics.correction_rate:.2%}")
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CorrectionRecord:
    """A single correction record."""
    timestamp: str
    query: str
    predicted_intent: str
    corrected_intent: str
    user_id: str = "anonymous"
    session_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "query": self.query,
            "predicted": self.predicted_intent,
            "corrected": self.corrected_intent,
            "user": self.user_id,
            "session": self.session_id,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CorrectionRecord":
        return cls(
            timestamp=data.get("timestamp", ""),
            query=data.get("query", ""),
            predicted_intent=data.get("predicted", ""),
            corrected_intent=data.get("corrected", ""),
            user_id=data.get("user", "anonymous"),
            session_id=data.get("session", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConfirmationRecord:
    """A confirmation that prediction was correct (positive signal)."""
    timestamp: str
    query: str
    intent: str
    confidence: float = 1.0
    user_id: str = "anonymous"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "query": self.query,
            "intent": self.intent,
            "confidence": self.confidence,
            "user": self.user_id,
            "type": "confirmation",
        }


@dataclass
class LearningMetrics:
    """Aggregated learning metrics."""
    total_queries: int = 0
    total_corrections: int = 0
    total_confirmations: int = 0
    correction_rate: float = 0.0
    top_confused_pairs: List[Tuple[str, str, int]] = field(default_factory=list)
    corrections_by_intent: Dict[str, int] = field(default_factory=dict)
    improvement_trend: float = 0.0  # Positive = improving
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "total_corrections": self.total_corrections,
            "total_confirmations": self.total_confirmations,
            "correction_rate": self.correction_rate,
            "top_confused_pairs": [
                {"predicted": p, "actual": a, "count": c}
                for p, a, c in self.top_confused_pairs
            ],
            "corrections_by_intent": self.corrections_by_intent,
            "improvement_trend": self.improvement_trend,
        }


# =============================================================================
# Active Learner
# =============================================================================

class ActiveLearner:
    """
    Track corrections and feedback for active learning.
    
    Stores data in JSONL format for easy appending and analysis.
    """
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        max_records: int = 10000,
    ):
        """
        Initialize the active learner.
        
        Args:
            data_dir: Directory for storing feedback data
            max_records: Maximum records to keep in memory
        """
        if data_dir is None:
            # Default to project data directory
            import workflow_composer
            package_dir = Path(workflow_composer.__file__).parent.parent.parent
            data_dir = package_dir / "data" / "feedback"
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.corrections_file = self.data_dir / "corrections.jsonl"
        self.confirmations_file = self.data_dir / "confirmations.jsonl"
        
        self.max_records = max_records
        
        # In-memory caches
        self._corrections: List[CorrectionRecord] = []
        self._confirmations: List[ConfirmationRecord] = []
        self._confusion_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Load existing data
        self._load_existing()
    
    def _load_existing(self) -> None:
        """Load existing feedback data from files."""
        # Load corrections
        if self.corrections_file.exists():
            try:
                with open(self.corrections_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            record = CorrectionRecord.from_dict(data)
                            self._corrections.append(record)
                            self._update_confusion_matrix(record)
                
                # Keep only recent records in memory
                if len(self._corrections) > self.max_records:
                    self._corrections = self._corrections[-self.max_records:]
                    
                logger.info(f"Loaded {len(self._corrections)} correction records")
            except Exception as e:
                logger.error(f"Failed to load corrections: {e}")
        
        # Load confirmations (lighter weight, just count)
        if self.confirmations_file.exists():
            try:
                count = 0
                with open(self.confirmations_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            count += 1
                logger.info(f"Found {count} confirmation records")
            except Exception as e:
                logger.error(f"Failed to count confirmations: {e}")
    
    def _update_confusion_matrix(self, record: CorrectionRecord) -> None:
        """Update confusion matrix with a correction."""
        self._confusion_matrix[record.predicted_intent][record.corrected_intent] += 1
    
    def record_correction(
        self,
        query: str,
        predicted: str,
        corrected: str,
        user_id: str = "anonymous",
        session_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a user correction (wrong intent predicted).
        
        Args:
            query: The user's query
            predicted: The intent that was predicted
            corrected: The correct intent (from user)
            user_id: User identifier
            session_id: Session identifier
            metadata: Additional context
        """
        record = CorrectionRecord(
            timestamp=datetime.utcnow().isoformat(),
            query=query,
            predicted_intent=predicted,
            corrected_intent=corrected,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
        )
        
        # Add to memory
        self._corrections.append(record)
        self._update_confusion_matrix(record)
        
        # Trim if needed
        if len(self._corrections) > self.max_records:
            self._corrections = self._corrections[-self.max_records:]
        
        # Append to file
        try:
            with open(self.corrections_file, 'a') as f:
                f.write(json.dumps(record.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to write correction: {e}")
        
        logger.info(
            f"Recorded correction: '{query[:50]}...' "
            f"{predicted} -> {corrected}"
        )
    
    def record_confirmation(
        self,
        query: str,
        intent: str,
        confidence: float = 1.0,
        user_id: str = "anonymous",
    ) -> None:
        """
        Record a confirmation (correct intent predicted).
        
        This is a positive signal for the learning system.
        
        Args:
            query: The user's query
            intent: The intent that was correctly predicted
            confidence: The prediction confidence
            user_id: User identifier
        """
        record = ConfirmationRecord(
            timestamp=datetime.utcnow().isoformat(),
            query=query,
            intent=intent,
            confidence=confidence,
            user_id=user_id,
        )
        
        self._confirmations.append(record)
        
        # Trim if needed
        if len(self._confirmations) > self.max_records:
            self._confirmations = self._confirmations[-self.max_records:]
        
        # Append to file
        try:
            with open(self.confirmations_file, 'a') as f:
                f.write(json.dumps(record.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to write confirmation: {e}")
    
    def get_confusion_matrix(self) -> Dict[str, Dict[str, int]]:
        """
        Get the confusion matrix of predicted vs actual intents.
        
        Returns:
            Dict mapping predicted -> actual -> count
        """
        return dict(self._confusion_matrix)
    
    def get_top_confused_pairs(
        self, 
        limit: int = 10
    ) -> List[Tuple[str, str, int]]:
        """
        Get the most commonly confused intent pairs.
        
        Returns:
            List of (predicted, actual, count) tuples, sorted by count descending
        """
        pairs = []
        for predicted, actuals in self._confusion_matrix.items():
            for actual, count in actuals.items():
                if predicted != actual:  # Only real confusions
                    pairs.append((predicted, actual, count))
        
        pairs.sort(key=lambda x: -x[2])
        return pairs[:limit]
    
    def get_problematic_queries(
        self, 
        min_corrections: int = 2
    ) -> List[Tuple[str, int]]:
        """
        Get queries that have been corrected multiple times.
        
        These are good candidates for adding to training data.
        
        Args:
            min_corrections: Minimum correction count to include
            
        Returns:
            List of (query, count) tuples
        """
        query_counts: Dict[str, int] = defaultdict(int)
        
        for record in self._corrections:
            # Normalize query for comparison
            normalized = record.query.lower().strip()
            query_counts[normalized] += 1
        
        problematic = [
            (q, c) for q, c in query_counts.items()
            if c >= min_corrections
        ]
        problematic.sort(key=lambda x: -x[1])
        
        return problematic
    
    def get_corrections_for_intent(self, intent: str) -> List[CorrectionRecord]:
        """Get all corrections where this intent was predicted or corrected."""
        return [
            r for r in self._corrections
            if r.predicted_intent == intent or r.corrected_intent == intent
        ]
    
    def get_metrics(self) -> LearningMetrics:
        """
        Get aggregated learning metrics.
        
        Returns:
            LearningMetrics with correction rates, confusion pairs, etc.
        """
        total_corrections = len(self._corrections)
        total_confirmations = len(self._confirmations)
        total_queries = total_corrections + total_confirmations
        
        correction_rate = (
            total_corrections / total_queries 
            if total_queries > 0 else 0.0
        )
        
        # Count corrections by predicted intent
        corrections_by_intent: Dict[str, int] = defaultdict(int)
        for record in self._corrections:
            corrections_by_intent[record.predicted_intent] += 1
        
        # Calculate improvement trend (compare recent vs older)
        trend = 0.0
        if len(self._corrections) >= 20:
            # Split into older half and recent half
            mid = len(self._corrections) // 2
            older = self._corrections[:mid]
            recent = self._corrections[mid:]
            
            # Compare correction rates (lower is better)
            # Positive trend means improving (fewer corrections recently)
            older_rate = len(older) / mid if mid > 0 else 0
            recent_rate = len(recent) / (len(self._corrections) - mid)
            trend = older_rate - recent_rate
        
        return LearningMetrics(
            total_queries=total_queries,
            total_corrections=total_corrections,
            total_confirmations=total_confirmations,
            correction_rate=correction_rate,
            top_confused_pairs=self.get_top_confused_pairs(),
            corrections_by_intent=dict(corrections_by_intent),
            improvement_trend=trend,
        )
    
    def export_for_retraining(
        self,
        output_path: Optional[Path] = None,
        min_count: int = 1,
    ) -> str:
        """
        Export corrections as YAML training examples.
        
        Args:
            output_path: Path to write YAML file
            min_count: Minimum correction count to include
            
        Returns:
            YAML string with training examples
        """
        # Group corrections by corrected intent
        intent_examples: Dict[str, List[str]] = defaultdict(list)
        
        for record in self._corrections:
            intent_examples[record.corrected_intent].append(record.query)
        
        # Build YAML structure
        import yaml
        
        intents_data = []
        for intent, queries in intent_examples.items():
            if len(queries) >= min_count:
                # Deduplicate
                unique_queries = list(set(queries))
                intents_data.append({
                    "intent": intent,
                    "examples": unique_queries,
                    "source": "corrections",
                })
        
        yaml_data = {
            "version": "1.0",
            "generated": datetime.utcnow().isoformat(),
            "source": "active_learning_corrections",
            "intents": intents_data,
        }
        
        yaml_str = yaml.dump(yaml_data, default_flow_style=False, sort_keys=False)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(yaml_str)
            logger.info(f"Exported {len(intents_data)} intents to {output_path}")
        
        return yaml_str
    
    def clear_all(self) -> None:
        """Clear all feedback data (use with caution!)."""
        self._corrections.clear()
        self._confirmations.clear()
        self._confusion_matrix.clear()
        
        # Remove files
        if self.corrections_file.exists():
            self.corrections_file.unlink()
        if self.confirmations_file.exists():
            self.confirmations_file.unlink()
        
        logger.warning("Cleared all active learning data")


# =============================================================================
# Singleton Instance
# =============================================================================

_active_learner: Optional[ActiveLearner] = None


def get_active_learner() -> ActiveLearner:
    """Get the singleton active learner instance."""
    global _active_learner
    if _active_learner is None:
        _active_learner = ActiveLearner()
    return _active_learner
