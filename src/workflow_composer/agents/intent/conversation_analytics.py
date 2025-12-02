"""
Conversation Analytics for Professional Chat Agent.

This module provides comprehensive analytics and metrics tracking for:
- Conversation quality and performance metrics
- Intent recognition accuracy tracking
- Response effectiveness analysis
- User satisfaction indicators
- Conversation flow patterns
- Error and recovery tracking
- A/B test metrics integration

Phase 3 of the Professional Chat Agent implementation.
"""

import json
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import statistics
import logging

from .dialog_state_machine import DialogState, DialogEvent


logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics tracked."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class ConversationOutcome(Enum):
    """Possible conversation outcomes."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    ABANDONED = "abandoned"
    ERROR = "error"
    ESCALATED = "escalated"
    TIMEOUT = "timeout"
    USER_CANCELLED = "user_cancelled"


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationMetrics:
    """Metrics for a single conversation."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    turn_count: int = 0
    intent_changes: int = 0
    clarification_count: int = 0
    error_count: int = 0
    recovery_count: int = 0
    slot_prompts: int = 0
    disambiguation_count: int = 0
    outcome: Optional[ConversationOutcome] = None
    user_satisfaction: Optional[float] = None  # 1-5 scale
    intents_detected: List[str] = field(default_factory=list)
    entities_extracted: List[str] = field(default_factory=list)
    states_visited: List[DialogState] = field(default_factory=list)
    response_times: List[float] = field(default_factory=list)
    
    @property
    def duration(self) -> Optional[float]:
        """Get conversation duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def avg_response_time(self) -> Optional[float]:
        """Get average response time."""
        if self.response_times:
            return statistics.mean(self.response_times)
        return None
    
    @property
    def success_rate_contribution(self) -> float:
        """Rate indicating conversation success (0-1)."""
        if self.outcome == ConversationOutcome.SUCCESS:
            return 1.0
        elif self.outcome == ConversationOutcome.PARTIAL_SUCCESS:
            return 0.5
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "turn_count": self.turn_count,
            "intent_changes": self.intent_changes,
            "clarification_count": self.clarification_count,
            "error_count": self.error_count,
            "recovery_count": self.recovery_count,
            "slot_prompts": self.slot_prompts,
            "disambiguation_count": self.disambiguation_count,
            "outcome": self.outcome.value if self.outcome else None,
            "user_satisfaction": self.user_satisfaction,
            "intents_detected": self.intents_detected,
            "entities_extracted": self.entities_extracted,
            "states_visited": [s.name for s in self.states_visited],
            "response_times": self.response_times,
            "duration": self.duration,
            "avg_response_time": self.avg_response_time,
        }


@dataclass
class IntentMetrics:
    """Metrics for a specific intent."""
    intent_name: str
    detection_count: int = 0
    successful_completions: int = 0
    failed_completions: int = 0
    avg_confidence: float = 0.0
    confidence_samples: List[float] = field(default_factory=list)
    avg_turns_to_complete: float = 0.0
    turns_samples: List[int] = field(default_factory=list)
    misclassification_count: int = 0
    slot_fill_rate: float = 0.0  # % of slots successfully filled
    
    @property
    def completion_rate(self) -> float:
        """Get successful completion rate."""
        total = self.successful_completions + self.failed_completions
        if total == 0:
            return 0.0
        return self.successful_completions / total
    
    def record_detection(self, confidence: float):
        """Record an intent detection."""
        self.detection_count += 1
        self.confidence_samples.append(confidence)
        self.avg_confidence = statistics.mean(self.confidence_samples)
    
    def record_completion(self, success: bool, turns: int):
        """Record an intent completion."""
        if success:
            self.successful_completions += 1
        else:
            self.failed_completions += 1
        self.turns_samples.append(turns)
        self.avg_turns_to_complete = statistics.mean(self.turns_samples)


@dataclass
class StateMetrics:
    """Metrics for a dialog state."""
    state: DialogState
    entry_count: int = 0
    exit_count: int = 0
    total_time_seconds: float = 0.0
    time_samples: List[float] = field(default_factory=list)
    timeout_count: int = 0
    error_count: int = 0
    transitions_from: Dict[DialogState, int] = field(default_factory=dict)
    transitions_to: Dict[DialogState, int] = field(default_factory=dict)
    
    @property
    def avg_time_in_state(self) -> float:
        """Get average time spent in state."""
        if self.time_samples:
            return statistics.mean(self.time_samples)
        return 0.0
    
    def record_entry(self):
        """Record state entry."""
        self.entry_count += 1
    
    def record_exit(self, duration: float, next_state: Optional[DialogState] = None):
        """Record state exit with duration."""
        self.exit_count += 1
        self.total_time_seconds += duration
        self.time_samples.append(duration)
        if next_state:
            self.transitions_to[next_state] = self.transitions_to.get(next_state, 0) + 1


@dataclass
class AggregateMetrics:
    """Aggregated metrics over a time window."""
    window_start: datetime
    window_end: datetime
    total_conversations: int = 0
    successful_conversations: int = 0
    avg_conversation_duration: float = 0.0
    avg_turns_per_conversation: float = 0.0
    avg_response_time: float = 0.0
    clarification_rate: float = 0.0  # clarifications per conversation
    error_rate: float = 0.0  # errors per conversation
    escalation_rate: float = 0.0  # % escalated to human
    abandonment_rate: float = 0.0  # % abandoned
    user_satisfaction_avg: Optional[float] = None
    top_intents: List[Tuple[str, int]] = field(default_factory=list)
    top_errors: List[Tuple[str, int]] = field(default_factory=list)
    state_distribution: Dict[str, float] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Get conversation success rate."""
        if self.total_conversations == 0:
            return 0.0
        return self.successful_conversations / self.total_conversations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "total_conversations": self.total_conversations,
            "successful_conversations": self.successful_conversations,
            "success_rate": self.success_rate,
            "avg_conversation_duration": self.avg_conversation_duration,
            "avg_turns_per_conversation": self.avg_turns_per_conversation,
            "avg_response_time": self.avg_response_time,
            "clarification_rate": self.clarification_rate,
            "error_rate": self.error_rate,
            "escalation_rate": self.escalation_rate,
            "abandonment_rate": self.abandonment_rate,
            "user_satisfaction_avg": self.user_satisfaction_avg,
            "top_intents": self.top_intents,
            "top_errors": self.top_errors,
            "state_distribution": self.state_distribution,
        }


class MetricsStorage(ABC):
    """Abstract base class for metrics storage."""
    
    @abstractmethod
    def store_conversation(self, metrics: ConversationMetrics) -> None:
        """Store conversation metrics."""
        pass
    
    @abstractmethod
    def get_conversations(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[ConversationMetrics]:
        """Retrieve conversation metrics."""
        pass
    
    @abstractmethod
    def store_metric(self, name: str, point: MetricPoint) -> None:
        """Store a single metric point."""
        pass
    
    @abstractmethod
    def get_metrics(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[MetricPoint]:
        """Retrieve metric points."""
        pass


class InMemoryMetricsStorage(MetricsStorage):
    """In-memory storage for metrics (for development/testing)."""
    
    def __init__(self, max_conversations: int = 10000, max_metrics: int = 100000):
        self.max_conversations = max_conversations
        self.max_metrics = max_metrics
        self.conversations: List[ConversationMetrics] = []
        self.metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def store_conversation(self, metrics: ConversationMetrics) -> None:
        """Store conversation metrics."""
        with self._lock:
            self.conversations.append(metrics)
            # Trim if over limit
            if len(self.conversations) > self.max_conversations:
                self.conversations = self.conversations[-self.max_conversations:]
    
    def get_conversations(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[ConversationMetrics]:
        """Retrieve conversation metrics."""
        with self._lock:
            result = self.conversations.copy()
        
        # Filter by time
        if start_time:
            result = [c for c in result if c.start_time >= start_time]
        if end_time:
            result = [c for c in result if c.start_time <= end_time]
        
        # Limit results
        return result[-limit:]
    
    def store_metric(self, name: str, point: MetricPoint) -> None:
        """Store a single metric point."""
        with self._lock:
            self.metrics[name].append(point)
            # Trim if over limit per metric
            if len(self.metrics[name]) > self.max_metrics:
                self.metrics[name] = self.metrics[name][-self.max_metrics:]
    
    def get_metrics(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[MetricPoint]:
        """Retrieve metric points."""
        with self._lock:
            points = self.metrics.get(name, []).copy()
        
        # Filter by time
        if start_time:
            points = [p for p in points if p.timestamp >= start_time]
        if end_time:
            points = [p for p in points if p.timestamp <= end_time]
        
        return points
    
    def clear(self) -> None:
        """Clear all stored metrics."""
        with self._lock:
            self.conversations.clear()
            self.metrics.clear()


class FileMetricsStorage(MetricsStorage):
    """File-based storage for metrics (for persistence)."""
    
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.conversations_file = self.base_path / "conversations.jsonl"
        self.metrics_dir = self.base_path / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)
    
    def store_conversation(self, metrics: ConversationMetrics) -> None:
        """Store conversation metrics to file."""
        with open(self.conversations_file, 'a') as f:
            f.write(json.dumps(metrics.to_dict()) + '\n')
    
    def get_conversations(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[ConversationMetrics]:
        """Retrieve conversation metrics from file."""
        if not self.conversations_file.exists():
            return []
        
        conversations = []
        with open(self.conversations_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    # Reconstruct ConversationMetrics (simplified)
                    conv = ConversationMetrics(
                        session_id=data["session_id"],
                        start_time=datetime.fromisoformat(data["start_time"]),
                        end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
                        turn_count=data.get("turn_count", 0),
                        intent_changes=data.get("intent_changes", 0),
                        clarification_count=data.get("clarification_count", 0),
                        error_count=data.get("error_count", 0),
                        outcome=ConversationOutcome(data["outcome"]) if data.get("outcome") else None,
                    )
                    
                    # Filter by time
                    if start_time and conv.start_time < start_time:
                        continue
                    if end_time and conv.start_time > end_time:
                        continue
                    
                    conversations.append(conv)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Failed to parse conversation: {e}")
                    continue
        
        return conversations[-limit:]
    
    def store_metric(self, name: str, point: MetricPoint) -> None:
        """Store a single metric point."""
        metric_file = self.metrics_dir / f"{name}.jsonl"
        data = {
            "timestamp": point.timestamp.isoformat(),
            "value": point.value,
            "tags": point.tags,
            "metadata": point.metadata
        }
        with open(metric_file, 'a') as f:
            f.write(json.dumps(data) + '\n')
    
    def get_metrics(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[MetricPoint]:
        """Retrieve metric points from file."""
        metric_file = self.metrics_dir / f"{name}.jsonl"
        if not metric_file.exists():
            return []
        
        points = []
        with open(metric_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    point = MetricPoint(
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        value=data["value"],
                        tags=data.get("tags", {}),
                        metadata=data.get("metadata", {})
                    )
                    
                    if start_time and point.timestamp < start_time:
                        continue
                    if end_time and point.timestamp > end_time:
                        continue
                    
                    points.append(point)
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        
        return points


class ConversationAnalytics:
    """
    Main analytics engine for conversation tracking and analysis.
    
    Features:
    - Real-time metrics collection
    - Conversation tracking
    - Intent/state metrics
    - Aggregation and reporting
    - Alert thresholds
    """
    
    def __init__(
        self,
        storage: Optional[MetricsStorage] = None,
        alert_callbacks: Optional[List[Callable[[str, Any], None]]] = None
    ):
        self.storage = storage or InMemoryMetricsStorage()
        self.alert_callbacks = alert_callbacks or []
        
        # Active conversations
        self._active_conversations: Dict[str, ConversationMetrics] = {}
        self._conversation_start_times: Dict[str, Dict[DialogState, datetime]] = {}
        
        # Per-intent metrics
        self._intent_metrics: Dict[str, IntentMetrics] = {}
        
        # Per-state metrics
        self._state_metrics: Dict[DialogState, StateMetrics] = {
            state: StateMetrics(state=state) for state in DialogState
        }
        
        # Alert thresholds
        self._alert_thresholds: Dict[str, Tuple[float, str]] = {
            "error_rate": (0.1, "Error rate exceeded 10%"),
            "avg_response_time": (5.0, "Average response time exceeded 5 seconds"),
            "abandonment_rate": (0.3, "Abandonment rate exceeded 30%"),
        }
        
        self._lock = threading.Lock()
    
    def start_conversation(self, session_id: str) -> ConversationMetrics:
        """Start tracking a new conversation."""
        with self._lock:
            metrics = ConversationMetrics(
                session_id=session_id,
                start_time=datetime.now()
            )
            self._active_conversations[session_id] = metrics
            self._conversation_start_times[session_id] = {}
            
        # Record metric
        self.storage.store_metric(
            "conversations.started",
            MetricPoint(timestamp=datetime.now(), value=1)
        )
        
        return metrics
    
    def end_conversation(
        self,
        session_id: str,
        outcome: ConversationOutcome,
        user_satisfaction: Optional[float] = None
    ) -> Optional[ConversationMetrics]:
        """End conversation tracking."""
        with self._lock:
            if session_id not in self._active_conversations:
                return None
            
            metrics = self._active_conversations.pop(session_id)
            self._conversation_start_times.pop(session_id, None)
            
            metrics.end_time = datetime.now()
            metrics.outcome = outcome
            metrics.user_satisfaction = user_satisfaction
        
        # Store conversation
        self.storage.store_conversation(metrics)
        
        # Record metric
        self.storage.store_metric(
            f"conversations.outcome.{outcome.value}",
            MetricPoint(timestamp=datetime.now(), value=1)
        )
        
        return metrics
    
    def record_turn(
        self,
        session_id: str,
        response_time: float,
        intent: Optional[str] = None,
        confidence: Optional[float] = None,
        entities: Optional[List[str]] = None
    ) -> None:
        """Record a conversation turn."""
        with self._lock:
            if session_id not in self._active_conversations:
                return
            
            metrics = self._active_conversations[session_id]
            metrics.turn_count += 1
            metrics.response_times.append(response_time)
            
            if intent:
                if intent not in metrics.intents_detected:
                    metrics.intents_detected.append(intent)
                else:
                    metrics.intent_changes += 1
                
                # Update intent metrics
                if intent not in self._intent_metrics:
                    self._intent_metrics[intent] = IntentMetrics(intent_name=intent)
                if confidence:
                    self._intent_metrics[intent].record_detection(confidence)
            
            if entities:
                for entity in entities:
                    if entity not in metrics.entities_extracted:
                        metrics.entities_extracted.append(entity)
        
        # Record response time metric
        self.storage.store_metric(
            "response_time",
            MetricPoint(
                timestamp=datetime.now(),
                value=response_time,
                tags={"session_id": session_id}
            )
        )
    
    def record_state_transition(
        self,
        session_id: str,
        from_state: DialogState,
        to_state: DialogState,
        event: Optional[DialogEvent] = None
    ) -> None:
        """Record a state transition."""
        with self._lock:
            if session_id not in self._active_conversations:
                return
            
            metrics = self._active_conversations[session_id]
            metrics.states_visited.append(to_state)
            
            # Update state metrics
            self._state_metrics[from_state].exit_count += 1
            self._state_metrics[from_state].transitions_to[to_state] = \
                self._state_metrics[from_state].transitions_to.get(to_state, 0) + 1
            
            self._state_metrics[to_state].entry_count += 1
            self._state_metrics[to_state].transitions_from[from_state] = \
                self._state_metrics[to_state].transitions_from.get(from_state, 0) + 1
            
            # Calculate time in previous state
            if session_id in self._conversation_start_times:
                start_times = self._conversation_start_times[session_id]
                if from_state in start_times:
                    duration = (datetime.now() - start_times[from_state]).total_seconds()
                    self._state_metrics[from_state].time_samples.append(duration)
                    self._state_metrics[from_state].total_time_seconds += duration
                
                # Record start time for new state
                start_times[to_state] = datetime.now()
        
        # Record metric
        self.storage.store_metric(
            f"state.transition.{from_state.name.lower()}_to_{to_state.name.lower()}",
            MetricPoint(timestamp=datetime.now(), value=1)
        )
    
    def record_clarification(self, session_id: str) -> None:
        """Record a clarification request."""
        with self._lock:
            if session_id in self._active_conversations:
                self._active_conversations[session_id].clarification_count += 1
        
        self.storage.store_metric(
            "clarifications",
            MetricPoint(timestamp=datetime.now(), value=1)
        )
    
    def record_error(self, session_id: str, error_type: str) -> None:
        """Record an error occurrence."""
        with self._lock:
            if session_id in self._active_conversations:
                self._active_conversations[session_id].error_count += 1
        
        self.storage.store_metric(
            f"errors.{error_type}",
            MetricPoint(timestamp=datetime.now(), value=1)
        )
        
        self.storage.store_metric(
            "errors.total",
            MetricPoint(timestamp=datetime.now(), value=1)
        )
    
    def record_recovery(self, session_id: str, recovery_type: str) -> None:
        """Record an error recovery."""
        with self._lock:
            if session_id in self._active_conversations:
                self._active_conversations[session_id].recovery_count += 1
        
        self.storage.store_metric(
            f"recoveries.{recovery_type}",
            MetricPoint(timestamp=datetime.now(), value=1)
        )
    
    def record_slot_prompt(self, session_id: str, slot_name: str) -> None:
        """Record a slot filling prompt."""
        with self._lock:
            if session_id in self._active_conversations:
                self._active_conversations[session_id].slot_prompts += 1
        
        self.storage.store_metric(
            f"slot_prompts.{slot_name}",
            MetricPoint(timestamp=datetime.now(), value=1)
        )
    
    def record_disambiguation(self, session_id: str, options_count: int) -> None:
        """Record a disambiguation request."""
        with self._lock:
            if session_id in self._active_conversations:
                self._active_conversations[session_id].disambiguation_count += 1
        
        self.storage.store_metric(
            "disambiguations",
            MetricPoint(
                timestamp=datetime.now(),
                value=1,
                metadata={"options_count": options_count}
            )
        )
    
    def record_intent_completion(
        self,
        intent: str,
        success: bool,
        turns: int
    ) -> None:
        """Record an intent completion."""
        with self._lock:
            if intent not in self._intent_metrics:
                self._intent_metrics[intent] = IntentMetrics(intent_name=intent)
            self._intent_metrics[intent].record_completion(success, turns)
        
        outcome = "success" if success else "failure"
        self.storage.store_metric(
            f"intent.{intent}.{outcome}",
            MetricPoint(timestamp=datetime.now(), value=1)
        )
    
    def get_aggregate_metrics(
        self,
        window_hours: int = 24
    ) -> AggregateMetrics:
        """Get aggregated metrics for a time window."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=window_hours)
        
        conversations = self.storage.get_conversations(
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        
        if not conversations:
            return AggregateMetrics(
                window_start=start_time,
                window_end=end_time
            )
        
        # Calculate aggregates
        total = len(conversations)
        successful = sum(1 for c in conversations 
                        if c.outcome in [ConversationOutcome.SUCCESS, ConversationOutcome.PARTIAL_SUCCESS])
        escalated = sum(1 for c in conversations if c.outcome == ConversationOutcome.ESCALATED)
        abandoned = sum(1 for c in conversations if c.outcome == ConversationOutcome.ABANDONED)
        
        durations = [c.duration for c in conversations if c.duration]
        turns = [c.turn_count for c in conversations]
        response_times = []
        for c in conversations:
            response_times.extend(c.response_times)
        
        clarifications = sum(c.clarification_count for c in conversations)
        errors = sum(c.error_count for c in conversations)
        
        satisfaction_scores = [c.user_satisfaction for c in conversations if c.user_satisfaction]
        
        # Top intents
        intent_counts: Dict[str, int] = defaultdict(int)
        for c in conversations:
            for intent in c.intents_detected:
                intent_counts[intent] += 1
        top_intents = sorted(intent_counts.items(), key=lambda x: -x[1])[:10]
        
        # State distribution
        state_counts: Dict[str, int] = defaultdict(int)
        for c in conversations:
            for state in c.states_visited:
                state_counts[state.name] += 1
        total_states = sum(state_counts.values()) or 1
        state_distribution = {k: v/total_states for k, v in state_counts.items()}
        
        aggregate = AggregateMetrics(
            window_start=start_time,
            window_end=end_time,
            total_conversations=total,
            successful_conversations=successful,
            avg_conversation_duration=statistics.mean(durations) if durations else 0,
            avg_turns_per_conversation=statistics.mean(turns) if turns else 0,
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            clarification_rate=clarifications / total if total else 0,
            error_rate=errors / total if total else 0,
            escalation_rate=escalated / total if total else 0,
            abandonment_rate=abandoned / total if total else 0,
            user_satisfaction_avg=statistics.mean(satisfaction_scores) if satisfaction_scores else None,
            top_intents=top_intents,
            state_distribution=state_distribution,
        )
        
        # Check alerts
        self._check_alerts(aggregate)
        
        return aggregate
    
    def get_intent_metrics(self, intent: str) -> Optional[IntentMetrics]:
        """Get metrics for a specific intent."""
        with self._lock:
            return self._intent_metrics.get(intent)
    
    def get_all_intent_metrics(self) -> Dict[str, IntentMetrics]:
        """Get metrics for all intents."""
        with self._lock:
            return self._intent_metrics.copy()
    
    def get_state_metrics(self, state: DialogState) -> StateMetrics:
        """Get metrics for a specific state."""
        with self._lock:
            return self._state_metrics[state]
    
    def get_all_state_metrics(self) -> Dict[DialogState, StateMetrics]:
        """Get metrics for all states."""
        with self._lock:
            return self._state_metrics.copy()
    
    def get_active_conversation_count(self) -> int:
        """Get number of active conversations."""
        with self._lock:
            return len(self._active_conversations)
    
    def set_alert_threshold(self, metric: str, threshold: float, message: str) -> None:
        """Set an alert threshold for a metric."""
        self._alert_thresholds[metric] = (threshold, message)
    
    def _check_alerts(self, aggregate: AggregateMetrics) -> None:
        """Check alert thresholds and trigger callbacks."""
        alerts = []
        
        if "error_rate" in self._alert_thresholds:
            threshold, message = self._alert_thresholds["error_rate"]
            if aggregate.error_rate > threshold:
                alerts.append(("error_rate", message, aggregate.error_rate))
        
        if "avg_response_time" in self._alert_thresholds:
            threshold, message = self._alert_thresholds["avg_response_time"]
            if aggregate.avg_response_time > threshold:
                alerts.append(("avg_response_time", message, aggregate.avg_response_time))
        
        if "abandonment_rate" in self._alert_thresholds:
            threshold, message = self._alert_thresholds["abandonment_rate"]
            if aggregate.abandonment_rate > threshold:
                alerts.append(("abandonment_rate", message, aggregate.abandonment_rate))
        
        # Trigger callbacks
        for metric, message, value in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(metric, {"message": message, "value": value})
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")
    
    def export_report(self, window_hours: int = 24) -> Dict[str, Any]:
        """Export a comprehensive analytics report."""
        aggregate = self.get_aggregate_metrics(window_hours)
        
        intent_metrics = {}
        for name, metrics in self.get_all_intent_metrics().items():
            intent_metrics[name] = {
                "detection_count": metrics.detection_count,
                "completion_rate": metrics.completion_rate,
                "avg_confidence": metrics.avg_confidence,
                "avg_turns_to_complete": metrics.avg_turns_to_complete,
            }
        
        state_metrics = {}
        for state, metrics in self.get_all_state_metrics().items():
            state_metrics[state.name] = {
                "entry_count": metrics.entry_count,
                "avg_time_in_state": metrics.avg_time_in_state,
                "timeout_count": metrics.timeout_count,
                "error_count": metrics.error_count,
            }
        
        return {
            "generated_at": datetime.now().isoformat(),
            "window_hours": window_hours,
            "aggregate": aggregate.to_dict(),
            "intent_metrics": intent_metrics,
            "state_metrics": state_metrics,
            "active_conversations": self.get_active_conversation_count(),
        }


class AnalyticsDashboard:
    """
    Dashboard for visualizing conversation analytics.
    
    Provides formatted reports and summaries.
    """
    
    def __init__(self, analytics: ConversationAnalytics):
        self.analytics = analytics
    
    def get_summary(self, window_hours: int = 24) -> str:
        """Get a formatted text summary of analytics."""
        aggregate = self.analytics.get_aggregate_metrics(window_hours)
        
        lines = [
            f"📊 Conversation Analytics Summary ({window_hours}h window)",
            "=" * 50,
            "",
            f"📈 Conversations: {aggregate.total_conversations}",
            f"✅ Success Rate: {aggregate.success_rate:.1%}",
            f"⏱️ Avg Duration: {aggregate.avg_conversation_duration:.1f}s",
            f"💬 Avg Turns: {aggregate.avg_turns_per_conversation:.1f}",
            f"⚡ Avg Response Time: {aggregate.avg_response_time:.2f}s",
            "",
            "📉 Rates:",
            f"  • Clarification: {aggregate.clarification_rate:.2f} per conversation",
            f"  • Error: {aggregate.error_rate:.2f} per conversation",
            f"  • Escalation: {aggregate.escalation_rate:.1%}",
            f"  • Abandonment: {aggregate.abandonment_rate:.1%}",
        ]
        
        if aggregate.user_satisfaction_avg:
            lines.append(f"")
            lines.append(f"⭐ User Satisfaction: {aggregate.user_satisfaction_avg:.1f}/5")
        
        if aggregate.top_intents:
            lines.append("")
            lines.append("🎯 Top Intents:")
            for intent, count in aggregate.top_intents[:5]:
                lines.append(f"  • {intent}: {count}")
        
        return "\n".join(lines)
    
    def get_intent_report(self) -> str:
        """Get a formatted report of intent metrics."""
        metrics = self.analytics.get_all_intent_metrics()
        
        if not metrics:
            return "No intent metrics recorded yet."
        
        lines = [
            "🎯 Intent Performance Report",
            "=" * 50,
        ]
        
        # Sort by detection count
        sorted_intents = sorted(
            metrics.items(),
            key=lambda x: -x[1].detection_count
        )
        
        for intent, m in sorted_intents:
            lines.append("")
            lines.append(f"Intent: {intent}")
            lines.append(f"  Detections: {m.detection_count}")
            lines.append(f"  Completion Rate: {m.completion_rate:.1%}")
            lines.append(f"  Avg Confidence: {m.avg_confidence:.2f}")
            lines.append(f"  Avg Turns to Complete: {m.avg_turns_to_complete:.1f}")
        
        return "\n".join(lines)
    
    def get_state_report(self) -> str:
        """Get a formatted report of state metrics."""
        metrics = self.analytics.get_all_state_metrics()
        
        lines = [
            "🔄 Dialog State Report",
            "=" * 50,
        ]
        
        # Sort by entry count
        sorted_states = sorted(
            metrics.items(),
            key=lambda x: -x[1].entry_count
        )
        
        for state, m in sorted_states:
            if m.entry_count == 0:
                continue
            lines.append("")
            lines.append(f"State: {state.name}")
            lines.append(f"  Entries: {m.entry_count}")
            lines.append(f"  Avg Time: {m.avg_time_in_state:.2f}s")
            lines.append(f"  Timeouts: {m.timeout_count}")
            lines.append(f"  Errors: {m.error_count}")
        
        return "\n".join(lines)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status based on metrics."""
        aggregate = self.analytics.get_aggregate_metrics(window_hours=1)
        
        status = "healthy"
        issues = []
        
        if aggregate.error_rate > 0.1:
            status = "degraded"
            issues.append(f"High error rate: {aggregate.error_rate:.1%}")
        
        if aggregate.avg_response_time > 5.0:
            status = "degraded"
            issues.append(f"Slow response time: {aggregate.avg_response_time:.1f}s")
        
        if aggregate.abandonment_rate > 0.3:
            status = "degraded"
            issues.append(f"High abandonment: {aggregate.abandonment_rate:.1%}")
        
        if aggregate.success_rate < 0.5 and aggregate.total_conversations > 10:
            status = "critical"
            issues.append(f"Low success rate: {aggregate.success_rate:.1%}")
        
        return {
            "status": status,
            "active_conversations": self.analytics.get_active_conversation_count(),
            "success_rate": aggregate.success_rate,
            "avg_response_time": aggregate.avg_response_time,
            "issues": issues,
            "checked_at": datetime.now().isoformat()
        }


# Singleton instances
_analytics_instance: Optional[ConversationAnalytics] = None
_dashboard_instance: Optional[AnalyticsDashboard] = None


def get_conversation_analytics(
    storage: Optional[MetricsStorage] = None
) -> ConversationAnalytics:
    """Get or create the singleton analytics instance."""
    global _analytics_instance
    if _analytics_instance is None:
        _analytics_instance = ConversationAnalytics(storage=storage)
    return _analytics_instance


def get_analytics_dashboard() -> AnalyticsDashboard:
    """Get or create the singleton dashboard instance."""
    global _dashboard_instance
    if _dashboard_instance is None:
        _dashboard_instance = AnalyticsDashboard(get_conversation_analytics())
    return _dashboard_instance


def reset_analytics() -> None:
    """Reset analytics (for testing)."""
    global _analytics_instance, _dashboard_instance
    _analytics_instance = None
    _dashboard_instance = None
