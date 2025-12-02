"""
A/B Testing Framework for Chat Agent.

Phase 5 of Professional Chat Agent implementation.

Features:
- Experiment variant management
- User/session assignment
- Metrics collection by variant
- Statistical analysis
- Experiment lifecycle management
"""

import hashlib
import json
import logging
import random
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
import math

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Experiment lifecycle status."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class AssignmentStrategy(Enum):
    """Strategy for assigning users to variants."""
    RANDOM = "random"
    DETERMINISTIC = "deterministic"  # Hash-based, consistent per user
    ROUND_ROBIN = "round_robin"
    WEIGHTED_RANDOM = "weighted_random"


class MetricType(Enum):
    """Type of metric being tracked."""
    COUNTER = "counter"
    GAUGE = "gauge"
    RATE = "rate"
    PERCENTAGE = "percentage"
    DURATION = "duration"


@dataclass
class Variant:
    """Represents a variant in an A/B test."""
    id: str
    name: str
    description: str = ""
    weight: float = 1.0  # Relative weight for assignment
    config: Dict[str, Any] = field(default_factory=dict)
    is_control: bool = False
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, Variant):
            return self.id == other.id
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "weight": self.weight,
            "config": self.config,
            "is_control": self.is_control,
        }


@dataclass
class MetricValue:
    """A single metric value with metadata."""
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricDefinition:
    """Definition for a metric to track."""
    name: str
    type: MetricType
    description: str = ""
    unit: str = ""
    higher_is_better: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "unit": self.unit,
            "higher_is_better": self.higher_is_better,
        }


@dataclass 
class VariantMetrics:
    """Metrics collected for a specific variant."""
    variant_id: str
    assigned_count: int = 0
    metrics: Dict[str, List[MetricValue]] = field(default_factory=dict)
    
    def add_metric(self, metric_name: str, value: float, metadata: Optional[Dict] = None) -> None:
        """Add a metric value."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(MetricValue(
            value=value,
            metadata=metadata or {}
        ))
    
    def get_count(self, metric_name: str) -> int:
        """Get count of metric values."""
        return len(self.metrics.get(metric_name, []))
    
    def get_sum(self, metric_name: str) -> float:
        """Get sum of metric values."""
        values = self.metrics.get(metric_name, [])
        return sum(v.value for v in values)
    
    def get_mean(self, metric_name: str) -> Optional[float]:
        """Get mean of metric values."""
        values = self.metrics.get(metric_name, [])
        if not values:
            return None
        return self.get_sum(metric_name) / len(values)
    
    def get_variance(self, metric_name: str) -> Optional[float]:
        """Get variance of metric values."""
        values = self.metrics.get(metric_name, [])
        if len(values) < 2:
            return None
        mean = self.get_mean(metric_name)
        if mean is None:
            return None
        return sum((v.value - mean) ** 2 for v in values) / (len(values) - 1)
    
    def get_std(self, metric_name: str) -> Optional[float]:
        """Get standard deviation of metric values."""
        variance = self.get_variance(metric_name)
        if variance is None:
            return None
        return math.sqrt(variance)
    
    def get_min(self, metric_name: str) -> Optional[float]:
        """Get minimum metric value."""
        values = self.metrics.get(metric_name, [])
        if not values:
            return None
        return min(v.value for v in values)
    
    def get_max(self, metric_name: str) -> Optional[float]:
        """Get maximum metric value."""
        values = self.metrics.get(metric_name, [])
        if not values:
            return None
        return max(v.value for v in values)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variant_id": self.variant_id,
            "assigned_count": self.assigned_count,
            "metrics": {
                name: [{"value": v.value, "timestamp": v.timestamp.isoformat(), "metadata": v.metadata}
                       for v in values]
                for name, values in self.metrics.items()
            }
        }


@dataclass
class Experiment:
    """Represents an A/B test experiment."""
    id: str
    name: str
    description: str = ""
    variants: List[Variant] = field(default_factory=list)
    metric_definitions: List[MetricDefinition] = field(default_factory=list)
    status: ExperimentStatus = ExperimentStatus.DRAFT
    strategy: AssignmentStrategy = AssignmentStrategy.DETERMINISTIC
    
    # Time bounds
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Traffic allocation
    traffic_percentage: float = 100.0  # Percentage of traffic to include
    
    # Targeting
    targeting_rules: Dict[str, Any] = field(default_factory=dict)
    
    # Results
    variant_metrics: Dict[str, VariantMetrics] = field(default_factory=dict)
    assignments: Dict[str, str] = field(default_factory=dict)  # user_id -> variant_id
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    owner: str = ""
    tags: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Initialize variant metrics."""
        for variant in self.variants:
            if variant.id not in self.variant_metrics:
                self.variant_metrics[variant.id] = VariantMetrics(variant_id=variant.id)
    
    def get_variant(self, variant_id: str) -> Optional[Variant]:
        """Get variant by ID."""
        for variant in self.variants:
            if variant.id == variant_id:
                return variant
        return None
    
    def get_control(self) -> Optional[Variant]:
        """Get control variant."""
        for variant in self.variants:
            if variant.is_control:
                return variant
        return None
    
    def is_running(self) -> bool:
        """Check if experiment is currently running."""
        if self.status != ExperimentStatus.RUNNING:
            return False
        
        now = datetime.now()
        if self.start_time and now < self.start_time:
            return False
        if self.end_time and now > self.end_time:
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "variants": [v.to_dict() for v in self.variants],
            "metric_definitions": [m.to_dict() for m in self.metric_definitions],
            "status": self.status.value,
            "strategy": self.strategy.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "traffic_percentage": self.traffic_percentage,
            "targeting_rules": self.targeting_rules,
            "variant_metrics": {k: v.to_dict() for k, v in self.variant_metrics.items()},
            "assignments": self.assignments,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "owner": self.owner,
            "tags": list(self.tags),
        }


class AssignmentAlgorithm(ABC):
    """Abstract base class for assignment algorithms."""
    
    @abstractmethod
    def assign(
        self,
        user_id: str,
        variants: List[Variant],
        **kwargs
    ) -> Variant:
        """Assign a user to a variant."""
        pass


class RandomAssignment(AssignmentAlgorithm):
    """Random assignment to variants."""
    
    def assign(self, user_id: str, variants: List[Variant], **kwargs) -> Variant:
        """Randomly assign user to a variant."""
        return random.choice(variants)


class DeterministicAssignment(AssignmentAlgorithm):
    """Hash-based deterministic assignment (consistent per user)."""
    
    def __init__(self, salt: str = ""):
        self.salt = salt
    
    def assign(self, user_id: str, variants: List[Variant], **kwargs) -> Variant:
        """Assign based on hash of user ID."""
        experiment_id = kwargs.get("experiment_id", "")
        hash_input = f"{self.salt}:{experiment_id}:{user_id}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()
        bucket = int(hash_value[:8], 16) % 100
        
        # Calculate cumulative weights
        total_weight = sum(v.weight for v in variants)
        cumulative = 0
        
        for variant in variants:
            cumulative += (variant.weight / total_weight) * 100
            if bucket < cumulative:
                return variant
        
        return variants[-1]


class RoundRobinAssignment(AssignmentAlgorithm):
    """Round-robin assignment to variants."""
    
    def __init__(self):
        self._index = 0
        self._lock = threading.Lock()
    
    def assign(self, user_id: str, variants: List[Variant], **kwargs) -> Variant:
        """Assign users in round-robin fashion."""
        with self._lock:
            variant = variants[self._index % len(variants)]
            self._index += 1
            return variant


class WeightedRandomAssignment(AssignmentAlgorithm):
    """Weighted random assignment based on variant weights."""
    
    def assign(self, user_id: str, variants: List[Variant], **kwargs) -> Variant:
        """Assign based on variant weights."""
        total_weight = sum(v.weight for v in variants)
        r = random.random() * total_weight
        
        cumulative = 0
        for variant in variants:
            cumulative += variant.weight
            if r < cumulative:
                return variant
        
        return variants[-1]


@dataclass
class StatisticalResult:
    """Result of statistical analysis."""
    control_mean: float
    treatment_mean: float
    lift: float  # Relative improvement
    p_value: float
    confidence_level: float
    is_significant: bool
    sample_size_control: int
    sample_size_treatment: int
    power: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "control_mean": self.control_mean,
            "treatment_mean": self.treatment_mean,
            "lift": self.lift,
            "lift_percentage": f"{self.lift * 100:.2f}%",
            "p_value": self.p_value,
            "confidence_level": self.confidence_level,
            "is_significant": self.is_significant,
            "sample_size_control": self.sample_size_control,
            "sample_size_treatment": self.sample_size_treatment,
            "power": self.power,
        }


class StatisticalAnalyzer:
    """Statistical analysis for A/B tests."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
    
    def welch_t_test(
        self,
        control_values: List[float],
        treatment_values: List[float]
    ) -> StatisticalResult:
        """
        Perform Welch's t-test (unequal variance t-test).
        
        Returns statistical significance result.
        """
        n1 = len(control_values)
        n2 = len(treatment_values)
        
        if n1 < 2 or n2 < 2:
            return StatisticalResult(
                control_mean=sum(control_values) / n1 if n1 else 0,
                treatment_mean=sum(treatment_values) / n2 if n2 else 0,
                lift=0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                is_significant=False,
                sample_size_control=n1,
                sample_size_treatment=n2
            )
        
        mean1 = sum(control_values) / n1
        mean2 = sum(treatment_values) / n2
        
        var1 = sum((x - mean1) ** 2 for x in control_values) / (n1 - 1)
        var2 = sum((x - mean2) ** 2 for x in treatment_values) / (n2 - 1)
        
        # Calculate t-statistic
        se = math.sqrt(var1/n1 + var2/n2) if var1/n1 + var2/n2 > 0 else 1e-10
        t_stat = (mean2 - mean1) / se
        
        # Welch-Satterthwaite degrees of freedom
        if var1 == 0 and var2 == 0:
            df = n1 + n2 - 2
        else:
            numerator = (var1/n1 + var2/n2) ** 2
            denominator = (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1) if (n1 > 1 and n2 > 1) else 1
            df = numerator / denominator if denominator > 0 else 1
        
        # Approximate p-value using normal distribution for large samples
        p_value = self._approximate_p_value(abs(t_stat), df)
        
        # Calculate lift
        lift = (mean2 - mean1) / mean1 if mean1 != 0 else 0
        
        return StatisticalResult(
            control_mean=mean1,
            treatment_mean=mean2,
            lift=lift,
            p_value=p_value,
            confidence_level=self.confidence_level,
            is_significant=p_value < (1 - self.confidence_level),
            sample_size_control=n1,
            sample_size_treatment=n2
        )
    
    def _approximate_p_value(self, t_stat: float, df: float) -> float:
        """
        Approximate two-tailed p-value using normal approximation.
        
        For large df, t-distribution approaches normal distribution.
        """
        # Use normal approximation (good for df > 30)
        if df > 30:
            # Using the complementary error function approximation
            x = abs(t_stat)
            # Simple approximation
            t = 1.0 / (1.0 + 0.2316419 * x)
            d = 0.3989423 * math.exp(-x * x / 2)
            p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
            return 2 * p  # Two-tailed
        else:
            # For small df, use a rougher approximation
            # This is simplified; in production, use scipy.stats
            return 2 * (1 - self._student_t_cdf(abs(t_stat), df))
    
    def _student_t_cdf(self, t: float, df: float) -> float:
        """Approximate Student's t CDF."""
        # Simplified approximation using beta incomplete function
        x = df / (df + t * t)
        # Very rough approximation
        return 0.5 + 0.5 * math.copysign(1, t) * (1 - x ** (0.5 * df))
    
    def calculate_sample_size(
        self,
        baseline_rate: float,
        minimum_detectable_effect: float,
        power: float = 0.8
    ) -> int:
        """
        Calculate required sample size per variant.
        
        Args:
            baseline_rate: Expected conversion rate of control
            minimum_detectable_effect: Minimum relative effect to detect
            power: Statistical power (default 0.8)
        
        Returns:
            Required sample size per variant
        """
        # Z-scores
        alpha = 1 - self.confidence_level
        z_alpha = self._z_score(1 - alpha/2)
        z_beta = self._z_score(power)
        
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)
        
        p_avg = (p1 + p2) / 2
        
        numerator = (z_alpha * math.sqrt(2 * p_avg * (1 - p_avg)) + 
                    z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        denominator = (p2 - p1) ** 2
        
        return int(math.ceil(numerator / denominator)) if denominator > 0 else 1000
    
    def _z_score(self, p: float) -> float:
        """Approximate z-score for probability p."""
        # Approximation for inverse normal CDF
        if p <= 0:
            return -10
        if p >= 1:
            return 10
        
        # Rational approximation
        if p < 0.5:
            t = math.sqrt(-2 * math.log(p))
            return -(2.515517 + 0.802853*t + 0.010328*t*t) / (1 + 1.432788*t + 0.189269*t*t + 0.001308*t*t*t) + t
        else:
            t = math.sqrt(-2 * math.log(1 - p))
            return (2.515517 + 0.802853*t + 0.010328*t*t) / (1 + 1.432788*t + 0.189269*t*t + 0.001308*t*t*t) - t
    
    def chi_square_test(
        self,
        control_successes: int,
        control_total: int,
        treatment_successes: int,
        treatment_total: int
    ) -> StatisticalResult:
        """
        Perform chi-square test for proportions.
        
        Useful for conversion rate comparisons.
        """
        if control_total == 0 or treatment_total == 0:
            return StatisticalResult(
                control_mean=0,
                treatment_mean=0,
                lift=0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                is_significant=False,
                sample_size_control=control_total,
                sample_size_treatment=treatment_total
            )
        
        p1 = control_successes / control_total
        p2 = treatment_successes / treatment_total
        
        # Pooled proportion
        p_pooled = (control_successes + treatment_successes) / (control_total + treatment_total)
        
        # Standard error
        se = math.sqrt(p_pooled * (1 - p_pooled) * (1/control_total + 1/treatment_total))
        
        if se == 0:
            return StatisticalResult(
                control_mean=p1,
                treatment_mean=p2,
                lift=0,
                p_value=1.0 if p1 == p2 else 0.0,
                confidence_level=self.confidence_level,
                is_significant=p1 != p2,
                sample_size_control=control_total,
                sample_size_treatment=treatment_total
            )
        
        z_stat = (p2 - p1) / se
        
        # Two-tailed p-value
        p_value = 2 * (1 - self._normal_cdf(abs(z_stat)))
        
        lift = (p2 - p1) / p1 if p1 != 0 else 0
        
        return StatisticalResult(
            control_mean=p1,
            treatment_mean=p2,
            lift=lift,
            p_value=p_value,
            confidence_level=self.confidence_level,
            is_significant=p_value < (1 - self.confidence_level),
            sample_size_control=control_total,
            sample_size_treatment=treatment_total
        )
    
    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF."""
        t = 1.0 / (1.0 + 0.2316419 * abs(x))
        d = 0.3989423 * math.exp(-x * x / 2)
        p = 1 - d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
        return p if x > 0 else 1 - p


class ExperimentStore(ABC):
    """Abstract base class for experiment persistence."""
    
    @abstractmethod
    def save(self, experiment: Experiment) -> None:
        """Save an experiment."""
        pass
    
    @abstractmethod
    def load(self, experiment_id: str) -> Optional[Experiment]:
        """Load an experiment by ID."""
        pass
    
    @abstractmethod
    def list_experiments(self, status: Optional[ExperimentStatus] = None) -> List[str]:
        """List experiment IDs."""
        pass
    
    @abstractmethod
    def delete(self, experiment_id: str) -> bool:
        """Delete an experiment."""
        pass


class MemoryExperimentStore(ExperimentStore):
    """In-memory experiment store."""
    
    def __init__(self):
        self._experiments: Dict[str, Experiment] = {}
        self._lock = threading.Lock()
    
    def save(self, experiment: Experiment) -> None:
        """Save experiment to memory."""
        with self._lock:
            experiment.updated_at = datetime.now()
            self._experiments[experiment.id] = experiment
    
    def load(self, experiment_id: str) -> Optional[Experiment]:
        """Load experiment from memory."""
        with self._lock:
            return self._experiments.get(experiment_id)
    
    def list_experiments(self, status: Optional[ExperimentStatus] = None) -> List[str]:
        """List experiment IDs."""
        with self._lock:
            if status:
                return [e.id for e in self._experiments.values() if e.status == status]
            return list(self._experiments.keys())
    
    def delete(self, experiment_id: str) -> bool:
        """Delete experiment from memory."""
        with self._lock:
            if experiment_id in self._experiments:
                del self._experiments[experiment_id]
                return True
            return False


class FileExperimentStore(ExperimentStore):
    """File-based experiment store."""
    
    def __init__(self, directory: str):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
    
    def _get_path(self, experiment_id: str) -> Path:
        """Get file path for experiment."""
        return self.directory / f"{experiment_id}.json"
    
    def save(self, experiment: Experiment) -> None:
        """Save experiment to file."""
        with self._lock:
            experiment.updated_at = datetime.now()
            path = self._get_path(experiment.id)
            with open(path, 'w') as f:
                json.dump(experiment.to_dict(), f, indent=2)
    
    def load(self, experiment_id: str) -> Optional[Experiment]:
        """Load experiment from file."""
        path = self._get_path(experiment_id)
        if not path.exists():
            return None
        
        with self._lock:
            with open(path) as f:
                data = json.load(f)
        
        # Reconstruct experiment
        variants = [Variant(**v) for v in data.get("variants", [])]
        metric_defs = [
            MetricDefinition(
                name=m["name"],
                type=MetricType(m["type"]),
                description=m.get("description", ""),
                unit=m.get("unit", ""),
                higher_is_better=m.get("higher_is_better", True)
            )
            for m in data.get("metric_definitions", [])
        ]
        
        return Experiment(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            variants=variants,
            metric_definitions=metric_defs,
            status=ExperimentStatus(data.get("status", "draft")),
            strategy=AssignmentStrategy(data.get("strategy", "deterministic")),
            start_time=datetime.fromisoformat(data["start_time"]) if data.get("start_time") else None,
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            traffic_percentage=data.get("traffic_percentage", 100),
            targeting_rules=data.get("targeting_rules", {}),
            assignments=data.get("assignments", {}),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat())),
            owner=data.get("owner", ""),
            tags=set(data.get("tags", []))
        )
    
    def list_experiments(self, status: Optional[ExperimentStatus] = None) -> List[str]:
        """List experiment IDs."""
        experiment_ids = []
        for path in self.directory.glob("*.json"):
            experiment_id = path.stem
            if status:
                exp = self.load(experiment_id)
                if exp and exp.status == status:
                    experiment_ids.append(experiment_id)
            else:
                experiment_ids.append(experiment_id)
        return experiment_ids
    
    def delete(self, experiment_id: str) -> bool:
        """Delete experiment file."""
        path = self._get_path(experiment_id)
        if path.exists():
            with self._lock:
                path.unlink()
            return True
        return False


@dataclass
class ExperimentReport:
    """Report for an experiment."""
    experiment: Experiment
    generated_at: datetime
    overall_winner: Optional[str] = None
    metric_results: Dict[str, Dict[str, StatisticalResult]] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment.id,
            "experiment_name": self.experiment.name,
            "generated_at": self.generated_at.isoformat(),
            "status": self.experiment.status.value,
            "overall_winner": self.overall_winner,
            "variants": [v.to_dict() for v in self.experiment.variants],
            "metric_results": {
                metric: {variant: result.to_dict() for variant, result in variants.items()}
                for metric, variants in self.metric_results.items()
            },
            "recommendations": self.recommendations,
        }
    
    def summary(self) -> str:
        """Generate text summary of results."""
        lines = [
            f"Experiment: {self.experiment.name}",
            f"Status: {self.experiment.status.value}",
            f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M')}",
            "",
            "Variants:",
        ]
        
        for variant in self.experiment.variants:
            metrics = self.experiment.variant_metrics.get(variant.id)
            lines.append(f"  - {variant.name}: {metrics.assigned_count if metrics else 0} users")
        
        lines.append("")
        lines.append("Results:")
        
        for metric, results in self.metric_results.items():
            lines.append(f"  {metric}:")
            for variant_id, result in results.items():
                variant = self.experiment.get_variant(variant_id)
                name = variant.name if variant else variant_id
                sig = "✓" if result.is_significant else "✗"
                lines.append(f"    {name}: {result.treatment_mean:.4f} (lift: {result.lift*100:.2f}%, p={result.p_value:.4f}) {sig}")
        
        if self.overall_winner:
            winner = self.experiment.get_variant(self.overall_winner)
            lines.append("")
            lines.append(f"Winner: {winner.name if winner else self.overall_winner}")
        
        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")
        
        return "\n".join(lines)


class ABTestingManager:
    """
    Main manager for A/B testing functionality.
    
    Features:
    - Experiment CRUD
    - User assignment to variants
    - Metric collection
    - Statistical analysis
    - Report generation
    """
    
    def __init__(
        self,
        store: Optional[ExperimentStore] = None,
        analyzer: Optional[StatisticalAnalyzer] = None
    ):
        self.store = store or MemoryExperimentStore()
        self.analyzer = analyzer or StatisticalAnalyzer()
        
        # Assignment algorithms
        self._algorithms: Dict[AssignmentStrategy, AssignmentAlgorithm] = {
            AssignmentStrategy.RANDOM: RandomAssignment(),
            AssignmentStrategy.DETERMINISTIC: DeterministicAssignment(),
            AssignmentStrategy.ROUND_ROBIN: RoundRobinAssignment(),
            AssignmentStrategy.WEIGHTED_RANDOM: WeightedRandomAssignment(),
        }
        
        self._lock = threading.Lock()
        
        # Cache active experiments
        self._active_experiments: Dict[str, Experiment] = {}
    
    def create_experiment(
        self,
        name: str,
        variants: List[Variant],
        description: str = "",
        metric_definitions: Optional[List[MetricDefinition]] = None,
        strategy: AssignmentStrategy = AssignmentStrategy.DETERMINISTIC,
        traffic_percentage: float = 100.0,
        owner: str = "",
        tags: Optional[Set[str]] = None
    ) -> Experiment:
        """Create a new experiment."""
        experiment = Experiment(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            variants=variants,
            metric_definitions=metric_definitions or [],
            strategy=strategy,
            traffic_percentage=traffic_percentage,
            owner=owner,
            tags=tags or set()
        )
        
        # Ensure at least one variant is marked as control
        if not any(v.is_control for v in variants):
            variants[0].is_control = True
        
        self.store.save(experiment)
        logger.info(f"Created experiment: {name} ({experiment.id})")
        
        return experiment
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID."""
        return self.store.load(experiment_id)
    
    def list_experiments(self, status: Optional[ExperimentStatus] = None) -> List[Experiment]:
        """List all experiments."""
        experiment_ids = self.store.list_experiments(status)
        return [self.store.load(eid) for eid in experiment_ids if self.store.load(eid)]
    
    def start_experiment(
        self,
        experiment_id: str,
        duration_days: Optional[int] = None
    ) -> bool:
        """Start an experiment."""
        experiment = self.store.load(experiment_id)
        if not experiment:
            return False
        
        if experiment.status not in [ExperimentStatus.DRAFT, ExperimentStatus.PAUSED]:
            logger.warning(f"Cannot start experiment in status: {experiment.status}")
            return False
        
        experiment.status = ExperimentStatus.RUNNING
        experiment.start_time = datetime.now()
        
        if duration_days:
            experiment.end_time = datetime.now() + timedelta(days=duration_days)
        
        self.store.save(experiment)
        
        with self._lock:
            self._active_experiments[experiment_id] = experiment
        
        logger.info(f"Started experiment: {experiment.name}")
        return True
    
    def pause_experiment(self, experiment_id: str) -> bool:
        """Pause an experiment."""
        experiment = self.store.load(experiment_id)
        if not experiment:
            return False
        
        if experiment.status != ExperimentStatus.RUNNING:
            return False
        
        experiment.status = ExperimentStatus.PAUSED
        self.store.save(experiment)
        
        with self._lock:
            self._active_experiments.pop(experiment_id, None)
        
        logger.info(f"Paused experiment: {experiment.name}")
        return True
    
    def complete_experiment(self, experiment_id: str) -> bool:
        """Complete an experiment."""
        experiment = self.store.load(experiment_id)
        if not experiment:
            return False
        
        experiment.status = ExperimentStatus.COMPLETED
        experiment.end_time = datetime.now()
        self.store.save(experiment)
        
        with self._lock:
            self._active_experiments.pop(experiment_id, None)
        
        logger.info(f"Completed experiment: {experiment.name}")
        return True
    
    def cancel_experiment(self, experiment_id: str) -> bool:
        """Cancel an experiment."""
        experiment = self.store.load(experiment_id)
        if not experiment:
            return False
        
        experiment.status = ExperimentStatus.CANCELLED
        experiment.end_time = datetime.now()
        self.store.save(experiment)
        
        with self._lock:
            self._active_experiments.pop(experiment_id, None)
        
        logger.info(f"Cancelled experiment: {experiment.name}")
        return True
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment."""
        with self._lock:
            self._active_experiments.pop(experiment_id, None)
        return self.store.delete(experiment_id)
    
    def assign_variant(
        self,
        experiment_id: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Variant]:
        """
        Assign a user to a variant.
        
        Args:
            experiment_id: The experiment ID
            user_id: The user/session ID
            context: Optional context for targeting
        
        Returns:
            Assigned variant or None if not eligible
        """
        experiment = self.store.load(experiment_id)
        if not experiment or not experiment.is_running():
            return None
        
        # Check traffic allocation
        if experiment.traffic_percentage < 100:
            hash_val = int(hashlib.md5(f"{experiment_id}:{user_id}".encode()).hexdigest()[:8], 16)
            if (hash_val % 100) >= experiment.traffic_percentage:
                return None
        
        # Check targeting rules
        if context and experiment.targeting_rules:
            if not self._matches_targeting(context, experiment.targeting_rules):
                return None
        
        # Check if already assigned
        if user_id in experiment.assignments:
            variant_id = experiment.assignments[user_id]
            return experiment.get_variant(variant_id)
        
        # Assign using algorithm
        algorithm = self._algorithms.get(experiment.strategy, self._algorithms[AssignmentStrategy.DETERMINISTIC])
        variant = algorithm.assign(user_id, experiment.variants, experiment_id=experiment_id)
        
        # Record assignment
        experiment.assignments[user_id] = variant.id
        if variant.id in experiment.variant_metrics:
            experiment.variant_metrics[variant.id].assigned_count += 1
        
        self.store.save(experiment)
        
        logger.debug(f"Assigned user {user_id} to variant {variant.name} in experiment {experiment.name}")
        return variant
    
    def _matches_targeting(self, context: Dict[str, Any], rules: Dict[str, Any]) -> bool:
        """Check if context matches targeting rules."""
        for key, rule in rules.items():
            value = context.get(key)
            
            if isinstance(rule, list):
                # List means "in"
                if value not in rule:
                    return False
            elif isinstance(rule, dict):
                # Dict can have operators
                if "eq" in rule and value != rule["eq"]:
                    return False
                if "ne" in rule and value == rule["ne"]:
                    return False
                if "gt" in rule and (value is None or value <= rule["gt"]):
                    return False
                if "lt" in rule and (value is None or value >= rule["lt"]):
                    return False
                if "in" in rule and value not in rule["in"]:
                    return False
            else:
                # Direct equality
                if value != rule:
                    return False
        
        return True
    
    def record_metric(
        self,
        experiment_id: str,
        user_id: str,
        metric_name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record a metric value for a user.
        
        Args:
            experiment_id: The experiment ID
            user_id: The user/session ID
            metric_name: Name of the metric
            value: Metric value
            metadata: Optional metadata
        
        Returns:
            True if recorded successfully
        """
        experiment = self.store.load(experiment_id)
        if not experiment:
            return False
        
        # Get user's variant
        variant_id = experiment.assignments.get(user_id)
        if not variant_id:
            logger.warning(f"User {user_id} not assigned to experiment {experiment_id}")
            return False
        
        # Record metric
        if variant_id in experiment.variant_metrics:
            experiment.variant_metrics[variant_id].add_metric(metric_name, value, metadata)
            self.store.save(experiment)
            return True
        
        return False
    
    def record_conversion(
        self,
        experiment_id: str,
        user_id: str,
        conversion_type: str = "conversion"
    ) -> bool:
        """Record a conversion (success=1)."""
        return self.record_metric(experiment_id, user_id, conversion_type, 1.0)
    
    def get_results(
        self,
        experiment_id: str,
        metric_name: str
    ) -> Optional[Dict[str, StatisticalResult]]:
        """
        Get statistical results for a metric.
        
        Returns comparison of each treatment variant vs control.
        """
        experiment = self.store.load(experiment_id)
        if not experiment:
            return None
        
        control = experiment.get_control()
        if not control:
            return None
        
        control_metrics = experiment.variant_metrics.get(control.id)
        if not control_metrics:
            return None
        
        control_values = [v.value for v in control_metrics.metrics.get(metric_name, [])]
        
        results = {}
        for variant in experiment.variants:
            if variant.is_control:
                continue
            
            variant_metrics = experiment.variant_metrics.get(variant.id)
            if not variant_metrics:
                continue
            
            treatment_values = [v.value for v in variant_metrics.metrics.get(metric_name, [])]
            
            result = self.analyzer.welch_t_test(control_values, treatment_values)
            results[variant.id] = result
        
        return results
    
    def generate_report(self, experiment_id: str) -> Optional[ExperimentReport]:
        """Generate a comprehensive report for an experiment."""
        experiment = self.store.load(experiment_id)
        if not experiment:
            return None
        
        report = ExperimentReport(
            experiment=experiment,
            generated_at=datetime.now()
        )
        
        # Analyze all metrics
        control = experiment.get_control()
        if not control:
            report.recommendations.append("No control variant defined")
            return report
        
        winners = {}  # metric -> variant_id
        
        for metric_def in experiment.metric_definitions:
            metric_name = metric_def.name
            results = self.get_results(experiment_id, metric_name)
            
            if results:
                report.metric_results[metric_name] = results
                
                # Find winner for this metric
                best_variant = None
                best_lift = 0 if metric_def.higher_is_better else float('inf')
                
                for variant_id, result in results.items():
                    if result.is_significant:
                        if metric_def.higher_is_better and result.lift > best_lift:
                            best_lift = result.lift
                            best_variant = variant_id
                        elif not metric_def.higher_is_better and result.lift < best_lift:
                            best_lift = result.lift
                            best_variant = variant_id
                
                if best_variant:
                    winners[metric_name] = best_variant
        
        # Determine overall winner
        if winners:
            variant_wins = {}
            for variant_id in winners.values():
                variant_wins[variant_id] = variant_wins.get(variant_id, 0) + 1
            
            if variant_wins:
                report.overall_winner = max(variant_wins, key=variant_wins.get)
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(experiment, report)
        
        return report
    
    def _generate_recommendations(
        self,
        experiment: Experiment,
        report: ExperimentReport
    ) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        # Check sample size
        for variant in experiment.variants:
            metrics = experiment.variant_metrics.get(variant.id)
            if metrics and metrics.assigned_count < 100:
                recommendations.append(
                    f"Variant '{variant.name}' has low sample size ({metrics.assigned_count}). "
                    "Consider running longer for reliable results."
                )
        
        # Check for significant results
        has_significant = any(
            result.is_significant
            for results in report.metric_results.values()
            for result in results.values()
        )
        
        if not has_significant:
            recommendations.append(
                "No statistically significant results yet. "
                "Consider running the experiment longer or increasing traffic."
            )
        
        # Winner recommendation
        if report.overall_winner:
            winner = experiment.get_variant(report.overall_winner)
            if winner:
                recommendations.append(
                    f"Consider rolling out '{winner.name}' as it shows the best overall performance."
                )
        else:
            recommendations.append(
                "No clear winner identified. Consider extending the experiment or adjusting variants."
            )
        
        return recommendations
    
    def get_variant_for_user(
        self,
        user_id: str,
        experiment_ids: Optional[List[str]] = None
    ) -> Dict[str, Optional[Variant]]:
        """
        Get all variant assignments for a user.
        
        Args:
            user_id: The user/session ID
            experiment_ids: Optional list of experiments to check
        
        Returns:
            Dict mapping experiment ID to assigned variant
        """
        if experiment_ids is None:
            experiment_ids = self.store.list_experiments(ExperimentStatus.RUNNING)
        
        assignments = {}
        for exp_id in experiment_ids:
            variant = self.assign_variant(exp_id, user_id)
            assignments[exp_id] = variant
        
        return assignments


# Singleton pattern for global access
_manager: Optional[ABTestingManager] = None
_manager_lock = threading.Lock()


def get_ab_testing_manager() -> ABTestingManager:
    """Get the singleton A/B testing manager."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = ABTestingManager()
    return _manager


def reset_ab_testing_manager() -> None:
    """Reset the singleton manager (for testing)."""
    global _manager
    with _manager_lock:
        _manager = None


# Convenience functions
def create_simple_ab_test(
    name: str,
    control_config: Dict[str, Any],
    treatment_config: Dict[str, Any],
    metrics: Optional[List[str]] = None
) -> Experiment:
    """Create a simple A/B test with two variants."""
    manager = get_ab_testing_manager()
    
    variants = [
        Variant(
            id="control",
            name="Control",
            config=control_config,
            is_control=True
        ),
        Variant(
            id="treatment",
            name="Treatment",
            config=treatment_config
        )
    ]
    
    metric_defs = [
        MetricDefinition(name=m, type=MetricType.COUNTER)
        for m in (metrics or ["conversion"])
    ]
    
    return manager.create_experiment(
        name=name,
        variants=variants,
        metric_definitions=metric_defs
    )


def get_variant_config(
    experiment_id: str,
    user_id: str,
    default_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Get the configuration for a user's assigned variant."""
    manager = get_ab_testing_manager()
    variant = manager.assign_variant(experiment_id, user_id)
    
    if variant:
        return variant.config
    return default_config or {}


__all__ = [
    # Enums
    "ExperimentStatus",
    "AssignmentStrategy",
    "MetricType",
    
    # Data classes
    "Variant",
    "MetricValue",
    "MetricDefinition",
    "VariantMetrics",
    "Experiment",
    "StatisticalResult",
    "ExperimentReport",
    
    # Assignment algorithms
    "AssignmentAlgorithm",
    "RandomAssignment",
    "DeterministicAssignment",
    "RoundRobinAssignment",
    "WeightedRandomAssignment",
    
    # Analysis
    "StatisticalAnalyzer",
    
    # Storage
    "ExperimentStore",
    "MemoryExperimentStore",
    "FileExperimentStore",
    
    # Manager
    "ABTestingManager",
    
    # Singleton
    "get_ab_testing_manager",
    "reset_ab_testing_manager",
    
    # Convenience functions
    "create_simple_ab_test",
    "get_variant_config",
]
