"""
Tests for A/B Testing Framework.

Phase 5 of Professional Chat Agent implementation.
"""

import pytest
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from workflow_composer.agents.intent.ab_testing import (
    ABTestingManager,
    Experiment,
    ExperimentStatus,
    ExperimentReport,
    Variant,
    MetricDefinition,
    MetricType,
    MetricValue,
    VariantMetrics,
    AssignmentStrategy,
    RandomAssignment,
    DeterministicAssignment,
    RoundRobinAssignment,
    WeightedRandomAssignment,
    StatisticalAnalyzer,
    StatisticalResult,
    MemoryExperimentStore,
    FileExperimentStore,
    get_ab_testing_manager,
    reset_ab_testing_manager,
    create_simple_ab_test,
    get_variant_config,
)


class TestExperimentStatus:
    """Test ExperimentStatus enum."""
    
    def test_all_statuses_exist(self):
        """Test all expected statuses are defined."""
        statuses = [
            ExperimentStatus.DRAFT,
            ExperimentStatus.RUNNING,
            ExperimentStatus.PAUSED,
            ExperimentStatus.COMPLETED,
            ExperimentStatus.CANCELLED,
        ]
        assert len(statuses) == 5
    
    def test_status_values(self):
        """Test status string values."""
        assert ExperimentStatus.DRAFT.value == "draft"
        assert ExperimentStatus.RUNNING.value == "running"


class TestAssignmentStrategy:
    """Test AssignmentStrategy enum."""
    
    def test_all_strategies_exist(self):
        """Test all assignment strategies are defined."""
        strategies = [
            AssignmentStrategy.RANDOM,
            AssignmentStrategy.DETERMINISTIC,
            AssignmentStrategy.ROUND_ROBIN,
            AssignmentStrategy.WEIGHTED_RANDOM,
        ]
        assert len(strategies) == 4


class TestMetricType:
    """Test MetricType enum."""
    
    def test_all_types_exist(self):
        """Test all metric types are defined."""
        types = [
            MetricType.COUNTER,
            MetricType.GAUGE,
            MetricType.RATE,
            MetricType.PERCENTAGE,
            MetricType.DURATION,
        ]
        assert len(types) == 5


class TestVariant:
    """Test Variant dataclass."""
    
    def test_create_variant(self):
        """Test creating a variant."""
        variant = Variant(
            id="test-variant",
            name="Test Variant",
            description="A test variant",
            weight=1.5,
            config={"key": "value"},
            is_control=True
        )
        
        assert variant.id == "test-variant"
        assert variant.name == "Test Variant"
        assert variant.weight == 1.5
        assert variant.config["key"] == "value"
        assert variant.is_control
    
    def test_variant_defaults(self):
        """Test variant default values."""
        variant = Variant(id="v1", name="V1")
        
        assert variant.description == ""
        assert variant.weight == 1.0
        assert variant.config == {}
        assert not variant.is_control
    
    def test_variant_hash(self):
        """Test variant is hashable."""
        v1 = Variant(id="v1", name="V1")
        v2 = Variant(id="v1", name="V1 Copy")
        v3 = Variant(id="v2", name="V2")
        
        assert hash(v1) == hash(v2)
        assert hash(v1) != hash(v3)
    
    def test_variant_equality(self):
        """Test variant equality based on ID."""
        v1 = Variant(id="v1", name="V1")
        v2 = Variant(id="v1", name="V1 Copy")
        v3 = Variant(id="v2", name="V2")
        
        assert v1 == v2
        assert v1 != v3
    
    def test_variant_to_dict(self):
        """Test converting variant to dictionary."""
        variant = Variant(id="v1", name="V1", is_control=True)
        d = variant.to_dict()
        
        assert d["id"] == "v1"
        assert d["name"] == "V1"
        assert d["is_control"]


class TestMetricDefinition:
    """Test MetricDefinition dataclass."""
    
    def test_create_metric_definition(self):
        """Test creating a metric definition."""
        metric = MetricDefinition(
            name="conversion_rate",
            type=MetricType.PERCENTAGE,
            description="User conversion rate",
            unit="%",
            higher_is_better=True
        )
        
        assert metric.name == "conversion_rate"
        assert metric.type == MetricType.PERCENTAGE
        assert metric.unit == "%"
        assert metric.higher_is_better
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        metric = MetricDefinition(name="clicks", type=MetricType.COUNTER)
        d = metric.to_dict()
        
        assert d["name"] == "clicks"
        assert d["type"] == "counter"


class TestVariantMetrics:
    """Test VariantMetrics class."""
    
    def test_create_variant_metrics(self):
        """Test creating variant metrics."""
        metrics = VariantMetrics(variant_id="v1")
        assert metrics.variant_id == "v1"
        assert metrics.assigned_count == 0
    
    def test_add_metric(self):
        """Test adding metric values."""
        metrics = VariantMetrics(variant_id="v1")
        
        metrics.add_metric("clicks", 5)
        metrics.add_metric("clicks", 3)
        metrics.add_metric("clicks", 7)
        
        assert metrics.get_count("clicks") == 3
        assert metrics.get_sum("clicks") == 15
    
    def test_get_mean(self):
        """Test calculating mean."""
        metrics = VariantMetrics(variant_id="v1")
        
        for val in [2, 4, 6, 8, 10]:
            metrics.add_metric("value", val)
        
        assert metrics.get_mean("value") == 6.0
    
    def test_get_mean_empty(self):
        """Test mean with no values."""
        metrics = VariantMetrics(variant_id="v1")
        assert metrics.get_mean("empty") is None
    
    def test_get_variance(self):
        """Test calculating variance."""
        metrics = VariantMetrics(variant_id="v1")
        
        for val in [2, 4, 6, 8, 10]:
            metrics.add_metric("value", val)
        
        variance = metrics.get_variance("value")
        assert variance is not None
        assert abs(variance - 10.0) < 0.01  # Sample variance
    
    def test_get_std(self):
        """Test calculating standard deviation."""
        metrics = VariantMetrics(variant_id="v1")
        
        for val in [2, 4, 6, 8, 10]:
            metrics.add_metric("value", val)
        
        std = metrics.get_std("value")
        assert std is not None
        assert abs(std - 3.162) < 0.01
    
    def test_get_min_max(self):
        """Test min and max."""
        metrics = VariantMetrics(variant_id="v1")
        
        for val in [5, 2, 8, 1, 9]:
            metrics.add_metric("value", val)
        
        assert metrics.get_min("value") == 1
        assert metrics.get_max("value") == 9


class TestExperiment:
    """Test Experiment dataclass."""
    
    @pytest.fixture
    def variants(self):
        """Create test variants."""
        return [
            Variant(id="control", name="Control", is_control=True),
            Variant(id="treatment", name="Treatment"),
        ]
    
    def test_create_experiment(self, variants):
        """Test creating an experiment."""
        exp = Experiment(
            id="exp-1",
            name="Test Experiment",
            variants=variants
        )
        
        assert exp.id == "exp-1"
        assert exp.name == "Test Experiment"
        assert len(exp.variants) == 2
        assert exp.status == ExperimentStatus.DRAFT
    
    def test_get_variant(self, variants):
        """Test getting variant by ID."""
        exp = Experiment(id="exp-1", name="Test", variants=variants)
        
        control = exp.get_variant("control")
        assert control is not None
        assert control.is_control
        
        missing = exp.get_variant("missing")
        assert missing is None
    
    def test_get_control(self, variants):
        """Test getting control variant."""
        exp = Experiment(id="exp-1", name="Test", variants=variants)
        
        control = exp.get_control()
        assert control is not None
        assert control.id == "control"
    
    def test_is_running(self, variants):
        """Test running status check."""
        exp = Experiment(id="exp-1", name="Test", variants=variants)
        
        assert not exp.is_running()
        
        exp.status = ExperimentStatus.RUNNING
        assert exp.is_running()
        
        # Test time bounds
        exp.start_time = datetime.now() + timedelta(hours=1)
        assert not exp.is_running()
        
        exp.start_time = datetime.now() - timedelta(hours=1)
        exp.end_time = datetime.now() - timedelta(minutes=1)
        assert not exp.is_running()
    
    def test_to_dict(self, variants):
        """Test converting to dictionary."""
        exp = Experiment(id="exp-1", name="Test", variants=variants)
        d = exp.to_dict()
        
        assert d["id"] == "exp-1"
        assert d["name"] == "Test"
        assert len(d["variants"]) == 2


class TestAssignmentAlgorithms:
    """Test assignment algorithms."""
    
    @pytest.fixture
    def variants(self):
        """Create test variants."""
        return [
            Variant(id="a", name="A", weight=1.0),
            Variant(id="b", name="B", weight=1.0),
            Variant(id="c", name="C", weight=1.0),
        ]
    
    def test_random_assignment(self, variants):
        """Test random assignment."""
        algorithm = RandomAssignment()
        
        assignments = [algorithm.assign("user1", variants).id for _ in range(100)]
        
        # Should have some distribution
        unique = set(assignments)
        assert len(unique) >= 2  # Should hit at least 2 variants
    
    def test_deterministic_assignment_consistency(self, variants):
        """Test deterministic assignment is consistent."""
        algorithm = DeterministicAssignment()
        
        first = algorithm.assign("user123", variants, experiment_id="exp1")
        
        # Same user should get same variant
        for _ in range(10):
            result = algorithm.assign("user123", variants, experiment_id="exp1")
            assert result.id == first.id
    
    def test_deterministic_assignment_distribution(self, variants):
        """Test deterministic assignment has distribution."""
        algorithm = DeterministicAssignment()
        
        assignments = {}
        for i in range(1000):
            variant = algorithm.assign(f"user_{i}", variants, experiment_id="exp1")
            assignments[variant.id] = assignments.get(variant.id, 0) + 1
        
        # Should have roughly even distribution (within 20%)
        for count in assignments.values():
            assert 200 < count < 500
    
    def test_round_robin_assignment(self, variants):
        """Test round robin assignment."""
        algorithm = RoundRobinAssignment()
        
        assignments = [algorithm.assign(f"user{i}", variants).id for i in range(9)]
        
        # Should cycle through variants
        assert assignments[:3] == ["a", "b", "c"]
        assert assignments[3:6] == ["a", "b", "c"]
    
    def test_weighted_random_assignment(self):
        """Test weighted random assignment."""
        variants = [
            Variant(id="heavy", name="Heavy", weight=9.0),
            Variant(id="light", name="Light", weight=1.0),
        ]
        
        algorithm = WeightedRandomAssignment()
        
        assignments = {}
        for i in range(1000):
            variant = algorithm.assign(f"user_{i}", variants)
            assignments[variant.id] = assignments.get(variant.id, 0) + 1
        
        # Heavy should be assigned ~9x more
        assert assignments["heavy"] > assignments["light"] * 5


class TestStatisticalAnalyzer:
    """Test StatisticalAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return StatisticalAnalyzer(confidence_level=0.95)
    
    def test_welch_t_test_significant(self, analyzer):
        """Test t-test with significant difference."""
        control = [10, 11, 9, 12, 10, 11, 10, 9, 11, 10] * 10
        treatment = [15, 14, 16, 15, 14, 15, 16, 14, 15, 15] * 10
        
        result = analyzer.welch_t_test(control, treatment)
        
        assert result.control_mean < result.treatment_mean
        assert result.lift > 0
        assert result.p_value < 0.05
        assert result.is_significant
    
    def test_welch_t_test_not_significant(self, analyzer):
        """Test t-test with no significant difference."""
        control = [10, 11, 9, 12]
        treatment = [10.5, 10.2, 11, 9.8]
        
        result = analyzer.welch_t_test(control, treatment)
        
        assert not result.is_significant
    
    def test_welch_t_test_insufficient_samples(self, analyzer):
        """Test t-test with insufficient samples."""
        result = analyzer.welch_t_test([10], [11])
        
        assert result.p_value == 1.0
        assert not result.is_significant
    
    def test_calculate_sample_size(self, analyzer):
        """Test sample size calculation."""
        sample_size = analyzer.calculate_sample_size(
            baseline_rate=0.1,
            minimum_detectable_effect=0.2,  # 20% lift
            power=0.8
        )
        
        # Should require reasonable sample size
        assert 500 < sample_size < 10000
    
    def test_chi_square_test_significant(self, analyzer):
        """Test chi-square test with significant difference."""
        result = analyzer.chi_square_test(
            control_successes=100,
            control_total=1000,
            treatment_successes=150,
            treatment_total=1000
        )
        
        assert result.control_mean == 0.1
        assert result.treatment_mean == 0.15
        assert abs(result.lift - 0.5) < 0.001  # 50% lift (with floating point tolerance)
        assert result.is_significant
    
    def test_chi_square_test_empty(self, analyzer):
        """Test chi-square with empty data."""
        result = analyzer.chi_square_test(0, 0, 0, 0)
        
        assert result.p_value == 1.0
        assert not result.is_significant


class TestMemoryExperimentStore:
    """Test MemoryExperimentStore class."""
    
    @pytest.fixture
    def store(self):
        """Create store instance."""
        return MemoryExperimentStore()
    
    @pytest.fixture
    def experiment(self):
        """Create test experiment."""
        return Experiment(
            id="exp-1",
            name="Test",
            variants=[Variant(id="v1", name="V1")]
        )
    
    def test_save_and_load(self, store, experiment):
        """Test saving and loading experiments."""
        store.save(experiment)
        
        loaded = store.load("exp-1")
        assert loaded is not None
        assert loaded.id == "exp-1"
        assert loaded.name == "Test"
    
    def test_load_missing(self, store):
        """Test loading non-existent experiment."""
        assert store.load("missing") is None
    
    def test_list_experiments(self, store):
        """Test listing experiments."""
        for i in range(3):
            exp = Experiment(id=f"exp-{i}", name=f"Test {i}", variants=[])
            if i == 0:
                exp.status = ExperimentStatus.RUNNING
            store.save(exp)
        
        all_ids = store.list_experiments()
        assert len(all_ids) == 3
        
        running_ids = store.list_experiments(ExperimentStatus.RUNNING)
        assert len(running_ids) == 1
    
    def test_delete(self, store, experiment):
        """Test deleting experiment."""
        store.save(experiment)
        assert store.load("exp-1") is not None
        
        assert store.delete("exp-1")
        assert store.load("exp-1") is None
        
        assert not store.delete("missing")


class TestFileExperimentStore:
    """Test FileExperimentStore class."""
    
    @pytest.fixture
    def store(self, tmp_path):
        """Create store instance with temp directory."""
        return FileExperimentStore(str(tmp_path / "experiments"))
    
    @pytest.fixture
    def experiment(self):
        """Create test experiment."""
        return Experiment(
            id="exp-1",
            name="Test",
            variants=[Variant(id="v1", name="V1", is_control=True)]
        )
    
    def test_save_and_load(self, store, experiment):
        """Test saving and loading experiments."""
        store.save(experiment)
        
        loaded = store.load("exp-1")
        assert loaded is not None
        assert loaded.id == "exp-1"
    
    def test_delete(self, store, experiment):
        """Test deleting experiment."""
        store.save(experiment)
        assert store.delete("exp-1")
        assert store.load("exp-1") is None


class TestABTestingManager:
    """Test ABTestingManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create manager instance."""
        return ABTestingManager()
    
    @pytest.fixture
    def variants(self):
        """Create test variants."""
        return [
            Variant(id="control", name="Control", is_control=True),
            Variant(id="treatment", name="Treatment"),
        ]
    
    def test_create_experiment(self, manager, variants):
        """Test creating an experiment."""
        exp = manager.create_experiment(
            name="Test Experiment",
            variants=variants,
            description="A test experiment"
        )
        
        assert exp.id is not None
        assert exp.name == "Test Experiment"
        assert exp.status == ExperimentStatus.DRAFT
    
    def test_create_sets_control(self, manager):
        """Test that create sets control if not specified."""
        variants = [
            Variant(id="a", name="A"),
            Variant(id="b", name="B"),
        ]
        
        exp = manager.create_experiment(name="Test", variants=variants)
        
        # First variant should be marked as control
        assert exp.variants[0].is_control
    
    def test_get_experiment(self, manager, variants):
        """Test getting experiment by ID."""
        created = manager.create_experiment(name="Test", variants=variants)
        
        fetched = manager.get_experiment(created.id)
        assert fetched is not None
        assert fetched.name == "Test"
    
    def test_list_experiments(self, manager, variants):
        """Test listing experiments."""
        for i in range(3):
            manager.create_experiment(name=f"Test {i}", variants=variants)
        
        all_exps = manager.list_experiments()
        assert len(all_exps) == 3
    
    def test_start_experiment(self, manager, variants):
        """Test starting an experiment."""
        exp = manager.create_experiment(name="Test", variants=variants)
        
        assert manager.start_experiment(exp.id)
        
        loaded = manager.get_experiment(exp.id)
        assert loaded.status == ExperimentStatus.RUNNING
        assert loaded.start_time is not None
    
    def test_start_with_duration(self, manager, variants):
        """Test starting experiment with duration."""
        exp = manager.create_experiment(name="Test", variants=variants)
        
        manager.start_experiment(exp.id, duration_days=7)
        
        loaded = manager.get_experiment(exp.id)
        assert loaded.end_time is not None
        assert (loaded.end_time - loaded.start_time).days == 7
    
    def test_pause_experiment(self, manager, variants):
        """Test pausing experiment."""
        exp = manager.create_experiment(name="Test", variants=variants)
        manager.start_experiment(exp.id)
        
        assert manager.pause_experiment(exp.id)
        
        loaded = manager.get_experiment(exp.id)
        assert loaded.status == ExperimentStatus.PAUSED
    
    def test_complete_experiment(self, manager, variants):
        """Test completing experiment."""
        exp = manager.create_experiment(name="Test", variants=variants)
        manager.start_experiment(exp.id)
        
        assert manager.complete_experiment(exp.id)
        
        loaded = manager.get_experiment(exp.id)
        assert loaded.status == ExperimentStatus.COMPLETED
    
    def test_cancel_experiment(self, manager, variants):
        """Test cancelling experiment."""
        exp = manager.create_experiment(name="Test", variants=variants)
        manager.start_experiment(exp.id)
        
        assert manager.cancel_experiment(exp.id)
        
        loaded = manager.get_experiment(exp.id)
        assert loaded.status == ExperimentStatus.CANCELLED
    
    def test_delete_experiment(self, manager, variants):
        """Test deleting experiment."""
        exp = manager.create_experiment(name="Test", variants=variants)
        
        assert manager.delete_experiment(exp.id)
        assert manager.get_experiment(exp.id) is None
    
    def test_assign_variant(self, manager, variants):
        """Test assigning user to variant."""
        exp = manager.create_experiment(name="Test", variants=variants)
        manager.start_experiment(exp.id)
        
        variant = manager.assign_variant(exp.id, "user-1")
        
        assert variant is not None
        assert variant.id in ["control", "treatment"]
    
    def test_assign_consistent(self, manager, variants):
        """Test assignment is consistent for same user."""
        exp = manager.create_experiment(name="Test", variants=variants)
        manager.start_experiment(exp.id)
        
        first = manager.assign_variant(exp.id, "user-1")
        
        for _ in range(10):
            variant = manager.assign_variant(exp.id, "user-1")
            assert variant.id == first.id
    
    def test_assign_not_running(self, manager, variants):
        """Test assignment fails for non-running experiment."""
        exp = manager.create_experiment(name="Test", variants=variants)
        
        variant = manager.assign_variant(exp.id, "user-1")
        assert variant is None
    
    def test_traffic_allocation(self, manager, variants):
        """Test traffic allocation."""
        exp = manager.create_experiment(
            name="Test",
            variants=variants,
            traffic_percentage=50.0
        )
        manager.start_experiment(exp.id)
        
        assigned_count = 0
        for i in range(100):
            variant = manager.assign_variant(exp.id, f"user-{i}")
            if variant:
                assigned_count += 1
        
        # Should be roughly 50%
        assert 30 < assigned_count < 70
    
    def test_targeting_rules(self, manager, variants):
        """Test targeting rules."""
        exp = manager.create_experiment(name="Test", variants=variants)
        exp.targeting_rules = {"country": ["US", "CA"]}
        manager.store.save(exp)
        manager.start_experiment(exp.id)
        
        # Should match
        variant = manager.assign_variant(exp.id, "user-1", context={"country": "US"})
        assert variant is not None
        
        # Should not match
        variant2 = manager.assign_variant(exp.id, "user-2", context={"country": "UK"})
        assert variant2 is None
    
    def test_record_metric(self, manager, variants):
        """Test recording metrics."""
        exp = manager.create_experiment(name="Test", variants=variants)
        manager.start_experiment(exp.id)
        
        manager.assign_variant(exp.id, "user-1")
        assert manager.record_metric(exp.id, "user-1", "clicks", 5)
        
        loaded = manager.get_experiment(exp.id)
        variant_id = loaded.assignments.get("user-1")
        metrics = loaded.variant_metrics.get(variant_id)
        
        assert metrics is not None
        assert metrics.get_sum("clicks") == 5
    
    def test_record_conversion(self, manager, variants):
        """Test recording conversion."""
        exp = manager.create_experiment(name="Test", variants=variants)
        manager.start_experiment(exp.id)
        
        manager.assign_variant(exp.id, "user-1")
        assert manager.record_conversion(exp.id, "user-1")
        
        loaded = manager.get_experiment(exp.id)
        variant_id = loaded.assignments.get("user-1")
        metrics = loaded.variant_metrics.get(variant_id)
        
        assert metrics.get_sum("conversion") == 1
    
    def test_get_results(self, manager, variants):
        """Test getting statistical results."""
        metric_defs = [MetricDefinition(name="value", type=MetricType.GAUGE)]
        exp = manager.create_experiment(
            name="Test",
            variants=variants,
            metric_definitions=metric_defs
        )
        manager.start_experiment(exp.id)
        
        # Add some data
        for i in range(50):
            variant = manager.assign_variant(exp.id, f"control-{i}")
            if variant and variant.is_control:
                manager.record_metric(exp.id, f"control-{i}", "value", 10 + (i % 5))
        
        for i in range(50):
            variant = manager.assign_variant(exp.id, f"treatment-{i}")
            if variant and not variant.is_control:
                manager.record_metric(exp.id, f"treatment-{i}", "value", 15 + (i % 5))
        
        results = manager.get_results(exp.id, "value")
        
        # Should have result for treatment variant
        assert results is not None
    
    def test_generate_report(self, manager, variants):
        """Test generating experiment report."""
        metric_defs = [MetricDefinition(name="conversion", type=MetricType.COUNTER)]
        exp = manager.create_experiment(
            name="Test Report",
            variants=variants,
            metric_definitions=metric_defs
        )
        manager.start_experiment(exp.id)
        
        # Add some data
        for i in range(20):
            manager.assign_variant(exp.id, f"user-{i}")
            manager.record_metric(exp.id, f"user-{i}", "conversion", 1 if i % 3 == 0 else 0)
        
        report = manager.generate_report(exp.id)
        
        assert report is not None
        assert report.experiment.id == exp.id
        assert report.generated_at is not None
    
    def test_report_summary(self, manager, variants):
        """Test report summary generation."""
        exp = manager.create_experiment(name="Test", variants=variants)
        manager.start_experiment(exp.id)
        
        report = manager.generate_report(exp.id)
        summary = report.summary()
        
        assert "Test" in summary
        assert "running" in summary.lower()
    
    def test_get_variant_for_user(self, manager, variants):
        """Test getting all variants for a user."""
        exp1 = manager.create_experiment(name="Test1", variants=variants)
        exp2 = manager.create_experiment(name="Test2", variants=variants)
        
        manager.start_experiment(exp1.id)
        manager.start_experiment(exp2.id)
        
        assignments = manager.get_variant_for_user("user-1", [exp1.id, exp2.id])
        
        assert exp1.id in assignments
        assert exp2.id in assignments


class TestSingletonFunctions:
    """Test singleton factory functions."""
    
    def setup_method(self):
        """Reset singleton before each test."""
        reset_ab_testing_manager()
    
    def test_get_ab_testing_manager(self):
        """Test getting singleton manager."""
        manager1 = get_ab_testing_manager()
        manager2 = get_ab_testing_manager()
        
        assert manager1 is manager2
    
    def test_reset_ab_testing_manager(self):
        """Test resetting singleton."""
        manager1 = get_ab_testing_manager()
        reset_ab_testing_manager()
        manager2 = get_ab_testing_manager()
        
        assert manager1 is not manager2


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def setup_method(self):
        """Reset singleton."""
        reset_ab_testing_manager()
    
    def test_create_simple_ab_test(self):
        """Test creating simple A/B test."""
        exp = create_simple_ab_test(
            name="Button Color Test",
            control_config={"color": "blue"},
            treatment_config={"color": "green"},
            metrics=["clicks", "conversions"]
        )
        
        assert exp.name == "Button Color Test"
        assert len(exp.variants) == 2
        assert exp.get_control().config["color"] == "blue"
    
    def test_get_variant_config(self):
        """Test getting variant config for user."""
        exp = create_simple_ab_test(
            name="Test",
            control_config={"enabled": False},
            treatment_config={"enabled": True}
        )
        
        manager = get_ab_testing_manager()
        manager.start_experiment(exp.id)
        
        config = get_variant_config(exp.id, "user-1")
        
        assert "enabled" in config
    
    def test_get_variant_config_not_running(self):
        """Test getting config when experiment not running."""
        exp = create_simple_ab_test(
            name="Test",
            control_config={"a": 1},
            treatment_config={"a": 2}
        )
        
        config = get_variant_config(exp.id, "user-1", default_config={"a": 0})
        
        assert config == {"a": 0}


class TestIntegration:
    """Integration tests for A/B testing."""
    
    def test_full_experiment_workflow(self):
        """Test complete experiment workflow."""
        manager = ABTestingManager()
        
        # Create experiment
        variants = [
            Variant(id="control", name="Control", config={"price": 9.99}, is_control=True),
            Variant(id="discount", name="Discount", config={"price": 7.99}),
        ]
        
        metric_defs = [
            MetricDefinition(name="purchase", type=MetricType.COUNTER, higher_is_better=True),
            MetricDefinition(name="revenue", type=MetricType.GAUGE, higher_is_better=True),
        ]
        
        exp = manager.create_experiment(
            name="Price Test",
            variants=variants,
            metric_definitions=metric_defs,
            owner="test@example.com"
        )
        
        # Start experiment
        manager.start_experiment(exp.id, duration_days=14)
        
        # Simulate users
        import random
        random.seed(42)
        
        for i in range(200):
            user_id = f"user-{i}"
            variant = manager.assign_variant(exp.id, user_id)
            
            if variant:
                # Simulate purchase behavior
                if variant.is_control:
                    purchase = random.random() < 0.1  # 10% conversion
                else:
                    purchase = random.random() < 0.15  # 15% conversion
                
                if purchase:
                    manager.record_metric(exp.id, user_id, "purchase", 1)
                    manager.record_metric(exp.id, user_id, "revenue", variant.config["price"])
        
        # Generate report
        report = manager.generate_report(exp.id)
        
        assert report is not None
        assert len(report.recommendations) > 0
        
        # Complete experiment
        manager.complete_experiment(exp.id)
        
        final = manager.get_experiment(exp.id)
        assert final.status == ExperimentStatus.COMPLETED
    
    def test_multiple_concurrent_experiments(self):
        """Test running multiple experiments concurrently."""
        manager = ABTestingManager()
        
        experiments = []
        for i in range(3):
            variants = [
                Variant(id=f"control-{i}", name="Control", is_control=True),
                Variant(id=f"treatment-{i}", name="Treatment"),
            ]
            exp = manager.create_experiment(name=f"Experiment {i}", variants=variants)
            manager.start_experiment(exp.id)
            experiments.append(exp)
        
        # Assign same user to all experiments
        user_id = "test-user"
        assignments = {}
        
        for exp in experiments:
            variant = manager.assign_variant(exp.id, user_id)
            assert variant is not None
            assignments[exp.id] = variant.id
        
        # Verify consistent assignments
        for exp in experiments:
            variant = manager.assign_variant(exp.id, user_id)
            assert variant.id == assignments[exp.id]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
