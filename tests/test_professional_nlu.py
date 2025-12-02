"""
Tests for Professional NLU Components
=====================================

Tests for:
- Training Data Loader (Phase 1)
- Active Learning (Phase 2)
- Slot Prompting (Phase 3)
- Balance Metrics (Phase 4)
- Entity Roles (Phase 5)
"""

import pytest
import tempfile
import json
from pathlib import Path
from typing import Dict, List

import yaml


# =============================================================================
# Phase 1: Training Data Tests
# =============================================================================

class TestTrainingDataLoader:
    """Tests for YAML-based training data loading."""
    
    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """Create a temporary config directory with test YAML files."""
        config_dir = tmp_path / "config" / "nlu"
        intents_dir = config_dir / "intents"
        entities_dir = config_dir / "entities"
        
        intents_dir.mkdir(parents=True)
        entities_dir.mkdir(parents=True)
        
        # Create test intent file
        intent_data = {
            "version": "1.0",
            "intents": [
                {
                    "intent": "DATA_SCAN",
                    "description": "Scan data directories",
                    "category": "data_operations",
                    "slots": [
                        {
                            "name": "data_type",
                            "type": "assay_type",
                            "required": False,
                            "prompt": "What type of data?"
                        }
                    ],
                    "examples": [
                        "scan my data",
                        "list available files",
                        "show me [rna-seq](data_type) data",
                        "what [chip-seq](data_type) samples do we have",
                    ]
                },
                {
                    "intent": "DATA_SEARCH",
                    "description": "Search for datasets",
                    "category": "data_operations",
                    "examples": [
                        "search for [human](organism) data",
                        "find [mouse](organism) [liver](tissue) samples",
                    ]
                }
            ]
        }
        
        with open(intents_dir / "test_intents.yaml", "w") as f:
            yaml.dump(intent_data, f)
        
        # Create test entity file
        entity_data = {
            "version": "1.0",
            "entities": [
                {
                    "entity": "organism",
                    "description": "Target organism",
                    "values": [
                        {
                            "canonical": "Homo sapiens",
                            "display": "human",
                            "aliases": ["human", "homo sapiens", "h. sapiens", "hsa"]
                        },
                        {
                            "canonical": "Mus musculus",
                            "display": "mouse",
                            "aliases": ["mouse", "mus musculus", "m. musculus", "mmu"]
                        }
                    ]
                }
            ]
        }
        
        with open(entities_dir / "test_entities.yaml", "w") as f:
            yaml.dump(entity_data, f)
        
        return config_dir
    
    def test_loader_initialization(self, temp_config_dir):
        """Test loader initializes correctly."""
        from src.workflow_composer.agents.intent.training_data import TrainingDataLoader
        
        loader = TrainingDataLoader(config_dir=temp_config_dir)
        assert loader.config_dir == temp_config_dir
    
    def test_load_intents(self, temp_config_dir):
        """Test loading intent definitions."""
        from src.workflow_composer.agents.intent.training_data import TrainingDataLoader
        
        loader = TrainingDataLoader(config_dir=temp_config_dir)
        loader.load_all()
        
        intents = loader.get_intents()
        assert "DATA_SCAN" in intents
        assert "DATA_SEARCH" in intents
        
        data_scan = intents["DATA_SCAN"]
        assert data_scan.description == "Scan data directories"
        assert len(data_scan.examples) == 4
        assert len(data_scan.slots) == 1
    
    def test_load_entities(self, temp_config_dir):
        """Test loading entity definitions."""
        from src.workflow_composer.agents.intent.training_data import TrainingDataLoader
        
        loader = TrainingDataLoader(config_dir=temp_config_dir)
        loader.load_all()
        
        entities = loader.get_entities()
        assert "organism" in entities
        
        organism = entities["organism"]
        assert len(organism.values) == 2
        assert organism.values[0].canonical == "Homo sapiens"
    
    def test_entity_alias_map(self, temp_config_dir):
        """Test alias-to-canonical mapping."""
        from src.workflow_composer.agents.intent.training_data import TrainingDataLoader
        
        loader = TrainingDataLoader(config_dir=temp_config_dir)
        loader.load_all()
        
        alias_map = loader.get_alias_map("organism")
        
        assert alias_map["human"] == "Homo sapiens"
        assert alias_map["mouse"] == "Mus musculus"
        assert alias_map["hsa"] == "Homo sapiens"
        assert alias_map["mmu"] == "Mus musculus"
    
    def test_normalize_entity(self, temp_config_dir):
        """Test entity normalization."""
        from src.workflow_composer.agents.intent.training_data import TrainingDataLoader
        
        loader = TrainingDataLoader(config_dir=temp_config_dir)
        loader.load_all()
        
        assert loader.normalize_entity("human", "organism") == "Homo sapiens"
        assert loader.normalize_entity("MOUSE", "organism") == "Mus musculus"
        assert loader.normalize_entity("unknown", "organism") is None
    
    def test_parse_example_with_slots(self, temp_config_dir):
        """Test parsing examples with slot annotations."""
        from src.workflow_composer.agents.intent.training_data import TrainingDataLoader
        
        loader = TrainingDataLoader(config_dir=temp_config_dir)
        loader.load_all()
        
        examples = loader.get_examples_for_intent("DATA_SEARCH")
        
        # Find the example with multiple slots
        multi_slot = [e for e in examples if "liver" in e.text][0]
        
        assert "organism" in multi_slot.slots
        assert multi_slot.slots["organism"] == "mouse"
        assert "tissue" in multi_slot.slots
        assert multi_slot.slots["tissue"] == "liver"
        assert multi_slot.clean_text == "find mouse liver samples"
    
    def test_balance_report(self, temp_config_dir):
        """Test training data balance report."""
        from src.workflow_composer.agents.intent.training_data import TrainingDataLoader
        
        loader = TrainingDataLoader(config_dir=temp_config_dir)
        loader.load_all()
        
        report = loader.get_intent_balance_report()
        
        assert report["total_examples"] == 6  # 4 + 2
        assert "DATA_SCAN" in report["intent_counts"]
        assert report["intent_counts"]["DATA_SCAN"] == 4


# =============================================================================
# Phase 2: Active Learning Tests
# =============================================================================

class TestActiveLearning:
    """Tests for active learning feedback system."""
    
    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """Create temporary data directory."""
        data_dir = tmp_path / "feedback"
        data_dir.mkdir(parents=True)
        return data_dir
    
    def test_record_correction(self, temp_data_dir):
        """Test recording a correction."""
        from src.workflow_composer.agents.intent.active_learning import ActiveLearner
        
        learner = ActiveLearner(data_dir=temp_data_dir)
        
        learner.record_correction(
            query="search for liver data",
            predicted="DATA_SCAN",
            corrected="DATA_SEARCH",
        )
        
        # Check it was recorded
        assert len(learner._corrections) == 1
        assert learner._corrections[0].predicted_intent == "DATA_SCAN"
        assert learner._corrections[0].corrected_intent == "DATA_SEARCH"
        
        # Check file was written
        assert (temp_data_dir / "corrections.jsonl").exists()
    
    def test_record_confirmation(self, temp_data_dir):
        """Test recording a confirmation."""
        from src.workflow_composer.agents.intent.active_learning import ActiveLearner
        
        learner = ActiveLearner(data_dir=temp_data_dir)
        
        learner.record_confirmation(
            query="scan my data",
            intent="DATA_SCAN",
            confidence=0.95,
        )
        
        assert len(learner._confirmations) == 1
        assert learner._confirmations[0].intent == "DATA_SCAN"
    
    def test_confusion_matrix(self, temp_data_dir):
        """Test confusion matrix tracking."""
        from src.workflow_composer.agents.intent.active_learning import ActiveLearner
        
        learner = ActiveLearner(data_dir=temp_data_dir)
        
        # Record some corrections
        learner.record_correction("q1", "A", "B")
        learner.record_correction("q2", "A", "B")
        learner.record_correction("q3", "A", "C")
        learner.record_correction("q4", "B", "A")
        
        matrix = learner.get_confusion_matrix()
        
        assert matrix["A"]["B"] == 2
        assert matrix["A"]["C"] == 1
        assert matrix["B"]["A"] == 1
    
    def test_top_confused_pairs(self, temp_data_dir):
        """Test getting top confused pairs."""
        from src.workflow_composer.agents.intent.active_learning import ActiveLearner
        
        learner = ActiveLearner(data_dir=temp_data_dir)
        
        # Record corrections
        for _ in range(5):
            learner.record_correction("q", "DATA_SCAN", "DATA_SEARCH")
        for _ in range(3):
            learner.record_correction("q", "WORKFLOW_CREATE", "WORKFLOW_RUN")
        
        pairs = learner.get_top_confused_pairs()
        
        assert len(pairs) >= 2
        assert pairs[0] == ("DATA_SCAN", "DATA_SEARCH", 5)
        assert pairs[1] == ("WORKFLOW_CREATE", "WORKFLOW_RUN", 3)
    
    def test_learning_metrics(self, temp_data_dir):
        """Test aggregated learning metrics."""
        from src.workflow_composer.agents.intent.active_learning import ActiveLearner
        
        learner = ActiveLearner(data_dir=temp_data_dir)
        
        # Record mixed feedback
        for _ in range(3):
            learner.record_correction("q", "A", "B")
        for _ in range(7):
            learner.record_confirmation("q", "A")
        
        metrics = learner.get_metrics()
        
        assert metrics.total_queries == 10
        assert metrics.total_corrections == 3
        assert metrics.total_confirmations == 7
        assert metrics.correction_rate == 0.3
    
    def test_export_for_retraining(self, temp_data_dir):
        """Test exporting corrections as training data."""
        from src.workflow_composer.agents.intent.active_learning import ActiveLearner
        
        learner = ActiveLearner(data_dir=temp_data_dir)
        
        learner.record_correction("search for liver data", "DATA_SCAN", "DATA_SEARCH")
        learner.record_correction("find brain samples", "DATA_SCAN", "DATA_SEARCH")
        
        yaml_str = learner.export_for_retraining()
        
        assert "DATA_SEARCH" in yaml_str
        assert "search for liver data" in yaml_str
        assert "corrections" in yaml_str


# =============================================================================
# Phase 3: Slot Prompting Tests
# =============================================================================

class TestSlotPrompting:
    """Tests for slot prompting system."""
    
    def test_check_required_slots(self):
        """Test checking for required slots."""
        from src.workflow_composer.agents.intent.slot_prompting import SlotPrompter
        
        prompter = SlotPrompter()
        
        # WORKFLOW_CREATE requires workflow_type
        result = prompter.check_slots(
            "WORKFLOW_CREATE",
            {}  # No slots filled
        )
        
        assert not result.is_complete
        assert result.needs_prompting
        assert "workflow_type" in result.missing_required
        assert result.prompt is not None
    
    def test_check_with_required_filled(self):
        """Test checking when required slots are filled."""
        from src.workflow_composer.agents.intent.slot_prompting import SlotPrompter
        
        prompter = SlotPrompter()
        
        result = prompter.check_slots(
            "WORKFLOW_CREATE",
            {"workflow_type": "RNA-seq"}
        )
        
        assert result.is_complete
        assert result.can_execute
        # May still have recommendations
    
    def test_recommended_slot_suggestion(self):
        """Test suggestion for recommended slots."""
        from src.workflow_composer.agents.intent.slot_prompting import SlotPrompter
        
        prompter = SlotPrompter()
        
        result = prompter.check_slots(
            "DATA_SEARCH",
            {}  # No slots filled, but none required
        )
        
        assert result.is_complete  # Can execute
        assert len(result.missing_recommended) > 0  # But has recommendations
    
    def test_slot_prompt_templates(self):
        """Test custom prompt templates."""
        from src.workflow_composer.agents.intent.slot_prompting import (
            SlotPrompter, PromptStyle
        )
        
        prompter = SlotPrompter()
        
        question_prompt = prompter.get_slot_prompt("organism", PromptStyle.QUESTION)
        suggestion_prompt = prompter.get_slot_prompt("organism", PromptStyle.SUGGESTION)
        
        assert "organism" in question_prompt.lower()
        assert "specify" in suggestion_prompt.lower() or "might" in suggestion_prompt.lower()
    
    def test_apply_defaults(self):
        """Test applying default values."""
        from src.workflow_composer.agents.intent.slot_prompting import SlotPrompter
        
        prompter = SlotPrompter()
        
        filled = prompter.apply_defaults(
            "WORKFLOW_CREATE",
            {"workflow_type": "RNA-seq"}
        )
        
        # reference_genome should get default value
        assert filled.get("reference_genome") == "GRCh38"
    
    def test_dialogue_state_tracking(self):
        """Test multi-turn dialogue state."""
        from src.workflow_composer.agents.intent.slot_prompting import SlotPrompter
        
        prompter = SlotPrompter()
        
        # Start dialogue for WORKFLOW_RUN (requires workflow_type and input_data)
        state = prompter.start_dialogue(
            session_id="test-123",
            intent="WORKFLOW_RUN",
            initial_slots={}
        )
        
        assert len(state.pending_slots) == 2
        assert state.current_prompt_slot is not None
        
        # Fill first slot
        result = prompter.fill_slot("test-123", "RNA-seq")
        assert result is not None
        
        # Check state updated
        state = prompter.get_dialogue_state("test-123")
        assert len(state.filled_slots) == 1
        assert state.turns == 1


# =============================================================================
# Phase 4: Balance Metrics Tests
# =============================================================================

class TestBalanceMetrics:
    """Tests for training data balance analysis."""
    
    def test_analyze_balanced_data(self):
        """Test analysis of balanced training data."""
        from src.workflow_composer.agents.intent.balance_metrics import (
            TrainingDataAnalyzer
        )
        
        # Balanced data
        examples = {
            "INTENT_A": [f"example {i} for A" for i in range(20)],
            "INTENT_B": [f"example {i} for B" for i in range(18)],
            "INTENT_C": [f"example {i} for C" for i in range(22)],
        }
        
        analyzer = TrainingDataAnalyzer(examples=examples)
        report = analyzer.analyze_balance()
        
        assert report.total_intents == 3
        assert report.total_examples == 60
        assert report.imbalance_ratio < 2.0  # Well balanced
        assert len(report.critical_issues) == 0
    
    def test_detect_imbalance(self):
        """Test detection of class imbalance."""
        from src.workflow_composer.agents.intent.balance_metrics import (
            TrainingDataAnalyzer, MAX_IMBALANCE_RATIO
        )
        
        # Severely imbalanced data
        examples = {
            "MAJORITY": [f"example {i}" for i in range(100)],
            "MINORITY": ["example 1", "example 2"],  # Only 2!
        }
        
        analyzer = TrainingDataAnalyzer(examples=examples)
        report = analyzer.analyze_balance()
        
        assert report.imbalance_ratio == 50.0  # 100/2
        assert len(report.critical_issues) > 0  # Should flag this
    
    def test_underrepresented_warning(self):
        """Test warning for underrepresented intents."""
        from src.workflow_composer.agents.intent.balance_metrics import (
            TrainingDataAnalyzer, MIN_EXAMPLES_PER_INTENT
        )
        
        examples = {
            "GOOD_INTENT": [f"example {i}" for i in range(15)],
            "POOR_INTENT": ["ex1", "ex2", "ex3"],  # Only 3
        }
        
        analyzer = TrainingDataAnalyzer(examples=examples)
        report = analyzer.analyze_balance()
        
        assert "POOR_INTENT" in report.intent_stats
        assert report.intent_stats["POOR_INTENT"].is_underrepresented
        
        # Should have warning
        has_warning = any("POOR_INTENT" in w for w in report.warnings)
        assert has_warning
    
    def test_vocabulary_overlap_detection(self):
        """Test detection of high vocabulary overlap."""
        from src.workflow_composer.agents.intent.balance_metrics import (
            TrainingDataAnalyzer
        )
        
        # Very similar intents (high overlap)
        examples = {
            "INTENT_A": [
                "search for human data",
                "find human samples",
                "look for human files",
            ],
            "INTENT_B": [
                "search for mouse data",
                "find mouse samples",
                "look for mouse files",
            ],
        }
        
        analyzer = TrainingDataAnalyzer(examples=examples)
        report = analyzer.analyze_balance()
        
        # Should detect high overlap
        assert len(report.high_overlap_pairs) > 0
        assert report.high_overlap_pairs[0][2] > 0.5  # High overlap
    
    def test_quality_scores(self):
        """Test quality score computation."""
        from src.workflow_composer.agents.intent.balance_metrics import (
            TrainingDataAnalyzer
        )
        
        examples = {
            "INTENT_A": [f"unique phrase {i} for intent A category" for i in range(15)],
            "INTENT_B": [f"different words {i} for intent B type" for i in range(15)],
        }
        
        analyzer = TrainingDataAnalyzer(examples=examples)
        report = analyzer.analyze_balance()
        
        assert 0 <= report.overall_score <= 1
        assert 0 <= report.balance_score <= 1
        assert 0 <= report.diversity_score <= 1
        assert 0 <= report.coverage_score <= 1
    
    def test_suggestions_generated(self):
        """Test that improvement suggestions are generated."""
        from src.workflow_composer.agents.intent.balance_metrics import (
            TrainingDataAnalyzer
        )
        
        # Problematic data
        examples = {
            "BIG_INTENT": [f"example {i}" for i in range(50)],
            "SMALL_INTENT": ["example 1", "example 2"],
        }
        
        analyzer = TrainingDataAnalyzer(examples=examples)
        report = analyzer.analyze_balance()
        
        # Should have suggestions
        assert len(report.suggestions) > 0


# =============================================================================
# Phase 5: Entity Roles Tests
# =============================================================================

class TestEntityRoles:
    """Tests for entity role resolution."""
    
    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """Create temporary config with roles."""
        config_dir = tmp_path / "config" / "nlu" / "entities"
        config_dir.mkdir(parents=True)
        
        roles_data = {
            "roles": [
                {
                    "role": "source",
                    "description": "Origin location",
                    "entity_types": ["file_path", "directory"],
                },
                {
                    "role": "destination",
                    "description": "Target location",
                    "entity_types": ["file_path", "directory"],
                },
                {
                    "role": "baseline",
                    "description": "Control condition",
                    "entity_types": ["condition"],
                },
            ],
            "disambiguation": [
                {"pattern": "from {entity}", "assign_role": "source"},
                {"pattern": "to {entity}", "assign_role": "destination"},
            ],
            "constraints": [
                {"exclusive": ["source", "destination"]},
            ]
        }
        
        with open(config_dir / "roles.yaml", "w") as f:
            yaml.dump(roles_data, f)
        
        return config_dir
    
    def test_resolve_source_destination(self, temp_config_dir):
        """Test resolving source/destination roles."""
        from src.workflow_composer.agents.intent.entity_roles import EntityRoleResolver
        
        resolver = EntityRoleResolver(config_dir=temp_config_dir)
        
        # "copy from /data/input to /data/output"
        #            ^10       ^21  ^25         ^37
        entities = [
            {"type": "file_path", "value": "/data/input", "start": 10, "end": 21},
            {"type": "file_path", "value": "/data/output", "start": 25, "end": 37},
        ]
        
        resolved = resolver.resolve_roles(
            "copy from /data/input to /data/output",
            entities
        )
        
        assert len(resolved) == 2
        assert resolved[0].role == "source"
        assert resolved[1].role == "destination"
    
    def test_role_from_context(self, temp_config_dir):
        """Test role detection from context patterns."""
        from src.workflow_composer.agents.intent.entity_roles import EntityRoleResolver
        
        resolver = EntityRoleResolver(config_dir=temp_config_dir)
        
        # "save output to results/"
        #                ^15     ^23
        entities = [
            {"type": "file_path", "value": "results/", "start": 15, "end": 23},
        ]
        
        resolved = resolver.resolve_roles(
            "save output to results/",
            entities
        )
        
        assert resolved[0].role == "destination"
    
    def test_suggest_roles_for_type(self, temp_config_dir):
        """Test suggesting roles for entity types."""
        from src.workflow_composer.agents.intent.entity_roles import EntityRoleResolver
        
        resolver = EntityRoleResolver(config_dir=temp_config_dir)
        
        suggestions = resolver.suggest_roles("file_path")
        
        assert "source" in suggestions
        assert "destination" in suggestions
    
    def test_default_patterns_without_yaml(self, tmp_path):
        """Test default patterns when YAML is missing."""
        from src.workflow_composer.agents.intent.entity_roles import EntityRoleResolver
        
        # Point to empty directory
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        resolver = EntityRoleResolver(config_dir=empty_dir)
        
        # Should still have default rules
        entities = [
            {"type": "path", "value": "/input", "start": 5, "end": 11},
        ]
        
        resolved = resolver.resolve_roles("from /input", entities)
        
        # Should resolve using defaults
        assert len(resolved) == 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestNLUIntegration:
    """Integration tests for the full NLU system."""
    
    def test_import_all_modules(self):
        """Test that all modules import correctly."""
        from src.workflow_composer.agents.intent import (
            TrainingDataLoader,
            ActiveLearner,
            SlotPrompter,
            TrainingDataAnalyzer,
            EntityRoleResolver,
        )
        
        # Just verify they import
        assert TrainingDataLoader is not None
        assert ActiveLearner is not None
        assert SlotPrompter is not None
        assert TrainingDataAnalyzer is not None
        assert EntityRoleResolver is not None
    
    def test_singleton_accessors(self):
        """Test singleton accessor functions."""
        from src.workflow_composer.agents.intent import (
            get_active_learner,
            get_slot_prompter,
            get_role_resolver,
        )
        
        # Get instances
        learner1 = get_active_learner()
        learner2 = get_active_learner()
        
        assert learner1 is learner2  # Same instance
        
        prompter = get_slot_prompter()
        assert prompter is not None
        
        resolver = get_role_resolver()
        assert resolver is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
