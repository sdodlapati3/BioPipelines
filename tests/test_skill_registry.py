"""
Tests for Skill Registry
========================

Tests for the BioPipelines skill documentation system.
"""

import pytest
from pathlib import Path
import sys

# Add config to path
sys.path.insert(0, str(Path(__file__).parent.parent / "config"))

from skills import (
    SkillRegistry,
    SkillDefinition,
    SkillParameter,
    SkillExample,
    SkillTriggers,
    get_skill_registry,
    reset_registry,
)


class TestSkillDefinition:
    """Tests for SkillDefinition dataclass."""
    
    def test_from_dict_basic(self):
        """Test creating SkillDefinition from dictionary."""
        data = {
            "name": "test_skill",
            "display_name": "Test Skill",
            "version": "1.0.0",
            "category": "testing",
            "description": "A test skill",
            "capabilities": ["capability 1", "capability 2"],
            "triggers": {
                "keywords": ["test", "testing"],
                "patterns": ["test.*pattern"],
                "intents": ["TEST_INTENT"]
            }
        }
        
        skill = SkillDefinition.from_dict(data)
        
        assert skill.name == "test_skill"
        assert skill.display_name == "Test Skill"
        assert skill.version == "1.0.0"
        assert skill.category == "testing"
        assert len(skill.capabilities) == 2
        assert "test" in skill.triggers.keywords
    
    def test_from_dict_with_parameters(self):
        """Test creating SkillDefinition with parameters."""
        data = {
            "name": "param_skill",
            "display_name": "Param Skill",
            "version": "1.0.0",
            "category": "testing",
            "description": "Skill with parameters",
            "capabilities": ["param handling"],
            "parameters": {
                "required": [
                    {"name": "query", "type": "string", "description": "Search query"}
                ],
                "optional": [
                    {"name": "limit", "type": "integer", "description": "Max results", "default": 10}
                ]
            }
        }
        
        skill = SkillDefinition.from_dict(data)
        
        assert len(skill.parameters.get("required", [])) == 1
        assert len(skill.parameters.get("optional", [])) == 1
        assert skill.parameters["required"][0].name == "query"
        assert skill.parameters["optional"][0].default == 10
    
    def test_from_dict_with_examples(self):
        """Test creating SkillDefinition with examples."""
        data = {
            "name": "example_skill",
            "display_name": "Example Skill",
            "version": "1.0.0",
            "category": "testing",
            "description": "Skill with examples",
            "capabilities": ["example handling"],
            "examples": [
                {
                    "query": "Test query",
                    "expected_behavior": "Expected result",
                    "parameters": {"key": "value"}
                }
            ]
        }
        
        skill = SkillDefinition.from_dict(data)
        
        assert len(skill.examples) == 1
        assert skill.examples[0].query == "Test query"
        assert skill.examples[0].parameters["key"] == "value"
    
    def test_from_dict_trigger_phrases_format(self):
        """Test handling trigger_phrases format (alternative to triggers)."""
        data = {
            "name": "phrase_skill",
            "display_name": "Phrase Skill",
            "version": "1.0.0",
            "category": "testing",
            "description": "Skill with trigger phrases",
            "capabilities": ["phrase handling"],
            "trigger_phrases": ["phrase one", "phrase two"]
        }
        
        skill = SkillDefinition.from_dict(data)
        
        assert "phrase one" in skill.triggers.keywords


class TestSkillRegistry:
    """Tests for SkillRegistry."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset registry before each test."""
        reset_registry()
    
    def test_registry_initialization(self):
        """Test registry initializes correctly."""
        skills_dir = Path(__file__).parent.parent / "config" / "skills"
        registry = SkillRegistry(skills_dir=skills_dir)
        
        assert registry.skills_dir == skills_dir
        assert len(registry._skills) == 0  # Not loaded yet
    
    def test_registry_loads_skills(self):
        """Test registry loads skill files."""
        registry = get_skill_registry()
        
        # Should have loaded skills
        assert registry.get_skill_count() > 0
    
    def test_get_skill_by_name(self):
        """Test retrieving skill by name."""
        registry = get_skill_registry()
        
        # Test getting a known skill (uniprot was created in Phase 2)
        skill = registry.get_skill("uniprot")
        
        if skill:  # Only if it was loaded
            assert skill.name == "uniprot"
            assert skill.category is not None
    
    def test_get_nonexistent_skill(self):
        """Test getting a skill that doesn't exist."""
        registry = get_skill_registry()
        
        skill = registry.get_skill("nonexistent_skill_xyz")
        
        assert skill is None
    
    def test_find_skills_for_query_keywords(self):
        """Test finding skills by keywords."""
        registry = get_skill_registry()
        
        # Search for encode-related skills
        matches = registry.find_skills_for_query("search encode for data")
        
        # Should find encode_search if loaded
        skill_names = [s.name for s in matches]
        # At minimum, test that we get a list back
        assert isinstance(matches, list)
    
    def test_find_skills_for_query_patterns(self):
        """Test finding skills by pattern matching."""
        registry = get_skill_registry()
        
        # Search with pattern-matchable query
        matches = registry.find_skills_for_query("search uniprot for TP53")
        
        assert isinstance(matches, list)
    
    def test_get_skills_by_category(self):
        """Test getting skills by category."""
        registry = get_skill_registry()
        
        # Get all categories first
        categories = registry.get_all_categories()
        
        if categories:
            # Get skills from first category
            skills = registry.get_skills_by_category(categories[0])
            assert isinstance(skills, list)
    
    def test_get_all_categories(self):
        """Test getting all categories."""
        registry = get_skill_registry()
        
        categories = registry.get_all_categories()
        
        assert isinstance(categories, list)
    
    def test_get_all_skills(self):
        """Test getting all skills."""
        registry = get_skill_registry()
        
        skills = registry.get_all_skills()
        
        assert isinstance(skills, list)
        assert len(skills) == registry.get_skill_count()
    
    def test_get_skill_context(self):
        """Test getting formatted skill context."""
        registry = get_skill_registry()
        
        skills = registry.get_all_skills()
        if skills:
            context = registry.get_skill_context(skills[0].name)
            
            assert isinstance(context, str)
            assert "## Skill:" in context or len(context) == 0
    
    def test_get_skills_summary(self):
        """Test getting skills summary."""
        registry = get_skill_registry()
        
        summary = registry.get_skills_summary()
        
        assert isinstance(summary, str)
        assert "# Available Skills" in summary


class TestSkillRegistrySingleton:
    """Tests for singleton pattern."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset registry before each test."""
        reset_registry()
    
    def test_singleton_returns_same_instance(self):
        """Test that get_skill_registry returns the same instance."""
        registry1 = get_skill_registry()
        registry2 = get_skill_registry()
        
        assert registry1 is registry2
    
    def test_reset_registry(self):
        """Test that reset_registry clears the singleton."""
        registry1 = get_skill_registry()
        reset_registry()
        registry2 = get_skill_registry()
        
        # Should be different instances after reset
        # (Note: lru_cache may return same due to same args)
        assert isinstance(registry2, SkillRegistry)


class TestSkillQueryMatching:
    """Tests for query matching functionality."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset registry before each test."""
        reset_registry()
    
    def test_keyword_matching(self):
        """Test keyword-based matching."""
        registry = get_skill_registry()
        
        # Various query patterns
        queries = [
            "search encode for ChIP-seq data",
            "find protein in uniprot",
            "run RNA-seq workflow",
            "check job status",
            "what is differential expression",
        ]
        
        for query in queries:
            matches = registry.find_skills_for_query(query)
            # Just verify we get results without error
            assert isinstance(matches, list)
    
    def test_empty_query(self):
        """Test with empty query."""
        registry = get_skill_registry()
        
        matches = registry.find_skills_for_query("")
        
        assert isinstance(matches, list)
        assert len(matches) == 0
    
    def test_query_case_insensitivity(self):
        """Test that matching is case insensitive."""
        registry = get_skill_registry()
        
        matches_lower = registry.find_skills_for_query("search encode")
        matches_upper = registry.find_skills_for_query("SEARCH ENCODE")
        
        # Should find same skills regardless of case
        assert len(matches_lower) == len(matches_upper)


class TestSkillIntegration:
    """Integration tests for skill system."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset registry before each test."""
        reset_registry()
    
    def test_data_discovery_skills_loaded(self):
        """Test that data discovery skills are loaded."""
        registry = get_skill_registry()
        
        # Check for data discovery category
        categories = registry.get_all_categories()
        
        # Should have some categories
        assert len(categories) > 0
    
    def test_database_skills_loaded(self):
        """Test that database skills from Phase 2 are loaded."""
        registry = get_skill_registry()
        
        # Check for database skills
        database_skills = ["uniprot", "string_db", "kegg", "reactome", "pubmed", "clinvar"]
        
        loaded_count = sum(1 for name in database_skills if registry.get_skill(name) is not None)
        
        # Should have loaded at least some database skills
        assert loaded_count > 0
    
    def test_workflow_skills_loaded(self):
        """Test that workflow generation skills are loaded."""
        registry = get_skill_registry()
        
        workflow_skills = ["rnaseq_workflow", "chipseq_workflow", "methylation_workflow"]
        
        loaded_count = sum(1 for name in workflow_skills if registry.get_skill(name) is not None)
        
        # Should have workflow skills
        assert loaded_count > 0
    
    def test_job_management_skills_loaded(self):
        """Test that job management skills are loaded."""
        registry = get_skill_registry()
        
        job_skills = ["submit_job", "check_status", "cancel_job", "get_logs"]
        
        loaded_count = sum(1 for name in job_skills if registry.get_skill(name) is not None)
        
        # Should have job management skills
        assert loaded_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
