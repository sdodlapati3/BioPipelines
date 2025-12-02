"""
BioPipelines Skill Registry
===========================

Loads and manages skill YAML definitions for intelligent tool selection.
Inspired by Claude Scientific Skills SKILL.md format.

Usage:
    from config.skills import get_skill_registry
    
    registry = get_skill_registry()
    matching_skills = registry.find_skills_for_query("search ENCODE for ChIP-seq")
    skill = registry.get_skill("encode_search")
"""

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Try to use PyYAML, fall back to basic parsing if not available
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    logger.warning("PyYAML not available - skill loading will be limited")


@dataclass
class SkillParameter:
    """Definition of a skill parameter."""
    name: str
    type: str
    description: str
    required: bool = False
    default: Any = None
    enum: Optional[List[str]] = None


@dataclass
class SkillExample:
    """Example usage of a skill."""
    query: str
    expected_behavior: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillTriggers:
    """Triggers that invoke a skill."""
    keywords: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    intents: List[str] = field(default_factory=list)


@dataclass
class SkillDefinition:
    """Complete skill definition loaded from YAML."""
    name: str
    display_name: str
    version: str
    category: str
    description: str
    capabilities: List[str]
    triggers: SkillTriggers
    parameters: Dict[str, List[SkillParameter]] = field(default_factory=dict)
    examples: List[SkillExample] = field(default_factory=list)
    related_skills: List[str] = field(default_factory=list)
    tool_binding: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    best_practices: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    prerequisites: Dict[str, Any] = field(default_factory=dict)
    references: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillDefinition":
        """Create SkillDefinition from dictionary (parsed YAML)."""
        # Extract triggers
        triggers_data = data.get("triggers", {})
        if isinstance(triggers_data, dict):
            triggers = SkillTriggers(
                keywords=triggers_data.get("keywords", []),
                patterns=triggers_data.get("patterns", []),
                intents=triggers_data.get("intents", [])
            )
        else:
            triggers = SkillTriggers(keywords=[], patterns=[], intents=[])
        
        # Handle trigger_phrases format (alternative to triggers.keywords)
        if "trigger_phrases" in data:
            triggers.keywords.extend(data.get("trigger_phrases", []))
        
        # Extract parameters
        params_data = data.get("parameters", {})
        parameters = {}
        if isinstance(params_data, dict):
            for key in ["required", "optional"]:
                if key in params_data:
                    parameters[key] = [
                        SkillParameter(
                            name=p.get("name", ""),
                            type=p.get("type", "string"),
                            description=p.get("description", ""),
                            required=key == "required",
                            default=p.get("default"),
                            enum=p.get("enum")
                        )
                        for p in params_data[key]
                    ]
        elif isinstance(params_data, list):
            # Handle list format
            parameters["all"] = [
                SkillParameter(
                    name=p.get("name", ""),
                    type=p.get("type", "string"),
                    description=p.get("description", ""),
                    required=p.get("required", False),
                    default=p.get("default"),
                    enum=p.get("enum")
                )
                for p in params_data
            ]
        
        # Extract examples
        examples_data = data.get("examples", [])
        examples = [
            SkillExample(
                query=ex.get("query", ""),
                expected_behavior=ex.get("expected_behavior", ""),
                parameters=ex.get("parameters", {})
            )
            for ex in examples_data
        ]
        
        return cls(
            name=data.get("name", "unknown"),
            display_name=data.get("display_name", data.get("name", "Unknown")),
            version=data.get("version", "1.0.0"),
            category=data.get("category", "general"),
            description=data.get("description", ""),
            capabilities=data.get("capabilities", []),
            triggers=triggers,
            parameters=parameters,
            examples=examples,
            related_skills=data.get("related_skills", []),
            tool_binding=data.get("tool_binding", {}),
            outputs=data.get("outputs", {}),
            best_practices=data.get("best_practices", []),
            limitations=data.get("limitations", []),
            prerequisites=data.get("prerequisites", {}),
            references=data.get("references", {}),
            aliases=data.get("aliases", [])
        )


class SkillRegistry:
    """
    Registry for all available skills.
    
    Loads skill definitions from YAML files and provides
    query-based skill matching for intelligent tool selection.
    """
    
    def __init__(self, skills_dir: Path = None):
        """
        Initialize the skill registry.
        
        Args:
            skills_dir: Directory containing skill YAML files
        """
        self.skills_dir = skills_dir or Path(__file__).parent
        self._skills: Dict[str, SkillDefinition] = {}
        self._categories: Dict[str, List[str]] = {}
        self._keyword_index: Dict[str, Set[str]] = {}
        self._loaded = False
    
    def _load_skills(self):
        """Load all skill YAML files from the skills directory."""
        if not HAS_YAML:
            logger.warning("Cannot load skills - PyYAML not available")
            self._loaded = True
            return
        
        if self._loaded:
            return
        
        for yaml_file in self.skills_dir.rglob("*.yaml"):
            # Skip schema file
            if yaml_file.name == "schema.yaml":
                continue
            
            try:
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)
                
                if not data or not isinstance(data, dict):
                    continue
                
                # Skip if this is the schema definition
                if "schema_version" in data or "example_template" in data:
                    continue
                
                skill = SkillDefinition.from_dict(data)
                self._skills[skill.name] = skill
                
                # Index by category
                if skill.category not in self._categories:
                    self._categories[skill.category] = []
                self._categories[skill.category].append(skill.name)
                
                # Build keyword index for fast lookup
                self._index_skill_keywords(skill)
                
                logger.debug(f"Loaded skill: {skill.name}")
                
            except Exception as e:
                logger.warning(f"Failed to load skill {yaml_file}: {e}")
        
        self._loaded = True
        logger.info(f"Loaded {len(self._skills)} skills from {self.skills_dir}")
    
    def _index_skill_keywords(self, skill: SkillDefinition):
        """Index skill keywords for fast lookup."""
        # Index trigger keywords
        for keyword in skill.triggers.keywords:
            kw_lower = keyword.lower()
            if kw_lower not in self._keyword_index:
                self._keyword_index[kw_lower] = set()
            self._keyword_index[kw_lower].add(skill.name)
        
        # Index aliases
        for alias in skill.aliases:
            alias_lower = alias.lower()
            if alias_lower not in self._keyword_index:
                self._keyword_index[alias_lower] = set()
            self._keyword_index[alias_lower].add(skill.name)
        
        # Index name
        name_lower = skill.name.lower()
        if name_lower not in self._keyword_index:
            self._keyword_index[name_lower] = set()
        self._keyword_index[name_lower].add(skill.name)
    
    def get_skill(self, name: str) -> Optional[SkillDefinition]:
        """
        Get a skill by name.
        
        Args:
            name: Skill name
            
        Returns:
            SkillDefinition if found, None otherwise
        """
        self._load_skills()
        return self._skills.get(name)
    
    def find_skills_for_query(self, query: str) -> List[SkillDefinition]:
        """
        Find skills that match a query based on triggers.
        
        Args:
            query: User query string
            
        Returns:
            List of matching SkillDefinitions, sorted by relevance
        """
        self._load_skills()
        matches: Dict[str, int] = {}  # skill_name -> score
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for skill in self._skills.values():
            score = 0
            
            # Check keywords (highest weight)
            for keyword in skill.triggers.keywords:
                if keyword.lower() in query_lower:
                    score += 10
            
            # Check patterns
            for pattern in skill.triggers.patterns:
                try:
                    if re.search(pattern, query_lower, re.IGNORECASE):
                        score += 8
                except re.error:
                    pass
            
            # Check skill name
            if skill.name.lower() in query_lower:
                score += 5
            
            # Check aliases
            for alias in skill.aliases:
                if alias.lower() in query_lower:
                    score += 5
            
            # Check capabilities for partial matches
            for capability in skill.capabilities:
                cap_words = set(capability.lower().split())
                overlap = query_words & cap_words
                if len(overlap) >= 2:
                    score += len(overlap)
            
            if score > 0:
                matches[skill.name] = score
        
        # Sort by score and return
        sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)
        return [self._skills[name] for name, _ in sorted_matches]
    
    def get_skills_by_category(self, category: str) -> List[SkillDefinition]:
        """
        Get all skills in a category.
        
        Args:
            category: Category name
            
        Returns:
            List of SkillDefinitions in the category
        """
        self._load_skills()
        skill_names = self._categories.get(category, [])
        return [self._skills[name] for name in skill_names if name in self._skills]
    
    def get_all_categories(self) -> List[str]:
        """Get list of all categories."""
        self._load_skills()
        return list(self._categories.keys())
    
    def get_all_skills(self) -> List[SkillDefinition]:
        """Get all loaded skills."""
        self._load_skills()
        return list(self._skills.values())
    
    def get_skill_count(self) -> int:
        """Get the number of loaded skills."""
        self._load_skills()
        return len(self._skills)
    
    def get_skill_context(self, skill_name: str) -> str:
        """
        Get formatted context for LLM consumption.
        
        Args:
            skill_name: Name of the skill
            
        Returns:
            Formatted markdown string describing the skill
        """
        skill = self.get_skill(skill_name)
        if not skill:
            return ""
        
        lines = [
            f"## Skill: {skill.display_name}",
            "",
            skill.description,
            "",
            "### Capabilities",
        ]
        
        for cap in skill.capabilities:
            lines.append(f"- {cap}")
        
        lines.append("")
        lines.append("### Parameters")
        
        # Required parameters
        required_params = skill.parameters.get("required", [])
        if required_params:
            lines.append("**Required:**")
            for p in required_params:
                lines.append(f"- `{p.name}` ({p.type}): {p.description}")
        
        # Optional parameters
        optional_params = skill.parameters.get("optional", [])
        if optional_params:
            lines.append("")
            lines.append("**Optional:**")
            for p in optional_params:
                default = f", default={p.default}" if p.default is not None else ""
                lines.append(f"- `{p.name}` ({p.type}{default}): {p.description}")
        
        # Examples
        if skill.examples:
            lines.append("")
            lines.append("### Examples")
            for ex in skill.examples[:3]:
                lines.append(f'- "{ex.query}"')
        
        # Best practices
        if skill.best_practices:
            lines.append("")
            lines.append("### Best Practices")
            for practice in skill.best_practices[:3]:
                lines.append(f"- {practice}")
        
        return "\n".join(lines)
    
    def get_skills_summary(self) -> str:
        """
        Get a summary of all available skills.
        
        Returns:
            Formatted markdown summary
        """
        self._load_skills()
        
        lines = ["# Available Skills", ""]
        
        for category in sorted(self._categories.keys()):
            skills = self.get_skills_by_category(category)
            if skills:
                lines.append(f"## {category.replace('_', ' ').title()}")
                for skill in skills:
                    lines.append(f"- **{skill.display_name}**: {skill.description[:80]}...")
                lines.append("")
        
        return "\n".join(lines)


# Singleton instance
_skill_registry: Optional[SkillRegistry] = None


@lru_cache(maxsize=1)
def get_skill_registry(skills_dir: str = None) -> SkillRegistry:
    """
    Get the singleton skill registry.
    
    Args:
        skills_dir: Optional path to skills directory
        
    Returns:
        SkillRegistry instance
    """
    global _skill_registry
    if _skill_registry is None:
        path = Path(skills_dir) if skills_dir else None
        _skill_registry = SkillRegistry(skills_dir=path)
    return _skill_registry


def reset_registry():
    """Reset the singleton registry (for testing)."""
    global _skill_registry
    _skill_registry = None
    get_skill_registry.cache_clear()


__all__ = [
    "SkillDefinition",
    "SkillParameter",
    "SkillExample",
    "SkillTriggers",
    "SkillRegistry",
    "get_skill_registry",
    "reset_registry",
]
