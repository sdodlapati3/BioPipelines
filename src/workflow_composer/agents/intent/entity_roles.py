"""
Entity Role Resolver
====================

Determine the semantic role of entities based on context.

Roles help distinguish between:
- Source vs destination locations
- Baseline vs target conditions
- Query vs reference organisms

Usage:
    resolver = EntityRoleResolver()
    
    entities = [
        {"type": "file_path", "value": "/data/input", "start": 10},
        {"type": "file_path", "value": "/data/output", "start": 35},
    ]
    
    resolved = resolver.resolve_roles(
        "copy from /data/input to /data/output",
        entities
    )
    
    print(resolved[0]["role"])  # "source"
    print(resolved[1]["role"])  # "destination"
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RoleDefinition:
    """Definition of an entity role."""
    role: str
    description: str
    entity_types: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)


@dataclass
class DisambiguationRule:
    """A rule for determining entity roles from context."""
    pattern: str
    assign_role: Optional[str] = None
    assign_roles: Optional[Dict[str, str]] = None  # For multi-entity patterns
    when_no_preceding: Optional[str] = None
    compiled_pattern: Optional[re.Pattern] = None
    
    def __post_init__(self):
        if self.compiled_pattern is None:
            # Convert pattern to regex
            regex = self.pattern
            regex = regex.replace("{entity}", r"([^,]+?)")
            regex = regex.replace("{entity1}", r"([^,]+?)")
            regex = regex.replace("{entity2}", r"([^,]+?)")
            self.compiled_pattern = re.compile(regex, re.IGNORECASE)


@dataclass
class RoleConstraint:
    """Constraints on role assignments."""
    exclusive: Optional[List[str]] = None  # Mutually exclusive roles
    paired: Optional[List[str]] = None     # Roles that come together


@dataclass
class ResolvedEntity:
    """An entity with resolved role information."""
    type: str
    value: str
    start: int
    end: int
    role: Optional[str] = None
    role_confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "value": self.value,
            "start": self.start,
            "end": self.end,
            "role": self.role,
            "role_confidence": self.role_confidence,
            **self.metadata,
        }


# =============================================================================
# Entity Role Resolver
# =============================================================================

class EntityRoleResolver:
    """
    Resolve semantic roles for entities based on context.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the role resolver.
        
        Args:
            config_dir: Path to config/nlu/entities directory
        """
        if config_dir is None:
            import workflow_composer
            package_dir = Path(workflow_composer.__file__).parent.parent.parent
            config_dir = package_dir / "config" / "nlu" / "entities"
        
        self.config_dir = Path(config_dir)
        
        # Loaded definitions
        self._roles: Dict[str, RoleDefinition] = {}
        self._rules: List[DisambiguationRule] = []
        self._constraints: List[RoleConstraint] = []
        
        # Context patterns for role detection
        self._context_patterns: Dict[str, List[Tuple[re.Pattern, str]]] = {}
        
        self._load_definitions()
        self._build_context_patterns()
    
    def _load_definitions(self) -> None:
        """Load role definitions from YAML."""
        roles_file = self.config_dir / "roles.yaml"
        
        if not roles_file.exists():
            logger.warning(f"Roles file not found: {roles_file}")
            self._build_default_patterns()
            return
        
        try:
            with open(roles_file, 'r') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load roles: {e}")
            self._build_default_patterns()
            return
        
        # Load roles
        for role_data in data.get("roles", []):
            role = RoleDefinition(
                role=role_data["role"],
                description=role_data.get("description", ""),
                entity_types=role_data.get("entity_types", []),
                examples=role_data.get("examples", []),
            )
            self._roles[role.role] = role
        
        # Load disambiguation rules
        for rule_data in data.get("disambiguation", []):
            rule = DisambiguationRule(
                pattern=rule_data["pattern"],
                assign_role=rule_data.get("assign_role"),
                assign_roles=rule_data.get("assign_roles"),
                when_no_preceding=rule_data.get("when_no_preceding"),
            )
            self._rules.append(rule)
        
        # Load constraints
        for constraint_data in data.get("constraints", []):
            constraint = RoleConstraint(
                exclusive=constraint_data.get("exclusive"),
                paired=constraint_data.get("paired"),
            )
            self._constraints.append(constraint)
        
        logger.info(
            f"Loaded {len(self._roles)} roles, "
            f"{len(self._rules)} disambiguation rules"
        )
    
    def _build_default_patterns(self) -> None:
        """Build default context patterns without YAML file."""
        self._rules = [
            # Source/destination patterns
            DisambiguationRule(
                pattern=r"from\s+([^\s]+)",
                assign_role="source",
            ),
            DisambiguationRule(
                pattern=r"to\s+([^\s]+)",
                assign_role="destination",
            ),
            # Comparison patterns
            DisambiguationRule(
                pattern=r"(\w+)\s+vs\.?\s+",
                assign_role="baseline",
            ),
            DisambiguationRule(
                pattern=r"vs\.?\s+(\w+)",
                assign_role="target",
            ),
            # Homolog patterns
            DisambiguationRule(
                pattern=r"(\w+)\s+homologs?\s+of",
                assign_role="query_organism",
            ),
            DisambiguationRule(
                pattern=r"homologs?\s+of\s+(\w+)",
                assign_role="reference_organism",
            ),
        ]
    
    def _build_context_patterns(self) -> None:
        """Build regex patterns for context-based role detection."""
        # Patterns that indicate source
        self._context_patterns["source"] = [
            (re.compile(r"from\s+", re.IGNORECASE), "before"),
            (re.compile(r"copy\s+", re.IGNORECASE), "after"),
            (re.compile(r"move\s+", re.IGNORECASE), "after"),
            (re.compile(r"download\s+", re.IGNORECASE), "after"),
            (re.compile(r"import\s+", re.IGNORECASE), "after"),
        ]
        
        # Patterns that indicate destination
        self._context_patterns["destination"] = [
            (re.compile(r"to\s+", re.IGNORECASE), "before"),
            (re.compile(r"into\s+", re.IGNORECASE), "before"),
            (re.compile(r"save\s+.*?\s+", re.IGNORECASE), "after"),
            (re.compile(r"output\s+", re.IGNORECASE), "after"),
            (re.compile(r"export\s+.*?\s+to", re.IGNORECASE), "after"),
        ]
        
        # Patterns that indicate baseline (control)
        self._context_patterns["baseline"] = [
            (re.compile(r"compared?\s+to\s+", re.IGNORECASE), "after"),
            (re.compile(r"against\s+", re.IGNORECASE), "after"),
            (re.compile(r"vs\.?\s+", re.IGNORECASE), "after"),
            (re.compile(r"versus\s+", re.IGNORECASE), "after"),
            (re.compile(r"relative\s+to\s+", re.IGNORECASE), "after"),
        ]
        
        # Patterns that indicate target (experimental)
        self._context_patterns["target"] = [
            (re.compile(r"compare\s+", re.IGNORECASE), "after"),
            (re.compile(r"analyze\s+", re.IGNORECASE), "after"),
            (re.compile(r"\s+vs\.?\s*$", re.IGNORECASE), "before"),
        ]
    
    def resolve_roles(
        self,
        text: str,
        entities: List[Dict[str, Any]],
    ) -> List[ResolvedEntity]:
        """
        Resolve roles for entities based on context.
        
        Args:
            text: The full query text
            entities: List of extracted entities with type, value, start, end
            
        Returns:
            List of ResolvedEntity with roles assigned
        """
        resolved = []
        text_lower = text.lower()
        
        for entity in entities:
            role = None
            confidence = 0.5  # Default confidence for unknown role
            
            entity_value = entity.get("value", "")
            entity_start = entity.get("start", 0)
            entity_end = entity.get("end", len(entity_value))
            entity_type = entity.get("type", "unknown")
            
            # Get text before and after the entity (DON'T strip - preserve whitespace for patterns)
            text_before = text_lower[:entity_start]
            text_after = text_lower[entity_end:]
            
            # Try context patterns first (more reliable)
            role, confidence = self._detect_role_from_context(
                text_before, text_after, entity_type
            )
            
            # If no role found from context, try disambiguation rules
            if role is None:
                for rule in self._rules:
                    if rule.compiled_pattern:
                        match = rule.compiled_pattern.search(text_lower)
                        if match:
                            # Check if entity is in the match
                            if entity_value.lower() in match.group(0).lower():
                                if rule.assign_role:
                                    role = rule.assign_role
                                    confidence = 0.9
                                    break
            
            # Validate role against entity type
            if role and not self._is_valid_role_for_type(role, entity_type):
                role = None
                confidence = 0.5
            
            resolved.append(ResolvedEntity(
                type=entity_type,
                value=entity_value,
                start=entity_start,
                end=entity_end,
                role=role,
                role_confidence=confidence,
                metadata=entity.get("metadata", {}),
            ))
        
        # Apply constraints
        resolved = self._apply_constraints(resolved)
        
        return resolved
    
    def _detect_role_from_context(
        self,
        text_before: str,
        text_after: str,
        entity_type: str,
    ) -> Tuple[Optional[str], float]:
        """
        Detect role from surrounding context.
        
        We check for patterns immediately before/after the entity.
        
        Returns:
            (role, confidence) tuple
        """
        best_role = None
        best_confidence = 0.0
        
        for role, patterns in self._context_patterns.items():
            for pattern, position in patterns:
                if position == "before":
                    # Pattern should appear IMMEDIATELY before the entity
                    # Only check the last ~20 chars of text_before
                    check_text = text_before[-20:] if len(text_before) > 20 else text_before
                    match = pattern.search(check_text)
                    if match:
                        # Check that the match is at the end of text_before
                        # (i.e., immediately precedes the entity)
                        match_end = match.end()
                        if match_end >= len(check_text) - 1:  # Allow for trailing space
                            if best_confidence < 0.85:
                                best_role = role
                                best_confidence = 0.85
                elif position == "after":
                    # Pattern should be in text after entity
                    check_text = text_after[:20] if len(text_after) > 20 else text_after
                    match = pattern.search(check_text)
                    if match:
                        # Check that match is near the start
                        if match.start() <= 2:  # Allow for leading space
                            if best_confidence < 0.85:
                                best_role = role
                                best_confidence = 0.85
        
        return best_role, best_confidence if best_role else 0.5
    
    def _is_valid_role_for_type(self, role: str, entity_type: str) -> bool:
        """Check if a role is valid for an entity type."""
        role_def = self._roles.get(role)
        if role_def is None:
            return True  # Unknown role, allow it
        
        if not role_def.entity_types:
            return True  # No type restrictions
        
        return entity_type in role_def.entity_types
    
    def _apply_constraints(
        self, 
        entities: List[ResolvedEntity]
    ) -> List[ResolvedEntity]:
        """Apply role constraints to resolved entities."""
        # Check for exclusive constraint violations
        for constraint in self._constraints:
            if constraint.exclusive:
                # Find entities with these roles
                role_entities = {}
                for entity in entities:
                    if entity.role in constraint.exclusive:
                        if entity.role not in role_entities:
                            role_entities[entity.role] = []
                        role_entities[entity.role].append(entity)
                
                # If same entity has multiple exclusive roles, keep highest confidence
                # (This shouldn't normally happen, but handle edge cases)
        
        # Check for paired constraints and infer missing roles
        for constraint in self._constraints:
            if constraint.paired:
                present_roles = set(
                    e.role for e in entities if e.role in constraint.paired
                )
                
                if len(present_roles) == 1:
                    # One role present, might want to infer the other
                    # This is informational - don't modify entities
                    missing = set(constraint.paired) - present_roles
                    logger.debug(
                        f"Paired role {list(missing)[0]} may be missing; "
                        f"found {list(present_roles)[0]}"
                    )
        
        return entities
    
    def get_role_info(self, role: str) -> Optional[RoleDefinition]:
        """Get information about a role."""
        return self._roles.get(role)
    
    def get_available_roles(self) -> List[str]:
        """Get list of all available roles."""
        return list(self._roles.keys())
    
    def suggest_roles(self, entity_type: str) -> List[str]:
        """Suggest possible roles for an entity type."""
        suggestions = []
        for role_name, role_def in self._roles.items():
            if not role_def.entity_types or entity_type in role_def.entity_types:
                suggestions.append(role_name)
        return suggestions


# =============================================================================
# Singleton Instance
# =============================================================================

_role_resolver: Optional[EntityRoleResolver] = None


def get_role_resolver() -> EntityRoleResolver:
    """Get the singleton role resolver instance."""
    global _role_resolver
    if _role_resolver is None:
        _role_resolver = EntityRoleResolver()
    return _role_resolver
