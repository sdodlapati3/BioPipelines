"""
NLU Training Data Loader
========================

Loads and parses YAML-based training data for intents, entities, and patterns.

Features:
- YAML schema validation
- Intent examples with slot annotations
- Entity definitions with synonyms
- Regex pattern loading
- Lookup table support
- Training data balance analysis

Usage:
    loader = TrainingDataLoader()
    loader.load_all()
    
    # Access loaded data
    intents = loader.get_intents()
    entities = loader.get_entities()
    patterns = loader.get_patterns()
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SlotDefinition:
    """Definition of a slot (parameter) for an intent."""
    name: str
    type: str  # "text", "file_path", "organism", "assay_type", etc.
    required: bool = False
    prompt: str = ""
    default: Any = None
    choices: Optional[List[str]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SlotDefinition":
        return cls(
            name=data["name"],
            type=data.get("type", "text"),
            required=data.get("required", False),
            prompt=data.get("prompt", ""),
            default=data.get("default"),
            choices=data.get("choices"),
        )


@dataclass
class IntentDefinition:
    """Definition of an intent with examples and slots."""
    name: str
    description: str = ""
    category: str = "general"
    slots: List[SlotDefinition] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    
    @property
    def required_slots(self) -> List[SlotDefinition]:
        return [s for s in self.slots if s.required]
    
    @property
    def optional_slots(self) -> List[SlotDefinition]:
        return [s for s in self.slots if not s.required]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntentDefinition":
        slots = [SlotDefinition.from_dict(s) for s in data.get("slots", [])]
        return cls(
            name=data["intent"],
            description=data.get("description", ""),
            category=data.get("category", "general"),
            slots=slots,
            examples=data.get("examples", []),
        )


@dataclass
class EntityValue:
    """A single entity value with aliases."""
    canonical: str
    display: str = ""
    aliases: List[str] = field(default_factory=list)
    category: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityValue":
        return cls(
            canonical=data["canonical"],
            display=data.get("display", data["canonical"]),
            aliases=data.get("aliases", []),
            category=data.get("category", ""),
            metadata={k: v for k, v in data.items() 
                     if k not in ("canonical", "display", "aliases", "category")},
        )


@dataclass
class EntityDefinition:
    """Definition of an entity type with values."""
    name: str
    description: str = ""
    values: List[EntityValue] = field(default_factory=list)
    
    def get_alias_map(self) -> Dict[str, str]:
        """Get mapping from all aliases to canonical values."""
        alias_map = {}
        for value in self.values:
            # Add canonical as its own alias
            alias_map[value.canonical.lower()] = value.canonical
            alias_map[value.display.lower()] = value.canonical
            for alias in value.aliases:
                alias_map[alias.lower()] = value.canonical
        return alias_map
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityDefinition":
        values = [EntityValue.from_dict(v) for v in data.get("values", [])]
        return cls(
            name=data["entity"],
            description=data.get("description", ""),
            values=values,
        )


@dataclass
class RegexPattern:
    """A regex pattern for entity extraction."""
    name: str
    entity_type: str
    pattern: str
    compiled: Optional[re.Pattern] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.compiled is None:
            self.compiled = re.compile(self.pattern, re.IGNORECASE)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegexPattern":
        return cls(
            name=data["name"],
            entity_type=data.get("entity_type", "unknown"),
            pattern=data["pattern"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class ParsedExample:
    """A parsed training example with extracted slots."""
    text: str  # Original text with annotations
    clean_text: str  # Text without annotations
    slots: Dict[str, str]  # Extracted slot values
    intent: str


# =============================================================================
# Training Data Loader
# =============================================================================

class TrainingDataLoader:
    """
    Load NLU training data from YAML files.
    
    Directory structure:
        config/nlu/
        ├── intents/
        │   ├── data_operations.yaml
        │   ├── workflow_operations.yaml
        │   └── ...
        ├── entities/
        │   ├── organisms.yaml
        │   ├── assay_types.yaml
        │   └── ...
        ├── synonyms.yaml
        ├── lookup_tables.yaml
        └── regex_patterns.yaml
    """
    
    # Regex to parse slot annotations: [value](slot_name) or [value](slot_name:role)
    SLOT_PATTERN = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the loader.
        
        Args:
            config_dir: Path to config/nlu directory. Auto-detected if None.
        """
        if config_dir is None:
            # Auto-detect from package location
            import workflow_composer
            package_dir = Path(workflow_composer.__file__).parent.parent.parent
            config_dir = package_dir / "config" / "nlu"
        
        self.config_dir = Path(config_dir)
        
        # Loaded data
        self._intents: Dict[str, IntentDefinition] = {}
        self._entities: Dict[str, EntityDefinition] = {}
        self._patterns: List[RegexPattern] = []
        self._synonyms: Dict[str, str] = {}  # alias -> canonical
        self._lookup_tables: Dict[str, List[str]] = {}
        
        # Derived data
        self._entity_alias_maps: Dict[str, Dict[str, str]] = {}
        self._parsed_examples: List[ParsedExample] = []
        
        # Stats
        self._load_errors: List[str] = []
    
    def load_all(self) -> None:
        """Load all training data from config directory."""
        if not self.config_dir.exists():
            logger.warning(f"Config directory not found: {self.config_dir}")
            return
        
        self._load_entities()
        self._load_intents()
        self._load_patterns()
        self._load_synonyms()
        self._load_lookup_tables()
        self._build_alias_maps()
        
        logger.info(
            f"Loaded training data: {len(self._intents)} intents, "
            f"{len(self._entities)} entity types, {len(self._patterns)} patterns"
        )
    
    def _load_yaml(self, path: Path) -> Optional[Dict[str, Any]]:
        """Load a YAML file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self._load_errors.append(f"Failed to load {path}: {e}")
            logger.error(f"Failed to load YAML: {path}: {e}")
            return None
    
    def _load_intents(self) -> None:
        """Load intent definitions from intents/ directory."""
        intents_dir = self.config_dir / "intents"
        if not intents_dir.exists():
            return
        
        for yaml_file in intents_dir.glob("*.yaml"):
            data = self._load_yaml(yaml_file)
            if data is None:
                continue
            
            for intent_data in data.get("intents", []):
                try:
                    intent = IntentDefinition.from_dict(intent_data)
                    self._intents[intent.name] = intent
                    
                    # Parse examples for this intent
                    for example in intent.examples:
                        parsed = self._parse_example(example, intent.name)
                        self._parsed_examples.append(parsed)
                        
                except Exception as e:
                    self._load_errors.append(
                        f"Failed to parse intent in {yaml_file}: {e}"
                    )
                    logger.error(f"Intent parse error: {e}")
    
    def _load_entities(self) -> None:
        """Load entity definitions from entities/ directory."""
        entities_dir = self.config_dir / "entities"
        if not entities_dir.exists():
            return
        
        for yaml_file in entities_dir.glob("*.yaml"):
            data = self._load_yaml(yaml_file)
            if data is None:
                continue
            
            for entity_data in data.get("entities", []):
                try:
                    entity = EntityDefinition.from_dict(entity_data)
                    self._entities[entity.name] = entity
                except Exception as e:
                    self._load_errors.append(
                        f"Failed to parse entity in {yaml_file}: {e}"
                    )
                    logger.error(f"Entity parse error: {e}")
    
    def _load_patterns(self) -> None:
        """Load regex patterns from regex_patterns.yaml."""
        patterns_file = self.config_dir / "regex_patterns.yaml"
        if not patterns_file.exists():
            return
        
        data = self._load_yaml(patterns_file)
        if data is None:
            return
        
        for pattern_data in data.get("patterns", []):
            try:
                pattern = RegexPattern.from_dict(pattern_data)
                self._patterns.append(pattern)
            except Exception as e:
                self._load_errors.append(f"Failed to parse pattern: {e}")
                logger.error(f"Pattern parse error: {e}")
    
    def _load_synonyms(self) -> None:
        """Load global synonyms from synonyms.yaml."""
        synonyms_file = self.config_dir / "synonyms.yaml"
        if not synonyms_file.exists():
            return
        
        data = self._load_yaml(synonyms_file)
        if data is None:
            return
        
        for synonym_data in data.get("synonyms", []):
            canonical = synonym_data.get("canonical", "")
            for alias in synonym_data.get("aliases", []):
                self._synonyms[alias.lower()] = canonical
    
    def _load_lookup_tables(self) -> None:
        """Load lookup tables from lookup_tables.yaml."""
        lookup_file = self.config_dir / "lookup_tables.yaml"
        if not lookup_file.exists():
            return
        
        data = self._load_yaml(lookup_file)
        if data is None:
            return
        
        for table_data in data.get("lookup_tables", []):
            name = table_data.get("name", "")
            values = table_data.get("values", [])
            if name:
                self._lookup_tables[name] = values
    
    def _build_alias_maps(self) -> None:
        """Build alias-to-canonical maps for all entity types."""
        for entity_name, entity_def in self._entities.items():
            self._entity_alias_maps[entity_name] = entity_def.get_alias_map()
    
    def _parse_example(self, example: str, intent: str) -> ParsedExample:
        """
        Parse a training example with slot annotations.
        
        Format: "search for [human](organism) [liver](tissue) data"
        Returns clean text and extracted slots.
        """
        slots = {}
        clean_text = example
        
        # Find all slot annotations
        for match in self.SLOT_PATTERN.finditer(example):
            value = match.group(1)
            slot_spec = match.group(2)
            
            # Handle role annotations: slot_name:role
            if ":" in slot_spec:
                slot_name, role = slot_spec.split(":", 1)
                # Store with role suffix for now
                slots[f"{slot_name}:{role}"] = value
            else:
                slots[slot_spec] = value
        
        # Remove annotations from text
        clean_text = self.SLOT_PATTERN.sub(r'\1', example)
        
        return ParsedExample(
            text=example,
            clean_text=clean_text,
            slots=slots,
            intent=intent,
        )
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    def get_intents(self) -> Dict[str, IntentDefinition]:
        """Get all loaded intent definitions."""
        return self._intents
    
    def get_intent(self, name: str) -> Optional[IntentDefinition]:
        """Get a specific intent definition."""
        return self._intents.get(name)
    
    def get_entities(self) -> Dict[str, EntityDefinition]:
        """Get all loaded entity definitions."""
        return self._entities
    
    def get_entity(self, name: str) -> Optional[EntityDefinition]:
        """Get a specific entity definition."""
        return self._entities.get(name)
    
    def get_patterns(self) -> List[RegexPattern]:
        """Get all loaded regex patterns."""
        return self._patterns
    
    def get_alias_map(self, entity_type: str) -> Dict[str, str]:
        """Get alias-to-canonical map for an entity type."""
        return self._entity_alias_maps.get(entity_type, {})
    
    def get_examples_for_intent(self, intent: str) -> List[ParsedExample]:
        """Get all parsed examples for an intent."""
        return [e for e in self._parsed_examples if e.intent == intent]
    
    def get_clean_examples_for_intent(self, intent: str) -> List[str]:
        """Get clean (no annotations) examples for an intent."""
        return [e.clean_text for e in self._parsed_examples if e.intent == intent]
    
    def normalize_entity(self, text: str, entity_type: str) -> Optional[str]:
        """Normalize an entity value to its canonical form."""
        alias_map = self._entity_alias_maps.get(entity_type, {})
        return alias_map.get(text.lower())
    
    def get_slot_definition(
        self, 
        intent: str, 
        slot_name: str
    ) -> Optional[SlotDefinition]:
        """Get slot definition for an intent."""
        intent_def = self._intents.get(intent)
        if intent_def is None:
            return None
        
        for slot in intent_def.slots:
            if slot.name == slot_name:
                return slot
        return None
    
    def get_required_slots(self, intent: str) -> List[SlotDefinition]:
        """Get required slots for an intent."""
        intent_def = self._intents.get(intent)
        if intent_def is None:
            return []
        return intent_def.required_slots
    
    def get_load_errors(self) -> List[str]:
        """Get any errors that occurred during loading."""
        return self._load_errors
    
    # =========================================================================
    # Balance Analysis
    # =========================================================================
    
    def get_intent_balance_report(self) -> Dict[str, Any]:
        """
        Analyze training data balance across intents.
        
        Returns:
            Report with counts, imbalance ratio, and warnings.
        """
        intent_counts = {}
        for intent_name, intent_def in self._intents.items():
            intent_counts[intent_name] = len(intent_def.examples)
        
        if not intent_counts:
            return {
                "total_examples": 0,
                "intent_counts": {},
                "is_balanced": True,
                "warnings": [],
            }
        
        counts = list(intent_counts.values())
        min_count = min(counts) if counts else 0
        max_count = max(counts) if counts else 0
        total = sum(counts)
        
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        warnings = []
        
        # Check for severely imbalanced intents
        if imbalance_ratio > 10:
            warnings.append(
                f"Severe imbalance: ratio is {imbalance_ratio:.1f}x "
                f"(max={max_count}, min={min_count})"
            )
        
        # Check for intents with too few examples
        for intent, count in intent_counts.items():
            if count < 5:
                warnings.append(
                    f"Intent '{intent}' has only {count} examples (recommend ≥10)"
                )
        
        # Sort by count for report
        sorted_counts = dict(sorted(
            intent_counts.items(), 
            key=lambda x: -x[1]
        ))
        
        return {
            "total_examples": total,
            "intent_counts": sorted_counts,
            "min_examples": min_count,
            "max_examples": max_count,
            "imbalance_ratio": imbalance_ratio,
            "is_balanced": imbalance_ratio <= 5.0,
            "warnings": warnings,
        }


# =============================================================================
# Slot Validation
# =============================================================================

@dataclass
class SlotValidationResult:
    """Result of slot validation."""
    is_complete: bool
    filled_slots: Dict[str, Any]
    missing_slots: List[SlotDefinition]
    clarification_prompt: str = ""


class SlotValidator:
    """
    Validate that required slots are filled for an intent.
    
    Generates clarification prompts for missing required slots.
    """
    
    def __init__(self, loader: TrainingDataLoader):
        self.loader = loader
    
    def validate(
        self, 
        intent: str, 
        extracted_slots: Dict[str, Any]
    ) -> SlotValidationResult:
        """
        Check if all required slots are filled.
        
        Args:
            intent: The detected intent name
            extracted_slots: Slots extracted from the query
            
        Returns:
            SlotValidationResult with completion status and prompts
        """
        intent_def = self.loader.get_intent(intent)
        if intent_def is None:
            # Unknown intent - no validation needed
            return SlotValidationResult(
                is_complete=True,
                filled_slots=extracted_slots,
                missing_slots=[],
            )
        
        missing = []
        filled = dict(extracted_slots)
        
        for slot in intent_def.required_slots:
            if slot.name not in filled or filled[slot.name] is None:
                # Check if default exists
                if slot.default is not None:
                    filled[slot.name] = slot.default
                else:
                    missing.append(slot)
        
        # Generate prompt for first missing slot
        prompt = ""
        if missing:
            first_missing = missing[0]
            prompt = self._generate_prompt(first_missing)
        
        return SlotValidationResult(
            is_complete=len(missing) == 0,
            filled_slots=filled,
            missing_slots=missing,
            clarification_prompt=prompt,
        )
    
    def _generate_prompt(self, slot: SlotDefinition) -> str:
        """Generate a natural language prompt for a missing slot."""
        if slot.prompt:
            return slot.prompt
        
        # Generate default prompts based on slot type
        type_prompts = {
            "file_path": f"Which directory or file for {slot.name}?",
            "organism": "Which organism? (e.g., human, mouse)",
            "tissue": "Which tissue type?",
            "disease": "Which disease or condition?",
            "assay_type": "What type of analysis? (e.g., RNA-seq, ChIP-seq)",
            "dataset_id": "Which dataset? (e.g., GSE12345)",
            "text": f"Please specify {slot.name}.",
            "number": f"What value for {slot.name}?",
            "boolean": f"Should {slot.name} be enabled?",
        }
        
        prompt = type_prompts.get(slot.type, f"Please specify {slot.name}.")
        
        # Add choices if available
        if slot.choices:
            choices_str = ", ".join(slot.choices[:5])
            if len(slot.choices) > 5:
                choices_str += ", ..."
            prompt += f" Options: {choices_str}"
        
        return prompt


# =============================================================================
# Singleton Instance
# =============================================================================

_training_data_loader: Optional[TrainingDataLoader] = None


def get_training_data_loader() -> TrainingDataLoader:
    """Get the singleton training data loader instance."""
    global _training_data_loader
    if _training_data_loader is None:
        _training_data_loader = TrainingDataLoader()
        _training_data_loader.load_all()
    return _training_data_loader


def reload_training_data() -> TrainingDataLoader:
    """Force reload of training data."""
    global _training_data_loader
    _training_data_loader = TrainingDataLoader()
    _training_data_loader.load_all()
    return _training_data_loader
