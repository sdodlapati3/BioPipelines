"""
Slot Prompting System
=====================

When intent is recognized but required slots are missing, prompt the user
for clarification in a natural way.

Features:
- Define required/optional slots per intent
- Natural language prompts
- Context-aware slot filling
- Multi-turn dialogue support

Usage:
    prompter = SlotPrompter()
    
    # Check if slots are complete
    result = prompter.check_slots("DATA_SEARCH", extracted_slots)
    
    if result.needs_prompting:
        print(result.prompt)  # "What organism are you looking for?"
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Types
# =============================================================================

class SlotPriority(Enum):
    """Slot importance level."""
    REQUIRED = "required"       # Must have to execute
    RECOMMENDED = "recommended" # Should have for best results
    OPTIONAL = "optional"       # Nice to have


class PromptStyle(Enum):
    """How to ask for missing slots."""
    QUESTION = "question"       # "What organism?"
    SUGGESTION = "suggestion"   # "You might want to specify an organism"
    EXAMPLE = "example"         # "e.g., human, mouse, zebrafish"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SlotDefinition:
    """Definition of a slot type."""
    name: str
    description: str
    priority: SlotPriority = SlotPriority.OPTIONAL
    entity_type: Optional[str] = None  # Maps to entity extractor type
    examples: List[str] = field(default_factory=list)
    default_value: Optional[Any] = None
    valid_values: Optional[List[str]] = None  # If constrained
    prompt_templates: Dict[PromptStyle, str] = field(default_factory=dict)
    
    def get_prompt(
        self, 
        style: PromptStyle = PromptStyle.QUESTION,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Get the prompt text for this slot."""
        if style in self.prompt_templates:
            template = self.prompt_templates[style]
            if context:
                return template.format(**context)
            return template
        
        # Default templates
        if style == PromptStyle.QUESTION:
            return f"What {self.description.lower()} would you like?"
        elif style == PromptStyle.SUGGESTION:
            return f"You might want to specify a {self.description.lower()}."
        else:  # EXAMPLE
            if self.examples:
                examples_str = ", ".join(self.examples[:3])
                return f"For {self.description.lower()}, options include: {examples_str}"
            return f"Please provide a {self.description.lower()}."


@dataclass
class IntentSlotSchema:
    """Defines which slots an intent requires."""
    intent: str
    required_slots: List[str] = field(default_factory=list)
    recommended_slots: List[str] = field(default_factory=list)
    optional_slots: List[str] = field(default_factory=list)
    slot_dependencies: Dict[str, List[str]] = field(default_factory=dict)  # slot -> depends on
    
    def get_all_slots(self) -> Set[str]:
        """Get all slots for this intent."""
        return set(
            self.required_slots + 
            self.recommended_slots + 
            self.optional_slots
        )


@dataclass
class SlotCheckResult:
    """Result of checking slot completeness."""
    intent: str
    is_complete: bool
    needs_prompting: bool
    missing_required: List[str]
    missing_recommended: List[str]
    prompt: Optional[str] = None
    prompt_slot: Optional[str] = None
    all_prompts: List[Tuple[str, str]] = field(default_factory=list)  # (slot, prompt)
    
    @property
    def can_execute(self) -> bool:
        """Whether we have enough to execute."""
        return len(self.missing_required) == 0


@dataclass
class DialogueState:
    """Track multi-turn dialogue for slot filling."""
    intent: str
    filled_slots: Dict[str, Any] = field(default_factory=dict)
    pending_slots: List[str] = field(default_factory=list)
    current_prompt_slot: Optional[str] = None
    turns: int = 0
    max_turns: int = 3
    
    def is_expired(self) -> bool:
        """Check if dialogue has gone on too long."""
        return self.turns >= self.max_turns


# =============================================================================
# Slot Definitions Registry
# =============================================================================

# Common slots used across intents
COMMON_SLOTS: Dict[str, SlotDefinition] = {
    "organism": SlotDefinition(
        name="organism",
        description="target organism",
        priority=SlotPriority.RECOMMENDED,
        entity_type="ORGANISM",
        examples=["human", "mouse", "zebrafish", "rat"],
        prompt_templates={
            PromptStyle.QUESTION: "What organism are you working with?",
            PromptStyle.SUGGESTION: "You might want to specify an organism (e.g., human, mouse).",
            PromptStyle.EXAMPLE: "Common organisms include: human, mouse, zebrafish, rat, drosophila.",
        }
    ),
    
    "data_type": SlotDefinition(
        name="data_type",
        description="sequencing data type",
        priority=SlotPriority.RECOMMENDED,
        entity_type="ASSAY_TYPE",
        examples=["RNA-seq", "scRNA-seq", "ChIP-seq", "ATAC-seq", "WGS"],
        prompt_templates={
            PromptStyle.QUESTION: "What type of sequencing data are you looking for?",
            PromptStyle.SUGGESTION: "Specifying a data type (e.g., RNA-seq, ChIP-seq) would help narrow results.",
            PromptStyle.EXAMPLE: "Data types include: RNA-seq, scRNA-seq, ChIP-seq, ATAC-seq, Hi-C, methylation.",
        }
    ),
    
    "tissue": SlotDefinition(
        name="tissue",
        description="tissue type",
        priority=SlotPriority.OPTIONAL,
        entity_type="TISSUE",
        examples=["liver", "brain", "heart", "blood"],
        prompt_templates={
            PromptStyle.QUESTION: "What tissue or cell type are you interested in?",
            PromptStyle.SUGGESTION: "You can optionally specify a tissue type (e.g., liver, brain).",
        }
    ),
    
    "disease": SlotDefinition(
        name="disease",
        description="disease or condition",
        priority=SlotPriority.OPTIONAL,
        entity_type="DISEASE",
        examples=["cancer", "diabetes", "alzheimers"],
        prompt_templates={
            PromptStyle.QUESTION: "Are you studying a specific disease or condition?",
        }
    ),
    
    "workflow_type": SlotDefinition(
        name="workflow_type",
        description="workflow type",
        priority=SlotPriority.REQUIRED,
        entity_type="WORKFLOW_TYPE",
        examples=["RNA-seq", "ChIP-seq", "variant calling", "differential expression"],
        prompt_templates={
            PromptStyle.QUESTION: "What type of analysis workflow do you need?",
            PromptStyle.EXAMPLE: "Available workflows: RNA-seq, ChIP-seq, variant calling, differential expression, etc.",
        }
    ),
    
    "input_data": SlotDefinition(
        name="input_data",
        description="input data path or files",
        priority=SlotPriority.REQUIRED,
        examples=["data/raw/*.fastq.gz", "/path/to/samples.csv"],
        prompt_templates={
            PromptStyle.QUESTION: "Where is your input data located?",
            PromptStyle.EXAMPLE: "Provide a path like: data/raw/*.fastq.gz or a sample sheet.",
        }
    ),
    
    "reference_genome": SlotDefinition(
        name="reference_genome",
        description="reference genome version",
        priority=SlotPriority.RECOMMENDED,
        entity_type="REFERENCE",
        examples=["GRCh38", "GRCm39", "hg38", "mm10"],
        default_value="GRCh38",
        prompt_templates={
            PromptStyle.QUESTION: "Which reference genome should be used?",
            PromptStyle.SUGGESTION: "You might want to specify a reference genome (default: GRCh38).",
        }
    ),
    
    "output_format": SlotDefinition(
        name="output_format",
        description="output format",
        priority=SlotPriority.OPTIONAL,
        examples=["BAM", "VCF", "BED", "counts matrix"],
        default_value="default",
    ),
}


# Intent-specific slot schemas
INTENT_SLOT_SCHEMAS: Dict[str, IntentSlotSchema] = {
    "DATA_SCAN": IntentSlotSchema(
        intent="DATA_SCAN",
        required_slots=[],  # None required, scan everything
        recommended_slots=["data_type"],
        optional_slots=["organism", "tissue"],
    ),
    
    "DATA_SEARCH": IntentSlotSchema(
        intent="DATA_SEARCH",
        required_slots=[],  # Keyword search doesn't require slots
        recommended_slots=["organism", "data_type"],
        optional_slots=["tissue", "disease"],
    ),
    
    "DATA_DOWNLOAD": IntentSlotSchema(
        intent="DATA_DOWNLOAD",
        required_slots=["input_data"],  # Must know what to download
        recommended_slots=["organism", "data_type"],
        optional_slots=[],
    ),
    
    "WORKFLOW_CREATE": IntentSlotSchema(
        intent="WORKFLOW_CREATE",
        required_slots=["workflow_type"],
        recommended_slots=["organism", "reference_genome"],
        optional_slots=["output_format"],
        slot_dependencies={
            "reference_genome": ["organism"],  # Reference depends on organism
        }
    ),
    
    "WORKFLOW_LIST": IntentSlotSchema(
        intent="WORKFLOW_LIST",
        required_slots=[],
        recommended_slots=["data_type"],
        optional_slots=[],
    ),
    
    "WORKFLOW_RUN": IntentSlotSchema(
        intent="WORKFLOW_RUN",
        required_slots=["workflow_type", "input_data"],
        recommended_slots=["organism", "reference_genome"],
        optional_slots=["output_format"],
    ),
    
    "ANALYSIS_REQUEST": IntentSlotSchema(
        intent="ANALYSIS_REQUEST",
        required_slots=[],
        recommended_slots=["organism", "data_type"],
        optional_slots=["tissue", "disease"],
    ),
}


# =============================================================================
# Slot Prompter
# =============================================================================

class SlotPrompter:
    """
    Check slot completeness and generate prompts for missing information.
    """
    
    def __init__(
        self,
        slots: Optional[Dict[str, SlotDefinition]] = None,
        schemas: Optional[Dict[str, IntentSlotSchema]] = None,
        prompt_style: PromptStyle = PromptStyle.QUESTION,
    ):
        """
        Initialize the slot prompter.
        
        Args:
            slots: Slot definitions (defaults to COMMON_SLOTS)
            schemas: Intent schemas (defaults to INTENT_SLOT_SCHEMAS)
            prompt_style: Default prompting style
        """
        self.slots = slots or COMMON_SLOTS.copy()
        self.schemas = schemas or INTENT_SLOT_SCHEMAS.copy()
        self.prompt_style = prompt_style
        
        # Track dialogue states by session
        self._dialogue_states: Dict[str, DialogueState] = {}
    
    def check_slots(
        self,
        intent: str,
        filled_slots: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> SlotCheckResult:
        """
        Check if an intent has all required slots filled.
        
        Args:
            intent: The recognized intent
            filled_slots: Slots that were extracted
            context: Additional context for prompts
            
        Returns:
            SlotCheckResult with completeness info and prompts
        """
        schema = self.schemas.get(intent)
        
        if schema is None:
            # Unknown intent, assume complete
            return SlotCheckResult(
                intent=intent,
                is_complete=True,
                needs_prompting=False,
                missing_required=[],
                missing_recommended=[],
            )
        
        # Find missing required slots
        missing_required = [
            slot for slot in schema.required_slots
            if slot not in filled_slots or filled_slots[slot] is None
        ]
        
        # Find missing recommended slots
        missing_recommended = [
            slot for slot in schema.recommended_slots
            if slot not in filled_slots or filled_slots[slot] is None
        ]
        
        is_complete = len(missing_required) == 0
        
        # Decide if we need to prompt
        needs_prompting = len(missing_required) > 0
        
        # Generate prompts
        all_prompts = []
        prompt = None
        prompt_slot = None
        
        # First, prompt for required slots
        if missing_required:
            prompt_slot = missing_required[0]
            slot_def = self.slots.get(prompt_slot)
            if slot_def:
                prompt = slot_def.get_prompt(self.prompt_style, context)
                for s in missing_required:
                    sd = self.slots.get(s)
                    if sd:
                        all_prompts.append((s, sd.get_prompt(self.prompt_style, context)))
        
        # If no required missing but we have recommendations
        elif missing_recommended:
            # Only prompt for recommended if we want to be thorough
            prompt_slot = missing_recommended[0]
            slot_def = self.slots.get(prompt_slot)
            if slot_def:
                # Use suggestion style for recommended slots
                prompt = slot_def.get_prompt(PromptStyle.SUGGESTION, context)
        
        return SlotCheckResult(
            intent=intent,
            is_complete=is_complete,
            needs_prompting=needs_prompting,
            missing_required=missing_required,
            missing_recommended=missing_recommended,
            prompt=prompt,
            prompt_slot=prompt_slot,
            all_prompts=all_prompts,
        )
    
    def fill_slot(
        self,
        session_id: str,
        value: str,
    ) -> Optional[Tuple[str, Any]]:
        """
        Fill a pending slot from user response.
        
        Args:
            session_id: Session identifier
            value: User's response value
            
        Returns:
            (slot_name, value) tuple if successful, None otherwise
        """
        state = self._dialogue_states.get(session_id)
        
        if state is None or state.current_prompt_slot is None:
            return None
        
        slot_name = state.current_prompt_slot
        slot_def = self.slots.get(slot_name)
        
        # Validate if constrained
        if slot_def and slot_def.valid_values:
            normalized = value.lower().strip()
            if normalized not in [v.lower() for v in slot_def.valid_values]:
                logger.warning(f"Invalid value '{value}' for slot {slot_name}")
                # Could return None or accept anyway
        
        # Store the value
        state.filled_slots[slot_name] = value
        state.pending_slots = [s for s in state.pending_slots if s != slot_name]
        state.current_prompt_slot = state.pending_slots[0] if state.pending_slots else None
        state.turns += 1
        
        return (slot_name, value)
    
    def start_dialogue(
        self,
        session_id: str,
        intent: str,
        initial_slots: Dict[str, Any],
    ) -> DialogueState:
        """
        Start a multi-turn slot filling dialogue.
        
        Args:
            session_id: Session identifier
            intent: The recognized intent
            initial_slots: Slots already filled
            
        Returns:
            DialogueState for tracking
        """
        schema = self.schemas.get(intent)
        
        if schema is None:
            # No schema, nothing to fill
            state = DialogueState(intent=intent, filled_slots=initial_slots)
            self._dialogue_states[session_id] = state
            return state
        
        # Determine what needs to be filled
        pending = []
        for slot in schema.required_slots:
            if slot not in initial_slots:
                pending.append(slot)
        
        state = DialogueState(
            intent=intent,
            filled_slots=initial_slots.copy(),
            pending_slots=pending,
            current_prompt_slot=pending[0] if pending else None,
        )
        
        self._dialogue_states[session_id] = state
        return state
    
    def get_dialogue_state(self, session_id: str) -> Optional[DialogueState]:
        """Get current dialogue state for a session."""
        return self._dialogue_states.get(session_id)
    
    def end_dialogue(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        End a dialogue and return filled slots.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Final filled slots dictionary
        """
        state = self._dialogue_states.pop(session_id, None)
        if state:
            return state.filled_slots
        return None
    
    def get_slot_prompt(
        self,
        slot_name: str,
        style: Optional[PromptStyle] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Get prompt for a specific slot.
        
        Args:
            slot_name: Name of the slot
            style: Prompting style (uses default if not specified)
            context: Context for template formatting
            
        Returns:
            Prompt string or None if slot not found
        """
        slot_def = self.slots.get(slot_name)
        if slot_def is None:
            return None
        
        return slot_def.get_prompt(style or self.prompt_style, context)
    
    def apply_defaults(
        self,
        intent: str,
        filled_slots: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply default values for missing optional slots.
        
        Args:
            intent: The recognized intent
            filled_slots: Currently filled slots
            
        Returns:
            Slots dict with defaults applied
        """
        result = filled_slots.copy()
        schema = self.schemas.get(intent)
        
        if schema is None:
            return result
        
        all_slots = schema.get_all_slots()
        
        for slot_name in all_slots:
            if slot_name not in result:
                slot_def = self.slots.get(slot_name)
                if slot_def and slot_def.default_value is not None:
                    result[slot_name] = slot_def.default_value
        
        return result
    
    def register_slot(self, slot: SlotDefinition) -> None:
        """Register a new slot definition."""
        self.slots[slot.name] = slot
    
    def register_schema(self, schema: IntentSlotSchema) -> None:
        """Register a new intent schema."""
        self.schemas[schema.intent] = schema


# =============================================================================
# Singleton Instance
# =============================================================================

_slot_prompter: Optional[SlotPrompter] = None


def get_slot_prompter() -> SlotPrompter:
    """Get the singleton slot prompter instance."""
    global _slot_prompter
    if _slot_prompter is None:
        _slot_prompter = SlotPrompter()
    return _slot_prompter
