"""
Response Generation System for Professional Chat Agent.

This module provides sophisticated response generation with:
- Template-based responses with multiple variations
- Context-aware response selection
- Personality/tone customization
- Rich response formatting (cards, suggestions, etc.)
- Response history for avoiding repetition
- Conditional response components
- Multi-modal response support

Phase 2 of the Professional Chat Agent implementation.
"""

import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json

from .dialog_state_machine import DialogState


class ResponseTone(Enum):
    """Available response tones/personalities."""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    CONCISE = "concise"
    HELPFUL = "helpful"
    TECHNICAL = "technical"


class ResponseType(Enum):
    """Types of responses the system can generate."""
    TEXT = "text"
    CARD = "card"
    SUGGESTION = "suggestion"
    CONFIRMATION = "confirmation"
    ERROR = "error"
    PROGRESS = "progress"
    CODE = "code"
    LIST = "list"
    TABLE = "table"
    WORKFLOW_PREVIEW = "workflow_preview"


@dataclass
class Suggestion:
    """A suggested action or response."""
    text: str
    action: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Card:
    """A rich card response with structured content."""
    title: str
    content: str
    image_url: Optional[str] = None
    actions: List[Suggestion] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeBlock:
    """A code block with syntax highlighting info."""
    code: str
    language: str = "text"
    title: Optional[str] = None
    copy_button: bool = True


@dataclass
class TableData:
    """Structured table data."""
    headers: List[str]
    rows: List[List[Any]]
    title: Optional[str] = None


@dataclass
class ProgressIndicator:
    """Progress indicator for long-running operations."""
    current_step: int
    total_steps: int
    current_step_name: str
    percentage: float
    estimated_time_remaining: Optional[float] = None


@dataclass
class ResponseComponent:
    """A component of a response (responses can have multiple components)."""
    type: ResponseType
    content: Union[str, Card, Suggestion, CodeBlock, TableData, ProgressIndicator, List, Dict]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Response:
    """Complete response with all components."""
    primary_text: str
    components: List[ResponseComponent] = field(default_factory=list)
    suggestions: List[Suggestion] = field(default_factory=list)
    tone: ResponseTone = ResponseTone.PROFESSIONAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for serialization."""
        return {
            "primary_text": self.primary_text,
            "components": [
                {
                    "type": c.type.value,
                    "content": self._serialize_content(c.content),
                    "metadata": c.metadata
                }
                for c in self.components
            ],
            "suggestions": [
                {"text": s.text, "action": s.action, "metadata": s.metadata}
                for s in self.suggestions
            ],
            "tone": self.tone.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    def _serialize_content(self, content: Any) -> Any:
        """Serialize component content."""
        if isinstance(content, (str, int, float, bool)):
            return content
        if isinstance(content, Card):
            return {
                "title": content.title,
                "content": content.content,
                "image_url": content.image_url,
                "actions": [{"text": a.text, "action": a.action} for a in content.actions]
            }
        if isinstance(content, CodeBlock):
            return {
                "code": content.code,
                "language": content.language,
                "title": content.title
            }
        if isinstance(content, TableData):
            return {
                "headers": content.headers,
                "rows": content.rows,
                "title": content.title
            }
        if isinstance(content, ProgressIndicator):
            return {
                "current_step": content.current_step,
                "total_steps": content.total_steps,
                "current_step_name": content.current_step_name,
                "percentage": content.percentage
            }
        if isinstance(content, list):
            return content
        if isinstance(content, dict):
            return content
        return str(content)


@dataclass
class ResponseTemplate:
    """A response template with variations."""
    intent: str
    variations: List[str]
    conditions: Dict[str, Any] = field(default_factory=dict)
    placeholders: List[str] = field(default_factory=list)
    tone: ResponseTone = ResponseTone.PROFESSIONAL
    follow_up_suggestions: List[str] = field(default_factory=list)
    priority: int = 0
    
    def matches_conditions(self, context: Dict[str, Any]) -> bool:
        """Check if this template matches the given context conditions."""
        for key, expected_value in self.conditions.items():
            actual_value = context.get(key)
            
            if callable(expected_value):
                if not expected_value(actual_value):
                    return False
            elif isinstance(expected_value, (list, tuple)):
                if actual_value not in expected_value:
                    return False
            elif actual_value != expected_value:
                return False
        
        return True


class ResponseHistoryTracker:
    """Tracks response history to avoid repetition."""
    
    def __init__(self, max_history: int = 50, repetition_window: int = 5):
        self.max_history = max_history
        self.repetition_window = repetition_window
        self.history: List[Tuple[str, datetime]] = []
        self.template_usage: Dict[str, List[datetime]] = {}
    
    def record_response(self, response_text: str, template_id: Optional[str] = None):
        """Record a response that was used."""
        now = datetime.now()
        self.history.append((response_text, now))
        
        # Trim history if needed
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        # Track template usage
        if template_id:
            if template_id not in self.template_usage:
                self.template_usage[template_id] = []
            self.template_usage[template_id].append(now)
    
    def was_recently_used(self, response_text: str) -> bool:
        """Check if a response was recently used."""
        recent = self.history[-self.repetition_window:]
        return any(text == response_text for text, _ in recent)
    
    def get_template_usage_count(self, template_id: str, window_hours: int = 24) -> int:
        """Get how many times a template was used in the given time window."""
        if template_id not in self.template_usage:
            return 0
        
        cutoff = datetime.now()
        from datetime import timedelta
        cutoff = cutoff - timedelta(hours=window_hours)
        
        return sum(1 for ts in self.template_usage[template_id] if ts > cutoff)


class TemplateRenderer:
    """Renders templates with placeholder substitution."""
    
    # Pattern for placeholders: {name}, {name|default}, {name:filter}, {name:filter|default}
    PLACEHOLDER_PATTERN = re.compile(r'\{(\w+)(?::(\w+))?(?:\|([^}]*))?\}')
    
    def __init__(self):
        self.filters: Dict[str, Callable[[Any], str]] = {
            'upper': lambda x: str(x).upper(),
            'lower': lambda x: str(x).lower(),
            'title': lambda x: str(x).title(),
            'pluralize': self._pluralize,
            'count': lambda x: str(len(x)) if hasattr(x, '__len__') else str(x),
        }
    
    def _pluralize(self, word: str, count: int = 2) -> str:
        """Simple pluralization."""
        if count == 1:
            return word
        if word.endswith('y'):
            return word[:-1] + 'ies'
        if word.endswith(('s', 'x', 'z', 'ch', 'sh')):
            return word + 'es'
        return word + 's'
    
    def render(self, template: str, variables: Dict[str, Any]) -> str:
        """Render a template with the given variables."""
        def replace_placeholder(match):
            name = match.group(1)
            filter_name = match.group(2)  # Optional filter
            default = match.group(3)  # Optional default
            
            # Get value or use default
            value = variables.get(name, default if default is not None else f"[{name}]")
            
            # Apply filter if specified
            if filter_name and filter_name in self.filters:
                value = self.filters[filter_name](value)
            
            return str(value) if value is not None else ""
        
        return self.PLACEHOLDER_PATTERN.sub(replace_placeholder, template)
    
    def add_filter(self, name: str, func: Callable[[Any], str]):
        """Add a custom filter function."""
        self.filters[name] = func


class ResponseGenerator:
    """
    Main response generator with template management and context-aware selection.
    
    Features:
    - Multiple templates per intent with variations
    - Context-aware template selection
    - Tone/personality customization
    - History tracking to avoid repetition
    - Rich response components
    """
    
    def __init__(
        self,
        default_tone: ResponseTone = ResponseTone.PROFESSIONAL,
        enable_history_tracking: bool = True
    ):
        self.default_tone = default_tone
        self.templates: Dict[str, List[ResponseTemplate]] = {}
        self.renderer = TemplateRenderer()
        self.history = ResponseHistoryTracker() if enable_history_tracking else None
        
        # State-specific templates
        self.state_templates: Dict[DialogState, Dict[str, List[str]]] = {}
        
        # Error response templates
        self.error_templates: Dict[str, List[str]] = {}
        
        # Initialize with default templates
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize default response templates."""
        # Welcome/greeting templates
        self.add_template(ResponseTemplate(
            intent="greeting",
            variations=[
                "Hello! I'm your bioinformatics workflow assistant. How can I help you today?",
                "Hi there! Ready to help you build bioinformatics workflows. What would you like to do?",
                "Welcome! I can help you create and run bioinformatics pipelines. What's your analysis task?"
            ],
            follow_up_suggestions=[
                "Create an RNA-seq pipeline",
                "Analyze ChIP-seq data",
                "Show available workflows"
            ],
            tone=ResponseTone.FRIENDLY
        ))
        
        # Workflow creation templates
        self.add_template(ResponseTemplate(
            intent="create_workflow",
            variations=[
                "I'll help you create a {workflow_type} workflow. Let me gather the necessary information.",
                "Great! Let's set up your {workflow_type} pipeline. I have a few questions to ensure it's configured correctly.",
                "Starting {workflow_type} workflow creation. I'll guide you through the setup process."
            ],
            placeholders=["workflow_type"],
            follow_up_suggestions=[
                "Use default parameters",
                "Customize configuration",
                "Show example inputs"
            ]
        ))
        
        # Slot filling prompts
        self.add_template(ResponseTemplate(
            intent="slot_prompt",
            variations=[
                "I need to know the {slot_name} for your analysis. {slot_description}",
                "What {slot_name} would you like to use? {slot_description}",
                "Please specify the {slot_name}. {slot_description}"
            ],
            placeholders=["slot_name", "slot_description"]
        ))
        
        # Confirmation templates
        self.add_template(ResponseTemplate(
            intent="confirm_action",
            variations=[
                "I'm about to {action_description}. Should I proceed?",
                "Please confirm: {action_description}. Is this correct?",
                "Ready to {action_description}. Do you want me to continue?"
            ],
            placeholders=["action_description"],
            follow_up_suggestions=["Yes, proceed", "No, make changes", "Show details"]
        ))
        
        # Disambiguation templates
        self.add_template(ResponseTemplate(
            intent="disambiguation",
            variations=[
                "I found multiple options for '{query}'. Which one did you mean?",
                "'{query}' could refer to several things. Please clarify:",
                "I need to clarify what you mean by '{query}'. Here are the options:"
            ],
            placeholders=["query"]
        ))
        
        # Success templates
        self.add_template(ResponseTemplate(
            intent="success",
            variations=[
                "Done! {result_summary}",
                "Successfully completed. {result_summary}",
                "All finished! {result_summary}"
            ],
            placeholders=["result_summary"],
            tone=ResponseTone.FRIENDLY
        ))
        
        # Error templates
        self.add_template(ResponseTemplate(
            intent="error",
            variations=[
                "I encountered an issue: {error_message}. {recovery_suggestion}",
                "Something went wrong: {error_message}. {recovery_suggestion}",
                "There was a problem: {error_message}. {recovery_suggestion}"
            ],
            placeholders=["error_message", "recovery_suggestion"]
        ))
        
        # Clarification templates
        self.add_template(ResponseTemplate(
            intent="clarification",
            variations=[
                "I'm not sure I understood. Could you rephrase that?",
                "I didn't quite catch that. Can you tell me more about what you'd like to do?",
                "I need a bit more information to help you. What specifically are you trying to accomplish?"
            ],
            follow_up_suggestions=[
                "Show examples",
                "List available commands",
                "Start over"
            ]
        ))
        
        # Help templates
        self.add_template(ResponseTemplate(
            intent="help",
            variations=[
                "I can help you with bioinformatics workflows. Here's what I can do:",
                "Here's how I can assist you with your analysis:",
                "These are the main things I can help you with:"
            ],
            follow_up_suggestions=[
                "Show workflow types",
                "View documentation",
                "See examples"
            ]
        ))
        
        # Progress update templates
        self.add_template(ResponseTemplate(
            intent="progress",
            variations=[
                "Working on it... {progress_description}",
                "In progress: {progress_description}",
                "Currently {progress_description}"
            ],
            placeholders=["progress_description"]
        ))
        
        # State-specific default responses
        self._initialize_state_templates()
    
    def _initialize_state_templates(self):
        """Initialize templates for each dialog state."""
        self.state_templates = {
            DialogState.IDLE: {
                "entry": [
                    "Ready for your next request!",
                    "What would you like to do next?",
                    "How can I help you?"
                ],
                "timeout": [
                    "Are you still there? Let me know if you need anything.",
                    "Feel free to ask if you have any questions.",
                    "I'm here whenever you're ready."
                ]
            },
            DialogState.UNDERSTANDING: {
                "entry": [
                    "Let me understand your request...",
                    "Processing your request...",
                    "Analyzing what you need..."
                ]
            },
            DialogState.SLOT_FILLING: {
                "entry": [
                    "I need some additional information to proceed.",
                    "Let me gather a few more details.",
                    "A few quick questions before we continue."
                ]
            },
            DialogState.DISAMBIGUATION: {
                "entry": [
                    "I need to clarify something.",
                    "Let me make sure I understand correctly.",
                    "Could you help me narrow this down?"
                ]
            },
            DialogState.CONFIRMING: {
                "entry": [
                    "Here's what I'm about to do:",
                    "Please review and confirm:",
                    "Before proceeding, let me confirm:"
                ]
            },
            DialogState.EXECUTING: {
                "entry": [
                    "Working on it now...",
                    "Executing your request...",
                    "Processing..."
                ]
            },
            DialogState.PRESENTING: {
                "entry": [
                    "Here are the results:",
                    "Done! Here's what I found:",
                    "Completed. Here's the summary:"
                ]
            },
            DialogState.FOLLOW_UP: {
                "entry": [
                    "Is there anything else you'd like to know about this?",
                    "Would you like me to explain any part in more detail?",
                    "Feel free to ask follow-up questions."
                ]
            },
            DialogState.ERROR_RECOVERY: {
                "entry": [
                    "I encountered an issue. Let me try to help resolve it.",
                    "Something went wrong. Here's what we can do:",
                    "There was a problem. Let me suggest some solutions."
                ],
                "retry": [
                    "Let's try that again.",
                    "I'll attempt a different approach.",
                    "Let me retry with different parameters."
                ]
            },
            DialogState.HUMAN_HANDOFF: {
                "entry": [
                    "I'm connecting you with a human expert for better assistance.",
                    "Let me transfer you to someone who can help further.",
                    "I'll get you connected with a specialist."
                ]
            }
        }
    
    def add_template(self, template: ResponseTemplate):
        """Add a response template."""
        if template.intent not in self.templates:
            self.templates[template.intent] = []
        self.templates[template.intent].append(template)
        
        # Sort by priority (higher priority first)
        self.templates[template.intent].sort(key=lambda t: -t.priority)
    
    def get_template(
        self,
        intent: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[ResponseTemplate]:
        """Get the best matching template for an intent and context."""
        if intent not in self.templates:
            return None
        
        context = context or {}
        
        # Find templates that match conditions
        matching = [t for t in self.templates[intent] if t.matches_conditions(context)]
        
        if not matching:
            # Fall back to templates without conditions
            matching = [t for t in self.templates[intent] if not t.conditions]
        
        if not matching:
            return None
        
        # Return highest priority match
        return matching[0]
    
    def generate(
        self,
        intent: str,
        variables: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        tone: Optional[ResponseTone] = None,
        include_suggestions: bool = True
    ) -> Response:
        """
        Generate a response for the given intent.
        
        Args:
            intent: The intent/response type to generate
            variables: Variables for placeholder substitution
            context: Context for conditional template selection
            tone: Override tone/personality
            include_suggestions: Whether to include follow-up suggestions
        
        Returns:
            Complete Response object
        """
        variables = variables or {}
        context = context or {}
        tone = tone or self.default_tone
        
        template = self.get_template(intent, context)
        
        if template:
            # Select a variation, avoiding recent ones if possible
            text = self._select_variation(template, variables)
            
            # Build suggestions
            suggestions = []
            if include_suggestions and template.follow_up_suggestions:
                suggestions = [
                    Suggestion(text=s) for s in template.follow_up_suggestions
                ]
            
            # Record usage
            if self.history:
                template_id = f"{intent}:{hash(text)}"
                self.history.record_response(text, template_id)
            
            return Response(
                primary_text=text,
                suggestions=suggestions,
                tone=template.tone if template.tone else tone,
                metadata={"intent": intent, "template_used": True}
            )
        
        # Fallback: Generate a basic response
        fallback_text = self._generate_fallback(intent, variables)
        return Response(
            primary_text=fallback_text,
            tone=tone,
            metadata={"intent": intent, "template_used": False}
        )
    
    def _select_variation(
        self,
        template: ResponseTemplate,
        variables: Dict[str, Any]
    ) -> str:
        """Select a variation from the template, avoiding recent ones."""
        rendered_variations = [
            self.renderer.render(v, variables) for v in template.variations
        ]
        
        if self.history:
            # Prefer variations that weren't recently used
            unused = [v for v in rendered_variations if not self.history.was_recently_used(v)]
            if unused:
                return random.choice(unused)
        
        return random.choice(rendered_variations)
    
    def _generate_fallback(self, intent: str, variables: Dict[str, Any]) -> str:
        """Generate a fallback response when no template matches."""
        # Simple intent-based fallbacks
        fallbacks = {
            "greeting": "Hello! How can I help you?",
            "success": "Done!",
            "error": "Sorry, something went wrong.",
            "help": "I'm here to help. What would you like to know?",
            "clarification": "Could you please clarify what you'd like to do?",
        }
        
        base = fallbacks.get(intent, f"Processing {intent} request...")
        return self.renderer.render(base, variables)
    
    def generate_state_response(
        self,
        state: DialogState,
        response_type: str = "entry",
        variables: Optional[Dict[str, Any]] = None
    ) -> Response:
        """Generate a response specific to a dialog state."""
        variables = variables or {}
        state_name = state.name.lower()  # e.g., "idle", "understanding"
        
        if state in self.state_templates and response_type in self.state_templates[state]:
            variations = self.state_templates[state][response_type]
            text = random.choice(variations)
            text = self.renderer.render(text, variables)
            
            return Response(
                primary_text=text,
                metadata={"state": state_name, "response_type": response_type}
            )
        
        # Fallback
        return Response(
            primary_text=f"[{state_name}]",
            metadata={"state": state_name, "fallback": True}
        )
    
    def generate_slot_prompt(
        self,
        slot_name: str,
        slot_description: str,
        examples: Optional[List[str]] = None,
        required: bool = True
    ) -> Response:
        """Generate a prompt for filling a slot."""
        variables = {
            "slot_name": slot_name,
            "slot_description": slot_description
        }
        
        response = self.generate("slot_prompt", variables)
        
        # Add examples as suggestions if provided
        if examples:
            response.suggestions = [Suggestion(text=e) for e in examples[:5]]
        
        # Add metadata
        response.metadata["slot_name"] = slot_name
        response.metadata["required"] = required
        
        return response
    
    def generate_disambiguation(
        self,
        query: str,
        options: List[Dict[str, Any]]
    ) -> Response:
        """Generate a disambiguation response with options."""
        variables = {"query": query}
        response = self.generate("disambiguation", variables)
        
        # Add options as components
        suggestions = []
        for i, option in enumerate(options[:5], 1):
            label = option.get("label", str(option))
            description = option.get("description", "")
            action = option.get("action", f"select:{i}")
            
            suggestions.append(Suggestion(
                text=f"{label}" + (f" - {description}" if description else ""),
                action=action,
                metadata=option
            ))
        
        response.suggestions = suggestions
        response.metadata["options_count"] = len(options)
        
        return response
    
    def generate_confirmation(
        self,
        action_description: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Response:
        """Generate a confirmation request."""
        variables = {"action_description": action_description}
        response = self.generate("confirm_action", variables)
        
        # Add details as a card if provided
        if details:
            card = Card(
                title="Action Summary",
                content=self._format_details(details),
                actions=[
                    Suggestion(text="✓ Confirm", action="confirm"),
                    Suggestion(text="✗ Cancel", action="cancel"),
                    Suggestion(text="Modify", action="modify")
                ]
            )
            response.components.append(ResponseComponent(
                type=ResponseType.CARD,
                content=card
            ))
        
        return response
    
    def _format_details(self, details: Dict[str, Any]) -> str:
        """Format details dictionary as readable text."""
        lines = []
        for key, value in details.items():
            formatted_key = key.replace('_', ' ').title()
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            lines.append(f"• {formatted_key}: {value}")
        return "\n".join(lines)
    
    def generate_error(
        self,
        error_message: str,
        recovery_suggestion: str = "Please try again or rephrase your request.",
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> Response:
        """Generate an error response with recovery suggestions."""
        variables = {
            "error_message": error_message,
            "recovery_suggestion": recovery_suggestion
        }
        
        response = self.generate("error", variables)
        response.metadata["error_code"] = error_code
        response.metadata["error_details"] = details
        
        # Add recovery suggestions
        response.suggestions = [
            Suggestion(text="Try again", action="retry"),
            Suggestion(text="Get help", action="help"),
            Suggestion(text="Start over", action="reset")
        ]
        
        return response
    
    def generate_progress(
        self,
        current_step: int,
        total_steps: int,
        current_step_name: str,
        estimated_time: Optional[float] = None
    ) -> Response:
        """Generate a progress update response."""
        percentage = (current_step / total_steps) * 100 if total_steps > 0 else 0
        
        variables = {
            "progress_description": f"{current_step_name} (step {current_step} of {total_steps})"
        }
        
        response = self.generate("progress", variables)
        
        # Add progress indicator component
        progress = ProgressIndicator(
            current_step=current_step,
            total_steps=total_steps,
            current_step_name=current_step_name,
            percentage=percentage,
            estimated_time_remaining=estimated_time
        )
        
        response.components.append(ResponseComponent(
            type=ResponseType.PROGRESS,
            content=progress
        ))
        
        return response
    
    def generate_workflow_preview(
        self,
        workflow_name: str,
        steps: List[Dict[str, str]],
        parameters: Dict[str, Any]
    ) -> Response:
        """Generate a workflow preview response."""
        response = Response(
            primary_text=f"Here's a preview of your {workflow_name} workflow:"
        )
        
        # Add workflow steps as a table
        table = TableData(
            headers=["Step", "Tool", "Description"],
            rows=[[s.get("name", f"Step {i+1}"), s.get("tool", "N/A"), s.get("description", "")] 
                  for i, s in enumerate(steps)],
            title="Workflow Steps"
        )
        
        response.components.append(ResponseComponent(
            type=ResponseType.TABLE,
            content=table
        ))
        
        # Add parameters card
        params_card = Card(
            title="Parameters",
            content=self._format_details(parameters),
            actions=[
                Suggestion(text="Run workflow", action="execute"),
                Suggestion(text="Modify parameters", action="modify"),
                Suggestion(text="Export config", action="export")
            ]
        )
        
        response.components.append(ResponseComponent(
            type=ResponseType.CARD,
            content=params_card
        ))
        
        return response
    
    def generate_code_response(
        self,
        code: str,
        language: str,
        title: Optional[str] = None,
        explanation: Optional[str] = None
    ) -> Response:
        """Generate a response with code content."""
        primary_text = explanation or f"Here's the {language} code:"
        
        response = Response(primary_text=primary_text)
        
        code_block = CodeBlock(
            code=code,
            language=language,
            title=title
        )
        
        response.components.append(ResponseComponent(
            type=ResponseType.CODE,
            content=code_block
        ))
        
        return response
    
    def generate_list_response(
        self,
        title: str,
        items: List[str],
        item_actions: Optional[Dict[str, str]] = None
    ) -> Response:
        """Generate a response with a list of items."""
        response = Response(primary_text=title)
        
        response.components.append(ResponseComponent(
            type=ResponseType.LIST,
            content=items
        ))
        
        if item_actions:
            for item, action in item_actions.items():
                response.suggestions.append(Suggestion(text=item, action=action))
        
        return response
    
    def set_tone(self, tone: ResponseTone):
        """Set the default response tone."""
        self.default_tone = tone
    
    def add_custom_template(
        self,
        intent: str,
        variations: List[str],
        conditions: Optional[Dict[str, Any]] = None,
        follow_up_suggestions: Optional[List[str]] = None,
        tone: Optional[ResponseTone] = None,
        priority: int = 0
    ):
        """Add a custom template programmatically."""
        template = ResponseTemplate(
            intent=intent,
            variations=variations,
            conditions=conditions or {},
            follow_up_suggestions=follow_up_suggestions or [],
            tone=tone or self.default_tone,
            priority=priority
        )
        self.add_template(template)
    
    def load_templates_from_config(self, config: Dict[str, Any]):
        """Load templates from a configuration dictionary."""
        templates_config = config.get("templates", [])
        
        for tc in templates_config:
            tone = ResponseTone(tc.get("tone", "professional")) if tc.get("tone") else None
            
            template = ResponseTemplate(
                intent=tc["intent"],
                variations=tc["variations"],
                conditions=tc.get("conditions", {}),
                placeholders=tc.get("placeholders", []),
                follow_up_suggestions=tc.get("follow_up_suggestions", []),
                tone=tone,
                priority=tc.get("priority", 0)
            )
            self.add_template(template)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get response generation statistics."""
        stats = {
            "total_templates": sum(len(templates) for templates in self.templates.values()),
            "intents_covered": list(self.templates.keys()),
            "default_tone": self.default_tone.value,
        }
        
        if self.history:
            stats["total_responses_generated"] = len(self.history.history)
            stats["unique_templates_used"] = len(self.history.template_usage)
        
        return stats


# Factory function for easy instantiation
def create_response_generator(
    tone: ResponseTone = ResponseTone.PROFESSIONAL,
    custom_templates: Optional[List[ResponseTemplate]] = None,
    config_path: Optional[str] = None
) -> ResponseGenerator:
    """
    Create a configured ResponseGenerator instance.
    
    Args:
        tone: Default response tone
        custom_templates: Additional templates to add
        config_path: Path to YAML/JSON config file with templates
    
    Returns:
        Configured ResponseGenerator
    """
    generator = ResponseGenerator(default_tone=tone)
    
    # Add custom templates
    if custom_templates:
        for template in custom_templates:
            generator.add_template(template)
    
    # Load from config file
    if config_path:
        import os
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    import yaml
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
                generator.load_templates_from_config(config)
    
    return generator
