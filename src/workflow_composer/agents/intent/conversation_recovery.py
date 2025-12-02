"""
Conversation Recovery & Clarification System
=============================================

Professional-grade error handling and clarification for chat agents.

Based on patterns from:
- Rasa: Conversation repair, clarification, fallback, error patterns
- Dialogflow: Fallback intents, follow-up handling
- ChatGPT: Graceful degradation, helpful suggestions

Patterns Implemented:
1. Clarification: Ask for more info when intent is unclear
2. Correction: Handle "no, I meant X" gracefully  
3. Error Acknowledgment: "Sorry, something went wrong"
4. Retry Strategy: Escalate to better models on failure
5. Fallback: Offer alternatives when stuck
6. Human Handoff: Suggest manual intervention when needed

Usage:
    recovery = ConversationRecovery(agent)
    
    # When intent confidence is low
    response = recovery.handle_low_confidence(query, confidence=0.25)
    
    # When an error occurs
    response = recovery.handle_error(error, query)
    
    # When user corrects the agent
    response = recovery.handle_correction(original_intent, corrected_intent)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from enum import Enum, auto
import re

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class RecoveryStrategy(Enum):
    """Strategies for conversation recovery."""
    CLARIFY = auto()           # Ask for clarification
    REPHRASE = auto()          # Ask user to rephrase
    SUGGEST = auto()           # Suggest alternatives
    RETRY_BETTER_MODEL = auto()  # Retry with better LLM
    ACKNOWLEDGE_ERROR = auto()   # Acknowledge and apologize
    OFFER_ALTERNATIVES = auto()  # Offer alternative actions
    HUMAN_HANDOFF = auto()       # Suggest human intervention
    FALLBACK_RESPONSE = auto()   # Default helpful response


class ErrorCategory(Enum):
    """Categories of errors for appropriate handling."""
    INTENT_UNCLEAR = auto()     # Couldn't understand intent
    TOOL_FAILED = auto()        # Tool execution failed
    API_ERROR = auto()          # External API error
    VALIDATION_ERROR = auto()   # Invalid parameters
    PERMISSION_ERROR = auto()   # Not allowed
    RESOURCE_NOT_FOUND = auto() # File/dataset not found
    TIMEOUT = auto()            # Operation timed out
    INTERNAL_ERROR = auto()     # Unexpected error
    USER_CORRECTION = auto()    # User correcting the agent


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RecoveryResponse:
    """Response from recovery system."""
    message: str
    strategy: RecoveryStrategy
    suggestions: List[str] = field(default_factory=list)
    should_retry: bool = False
    retry_with_model: Optional[str] = None
    needs_user_input: bool = True
    context_update: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorContext:
    """Context about an error for smart recovery."""
    error: Exception
    query: str
    category: ErrorCategory
    tool_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# CLARIFICATION TEMPLATES
# =============================================================================

CLARIFICATION_TEMPLATES = {
    "general": [
        "I'm not quite sure what you mean. Could you rephrase that?",
        "I want to make sure I understand correctly. Are you asking me to {guess}?",
        "Could you give me a bit more detail about what you'd like to do?",
    ],
    
    "data": [
        "I'm not sure which data you're referring to. Could you specify:\n"
        "• A path (e.g., `/data/methylation`)\n"
        "• A dataset ID (e.g., `GSE12345`)\n"
        "• Or describe what kind of data you're looking for?",
    ],
    
    "workflow": [
        "I'd be happy to help with your workflow! Could you tell me:\n"
        "• What type of analysis? (RNA-seq, ChIP-seq, methylation, etc.)\n"
        "• Where is your input data?",
    ],
    
    "ambiguous_intent": [
        "I detected a few possible interpretations:\n{options}\n\nWhich did you mean?",
    ],
    
    "missing_parameter": [
        "I need a bit more information to help you.\n{missing_info}",
    ],
}

ERROR_TEMPLATES = {
    ErrorCategory.TOOL_FAILED: [
        "I ran into an issue while {action}. Let me try a different approach.",
        "That didn't work as expected. Here's what happened: {brief_error}\n\nWould you like me to try again or try something else?",
    ],
    
    ErrorCategory.API_ERROR: [
        "I'm having trouble connecting to {service}. This might be temporary.\n\n"
        "Would you like me to:\n• Try again in a moment\n• Search a different database\n• Work with local data instead?",
    ],
    
    ErrorCategory.RESOURCE_NOT_FOUND: [
        "I couldn't find {resource}. Let me help you locate it.\n\n"
        "• Would you like me to scan for available files?\n"
        "• Or search for a different dataset?",
    ],
    
    ErrorCategory.VALIDATION_ERROR: [
        "There's an issue with the parameters: {issue}\n\n"
        "Could you check and provide the correct {parameter}?",
    ],
    
    ErrorCategory.TIMEOUT: [
        "The operation is taking longer than expected. This can happen with large datasets.\n\n"
        "Options:\n• Wait a bit longer\n• Try with a smaller subset\n• Cancel and try something else",
    ],
    
    ErrorCategory.INTERNAL_ERROR: [
        "Something unexpected happened on my end. I apologize for the inconvenience.\n\n"
        "I've logged the issue. Would you like to try again, or can I help with something else?",
    ],
}

CORRECTION_TEMPLATES = [
    "Got it! I understand now - you want to {correct_action}. Let me do that instead.",
    "Thanks for the clarification! Switching to {correct_action}.",
    "No problem, I'll {correct_action} instead.",
]


# =============================================================================
# CONVERSATION RECOVERY
# =============================================================================

class ConversationRecovery:
    """
    Handles conversation recovery, clarification, and error handling.
    
    Design Philosophy:
    1. Never leave the user stuck - always offer a path forward
    2. Be honest about mistakes - acknowledge when things go wrong
    3. Be helpful - suggest alternatives, don't just say "error"
    4. Learn from corrections - record for future improvement
    """
    
    # Confidence thresholds
    HIGH_CONFIDENCE = 0.7
    MEDIUM_CONFIDENCE = 0.4
    LOW_CONFIDENCE = 0.25
    
    def __init__(
        self,
        retry_with_better_model: bool = True,
        max_retries: int = 2,
        fallback_model: str = "claude-3-opus",
    ):
        """
        Initialize recovery system.
        
        Args:
            retry_with_better_model: Whether to escalate to better LLM on failure
            max_retries: Maximum retry attempts
            fallback_model: Model to use for difficult queries
        """
        self.retry_with_better_model = retry_with_better_model
        self.max_retries = max_retries
        self.fallback_model = fallback_model
        
        # Track retry state
        self._retry_counts: Dict[str, int] = {}
        self._last_error: Optional[ErrorContext] = None
    
    # =========================================================================
    # LOW CONFIDENCE HANDLING
    # =========================================================================
    
    def handle_low_confidence(
        self,
        query: str,
        confidence: float,
        detected_intent: str = None,
        alternative_intents: List[tuple] = None,  # [(intent, confidence), ...]
    ) -> RecoveryResponse:
        """
        Handle queries where intent confidence is low.
        
        Strategy based on confidence level:
        - Very low (<0.25): Ask for clarification
        - Low (0.25-0.4): Suggest interpretation and ask
        - Medium (0.4-0.7): Proceed but verify
        """
        if confidence < self.LOW_CONFIDENCE:
            # Very uncertain - ask for clarification
            return self._create_clarification_response(query)
        
        elif confidence < self.MEDIUM_CONFIDENCE:
            # Low confidence - suggest interpretation
            if alternative_intents and len(alternative_intents) > 1:
                return self._create_disambiguation_response(
                    query, detected_intent, alternative_intents
                )
            else:
                return self._create_verification_response(
                    query, detected_intent, confidence
                )
        
        else:
            # Medium confidence - proceed but note uncertainty
            return RecoveryResponse(
                message="",  # No message - proceed with action
                strategy=RecoveryStrategy.SUGGEST,
                should_retry=False,
                needs_user_input=False,
            )
    
    def _create_clarification_response(self, query: str) -> RecoveryResponse:
        """Create response asking for clarification."""
        # Detect query category for appropriate template
        category = self._detect_query_category(query)
        
        templates = CLARIFICATION_TEMPLATES.get(category, CLARIFICATION_TEMPLATES["general"])
        message = templates[0]  # Use first template
        
        # Generate helpful suggestions based on query
        suggestions = self._generate_suggestions(query, category)
        
        return RecoveryResponse(
            message=message,
            strategy=RecoveryStrategy.CLARIFY,
            suggestions=suggestions,
            needs_user_input=True,
        )
    
    def _create_disambiguation_response(
        self,
        query: str,
        detected_intent: str,
        alternatives: List[tuple],
    ) -> RecoveryResponse:
        """Create response for ambiguous intents."""
        # Format alternatives
        options = []
        for i, (intent, conf) in enumerate(alternatives[:3], 1):
            intent_desc = self._intent_to_description(intent)
            options.append(f"{i}. {intent_desc}")
        
        options_str = "\n".join(options)
        
        message = CLARIFICATION_TEMPLATES["ambiguous_intent"][0].format(
            options=options_str
        )
        
        return RecoveryResponse(
            message=message,
            strategy=RecoveryStrategy.CLARIFY,
            suggestions=[self._intent_to_description(alt[0]) for alt in alternatives[:3]],
            needs_user_input=True,
            context_update={"disambiguation_options": alternatives},
        )
    
    def _create_verification_response(
        self,
        query: str,
        detected_intent: str,
        confidence: float,
    ) -> RecoveryResponse:
        """Create response verifying interpretation."""
        intent_desc = self._intent_to_description(detected_intent)
        
        message = (
            f"Just to confirm - you'd like me to {intent_desc}?\n\n"
            f"Say 'yes' to proceed, or clarify if I misunderstood."
        )
        
        return RecoveryResponse(
            message=message,
            strategy=RecoveryStrategy.CLARIFY,
            suggestions=["Yes, proceed", "No, I meant something else"],
            needs_user_input=True,
            context_update={"pending_confirmation": detected_intent},
        )
    
    # =========================================================================
    # ERROR HANDLING
    # =========================================================================
    
    def handle_error(
        self,
        error: Exception,
        query: str,
        tool_name: str = None,
        parameters: Dict = None,
    ) -> RecoveryResponse:
        """
        Handle an error gracefully.
        
        Args:
            error: The exception that occurred
            query: The user's query
            tool_name: Tool that failed (if any)
            parameters: Parameters that were used
            
        Returns:
            RecoveryResponse with appropriate message and suggestions
        """
        # Categorize the error
        category = self._categorize_error(error, tool_name)
        
        # Create error context
        context = ErrorContext(
            error=error,
            query=query,
            category=category,
            tool_name=tool_name,
            parameters=parameters or {},
            retry_count=self._retry_counts.get(query, 0),
        )
        self._last_error = context
        
        # Get appropriate response based on category
        return self._get_error_response(context)
    
    def _categorize_error(
        self, 
        error: Exception, 
        tool_name: str = None
    ) -> ErrorCategory:
        """Categorize an error for appropriate handling."""
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Check for specific error types
        if "timeout" in error_str or "timed out" in error_str:
            return ErrorCategory.TIMEOUT
        
        if "not found" in error_str or "404" in error_str:
            return ErrorCategory.RESOURCE_NOT_FOUND
        
        if "permission" in error_str or "unauthorized" in error_str or "403" in error_str:
            return ErrorCategory.PERMISSION_ERROR
        
        if "validation" in error_str or "invalid" in error_str:
            return ErrorCategory.VALIDATION_ERROR
        
        if "connection" in error_str or "api" in error_str or "request" in error_str:
            return ErrorCategory.API_ERROR
        
        if tool_name:
            return ErrorCategory.TOOL_FAILED
        
        return ErrorCategory.INTERNAL_ERROR
    
    def _get_error_response(self, context: ErrorContext) -> RecoveryResponse:
        """Generate appropriate error response."""
        templates = ERROR_TEMPLATES.get(
            context.category, 
            ERROR_TEMPLATES[ErrorCategory.INTERNAL_ERROR]
        )
        
        # Format template with context
        template = templates[0]
        message = self._format_error_template(template, context)
        
        # Determine if we should retry
        should_retry = (
            context.retry_count < self.max_retries and
            context.category in [ErrorCategory.TOOL_FAILED, ErrorCategory.API_ERROR, ErrorCategory.TIMEOUT]
        )
        
        # Generate suggestions
        suggestions = self._get_error_suggestions(context)
        
        return RecoveryResponse(
            message=message,
            strategy=RecoveryStrategy.ACKNOWLEDGE_ERROR,
            suggestions=suggestions,
            should_retry=should_retry,
            retry_with_model=self.fallback_model if self.retry_with_better_model else None,
            needs_user_input=True,
        )
    
    def _format_error_template(
        self, 
        template: str, 
        context: ErrorContext
    ) -> str:
        """Format error template with context."""
        replacements = {
            "action": context.tool_name or "processing your request",
            "brief_error": str(context.error)[:100],
            "service": self._get_service_name(context.tool_name),
            "resource": context.parameters.get("path") or context.parameters.get("dataset_id") or "the requested item",
            "issue": str(context.error)[:150],
            "parameter": list(context.parameters.keys())[0] if context.parameters else "information",
        }
        
        result = template
        for key, value in replacements.items():
            result = result.replace("{" + key + "}", str(value))
        
        return result
    
    def _get_error_suggestions(self, context: ErrorContext) -> List[str]:
        """Generate helpful suggestions for error recovery."""
        suggestions = []
        
        if context.category == ErrorCategory.RESOURCE_NOT_FOUND:
            suggestions.extend([
                "Scan for available files",
                "Search online databases",
                "Check the path and try again",
            ])
        
        elif context.category == ErrorCategory.API_ERROR:
            suggestions.extend([
                "Try again",
                "Search a different database",
                "Work with local data",
            ])
        
        elif context.category == ErrorCategory.VALIDATION_ERROR:
            suggestions.extend([
                "Provide correct parameters",
                "Show available options",
                "Start over",
            ])
        
        elif context.category == ErrorCategory.TIMEOUT:
            suggestions.extend([
                "Wait and retry",
                "Try with smaller data",
                "Cancel operation",
            ])
        
        else:
            suggestions.extend([
                "Try again",
                "Try a different approach",
                "Get help",
            ])
        
        return suggestions[:3]
    
    # =========================================================================
    # CORRECTION HANDLING
    # =========================================================================
    
    def handle_correction(
        self,
        original_intent: str,
        corrected_intent: str,
        query: str,
    ) -> RecoveryResponse:
        """
        Handle user correction ("No, I meant X").
        
        Args:
            original_intent: What the agent thought
            corrected_intent: What the user actually wanted
            query: The original query
            
        Returns:
            RecoveryResponse acknowledging correction
        """
        correct_desc = self._intent_to_description(corrected_intent)
        
        message = CORRECTION_TEMPLATES[0].format(correct_action=correct_desc)
        
        return RecoveryResponse(
            message=message,
            strategy=RecoveryStrategy.SUGGEST,
            suggestions=[],
            should_retry=False,
            needs_user_input=False,
            context_update={
                "corrected_from": original_intent,
                "corrected_to": corrected_intent,
            },
        )
    
    # =========================================================================
    # FALLBACK HANDLING
    # =========================================================================
    
    def get_fallback_response(self, query: str) -> RecoveryResponse:
        """
        Get a fallback response when nothing else works.
        
        This is the "cannot handle" pattern from Rasa.
        """
        message = (
            "I'm not sure how to help with that specific request, "
            "but I can help you with:\n\n"
            "📊 **Data Management**\n"
            "• `scan data` - See available files\n"
            "• `search for [organism] [assay]` - Find datasets online\n\n"
            "🔬 **Workflows**\n"
            "• `create RNA-seq workflow` - Generate analysis pipeline\n"
            "• `show jobs` - Check running jobs\n\n"
            "❓ **Help**\n"
            "• `explain [concept]` - Learn about bioinformatics concepts\n"
            "• `help` - See all available commands\n\n"
            "What would you like to do?"
        )
        
        return RecoveryResponse(
            message=message,
            strategy=RecoveryStrategy.FALLBACK_RESPONSE,
            suggestions=[
                "Scan my data",
                "Search for datasets",
                "Show help",
            ],
            needs_user_input=True,
        )
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _detect_query_category(self, query: str) -> str:
        """Detect the category of a query for appropriate templates."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["data", "file", "path", "directory", "folder"]):
            return "data"
        
        if any(word in query_lower for word in ["workflow", "pipeline", "analysis", "run"]):
            return "workflow"
        
        return "general"
    
    def _generate_suggestions(self, query: str, category: str) -> List[str]:
        """Generate helpful suggestions based on query."""
        if category == "data":
            return [
                "Scan local data",
                "Search for datasets",
                "Specify a path",
            ]
        
        if category == "workflow":
            return [
                "Create RNA-seq workflow",
                "Create ChIP-seq workflow",
                "Show available workflows",
            ]
        
        return [
            "Search for data",
            "Create a workflow",
            "Show help",
        ]
    
    def _intent_to_description(self, intent: str) -> str:
        """Convert intent name to human-readable description."""
        descriptions = {
            "DATA_SCAN": "scan your local data",
            "DATA_SEARCH": "search for datasets online",
            "DATA_DOWNLOAD": "download a dataset",
            "DATA_VALIDATE": "validate your data",
            "WORKFLOW_CREATE": "create an analysis workflow",
            "WORKFLOW_GENERATE": "generate a workflow",
            "JOB_SUBMIT": "submit a job",
            "JOB_STATUS": "check job status",
            "JOB_LIST": "list your jobs",
            "DIAGNOSE_ERROR": "diagnose an error",
            "EDUCATION_EXPLAIN": "explain a concept",
            "EDUCATION_HELP": "show help",
        }
        
        # Handle various formats
        intent_key = intent.upper().replace("-", "_").replace(" ", "_")
        
        return descriptions.get(intent_key, intent.lower().replace("_", " "))
    
    def _get_service_name(self, tool_name: str) -> str:
        """Get friendly service name from tool name."""
        if not tool_name:
            return "the service"
        
        service_map = {
            "search_databases": "the genomics databases",
            "download_dataset": "the data server",
            "submit_job": "the HPC cluster",
            "get_job_status": "the job scheduler",
        }
        
        return service_map.get(tool_name, f"the {tool_name.replace('_', ' ')} service")
    
    def record_retry(self, query: str):
        """Record a retry attempt for a query."""
        self._retry_counts[query] = self._retry_counts.get(query, 0) + 1
    
    def clear_retry_count(self, query: str = None):
        """Clear retry count for a query or all queries."""
        if query:
            self._retry_counts.pop(query, None)
        else:
            self._retry_counts.clear()


# =============================================================================
# SINGLETON
# =============================================================================

_recovery: Optional[ConversationRecovery] = None


def get_conversation_recovery() -> ConversationRecovery:
    """Get the singleton recovery instance."""
    global _recovery
    if _recovery is None:
        _recovery = ConversationRecovery()
    return _recovery
