"""
Concise Memory Agent
====================

Implements the "clean slate" memory pattern from DeepCode.

Key principles:
1. Preserve: System prompt + initial query + completed summaries
2. Discard: Full conversation history, file contents, intermediate outputs
3. Compress: Convert completed steps to one-line summaries
4. Result: 70-80% token reduction in long-running workflows

This pattern is essential for multi-step agent workflows that would
otherwise exceed context limits.

References:
    - DeepCode: workflows/agents/memory_agent_concise.py
"""

import json
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .token_tracker import TokenTracker, TokenBudget, create_tracker_for_model

logger = logging.getLogger(__name__)


@dataclass
class CompletedStep:
    """
    Summary of a completed workflow step.
    
    After a step completes, we discard the full details and keep
    only this minimal summary.
    """
    step_num: int
    action: str
    summary: str  # One-line summary of what was accomplished
    timestamp: datetime = field(default_factory=datetime.now)
    tokens_saved: int = 0  # How many tokens we saved by compressing
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"✓ Step {self.step_num}: {self.summary}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_num": self.step_num,
            "action": self.action,
            "summary": self.summary,
            "timestamp": self.timestamp.isoformat(),
            "tokens_saved": self.tokens_saved,
            "metadata": self.metadata,
        }


@dataclass 
class ConciseState:
    """
    Minimal state preserved across clean slate resets.
    
    This is the core of the concise memory pattern: we only keep
    what's absolutely necessary for the agent to continue.
    """
    system_prompt: str
    initial_query: str
    completed_steps: List[CompletedStep] = field(default_factory=list)
    current_context: Dict[str, Any] = field(default_factory=dict)
    
    # Track compression statistics
    total_tokens_saved: int = 0
    compression_count: int = 0
    
    def get_context_summary(self) -> str:
        """Get formatted summary for prompt injection."""
        parts = []
        
        if self.completed_steps:
            parts.append("## Completed Steps")
            for step in self.completed_steps:
                parts.append(str(step))
        
        if self.current_context:
            parts.append("")
            parts.append("## Current Context")
            for key, value in self.current_context.items():
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, default=str)[:100]
                parts.append(f"- {key}: {value}")
        
        return "\n".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_prompt_length": len(self.system_prompt),
            "initial_query": self.initial_query[:200],
            "completed_steps": [s.to_dict() for s in self.completed_steps],
            "current_context": self.current_context,
            "total_tokens_saved": self.total_tokens_saved,
            "compression_count": self.compression_count,
        }


class ConciseMemory:
    """
    Memory manager with aggressive compression.
    
    After each major action (file write, tool completion), triggers
    compression to prevent context overflow. This implements the
    "clean slate" pattern from DeepCode.
    
    Key Insight:
        After completing a step, we don't need the full conversation
        that led to it. We only need:
        1. The system prompt (instructions)
        2. The original query (what we're trying to do)
        3. Summaries of what we've done (not the details)
        
    This typically reduces token usage by 70-80%.
    
    Examples:
        >>> memory = ConciseMemory(system_prompt=SYSTEM_PROMPT)
        >>> memory.set_initial_query(user_query)
        >>> 
        >>> # During workflow execution...
        >>> memory.add_working_message("user", "Align these samples")
        >>> memory.add_working_message("assistant", "I'll run STAR alignment...")
        >>> memory.add_tool_result("Aligned 48 samples, 96% mapping rate")
        >>> 
        >>> # After completing the alignment step
        >>> memory.complete_step(
        ...     step_num=1,
        ...     action="STAR_ALIGN",
        ...     summary="Aligned 48 samples with STAR (96% mapped)"
        ... )
        >>> # Working context is now cleared, only summary remains
        >>> 
        >>> # Get compressed context for next LLM call
        >>> messages = memory.get_prompt_context()
    """
    
    def __init__(
        self,
        system_prompt: str,
        token_budget: Optional[TokenBudget] = None,
        model_name: str = "gpt-4",
        auto_compress: bool = True,
        summarizer: Optional[Callable[[List[Dict]], str]] = None,
    ):
        """
        Initialize concise memory manager.
        
        Args:
            system_prompt: System prompt (preserved across resets)
            token_budget: Token budget configuration
            model_name: Model name for token counting
            auto_compress: Automatically compress when budget exceeded
            summarizer: Optional function to summarize conversations
        """
        self.state = ConciseState(
            system_prompt=system_prompt,
            initial_query="",
        )
        
        # Initialize token tracker
        if token_budget:
            self.token_tracker = TokenTracker(budget=token_budget, model_name=model_name)
        else:
            self.token_tracker = create_tracker_for_model(model_name)
        
        self.auto_compress = auto_compress
        self.summarizer = summarizer
        
        # Working context (will be compressed after each step)
        self._working_context: List[Dict[str, str]] = []
        
        # Full history for debugging (not sent to LLM)
        self._full_history: List[Dict[str, Any]] = []
        
        # Checkpoints for recovery
        self._checkpoints: List[Dict[str, Any]] = []
        
        # Track system prompt tokens
        self.token_tracker.add_system_prompt(system_prompt)
    
    def set_initial_query(self, query: str):
        """
        Set the initial user query (preserved across resets).
        
        This is the "what we're trying to accomplish" that never changes.
        """
        self.state.initial_query = query
        
        # Log to history
        self._full_history.append({
            "type": "initial_query",
            "content": query,
            "timestamp": datetime.now().isoformat(),
        })
    
    def add_working_message(self, role: str, content: str):
        """
        Add a message to working context.
        
        Working messages may be compressed after step completion.
        """
        self._working_context.append({
            "role": role,
            "content": content,
        })
        
        # Track in full history
        self._full_history.append({
            "type": "message",
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Update token tracking
        self.token_tracker.add_message(role, content)
        
        # Check if compression needed
        if self.auto_compress and self.token_tracker.should_compress():
            logger.info(
                f"Token budget {self.token_tracker.get_usage_percentage():.1f}% used, "
                f"triggering compression"
            )
            self._compress()
    
    def add_tool_result(self, result: str, tool_name: str = "tool"):
        """
        Add tool execution result to working context.
        
        Tool results are often large (file listings, error logs) and
        are prime candidates for compression.
        """
        # Add as assistant message with tool indicator
        message = f"[{tool_name}] {result}"
        self.add_working_message("assistant", message)
        
        # Separately track tool tokens
        self.token_tracker.add_tool_result(result)
    
    def complete_step(
        self,
        step_num: int,
        action: str,
        summary: str,
        context_updates: Optional[Dict[str, Any]] = None,
    ):
        """
        Mark a step as complete and apply clean slate.
        
        This is the key operation: after completing a step, we discard
        the detailed working context and keep only the summary.
        
        Args:
            step_num: Step number for tracking
            action: What action was performed
            summary: One-line summary of the outcome
            context_updates: Any state to preserve (e.g., file paths created)
        """
        # Calculate tokens saved
        tokens_before = self.token_tracker.usage.total
        
        # Create completion record
        completed = CompletedStep(
            step_num=step_num,
            action=action,
            summary=summary,
        )
        self.state.completed_steps.append(completed)
        
        # Update persistent context
        if context_updates:
            self.state.current_context.update(context_updates)
        
        # CLEAN SLATE: Discard working context
        working_tokens = sum(
            self.token_tracker.count_tokens(m.get("content", ""))
            for m in self._working_context
        )
        self._working_context = []
        
        # Reset token tracker (keep only system prompt)
        self.token_tracker = create_tracker_for_model(
            self.token_tracker.model_name,
            compression_trigger=self.token_tracker.budget.compression_trigger,
        )
        self.token_tracker.budget = TokenBudget(
            max_context=self.token_tracker.budget.max_context,
        )
        self.token_tracker.add_system_prompt(self.state.system_prompt)
        
        # Track tokens saved
        tokens_saved = working_tokens
        completed.tokens_saved = tokens_saved
        self.state.total_tokens_saved += tokens_saved
        self.state.compression_count += 1
        
        # Log to history
        self._full_history.append({
            "type": "step_complete",
            "step_num": step_num,
            "action": action,
            "summary": summary,
            "tokens_saved": tokens_saved,
            "timestamp": datetime.now().isoformat(),
        })
        
        logger.info(
            f"Completed step {step_num} ({action}), "
            f"saved {tokens_saved} tokens via clean slate"
        )
    
    def _compress(self):
        """
        Compress working context to reduce token usage.
        
        Called automatically when token budget is exceeded.
        Creates a summary of the working context if a summarizer
        is available, otherwise just truncates old messages.
        """
        if not self._working_context:
            return
        
        tokens_before = self.token_tracker.usage.total
        
        if self.summarizer:
            # Use provided summarizer function
            try:
                summary = self.summarizer(self._working_context)
                # Replace working context with summary
                self._working_context = [{
                    "role": "system",
                    "content": f"[Previous context summary: {summary}]"
                }]
            except Exception as e:
                logger.warning(f"Summarizer failed: {e}, using truncation")
                # Fallback to truncation
                self._truncate_context()
        else:
            # Simple truncation: keep only recent messages
            self._truncate_context()
        
        # Recalculate token usage
        self.token_tracker.reset_history()
        self.token_tracker.reset_tool_results()
        for msg in self._working_context:
            self.token_tracker.add_message(
                msg.get("role", "user"),
                msg.get("content", "")
            )
        
        tokens_after = self.token_tracker.usage.total
        tokens_saved = tokens_before - tokens_after
        self.state.total_tokens_saved += tokens_saved
        
        # Log compression event
        self._full_history.append({
            "type": "compression",
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "tokens_saved": tokens_saved,
            "timestamp": datetime.now().isoformat(),
        })
        
        logger.info(f"Compressed context, saved {tokens_saved} tokens")
    
    def _truncate_context(self, keep_recent: int = 4):
        """Truncate working context to recent messages."""
        if len(self._working_context) > keep_recent:
            removed = len(self._working_context) - keep_recent
            self._working_context = self._working_context[-keep_recent:]
            logger.debug(f"Truncated {removed} old messages from context")
    
    def get_prompt_context(self) -> List[Dict[str, str]]:
        """
        Get messages for LLM prompt.
        
        Returns minimal context:
        1. System prompt
        2. Completed steps summary (if any)
        3. Initial query
        4. Recent working context
        """
        messages = [
            {"role": "system", "content": self.state.system_prompt}
        ]
        
        # Add completed steps summary if any
        if self.state.completed_steps:
            context_summary = self.state.get_context_summary()
            messages.append({
                "role": "system",
                "content": f"Progress so far:\n{context_summary}"
            })
        
        # Add initial query
        if self.state.initial_query:
            messages.append({
                "role": "user",
                "content": self.state.initial_query
            })
        
        # Add working context
        messages.extend(self._working_context)
        
        return messages
    
    def get_token_status(self) -> Dict[str, Any]:
        """Get current token usage status."""
        return {
            **self.token_tracker.get_status(),
            "total_tokens_saved": self.state.total_tokens_saved,
            "compression_count": self.state.compression_count,
            "completed_steps": len(self.state.completed_steps),
        }
    
    def get_full_history(self) -> List[Dict[str, Any]]:
        """Get full history for debugging (not sent to LLM)."""
        return self._full_history.copy()
    
    def save_checkpoint(self, name: str = "auto") -> Dict[str, Any]:
        """
        Save state for recovery.
        
        Checkpoints allow recovering from failures without losing
        all progress.
        """
        checkpoint = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "state": self.state.to_dict(),
            "working_context": self._working_context.copy(),
        }
        self._checkpoints.append(checkpoint)
        
        logger.info(f"Saved checkpoint '{name}' with {len(self.state.completed_steps)} steps")
        return checkpoint
    
    def restore_checkpoint(self, checkpoint: Dict[str, Any]):
        """
        Restore state from checkpoint.
        
        Use after a failure to resume from a known good state.
        """
        state_dict = checkpoint.get("state", {})
        
        # Restore completed steps
        self.state.completed_steps = [
            CompletedStep(
                step_num=s["step_num"],
                action=s["action"],
                summary=s["summary"],
                timestamp=datetime.fromisoformat(s["timestamp"]),
                tokens_saved=s.get("tokens_saved", 0),
                metadata=s.get("metadata", {}),
            )
            for s in state_dict.get("completed_steps", [])
        ]
        
        self.state.current_context = state_dict.get("current_context", {})
        self.state.total_tokens_saved = state_dict.get("total_tokens_saved", 0)
        self.state.compression_count = state_dict.get("compression_count", 0)
        
        # Restore working context
        self._working_context = checkpoint.get("working_context", [])
        
        logger.info(f"Restored checkpoint with {len(self.state.completed_steps)} steps")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory management statistics."""
        return {
            "completed_steps": len(self.state.completed_steps),
            "working_context_size": len(self._working_context),
            "total_tokens_saved": self.state.total_tokens_saved,
            "compression_count": self.state.compression_count,
            "current_token_usage": self.token_tracker.usage.total,
            "token_budget_remaining": self.token_tracker.get_remaining_budget(),
            "checkpoints_saved": len(self._checkpoints),
        }
    
    def clear(self):
        """Clear all state (for reuse with new query)."""
        self.state = ConciseState(
            system_prompt=self.state.system_prompt,
            initial_query="",
        )
        self._working_context = []
        self._full_history = []
        self._checkpoints = []
        
        # Reset token tracker
        self.token_tracker = create_tracker_for_model(
            self.token_tracker.model_name,
        )
        self.token_tracker.add_system_prompt(self.state.system_prompt)


def create_concise_memory(
    system_prompt: str,
    model_name: str = "gpt-4",
    compression_trigger: float = 0.75,
) -> ConciseMemory:
    """
    Factory function to create concise memory with defaults.
    
    Args:
        system_prompt: System prompt for the agent
        model_name: Model name (for token counting and budget)
        compression_trigger: When to trigger compression (0.0-1.0)
        
    Returns:
        Configured ConciseMemory instance
    """
    budget = TokenBudget(compression_trigger=compression_trigger)
    
    return ConciseMemory(
        system_prompt=system_prompt,
        token_budget=budget,
        model_name=model_name,
        auto_compress=True,
    )
