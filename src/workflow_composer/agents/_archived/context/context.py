"""
Conversation Context
====================

Manages conversation state for multi-turn dialogue with the AI agent.

This enables:
- Remembering what data was scanned
- Tracking current workflow being discussed
- Understanding pronouns ("it", "this", "that data")
- Follow-up questions without repeating context

Usage:
    context = ConversationContext()
    
    # After scanning data
    context.set_scanned_data(path="/data/samples", samples=sample_list)
    
    # User says "create a workflow for this data"
    # Context knows what "this data" refers to
    data_info = context.get_current_data()
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ConversationContext:
    """
    Tracks conversation state across multiple chat messages.
    
    Enables natural follow-up questions like:
    - "Create a workflow for this data" (references last scanned data)
    - "Run it on SLURM" (references last generated workflow)
    - "What's the status?" (references last submitted job)
    """
    
    # Last scanned data
    last_scan_path: Optional[str] = None
    last_scan_samples: List[Any] = field(default_factory=list)
    last_scan_time: Optional[datetime] = None
    
    # Current data manifest state
    manifest_sample_count: int = 0
    manifest_organism: Optional[str] = None
    manifest_assay_type: Optional[str] = None
    
    # Last generated workflow
    last_workflow_path: Optional[str] = None
    last_workflow_name: Optional[str] = None
    last_workflow_time: Optional[datetime] = None
    
    # Last submitted job
    last_job_id: Optional[str] = None
    last_job_workflow: Optional[str] = None
    last_job_status: Optional[str] = None
    last_job_time: Optional[datetime] = None
    
    # Search context
    last_search_query: Optional[str] = None
    last_search_results: List[Dict] = field(default_factory=list)
    
    # Reference context
    last_organism_checked: Optional[str] = None
    last_assembly_checked: Optional[str] = None
    
    # Conversation intent tracking
    pending_action: Optional[str] = None  # "create_workflow", "run_job", etc.
    clarification_needed: Optional[str] = None
    
    def set_scanned_data(self, path: str, samples: List[Any], organism: str = None, assay_type: str = None):
        """Record that data was scanned."""
        self.last_scan_path = path
        self.last_scan_samples = samples
        self.last_scan_time = datetime.now()
        self.manifest_sample_count = len(samples)
        if organism:
            self.manifest_organism = organism
        if assay_type:
            self.manifest_assay_type = assay_type
        logger.debug(f"Context updated: scanned {len(samples)} samples from {path}")
    
    def set_generated_workflow(self, path: str, name: str):
        """Record that a workflow was generated."""
        self.last_workflow_path = path
        self.last_workflow_name = name
        self.last_workflow_time = datetime.now()
        logger.debug(f"Context updated: generated workflow {name}")
    
    def set_submitted_job(self, job_id: str, workflow: str = None):
        """Record that a job was submitted."""
        self.last_job_id = job_id
        self.last_job_workflow = workflow or self.last_workflow_name
        self.last_job_status = "submitted"
        self.last_job_time = datetime.now()
        logger.debug(f"Context updated: submitted job {job_id}")
    
    def set_search_results(self, query: str, results: List[Dict]):
        """Record search results."""
        self.last_search_query = query
        self.last_search_results = results
        logger.debug(f"Context updated: search for '{query}' returned {len(results)} results")
    
    def get_current_data(self) -> Dict[str, Any]:
        """Get information about current data context."""
        return {
            "path": self.last_scan_path,
            "sample_count": self.manifest_sample_count,
            "organism": self.manifest_organism,
            "assay_type": self.manifest_assay_type,
            "samples": self.last_scan_samples,
        }
    
    def get_current_workflow(self) -> Dict[str, Any]:
        """Get information about current workflow context."""
        return {
            "path": self.last_workflow_path,
            "name": self.last_workflow_name,
            "created": self.last_workflow_time,
        }
    
    def get_current_job(self) -> Dict[str, Any]:
        """Get information about current job context."""
        return {
            "job_id": self.last_job_id,
            "workflow": self.last_job_workflow,
            "status": self.last_job_status,
            "submitted": self.last_job_time,
        }
    
    def resolve_pronoun(self, message: str) -> Dict[str, Any]:
        """
        Resolve pronouns in a message to actual context.
        
        Examples:
        - "run it" → references last workflow
        - "for this data" → references last scanned data
        - "check that job" → references last submitted job
        """
        message_lower = message.lower()
        resolved = {}
        
        # Data references
        if any(phrase in message_lower for phrase in ["this data", "that data", "the data", "these samples", "those samples"]):
            if self.last_scan_path:
                resolved["data_path"] = self.last_scan_path
                resolved["samples"] = self.last_scan_samples
        
        # Workflow references
        if any(phrase in message_lower for phrase in ["run it", "execute it", "this workflow", "the workflow", "that pipeline"]):
            if self.last_workflow_path:
                resolved["workflow_path"] = self.last_workflow_path
                resolved["workflow_name"] = self.last_workflow_name
        
        # Job references
        if any(phrase in message_lower for phrase in ["the job", "that job", "my job", "its status", "the status"]):
            if self.last_job_id:
                resolved["job_id"] = self.last_job_id
        
        return resolved
    
    def get_context_summary(self) -> str:
        """Get a summary of current context for LLM prompts."""
        parts = []
        
        if self.manifest_sample_count > 0:
            parts.append(f"Data: {self.manifest_sample_count} samples from {self.last_scan_path}")
            if self.manifest_organism:
                parts.append(f"Organism: {self.manifest_organism}")
        
        if self.last_workflow_name:
            parts.append(f"Last workflow: {self.last_workflow_name}")
        
        if self.last_job_id:
            parts.append(f"Active job: {self.last_job_id} ({self.last_job_status})")
        
        return "; ".join(parts) if parts else "No active context"
    
    def needs_clarification(self, message: str) -> Optional[str]:
        """
        Check if the message needs clarification.
        
        Returns a clarification question if needed, None otherwise.
        """
        message_lower = message.lower()
        
        # User says "create workflow" but no data scanned
        if any(kw in message_lower for kw in ["create workflow", "generate workflow", "build pipeline"]):
            if self.manifest_sample_count == 0:
                return "I don't have any data loaded yet. Would you like me to scan a directory for samples first? (e.g., 'scan data in /path/to/data')"
        
        # User says "run it" but no workflow generated
        if any(kw in message_lower for kw in ["run it", "execute it", "submit it"]):
            if not self.last_workflow_path:
                return "I don't have a workflow ready to run. Would you like me to generate one first?"
        
        # User says "check status" but no job submitted
        if any(kw in message_lower for kw in ["check status", "job status", "how's it going"]):
            if not self.last_job_id:
                return "I don't have any active jobs to check. Would you like to submit a workflow?"
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return {
            "last_scan_path": self.last_scan_path,
            "manifest_sample_count": self.manifest_sample_count,
            "manifest_organism": self.manifest_organism,
            "manifest_assay_type": self.manifest_assay_type,
            "last_workflow_path": self.last_workflow_path,
            "last_workflow_name": self.last_workflow_name,
            "last_job_id": self.last_job_id,
            "last_job_status": self.last_job_status,
            "last_search_query": self.last_search_query,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationContext":
        """Create context from dictionary."""
        ctx = cls()
        for key, value in data.items():
            if hasattr(ctx, key):
                setattr(ctx, key, value)
        return ctx
