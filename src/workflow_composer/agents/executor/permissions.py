"""
Permission Manager
===================

Control what the agent is allowed to do.

Features:
- Multiple autonomy levels
- Action-based permissions
- Human approval workflow
- Configurable policies

Example:
    pm = PermissionManager(autonomy_level=AutonomyLevel.ASSISTED)
    
    # Check if action is allowed
    if pm.can_execute("run_command"):
        result = sandbox.execute(cmd)
    else:
        # Need human approval
        approval = pm.request_approval(cmd)
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Set, Callable
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================

class PermissionLevel(Enum):
    """
    Autonomy levels for the agent.
    
    OBSERVE: Read-only, can only observe and report
    SUGGEST: Can suggest actions but not execute
    APPROVE: Can execute after human approval
    AUTONOMOUS: Full autonomy (with safety limits)
    """
    OBSERVE = "observe"
    SUGGEST = "suggest"
    APPROVE = "approve"
    AUTONOMOUS = "autonomous"


class AutonomyLevel(Enum):
    """
    User-facing autonomy levels (alias for PermissionLevel).
    
    READONLY: Read-only access
    MONITORED: Read + logged actions
    ASSISTED: Read + write with confirmation
    SUPERVISED: Most actions without confirmation
    AUTONOMOUS: Full autonomy
    """
    READONLY = "readonly"
    MONITORED = "monitored"
    ASSISTED = "assisted"
    SUPERVISED = "supervised"
    AUTONOMOUS = "autonomous"
    
    def to_permission_level(self) -> PermissionLevel:
        """Convert to internal PermissionLevel."""
        mapping = {
            AutonomyLevel.READONLY: PermissionLevel.OBSERVE,
            AutonomyLevel.MONITORED: PermissionLevel.OBSERVE,
            AutonomyLevel.ASSISTED: PermissionLevel.APPROVE,
            AutonomyLevel.SUPERVISED: PermissionLevel.APPROVE,
            AutonomyLevel.AUTONOMOUS: PermissionLevel.AUTONOMOUS,
        }
        return mapping[self]


class ActionCategory(Enum):
    """Categories of actions."""
    READ = "read"           # Read files, list directories
    WRITE = "write"         # Write/patch files
    DELETE = "delete"       # Delete files
    EXECUTE = "execute"     # Run commands
    PROCESS = "process"     # Start/stop processes
    NETWORK = "network"     # Network operations
    SYSTEM = "system"       # System administration
    RECOVERY = "recovery"   # Automated recovery


@dataclass
class ApprovalRequest:
    """A request for human approval."""
    id: str
    action: str
    description: str
    details: Dict[str, Any]
    timestamp: str
    status: str = "pending"  # pending, approved, denied, expired
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    expires_at: Optional[str] = None


@dataclass
class PermissionResult:
    """Result of a permission check."""
    allowed: bool
    requires_confirmation: bool
    reason: str


@dataclass
class PermissionPolicy:
    """A permission policy."""
    name: str
    level: PermissionLevel
    allowed_actions: Set[str]
    denied_actions: Set[str]
    require_approval: Set[str]
    max_retries: int = 3
    timeout_seconds: int = 300
    

# =============================================================================
# Default Policies
# =============================================================================

DEFAULT_POLICIES = {
    PermissionLevel.OBSERVE: PermissionPolicy(
        name="observe",
        level=PermissionLevel.OBSERVE,
        allowed_actions={
            "read_file", "list_directory", "get_job_status",
            "tail_log", "search", "scan_data",
        },
        denied_actions={"*"},  # Deny all by default
        require_approval=set(),
    ),
    
    PermissionLevel.SUGGEST: PermissionPolicy(
        name="suggest",
        level=PermissionLevel.SUGGEST,
        allowed_actions={
            "read_file", "list_directory", "get_job_status",
            "tail_log", "search", "scan_data",
            "run_command:safe",  # Safe subset of commands
        },
        denied_actions={"write_file", "patch_file", "delete_file", "run_command:unsafe"},
        require_approval={"write_file", "patch_file", "run_command"},
    ),
    
    PermissionLevel.APPROVE: PermissionPolicy(
        name="approve",
        level=PermissionLevel.APPROVE,
        allowed_actions={
            "read_file", "list_directory", "get_job_status",
            "tail_log", "search", "scan_data",
            "run_command:safe",
        },
        denied_actions={"delete_file", "run_command:dangerous"},
        require_approval={"write_file", "patch_file", "run_command:unsafe", "start_process"},
    ),
    
    PermissionLevel.AUTONOMOUS: PermissionPolicy(
        name="autonomous",
        level=PermissionLevel.AUTONOMOUS,
        allowed_actions={"*"},  # Allow all
        denied_actions={"run_command:dangerous", "delete_file:/"},
        require_approval={"delete_file"},  # Still require approval for deletes
        max_retries=5,
    ),
}


# =============================================================================
# Safe Command Categories
# =============================================================================

SAFE_COMMANDS = {
    "ls", "cat", "head", "tail", "grep", "find", "wc",
    "python", "pip", "conda",
    "squeue", "sinfo", "sacct",
    "git", "diff",
    "echo", "date", "pwd",
}

UNSAFE_COMMANDS = {
    "rm", "mv", "cp",  # These need approval
    "sbatch", "scancel",  # Job operations need approval
}

DANGEROUS_COMMANDS = {
    "rm -rf", "mkfs", "dd if=",
    "chmod 777", "chown -R",
}


# =============================================================================
# Permission Manager
# =============================================================================

class PermissionManager:
    """
    Manage permissions for agent actions.
    
    Controls what the agent can do at different autonomy levels.
    """
    
    def __init__(
        self,
        level: Optional[PermissionLevel] = None,
        autonomy_level: Optional[AutonomyLevel] = None,
        policy: Optional[PermissionPolicy] = None,
        approval_callback: Optional[Callable[[ApprovalRequest], bool]] = None,
        audit_logger: Optional["AuditLogger"] = None,
        workspace_root: Optional[Any] = None,  # Accept but don't use for compatibility
    ):
        """
        Initialize the permission manager.
        
        Args:
            level: The internal permission level
            autonomy_level: The user-facing autonomy level (converted to internal level)
            policy: Custom policy (default: use level's default policy)
            approval_callback: Function to request human approval
            audit_logger: Logger for audit trail
            workspace_root: Accepted for compatibility but not used
        """
        # Determine level from autonomy_level if provided
        if autonomy_level is not None:
            self.level = autonomy_level.to_permission_level()
            self._autonomy_level = autonomy_level
        elif level is not None:
            self.level = level
            self._autonomy_level = None
        else:
            self.level = PermissionLevel.APPROVE
            self._autonomy_level = AutonomyLevel.ASSISTED
            
        self.policy = policy or DEFAULT_POLICIES.get(self.level, DEFAULT_POLICIES[PermissionLevel.SUGGEST])
        self.approval_callback = approval_callback
        self.audit_logger = audit_logger
        
        # Pending approvals
        self._pending_approvals: Dict[str, ApprovalRequest] = {}
    
    @property
    def autonomy_level(self) -> AutonomyLevel:
        """Get the current autonomy level."""
        return self._autonomy_level or AutonomyLevel.ASSISTED
    
    @autonomy_level.setter
    def autonomy_level(self, level: AutonomyLevel):
        """Set the autonomy level."""
        self._autonomy_level = level
        self.level = level.to_permission_level()
        self.policy = DEFAULT_POLICIES.get(self.level, DEFAULT_POLICIES[PermissionLevel.SUGGEST])
        
    def can_execute(
        self,
        action: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Check if an action is allowed.
        
        Args:
            action: The action to check
            details: Additional details for the check
            
        Returns:
            True if action is allowed without approval
        """
        # Check if explicitly denied
        if self._is_denied(action, details):
            return False
            
        # Check if explicitly allowed
        if self._is_allowed(action, details):
            return True
            
        # Default: denied
        return False
    
    def check_permission(
        self,
        action: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> "PermissionResult":
        """
        Check permission for an action, returning detailed result.
        
        Args:
            action: The action to check
            details: Additional details
            
        Returns:
            PermissionResult with allowed, requires_confirmation, reason
        """
        # Check if denied
        if self._is_denied(action, details):
            return PermissionResult(
                allowed=False,
                requires_confirmation=False,
                reason=f"Action '{action}' is explicitly denied",
            )
        
        # Check if allowed
        is_allowed = self._is_allowed(action, details)
        
        # Check if needs approval
        needs_approval = False
        for pattern in self.policy.require_approval:
            if self._matches(action, pattern, details):
                needs_approval = True
                break
        
        if is_allowed and not needs_approval:
            return PermissionResult(
                allowed=True,
                requires_confirmation=False,
                reason="Action allowed",
            )
        elif is_allowed and needs_approval:
            return PermissionResult(
                allowed=True,
                requires_confirmation=True,
                reason=f"Action '{action}' requires confirmation",
            )
        else:
            return PermissionResult(
                allowed=False,
                requires_confirmation=False,
                reason=f"Action '{action}' not in allowed list",
            )
        
    def requires_approval(
        self,
        action: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Check if an action requires human approval.
        
        Args:
            action: The action to check
            details: Additional details
            
        Returns:
            True if approval is required
        """
        if self._is_denied(action, details):
            return False  # Can't approve denied actions
            
        # Check if in require_approval set
        for pattern in self.policy.require_approval:
            if self._matches(action, pattern, details):
                return True
                
        return False
        
    def request_approval(
        self,
        action: str,
        description: str,
        details: Optional[Dict[str, Any]] = None,
        timeout_seconds: int = 300,
    ) -> ApprovalRequest:
        """
        Request human approval for an action.
        
        Args:
            action: The action needing approval
            description: Human-readable description
            details: Action details
            timeout_seconds: How long approval is valid
            
        Returns:
            ApprovalRequest object
        """
        request_id = f"approval_{int(datetime.now().timestamp() * 1000)}"
        
        request = ApprovalRequest(
            id=request_id,
            action=action,
            description=description,
            details=details or {},
            timestamp=datetime.now().isoformat(),
            expires_at=(datetime.now().timestamp() + timeout_seconds).__str__(),
        )
        
        self._pending_approvals[request_id] = request
        
        logger.info(f"Approval requested: {request_id} - {description}")
        
        # If we have a callback, try to get approval
        if self.approval_callback:
            try:
                approved = self.approval_callback(request)
                if approved:
                    self.approve(request_id, "callback")
                else:
                    self.deny(request_id, "callback")
            except Exception as e:
                logger.error(f"Approval callback failed: {e}")
                
        return request
        
    def approve(self, request_id: str, approver: str = "human") -> bool:
        """Approve a pending request."""
        request = self._pending_approvals.get(request_id)
        if not request:
            return False
            
        request.status = "approved"
        request.approved_by = approver
        request.approved_at = datetime.now().isoformat()
        
        logger.info(f"Request {request_id} approved by {approver}")
        return True
        
    def deny(self, request_id: str, denier: str = "human") -> bool:
        """Deny a pending request."""
        request = self._pending_approvals.get(request_id)
        if not request:
            return False
            
        request.status = "denied"
        request.approved_by = denier
        request.approved_at = datetime.now().isoformat()
        
        logger.info(f"Request {request_id} denied by {denier}")
        return True
        
    def is_approved(self, request_id: str) -> bool:
        """Check if a request was approved."""
        request = self._pending_approvals.get(request_id)
        if not request:
            return False
        return request.status == "approved"
        
    def get_pending_approvals(self) -> List[ApprovalRequest]:
        """Get all pending approval requests."""
        return [
            r for r in self._pending_approvals.values()
            if r.status == "pending"
        ]
        
    def set_level(self, level: PermissionLevel):
        """Change the autonomy level."""
        self.level = level
        self.policy = DEFAULT_POLICIES.get(level, self.policy)
        logger.info(f"Permission level set to: {level.value}")
        
    def classify_command(self, command: str) -> str:
        """
        Classify a command as safe, unsafe, or dangerous.
        
        Returns:
            "safe", "unsafe", or "dangerous"
        """
        command_lower = command.lower().strip()
        
        # Check for dangerous patterns
        for pattern in DANGEROUS_COMMANDS:
            if pattern in command_lower:
                return "dangerous"
                
        # Get the base command
        parts = command_lower.split()
        if not parts:
            return "unsafe"
        base_cmd = parts[0]
        
        if base_cmd in SAFE_COMMANDS:
            return "safe"
        elif base_cmd in UNSAFE_COMMANDS:
            return "unsafe"
        else:
            return "unsafe"  # Unknown = unsafe
            
    def _is_allowed(
        self,
        action: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check if action is in allowed set."""
        for pattern in self.policy.allowed_actions:
            if self._matches(action, pattern, details):
                return True
        return False
        
    def _is_denied(
        self,
        action: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check if action is in denied set."""
        for pattern in self.policy.denied_actions:
            if self._matches(action, pattern, details):
                return True
        return False
        
    def _matches(
        self,
        action: str,
        pattern: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check if action matches a pattern."""
        if pattern == "*":
            return True
            
        # Check for command classification
        if ":" in pattern:
            base, modifier = pattern.split(":", 1)
            if action.startswith(base):
                # Check modifier
                if modifier == "safe":
                    cmd = details.get("command", "") if details else ""
                    return self.classify_command(cmd) == "safe"
                elif modifier == "unsafe":
                    cmd = details.get("command", "") if details else ""
                    return self.classify_command(cmd) == "unsafe"
                elif modifier == "dangerous":
                    cmd = details.get("command", "") if details else ""
                    return self.classify_command(cmd) == "dangerous"
                else:
                    # Path-based modifier
                    path = details.get("path", "") if details else ""
                    return modifier in path
                    
        # Simple match
        return action == pattern or action.startswith(pattern + ":")


# =============================================================================
# Convenience Functions
# =============================================================================

_default_manager: Optional[PermissionManager] = None

def get_permission_manager() -> PermissionManager:
    """Get the default permission manager."""
    global _default_manager
    if _default_manager is None:
        # Default to APPROVE level for safety
        _default_manager = PermissionManager(level=PermissionLevel.APPROVE)
    return _default_manager


def set_autonomy_level(level: PermissionLevel):
    """Set the autonomy level."""
    get_permission_manager().set_level(level)


def can_execute(action: str, details: Optional[Dict[str, Any]] = None) -> bool:
    """Check if an action is allowed."""
    return get_permission_manager().can_execute(action, details)


def requires_approval(action: str, details: Optional[Dict[str, Any]] = None) -> bool:
    """Check if an action requires approval."""
    return get_permission_manager().requires_approval(action, details)
