"""
File Operations
================

Safe file operations with automatic backup and validation.

Features:
- Path validation (no escape from workspace)
- Automatic backup before modifications
- Atomic writes
- Diff generation
- Size limits

Example:
    file_ops = FileOperations(workspace=Path("/project"))
    
    # Read file
    content = file_ops.read_file("config.yaml")
    print(content.text)
    
    # Write with backup
    result = file_ops.write_file("config.yaml", new_content)
    print(f"Backup at: {result.backup_path}")
    
    # Targeted patch
    result = file_ops.patch_file(
        "script.py",
        old="param = 10",
        new="param = 20"
    )
"""

import os
import re
import shutil
import difflib
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================

@dataclass
class FileContent:
    """Content of a file read operation."""
    path: str
    text: str
    lines: List[str]
    size_bytes: int
    line_count: int
    truncated: bool = False
    encoding: str = "utf-8"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "text": self.text,
            "size_bytes": self.size_bytes,
            "line_count": self.line_count,
            "truncated": self.truncated,
        }


@dataclass
class WriteResult:
    """Result of a file write operation."""
    success: bool
    path: str
    backup_path: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    bytes_written: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "path": self.path,
            "backup_path": self.backup_path,
            "message": self.message,
            "error": self.error,
            "bytes_written": self.bytes_written,
        }


@dataclass
class PatchResult:
    """Result of a file patch operation."""
    success: bool
    path: str
    backup_path: Optional[str] = None
    diff: str = ""
    matches_found: int = 0
    matches_replaced: int = 0
    message: str = ""
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "path": self.path,
            "backup_path": self.backup_path,
            "diff": self.diff,
            "matches_found": self.matches_found,
            "matches_replaced": self.matches_replaced,
            "message": self.message,
            "error": self.error,
        }


@dataclass
class DirectoryListing:
    """Result of a directory listing."""
    path: str
    entries: List[Dict[str, Any]]
    total_files: int
    total_dirs: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "entries": self.entries,
            "total_files": self.total_files,
            "total_dirs": self.total_dirs,
        }


class PathValidationError(Exception):
    """Raised when a path fails validation."""
    pass


# =============================================================================
# File Operations
# =============================================================================

class FileOperations:
    """
    Safe file operations with backup and validation.
    
    All operations are confined to the workspace directory.
    Modifications are backed up automatically.
    """
    
    # Maximum file size to read (10MB)
    MAX_READ_SIZE = 10 * 1024 * 1024
    
    # Maximum lines to read
    MAX_READ_LINES = 50000
    
    # Backup directory name
    BACKUP_DIR = ".agent_backups"
    
    # Files that should never be modified
    PROTECTED_PATTERNS = [
        r"\.git/",
        r"\.ssh/",
        r"id_rsa",
        r"\.env$",
        r"\.secrets",
        r"password",
        r"credentials",
    ]
    
    def __init__(
        self,
        workspace: Optional[Path] = None,
        backup_dir: Optional[Path] = None,
        max_read_size: int = MAX_READ_SIZE,
        max_read_lines: int = MAX_READ_LINES,
        audit_logger: Optional["AuditLogger"] = None,
    ):
        """
        Initialize file operations.
        
        Args:
            workspace: Root directory for all operations
            backup_dir: Directory for backups (default: workspace/.agent_backups)
            max_read_size: Maximum file size to read
            max_read_lines: Maximum lines to read
            audit_logger: Logger for audit trail
        """
        self.workspace = Path(workspace or os.getcwd()).resolve()
        self.backup_dir = Path(backup_dir) if backup_dir else self.workspace / self.BACKUP_DIR
        self.max_read_size = max_read_size
        self.max_read_lines = max_read_lines
        self.audit_logger = audit_logger
        
        # Compile protected patterns
        self._protected_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.PROTECTED_PATTERNS
        ]
        
    def _validate_path(self, path: Union[str, Path]) -> Path:
        """
        Validate that a path is within the workspace.
        
        Args:
            path: The path to validate
            
        Returns:
            Resolved absolute Path
            
        Raises:
            PathValidationError: If path is outside workspace or protected
        """
        # Convert to Path and resolve
        if isinstance(path, str):
            # Handle relative paths
            if not os.path.isabs(path):
                path = self.workspace / path
            path = Path(path)
            
        resolved = path.resolve()
        
        # Check if within workspace
        try:
            resolved.relative_to(self.workspace)
        except ValueError:
            raise PathValidationError(
                f"Path '{resolved}' is outside workspace '{self.workspace}'"
            )
            
        # Check protected patterns
        path_str = str(resolved)
        for pattern in self._protected_patterns:
            if pattern.search(path_str):
                raise PathValidationError(
                    f"Path '{resolved}' matches protected pattern"
                )
                
        return resolved
        
    def _create_backup(self, path: Path) -> Optional[Path]:
        """
        Create a backup of a file.
        
        Args:
            path: Path to the file to backup
            
        Returns:
            Path to the backup file, or None if file doesn't exist
        """
        if not path.exists():
            return None
            
        # Create backup directory
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate backup filename with timestamp and hash
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_hash = hashlib.md5(str(path).encode()).hexdigest()[:8]
        backup_name = f"{path.name}.{timestamp}.{file_hash}.bak"
        backup_path = self.backup_dir / backup_name
        
        # Copy file
        shutil.copy2(path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        
        return backup_path
        
    def read_file(
        self,
        path: Union[str, Path],
        max_lines: Optional[int] = None,
        encoding: str = "utf-8",
    ) -> FileContent:
        """
        Read a file with size limits.
        
        Args:
            path: Path to the file
            max_lines: Maximum lines to read (default: MAX_READ_LINES)
            encoding: File encoding
            
        Returns:
            FileContent with the file contents
        """
        resolved_path = self._validate_path(path)
        max_lines = max_lines or self.max_read_lines
        
        if not resolved_path.exists():
            raise FileNotFoundError(f"File not found: {resolved_path}")
            
        if not resolved_path.is_file():
            raise ValueError(f"Not a file: {resolved_path}")
            
        # Check file size
        size = resolved_path.stat().st_size
        if size > self.max_read_size:
            raise ValueError(
                f"File too large ({size} bytes, max {self.max_read_size})"
            )
            
        # Read file
        try:
            with open(resolved_path, "r", encoding=encoding) as f:
                lines = []
                truncated = False
                for i, line in enumerate(f):
                    if i >= max_lines:
                        truncated = True
                        break
                    lines.append(line)
                    
                text = "".join(lines)
                
        except UnicodeDecodeError:
            # Try binary read and decode with errors='replace'
            with open(resolved_path, "rb") as f:
                raw = f.read(self.max_read_size)
                text = raw.decode(encoding, errors="replace")
                lines = text.splitlines(keepends=True)
                truncated = len(lines) > max_lines
                if truncated:
                    lines = lines[:max_lines]
                    text = "".join(lines)
                    
        return FileContent(
            path=str(resolved_path),
            text=text,
            lines=lines,
            size_bytes=size,
            line_count=len(lines),
            truncated=truncated,
            encoding=encoding,
        )
        
    def write_file(
        self,
        path: Union[str, Path],
        content: str,
        backup: bool = True,
        create_dirs: bool = True,
        encoding: str = "utf-8",
    ) -> WriteResult:
        """
        Write content to a file with automatic backup.
        
        Args:
            path: Path to the file
            content: Content to write
            backup: Whether to create a backup first
            create_dirs: Whether to create parent directories
            encoding: File encoding
            
        Returns:
            WriteResult with operation details
        """
        try:
            resolved_path = self._validate_path(path)
        except PathValidationError as e:
            return WriteResult(
                success=False,
                path=str(path),
                error=str(e),
                message=f"Path validation failed: {e}",
            )
            
        backup_path = None
        
        try:
            # Create parent directories if needed
            if create_dirs:
                resolved_path.parent.mkdir(parents=True, exist_ok=True)
                
            # Create backup if file exists
            if backup and resolved_path.exists():
                backup_path = self._create_backup(resolved_path)
                
            # Write file atomically (write to temp, then rename)
            temp_path = resolved_path.with_suffix(resolved_path.suffix + ".tmp")
            with open(temp_path, "w", encoding=encoding) as f:
                f.write(content)
                
            # Rename to final path
            temp_path.rename(resolved_path)
            
            bytes_written = len(content.encode(encoding))
            
            logger.info(f"Wrote {bytes_written} bytes to {resolved_path}")
            
            # Audit log
            if self.audit_logger:
                self.audit_logger.log_file_change(
                    str(resolved_path), "write", str(backup_path)
                )
                
            return WriteResult(
                success=True,
                path=str(resolved_path),
                backup_path=str(backup_path) if backup_path else None,
                message=f"Successfully wrote {bytes_written} bytes",
                bytes_written=bytes_written,
            )
            
        except Exception as e:
            logger.error(f"Failed to write {path}: {e}")
            return WriteResult(
                success=False,
                path=str(path),
                backup_path=str(backup_path) if backup_path else None,
                error=str(e),
                message=f"Write failed: {e}",
            )
            
    def patch_file(
        self,
        path: Union[str, Path],
        old: str,
        new: str,
        backup: bool = True,
        count: int = 1,
        encoding: str = "utf-8",
    ) -> PatchResult:
        """
        Apply a targeted patch to a file.
        
        Replaces exact occurrences of 'old' with 'new'.
        
        Args:
            path: Path to the file
            old: Text to find (exact match)
            new: Text to replace with
            backup: Whether to create a backup first
            count: Maximum number of replacements (0 = all)
            encoding: File encoding
            
        Returns:
            PatchResult with operation details
        """
        try:
            resolved_path = self._validate_path(path)
        except PathValidationError as e:
            return PatchResult(
                success=False,
                path=str(path),
                error=str(e),
                message=f"Path validation failed: {e}",
            )
            
        if not resolved_path.exists():
            return PatchResult(
                success=False,
                path=str(path),
                error="File not found",
                message=f"File does not exist: {path}",
            )
            
        try:
            # Read current content
            with open(resolved_path, "r", encoding=encoding) as f:
                original_content = f.read()
                
            # Count matches
            matches_found = original_content.count(old)
            
            if matches_found == 0:
                return PatchResult(
                    success=False,
                    path=str(resolved_path),
                    matches_found=0,
                    matches_replaced=0,
                    error="Pattern not found",
                    message=f"Could not find the text to replace",
                )
                
            # Create backup
            backup_path = None
            if backup:
                backup_path = self._create_backup(resolved_path)
                
            # Apply patch
            if count == 0:
                new_content = original_content.replace(old, new)
                matches_replaced = matches_found
            else:
                new_content = original_content.replace(old, new, count)
                matches_replaced = min(count, matches_found)
                
            # Generate diff
            diff = self._generate_diff(
                original_content, 
                new_content, 
                str(resolved_path)
            )
            
            # Write patched content
            with open(resolved_path, "w", encoding=encoding) as f:
                f.write(new_content)
                
            logger.info(
                f"Patched {resolved_path}: {matches_replaced} replacement(s)"
            )
            
            # Audit log
            if self.audit_logger:
                self.audit_logger.log_file_change(
                    str(resolved_path), "patch", str(backup_path)
                )
                
            return PatchResult(
                success=True,
                path=str(resolved_path),
                backup_path=str(backup_path) if backup_path else None,
                diff=diff,
                matches_found=matches_found,
                matches_replaced=matches_replaced,
                message=f"Successfully replaced {matches_replaced} occurrence(s)",
            )
            
        except Exception as e:
            logger.error(f"Failed to patch {path}: {e}")
            return PatchResult(
                success=False,
                path=str(path),
                error=str(e),
                message=f"Patch failed: {e}",
            )
            
    def apply_diff(
        self,
        path: Union[str, Path],
        diff_text: str,
        backup: bool = True,
    ) -> PatchResult:
        """
        Apply a unified diff to a file.
        
        Args:
            path: Path to the file
            diff_text: Unified diff text
            backup: Whether to create a backup first
            
        Returns:
            PatchResult with operation details
        """
        # This is more complex and requires parsing unified diff format
        # For now, we recommend using patch_file for simple changes
        return PatchResult(
            success=False,
            path=str(path),
            error="Not implemented",
            message="Use patch_file() for targeted replacements",
        )
        
    def list_directory(
        self,
        path: Union[str, Path] = ".",
        pattern: str = "*",
        recursive: bool = False,
        max_entries: int = 1000,
    ) -> DirectoryListing:
        """
        List contents of a directory.
        
        Args:
            path: Directory path
            pattern: Glob pattern to filter
            recursive: Whether to list recursively
            max_entries: Maximum entries to return
            
        Returns:
            DirectoryListing with entries
        """
        resolved_path = self._validate_path(path)
        
        if not resolved_path.exists():
            raise FileNotFoundError(f"Directory not found: {resolved_path}")
            
        if not resolved_path.is_dir():
            raise ValueError(f"Not a directory: {resolved_path}")
            
        entries = []
        total_files = 0
        total_dirs = 0
        
        if recursive:
            glob_iter = resolved_path.rglob(pattern)
        else:
            glob_iter = resolved_path.glob(pattern)
            
        for i, entry_path in enumerate(glob_iter):
            if i >= max_entries:
                break
                
            is_dir = entry_path.is_dir()
            if is_dir:
                total_dirs += 1
            else:
                total_files += 1
                
            try:
                stat = entry_path.stat()
                entries.append({
                    "name": entry_path.name,
                    "path": str(entry_path.relative_to(self.workspace)),
                    "is_dir": is_dir,
                    "size": stat.st_size if not is_dir else 0,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
            except (OSError, PermissionError):
                entries.append({
                    "name": entry_path.name,
                    "path": str(entry_path.relative_to(self.workspace)),
                    "is_dir": is_dir,
                    "size": 0,
                    "modified": None,
                    "error": "Permission denied",
                })
                
        return DirectoryListing(
            path=str(resolved_path),
            entries=entries,
            total_files=total_files,
            total_dirs=total_dirs,
        )
        
    def file_exists(self, path: Union[str, Path]) -> bool:
        """Check if a file exists."""
        try:
            resolved = self._validate_path(path)
            return resolved.exists()
        except (PathValidationError, Exception):
            return False
            
    def delete_file(
        self,
        path: Union[str, Path],
        backup: bool = True,
    ) -> WriteResult:
        """
        Delete a file (move to backup directory).
        
        Args:
            path: Path to the file
            backup: Whether to keep a backup (default: True)
            
        Returns:
            WriteResult with operation details
        """
        try:
            resolved_path = self._validate_path(path)
        except PathValidationError as e:
            return WriteResult(
                success=False,
                path=str(path),
                error=str(e),
            )
            
        if not resolved_path.exists():
            return WriteResult(
                success=False,
                path=str(path),
                error="File not found",
            )
            
        backup_path = None
        try:
            if backup:
                backup_path = self._create_backup(resolved_path)
                
            resolved_path.unlink()
            
            return WriteResult(
                success=True,
                path=str(resolved_path),
                backup_path=str(backup_path) if backup_path else None,
                message="File deleted successfully",
            )
            
        except Exception as e:
            return WriteResult(
                success=False,
                path=str(path),
                error=str(e),
            )
            
    def restore_backup(self, backup_path: Union[str, Path]) -> WriteResult:
        """
        Restore a file from backup.
        
        Args:
            backup_path: Path to the backup file
            
        Returns:
            WriteResult with operation details
        """
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            return WriteResult(
                success=False,
                path=str(backup_path),
                error="Backup not found",
            )
            
        # Parse original filename from backup name
        # Format: filename.ext.TIMESTAMP.HASH.bak
        name = backup_path.name
        if not name.endswith(".bak"):
            return WriteResult(
                success=False,
                path=str(backup_path),
                error="Invalid backup file format",
            )
            
        # Remove .TIMESTAMP.HASH.bak
        parts = name.rsplit(".", 4)
        if len(parts) >= 4:
            original_name = ".".join(parts[:-3])
        else:
            original_name = parts[0]
            
        # Restore to workspace
        original_path = self.workspace / original_name
        
        try:
            shutil.copy2(backup_path, original_path)
            return WriteResult(
                success=True,
                path=str(original_path),
                backup_path=str(backup_path),
                message=f"Restored from backup",
            )
        except Exception as e:
            return WriteResult(
                success=False,
                path=str(original_path),
                error=str(e),
            )
            
    def _generate_diff(
        self,
        old_content: str,
        new_content: str,
        filename: str,
    ) -> str:
        """Generate unified diff between old and new content."""
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
        )
        
        return "".join(diff)


# =============================================================================
# Convenience Functions
# =============================================================================

_default_file_ops: Optional[FileOperations] = None

def get_file_ops(workspace: Optional[Path] = None) -> FileOperations:
    """Get the default file operations instance."""
    global _default_file_ops
    if _default_file_ops is None or (workspace and _default_file_ops.workspace != workspace):
        _default_file_ops = FileOperations(workspace=workspace)
    return _default_file_ops


def read_file(path: str, max_lines: int = 1000) -> FileContent:
    """Read a file using default file operations."""
    return get_file_ops().read_file(path, max_lines=max_lines)


def write_file(path: str, content: str) -> WriteResult:
    """Write a file using default file operations."""
    return get_file_ops().write_file(path, content)


def patch_file(path: str, old: str, new: str) -> PatchResult:
    """Patch a file using default file operations."""
    return get_file_ops().patch_file(path, old, new)
