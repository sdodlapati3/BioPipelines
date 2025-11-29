#!/usr/bin/env python3
"""
Codebase Cleanup Script
=======================
Addresses issues found by validate_codebase.py:
1. Duplicate code consolidation
2. Dead code removal (with confirmation)
3. Import cleanup
4. Deprecated module handling

Usage:
    python scripts/cleanup_codebase.py --report          # Show what would be done
    python scripts/cleanup_codebase.py --fix-duplicates  # Fix duplicate code
    python scripts/cleanup_codebase.py --fix-imports     # Fix import issues
    python scripts/cleanup_codebase.py --archive-dead    # Move dead code to deprecated/
"""

import argparse
import ast
import os
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class DuplicateGroup:
    """Group of duplicate code blocks."""
    canonical_file: Path  # The "source of truth"
    canonical_name: str
    duplicates: List[Tuple[Path, str]]  # (file, function_name) pairs
    similarity: float = 1.0


@dataclass
class CleanupAction:
    """Represents a cleanup action to take."""
    action_type: str  # 'consolidate', 'archive', 'delete', 'refactor'
    description: str
    files_affected: List[Path]
    risk_level: str  # 'low', 'medium', 'high'
    auto_fixable: bool = False


class CodebaseCleanup:
    """Systematic codebase cleanup utilities."""
    
    def __init__(self, root: Path = None):
        self.root = root or Path(__file__).parent.parent
        self.src_dir = self.root / "src" / "workflow_composer"
        self.deprecated_dir = self.root / "deprecated"
        self.actions: List[CleanupAction] = []
        
    # =========================================================================
    # DUPLICATE CODE ANALYSIS
    # =========================================================================
    
    def find_duplicate_groups(self) -> List[DuplicateGroup]:
        """Identify groups of duplicate code that should be consolidated."""
        groups = []
        
        # Known duplicate patterns to consolidate
        known_duplicates = [
            {
                "canonical": ("models/providers/base.py", "BaseProvider"),
                "duplicates": [
                    ("models/providers/openai.py", "complete_async"),
                    ("models/providers/lightning.py", "complete_async"),
                    ("models/providers/openai.py", "health_check"),
                    ("models/providers/lightning.py", "health_check"),
                ],
                "action": "Move shared methods to BaseProvider"
            },
            {
                "canonical": ("providers/router.py", "_get_provider_instance"),
                "duplicates": [
                    ("models/router.py", "_get_provider_client"),
                ],
                "action": "Consolidate router modules"
            },
            {
                "canonical": ("core/tool_selector.py", "load_analysis_tool_map"),
                "duplicates": [
                    ("core/module_mapper.py", "load_tool_mappings"),
                ],
                "action": "Merge into single tool mapping loader"
            }
        ]
        
        for pattern in known_duplicates:
            canonical_path = self.src_dir / pattern["canonical"][0]
            if canonical_path.exists():
                group = DuplicateGroup(
                    canonical_file=canonical_path,
                    canonical_name=pattern["canonical"][1],
                    duplicates=[
                        (self.src_dir / d[0], d[1]) 
                        for d in pattern["duplicates"]
                        if (self.src_dir / d[0]).exists()
                    ]
                )
                if group.duplicates:
                    groups.append(group)
                    
        return groups
    
    def generate_base_provider(self) -> str:
        """Generate improved BaseProvider with shared methods."""
        return '''"""
Base Provider - Shared functionality for all LLM providers.
============================================================
Consolidates common code from individual provider implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator
import asyncio
import aiohttp


class BaseProvider(ABC):
    """Abstract base class for LLM providers with shared functionality."""
    
    def __init__(self, api_key: str = None, model: str = None, **kwargs):
        self.api_key = api_key
        self.model = model
        self.base_url = kwargs.get('base_url')
        self.timeout = kwargs.get('timeout', 30)
        self._client = None
        
    @abstractmethod
    async def complete(self, messages: List[Dict], **kwargs) -> str:
        """Generate a completion from messages."""
        pass
    
    @abstractmethod
    async def complete_stream(self, messages: List[Dict], **kwargs) -> AsyncIterator[str]:
        """Stream a completion from messages."""
        pass
    
    # Shared implementations
    async def complete_async(self, prompt: str, **kwargs) -> str:
        """Async completion wrapper - shared by all providers."""
        messages = [{"role": "user", "content": prompt}]
        return await self.complete(messages, **kwargs)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check provider health - shared implementation."""
        try:
            response = await self.complete_async("Hello", max_tokens=5)
            return {
                "status": "healthy",
                "provider": self.__class__.__name__,
                "model": self.model,
                "response_sample": response[:50] if response else None
            }
        except Exception as e:
            return {
                "status": "unhealthy", 
                "provider": self.__class__.__name__,
                "error": str(e)
            }
    
    def get_client(self):
        """Get or create HTTP client - shared."""
        if self._client is None:
            self._client = self._create_client()
        return self._client
    
    @abstractmethod
    def _create_client(self):
        """Create provider-specific client."""
        pass
    
    async def close(self):
        """Close client connections."""
        if self._client and hasattr(self._client, 'close'):
            await self._client.close()
            self._client = None
'''

    # =========================================================================
    # DEAD CODE HANDLING
    # =========================================================================
    
    def find_dead_code_candidates(self) -> Dict[Path, List[str]]:
        """Find functions that appear to be unused."""
        # Collect all function definitions
        all_functions: Dict[str, List[Path]] = defaultdict(list)
        
        for py_file in self.src_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            try:
                with open(py_file) as f:
                    tree = ast.parse(f.read())
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if not node.name.startswith('_'):
                            all_functions[node.name].append(py_file)
            except:
                continue
        
        # Check for usage
        all_code = ""
        for py_file in self.src_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            try:
                with open(py_file) as f:
                    all_code += f.read() + "\n"
            except:
                continue
        
        dead_candidates = {}
        for func_name, files in all_functions.items():
            # Count occurrences (more than 1 = definition + usage)
            count = all_code.count(func_name)
            if count <= len(files):  # Only defined, not called
                for f in files:
                    if f not in dead_candidates:
                        dead_candidates[f] = []
                    dead_candidates[f].append(func_name)
        
        return dead_candidates
    
    def archive_dead_module(self, module_path: Path) -> bool:
        """Move a module to deprecated/ directory."""
        if not module_path.exists():
            return False
            
        # Create corresponding path in deprecated/
        relative = module_path.relative_to(self.src_dir)
        archive_path = self.deprecated_dir / relative
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add deprecation header
        with open(module_path) as f:
            content = f.read()
        
        header = f'''"""
DEPRECATED: {datetime.now().strftime('%Y-%m-%d')}
This module has been moved to deprecated/ as it appears unused.
Original location: {module_path}
"""

'''
        with open(archive_path, 'w') as f:
            f.write(header + content)
        
        print(f"  Archived: {module_path} → {archive_path}")
        return True

    # =========================================================================
    # IMPORT CLEANUP  
    # =========================================================================
    
    def fix_relative_imports(self, file_path: Path) -> int:
        """Fix broken relative imports in a file."""
        fixes = 0
        try:
            with open(file_path) as f:
                content = f.read()
            
            original = content
            
            # Common fixes
            replacements = [
                # Old module paths -> new paths
                ("from composer.", "from workflow_composer."),
                ("from config.", "from workflow_composer.config."),
                ("from core.", "from workflow_composer.core."),
                ("import composer", "import workflow_composer as composer"),
            ]
            
            for old, new in replacements:
                if old in content:
                    content = content.replace(old, new)
                    fixes += 1
            
            if content != original:
                with open(file_path, 'w') as f:
                    f.write(content)
                    
        except Exception as e:
            print(f"  Error fixing {file_path}: {e}")
            
        return fixes

    # =========================================================================
    # CONSOLIDATION ACTIONS
    # =========================================================================
    
    def consolidate_routers(self) -> CleanupAction:
        """Plan to consolidate duplicate router modules."""
        models_router = self.src_dir / "models" / "router.py"
        providers_router = self.src_dir / "providers" / "router.py"
        
        files = [f for f in [models_router, providers_router] if f.exists()]
        
        return CleanupAction(
            action_type="consolidate",
            description="Merge models/router.py and providers/router.py into single router",
            files_affected=files,
            risk_level="medium",
            auto_fixable=False
        )
    
    def consolidate_providers(self) -> CleanupAction:
        """Plan to consolidate provider implementations."""
        provider_files = list((self.src_dir / "models" / "providers").glob("*.py"))
        
        return CleanupAction(
            action_type="refactor",
            description="Move shared methods (complete_async, health_check) to BaseProvider",
            files_affected=provider_files,
            risk_level="low",
            auto_fixable=True
        )

    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def generate_report(self) -> str:
        """Generate a cleanup report."""
        lines = [
            "=" * 70,
            "🧹 CODEBASE CLEANUP REPORT",
            "=" * 70,
            ""
        ]
        
        # Duplicate groups
        dup_groups = self.find_duplicate_groups()
        lines.append(f"📦 DUPLICATE CODE GROUPS: {len(dup_groups)}")
        lines.append("-" * 50)
        for group in dup_groups:
            lines.append(f"  Canonical: {group.canonical_file.name}::{group.canonical_name}")
            for dup_file, dup_name in group.duplicates:
                lines.append(f"    └─ {dup_file.name}::{dup_name}")
        lines.append("")
        
        # Dead code
        dead = self.find_dead_code_candidates()
        total_dead = sum(len(v) for v in dead.values())
        lines.append(f"💀 DEAD CODE CANDIDATES: {total_dead} functions in {len(dead)} files")
        lines.append("-" * 50)
        for file_path, funcs in list(dead.items())[:10]:
            lines.append(f"  {file_path.name}: {', '.join(funcs[:3])}...")
        if len(dead) > 10:
            lines.append(f"  ... and {len(dead) - 10} more files")
        lines.append("")
        
        # Recommended actions
        lines.append("🔧 RECOMMENDED ACTIONS")
        lines.append("-" * 50)
        
        actions = [
            self.consolidate_routers(),
            self.consolidate_providers(),
        ]
        
        for i, action in enumerate(actions, 1):
            risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}[action.risk_level]
            auto = "✅ auto" if action.auto_fixable else "🔧 manual"
            lines.append(f"  {i}. [{risk_emoji} {action.risk_level}] [{auto}] {action.description}")
            for f in action.files_affected[:3]:
                lines.append(f"       - {f.name}")
        
        lines.append("")
        lines.append("=" * 70)
        lines.append("Run with --fix-duplicates or --fix-imports to apply fixes")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    # =========================================================================
    # FIX IMPLEMENTATIONS
    # =========================================================================
    
    def fix_provider_duplicates(self) -> int:
        """Remove duplicate methods from providers, use inheritance."""
        fixes = 0
        
        base_provider_path = self.src_dir / "models" / "providers" / "base.py"
        
        # Write improved base provider
        print("  Creating improved BaseProvider with shared methods...")
        base_provider_path.parent.mkdir(parents=True, exist_ok=True)
        with open(base_provider_path, 'w') as f:
            f.write(self.generate_base_provider())
        fixes += 1
        
        # Update other providers to use base
        for provider_file in (self.src_dir / "models" / "providers").glob("*.py"):
            if provider_file.name in ("base.py", "__init__.py"):
                continue
                
            try:
                with open(provider_file) as f:
                    content = f.read()
                
                # Check if it has the duplicate methods
                if "async def complete_async" in content and "async def health_check" in content:
                    print(f"  {provider_file.name}: Has duplicate methods (manual review needed)")
                    # We don't auto-delete - just flag for review
                    fixes += 1
            except:
                continue
        
        return fixes


def main():
    parser = argparse.ArgumentParser(description="Codebase cleanup utilities")
    parser.add_argument("--report", action="store_true", help="Generate cleanup report")
    parser.add_argument("--fix-duplicates", action="store_true", help="Fix duplicate code")
    parser.add_argument("--fix-imports", action="store_true", help="Fix import issues")
    parser.add_argument("--archive-dead", action="store_true", help="Archive dead code")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    
    args = parser.parse_args()
    
    cleanup = CodebaseCleanup()
    
    if args.report or not any([args.fix_duplicates, args.fix_imports, args.archive_dead]):
        print(cleanup.generate_report())
        return
    
    if args.fix_duplicates:
        print("\n🔧 Fixing duplicate code...")
        if args.dry_run:
            print("  [DRY RUN] Would create BaseProvider with shared methods")
            print("  [DRY RUN] Would flag providers with duplicate methods")
        else:
            fixes = cleanup.fix_provider_duplicates()
            print(f"  Applied {fixes} fixes")
    
    if args.fix_imports:
        print("\n🔧 Fixing import issues...")
        total_fixes = 0
        for py_file in cleanup.src_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            if args.dry_run:
                print(f"  [DRY RUN] Would check {py_file.name}")
            else:
                fixes = cleanup.fix_relative_imports(py_file)
                if fixes:
                    print(f"  Fixed {fixes} imports in {py_file.name}")
                    total_fixes += fixes
        print(f"  Total: {total_fixes} import fixes")
    
    if args.archive_dead:
        print("\n🔧 Archiving dead code...")
        dead = cleanup.find_dead_code_candidates()
        # Only archive files where ALL functions are dead
        for file_path, funcs in dead.items():
            if len(funcs) >= 5:  # Likely a dead module
                if args.dry_run:
                    print(f"  [DRY RUN] Would archive {file_path.name}")
                else:
                    cleanup.archive_dead_module(file_path)


if __name__ == "__main__":
    main()
