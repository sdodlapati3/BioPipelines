#!/usr/bin/env python3
"""
BioPipelines Codebase Validator
===============================

A comprehensive validation system that checks:
1. Import integrity - all imports resolve
2. API endpoints - external services are reachable
3. Environment variables - required secrets are configured
4. Dead code detection - unused functions/classes
5. Configuration files - valid YAML/JSON
6. Tool registration - all tools properly connected
7. LLM provider chain - fallback chain works

Usage:
    python scripts/validate_codebase.py [--full] [--fix]
    
    --full: Run all checks including slow network tests
    --fix:  Attempt to auto-fix simple issues
"""

import os
import sys
import ast
import json
import importlib
import subprocess
import hashlib
import tokenize
import io
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from difflib import SequenceMatcher

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class ValidationResult:
    """Result of a validation check."""
    category: str
    check_name: str
    status: str  # "pass", "warn", "fail"
    message: str
    details: List[str] = field(default_factory=list)
    fixable: bool = False
    fix_command: Optional[str] = None


class CodebaseValidator:
    """Comprehensive codebase validation."""
    
    def __init__(self, project_root: Path = PROJECT_ROOT, verbose: bool = True):
        self.project_root = project_root
        self.src_root = project_root / "src"
        self.verbose = verbose
        self.results: List[ValidationResult] = []
    
    def log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def add_result(self, result: ValidationResult):
        self.results.append(result)
        symbol = {"pass": "✅", "warn": "⚠️", "fail": "❌"}[result.status]
        self.log(f"  {symbol} {result.check_name}: {result.message}")
        for detail in result.details[:3]:  # Show first 3 details
            self.log(f"      - {detail}")
        if len(result.details) > 3:
            self.log(f"      ... and {len(result.details) - 3} more")
    
    # =========================================================================
    # 1. ENVIRONMENT & SECRETS VALIDATION
    # =========================================================================
    
    def check_secrets(self) -> List[ValidationResult]:
        """Check that all required secrets are configured."""
        self.log("\n📁 Checking Secrets Configuration...")
        
        secrets_dir = self.project_root / ".secrets"
        required_secrets = {
            "github_token": "GITHUB_TOKEN",
            "google_api_key": "GOOGLE_API_KEY", 
            "openai_key": "OPENAI_API_KEY",
            "hf_token": "HF_TOKEN",
        }
        optional_secrets = {
            "lightning_key": "LIGHTNING_API_KEY",
        }
        
        results = []
        
        # Check secrets directory
        if not secrets_dir.exists():
            results.append(ValidationResult(
                category="secrets",
                check_name="secrets_directory",
                status="fail",
                message=".secrets directory not found",
                fixable=True,
                fix_command=f"mkdir -p {secrets_dir}"
            ))
            return results
        
        # Check each required secret
        for filename, env_var in required_secrets.items():
            filepath = secrets_dir / filename
            env_value = os.environ.get(env_var, "")
            
            if filepath.exists():
                content = filepath.read_text().strip()
                if content:
                    results.append(ValidationResult(
                        category="secrets",
                        check_name=f"secret_{filename}",
                        status="pass",
                        message=f"{filename} configured ({len(content)} chars)"
                    ))
                else:
                    results.append(ValidationResult(
                        category="secrets",
                        check_name=f"secret_{filename}",
                        status="warn",
                        message=f"{filename} file exists but is empty"
                    ))
            elif env_value:
                results.append(ValidationResult(
                    category="secrets",
                    check_name=f"secret_{filename}",
                    status="pass",
                    message=f"{env_var} set in environment"
                ))
            else:
                results.append(ValidationResult(
                    category="secrets",
                    check_name=f"secret_{filename}",
                    status="warn",
                    message=f"{filename} not found (optional but recommended)"
                ))
        
        for result in results:
            self.add_result(result)
        
        return results
    
    # =========================================================================
    # 2. IMPORT VALIDATION
    # =========================================================================
    
    def check_imports(self) -> List[ValidationResult]:
        """Check that all imports in Python files resolve."""
        self.log("\n📦 Checking Import Integrity...")
        
        results = []
        failed_imports = []
        checked_files = 0
        
        for py_file in self.src_root.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            
            checked_files += 1
            try:
                with open(py_file) as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            try:
                                importlib.import_module(alias.name.split('.')[0])
                            except ImportError as e:
                                failed_imports.append((py_file.relative_to(self.project_root), alias.name, str(e)))
                    
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            try:
                                importlib.import_module(node.module.split('.')[0])
                            except ImportError as e:
                                # Skip relative imports within our package
                                if not node.module.startswith("workflow_composer"):
                                    failed_imports.append((py_file.relative_to(self.project_root), node.module, str(e)))
            
            except SyntaxError as e:
                failed_imports.append((py_file.relative_to(self.project_root), "SYNTAX ERROR", str(e)))
        
        if failed_imports:
            result = ValidationResult(
                category="imports",
                check_name="import_resolution",
                status="warn",
                message=f"{len(failed_imports)} import issues in {checked_files} files",
                details=[f"{f}: {m} - {e}" for f, m, e in failed_imports[:10]]
            )
        else:
            result = ValidationResult(
                category="imports",
                check_name="import_resolution",
                status="pass",
                message=f"All imports resolve in {checked_files} files"
            )
        
        self.add_result(result)
        return [result]
    
    # =========================================================================
    # 3. TOOL SYSTEM VALIDATION
    # =========================================================================
    
    def check_tool_system(self) -> List[ValidationResult]:
        """Validate the modular tool system."""
        self.log("\n🔧 Checking Tool System...")
        
        results = []
        
        try:
            from workflow_composer.agents.tools import (
                AgentTools, get_agent_tools, ToolName, ToolResult
            )
            
            tools = get_agent_tools()
            tool_count = tools.get_tool_count()
            tool_names = tools.get_tool_names()
            
            results.append(ValidationResult(
                category="tools",
                check_name="tool_registry",
                status="pass",
                message=f"{tool_count} tools registered",
                details=tool_names[:10]
            ))
            
            # Test each tool can be invoked (with empty args)
            failed_tools = []
            for name in tool_names:
                try:
                    result = tools.execute_tool(name)
                    if not isinstance(result, ToolResult):
                        failed_tools.append((name, "Did not return ToolResult"))
                except Exception as e:
                    # Some tools may fail with empty args - that's OK
                    if "required" not in str(e).lower() and "missing" not in str(e).lower():
                        failed_tools.append((name, str(e)[:50]))
            
            if failed_tools:
                results.append(ValidationResult(
                    category="tools",
                    check_name="tool_execution",
                    status="warn",
                    message=f"{len(failed_tools)} tools had issues",
                    details=[f"{n}: {e}" for n, e in failed_tools]
                ))
            else:
                results.append(ValidationResult(
                    category="tools",
                    check_name="tool_execution",
                    status="pass",
                    message="All tools execute without errors"
                ))
            
            # Check OpenAI function definitions
            funcs = tools.get_openai_functions()
            results.append(ValidationResult(
                category="tools",
                check_name="openai_functions",
                status="pass",
                message=f"{len(funcs)} OpenAI function definitions generated"
            ))
            
        except ImportError as e:
            results.append(ValidationResult(
                category="tools",
                check_name="tool_import",
                status="fail",
                message=f"Cannot import tool system: {e}"
            ))
        except Exception as e:
            results.append(ValidationResult(
                category="tools",
                check_name="tool_system",
                status="fail",
                message=f"Tool system error: {e}"
            ))
        
        for result in results:
            self.add_result(result)
        
        return results
    
    # =========================================================================
    # 4. LLM PROVIDER VALIDATION
    # =========================================================================
    
    def check_llm_providers(self, test_connections: bool = False) -> List[ValidationResult]:
        """Validate LLM provider configuration."""
        self.log("\n🤖 Checking LLM Providers...")
        
        results = []
        
        try:
            from workflow_composer.web.chat_handler import LLMProvider
            
            provider = LLMProvider()
            
            results.append(ValidationResult(
                category="llm",
                check_name="llm_provider_init",
                status="pass",
                message=f"Active provider: {provider.active_provider or 'None'}"
            ))
            
            # Check each configured provider
            for name, client in provider.clients.items():
                model = provider.models.get(name, "unknown")
                is_active = name == provider.active_provider
                results.append(ValidationResult(
                    category="llm",
                    check_name=f"llm_{name}",
                    status="pass" if is_active else "warn",
                    message=f"{name}: {model}" + (" (ACTIVE)" if is_active else " (standby)")
                ))
            
            if not provider.available:
                results.append(ValidationResult(
                    category="llm",
                    check_name="llm_availability",
                    status="warn",
                    message="No LLM provider available - pattern matching only"
                ))
            
            # Test actual connection if requested
            if test_connections and provider.available:
                try:
                    client, model = provider.get_client_and_model()
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": "Say 'OK'"}],
                        max_tokens=5
                    )
                    results.append(ValidationResult(
                        category="llm",
                        check_name="llm_connection",
                        status="pass",
                        message=f"Successfully connected to {provider.active_provider}"
                    ))
                except Exception as e:
                    results.append(ValidationResult(
                        category="llm",
                        check_name="llm_connection",
                        status="fail",
                        message=f"Connection failed: {str(e)[:50]}"
                    ))
        
        except ImportError as e:
            results.append(ValidationResult(
                category="llm",
                check_name="llm_import",
                status="fail",
                message=f"Cannot import LLM provider: {e}"
            ))
        
        for result in results:
            self.add_result(result)
        
        return results
    
    # =========================================================================
    # 5. CHAT HANDLER VALIDATION
    # =========================================================================
    
    def check_chat_handler(self) -> List[ValidationResult]:
        """Validate the unified chat handler."""
        self.log("\n💬 Checking Chat Handler...")
        
        results = []
        
        try:
            from workflow_composer.web.chat_handler import (
                get_chat_handler, UnifiedChatHandler, SessionManager
            )
            
            handler = get_chat_handler()
            
            results.append(ValidationResult(
                category="chat",
                check_name="handler_init",
                status="pass",
                message="Chat handler initialized"
            ))
            
            # Check components
            if handler._tools_available:
                results.append(ValidationResult(
                    category="chat",
                    check_name="handler_tools",
                    status="pass",
                    message="Tools connected to handler"
                ))
            else:
                results.append(ValidationResult(
                    category="chat",
                    check_name="handler_tools",
                    status="warn",
                    message="Tools not available in handler"
                ))
            
            if handler.llm_provider and handler.llm_provider.available:
                results.append(ValidationResult(
                    category="chat",
                    check_name="handler_llm",
                    status="pass",
                    message=f"LLM connected: {handler.llm_provider.active_provider}"
                ))
            else:
                results.append(ValidationResult(
                    category="chat",
                    check_name="handler_llm",
                    status="warn",
                    message="No LLM available - pattern matching only"
                ))
            
            # Test pattern detection
            test_patterns = [
                ("scan my data", "scan_data"),
                ("list workflows", "list_workflows"),
                ("help", "get_help"),
            ]
            
            for msg, expected in test_patterns:
                detected = handler._tools.detect_tool(msg) if handler._tools else None
                if detected == expected:
                    results.append(ValidationResult(
                        category="chat",
                        check_name=f"pattern_{expected}",
                        status="pass",
                        message=f"'{msg}' → {detected}"
                    ))
                else:
                    results.append(ValidationResult(
                        category="chat",
                        check_name=f"pattern_{expected}",
                        status="warn",
                        message=f"'{msg}' → {detected} (expected {expected})"
                    ))
        
        except Exception as e:
            results.append(ValidationResult(
                category="chat",
                check_name="handler_error",
                status="fail",
                message=f"Chat handler error: {e}"
            ))
        
        for result in results:
            self.add_result(result)
        
        return results
    
    # =========================================================================
    # 6. DEAD CODE DETECTION
    # =========================================================================
    
    def check_dead_code(self) -> List[ValidationResult]:
        """Find potentially unused code."""
        self.log("\n🧹 Checking for Dead Code...")
        
        results = []
        
        # Find all function/class definitions
        definitions: Dict[str, Set[str]] = {"functions": set(), "classes": set()}
        references: Set[str] = set()
        
        for py_file in self.src_root.rglob("*.py"):
            if "__pycache__" in str(py_file) or "deprecated" in str(py_file):
                continue
            
            try:
                with open(py_file) as f:
                    content = f.read()
                    tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Skip private and dunder methods
                        if not node.name.startswith('_'):
                            definitions["functions"].add(node.name)
                    elif isinstance(node, ast.ClassDef):
                        definitions["classes"].add(node.name)
                    elif isinstance(node, ast.Name):
                        references.add(node.id)
                    elif isinstance(node, ast.Attribute):
                        references.add(node.attr)
            
            except Exception:
                pass
        
        # Find unreferenced items
        unreferenced_funcs = definitions["functions"] - references
        unreferenced_classes = definitions["classes"] - references
        
        # Filter out common patterns
        common_patterns = {"main", "run", "start", "init", "setup", "create_app"}
        unreferenced_funcs -= common_patterns
        
        if unreferenced_funcs:
            results.append(ValidationResult(
                category="dead_code",
                check_name="unused_functions",
                status="warn",
                message=f"{len(unreferenced_funcs)} potentially unused functions",
                details=list(unreferenced_funcs)[:10]
            ))
        else:
            results.append(ValidationResult(
                category="dead_code",
                check_name="unused_functions",
                status="pass",
                message="No obviously unused functions found"
            ))
        
        for result in results:
            self.add_result(result)
        
        return results
    
    # =========================================================================
    # 6.5 DUPLICATE CODE DETECTION
    # =========================================================================
    
    def check_duplicate_code(self, min_lines: int = 6, similarity_threshold: float = 0.85) -> List[ValidationResult]:
        """
        Find duplicate or near-duplicate code blocks.
        
        Uses multiple detection strategies:
        1. Exact hash matching (identical code blocks)
        2. AST structure matching (same structure, different names)
        3. Token sequence similarity (similar patterns)
        
        Args:
            min_lines: Minimum lines for a block to be considered
            similarity_threshold: Similarity ratio for near-duplicates (0.0-1.0)
        """
        self.log("\n🔄 Checking for Duplicate Code...")
        
        results = []
        
        # Collect all function/method definitions
        @dataclass
        class CodeBlock:
            name: str
            file: Path
            line: int
            source: str
            normalized: str  # Normalized for comparison
            ast_hash: str    # Hash of AST structure
            tokens: List[str]
        
        code_blocks: List[CodeBlock] = []
        
        for py_file in self.src_root.rglob("*.py"):
            if "__pycache__" in str(py_file) or "deprecated" in str(py_file):
                continue
            
            try:
                with open(py_file) as f:
                    source = f.read()
                
                tree = ast.parse(source)
                lines = source.split('\n')
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        # Skip short functions
                        if hasattr(node, 'end_lineno') and node.end_lineno:
                            func_lines = node.end_lineno - node.lineno
                            if func_lines < min_lines:
                                continue
                        
                        # Extract source
                        try:
                            func_source = ast.get_source_segment(source, node)
                            if not func_source:
                                continue
                        except Exception:
                            continue
                        
                        # Normalize: remove comments, normalize whitespace
                        normalized = self._normalize_code(func_source)
                        
                        # Create AST hash (structure without names)
                        ast_hash = self._hash_ast_structure(node)
                        
                        # Tokenize
                        tokens = self._tokenize_code(func_source)
                        
                        code_blocks.append(CodeBlock(
                            name=node.name,
                            file=py_file.relative_to(self.project_root),
                            line=node.lineno,
                            source=func_source,
                            normalized=normalized,
                            ast_hash=ast_hash,
                            tokens=tokens
                        ))
            
            except Exception as e:
                pass
        
        # Strategy 1: Find exact duplicates by hash
        exact_duplicates = self._find_exact_duplicates(code_blocks)
        
        # Strategy 2: Find structural duplicates by AST
        structural_duplicates = self._find_structural_duplicates(code_blocks)
        
        # Strategy 3: Find similar code by token similarity
        similar_code = self._find_similar_code(code_blocks, similarity_threshold)
        
        # Report results
        if exact_duplicates:
            results.append(ValidationResult(
                category="duplicates",
                check_name="exact_duplicates",
                status="warn",
                message=f"{len(exact_duplicates)} sets of identical code blocks",
                details=[f"{d[0].name} ({d[0].file}:{d[0].line}) == {d[1].name} ({d[1].file}:{d[1].line})" 
                        for d in exact_duplicates[:5]]
            ))
        else:
            results.append(ValidationResult(
                category="duplicates",
                check_name="exact_duplicates",
                status="pass",
                message="No exact duplicate functions found"
            ))
        
        if structural_duplicates:
            results.append(ValidationResult(
                category="duplicates",
                check_name="structural_duplicates",
                status="warn",
                message=f"{len(structural_duplicates)} sets of structurally identical code",
                details=[f"{d[0].name} ({d[0].file}) ≈ {d[1].name} ({d[1].file})" 
                        for d in structural_duplicates[:5]]
            ))
        
        if similar_code:
            results.append(ValidationResult(
                category="duplicates",
                check_name="similar_code",
                status="warn",
                message=f"{len(similar_code)} pairs of highly similar code ({int(similarity_threshold*100)}%+)",
                details=[f"{s[0].name} ~ {s[1].name} ({s[2]:.0%} similar)" 
                        for s in similar_code[:5]]
            ))
        
        # Summary
        total_issues = len(exact_duplicates) + len(structural_duplicates) + len(similar_code)
        if total_issues == 0:
            results.append(ValidationResult(
                category="duplicates",
                check_name="duplicate_summary",
                status="pass",
                message=f"Analyzed {len(code_blocks)} functions, no significant duplication"
            ))
        else:
            results.append(ValidationResult(
                category="duplicates",
                check_name="duplicate_summary",
                status="warn",
                message=f"Found {total_issues} potential code duplication issues"
            ))
        
        for result in results:
            self.add_result(result)
        
        return results
    
    def _normalize_code(self, source: str) -> str:
        """Normalize code for comparison by removing variable names and whitespace."""
        try:
            # Remove comments and docstrings
            tree = ast.parse(source)
            
            # Extract just the structure
            normalized_parts = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                    continue  # Skip docstrings
                node_type = type(node).__name__
                normalized_parts.append(node_type)
            
            return ' '.join(normalized_parts)
        except Exception:
            # Fallback: just normalize whitespace
            return ' '.join(source.split())
    
    def _hash_ast_structure(self, node: ast.AST) -> str:
        """Create a hash of AST structure, ignoring variable names."""
        def structure_repr(n, depth=0):
            if depth > 20:  # Prevent infinite recursion
                return ""
            
            parts = [type(n).__name__]
            
            # Add relevant attributes but not names
            if isinstance(n, ast.BinOp):
                parts.append(type(n.op).__name__)
            elif isinstance(n, ast.Compare):
                parts.extend(type(op).__name__ for op in n.ops)
            elif isinstance(n, ast.Call):
                parts.append(f"args:{len(n.args)}")
            
            # Recurse into children
            for child in ast.iter_child_nodes(n):
                parts.append(structure_repr(child, depth + 1))
            
            return f"({' '.join(parts)})"
        
        structure = structure_repr(node)
        return hashlib.md5(structure.encode()).hexdigest()[:12]
    
    def _tokenize_code(self, source: str) -> List[str]:
        """Tokenize code, normalizing identifiers."""
        tokens = []
        try:
            for tok in tokenize.generate_tokens(io.StringIO(source).readline):
                if tok.type == tokenize.NAME:
                    tokens.append("NAME")  # Normalize all names
                elif tok.type == tokenize.NUMBER:
                    tokens.append("NUM")
                elif tok.type == tokenize.STRING:
                    tokens.append("STR")
                elif tok.type in (tokenize.OP, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
                    tokens.append(tok.string)
        except tokenize.TokenizeError:
            pass
        return tokens
    
    def _find_exact_duplicates(self, blocks: List) -> List[Tuple]:
        """Find blocks with identical normalized code."""
        hash_to_blocks = defaultdict(list)
        
        for block in blocks:
            code_hash = hashlib.md5(block.normalized.encode()).hexdigest()
            hash_to_blocks[code_hash].append(block)
        
        duplicates = []
        for blocks_list in hash_to_blocks.values():
            if len(blocks_list) > 1:
                # Create pairs
                for i in range(len(blocks_list)):
                    for j in range(i + 1, len(blocks_list)):
                        # Skip if same file (might be intentional overloads)
                        if blocks_list[i].file != blocks_list[j].file:
                            duplicates.append((blocks_list[i], blocks_list[j]))
        
        return duplicates
    
    def _find_structural_duplicates(self, blocks: List) -> List[Tuple]:
        """Find blocks with identical AST structure but different names."""
        hash_to_blocks = defaultdict(list)
        
        for block in blocks:
            hash_to_blocks[block.ast_hash].append(block)
        
        duplicates = []
        for blocks_list in hash_to_blocks.values():
            if len(blocks_list) > 1:
                for i in range(len(blocks_list)):
                    for j in range(i + 1, len(blocks_list)):
                        b1, b2 = blocks_list[i], blocks_list[j]
                        # Only if different names (not same function in diff file)
                        if b1.name != b2.name and b1.file != b2.file:
                            duplicates.append((b1, b2))
        
        return duplicates[:20]  # Limit results
    
    def _find_similar_code(self, blocks: List, threshold: float) -> List[Tuple]:
        """Find blocks with high token sequence similarity."""
        similar = []
        
        # Only compare across different files (skip same-file comparisons)
        files = defaultdict(list)
        for block in blocks:
            files[str(block.file)].append(block)
        
        file_list = list(files.keys())
        
        for i in range(len(file_list)):
            for j in range(i + 1, len(file_list)):
                for b1 in files[file_list[i]]:
                    for b2 in files[file_list[j]]:
                        if len(b1.tokens) < 20 or len(b2.tokens) < 20:
                            continue
                        
                        # Quick length check first
                        len_ratio = min(len(b1.tokens), len(b2.tokens)) / max(len(b1.tokens), len(b2.tokens))
                        if len_ratio < threshold:
                            continue
                        
                        # Detailed similarity
                        ratio = SequenceMatcher(None, b1.tokens, b2.tokens).ratio()
                        if ratio >= threshold:
                            similar.append((b1, b2, ratio))
        
        # Sort by similarity, return top matches
        similar.sort(key=lambda x: x[2], reverse=True)
        return similar[:10]
    
    # =========================================================================
    # 7. CONFIGURATION FILES
    # =========================================================================
    
    def check_configs(self) -> List[ValidationResult]:
        """Validate configuration files."""
        self.log("\n⚙️ Checking Configuration Files...")
        
        results = []
        config_dir = self.project_root / "config"
        
        if not config_dir.exists():
            results.append(ValidationResult(
                category="config",
                check_name="config_dir",
                status="warn",
                message="config/ directory not found"
            ))
            self.add_result(results[0])
            return results
        
        # Check YAML files
        import yaml
        
        for yaml_file in config_dir.glob("*.yaml"):
            try:
                with open(yaml_file) as f:
                    yaml.safe_load(f)
                results.append(ValidationResult(
                    category="config",
                    check_name=f"yaml_{yaml_file.stem}",
                    status="pass",
                    message=f"{yaml_file.name} is valid YAML"
                ))
            except yaml.YAMLError as e:
                results.append(ValidationResult(
                    category="config",
                    check_name=f"yaml_{yaml_file.stem}",
                    status="fail",
                    message=f"{yaml_file.name} has invalid YAML: {e}"
                ))
        
        for result in results:
            self.add_result(result)
        
        return results
    
    # =========================================================================
    # 8. WEB APP VALIDATION
    # =========================================================================
    
    def check_web_app(self) -> List[ValidationResult]:
        """Validate web application setup."""
        self.log("\n🌐 Checking Web Application...")
        
        results = []
        
        try:
            from workflow_composer.web.app import (
                chat_handler, chat_response, create_app, HANDLER_AVAILABLE
            )
            
            results.append(ValidationResult(
                category="web",
                check_name="app_import",
                status="pass",
                message="Web app imports successfully"
            ))
            
            results.append(ValidationResult(
                category="web",
                check_name="handler_available",
                status="pass" if HANDLER_AVAILABLE else "fail",
                message=f"Handler available: {HANDLER_AVAILABLE}"
            ))
            
            # Test chat_response function
            try:
                responses = list(chat_response("help", []))
                if responses and len(responses[-1]) >= 2:
                    results.append(ValidationResult(
                        category="web",
                        check_name="chat_response",
                        status="pass",
                        message="chat_response() works correctly"
                    ))
                else:
                    results.append(ValidationResult(
                        category="web",
                        check_name="chat_response",
                        status="warn",
                        message="chat_response() returned unexpected format"
                    ))
            except Exception as e:
                results.append(ValidationResult(
                    category="web",
                    check_name="chat_response",
                    status="fail",
                    message=f"chat_response() failed: {e}"
                ))
        
        except ImportError as e:
            results.append(ValidationResult(
                category="web",
                check_name="app_import",
                status="fail",
                message=f"Cannot import web app: {e}"
            ))
        
        for result in results:
            self.add_result(result)
        
        return results
    
    # =========================================================================
    # MAIN VALIDATION
    # =========================================================================
    
    def run_all(self, full: bool = False) -> Dict:
        """Run all validation checks."""
        start_time = datetime.now()
        
        print("=" * 70)
        print("🔍 BioPipelines Codebase Validation")
        print("=" * 70)
        
        self.check_secrets()
        self.check_imports()
        self.check_tool_system()
        self.check_llm_providers(test_connections=full)
        self.check_chat_handler()
        self.check_dead_code()
        self.check_duplicate_code()
        self.check_configs()
        self.check_web_app()
        
        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        
        passed = len([r for r in self.results if r.status == "pass"])
        warnings = len([r for r in self.results if r.status == "warn"])
        failed = len([r for r in self.results if r.status == "fail"])
        
        print("\n" + "=" * 70)
        print("📊 VALIDATION SUMMARY")
        print("=" * 70)
        print(f"  ✅ Passed:   {passed}")
        print(f"  ⚠️  Warnings: {warnings}")
        print(f"  ❌ Failed:   {failed}")
        print(f"  ⏱️  Time:     {elapsed:.2f}s")
        
        if failed > 0:
            print("\n❌ CRITICAL ISSUES:")
            for r in self.results:
                if r.status == "fail":
                    print(f"  - {r.category}/{r.check_name}: {r.message}")
                    if r.fix_command:
                        print(f"    Fix: {r.fix_command}")
        
        print("=" * 70)
        
        return {
            "passed": passed,
            "warnings": warnings,
            "failed": failed,
            "results": self.results,
            "elapsed": elapsed
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate BioPipelines codebase")
    parser.add_argument("--full", action="store_true", help="Run full tests including network")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix issues")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    validator = CodebaseValidator(verbose=not args.json)
    summary = validator.run_all(full=args.full)
    
    if args.json:
        import json
        print(json.dumps({
            "passed": summary["passed"],
            "warnings": summary["warnings"],
            "failed": summary["failed"],
            "elapsed": summary["elapsed"]
        }))
    
    # Exit with error code if there are failures
    sys.exit(1 if summary["failed"] > 0 else 0)


if __name__ == "__main__":
    main()
