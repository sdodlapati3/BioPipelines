"""
Knowledge Base
==============

Multi-source knowledge base for enhanced RAG.

Sources:
- Tool catalog: BioPipelines tool information
- nf-core modules: Nextflow module documentation
- Paper abstracts: Bioinformatics paper summaries
- Error patterns: Common error solutions
- Best practices: Workflow design guidelines
"""

import logging
import yaml
import json
import sqlite3
import hashlib
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class KnowledgeSource(Enum):
    """Types of knowledge sources."""
    TOOL_CATALOG = "tool_catalog"
    NF_CORE_MODULES = "nf_core_modules"
    PAPER_ABSTRACTS = "paper_abstracts"
    ERROR_PATTERNS = "error_patterns"
    BEST_PRACTICES = "best_practices"


@dataclass
class KnowledgeBaseConfig:
    """Configuration for knowledge base."""
    base_path: str = "~/.biopipelines/knowledge"
    auto_index_nf_core: bool = False
    auto_index_tools: bool = True
    max_results: int = 10
    embedding_enabled: bool = False
    cache_embeddings: bool = True


@dataclass
class KnowledgeDocument:
    """A document in the knowledge base."""
    id: str
    source: KnowledgeSource
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "source": self.source.value,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeDocument":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            source=KnowledgeSource(data["source"]),
            title=data["title"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
        )


class KnowledgeBase:
    """
    Multi-source knowledge base for RAG.
    
    Indexes and retrieves knowledge from multiple sources
    to enhance workflow generation with contextual information.
    """
    
    def __init__(self, base_path: str = "~/.biopipelines/knowledge"):
        """
        Initialize knowledge base.
        
        Args:
            base_path: Directory for knowledge storage
        """
        self.base_path = Path(base_path).expanduser()
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.base_path / "knowledge.db"
        self._init_db()
        
        # Source-specific indexers
        self._indexers = {}
    
    @contextmanager
    def _get_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    embedding BLOB,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_source 
                ON documents(source)
            """)
            
            # Full-text search index
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts 
                USING fts5(id, title, content, source)
            """)
            
            conn.commit()
    
    def add_document(self, doc: KnowledgeDocument):
        """
        Add a document to the knowledge base.
        
        Args:
            doc: Document to add
        """
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO documents 
                (id, source, title, content, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                doc.id,
                doc.source.value,
                doc.title,
                doc.content,
                json.dumps(doc.metadata),
                doc.created_at.isoformat(),
            ))
            
            # Update FTS index
            conn.execute("""
                INSERT OR REPLACE INTO documents_fts (id, title, content, source)
                VALUES (?, ?, ?, ?)
            """, (doc.id, doc.title, doc.content, doc.source.value))
            
            conn.commit()
    
    def search(self, query: str, sources: List[KnowledgeSource] = None,
               limit: int = 10) -> List[KnowledgeDocument]:
        """
        Search the knowledge base.
        
        Args:
            query: Search query
            sources: Filter by sources (None = all)
            limit: Maximum results
            
        Returns:
            List of matching documents
        """
        # Escape special FTS5 characters and quote terms
        # FTS5 treats - as NOT, so we need to quote terms with hyphens
        escaped_query = self._escape_fts_query(query)
        
        with self._get_connection() as conn:
            try:
                if sources:
                    source_filter = ", ".join(f"'{s.value}'" for s in sources)
                    rows = conn.execute(f"""
                        SELECT d.* FROM documents d
                        JOIN documents_fts fts ON d.id = fts.id
                        WHERE documents_fts MATCH ?
                        AND d.source IN ({source_filter})
                        ORDER BY rank
                        LIMIT ?
                    """, (escaped_query, limit)).fetchall()
                else:
                    rows = conn.execute("""
                        SELECT d.* FROM documents d
                        JOIN documents_fts fts ON d.id = fts.id
                        WHERE documents_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                    """, (escaped_query, limit)).fetchall()
            except sqlite3.OperationalError:
                # If FTS query fails, fall back to LIKE search
                like_query = f"%{query}%"
                if sources:
                    source_filter = ", ".join(f"'{s.value}'" for s in sources)
                    rows = conn.execute(f"""
                        SELECT * FROM documents
                        WHERE (title LIKE ? OR content LIKE ?)
                        AND source IN ({source_filter})
                        LIMIT ?
                    """, (like_query, like_query, limit)).fetchall()
                else:
                    rows = conn.execute("""
                        SELECT * FROM documents
                        WHERE title LIKE ? OR content LIKE ?
                        LIMIT ?
                    """, (like_query, like_query, limit)).fetchall()
            
            return [
                KnowledgeDocument(
                    id=row["id"],
                    source=KnowledgeSource(row["source"]),
                    title=row["title"],
                    content=row["content"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                for row in rows
            ]
    
    def _escape_fts_query(self, query: str) -> str:
        """
        Escape FTS5 special characters in query.
        
        Args:
            query: Raw query string
            
        Returns:
            Escaped query safe for FTS5
        """
        # Quote each term to handle special characters like hyphens
        terms = query.split()
        quoted_terms = [f'"{term}"' for term in terms]
        return " ".join(quoted_terms)
    
    def get_by_source(self, source: KnowledgeSource, 
                      limit: int = 100) -> List[KnowledgeDocument]:
        """Get documents by source."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM documents
                WHERE source = ?
                LIMIT ?
            """, (source.value, limit)).fetchall()
            
            return [
                KnowledgeDocument(
                    id=row["id"],
                    source=KnowledgeSource(row["source"]),
                    title=row["title"],
                    content=row["content"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                for row in rows
            ]
    
    def get_stats(self) -> Dict[str, int]:
        """Get knowledge base statistics."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT source, COUNT(*) as count
                FROM documents
                GROUP BY source
            """).fetchall()
            
            return {row["source"]: row["count"] for row in rows}
    
    async def index_nf_core(self, modules_path: str = None):
        """
        Index nf-core modules.
        
        Args:
            modules_path: Path to nf-core/modules clone
        """
        if modules_path is None:
            modules_path = self.base_path / "nf-core-modules"
        else:
            modules_path = Path(modules_path)
        
        # Clone if not exists
        if not modules_path.exists():
            logger.info("Cloning nf-core/modules repository...")
            try:
                subprocess.run([
                    "git", "clone", "--depth", "1",
                    "https://github.com/nf-core/modules.git",
                    str(modules_path)
                ], check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to clone nf-core/modules: {e}")
                return
        
        # Index all module meta.yml files
        count = 0
        for meta_file in modules_path.glob("modules/nf-core/**/meta.yml"):
            try:
                module_info = yaml.safe_load(meta_file.read_text())
                
                # Extract module name from path
                module_name = "/".join(meta_file.parent.relative_to(modules_path / "modules/nf-core").parts)
                
                # Create document
                doc = KnowledgeDocument(
                    id=f"nfcore_{module_name.replace('/', '_')}",
                    source=KnowledgeSource.NF_CORE_MODULES,
                    title=module_info.get("name", module_name),
                    content=self._format_module_content(module_info, module_name),
                    metadata={
                        "module_name": module_name,
                        "tools": module_info.get("tools", []),
                        "keywords": module_info.get("keywords", []),
                    },
                )
                
                self.add_document(doc)
                count += 1
                
            except Exception as e:
                logger.debug(f"Failed to index {meta_file}: {e}")
        
        logger.info(f"Indexed {count} nf-core modules")
    
    def _format_module_content(self, info: Dict, name: str) -> str:
        """Format module info for indexing."""
        parts = [
            f"Module: {name}",
            f"Description: {info.get('description', 'No description')}",
        ]
        
        if "tools" in info:
            for tool in info["tools"]:
                if isinstance(tool, dict):
                    for tool_name, tool_info in tool.items():
                        parts.append(f"Tool: {tool_name}")
                        if isinstance(tool_info, dict):
                            parts.append(f"  Description: {tool_info.get('description', '')}")
                            parts.append(f"  Homepage: {tool_info.get('homepage', '')}")
        
        if "input" in info:
            parts.append("Inputs:")
            for inp in info["input"]:
                if isinstance(inp, dict):
                    for inp_name, inp_info in inp.items():
                        parts.append(f"  - {inp_name}: {inp_info.get('description', '')}")
        
        if "output" in info:
            parts.append("Outputs:")
            for out in info["output"]:
                if isinstance(out, dict):
                    for out_name, out_info in out.items():
                        parts.append(f"  - {out_name}: {out_info.get('description', '')}")
        
        if "keywords" in info:
            parts.append(f"Keywords: {', '.join(info['keywords'])}")
        
        return "\n".join(parts)
    
    def index_tool_catalog(self, catalog_path: str):
        """
        Index tool catalog YAML files.
        
        Args:
            catalog_path: Path to tool catalog directory
        """
        catalog_path = Path(catalog_path)
        count = 0
        
        for yaml_file in catalog_path.glob("**/*.yaml"):
            try:
                tools = yaml.safe_load(yaml_file.read_text())
                
                if isinstance(tools, list):
                    for tool in tools:
                        doc = self._tool_to_document(tool)
                        if doc:
                            self.add_document(doc)
                            count += 1
                elif isinstance(tools, dict) and "tools" in tools:
                    for tool in tools["tools"]:
                        doc = self._tool_to_document(tool)
                        if doc:
                            self.add_document(doc)
                            count += 1
                            
            except Exception as e:
                logger.debug(f"Failed to index {yaml_file}: {e}")
        
        logger.info(f"Indexed {count} tools from catalog")
    
    def _tool_to_document(self, tool: Dict) -> Optional[KnowledgeDocument]:
        """Convert tool info to document."""
        if "name" not in tool:
            return None
        
        content_parts = [
            f"Tool: {tool['name']}",
            f"Description: {tool.get('description', 'No description')}",
            f"Category: {tool.get('category', 'Unknown')}",
        ]
        
        if "homepage" in tool:
            content_parts.append(f"Homepage: {tool['homepage']}")
        
        if "parameters" in tool:
            content_parts.append("Parameters:")
            for param in tool["parameters"]:
                if isinstance(param, dict):
                    content_parts.append(f"  - {param.get('name', '')}: {param.get('description', '')}")
        
        return KnowledgeDocument(
            id=f"tool_{tool['name'].replace(' ', '_').lower()}",
            source=KnowledgeSource.TOOL_CATALOG,
            title=tool["name"],
            content="\n".join(content_parts),
            metadata={
                "category": tool.get("category"),
                "version": tool.get("version"),
                "homepage": tool.get("homepage"),
            },
        )
    
    def index_best_practices(self, practices: List[Dict[str, str]]):
        """
        Index best practices guidelines.
        
        Args:
            practices: List of best practice dictionaries
        """
        for i, practice in enumerate(practices):
            doc = KnowledgeDocument(
                id=f"bp_{i}_{hashlib.md5(practice.get('title', '').encode()).hexdigest()[:8]}",
                source=KnowledgeSource.BEST_PRACTICES,
                title=practice.get("title", f"Best Practice {i+1}"),
                content=practice.get("content", ""),
                metadata={
                    "category": practice.get("category"),
                    "analysis_types": practice.get("analysis_types", []),
                },
            )
            self.add_document(doc)
        
        logger.info(f"Indexed {len(practices)} best practices")
    
    def cleanup(self, source: KnowledgeSource = None):
        """Remove documents by source or all."""
        with self._get_connection() as conn:
            if source:
                conn.execute("DELETE FROM documents WHERE source = ?", (source.value,))
                conn.execute("DELETE FROM documents_fts WHERE source = ?", (source.value,))
            else:
                conn.execute("DELETE FROM documents")
                conn.execute("DELETE FROM documents_fts")
            conn.commit()
