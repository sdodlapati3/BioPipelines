# Claude Scientific Skills Integration Plan for BioPipelines

## Executive Summary

This document outlines a comprehensive plan to integrate best practices from Claude Scientific Skills into BioPipelines, enhancing our system's capabilities while maintaining our core strength: **execution over documentation**.

**Goal**: Borrow the best patterns from Claude Scientific Skills to improve tool discovery, add valuable database integrations, provide workflow templates, and optionally expose BioPipelines via MCP protocol.

**Timeline**: 4 Phases over estimated 4-6 development sessions

---

## Phase 1: Skill Documentation System

### 1.1 Objective
Create structured YAML-based skill documentation that helps the ChatAgent/UnifiedAgent better understand and select tools.

### 1.2 Architecture

```
config/
└── skills/
    ├── __init__.py           # Skill loader
    ├── schema.py             # Pydantic models for skill validation
    │
    ├── data_discovery/
    │   ├── encode_search.yaml
    │   ├── geo_search.yaml
    │   ├── tcga_search.yaml
    │   └── data_scan.yaml
    │
    ├── data_management/
    │   ├── download_dataset.yaml
    │   ├── validate_data.yaml
    │   └── cleanup_data.yaml
    │
    ├── workflow_generation/
    │   ├── rnaseq_workflow.yaml
    │   ├── chipseq_workflow.yaml
    │   ├── methylation_workflow.yaml
    │   └── variant_calling.yaml
    │
    ├── job_management/
    │   ├── submit_job.yaml
    │   ├── check_status.yaml
    │   └── cancel_job.yaml
    │
    ├── education/
    │   ├── explain_concept.yaml
    │   └── compare_tools.yaml
    │
    └── databases/
        ├── uniprot.yaml       # NEW
        ├── string_db.yaml     # NEW
        ├── kegg.yaml          # NEW
        ├── reactome.yaml      # NEW
        └── pubmed.yaml        # NEW
```

### 1.3 Skill YAML Schema

```yaml
# Example: config/skills/data_discovery/encode_search.yaml
---
# Metadata
name: encode_search
version: "1.0.0"
category: data_discovery
description: |
  Search the ENCODE database for experiments and datasets.
  ENCODE contains chromatin accessibility, histone modifications,
  transcription factor binding, and gene expression data.

# When to use this skill
triggers:
  keywords:
    - "search encode"
    - "find encode"
    - "encode datasets"
    - "encode experiments"
  patterns:
    - "search.*encode"
    - "find.*in encode"
    - "encode.*data"
  intents:
    - DATA_SEARCH
    - ENCODE_QUERY

# Capability details
capabilities:
  - Search by assay type (ChIP-seq, ATAC-seq, RNA-seq, etc.)
  - Filter by organism (human, mouse)
  - Filter by biosample (cell line, tissue)
  - Filter by target (transcription factors, histone marks)
  - Download experiment metadata
  - Retrieve file URLs

# Parameters the tool accepts
parameters:
  required:
    - name: query
      type: string
      description: Search terms (organism, cell type, assay, etc.)
  optional:
    - name: assay_type
      type: string
      enum: [ChIP-seq, ATAC-seq, RNA-seq, WGBS, Hi-C]
      description: Filter by assay type
    - name: organism
      type: string
      enum: [human, mouse]
      default: human
      description: Filter by organism
    - name: biosample
      type: string
      description: Cell line or tissue type
    - name: limit
      type: integer
      default: 10
      description: Maximum results to return

# Example prompts and expected behavior
examples:
  - prompt: "Search ENCODE for K562 ChIP-seq"
    expected_params:
      query: "K562 ChIP-seq"
      biosample: "K562"
      assay_type: "ChIP-seq"
    expected_behavior: "Returns ChIP-seq experiments from K562 cells"
    
  - prompt: "Find ATAC-seq data for mouse liver"
    expected_params:
      query: "mouse liver ATAC-seq"
      organism: "mouse"
      assay_type: "ATAC-seq"
      biosample: "liver"
    expected_behavior: "Returns ATAC-seq experiments from mouse liver tissue"

  - prompt: "Search for H3K27ac histone data in human"
    expected_params:
      query: "H3K27ac human"
      organism: "human"
    expected_behavior: "Returns ChIP-seq experiments targeting H3K27ac"

# Related skills
related_skills:
  - geo_search
  - tcga_search
  - download_dataset

# Tool mapping
tool_binding:
  tool_name: search_data
  tool_module: workflow_composer.agents.tools
  default_kwargs:
    source: "ENCODE"

# Output format
output:
  type: list
  item_schema:
    id: string
    title: string
    assay_type: string
    organism: string
    biosample: string
    files: list
```

### 1.4 Skill Loader Implementation

```python
# config/skills/__init__.py
"""
Skill Documentation System

Loads and manages skill definitions for improved tool selection.
"""

from pathlib import Path
from typing import Dict, List, Optional
import yaml
from pydantic import BaseModel, Field
from functools import lru_cache

class SkillParameter(BaseModel):
    name: str
    type: str
    description: str
    enum: Optional[List[str]] = None
    default: Optional[Any] = None

class SkillExample(BaseModel):
    prompt: str
    expected_params: Dict[str, Any]
    expected_behavior: str

class SkillTriggers(BaseModel):
    keywords: List[str] = Field(default_factory=list)
    patterns: List[str] = Field(default_factory=list)
    intents: List[str] = Field(default_factory=list)

class SkillDefinition(BaseModel):
    name: str
    version: str
    category: str
    description: str
    triggers: SkillTriggers
    capabilities: List[str]
    parameters: Dict[str, List[SkillParameter]]
    examples: List[SkillExample]
    related_skills: List[str] = Field(default_factory=list)
    tool_binding: Dict[str, Any]
    output: Dict[str, Any]

class SkillRegistry:
    """Registry for all available skills."""
    
    def __init__(self, skills_dir: Path = None):
        self.skills_dir = skills_dir or Path(__file__).parent
        self._skills: Dict[str, SkillDefinition] = {}
        self._load_skills()
    
    def _load_skills(self):
        """Load all skill YAML files."""
        for yaml_file in self.skills_dir.rglob("*.yaml"):
            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                skill = SkillDefinition(**data)
                self._skills[skill.name] = skill
            except Exception as e:
                logger.warning(f"Failed to load skill {yaml_file}: {e}")
    
    def get_skill(self, name: str) -> Optional[SkillDefinition]:
        """Get a skill by name."""
        return self._skills.get(name)
    
    def find_skills_for_query(self, query: str) -> List[SkillDefinition]:
        """Find skills that match a query based on triggers."""
        matches = []
        query_lower = query.lower()
        
        for skill in self._skills.values():
            # Check keywords
            if any(kw in query_lower for kw in skill.triggers.keywords):
                matches.append(skill)
                continue
            
            # Check patterns
            for pattern in skill.triggers.patterns:
                if re.search(pattern, query_lower):
                    matches.append(skill)
                    break
        
        return matches
    
    def get_skills_by_category(self, category: str) -> List[SkillDefinition]:
        """Get all skills in a category."""
        return [s for s in self._skills.values() if s.category == category]
    
    def get_skill_context(self, skill_name: str) -> str:
        """Get formatted context for LLM consumption."""
        skill = self.get_skill(skill_name)
        if not skill:
            return ""
        
        return f"""
## Skill: {skill.name}

{skill.description}

### Capabilities
{chr(10).join(f"- {cap}" for cap in skill.capabilities)}

### Parameters
Required:
{chr(10).join(f"- {p.name} ({p.type}): {p.description}" for p in skill.parameters.get('required', []))}

Optional:
{chr(10).join(f"- {p.name} ({p.type}, default={p.default}): {p.description}" for p in skill.parameters.get('optional', []))}

### Examples
{chr(10).join(f"- \"{ex.prompt}\"" for ex in skill.examples[:3])}
"""

@lru_cache()
def get_skill_registry() -> SkillRegistry:
    """Get the singleton skill registry."""
    return SkillRegistry()
```

### 1.5 Integration with ChatAgent

Modify `chat_agent.py` to use skill documentation for better tool selection:

```python
# In _is_tool_query method
def _is_tool_query(self, content: str, intent: Optional[str]) -> bool:
    # Use skill registry for more accurate detection
    from config.skills import get_skill_registry
    
    registry = get_skill_registry()
    matching_skills = registry.find_skills_for_query(content)
    
    if matching_skills:
        return True
    
    # Fall back to keyword matching...
```

### 1.6 Files to Create

| File | Description |
|------|-------------|
| `config/skills/__init__.py` | Skill loader and registry |
| `config/skills/schema.py` | Pydantic models |
| `config/skills/data_discovery/encode_search.yaml` | ENCODE search skill |
| `config/skills/data_discovery/geo_search.yaml` | GEO search skill |
| `config/skills/data_discovery/tcga_search.yaml` | TCGA search skill |
| `config/skills/data_discovery/data_scan.yaml` | Local file scan skill |
| `config/skills/workflow_generation/rnaseq_workflow.yaml` | RNA-seq workflow skill |
| `config/skills/workflow_generation/chipseq_workflow.yaml` | ChIP-seq workflow skill |
| `config/skills/job_management/submit_job.yaml` | Job submission skill |
| `config/skills/job_management/check_status.yaml` | Job status skill |
| `config/skills/education/explain_concept.yaml` | Concept explanation skill |
| `tests/test_skill_registry.py` | Unit tests |

### 1.7 Success Criteria

- [x] All existing tools have corresponding skill YAML files
- [x] SkillRegistry can load and query skills
- [ ] ChatAgent uses skill registry for tool detection (optional enhancement)
- [x] Tests pass for skill loading and matching (24 tests passing)
- [x] No regression in existing functionality

**Phase 1 Status: COMPLETE** (Implemented 2024-01)

Files created:
- `config/skills/__init__.py` - SkillRegistry, SkillDefinition, loader
- `config/skills/data_discovery/encode_search.yaml`
- `config/skills/data_discovery/geo_search.yaml`
- `config/skills/data_discovery/tcga_search.yaml`
- `config/skills/data_discovery/data_scan.yaml`
- `config/skills/workflow_generation/rnaseq_workflow.yaml`
- `config/skills/workflow_generation/chipseq_workflow.yaml`
- `config/skills/workflow_generation/methylation_workflow.yaml`
- `config/skills/job_management/submit_job.yaml`
- `config/skills/job_management/check_status.yaml`
- `config/skills/job_management/cancel_job.yaml`
- `config/skills/job_management/get_logs.yaml`
- `config/skills/education/explain_concept.yaml`
- `config/skills/education/compare_samples.yaml`
- `tests/test_skill_registry.py`

Total skills: 25+ (6 database + 4 data discovery + 3 workflow + 4 job management + 2 education + existing tools)

---

## Phase 2: Additional Database Integrations

### 2.1 Objective

Add new database integrations inspired by Claude Scientific Skills' 26+ database access:

| Database | Priority | Use Case |
|----------|----------|----------|
| **UniProt** | High | Protein sequences, annotations, functions |
| **STRING** | High | Protein-protein interactions, networks |
| **KEGG** | High | Pathway enrichment, metabolic pathways |
| **Reactome** | High | Pathway analysis, biological processes |
| **PubMed** | Medium | Literature search, citations |
| **ClinVar** | Medium | Variant pathogenicity |
| **PDB** | Low | Protein structures |

### 2.2 Architecture

```
src/workflow_composer/agents/tools/
├── __init__.py              # Updated with new tools
├── databases/
│   ├── __init__.py
│   ├── base.py              # Base database client
│   ├── uniprot.py           # UniProt API client
│   ├── string_db.py         # STRING API client
│   ├── kegg.py              # KEGG API client
│   ├── reactome.py          # Reactome API client
│   ├── pubmed.py            # PubMed/Entrez client
│   └── clinvar.py           # ClinVar API client
```

### 2.3 Base Database Client

```python
# src/workflow_composer/agents/tools/databases/base.py
"""Base class for database API clients."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import httpx
import logging

logger = logging.getLogger(__name__)

@dataclass
class DatabaseResult:
    """Standardized result from database queries."""
    success: bool
    data: Any
    count: int
    query: str
    source: str
    message: str = ""
    
class DatabaseClient(ABC):
    """Abstract base class for database API clients."""
    
    BASE_URL: str = ""
    NAME: str = ""
    RATE_LIMIT: float = 0.34  # requests per second
    
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self._last_request_time = 0
        self.client = httpx.Client(timeout=timeout)
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        import time
        elapsed = time.time() - self._last_request_time
        if elapsed < (1.0 / self.RATE_LIMIT):
            time.sleep((1.0 / self.RATE_LIMIT) - elapsed)
        self._last_request_time = time.time()
    
    @abstractmethod
    def search(self, query: str, **kwargs) -> DatabaseResult:
        """Search the database."""
        pass
    
    @abstractmethod
    def get_by_id(self, identifier: str) -> DatabaseResult:
        """Get a specific entry by ID."""
        pass
    
    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make a rate-limited request."""
        self._rate_limit()
        return self.client.request(method, url, **kwargs)
```

### 2.4 UniProt Implementation

```python
# src/workflow_composer/agents/tools/databases/uniprot.py
"""UniProt database client for protein information."""

from .base import DatabaseClient, DatabaseResult
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class UniProtClient(DatabaseClient):
    """Client for UniProt REST API."""
    
    BASE_URL = "https://rest.uniprot.org"
    NAME = "UniProt"
    RATE_LIMIT = 10  # UniProt allows higher rate
    
    def search(
        self,
        query: str,
        organism: Optional[str] = None,
        reviewed: bool = True,
        limit: int = 25,
        fields: Optional[List[str]] = None
    ) -> DatabaseResult:
        """
        Search UniProt for proteins.
        
        Args:
            query: Search terms (gene name, protein name, keywords)
            organism: Organism filter (e.g., "human", "9606", "Homo sapiens")
            reviewed: If True, only return Swiss-Prot (reviewed) entries
            limit: Maximum results
            fields: Specific fields to return
            
        Returns:
            DatabaseResult with protein entries
        """
        # Build query
        full_query = query
        if organism:
            organism_id = self._resolve_organism(organism)
            full_query += f" AND organism_id:{organism_id}"
        if reviewed:
            full_query += " AND reviewed:true"
        
        # Default fields
        if fields is None:
            fields = [
                "accession", "id", "protein_name", "gene_names",
                "organism_name", "length", "cc_function",
                "go_p", "go_c", "go_f"
            ]
        
        try:
            response = self._request(
                "GET",
                f"{self.BASE_URL}/uniprotkb/search",
                params={
                    "query": full_query,
                    "format": "json",
                    "size": limit,
                    "fields": ",".join(fields)
                }
            )
            response.raise_for_status()
            data = response.json()
            
            results = data.get("results", [])
            return DatabaseResult(
                success=True,
                data=results,
                count=len(results),
                query=query,
                source=self.NAME,
                message=f"Found {len(results)} proteins"
            )
            
        except Exception as e:
            logger.error(f"UniProt search failed: {e}")
            return DatabaseResult(
                success=False,
                data=[],
                count=0,
                query=query,
                source=self.NAME,
                message=f"Search failed: {e}"
            )
    
    def get_by_id(self, accession: str) -> DatabaseResult:
        """Get protein by UniProt accession."""
        try:
            response = self._request(
                "GET",
                f"{self.BASE_URL}/uniprotkb/{accession}.json"
            )
            response.raise_for_status()
            data = response.json()
            
            return DatabaseResult(
                success=True,
                data=data,
                count=1,
                query=accession,
                source=self.NAME,
                message=f"Retrieved {accession}"
            )
        except Exception as e:
            return DatabaseResult(
                success=False,
                data=None,
                count=0,
                query=accession,
                source=self.NAME,
                message=f"Failed to retrieve {accession}: {e}"
            )
    
    def get_sequence(self, accession: str) -> Optional[str]:
        """Get protein sequence in FASTA format."""
        try:
            response = self._request(
                "GET",
                f"{self.BASE_URL}/uniprotkb/{accession}.fasta"
            )
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Failed to get sequence for {accession}: {e}")
            return None
    
    def _resolve_organism(self, organism: str) -> str:
        """Resolve organism name to taxonomy ID."""
        organism_map = {
            "human": "9606",
            "homo sapiens": "9606",
            "mouse": "10090",
            "mus musculus": "10090",
            "rat": "10116",
            "zebrafish": "7955",
            "drosophila": "7227",
            "yeast": "559292",
            "e. coli": "83333",
        }
        return organism_map.get(organism.lower(), organism)
```

### 2.5 STRING Database Implementation

```python
# src/workflow_composer/agents/tools/databases/string_db.py
"""STRING database client for protein-protein interactions."""

from .base import DatabaseClient, DatabaseResult
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class STRINGClient(DatabaseClient):
    """Client for STRING protein interaction database."""
    
    BASE_URL = "https://string-db.org/api"
    NAME = "STRING"
    RATE_LIMIT = 1  # STRING has lower rate limit
    
    def search(
        self,
        identifiers: List[str],
        species: int = 9606,  # Human
        network_type: str = "functional",
        required_score: int = 400,
        limit: int = 50
    ) -> DatabaseResult:
        """
        Get protein interaction network.
        
        Args:
            identifiers: Gene names or protein IDs
            species: NCBI taxonomy ID (9606=human, 10090=mouse)
            network_type: "functional" or "physical"
            required_score: Minimum interaction score (0-1000)
            limit: Maximum interactions per protein
            
        Returns:
            DatabaseResult with interaction network
        """
        try:
            response = self._request(
                "POST",
                f"{self.BASE_URL}/json/network",
                data={
                    "identifiers": "\r".join(identifiers),
                    "species": species,
                    "network_type": network_type,
                    "required_score": required_score,
                    "limit": limit
                }
            )
            response.raise_for_status()
            interactions = response.json()
            
            return DatabaseResult(
                success=True,
                data=interactions,
                count=len(interactions),
                query=", ".join(identifiers[:5]),
                source=self.NAME,
                message=f"Found {len(interactions)} interactions"
            )
            
        except Exception as e:
            logger.error(f"STRING search failed: {e}")
            return DatabaseResult(
                success=False,
                data=[],
                count=0,
                query=", ".join(identifiers[:5]),
                source=self.NAME,
                message=f"Search failed: {e}"
            )
    
    def get_enrichment(
        self,
        identifiers: List[str],
        species: int = 9606,
        background_count: Optional[int] = None
    ) -> DatabaseResult:
        """
        Get functional enrichment for gene list.
        
        Args:
            identifiers: Gene names
            species: NCBI taxonomy ID
            background_count: Size of background gene set
            
        Returns:
            DatabaseResult with enrichment results (GO, KEGG, etc.)
        """
        try:
            params = {
                "identifiers": "\r".join(identifiers),
                "species": species
            }
            if background_count:
                params["background_count"] = background_count
            
            response = self._request(
                "POST",
                f"{self.BASE_URL}/json/enrichment",
                data=params
            )
            response.raise_for_status()
            enrichment = response.json()
            
            return DatabaseResult(
                success=True,
                data=enrichment,
                count=len(enrichment),
                query=f"{len(identifiers)} genes",
                source=f"{self.NAME} Enrichment",
                message=f"Found {len(enrichment)} enriched terms"
            )
            
        except Exception as e:
            logger.error(f"STRING enrichment failed: {e}")
            return DatabaseResult(
                success=False,
                data=[],
                count=0,
                query=f"{len(identifiers)} genes",
                source=f"{self.NAME} Enrichment",
                message=f"Enrichment failed: {e}"
            )
    
    def get_by_id(self, identifier: str, species: int = 9606) -> DatabaseResult:
        """Get protein info by identifier."""
        return self.search([identifier], species=species, limit=1)
```

### 2.6 KEGG Implementation

```python
# src/workflow_composer/agents/tools/databases/kegg.py
"""KEGG database client for pathway information."""

from .base import DatabaseClient, DatabaseResult
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class KEGGClient(DatabaseClient):
    """Client for KEGG REST API."""
    
    BASE_URL = "https://rest.kegg.jp"
    NAME = "KEGG"
    RATE_LIMIT = 3  # KEGG rate limit
    
    def search(
        self,
        query: str,
        database: str = "pathway",
        organism: str = "hsa"
    ) -> DatabaseResult:
        """
        Search KEGG database.
        
        Args:
            query: Search terms
            database: "pathway", "module", "compound", "drug", etc.
            organism: KEGG organism code (hsa=human, mmu=mouse)
            
        Returns:
            DatabaseResult with matching entries
        """
        try:
            response = self._request(
                "GET",
                f"{self.BASE_URL}/find/{database}/{query}"
            )
            response.raise_for_status()
            
            # Parse KEGG text format
            results = []
            for line in response.text.strip().split("\n"):
                if line:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        results.append({
                            "id": parts[0],
                            "name": parts[1] if len(parts) > 1 else ""
                        })
            
            return DatabaseResult(
                success=True,
                data=results,
                count=len(results),
                query=query,
                source=self.NAME,
                message=f"Found {len(results)} entries in {database}"
            )
            
        except Exception as e:
            logger.error(f"KEGG search failed: {e}")
            return DatabaseResult(
                success=False,
                data=[],
                count=0,
                query=query,
                source=self.NAME,
                message=f"Search failed: {e}"
            )
    
    def get_pathway(self, pathway_id: str) -> DatabaseResult:
        """Get pathway details."""
        try:
            response = self._request(
                "GET",
                f"{self.BASE_URL}/get/{pathway_id}"
            )
            response.raise_for_status()
            
            return DatabaseResult(
                success=True,
                data=self._parse_kegg_entry(response.text),
                count=1,
                query=pathway_id,
                source=self.NAME,
                message=f"Retrieved pathway {pathway_id}"
            )
            
        except Exception as e:
            return DatabaseResult(
                success=False,
                data=None,
                count=0,
                query=pathway_id,
                source=self.NAME,
                message=f"Failed: {e}"
            )
    
    def get_pathway_genes(self, pathway_id: str) -> List[str]:
        """Get genes in a pathway."""
        result = self.get_pathway(pathway_id)
        if result.success and result.data:
            return result.data.get("genes", [])
        return []
    
    def get_by_id(self, identifier: str) -> DatabaseResult:
        """Get entry by KEGG ID."""
        return self.get_pathway(identifier)
    
    def _parse_kegg_entry(self, text: str) -> dict:
        """Parse KEGG flat file format."""
        entry = {}
        current_field = None
        
        for line in text.split("\n"):
            if line.startswith(" "):
                # Continuation of previous field
                if current_field:
                    if isinstance(entry[current_field], list):
                        entry[current_field].append(line.strip())
                    else:
                        entry[current_field] += " " + line.strip()
            elif line:
                parts = line.split(None, 1)
                if parts:
                    current_field = parts[0].lower()
                    value = parts[1] if len(parts) > 1 else ""
                    
                    if current_field in ["gene", "compound"]:
                        entry[current_field + "s"] = [value]
                    else:
                        entry[current_field] = value
        
        return entry
```

### 2.7 Tool Registration

```python
# Update src/workflow_composer/agents/tools/__init__.py

# Add new database tools
from .databases import (
    UniProtClient,
    STRINGClient,
    KEGGClient,
    ReactomeClient,
    PubMedClient,
)

# Singleton instances
_uniprot_client = None
_string_client = None
_kegg_client = None

def get_uniprot_client():
    global _uniprot_client
    if _uniprot_client is None:
        _uniprot_client = UniProtClient()
    return _uniprot_client

# Add to TOOL_REGISTRY
TOOL_REGISTRY = {
    # ... existing tools ...
    
    # New database tools
    "search_uniprot": lambda **kw: get_uniprot_client().search(**kw),
    "search_string": lambda **kw: get_string_client().search(**kw),
    "get_enrichment": lambda **kw: get_string_client().get_enrichment(**kw),
    "search_kegg": lambda **kw: get_kegg_client().search(**kw),
    "get_pathway": lambda **kw: get_kegg_client().get_pathway(**kw),
    "search_pubmed": lambda **kw: get_pubmed_client().search(**kw),
}
```

### 2.8 Files to Create

| File | Description |
|------|-------------|
| `src/workflow_composer/agents/tools/databases/__init__.py` | Package init |
| `src/workflow_composer/agents/tools/databases/base.py` | Base client |
| `src/workflow_composer/agents/tools/databases/uniprot.py` | UniProt client |
| `src/workflow_composer/agents/tools/databases/string_db.py` | STRING client |
| `src/workflow_composer/agents/tools/databases/kegg.py` | KEGG client |
| `src/workflow_composer/agents/tools/databases/reactome.py` | Reactome client |
| `src/workflow_composer/agents/tools/databases/pubmed.py` | PubMed client |
| `config/skills/databases/uniprot.yaml` | UniProt skill |
| `config/skills/databases/string.yaml` | STRING skill |
| `config/skills/databases/kegg.yaml` | KEGG skill |
| `tests/test_database_clients.py` | Unit tests |

### 2.9 Success Criteria

- [x] All 5 high/medium priority databases implemented (UniProt, STRING, KEGG, Reactome, PubMed, ClinVar)
- [x] Each client has search and get_by_id methods
- [x] Rate limiting works correctly
- [x] Error handling is robust
- [x] Corresponding skill YAML files created (config/skills/*.yaml)
- [x] Integration tests pass (52 tests, 49 passed, 3 skipped integration tests)
- [x] No regression in existing tests

**Phase 2 Status: COMPLETE** (Implemented 2024-01)

Files created:
- `src/workflow_composer/agents/tools/databases/__init__.py`
- `src/workflow_composer/agents/tools/databases/base.py`
- `src/workflow_composer/agents/tools/databases/uniprot.py`
- `src/workflow_composer/agents/tools/databases/string_db.py`
- `src/workflow_composer/agents/tools/databases/kegg.py`
- `src/workflow_composer/agents/tools/databases/reactome.py`
- `src/workflow_composer/agents/tools/databases/pubmed.py`
- `src/workflow_composer/agents/tools/databases/clinvar.py`
- `config/skills/uniprot.yaml`
- `config/skills/string_db.yaml`
- `config/skills/kegg.yaml`
- `config/skills/reactome.yaml`
- `config/skills/pubmed.yaml`
- `config/skills/clinvar.yaml`
- `tests/test_database_clients.py`

AgentTools integration:
- 13 new tools added (search_uniprot, get_protein, search_string, get_interactions, get_enrichment, search_kegg, get_pathway, search_reactome, analyze_genes, search_pubmed, get_article, search_clinvar, get_variants)
- OpenAI function definitions added for all database tools
- Pattern detection integrated for natural language queries

---

## Phase 3: Workflow Templates

### 3.1 Objective

Create pre-built workflow templates that users can invoke with minimal customization, similar to Claude Scientific Skills' multi-step workflow prompts.

### 3.2 Architecture

```
config/
└── workflow_templates/
    ├── __init__.py              # Template loader
    ├── schema.py                # Template models
    │
    ├── rnaseq/
    │   ├── basic_de.yaml        # Basic differential expression
    │   ├── full_analysis.yaml   # Complete RNA-seq pipeline
    │   └── custom_genome.yaml   # With custom reference
    │
    ├── chipseq/
    │   ├── peak_calling.yaml    # Basic peak calling
    │   ├── differential.yaml    # Differential binding
    │   └── motif_analysis.yaml  # With motif discovery
    │
    ├── methylation/
    │   ├── wgbs_analysis.yaml   # Whole genome bisulfite
    │   └── dmr_calling.yaml     # Differential methylation
    │
    ├── variant/
    │   ├── germline.yaml        # Germline variant calling
    │   └── somatic.yaml         # Tumor/normal somatic
    │
    └── multiomics/
        ├── rnaseq_chipseq.yaml  # Integrated RNA + ChIP
        └── multi_sample.yaml    # Multi-sample integration
```

### 3.3 Template Schema

```yaml
# config/workflow_templates/rnaseq/full_analysis.yaml
---
name: complete_rnaseq_analysis
version: "1.0.0"
display_name: "Complete RNA-seq Analysis Pipeline"
description: |
  End-to-end RNA-seq analysis from raw FASTQ files to differential
  expression results with quality control at each step.

# Template category and tags
category: rnaseq
tags:
  - differential-expression
  - quality-control
  - full-pipeline

# Required inputs from user
inputs:
  required:
    - name: input_dir
      type: path
      description: Directory containing FASTQ files
      
    - name: sample_sheet
      type: file
      description: CSV with sample metadata (sample_id, condition, replicate)
      
    - name: organism
      type: string
      enum: [human, mouse, rat]
      description: Reference organism
      
    - name: comparisons
      type: list
      description: Condition comparisons for DE analysis
      example: ["treated_vs_control", "timepoint2_vs_timepoint1"]
      
  optional:
    - name: strandedness
      type: string
      enum: [unstranded, forward, reverse]
      default: reverse
      description: Library strandedness
      
    - name: aligner
      type: string
      enum: [STAR, hisat2, salmon]
      default: STAR
      description: Alignment tool
      
    - name: de_tool
      type: string
      enum: [DESeq2, edgeR, limma]
      default: DESeq2
      description: Differential expression tool

# Pipeline steps
steps:
  - name: quality_control
    description: "Initial QC with FastQC"
    tool: fastqc
    inputs:
      reads: "${input_dir}/*.fastq.gz"
    outputs:
      qc_reports: "${output_dir}/qc/fastqc/"
      
  - name: trimming
    description: "Adapter trimming with Trim Galore"
    tool: trim_galore
    inputs:
      reads: "${input_dir}/*.fastq.gz"
    outputs:
      trimmed_reads: "${output_dir}/trimmed/"
    parameters:
      quality: 20
      length: 20
      
  - name: alignment
    description: "Align reads to reference genome"
    tool: "${aligner}"
    inputs:
      reads: "${steps.trimming.outputs.trimmed_reads}"
      genome: "${reference_dir}/${organism}/genome"
    outputs:
      bam_files: "${output_dir}/aligned/"
    parameters:
      threads: 8
      
  - name: quantification
    description: "Count reads per gene"
    tool: featurecounts
    inputs:
      bam: "${steps.alignment.outputs.bam_files}"
      annotation: "${reference_dir}/${organism}/annotation.gtf"
    outputs:
      counts: "${output_dir}/counts/counts.txt"
    parameters:
      strandedness: "${strandedness}"
      
  - name: differential_expression
    description: "Differential expression analysis"
    tool: "${de_tool}"
    inputs:
      counts: "${steps.quantification.outputs.counts}"
      sample_info: "${sample_sheet}"
    outputs:
      results: "${output_dir}/de_results/"
    parameters:
      comparisons: "${comparisons}"
      fdr_threshold: 0.05
      log2fc_threshold: 1.0
      
  - name: multiqc_report
    description: "Aggregate QC metrics"
    tool: multiqc
    inputs:
      dirs:
        - "${steps.quality_control.outputs.qc_reports}"
        - "${steps.alignment.outputs.bam_files}"
    outputs:
      report: "${output_dir}/multiqc_report.html"

# Output structure
outputs:
  qc_reports: "Quality control reports for each sample"
  aligned_bams: "Aligned and sorted BAM files"
  count_matrix: "Gene-level count matrix"
  de_results: "Differential expression tables per comparison"
  multiqc_report: "Aggregated QC report"

# Example usage
examples:
  - prompt: |
      Run complete RNA-seq analysis on my mouse samples.
      Input: /data/rnaseq/fastq/
      Sample sheet: /data/rnaseq/samples.csv
      Comparisons: KO_vs_WT
    parameters:
      input_dir: "/data/rnaseq/fastq/"
      sample_sheet: "/data/rnaseq/samples.csv"
      organism: "mouse"
      comparisons: ["KO_vs_WT"]

# Resource requirements
resources:
  memory: "32GB"
  cpus: 8
  time: "8h"
  gpu: false

# Related templates
related_templates:
  - basic_de
  - custom_genome
```

### 3.4 Template Engine

```python
# config/workflow_templates/__init__.py
"""
Workflow Template System

Pre-built workflow templates for common analysis patterns.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
from pydantic import BaseModel, Field
from jinja2 import Environment, BaseLoader

class TemplateInput(BaseModel):
    name: str
    type: str
    description: str
    enum: Optional[List[str]] = None
    default: Optional[Any] = None
    example: Optional[Any] = None

class TemplateStep(BaseModel):
    name: str
    description: str
    tool: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    parameters: Dict[str, Any] = Field(default_factory=dict)

class WorkflowTemplate(BaseModel):
    name: str
    version: str
    display_name: str
    description: str
    category: str
    tags: List[str]
    inputs: Dict[str, List[TemplateInput]]
    steps: List[TemplateStep]
    outputs: Dict[str, str]
    examples: List[Dict[str, Any]]
    resources: Dict[str, Any]
    related_templates: List[str] = Field(default_factory=list)

class TemplateEngine:
    """Engine for loading and rendering workflow templates."""
    
    def __init__(self, templates_dir: Path = None):
        self.templates_dir = templates_dir or Path(__file__).parent
        self._templates: Dict[str, WorkflowTemplate] = {}
        self._jinja_env = Environment(loader=BaseLoader())
        self._load_templates()
    
    def _load_templates(self):
        """Load all template YAML files."""
        for yaml_file in self.templates_dir.rglob("*.yaml"):
            if yaml_file.name == "schema.py":
                continue
            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                template = WorkflowTemplate(**data)
                self._templates[template.name] = template
            except Exception as e:
                logger.warning(f"Failed to load template {yaml_file}: {e}")
    
    def get_template(self, name: str) -> Optional[WorkflowTemplate]:
        """Get a template by name."""
        return self._templates.get(name)
    
    def list_templates(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[WorkflowTemplate]:
        """List available templates, optionally filtered."""
        templates = list(self._templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        if tags:
            templates = [
                t for t in templates 
                if any(tag in t.tags for tag in tags)
            ]
        
        return templates
    
    def render_template(
        self,
        template_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Render a template with user parameters.
        
        Returns a workflow configuration ready for generation.
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")
        
        # Validate required parameters
        for input_def in template.inputs.get("required", []):
            if input_def.name not in parameters:
                raise ValueError(f"Missing required parameter: {input_def.name}")
        
        # Apply defaults for optional parameters
        for input_def in template.inputs.get("optional", []):
            if input_def.name not in parameters and input_def.default is not None:
                parameters[input_def.name] = input_def.default
        
        # Render template variables
        rendered_steps = []
        step_outputs = {}
        
        for step in template.steps:
            rendered_step = {
                "name": step.name,
                "description": step.description,
                "tool": self._render_value(step.tool, parameters, step_outputs),
                "inputs": {},
                "outputs": {},
                "parameters": {}
            }
            
            # Render inputs
            for key, value in step.inputs.items():
                rendered_step["inputs"][key] = self._render_value(
                    value, parameters, step_outputs
                )
            
            # Render outputs
            for key, value in step.outputs.items():
                rendered_value = self._render_value(value, parameters, step_outputs)
                rendered_step["outputs"][key] = rendered_value
                step_outputs[f"steps.{step.name}.outputs.{key}"] = rendered_value
            
            # Render parameters
            for key, value in step.parameters.items():
                rendered_step["parameters"][key] = self._render_value(
                    value, parameters, step_outputs
                )
            
            rendered_steps.append(rendered_step)
        
        return {
            "template": template_name,
            "version": template.version,
            "category": template.category,
            "parameters": parameters,
            "steps": rendered_steps,
            "resources": template.resources
        }
    
    def _render_value(
        self,
        value: Any,
        parameters: Dict[str, Any],
        step_outputs: Dict[str, Any]
    ) -> Any:
        """Render a template value with variable substitution."""
        if not isinstance(value, str):
            return value
        
        # Simple variable substitution
        context = {**parameters, **step_outputs}
        
        # Replace ${var} patterns
        import re
        pattern = r'\$\{([^}]+)\}'
        
        def replace(match):
            var_name = match.group(1)
            if var_name in context:
                return str(context[var_name])
            return match.group(0)
        
        return re.sub(pattern, replace, value)
    
    def generate_workflow(
        self,
        template_name: str,
        parameters: Dict[str, Any],
        output_format: str = "nextflow"
    ) -> str:
        """
        Generate actual workflow code from template.
        
        Args:
            template_name: Name of template to use
            parameters: User-provided parameters
            output_format: "nextflow" or "snakemake"
            
        Returns:
            Generated workflow code as string
        """
        config = self.render_template(template_name, parameters)
        
        # Use BioPipelines workflow generator
        from workflow_composer import BioPipelines
        bp = BioPipelines()
        
        # Convert template config to workflow
        workflow = bp.compose(
            analysis_type=config["category"],
            organism=parameters.get("organism", "human"),
            steps=config["steps"]
        )
        
        return workflow

# Singleton
_template_engine = None

def get_template_engine() -> TemplateEngine:
    global _template_engine
    if _template_engine is None:
        _template_engine = TemplateEngine()
    return _template_engine
```

### 3.5 Files to Create

| File | Description |
|------|-------------|
| `config/workflow_templates/__init__.py` | Template engine |
| `config/workflow_templates/schema.py` | Pydantic models |
| `config/workflow_templates/rnaseq/basic_de.yaml` | Basic DE template |
| `config/workflow_templates/rnaseq/full_analysis.yaml` | Full RNA-seq |
| `config/workflow_templates/chipseq/peak_calling.yaml` | ChIP-seq peaks |
| `config/workflow_templates/chipseq/differential.yaml` | Differential binding |
| `config/workflow_templates/methylation/wgbs_analysis.yaml` | WGBS analysis |
| `config/workflow_templates/variant/germline.yaml` | Germline variants |
| `tests/test_workflow_templates.py` | Unit tests |

### 3.6 Success Criteria

- [ ] Template engine loads and validates templates
- [ ] Variable substitution works correctly
- [ ] Templates can generate actual Nextflow/Snakemake code
- [ ] At least 6 templates created (2 per major category)
- [ ] ChatAgent can invoke templates via natural language
- [ ] Tests pass

---

## Phase 4: MCP Server (Optional)

### 4.1 Objective

Expose BioPipelines tools via Model Context Protocol (MCP) for integration with Claude Code, Cursor, and other MCP-compatible clients.

### 4.2 Architecture

```
src/workflow_composer/
└── mcp/
    ├── __init__.py
    ├── server.py          # MCP server implementation
    ├── tools.py           # Tool definitions for MCP
    └── resources.py       # Resource definitions
```

### 4.3 MCP Server Implementation

```python
# src/workflow_composer/mcp/server.py
"""
BioPipelines MCP Server

Exposes BioPipelines tools via Model Context Protocol.
"""

from mcp.server import Server, Tool, Resource
from mcp.types import TextContent, ImageContent
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)

class BioPipelinesMCPServer:
    """MCP Server exposing BioPipelines capabilities."""
    
    def __init__(self):
        self.server = Server("biopipelines")
        self._setup_tools()
        self._setup_resources()
    
    def _setup_tools(self):
        """Register all BioPipelines tools with MCP."""
        
        @self.server.tool("search_encode")
        async def search_encode(
            query: str,
            assay_type: str = None,
            organism: str = "human",
            limit: int = 10
        ) -> TextContent:
            """
            Search ENCODE database for experiments.
            
            Args:
                query: Search terms (cell type, target, etc.)
                assay_type: Filter by assay (ChIP-seq, RNA-seq, etc.)
                organism: Filter by organism (human, mouse)
                limit: Maximum results
            """
            from workflow_composer import BioPipelines
            bp = BioPipelines()
            
            result = bp.tools.search_data(
                query=query,
                source="ENCODE",
                assay_type=assay_type,
                organism=organism,
                limit=limit
            )
            
            return TextContent(text=format_search_results(result))
        
        @self.server.tool("search_geo")
        async def search_geo(
            query: str,
            organism: str = None,
            limit: int = 10
        ) -> TextContent:
            """Search GEO database for datasets."""
            from workflow_composer import BioPipelines
            bp = BioPipelines()
            
            result = bp.tools.search_data(
                query=query,
                source="GEO",
                organism=organism,
                limit=limit
            )
            
            return TextContent(text=format_search_results(result))
        
        @self.server.tool("create_workflow")
        async def create_workflow(
            analysis_type: str,
            organism: str = "human",
            input_dir: str = None,
            output_dir: str = None
        ) -> TextContent:
            """
            Generate a bioinformatics workflow.
            
            Args:
                analysis_type: Type of analysis (rnaseq, chipseq, etc.)
                organism: Reference organism
                input_dir: Input data directory
                output_dir: Output directory
            """
            from workflow_composer import BioPipelines
            bp = BioPipelines()
            
            workflow = bp.compose(
                analysis_type=analysis_type,
                organism=organism,
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            return TextContent(text=f"Generated workflow:\n\n```nextflow\n{workflow}\n```")
        
        @self.server.tool("explain_concept")
        async def explain_concept(concept: str) -> TextContent:
            """
            Explain a bioinformatics concept.
            
            Args:
                concept: The concept to explain (DESeq2, peak calling, etc.)
            """
            from workflow_composer import BioPipelines
            bp = BioPipelines()
            
            result = bp.tools.explain_concept(concept=concept)
            return TextContent(text=result.message)
        
        @self.server.tool("search_uniprot")
        async def search_uniprot(
            query: str,
            organism: str = "human",
            reviewed: bool = True,
            limit: int = 25
        ) -> TextContent:
            """
            Search UniProt for proteins.
            
            Args:
                query: Gene name, protein name, or keywords
                organism: Filter by organism
                reviewed: Only return reviewed entries
                limit: Maximum results
            """
            from workflow_composer.agents.tools.databases import get_uniprot_client
            client = get_uniprot_client()
            
            result = client.search(
                query=query,
                organism=organism,
                reviewed=reviewed,
                limit=limit
            )
            
            return TextContent(text=format_protein_results(result))
        
        @self.server.tool("get_enrichment")
        async def get_enrichment(
            genes: List[str],
            organism: str = "human"
        ) -> TextContent:
            """
            Get functional enrichment for a gene list.
            
            Args:
                genes: List of gene names
                organism: Organism for enrichment
            """
            from workflow_composer.agents.tools.databases import get_string_client
            client = get_string_client()
            
            species_map = {"human": 9606, "mouse": 10090}
            result = client.get_enrichment(
                identifiers=genes,
                species=species_map.get(organism, 9606)
            )
            
            return TextContent(text=format_enrichment_results(result))
    
    def _setup_resources(self):
        """Register resources (documentation, templates, etc.)."""
        
        @self.server.resource("skills")
        async def get_skills() -> TextContent:
            """Get list of available BioPipelines skills."""
            from config.skills import get_skill_registry
            registry = get_skill_registry()
            
            skills_text = "# Available BioPipelines Skills\n\n"
            for category in ["data_discovery", "workflow_generation", "databases"]:
                skills = registry.get_skills_by_category(category)
                skills_text += f"## {category.replace('_', ' ').title()}\n\n"
                for skill in skills:
                    skills_text += f"- **{skill.name}**: {skill.description[:100]}...\n"
                skills_text += "\n"
            
            return TextContent(text=skills_text)
        
        @self.server.resource("templates")
        async def get_templates() -> TextContent:
            """Get list of available workflow templates."""
            from config.workflow_templates import get_template_engine
            engine = get_template_engine()
            
            templates_text = "# Available Workflow Templates\n\n"
            for template in engine.list_templates():
                templates_text += f"## {template.display_name}\n"
                templates_text += f"{template.description}\n\n"
                templates_text += f"**Category**: {template.category}\n"
                templates_text += f"**Tags**: {', '.join(template.tags)}\n\n"
            
            return TextContent(text=templates_text)
    
    async def run(self, transport: str = "stdio"):
        """Run the MCP server."""
        if transport == "stdio":
            await self.server.run_stdio()
        elif transport == "http":
            await self.server.run_http(host="0.0.0.0", port=8080)


def format_search_results(result) -> str:
    """Format search results for display."""
    if not result.success:
        return f"Search failed: {result.message}"
    
    text = f"Found {result.count} results:\n\n"
    for item in result.data[:10]:
        text += f"- **{item.get('id', 'N/A')}**: {item.get('title', item.get('name', 'N/A'))}\n"
    
    return text


def format_protein_results(result) -> str:
    """Format UniProt results for display."""
    if not result.success:
        return f"Search failed: {result.message}"
    
    text = f"Found {result.count} proteins:\n\n"
    for protein in result.data[:10]:
        accession = protein.get("primaryAccession", "N/A")
        name = protein.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "N/A")
        gene = protein.get("genes", [{}])[0].get("geneName", {}).get("value", "N/A")
        
        text += f"- **{accession}** ({gene}): {name}\n"
    
    return text


def format_enrichment_results(result) -> str:
    """Format enrichment results for display."""
    if not result.success:
        return f"Enrichment failed: {result.message}"
    
    text = f"Found {result.count} enriched terms:\n\n"
    
    # Group by category
    categories = {}
    for term in result.data[:50]:
        cat = term.get("category", "Other")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(term)
    
    for cat, terms in categories.items():
        text += f"## {cat}\n"
        for term in terms[:5]:
            desc = term.get("description", term.get("term", "N/A"))
            pvalue = term.get("p_value", term.get("fdr", "N/A"))
            text += f"- {desc} (p={pvalue:.2e})\n"
        text += "\n"
    
    return text


# Entry point for running server
async def main():
    server = BioPipelinesMCPServer()
    await server.run()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### 4.4 MCP Configuration

```json
// mcp-config.json (for Claude Code / Cursor)
{
  "mcpServers": {
    "biopipelines": {
      "command": "python",
      "args": ["-m", "workflow_composer.mcp.server"],
      "env": {
        "PYTHONPATH": "/path/to/BioPipelines/src"
      }
    }
  }
}
```

### 4.5 Files to Create

| File | Description |
|------|-------------|
| `src/workflow_composer/mcp/__init__.py` | Package init |
| `src/workflow_composer/mcp/server.py` | MCP server |
| `src/workflow_composer/mcp/tools.py` | Tool definitions |
| `mcp-config.json` | Config for clients |
| `docs/MCP_INTEGRATION.md` | Documentation |
| `tests/test_mcp_server.py` | Unit tests |

### 4.6 Success Criteria

- [ ] MCP server starts and runs
- [ ] All major tools exposed via MCP
- [ ] Works with Claude Code (manual test)
- [ ] Works with Cursor IDE (manual test)
- [ ] Documentation complete

---

## Implementation Order

### Session 1: Phase 1 - Skill Documentation (Foundation)
1. Create `config/skills/` directory structure
2. Implement `schema.py` with Pydantic models
3. Implement `__init__.py` with SkillRegistry
4. Create skill YAML files for existing tools
5. Write tests
6. Integrate with ChatAgent

### Session 2: Phase 2A - Database Clients (UniProt, STRING)
1. Create `databases/` package structure
2. Implement `base.py` with DatabaseClient
3. Implement `uniprot.py`
4. Implement `string_db.py`
5. Write tests
6. Create corresponding skill YAML files

### Session 3: Phase 2B - Database Clients (KEGG, Reactome, PubMed)
1. Implement `kegg.py`
2. Implement `reactome.py`
3. Implement `pubmed.py`
4. Register tools in `__init__.py`
5. Write tests
6. Create skill YAML files

### Session 4: Phase 3 - Workflow Templates
1. Create `workflow_templates/` directory structure
2. Implement template engine
3. Create RNA-seq templates
4. Create ChIP-seq templates
5. Write tests
6. Integrate with ChatAgent

### Session 5: Phase 4 - MCP Server (Optional)
1. Create `mcp/` package
2. Implement server.py
3. Add all tool definitions
4. Test with Claude Code
5. Write documentation

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Database API changes | Version pin API clients, implement fallbacks |
| Rate limiting | Built-in rate limiting in base client |
| Breaking existing tests | Run full test suite after each phase |
| MCP protocol changes | Pin MCP library version |
| Performance impact | Lazy loading, caching |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| New database integrations | 5+ |
| Skill YAML files | 15+ |
| Workflow templates | 8+ |
| Test coverage | Maintain 44%+ |
| All existing tests pass | 1246+ |

---

## Appendix: Dependencies to Add

```txt
# requirements.txt additions
httpx>=0.24.0          # Async HTTP client for databases
pydantic>=2.0          # Already have, ensure version
pyyaml>=6.0            # Already have
jinja2>=3.0            # Template rendering
mcp>=0.1.0             # MCP protocol (Phase 4 only)
```

---

*Document created: December 2, 2025*
*Author: BioPipelines Development Team*
