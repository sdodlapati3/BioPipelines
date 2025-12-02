"""
BioPipelines Database Clients
=============================

API clients for biological databases:
- UniProt: Protein sequences, annotations, functions
- STRING: Protein-protein interactions, networks
- KEGG: Pathway enrichment, metabolic pathways
- Reactome: Pathway analysis, biological processes
- PubMed: Literature search, citations
- ClinVar: Variant pathogenicity

Usage:
    from workflow_composer.agents.tools.databases import (
        UniProtClient,
        STRINGClient,
        KEGGClient,
        ReactomeClient,
        PubMedClient,
        ClinVarClient,
    )
    
    # Search UniProt for a protein
    client = UniProtClient()
    result = client.search("BRCA1", organism="human")
    
    # Get protein interactions from STRING
    string = STRINGClient()
    interactions = string.get_interactions(["TP53", "MDM2"])
"""

import logging
import threading
from typing import Optional

from .base import DatabaseClient, DatabaseResult
from .uniprot import UniProtClient
from .string_db import STRINGClient
from .kegg import KEGGClient
from .reactome import ReactomeClient
from .pubmed import PubMedClient
from .clinvar import ClinVarClient

logger = logging.getLogger(__name__)

# =============================================================================
# Singleton Instances
# =============================================================================

_uniprot_client: Optional[UniProtClient] = None
_string_client: Optional[STRINGClient] = None
_kegg_client: Optional[KEGGClient] = None
_reactome_client: Optional[ReactomeClient] = None
_pubmed_client: Optional[PubMedClient] = None
_clinvar_client: Optional[ClinVarClient] = None
_lock = threading.Lock()


def get_uniprot_client() -> UniProtClient:
    """Get singleton UniProt client."""
    global _uniprot_client
    with _lock:
        if _uniprot_client is None:
            _uniprot_client = UniProtClient()
        return _uniprot_client


def get_string_client() -> STRINGClient:
    """Get singleton STRING client."""
    global _string_client
    with _lock:
        if _string_client is None:
            _string_client = STRINGClient()
        return _string_client


def get_kegg_client() -> KEGGClient:
    """Get singleton KEGG client."""
    global _kegg_client
    with _lock:
        if _kegg_client is None:
            _kegg_client = KEGGClient()
        return _kegg_client


def get_reactome_client() -> ReactomeClient:
    """Get singleton Reactome client."""
    global _reactome_client
    with _lock:
        if _reactome_client is None:
            _reactome_client = ReactomeClient()
        return _reactome_client


def get_pubmed_client() -> PubMedClient:
    """Get singleton PubMed client."""
    global _pubmed_client
    with _lock:
        if _pubmed_client is None:
            _pubmed_client = PubMedClient()
        return _pubmed_client


def get_clinvar_client() -> ClinVarClient:
    """Get singleton ClinVar client."""
    global _clinvar_client
    with _lock:
        if _clinvar_client is None:
            _clinvar_client = ClinVarClient()
        return _clinvar_client


def reset_clients():
    """Reset all singleton clients (for testing)."""
    global _uniprot_client, _string_client, _kegg_client
    global _reactome_client, _pubmed_client, _clinvar_client
    with _lock:
        _uniprot_client = None
        _string_client = None
        _kegg_client = None
        _reactome_client = None
        _pubmed_client = None
        _clinvar_client = None


__all__ = [
    # Base types
    "DatabaseClient",
    "DatabaseResult",
    
    # Clients
    "UniProtClient",
    "STRINGClient",
    "KEGGClient",
    "ReactomeClient",
    "PubMedClient",
    "ClinVarClient",
    
    # Singleton accessors
    "get_uniprot_client",
    "get_string_client",
    "get_kegg_client",
    "get_reactome_client",
    "get_pubmed_client",
    "get_clinvar_client",
    "reset_clients",
]
