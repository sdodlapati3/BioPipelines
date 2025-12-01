"""
Provisioning Module
===================

Automatic provisioning of resources:
- Reference genomes (Ensembl, UCSC, GENCODE)
- Aligner indices (STAR, BWA, Bowtie2, etc.)
- Container images (Singularity/Docker)

Used by PreflightValidator to auto-fix missing resources.
"""

from .reference_manager import (
    ReferenceManager,
    ReferenceGenome,
    REFERENCE_CATALOG,
    get_reference_manager,
)

from .container_manager import (
    ContainerManager,
    CONTAINER_REGISTRY,
    get_container_manager,
)

__all__ = [
    # Reference Management
    "ReferenceManager",
    "ReferenceGenome",
    "REFERENCE_CATALOG",
    "get_reference_manager",
    # Container Management
    "ContainerManager",
    "CONTAINER_REGISTRY",
    "get_container_manager",
]
