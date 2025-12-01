#!/usr/bin/env python3
"""
Test Phase 2.3: Auto-Provisioning

Tests:
- Reference catalog lookup
- Container registry lookup
- Manager initialization
- Reference resolution
"""

import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_reference_catalog():
    """Test reference catalog contains expected entries."""
    from workflow_composer.provisioning.reference_manager import (
        REFERENCE_CATALOG,
        ORGANISM_ALIASES,
    )
    
    # Check expected organisms exist
    expected_organisms = ["human", "mouse", "rat", "zebrafish", "yeast"]
    found = set()
    
    for key, ref in REFERENCE_CATALOG.items():
        found.add(ref.organism)
    
    missing = set(expected_organisms) - found
    assert not missing, f"Missing organisms: {missing}"
    
    # Check human GRCh38
    human_keys = [k for k in REFERENCE_CATALOG if k.startswith("human_GRCh38")]
    assert len(human_keys) >= 1, "Should have human GRCh38 reference"
    
    # Check aliases
    assert ORGANISM_ALIASES["homo_sapiens"] == "human"
    assert ORGANISM_ALIASES["mus_musculus"] == "mouse"
    
    print("✅ Reference catalog: PASS")
    return True


def test_container_registry():
    """Test container registry contains expected entries."""
    from workflow_composer.provisioning.container_manager import (
        CONTAINER_REGISTRY,
        TOOL_CONTAINER_MAP,
    )
    
    # Check expected containers
    expected = ["base", "rna-seq", "chip-seq", "dna-seq"]
    for c in expected:
        assert c in CONTAINER_REGISTRY, f"Missing container: {c}"
    
    # Check tool mappings
    assert TOOL_CONTAINER_MAP.get("star") == "rna-seq"
    assert TOOL_CONTAINER_MAP.get("bwa") == "dna-seq"
    assert TOOL_CONTAINER_MAP.get("macs2") == "chip-seq"
    
    print("✅ Container registry: PASS")
    return True


def test_reference_manager_init():
    """Test reference manager initialization."""
    from workflow_composer.provisioning import ReferenceManager
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = ReferenceManager(base_path=tmpdir)
        
        assert mgr.base_path.exists()
        
        # Test organism normalization
        assert mgr._normalize_organism("Homo sapiens") == "human"
        assert mgr._normalize_organism("MUS_MUSCULUS") == "mouse"
        assert mgr._normalize_organism("human") == "human"
        
        # Test key resolution
        key = mgr._resolve_reference_key("human", "GRCh38")
        assert key is not None
        assert "human" in key
    
    print("✅ Reference manager initialization: PASS")
    return True


def test_container_manager_init():
    """Test container manager initialization."""
    from workflow_composer.provisioning import ContainerManager
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = ContainerManager(cache_dir=tmpdir)
        
        assert mgr.cache_dir.exists()
        
        # Test tool lookup
        from workflow_composer.provisioning.container_manager import TOOL_CONTAINER_MAP
        container = TOOL_CONTAINER_MAP.get("star")
        assert container == "rna-seq"
    
    print("✅ Container manager initialization: PASS")
    return True


def test_reference_listing():
    """Test listing available references."""
    from workflow_composer.provisioning import ReferenceManager
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = ReferenceManager(base_path=tmpdir)
        available = mgr.list_available()
        
        # Should list all catalog entries
        assert len(available) > 0
        
        # Check structure
        for key, info in available.items():
            assert "organism" in info
            assert "build" in info
            assert "fasta_available" in info
            assert "indices" in info
    
    print("✅ Reference listing: PASS")
    return True


def test_container_listing():
    """Test listing available containers."""
    from workflow_composer.provisioning import ContainerManager
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = ContainerManager(cache_dir=tmpdir)
        available = mgr.list_available()
        
        # Should list registry entries
        assert len(available) > 0
        assert "rna-seq" in available
        
        # Check structure
        info = available["rna-seq"]
        assert info.name == "rna-seq"
        assert "star" in info.tools
    
    print("✅ Container listing: PASS")
    return True


def test_preflight_with_managers():
    """Test preflight validator uses managers."""
    from workflow_composer.core.preflight_validator import PreflightValidator
    
    validator = PreflightValidator()
    
    # Should have manager properties
    assert hasattr(validator, 'ref_manager')
    assert hasattr(validator, 'container_manager')
    
    # Run a validation
    report = validator.validate(
        analysis_type="rna_seq",
        tools=["fastqc", "star"],
        organism="mouse",
        genome_build="GRCm39",
    )
    
    assert hasattr(report, 'can_proceed')
    assert hasattr(report, 'items')
    
    print("✅ Preflight with managers: PASS")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Phase 2.3: Auto-Provisioning Tests")
    print("=" * 60 + "\n")
    
    tests = [
        ("Reference Catalog", test_reference_catalog),
        ("Container Registry", test_container_registry),
        ("Reference Manager Init", test_reference_manager_init),
        ("Container Manager Init", test_container_manager_init),
        ("Reference Listing", test_reference_listing),
        ("Container Listing", test_container_listing),
        ("Preflight with Managers", test_preflight_with_managers),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {name}: FAIL - {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "-" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
