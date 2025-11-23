#!/bin/bash
# Rebuild RNA-seq with new base + 4 failed containers

set -e

echo "════════════════════════════════════════════════════════"
echo "Rebuilding Remaining Containers"
echo "Start time: $(date)"
echo "════════════════════════════════════════════════════════"

# Rebuild RNA-seq with optimized base
echo "✓ Submitting rna-seq rebuild (new base): $(sbatch scripts/containers/build_rna_seq_container.slurm | awk '{print $4}')"

# Rebuild 4 failed containers
echo "✓ Submitted metagenomics build: $(sbatch scripts/containers/build_metagenomics_container.slurm | awk '{print $4}')"
echo "✓ Submitted methylation build: $(sbatch scripts/containers/build_methylation_container.slurm | awk '{print $4}')"
echo "✓ Submitted scrna-seq build: $(sbatch scripts/containers/build_scrna_seq_container.slurm | awk '{print $4}')"
echo "✓ Submitted structural-variants build: $(sbatch scripts/containers/build_structural_variants_container.slurm | awk '{print $4}')"

echo ""
echo "════════════════════════════════════════════════════════"
echo "Submitted 5 container rebuild jobs"
echo "Monitor: squeue -u \$USER"
echo "════════════════════════════════════════════════════════"
