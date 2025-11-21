#!/usr/bin/env python3
"""
Filter cells and normalize counts
"""

import sys
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Snakemake inputs/outputs
input_h5ad = snakemake.input.h5ad
output_h5ad = snakemake.output.h5ad
filter_report = snakemake.output.filter_report

# Parameters
min_counts = snakemake.params.min_counts
max_counts = snakemake.params.max_counts
min_genes = snakemake.params.min_genes
min_cells = snakemake.params.min_cells
max_mito = snakemake.params.max_mito
max_ribo = snakemake.params.max_ribo
target_sum = snakemake.params.target_sum

print(f"Loading data from {input_h5ad}...")
adata = sc.read_h5ad(input_h5ad)

# Store original counts
n_cells_before = adata.n_obs
n_genes_before = adata.n_vars

print(f"\nBefore filtering: {n_cells_before:,} cells, {n_genes_before:,} genes")

# Create filtering report
report_lines = []
report_lines.append("="*80)
report_lines.append("CELL FILTERING REPORT")
report_lines.append("="*80)
report_lines.append(f"\nInitial: {n_cells_before:,} cells, {n_genes_before:,} genes")
report_lines.append(f"\nFiltering criteria:")
report_lines.append(f"  Min counts per cell: {min_counts}")
report_lines.append(f"  Max counts per cell: {max_counts}")
report_lines.append(f"  Min genes per cell: {min_genes}")
report_lines.append(f"  Min cells per gene: {min_cells}")
report_lines.append(f"  Max mitochondrial %: {max_mito}%")
report_lines.append(f"  Max ribosomal %: {max_ribo}%")

# Filter cells
print("\nFiltering cells...")

# Remove predicted doublets
n_doublets = adata.obs['predicted_doublet'].sum()
sc.pp.filter_cells(adata, min_counts=min_counts)
report_lines.append(f"\nAfter removing doublets: {adata.n_obs:,} cells ({n_doublets} doublets removed)")

# Filter by counts
adata = adata[adata.obs['total_counts'] >= min_counts, :]
report_lines.append(f"After min_counts filter: {adata.n_obs:,} cells")

adata = adata[adata.obs['total_counts'] <= max_counts, :]
report_lines.append(f"After max_counts filter: {adata.n_obs:,} cells")

# Filter by genes
adata = adata[adata.obs['n_genes_by_counts'] >= min_genes, :]
report_lines.append(f"After min_genes filter: {adata.n_obs:,} cells")

# Filter by mitochondrial content
adata = adata[adata.obs['pct_counts_mt'] <= max_mito, :]
report_lines.append(f"After max_mito filter: {adata.n_obs:,} cells")

# Filter by ribosomal content
adata = adata[adata.obs['pct_counts_ribo'] <= max_ribo, :]
report_lines.append(f"After max_ribo filter: {adata.n_obs:,} cells")

# Filter genes (must be expressed in at least min_cells)
print("Filtering genes...")
sc.pp.filter_genes(adata, min_cells=min_cells)
report_lines.append(f"\nAfter gene filtering: {adata.n_vars:,} genes")

n_cells_after = adata.n_obs
n_genes_after = adata.n_vars

report_lines.append(f"\nFinal: {n_cells_after:,} cells, {n_genes_after:,} genes")
report_lines.append(f"Removed: {n_cells_before - n_cells_after:,} cells ({(n_cells_before - n_cells_after)/n_cells_before*100:.2f}%)")
report_lines.append(f"Removed: {n_genes_before - n_genes_after:,} genes ({(n_genes_before - n_genes_after)/n_genes_before*100:.2f}%)")

# Normalization
print(f"\nNormalizing to {target_sum} total counts per cell...")
report_lines.append(f"\nNormalization:")
report_lines.append(f"  Method: Total count normalization")
report_lines.append(f"  Target sum: {target_sum}")

# Store raw counts in a layer
adata.layers['counts'] = adata.X.copy()

# Normalize to target_sum
sc.pp.normalize_total(adata, target_sum=target_sum)

# Log-transform
print("Log-transforming...")
sc.pp.log1p(adata)
report_lines.append(f"  Transformation: log1p (natural log of (x+1))")

# Store normalized data
adata.layers['log1p_normalized'] = adata.X.copy()

# Final statistics
print("\n" + "="*80)
print("FILTERING AND NORMALIZATION COMPLETE")
print("="*80)
print(f"\nFinal dataset: {n_cells_after:,} cells × {n_genes_after:,} genes")
print(f"Cells retained: {n_cells_after/n_cells_before*100:.2f}%")
print(f"Genes retained: {n_genes_after/n_genes_before*100:.2f}%")

report_lines.append(f"\nData layers:")
report_lines.append(f"  'counts': Raw UMI counts")
report_lines.append(f"  'log1p_normalized': Log-normalized data (current X)")

# Save report
with open(filter_report, 'w') as f:
    f.write('\n'.join(report_lines))
print(f"\nReport saved to {filter_report}")

# Save filtered and normalized data
adata.write(output_h5ad)
print(f"Filtered data saved to {output_h5ad}")

print("\nFiltering and normalization complete!")
