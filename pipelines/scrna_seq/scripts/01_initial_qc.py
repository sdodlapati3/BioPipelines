#!/usr/bin/env python3
"""
Initial QC - Load STARsolo output and calculate QC metrics
"""

import sys
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Snakemake inputs/outputs
matrix_file = snakemake.input.matrix
barcodes_file = snakemake.input.barcodes
features_file = snakemake.input.features
output_h5ad = snakemake.output.h5ad
qc_report = snakemake.output.qc_report
qc_plots_dir = snakemake.output.qc_plots

# Parameters
min_counts = snakemake.params.min_counts
max_counts = snakemake.params.max_counts
min_genes = snakemake.params.min_genes
max_mito = snakemake.params.max_mito

# Create output directory
Path(qc_plots_dir).mkdir(parents=True, exist_ok=True)

# Set plot style
sc.settings.set_figure_params(dpi=150, frameon=False, figsize=(8, 6))

print("Loading STARsolo output...")
# Read 10x-formatted matrix
adata = sc.read_10x_mtx(
    Path(matrix_file).parent,
    var_names='gene_symbols',
    cache=True
)

print(f"Loaded {adata.n_obs} cells and {adata.n_vars} genes")

# Calculate QC metrics
print("Calculating QC metrics...")

# Identify mitochondrial genes
adata.var['mt'] = adata.var_names.str.startswith('MT-')
# Identify ribosomal genes
adata.var['ribo'] = adata.var_names.str.startswith(('RPS', 'RPL'))
# Identify hemoglobin genes (common in blood samples)
adata.var['hb'] = adata.var_names.str.contains('^HB[^(P)]')

# Calculate QC metrics
sc.pp.calculate_qc_metrics(
    adata,
    qc_vars=['mt', 'ribo', 'hb'],
    percent_top=None,
    log1p=False,
    inplace=True
)

# Generate QC plots
print("Generating QC plots...")

# 1. Violin plots for key metrics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'pct_counts_ribo'],
             jitter=0.4, multi_panel=True, ax=axes.ravel(), show=False)
plt.tight_layout()
plt.savefig(f"{qc_plots_dir}/qc_violin_plots.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Scatter plots showing relationships
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Total counts vs genes
axes[0].scatter(adata.obs['total_counts'], adata.obs['n_genes_by_counts'], 
                alpha=0.3, s=1, c='steelblue')
axes[0].set_xlabel('Total counts per cell')
axes[0].set_ylabel('Number of genes')
axes[0].set_title('Counts vs Genes')
axes[0].axhline(y=min_genes, color='red', linestyle='--', alpha=0.5, label=f'min_genes={min_genes}')
axes[0].axvline(x=min_counts, color='red', linestyle='--', alpha=0.5, label=f'min_counts={min_counts}')
axes[0].legend()

# Total counts vs mitochondrial percentage
axes[1].scatter(adata.obs['total_counts'], adata.obs['pct_counts_mt'], 
                alpha=0.3, s=1, c='coral')
axes[1].set_xlabel('Total counts per cell')
axes[1].set_ylabel('Mitochondrial %')
axes[1].set_title('Counts vs Mitochondrial %')
axes[1].axhline(y=max_mito, color='red', linestyle='--', alpha=0.5, label=f'max_mito={max_mito}%')
axes[1].legend()

# Genes vs mitochondrial percentage
axes[2].scatter(adata.obs['n_genes_by_counts'], adata.obs['pct_counts_mt'], 
                alpha=0.3, s=1, c='mediumseagreen')
axes[2].set_xlabel('Number of genes')
axes[2].set_ylabel('Mitochondrial %')
axes[2].set_title('Genes vs Mitochondrial %')
axes[2].axhline(y=max_mito, color='red', linestyle='--', alpha=0.5)
axes[2].axvline(x=min_genes, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(f"{qc_plots_dir}/qc_scatter_plots.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Highest expressed genes
fig = sc.pl.highest_expr_genes(adata, n_top=20, show=False)
plt.savefig(f"{qc_plots_dir}/highest_expr_genes.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Distribution histograms
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0].hist(adata.obs['total_counts'], bins=100, color='steelblue', alpha=0.7)
axes[0].set_xlabel('Total counts')
axes[0].set_ylabel('Number of cells')
axes[0].set_title('Distribution of total counts')
axes[0].axvline(x=min_counts, color='red', linestyle='--', label=f'min={min_counts}')
axes[0].axvline(x=max_counts, color='red', linestyle='--', label=f'max={max_counts}')
axes[0].set_yscale('log')
axes[0].legend()

axes[1].hist(adata.obs['n_genes_by_counts'], bins=100, color='coral', alpha=0.7)
axes[1].set_xlabel('Number of genes')
axes[1].set_ylabel('Number of cells')
axes[1].set_title('Distribution of detected genes')
axes[1].axvline(x=min_genes, color='red', linestyle='--', label=f'min={min_genes}')
axes[1].set_yscale('log')
axes[1].legend()

axes[2].hist(adata.obs['pct_counts_mt'], bins=100, color='mediumseagreen', alpha=0.7)
axes[2].set_xlabel('Mitochondrial %')
axes[2].set_ylabel('Number of cells')
axes[2].set_title('Distribution of mitochondrial %')
axes[2].axvline(x=max_mito, color='red', linestyle='--', label=f'max={max_mito}%')
axes[2].legend()

axes[3].hist(adata.obs['pct_counts_ribo'], bins=100, color='mediumpurple', alpha=0.7)
axes[3].set_xlabel('Ribosomal %')
axes[3].set_ylabel('Number of cells')
axes[3].set_title('Distribution of ribosomal %')

plt.tight_layout()
plt.savefig(f"{qc_plots_dir}/qc_distributions.png", dpi=300, bbox_inches='tight')
plt.close()

# Save QC metrics to CSV
print("Saving QC metrics...")
qc_df = adata.obs[[
    'total_counts', 'n_genes_by_counts',
    'pct_counts_mt', 'pct_counts_ribo', 'pct_counts_hb'
]]
qc_df.to_csv(qc_report)

# Print summary statistics
print("\n" + "="*80)
print("QC SUMMARY STATISTICS")
print("="*80)
print(f"\nTotal cells: {adata.n_obs:,}")
print(f"Total genes: {adata.n_vars:,}")
print(f"\nMedian counts per cell: {adata.obs['total_counts'].median():.0f}")
print(f"Median genes per cell: {adata.obs['n_genes_by_counts'].median():.0f}")
print(f"Median mitochondrial %: {adata.obs['pct_counts_mt'].median():.2f}%")
print(f"Median ribosomal %: {adata.obs['pct_counts_ribo'].median():.2f}%")

# Estimate cells to filter
cells_low_counts = (adata.obs['total_counts'] < min_counts).sum()
cells_high_counts = (adata.obs['total_counts'] > max_counts).sum()
cells_low_genes = (adata.obs['n_genes_by_counts'] < min_genes).sum()
cells_high_mito = (adata.obs['pct_counts_mt'] > max_mito).sum()

print(f"\nCells with < {min_counts} counts: {cells_low_counts} ({cells_low_counts/adata.n_obs*100:.2f}%)")
print(f"Cells with > {max_counts} counts: {cells_high_counts} ({cells_high_counts/adata.n_obs*100:.2f}%)")
print(f"Cells with < {min_genes} genes: {cells_low_genes} ({cells_low_genes/adata.n_obs*100:.2f}%)")
print(f"Cells with > {max_mito}% mito: {cells_high_mito} ({cells_high_mito/adata.n_obs*100:.2f}%)")

# Save raw data
print(f"\nSaving raw data to {output_h5ad}")
adata.write(output_h5ad)

print("Initial QC complete!")
