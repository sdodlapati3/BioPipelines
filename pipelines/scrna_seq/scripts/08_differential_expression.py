#!/usr/bin/env python3
"""
Differential expression analysis between cell types
"""

import sys
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Snakemake inputs/outputs
input_h5ad = snakemake.input.h5ad
deg_csv = snakemake.output.deg_csv
deg_plots_dir = snakemake.output.deg_plots
rank_genes_csv = snakemake.output.rank_genes

# Parameters
method = snakemake.params.method
min_pct = snakemake.params.min_pct
logfc_threshold = snakemake.params.logfc_threshold
pval_cutoff = snakemake.params.pval_cutoff

# Create output directory
Path(deg_plots_dir).mkdir(parents=True, exist_ok=True)

print(f"Loading data from {input_h5ad}...")
adata = sc.read_h5ad(input_h5ad)

print(f"\nDataset: {adata.n_obs:,} cells")
print(f"Cell types: {adata.obs['celltype'].unique()}")

# Differential expression analysis
print(f"\nRunning differential expression analysis ({method})...")
sc.tl.rank_genes_groups(
    adata,
    groupby='celltype',
    method=method,
    use_raw=True,
    key_added='rank_genes_celltype',
    pts=True  # Calculate fraction of cells expressing
)

# Extract results for all cell types
print("\nExtracting DEG results...")
all_degs = []

for celltype in adata.obs['celltype'].unique():
    deg_df = sc.get.rank_genes_groups_df(
        adata,
        group=celltype,
        key='rank_genes_celltype'
    )
    
    # Add cell type column
    deg_df['celltype'] = celltype
    
    # Filter by thresholds
    deg_filtered = deg_df[
        (deg_df['pvals_adj'] < pval_cutoff) &
        (np.abs(deg_df['logfoldchanges']) > logfc_threshold)
    ]
    
    n_degs = len(deg_filtered)
    print(f"  {celltype}: {n_degs} DEGs")
    
    all_degs.append(deg_df)

# Combine all results
combined_degs = pd.concat(all_degs, ignore_index=True)
combined_degs.to_csv(deg_csv, index=False)
print(f"\nAll DEG results saved to {deg_csv}")

# Summary statistics
print("\n" + "="*80)
print("DIFFERENTIAL EXPRESSION SUMMARY")
print("="*80)
print(f"\nMethod: {method}")
print(f"P-value cutoff: {pval_cutoff}")
print(f"Log fold-change threshold: {logfc_threshold}")
print(f"\nSignificant DEGs per cell type:")
for celltype in adata.obs['celltype'].unique():
    celltype_degs = combined_degs[
        (combined_degs['celltype'] == celltype) &
        (combined_degs['pvals_adj'] < pval_cutoff) &
        (np.abs(combined_degs['logfoldchanges']) > logfc_threshold)
    ]
    n_up = (celltype_degs['logfoldchanges'] > 0).sum()
    n_down = (celltype_degs['logfoldchanges'] < 0).sum()
    print(f"  {celltype}: {len(celltype_degs)} total ({n_up} up, {n_down} down)")

# Visualizations
print("\nGenerating visualizations...")

# 1. Rank genes dotplot
fig = sc.pl.rank_genes_groups_dotplot(
    adata,
    n_genes=5,
    key='rank_genes_celltype',
    groupby='celltype',
    standard_scale='var',
    show=False
)
plt.savefig(f"{deg_plots_dir}/rank_genes_dotplot.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Rank genes heatmap
fig = sc.pl.rank_genes_groups_heatmap(
    adata,
    n_genes=10,
    key='rank_genes_celltype',
    groupby='celltype',
    use_raw=True,
    show_gene_labels=True,
    show=False
)
plt.savefig(f"{deg_plots_dir}/rank_genes_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Rank genes stacked violin
fig = sc.pl.rank_genes_groups_stacked_violin(
    adata,
    n_genes=5,
    key='rank_genes_celltype',
    groupby='celltype',
    show=False
)
plt.savefig(f"{deg_plots_dir}/rank_genes_violin.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Volcano plots for each cell type
for celltype in adata.obs['celltype'].unique():
    celltype_degs = combined_degs[combined_degs['celltype'] == celltype]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Categorize genes
    sig_up = (celltype_degs['pvals_adj'] < pval_cutoff) & (celltype_degs['logfoldchanges'] > logfc_threshold)
    sig_down = (celltype_degs['pvals_adj'] < pval_cutoff) & (celltype_degs['logfoldchanges'] < -logfc_threshold)
    not_sig = ~(sig_up | sig_down)
    
    # Plot
    ax.scatter(celltype_degs.loc[not_sig, 'logfoldchanges'],
              -np.log10(celltype_degs.loc[not_sig, 'pvals_adj']),
              c='gray', alpha=0.3, s=5, label='Not significant')
    ax.scatter(celltype_degs.loc[sig_down, 'logfoldchanges'],
              -np.log10(celltype_degs.loc[sig_down, 'pvals_adj']),
              c='blue', alpha=0.6, s=10, label='Down-regulated')
    ax.scatter(celltype_degs.loc[sig_up, 'logfoldchanges'],
              -np.log10(celltype_degs.loc[sig_up, 'pvals_adj']),
              c='red', alpha=0.6, s=10, label='Up-regulated')
    
    # Add threshold lines
    ax.axhline(y=-np.log10(pval_cutoff), color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=logfc_threshold, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=-logfc_threshold, color='black', linestyle='--', alpha=0.5)
    
    # Labels for top genes
    top_genes = celltype_degs.nsmallest(10, 'pvals_adj')
    for _, gene in top_genes.iterrows():
        if abs(gene['logfoldchanges']) > logfc_threshold:
            ax.annotate(gene['names'],
                       xy=(gene['logfoldchanges'], -np.log10(gene['pvals_adj'])),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Log2 Fold Change')
    ax.set_ylabel('-Log10 Adjusted P-value')
    ax.set_title(f'Volcano Plot: {celltype}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    safe_celltype = celltype.replace(' ', '_').replace('/', '_')
    plt.savefig(f"{deg_plots_dir}/volcano_{safe_celltype}.png", dpi=300, bbox_inches='tight')
    plt.close()

# 5. Tracksplot (marker genes across cell types)
fig = sc.pl.rank_genes_groups_tracksplot(
    adata,
    n_genes=5,
    key='rank_genes_celltype',
    groupby='celltype',
    show=False
)
plt.savefig(f"{deg_plots_dir}/rank_genes_tracksplot.png", dpi=300, bbox_inches='tight')
plt.close()

# 6. Matrix plot of top DEGs
fig = sc.pl.rank_genes_groups_matrixplot(
    adata,
    n_genes=10,
    key='rank_genes_celltype',
    groupby='celltype',
    use_raw=True,
    standard_scale='var',
    show=False
)
plt.savefig(f"{deg_plots_dir}/rank_genes_matrixplot.png", dpi=300, bbox_inches='tight')
plt.close()

# Save top ranked genes per cell type
print("\nSaving ranked gene lists...")
rank_genes_list = []

for celltype in adata.obs['celltype'].unique():
    genes_df = sc.get.rank_genes_groups_df(
        adata,
        group=celltype,
        key='rank_genes_celltype'
    )
    genes_df['celltype'] = celltype
    rank_genes_list.append(genes_df.head(50))  # Top 50 per cell type

rank_genes_combined = pd.concat(rank_genes_list, ignore_index=True)
rank_genes_combined.to_csv(rank_genes_csv, index=False)
print(f"Top ranked genes saved to {rank_genes_csv}")

# Print top genes per cell type
print("\n" + "="*80)
print("TOP 5 MARKER GENES PER CELL TYPE")
print("="*80)
for celltype in adata.obs['celltype'].unique():
    print(f"\n{celltype}:")
    top_genes = rank_genes_combined[rank_genes_combined['celltype'] == celltype].head(5)
    for _, gene in top_genes.iterrows():
        print(f"  {gene['names']:<15} logFC={gene['logfoldchanges']:>6.2f}  " +
              f"pval_adj={gene['pvals_adj']:.2e}")

# Save results
adata.write(input_h5ad)  # Save back with DE results
print(f"\nData with DE results saved to {input_h5ad}")

print("\nDifferential expression analysis complete!")
