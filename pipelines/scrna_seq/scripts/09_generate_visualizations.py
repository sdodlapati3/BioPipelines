#!/usr/bin/env python3
"""
Generate comprehensive visualizations
"""

import sys
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Snakemake inputs/outputs
input_h5ad = snakemake.input.h5ad
plot_dir = snakemake.output.plot_dir

# Parameters
plots_to_generate = snakemake.params.plots
dpi = snakemake.params.dpi

# Create output directory
Path(plot_dir).mkdir(parents=True, exist_ok=True)

# Set plotting parameters
sc.settings.set_figure_params(dpi=dpi, frameon=False, figsize=(8, 6))
sc.settings.verbosity = 1

print(f"Loading data from {input_h5ad}...")
adata = sc.read_h5ad(input_h5ad)

print(f"\nDataset: {adata.n_obs:,} cells, {adata.n_vars:,} genes")
print(f"Cell types: {adata.obs['celltype'].nunique()}")
print(f"Clusters: {adata.obs['leiden'].nunique()}")

print(f"\nGenerating {len(plots_to_generate)} plot types...")

# Define available markers from annotation
if 'marker_genes' in snakemake.config['annotation']:
    marker_genes_dict = snakemake.config['annotation']['marker_genes']
    all_markers = []
    for markers in marker_genes_dict.values():
        all_markers.extend(markers)
    available_markers = [g for g in all_markers if g in adata.raw.var_names]
else:
    available_markers = []

for plot_type in plots_to_generate:
    print(f"\nGenerating: {plot_type}...")
    
    if plot_type == "qc_violin":
        # QC metrics violin plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        sc.pl.violin(adata, ['total_counts', 'n_genes_by_counts', 
                            'pct_counts_mt', 'pct_counts_ribo'],
                    groupby='celltype', ax=axes.ravel(), show=False, rotation=45)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/qc_violin_by_celltype.png", dpi=dpi, bbox_inches='tight')
        plt.close()
    
    elif plot_type == "highest_expr_genes":
        # Highest expressed genes
        fig = sc.pl.highest_expr_genes(adata, n_top=30, show=False)
        plt.savefig(f"{plot_dir}/highest_expr_genes.png", dpi=dpi, bbox_inches='tight')
        plt.close()
    
    elif plot_type == "mito_scatter":
        # Mitochondrial vs total counts
        fig, ax = plt.subplots(figsize=(8, 6))
        sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', 
                     color='celltype', ax=ax, show=False)
        plt.savefig(f"{plot_dir}/mito_scatter.png", dpi=dpi, bbox_inches='tight')
        plt.close()
    
    elif plot_type == "pca_variance":
        # PCA variance ratio
        fig = sc.pl.pca_variance_ratio(adata, log=True, n_pcs=50, show=False)
        plt.savefig(f"{plot_dir}/pca_variance_ratio.png", dpi=dpi, bbox_inches='tight')
        plt.close()
    
    elif plot_type == "umap_clusters":
        # UMAP with multiple colorings
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        sc.pl.umap(adata, color='leiden', ax=axes[0,0], show=False, title='Leiden clusters')
        sc.pl.umap(adata, color='celltype', ax=axes[0,1], show=False, title='Cell types')
        sc.pl.umap(adata, color='total_counts', ax=axes[0,2], show=False, title='Total counts')
        sc.pl.umap(adata, color='n_genes_by_counts', ax=axes[1,0], show=False, title='N genes')
        sc.pl.umap(adata, color='pct_counts_mt', ax=axes[1,1], show=False, title='Mitochondrial %')
        sc.pl.umap(adata, color='doublet_score', ax=axes[1,2], show=False, title='Doublet score')
        
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/umap_comprehensive.png", dpi=dpi, bbox_inches='tight')
        plt.close()
    
    elif plot_type == "umap_genes":
        # UMAP colored by key marker genes
        if len(available_markers) >= 6:
            markers_to_plot = available_markers[:6]
            fig = sc.pl.umap(adata, color=markers_to_plot, use_raw=True, 
                           ncols=3, show=False)
            plt.savefig(f"{plot_dir}/umap_marker_genes.png", dpi=dpi, bbox_inches='tight')
            plt.close()
    
    elif plot_type == "dotplot_markers":
        # Dotplot of marker genes
        if len(available_markers) > 0:
            fig = sc.pl.dotplot(adata, available_markers, groupby='celltype',
                              use_raw=True, standard_scale='var', show=False)
            plt.savefig(f"{plot_dir}/dotplot_markers.png", dpi=dpi, bbox_inches='tight')
            plt.close()
    
    elif plot_type == "heatmap_markers":
        # Heatmap of marker genes
        if len(available_markers) > 0:
            fig = sc.pl.heatmap(adata, available_markers, groupby='celltype',
                              use_raw=True, swap_axes=True, show_gene_labels=True,
                              show=False)
            plt.savefig(f"{plot_dir}/heatmap_markers.png", dpi=dpi, bbox_inches='tight')
            plt.close()
    
    elif plot_type == "rank_genes":
        # Ranked genes visualization
        if 'rank_genes_celltype' in adata.uns:
            fig = sc.pl.rank_genes_groups(adata, n_genes=20, 
                                         key='rank_genes_celltype',
                                         sharey=False, show=False)
            plt.savefig(f"{plot_dir}/rank_genes_groups.png", dpi=dpi, bbox_inches='tight')
            plt.close()

# Additional comprehensive plots

# Cell type composition
print("\nGenerating cell type composition plots...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

celltype_counts = adata.obs['celltype'].value_counts()
axes[0].bar(range(len(celltype_counts)), celltype_counts.values, color='steelblue', alpha=0.7)
axes[0].set_xticks(range(len(celltype_counts)))
axes[0].set_xticklabels(celltype_counts.index, rotation=45, ha='right')
axes[0].set_ylabel('Number of cells')
axes[0].set_title('Cell Type Distribution')
axes[0].grid(axis='y', alpha=0.3)

# Pie chart
colors = plt.cm.tab20(np.linspace(0, 1, len(celltype_counts)))
axes[1].pie(celltype_counts.values, labels=celltype_counts.index, 
           autopct='%1.1f%%', colors=colors)
axes[1].set_title('Cell Type Proportions')

plt.tight_layout()
plt.savefig(f"{plot_dir}/celltype_composition.png", dpi=dpi, bbox_inches='tight')
plt.close()

# Dendrogram of cell types
print("\nGenerating cell type dendrogram...")
sc.tl.dendrogram(adata, groupby='celltype')
fig = sc.pl.dendrogram(adata, groupby='celltype', show=False)
plt.savefig(f"{plot_dir}/celltype_dendrogram.png", dpi=dpi, bbox_inches='tight')
plt.close()

# Correlation heatmap of cell types
print("\nGenerating cell type correlation heatmap...")
# Get mean expression per cell type
celltype_means = []
for ct in adata.obs['celltype'].unique():
    ct_cells = adata[adata.obs['celltype'] == ct]
    celltype_means.append(ct_cells.X.mean(axis=0).A1)

celltype_means_df = pd.DataFrame(
    celltype_means,
    index=adata.obs['celltype'].unique(),
    columns=adata.var_names
)

# Calculate correlation
corr_matrix = celltype_means_df.T.corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
           center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title('Cell Type Expression Correlation')
plt.tight_layout()
plt.savefig(f"{plot_dir}/celltype_correlation.png", dpi=dpi, bbox_inches='tight')
plt.close()

# Embedding density
print("\nGenerating UMAP density plot...")
fig = sc.pl.embedding_density(adata, basis='umap', key='celltype', show=False)
plt.savefig(f"{plot_dir}/umap_density.png", dpi=dpi, bbox_inches='tight')
plt.close()

# Summary statistics plot
print("\nGenerating summary statistics plot...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Cells per cell type
celltype_counts.plot(kind='barh', ax=axes[0, 0], color='steelblue', alpha=0.7)
axes[0, 0].set_xlabel('Number of cells')
axes[0, 0].set_title('Cells per Cell Type')
axes[0, 0].grid(axis='x', alpha=0.3)

# Mean counts per cell type
mean_counts = adata.obs.groupby('celltype')['total_counts'].mean().sort_values()
mean_counts.plot(kind='barh', ax=axes[0, 1], color='coral', alpha=0.7)
axes[0, 1].set_xlabel('Mean total counts')
axes[0, 1].set_title('Mean Counts per Cell Type')
axes[0, 1].grid(axis='x', alpha=0.3)

# Mean genes per cell type
mean_genes = adata.obs.groupby('celltype')['n_genes_by_counts'].mean().sort_values()
mean_genes.plot(kind='barh', ax=axes[1, 0], color='mediumseagreen', alpha=0.7)
axes[1, 0].set_xlabel('Mean genes detected')
axes[1, 0].set_title('Mean Genes per Cell Type')
axes[1, 0].grid(axis='x', alpha=0.3)

# Mean mito % per cell type
mean_mito = adata.obs.groupby('celltype')['pct_counts_mt'].mean().sort_values()
mean_mito.plot(kind='barh', ax=axes[1, 1], color='mediumpurple', alpha=0.7)
axes[1, 1].set_xlabel('Mean mitochondrial %')
axes[1, 1].set_title('Mean Mitochondrial % per Cell Type')
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{plot_dir}/summary_statistics.png", dpi=dpi, bbox_inches='tight')
plt.close()

print(f"\nAll plots saved to {plot_dir}/")
print("Visualization generation complete!")
