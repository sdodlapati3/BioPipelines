#!/usr/bin/env python3
"""
Clustering - Compute neighbors, UMAP, and Leiden/Louvain clustering
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
umap_plot = snakemake.output.umap_plot
leiden_plots_dir = snakemake.output.leiden_plots

# Parameters
n_neighbors = snakemake.params.n_neighbors
n_pcs = snakemake.params.n_pcs
leiden_resolutions = snakemake.params.leiden_res

# Create output directory
Path(leiden_plots_dir).mkdir(parents=True, exist_ok=True)

print(f"Loading data from {input_h5ad}...")
adata = sc.read_h5ad(input_h5ad)

print(f"\nDataset: {adata.n_obs:,} cells")

# Compute neighbors
print(f"\nComputing neighborhood graph (n_neighbors={n_neighbors}, n_pcs={n_pcs})...")
sc.pp.neighbors(
    adata,
    n_neighbors=n_neighbors,
    n_pcs=n_pcs,
    metric='euclidean',
    random_state=42
)

# Compute UMAP
print("Computing UMAP embedding...")
sc.tl.umap(adata, min_dist=0.3, spread=1.0, random_state=42)

# Clustering at multiple resolutions
print(f"\nRunning Leiden clustering at resolutions: {leiden_resolutions}")

for res in leiden_resolutions:
    print(f"  Resolution {res}...")
    sc.tl.leiden(
        adata,
        resolution=res,
        key_added=f'leiden_{res}',
        random_state=42
    )
    n_clusters = adata.obs[f'leiden_{res}'].nunique()
    print(f"    Found {n_clusters} clusters")

# Use the middle resolution as default
default_res = leiden_resolutions[len(leiden_resolutions)//2]
adata.obs['leiden'] = adata.obs[f'leiden_{default_res}']
print(f"\nUsing leiden_{default_res} as default clustering")

# Also compute Louvain for comparison
print("\nRunning Louvain clustering (resolution=0.8)...")
sc.tl.louvain(adata, resolution=0.8, key_added='louvain', random_state=42)
n_louvain = adata.obs['louvain'].nunique()
print(f"  Found {n_louvain} clusters")

# Visualizations
print("\nGenerating visualizations...")

# 1. UMAP colored by different resolutions
n_res = len(leiden_resolutions)
ncols = min(3, n_res)
nrows = (n_res + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
if nrows == 1:
    axes = axes.reshape(1, -1)

for idx, res in enumerate(leiden_resolutions):
    row = idx // ncols
    col = idx % ncols
    ax = axes[row, col] if nrows > 1 else axes[col]
    
    sc.pl.umap(
        adata,
        color=f'leiden_{res}',
        ax=ax,
        show=False,
        title=f'Leiden clustering (res={res})',
        legend_loc='on data',
        legend_fontsize=8
    )

# Hide empty subplots
for idx in range(n_res, nrows * ncols):
    row = idx // ncols
    col = idx % ncols
    ax = axes[row, col] if nrows > 1 else axes[col]
    ax.axis('off')

plt.tight_layout()
plt.savefig(f"{leiden_plots_dir}/leiden_all_resolutions.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. UMAP with default clustering
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sc.pl.umap(adata, color='leiden', ax=axes[0], show=False, title='Leiden clusters', legend_loc='right margin')
sc.pl.umap(adata, color='louvain', ax=axes[1], show=False, title='Louvain clusters', legend_loc='right margin')

plt.tight_layout()
plt.savefig(umap_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"UMAP plot saved to {umap_plot}")

# 3. UMAP colored by QC metrics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sc.pl.umap(adata, color='total_counts', ax=axes[0, 0], show=False, title='Total counts')
sc.pl.umap(adata, color='n_genes_by_counts', ax=axes[0, 1], show=False, title='N genes')
sc.pl.umap(adata, color='pct_counts_mt', ax=axes[1, 0], show=False, title='Mitochondrial %')
sc.pl.umap(adata, color='doublet_score', ax=axes[1, 1], show=False, title='Doublet score')

plt.tight_layout()
plt.savefig(f"{leiden_plots_dir}/umap_qc_metrics.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Cluster sizes
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Leiden
cluster_sizes = adata.obs['leiden'].value_counts().sort_index()
axes[0].bar(range(len(cluster_sizes)), cluster_sizes.values, color='steelblue', alpha=0.7)
axes[0].set_xlabel('Cluster')
axes[0].set_ylabel('Number of cells')
axes[0].set_title(f'Leiden cluster sizes (res={default_res})')
axes[0].set_xticks(range(len(cluster_sizes)))
axes[0].set_xticklabels(cluster_sizes.index)
axes[0].grid(axis='y', alpha=0.3)

# Louvain
cluster_sizes_l = adata.obs['louvain'].value_counts().sort_index()
axes[1].bar(range(len(cluster_sizes_l)), cluster_sizes_l.values, color='coral', alpha=0.7)
axes[1].set_xlabel('Cluster')
axes[1].set_ylabel('Number of cells')
axes[1].set_title('Louvain cluster sizes')
axes[1].set_xticks(range(len(cluster_sizes_l)))
axes[1].set_xticklabels(cluster_sizes_l.index)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{leiden_plots_dir}/cluster_sizes.png", dpi=300, bbox_inches='tight')
plt.close()

# Summary statistics
print("\n" + "="*80)
print("CLUSTERING SUMMARY")
print("="*80)
print(f"\nDataset: {adata.n_obs:,} cells")
print(f"\nNeighborhood graph: {n_neighbors} neighbors, {n_pcs} PCs")
print(f"\nLeiden clustering:")
for res in leiden_resolutions:
    n_clusters = adata.obs[f'leiden_{res}'].nunique()
    sizes = adata.obs[f'leiden_{res}'].value_counts()
    print(f"  Resolution {res}: {n_clusters} clusters (min={sizes.min()}, max={sizes.max()}, median={sizes.median():.0f})")

print(f"\nLouvain clustering:")
n_clusters_l = adata.obs['louvain'].nunique()
sizes_l = adata.obs['louvain'].value_counts()
print(f"  {n_clusters_l} clusters (min={sizes_l.min()}, max={sizes_l.max()}, median={sizes_l.median():.0f})")

# Save cluster assignments
cluster_file = Path(output_h5ad).parent / f"{Path(output_h5ad).stem}_cluster_assignments.csv"
cluster_cols = [col for col in adata.obs.columns if 'leiden' in col or 'louvain' in col]
adata.obs[cluster_cols].to_csv(cluster_file)
print(f"\nCluster assignments saved to {cluster_file}")

# Save clustered data
adata.write(output_h5ad)
print(f"Clustered data saved to {output_h5ad}")

print("\nClustering complete!")
