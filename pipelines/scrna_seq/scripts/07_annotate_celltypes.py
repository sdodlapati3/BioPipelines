#!/usr/bin/env python3
"""
Cell type annotation using marker genes
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
marker_plot = snakemake.output.marker_plot
dotplot = snakemake.output.dotplot
annotation_csv = snakemake.output.annotation_csv

# Parameters
method = snakemake.params.method
marker_genes = snakemake.params.marker_genes

print(f"Loading data from {input_h5ad}...")
adata = sc.read_h5ad(input_h5ad)

print(f"\nDataset: {adata.n_obs:,} cells in {adata.obs['leiden'].nunique()} clusters")

# Compute marker genes for each cluster
print("\nComputing marker genes for each cluster...")
sc.tl.rank_genes_groups(
    adata,
    groupby='leiden',
    method='wilcoxon',
    use_raw=False,
    key_added='rank_genes_leiden'
)

# Display top marker genes
print("\nTop 5 marker genes per cluster:")
print("="*80)
for cluster in adata.obs['leiden'].unique():
    genes = sc.get.rank_genes_groups_df(adata, group=cluster, key='rank_genes_leiden').head(5)
    print(f"\nCluster {cluster}:")
    print(genes[['names', 'scores', 'logfoldchanges', 'pvals_adj']].to_string(index=False))

# Annotate clusters based on marker genes
print("\n" + "="*80)
print("CELL TYPE ANNOTATION")
print("="*80)

if method == "marker_genes":
    # Manual annotation based on known marker genes
    print(f"\nUsing marker gene expression for annotation...")
    print(f"Cell types: {list(marker_genes.keys())}")
    
    # Calculate mean expression of marker genes in each cluster
    cluster_annotations = {}
    cluster_scores = {celltype: [] for celltype in marker_genes.keys()}
    
    for cluster in sorted(adata.obs['leiden'].unique(), key=int):
        cluster_cells = adata[adata.obs['leiden'] == cluster]
        
        print(f"\nCluster {cluster}:")
        celltype_scores = {}
        
        for celltype, markers in marker_genes.items():
            # Check which markers are present in dataset
            available_markers = [g for g in markers if g in adata.raw.var_names]
            
            if len(available_markers) == 0:
                celltype_scores[celltype] = 0
                continue
            
            # Calculate mean expression of available markers
            marker_expr = []
            for marker in available_markers:
                expr = cluster_cells.raw[:, marker].X.toarray().mean()
                marker_expr.append(expr)
            
            score = np.mean(marker_expr)
            celltype_scores[celltype] = score
            
            print(f"  {celltype}: {score:.3f} ({len(available_markers)}/{len(markers)} markers)")
        
        # Assign cell type with highest score
        if max(celltype_scores.values()) > 0:
            best_celltype = max(celltype_scores, key=celltype_scores.get)
            cluster_annotations[cluster] = best_celltype
        else:
            cluster_annotations[cluster] = "Unknown"
        
        for celltype in marker_genes.keys():
            cluster_scores[celltype].append(celltype_scores[celltype])
    
    # Add annotations to adata
    adata.obs['celltype'] = adata.obs['leiden'].map(cluster_annotations)
    
    print("\n" + "="*80)
    print("CLUSTER ANNOTATIONS:")
    print("="*80)
    for cluster, celltype in sorted(cluster_annotations.items(), key=lambda x: int(x[0])):
        n_cells = (adata.obs['leiden'] == cluster).sum()
        print(f"Cluster {cluster}: {celltype} ({n_cells:,} cells)")

elif method == "celltypist":
    try:
        import celltypist
        from celltypist import models
        
        # Download and use pre-trained model
        print("Downloading CellTypist model...")
        model = models.Model.load(model='Immune_All_Low.pkl')
        
        # Predict cell types
        print("Predicting cell types...")
        predictions = celltypist.annotate(adata, model=model, majority_voting=True)
        adata = predictions.to_adata()
        
        print("\nCell type predictions:")
        print(adata.obs['majority_voting'].value_counts())
        
    except ImportError:
        print("ERROR: celltypist not installed")
        print("Install with: pip install celltypist")
        print("Falling back to marker gene annotation...")
        method = "marker_genes"

# Visualizations
print("\nGenerating visualizations...")

# 1. UMAP with cell type annotation
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sc.pl.umap(adata, color='leiden', ax=axes[0], show=False, 
          title='Leiden clusters', legend_loc='right margin')
sc.pl.umap(adata, color='celltype', ax=axes[1], show=False, 
          title='Cell type annotation', legend_loc='right margin')

plt.tight_layout()
plt.savefig(marker_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"Cell type UMAP saved to {marker_plot}")

# 2. Dotplot of marker genes
print("\nGenerating dotplot of marker genes...")

# Flatten marker genes
all_markers = []
for markers in marker_genes.values():
    all_markers.extend(markers)
# Remove duplicates while preserving order
all_markers = list(dict.fromkeys(all_markers))
# Keep only markers present in dataset
available_markers = [g for g in all_markers if g in adata.raw.var_names]

if len(available_markers) > 0:
    fig = sc.pl.dotplot(
        adata,
        available_markers,
        groupby='celltype',
        use_raw=True,
        standard_scale='var',  # Normalize per gene
        show=False
    )
    plt.savefig(dotplot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Dotplot saved to {dotplot}")
else:
    print("WARNING: No marker genes found in dataset")
    # Create empty plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'No marker genes available', 
           ha='center', va='center', fontsize=14)
    ax.axis('off')
    plt.savefig(dotplot, dpi=300, bbox_inches='tight')
    plt.close()

# 3. Stacked violin plots for top markers
if len(available_markers) > 0:
    top_markers = available_markers[:min(10, len(available_markers))]
    fig = sc.pl.stacked_violin(
        adata,
        top_markers,
        groupby='celltype',
        use_raw=True,
        show=False
    )
    violin_path = Path(marker_plot).parent / f"{Path(marker_plot).stem}_violin.png"
    plt.savefig(violin_path, dpi=300, bbox_inches='tight')
    plt.close()

# 4. Matrix plot
if len(available_markers) > 0:
    fig = sc.pl.matrixplot(
        adata,
        available_markers,
        groupby='celltype',
        use_raw=True,
        standard_scale='var',
        show=False
    )
    matrix_path = Path(marker_plot).parent / f"{Path(marker_plot).stem}_matrix.png"
    plt.savefig(matrix_path, dpi=300, bbox_inches='tight')
    plt.close()

# Cell type distribution
print("\n" + "="*80)
print("CELL TYPE DISTRIBUTION")
print("="*80)
celltype_counts = adata.obs['celltype'].value_counts()
for celltype, count in celltype_counts.items():
    pct = count / adata.n_obs * 100
    print(f"{celltype}: {count:,} cells ({pct:.2f}%)")

# Save annotation mapping
annotation_df = pd.DataFrame({
    'cluster': list(cluster_annotations.keys()),
    'celltype': list(cluster_annotations.values())
})
annotation_df.to_csv(annotation_csv, index=False)
print(f"\nAnnotation mapping saved to {annotation_csv}")

# Save annotated data
adata.write(output_h5ad)
print(f"Annotated data saved to {output_h5ad}")

print("\nCell type annotation complete!")
