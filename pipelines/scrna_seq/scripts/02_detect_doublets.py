#!/usr/bin/env python3
"""
Doublet Detection - Identify potential doublets using Scrublet
"""

import sys
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scrublet as scr
from pathlib import Path

# Snakemake inputs/outputs
input_h5ad = snakemake.input.h5ad
output_h5ad = snakemake.output.h5ad
doublets_csv = snakemake.output.doublets_csv

# Parameters
method = snakemake.params.method
expected_rate = snakemake.params.expected_rate
threshold = snakemake.params.threshold

print(f"Loading data from {input_h5ad}...")
adata = sc.read_h5ad(input_h5ad)

print(f"Running {method} doublet detection...")
print(f"Expected doublet rate: {expected_rate:.2%}")

if method == "scrublet":
    # Initialize Scrublet
    scrub = scr.Scrublet(
        adata.X,
        expected_doublet_rate=expected_rate,
        random_state=42
    )
    
    # Run doublet detection
    doublet_scores, predicted_doublets = scrub.scrub_doublets(
        min_counts=2,
        min_cells=3,
        min_gene_variability_pct=85,
        n_prin_comps=30
    )
    
    # Add results to adata
    adata.obs['doublet_score'] = doublet_scores
    adata.obs['predicted_doublet'] = predicted_doublets
    
    # Scrublet plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot doublet score histogram
    scrub.plot_histogram(ax=axes[0])
    axes[0].set_title('Doublet Score Distribution')
    
    # Plot UMAP with doublet scores
    try:
        scrub.set_embedding('UMAP', scr.get_umap(scrub.manifold_obs_, 10, min_dist=0.3))
        scrub.plot_embedding('UMAP', order_points=True, ax=axes[1])
        axes[1].set_title('Doublet Scores (UMAP)')
    except Exception as e:
        print(f"Could not plot UMAP: {e}")
    
    plt.tight_layout()
    plot_path = Path(output_h5ad).parent / f"{Path(output_h5ad).stem}_doublet_detection.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
elif method == "doubletfinder":
    # Note: DoubletFinder is R-based, would need rpy2 integration
    print("DoubletFinder requires R/rpy2 integration - using Scrublet instead")
    method = "scrublet"
    # (repeat scrublet code above)

# Summary statistics
n_doublets = adata.obs['predicted_doublet'].sum()
doublet_rate = n_doublets / adata.n_obs

print("\n" + "="*80)
print("DOUBLET DETECTION SUMMARY")
print("="*80)
print(f"\nMethod: {method}")
print(f"Total cells analyzed: {adata.n_obs:,}")
print(f"Predicted doublets: {n_doublets:,} ({doublet_rate:.2%})")
print(f"Expected doublet rate: {expected_rate:.2%}")
print(f"\nMean doublet score: {adata.obs['doublet_score'].mean():.4f}")
print(f"Median doublet score: {adata.obs['doublet_score'].median():.4f}")
print(f"Max doublet score: {adata.obs['doublet_score'].max():.4f}")

# Save doublet predictions
doublet_df = adata.obs[['doublet_score', 'predicted_doublet']].copy()
doublet_df.to_csv(doublets_csv)
print(f"\nDoublet predictions saved to {doublets_csv}")

# Save annotated data
adata.write(output_h5ad)
print(f"Annotated data saved to {output_h5ad}")

print("\nDoublet detection complete!")
