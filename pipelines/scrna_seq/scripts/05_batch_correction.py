#!/usr/bin/env python3
"""
Batch correction (optional, if multiple batches present)
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

# Parameters
enable = snakemake.params.enable
method = snakemake.params.method

print(f"Loading data from {input_h5ad}...")
adata = sc.read_h5ad(input_h5ad)

if not enable:
    print("\nBatch correction is DISABLED in config")
    print("Saving data without batch correction...")
    adata.write(output_h5ad)
    print("Done!")
    sys.exit(0)

# Check if batch column exists
if 'batch' not in adata.obs.columns:
    print("\nWARNING: 'batch' column not found in adata.obs")
    print("Skipping batch correction...")
    adata.write(output_h5ad)
    sys.exit(0)

n_batches = adata.obs['batch'].nunique()
print(f"\nFound {n_batches} batches: {adata.obs['batch'].unique()}")

if n_batches == 1:
    print("Only one batch detected, skipping batch correction")
    adata.write(output_h5ad)
    sys.exit(0)

print(f"\nApplying batch correction using {method}...")

if method == "harmony":
    try:
        import harmonypy as hm
        
        # Run Harmony on PCA
        print("Running Harmony...")
        ho = hm.run_harmony(
            adata.obsm['X_pca'],
            adata.obs,
            'batch',
            max_iter_harmony=20
        )
        
        # Store corrected PCA
        adata.obsm['X_pca_harmony'] = ho.Z_corr.T
        adata.obsm['X_pca_original'] = adata.obsm['X_pca'].copy()
        adata.obsm['X_pca'] = adata.obsm['X_pca_harmony']
        
        print("Harmony batch correction complete")
        
    except ImportError:
        print("ERROR: harmonypy not installed")
        print("Install with: pip install harmonypy")
        sys.exit(1)

elif method == "bbknn":
    try:
        import bbknn
        
        # Run BBKNN (Batch Balanced K-Nearest Neighbors)
        print("Running BBKNN...")
        sc.external.pp.bbknn(adata, batch_key='batch', n_pcs=30)
        print("BBKNN batch correction complete")
        
    except ImportError:
        print("ERROR: bbknn not installed")
        print("Install with: pip install bbknn")
        sys.exit(1)

elif method == "scanorama":
    try:
        import scanorama
        
        # Run Scanorama
        print("Running Scanorama...")
        batches = []
        for batch in adata.obs['batch'].unique():
            batches.append(adata[adata.obs['batch'] == batch].copy())
        
        # Integrate
        corrected = scanorama.correct_scanpy(batches, return_dimred=True)
        
        # Merge back
        adata = sc.concat(corrected)
        print("Scanorama batch correction complete")
        
    except ImportError:
        print("ERROR: scanorama not installed")
        print("Install with: pip install scanorama")
        sys.exit(1)

elif method == "combat":
    # ComBat (adjusts gene expression directly)
    print("Running ComBat...")
    sc.pp.combat(adata, key='batch')
    print("ComBat batch correction complete")
    
    # Re-run PCA after ComBat
    print("Re-running PCA after ComBat...")
    sc.tl.pca(adata, n_comps=50, svd_solver='arpack')

else:
    print(f"ERROR: Unknown batch correction method: {method}")
    print("Available methods: harmony, bbknn, scanorama, combat")
    sys.exit(1)

# Visualization comparison (before/after)
if 'X_pca_original' in adata.obsm:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Before correction
    sc.pl.pca(
        adata,
        color='batch',
        use_raw=False,
        ax=axes[0],
        show=False,
        title='Before Batch Correction'
    )
    
    # After correction (using harmony-corrected PCA)
    sc.pl.embedding(
        adata,
        basis='pca_harmony' if method == 'harmony' else 'pca',
        color='batch',
        ax=axes[1],
        show=False,
        title=f'After {method.upper()} Correction'
    )
    
    plt.tight_layout()
    plot_path = Path(output_h5ad).parent / f"{Path(output_h5ad).stem}_batch_correction.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nBatch correction comparison plot saved to {plot_path}")

# Save batch-corrected data
adata.write(output_h5ad)
print(f"Batch-corrected data saved to {output_h5ad}")

print(f"\nBatch correction with {method} complete!")
