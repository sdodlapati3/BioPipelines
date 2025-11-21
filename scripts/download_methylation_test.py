#!/usr/bin/env python3
"""
Download test datasets for DNA methylation pipeline validation.
Uses the MethylationDownloader module to fetch ENCODE WGBS data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biopipelines.data_download.methylation_downloader import MethylationDownloader
import logging

logging.basicConfig(level=logging.INFO)

def main():
    # Initialize downloader
    downloader = MethylationDownloader(
        output_dir="/home/sdodl001_odu_edu/BioPipelines/data/raw/methylation"
    )
    
    print("=" * 80)
    print("Searching for ENCODE WGBS datasets (brain tissue)")
    print("=" * 80)
    
    # Search for brain WGBS data
    datasets = downloader.search_encode_wgbs(
        tissue="brain",
        assay_type="WGBS",
        limit=5
    )
    
    if not datasets:
        print("No datasets found. Trying broader search...")
        datasets = downloader.search_encode_wgbs(limit=5)
    
    print(f"\nFound {len(datasets)} WGBS datasets:")
    for i, ds in enumerate(datasets, 1):
        print(f"\n{i}. {ds.experiment_id}")
        print(f"   Organism: {ds.organism}")
        print(f"   Tissue: {ds.tissue}")
        print(f"   Cell Type: {ds.cell_type}")
        print(f"   Assay: {ds.assay_type}")
        print(f"   Files: {ds.num_files}")
    
    # Download first suitable dataset (small size preferred)
    if datasets:
        print(f"\n{'=' * 80}")
        print(f"Downloading experiment: {datasets[0].experiment_id}")
        print(f"{'=' * 80}\n")
        
        files = downloader.download_encode_experiment(
            experiment_id=datasets[0].experiment_id,
            max_files=2  # Just 2 replicates for testing
        )
        
        print(f"\n{'=' * 80}")
        print(f"Downloaded {len(files)} files successfully!")
        print(f"{'=' * 80}")
        for f in files:
            print(f"  - {f}")
        
        print("\nUpdate config.yaml with these sample names:")
        for f in files:
            sample_name = f.stem.replace('.fastq', '').replace('.gz', '')
            print(f"  - {sample_name}")

if __name__ == "__main__":
    main()
