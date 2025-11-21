#!/usr/bin/env python3
"""
Download test datasets for Hi-C pipeline validation.
Uses the HiCDownloader module to fetch ENCODE Hi-C data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biopipelines.data_download.hic_downloader import HiCDownloader
import logging

logging.basicConfig(level=logging.INFO)

def main():
    # Initialize downloader
    downloader = HiCDownloader(
        output_dir="/home/sdodl001_odu_edu/BioPipelines/data/raw/hic"
    )
    
    print("=" * 80)
    print("Searching for ENCODE Hi-C datasets (GM12878 cell line)")
    print("=" * 80)
    
    # Search for GM12878 Hi-C data (well-characterized cell line)
    datasets = downloader.search_encode_hic(
        cell_line="GM12878",
        limit=5
    )
    
    if not datasets:
        print("No GM12878 datasets found. Trying broader search...")
        datasets = downloader.search_encode_hic(limit=5)
    
    print(f"\nFound {len(datasets)} Hi-C datasets:")
    for i, ds in enumerate(datasets, 1):
        print(f"\n{i}. {ds.experiment_id}")
        print(f"   Organism: {ds.organism}")
        print(f"   Cell Type: {ds.cell_type}")
        print(f"   Protocol: {ds.protocol_type}")
        print(f"   Enzyme: {ds.restriction_enzyme}")
        print(f"   Files: {ds.num_files}")
    
    # Download first suitable dataset
    if datasets:
        print(f"\n{'=' * 80}")
        print(f"Downloading experiment: {datasets[0].experiment_id}")
        print(f"{'=' * 80}\n")
        
        # Download fastq files for full pipeline test
        files = downloader.download_encode_hic(
            experiment_id=datasets[0].experiment_id,
            file_format="fastq",
            max_files=4  # R1 + R2 for 2 replicates
        )
        
        print(f"\n{'=' * 80}")
        print(f"Downloaded {len(files)} files successfully!")
        print(f"{'=' * 80}")
        for f in files:
            print(f"  - {f}")
        
        print("\nUpdate config.yaml with these sample names:")
        # Extract unique sample names (remove _R1/_R2 suffixes)
        samples = set()
        for f in files:
            sample_name = f.stem.replace('.fastq', '').replace('.gz', '')
            # Remove _R1 or _R2 suffix
            if sample_name.endswith('_R1') or sample_name.endswith('_R2'):
                sample_name = sample_name[:-3]
            samples.add(sample_name)
        
        for sample in sorted(samples):
            print(f"  - {sample}")

if __name__ == "__main__":
    main()
