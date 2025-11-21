#!/usr/bin/env python3
"""
Download ChIP-seq data from ENCODE using proper experiment IDs
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biopipelines.data_download.encode_downloader import ENCODEDownloader
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    output_dir = Path("/scratch/sdodl001/BioPipelines/data/raw/chip_seq")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    downloader = ENCODEDownloader(output_dir)
    
    # Download from ENCODE experiment ENCSR000DRY (GM12878 H3K4me3)
    experiment_id = "ENCSR000DRY"
    
    print(f"\n{'='*60}")
    print(f"Downloading ENCODE experiment: {experiment_id}")
    print(f"Target: H3K4me3 ChIP-seq on GM12878")
    print(f"{'='*60}\n")
    
    try:
        files = downloader.download_experiment(
            experiment_id,
            dataset_type="chip_seq",
            file_format="fastq"
        )
        
        print(f"\n✓ Successfully downloaded {len(files)} files:")
        for f in files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {size_mb:.1f} MB")
        
        print(f"\n{'='*60}")
        print("✓ ChIP-seq data downloaded successfully!")
        print(f"{'='*60}")
        return 0
        
    except Exception as e:
        print(f"\n✗ Failed to download: {e}")
        logging.exception("Download failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
