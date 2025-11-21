#!/usr/bin/env python3
"""
Download ChIP-seq input control from ENCODE
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biopipelines.data_download.encode_downloader import ENCODEDownloader
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    output_dir = Path("/scratch/sdodl001/BioPipelines/data/raw/chip_seq")
    downloader = ENCODEDownloader(output_dir)
    
    # Download input control from ENCSR000DRV
    experiment_id = "ENCSR000DRV"
    
    print(f"\n{'='*60}")
    print(f"Downloading input control: {experiment_id}")
    print(f"{'='*60}\n")
    
    try:
        files = downloader.download_experiment(
            experiment_id,
            dataset_type="chip_seq",
            file_format="fastq"
        )
        
        print(f"\n✓ Downloaded {len(files)} files:")
        for f in files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {size_mb:.1f} MB")
            
        return 0
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        logging.exception("Download failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
