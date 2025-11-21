#!/usr/bin/env python3
"""
Download ChIP-seq data using proper BioPipelines module
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biopipelines.data_download.sra_downloader import SRADownloader
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # ChIP-seq accessions: GM12878 H3K4me3 from ENCODE
    accessions = {
        'SRR1552484': 'h3k4me3_rep1',  # Replicate 1
        'SRR1552485': 'h3k4me3_rep2',  # Replicate 2
        'SRR1552480': 'input_control'  # Input control
    }
    
    output_dir = Path("/scratch/sdodl001/BioPipelines/data/raw/chip_seq")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    downloader = SRADownloader(output_dir)
    
    for accession, label in accessions.items():
        print(f"\n{'='*60}")
        print(f"Downloading {label} ({accession})")
        print(f"{'='*60}\n")
        
        try:
            files = downloader.download(accession, "chip_seq")
            
            print(f"\n✓ Successfully downloaded {len(files)} files:")
            for f in files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  {f.name}: {size_mb:.1f} MB")
                
        except Exception as e:
            print(f"\n✗ Failed to download {accession}: {e}")
            sys.exit(1)
    
    print(f"\n{'='*60}")
    print("✓ All ChIP-seq data downloaded successfully!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
