#!/usr/bin/env python3
"""
Download ChIP-seq data using aspera-connect (no SSL verification needed)
or direct HTTP download from ENA
"""

import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def download_from_ena_http(accession, output_dir):
    """Download directly from ENA using HTTP (no SSL cert issues)"""
    
    # ENA HTTP URLs (not FTP which has issues)
    # Format: http://ftp.sra.ebi.ac.uk/vol1/fastq/{SRR_PREFIX}/{ACCESSION}/{ACCESSION}_{READ}.fastq.gz
    
    srr_prefix = accession[:6]  # First 6 chars, e.g., SRR155
    
    files = []
    for read in ['1', '2']:
        url = f"http://ftp.sra.ebi.ac.uk/vol1/fastq/{srr_prefix}/00{accession[-1]}/{accession}/{accession}_{read}.fastq.gz"
        output_file = output_dir / f"{accession}_{read}.fastq.gz"
        
        if output_file.exists():
            logging.info(f"File already exists: {output_file.name}")
            files.append(output_file)
            continue
        
        logging.info(f"Downloading {accession} read {read} from ENA (HTTP)...")
        logging.info(f"URL: {url}")
        
        # Use wget with no certificate checking
        cmd = [
            "wget",
            "--no-check-certificate",  # Skip SSL verification
            "-q",
            "--show-progress",
            "-O", str(output_file),
            url
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Check file size
            size_mb = output_file.stat().st_size / (1024 * 1024)
            if size_mb < 1:
                logging.error(f"Downloaded file is too small: {size_mb:.2f} MB")
                output_file.unlink()
                return None
            
            logging.info(f"✓ Downloaded {output_file.name}: {size_mb:.1f} MB")
            files.append(output_file)
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Download failed: {e}")
            if output_file.exists():
                output_file.unlink()
            return None
    
    return files if files else None


def main():
    # ChIP-seq accessions: GM12878 H3K4me3 from ENCODE
    accessions = {
        'SRR1552484': 'h3k4me3_rep1',
        'SRR1552485': 'h3k4me3_rep2',
        'SRR1552480': 'input_control'
    }
    
    output_dir = Path("/scratch/sdodl001/BioPipelines/data/raw/chip_seq")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_success = True
    
    for accession, label in accessions.items():
        print(f"\n{'='*60}")
        print(f"Downloading {label} ({accession})")
        print(f"{'='*60}\n")
        
        files = download_from_ena_http(accession, output_dir)
        
        if files:
            print(f"\n✓ Successfully downloaded {label}:")
            for f in files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  {f.name}: {size_mb:.1f} MB")
        else:
            print(f"\n✗ Failed to download {label}")
            all_success = False
    
    if all_success:
        print(f"\n{'='*60}")
        print("✓ All ChIP-seq data downloaded successfully!")
        print(f"{'='*60}")
        return 0
    else:
        print(f"\n{'='*60}")
        print("✗ Some downloads failed")
        print(f"{'='*60}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
