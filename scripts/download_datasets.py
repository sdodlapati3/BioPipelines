#!/usr/bin/env python3
"""
Download validated test datasets for BioPipelines using pysradb and wget.
"""

import os
import subprocess
import sys
from pathlib import Path
import pysradb

def download_file(url, output_path):
    """Download file using wget."""
    cmd = ["wget", "-O", str(output_path), url]
    print(f"Downloading: {url}")
    print(f"  -> {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ Downloaded successfully")
        return True
    else:
        print(f"✗ Download failed: {result.stderr}")
        return False

def get_ena_http_url(accession, db):
    """Get ENA HTTP URL for an accession."""
    try:
        metadata = db.sra_metadata(accession, detailed=True)
        
        # Check for paired-end reads
        if metadata['ena_fastq_http_1'].iloc[0]:
            urls = [
                metadata['ena_fastq_http_1'].iloc[0],
                metadata['ena_fastq_http_2'].iloc[0]
            ]
            return urls
        else:
            # Single-end
            return [metadata['ena_fastq_http'].iloc[0]]
    except Exception as e:
        print(f"Error getting metadata for {accession}: {e}")
        return None

def main():
    # Initialize pysradb
    db = pysradb.SRAweb()
    
    # Define datasets to download
    datasets = {
        'rna_seq': {
            'dir': Path.home() / 'BioPipelines/data/raw/rna_seq',
            'accessions': [
                ('ERR458493', 'wt_rep1'),  # Yeast WT replicate 1
                ('ERR458494', 'wt_rep2'),  # Yeast WT replicate 2
                ('ERR458495', 'mut_rep1'), # Yeast mutant replicate 1
                ('ERR458496', 'mut_rep2'), # Yeast mutant replicate 2
            ]
        },
        'chip_seq': {
            'dir': Path.home() / 'BioPipelines/data/raw/chip_seq',
            'accessions': [
                # ENCODE GM12878 H3K4me3 ChIP-seq (human, hg38-compatible)
                ('SRR1552484', 'h3k4me3_rep1'),  # ENCODE ENCSR000AKP rep1
                ('SRR1552485', 'h3k4me3_rep2'),  # ENCODE ENCSR000AKP rep2
                ('SRR1552480', 'input_control'), # Input control
            ]
        },
        'atac_seq': {
            'dir': Path.home() / 'BioPipelines/data/raw/atac_seq',
            'accessions': [
                ('SRR891268', 'sample1'),
                ('SRR891269', 'sample2'),
            ]
        }
    }
    
    # Download each dataset
    for data_type, config in datasets.items():
        print(f"\n{'='*60}")
        print(f"Downloading {data_type.upper()} datasets")
        print(f"{'='*60}")
        
        output_dir = config['dir']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for accession, sample_name in config['accessions']:
            print(f"\nProcessing {accession} ({sample_name})...")
            
            urls = get_ena_http_url(accession, db)
            if not urls:
                print(f"  ✗ Could not get URLs for {accession}")
                continue
            
            # Download files
            if len(urls) == 2:
                # Paired-end
                r1_path = output_dir / f"{sample_name}_R1.fastq.gz"
                r2_path = output_dir / f"{sample_name}_R2.fastq.gz"
                download_file(urls[0], r1_path)
                download_file(urls[1], r2_path)
            else:
                # Single-end
                output_path = output_dir / f"{sample_name}.fastq.gz"
                download_file(urls[0], output_path)
    
    print(f"\n{'='*60}")
    print("Download complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
