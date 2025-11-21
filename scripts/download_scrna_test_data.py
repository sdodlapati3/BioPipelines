#!/usr/bin/env python3
"""
Download 10x Genomics PBMC 3k dataset for testing scRNA-seq pipeline
"""

import os
import sys
import requests
from pathlib import Path
import gzip
import shutil

def download_file(url, output_path):
    """Download a file with progress indicator"""
    print(f"Downloading: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    downloaded = 0
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\r  Progress: {percent:.1f}% ({downloaded // 1024 // 1024} MB)", 
                          end='', flush=True)
    print("\n  ✓ Complete")

def main():
    # Setup directories
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "data" / "raw" / "scrna_seq"
    ref_dir = base_dir / "data" / "references"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Downloading 10x Genomics PBMC 3k Dataset")
    print("="*80)
    print("\nDataset: 3k PBMCs from a Healthy Donor")
    print("Platform: 10x Genomics Chromium v2 Chemistry")
    print("Expected cells: ~2,700")
    print("Expected UMI: ~1,000 median per cell")
    print("\n")
    
    # 10x Genomics PBMC 3k dataset URLs (smaller dataset for testing)
    # From: https://support.10xgenomics.com/single-cell-gene-expression/datasets
    
    urls = {
        "fastq_r1": "http://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_1k_v3/pbmc_1k_v3_fastqs.tar",
        "filtered_matrix": "http://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_1k_v3/pbmc_1k_v3_filtered_feature_bc_matrix.tar.gz"
    }
    
    # Alternative: Use actual FASTQs from SRA
    print("NOTE: For testing, we'll use 10x PBMC 1k dataset")
    print("This is a smaller dataset (~1000 cells) suitable for quick testing")
    print("\nFor production analysis, use the full PBMC 3k or 10k datasets\n")
    
    # Download filtered matrix (for quick testing without alignment)
    matrix_path = raw_dir / "pbmc_1k_v3_filtered_feature_bc_matrix.tar.gz"
    if not matrix_path.exists():
        print("\n1. Downloading filtered feature-barcode matrix...")
        download_file(urls["filtered_matrix"], matrix_path)
        
        # Extract
        print("   Extracting...")
        import tarfile
        with tarfile.open(matrix_path, 'r:gz') as tar:
            tar.extractall(raw_dir)
        print("   ✓ Extracted")
    else:
        print("\n1. Filtered matrix already downloaded")
    
    # For FASTQ files, we'll use a smaller test dataset
    # or demonstrate with simulated data
    
    print("\n" + "="*80)
    print("Alternative: Download from SRA (actual FASTQ files)")
    print("="*80)
    
    # SRA accession for 10x PBMC data
    sra_accessions = [
        "SRR8206317",  # 10x PBMC ~5k cells, ~500MB
    ]
    
    print("\nFor full FASTQ-based analysis, download from SRA:")
    for acc in sra_accessions:
        print(f"  fastq-dump --split-files --gzip {acc}")
    
    print("\nOr use the 10x website downloads (requires registration):")
    print("  https://www.10xgenomics.com/resources/datasets")
    
    # Download cell barcode whitelist for 10x v3 chemistry
    print("\n" + "="*80)
    print("Downloading 10x Cell Barcode Whitelist (v3)")
    print("="*80)
    
    whitelist_url = "https://github.com/10XGenomics/cellranger/raw/master/lib/python/cellranger/barcodes/3M-february-2018.txt.gz"
    whitelist_path = ref_dir / "10x_whitelist_v3.txt.gz"
    whitelist_final = ref_dir / "10x_whitelist_v3.txt"
    
    if not whitelist_final.exists():
        print("\nDownloading v3 chemistry whitelist...")
        download_file(whitelist_url, whitelist_path)
        
        # Decompress
        print("   Decompressing...")
        with gzip.open(whitelist_path, 'rb') as f_in:
            with open(whitelist_final, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        whitelist_path.unlink()
        print("   ✓ Complete")
    else:
        print("\nWhitelist already downloaded")
    
    # Summary
    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)
    print(f"\nData location: {raw_dir}")
    print(f"Whitelist location: {whitelist_final}")
    
    print("\n📋 Next steps:")
    print("1. Download actual FASTQ files using:")
    print("   - SRA Toolkit: fastq-dump or fasterq-dump")
    print("   - 10x website: https://www.10xgenomics.com/resources/datasets")
    print("\n2. Ensure FASTQ files are named:")
    print("   - sample1_R1.fastq.gz (Read 1: Cell barcode + UMI)")
    print("   - sample1_R2.fastq.gz (Read 2: cDNA)")
    print("\n3. Build STAR index for hg38 with gene annotations")
    print("\n4. Run the scRNA-seq pipeline:")
    print("   sbatch scripts/submit_scrna_seq.sh")
    
    print("\n✓ Test data download complete!")

if __name__ == "__main__":
    main()
