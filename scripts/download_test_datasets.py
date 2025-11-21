#!/usr/bin/env python3
"""
Download validated test datasets using BioPipelines download module

This script downloads small, working datasets for testing all pipelines.
Uses the pysradb package to get ENA FTP URLs and download directly.
"""

import subprocess
import sys
from pathlib import Path

def download_with_wget(url, output_path):
    """Download file using wget"""
    if output_path.exists():
        print(f"  ✓ File exists: {output_path.name}")
        return True
    
    print(f"  Downloading: {output_path.name}")
    try:
        cmd = ["wget", "-q", "--show-progress", "--tries=3", "-O", str(output_path), url]
        subprocess.run(cmd, check=True)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Downloaded: {output_path.name} ({size_mb:.1f} MB)")
        return True
    except subprocess.CalledProcessError:
        print(f"  ✗ Download failed: {url}")
        if output_path.exists():
            output_path.unlink()
        return False

def get_ena_urls(accession):
    """Get ENA FTP URLs for an SRA accession using pysradb"""
    try:
        import pysradb
        db = pysradb.SRAweb()
        metadata = db.sra_metadata(accession, detailed=True)
        
        if metadata.empty:
            return []
        
        fastq_ftp = metadata['fastq_ftp'].iloc[0]
        if isinstance(fastq_ftp, str) and fastq_ftp:
            urls = [f"ftp://{url}" for url in fastq_ftp.split(';')]
            return urls
        return []
    except Exception as e:
        print(f"  Warning: Could not get ENA URLs for {accession}: {e}")
        return []

def main():
    base_dir = Path.home() / "BioPipelines" / "data" / "raw"
    
    print("=" * 60)
    print("BioPipelines - Download Validated Test Data")
    print("=" * 60)
    print()
    print("Downloading small datasets for pipeline testing")
    print()
    
    # RNA-seq - E. coli
    print("[1/3] RNA-seq test data (E. coli)")
    rna_dir = base_dir / "rna_seq"
    rna_dir.mkdir(parents=True, exist_ok=True)
    
    rna_samples = {
        "ERR458493": ("treat_rep1_R1.fastq.gz", "treat_rep1_R2.fastq.gz"),
        "ERR458494": ("treat_rep2_R1.fastq.gz", "treat_rep2_R2.fastq.gz"),
        "ERR458495": ("ctrl_rep1_R1.fastq.gz", "ctrl_rep1_R2.fastq.gz"),
        "ERR458496": ("ctrl_rep2_R1.fastq.gz", "ctrl_rep2_R2.fastq.gz"),
    }
    
    for accession, (r1_name, r2_name) in rna_samples.items():
        urls = get_ena_urls(accession)
        if len(urls) >= 2:
            download_with_wget(urls[0], rna_dir / r1_name)
            download_with_wget(urls[1], rna_dir / r2_name)
        else:
            print(f"  ⚠ Could not get URLs for {accession}")
    
    # ChIP-seq
    print()
    print("[2/3] ChIP-seq test data (H3K27ac)")
    chip_dir = base_dir / "chip_seq"
    chip_dir.mkdir(parents=True, exist_ok=True)
    
    chip_samples = {
        "SRR5344681": "sample1.fastq.gz",
        "SRR5344682": "sample2.fastq.gz",
        "SRR5344683": "input.fastq.gz",
    }
    
    for accession, filename in chip_samples.items():
        urls = get_ena_urls(accession)
        if urls:
            download_with_wget(urls[0], chip_dir / filename)
        else:
            print(f"  ⚠ Could not get URL for {accession}")
    
    # ATAC-seq
    print()
    print("[3/3] ATAC-seq test data (paired-end)")
    atac_dir = base_dir / "atac_seq"
    atac_dir.mkdir(parents=True, exist_ok=True)
    
    atac_samples = {
        "SRR891268": ("sample1_R1.fastq.gz", "sample1_R2.fastq.gz"),
        "SRR891269": ("sample2_R1.fastq.gz", "sample2_R2.fastq.gz"),
    }
    
    for accession, (r1_name, r2_name) in atac_samples.items():
        urls = get_ena_urls(accession)
        if len(urls) >= 2:
            download_with_wget(urls[0], atac_dir / r1_name)
            download_with_wget(urls[1], atac_dir / r2_name)
        else:
            print(f"  ⚠ Could not get URLs for {accession}")
    
    # Summary
    print()
    print("=" * 60)
    print("✓ Download Complete!")
    print("=" * 60)
    print()
    
    for subdir in ["rna_seq", "chip_seq", "atac_seq"]:
        data_dir = base_dir / subdir
        if data_dir.exists():
            files = list(data_dir.glob("*.fastq.gz"))
            if files:
                print(f"{subdir}:")
                for f in sorted(files):
                    size_mb = f.stat().st_size / (1024 * 1024)
                    print(f"  {f.name}: {size_mb:.1f} MB")
                print()
    
    print("🎯 Next steps:")
    print("  1. Update pipeline configs with new sample names")
    print("  2. Run pipelines: sbatch scripts/submit_*.sh")
    print()

if __name__ == "__main__":
    main()
