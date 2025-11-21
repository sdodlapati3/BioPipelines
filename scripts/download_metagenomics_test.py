#!/usr/bin/env python3
"""
Download small metagenomics test dataset
Human gut microbiome sample for quick pipeline testing
"""

import sys
import subprocess
import requests
from pathlib import Path

def download_file(url, output_path):
    """Download file with progress"""
    print(f"Downloading from ENA: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    block_size = 8192
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    mb = downloaded / 1024 / 1024
                    print(f"\r  Progress: {percent:.1f}% ({mb:.1f} MB)", end='', flush=True)
    print("\n  ✓ Complete")


def main():
    # Setup output directory
    output_dir = Path(__file__).parent.parent / "data" / "raw" / "metagenomics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Downloading Metagenomics Test Dataset")
    print("=" * 80)
    print("\nDataset: Human gut microbiome (SRR1927149)")
    print("Source: ENA (European Nucleotide Archive)")
    print("Size: ~300-400 MB per file (subsampled)")
    print("Technology: Illumina HiSeq")
    print()
    
    # Use ENA for faster download (already subsampled)
    # SRR1927149: Human gut metagenome, good quality
    accession = "SRR1927149"
    
    # ENA FTP links (faster than SRA)
    base_url = f"ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR192/009/{accession}"
    r1_url = f"{base_url}/{accession}_1.fastq.gz"
    r2_url = f"{base_url}/{accession}_2.fastq.gz"
    
    r1_path = output_dir / "sample1_R1.fastq.gz"
    r2_path = output_dir / "sample1_R2.fastq.gz"
    
    try:
        if not r1_path.exists():
            print("Downloading Read 1...")
            download_file(r1_url, r1_path)
        else:
            print(f"✓ Read 1 already exists: {r1_path}")
        
        if not r2_path.exists():
            print("\nDownloading Read 2...")
            download_file(r2_url, r2_path)
        else:
            print(f"✓ Read 2 already exists: {r2_path}")
        
        print("\n" + "=" * 80)
        print("✓ Download Complete!")
        print("=" * 80)
        print(f"\nFiles in: {output_dir}/")
        
        # Show file sizes
        if r1_path.exists() and r2_path.exists():
            r1_mb = r1_path.stat().st_size / 1024 / 1024
            r2_mb = r2_path.stat().st_size / 1024 / 1024
            print(f"  sample1_R1.fastq.gz: {r1_mb:.1f} MB")
            print(f"  sample1_R2.fastq.gz: {r2_mb:.1f} MB")
        
        print("\n📋 Next steps:")
        print("1. Verify Kraken2 DB exists:")
        print("   ls /scratch/.../kraken2_db/hash.k2d")
        print("2. Update config if needed:")
        print("   pipelines/metagenomics/taxonomic_profiling/config.yaml")
        print("3. Submit pipeline:")
        print("   sbatch scripts/submit_metagenomics.sh")
        print()
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nAlternative: Use wget directly:")
        print(f"  wget {r1_url} -O {r1_path}")
        print(f"  wget {r2_url} -O {r2_path}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
