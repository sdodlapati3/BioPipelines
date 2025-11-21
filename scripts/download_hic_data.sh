#!/bin/bash
#SBATCH --job-name=download_hic
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=3:00:00

echo "Downloading Hi-C data using proper Python module"
echo "Job ID: $SLURM_JOB_ID"

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/sdodl001_odu_edu/envs/biopipelines

cd ~/BioPipelines

# Download smaller, cleaner Hi-C dataset
python << 'PYTHON_SCRIPT'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

from biopipelines.data_download.hic_downloader import HiCDownloader

output_dir = Path("data/raw/hic")
output_dir.mkdir(parents=True, exist_ok=True)

downloader = HiCDownloader(output_dir=str(output_dir))

print("=" * 80)
print("Downloading Hi-C experiment: ENCSR862OGI")
print("Cell type: GM12878 (lymphoblastoid)")
print("Smaller dataset for testing (~15GB total)")
print("=" * 80)

try:
    # Remove old corrupted files
    import glob, os
    for f in glob.glob(str(output_dir / "*.fastq.gz")):
        os.remove(f)
        print(f"Removed old file: {f}")
    
    files = downloader.download_encode_hic(
        experiment_id="ENCSR862OGI",
        file_type="fastq"
    )
    
    print(f"\n✓ Downloaded {len(files)} files")
    
except Exception as e:
    print(f"\n✗ Failed: {e}")
    sys.exit(1)

PYTHON_SCRIPT

echo "Download complete: $(date)"
