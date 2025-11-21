#!/bin/bash
#SBATCH --job-name=download_metagenomics
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=cpuspot

echo "=========================================="
echo "Downloading Metagenomics Test Data"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Create output directory
OUTPUT_DIR="/scratch/sdodl001/BioPipelines/data/raw/metagenomics"
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# Download from ENA (faster than SRA)
echo "Dataset: Human gut microbiome (SRR1927149)"
echo "Source: ENA FTP"
echo "Expected size: ~350MB per file"
echo ""

# Download Read 1
if [ ! -f "sample1_R1.fastq.gz" ]; then
    echo "Downloading Read 1..."
    wget -q --show-progress \
        ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR192/009/SRR1927149/SRR1927149_1.fastq.gz \
        -O sample1_R1.fastq.gz
    
    if [ $? -eq 0 ]; then
        echo "✓ Read 1 downloaded"
    else
        echo "✗ Read 1 download failed"
        exit 1
    fi
else
    echo "✓ Read 1 already exists"
fi

echo ""

# Download Read 2
if [ ! -f "sample1_R2.fastq.gz" ]; then
    echo "Downloading Read 2..."
    wget -q --show-progress \
        ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR192/009/SRR1927149/SRR1927149_2.fastq.gz \
        -O sample1_R2.fastq.gz
    
    if [ $? -eq 0 ]; then
        echo "✓ Read 2 downloaded"
    else
        echo "✗ Read 2 download failed"
        exit 1
    fi
else
    echo "✓ Read 2 already exists"
fi

echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""
echo "Files:"
ls -lh sample1_R*.fastq.gz

echo ""
echo "Verify files:"
zcat sample1_R1.fastq.gz | head -4
echo "..."
echo ""

echo "✓ Ready for metagenomics pipeline"
echo "Next: sbatch scripts/submit_metagenomics.sh"
echo ""
echo "End time: $(date)"
