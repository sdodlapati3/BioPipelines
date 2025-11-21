#!/bin/bash
#
# Download validated test datasets using the BioPipelines download module
#
# This script downloads small, working datasets for all pipeline types
# using known-good SRA accessions from ENA
#

set -e

# Activate conda environment
source ~/miniconda3/bin/activate ~/envs/biopipelines

BASE_DIR="$HOME/BioPipelines/data/raw"

echo "========================================="
echo "BioPipelines - Download Validated Test Data"
echo "========================================="
echo ""
echo "Using BioPipelines data download module"
echo "This will download ~2-3GB of validated test data"
echo ""

# ========================================
# Download using direct ENA FTP (fastest, most reliable)
# ========================================

download_ena() {
    local accession=$1
    local output_dir=$2
    local file_num=$3
    
    echo "  Downloading $accession file $file_num from ENA..."
    
    # ENA FTP URL pattern
    local base_url="ftp://ftp.sra.ebi.ac.uk/vol1/fastq"
    local acc_prefix=$(echo $accession | cut -c 1-6)
    local acc_last=$(echo $accession | rev | cut -c 1-3 | rev)
    
    # Construct URL
    if [ "$file_num" = "1" ]; then
        local url="${base_url}/${acc_prefix}/${acc_last}/${accession}/${accession}_1.fastq.gz"
        local output="${output_dir}/${accession}_R1.fastq.gz"
    elif [ "$file_num" = "2" ]; then
        local url="${base_url}/${acc_prefix}/${acc_last}/${accession}/${accession}_2.fastq.gz"
        local output="${output_dir}/${accession}_R2.fastq.gz"
    else
        local url="${base_url}/${acc_prefix}/${acc_last}/${accession}/${accession}.fastq.gz"
        local output="${output_dir}/${accession}.fastq.gz"
    fi
    
    if [ -f "$output" ]; then
        echo "    ✓ File exists: $(basename $output)"
        return 0
    fi
    
    wget -q --show-progress --tries=3 -O "$output" "$url" || {
        echo "    ✗ Download failed: $url"
        return 1
    }
    
    echo "    ✓ Downloaded: $(basename $output) ($(du -h $output | cut -f1))"
}

# ========================================
# 1. RNA-seq - Small E.coli dataset
# ========================================
echo ""
echo "[1/4] RNA-seq test data (E. coli, paired-end)"
echo "      Small dataset for quick testing (~100MB/file)"

mkdir -p "${BASE_DIR}/rna_seq"
cd "${BASE_DIR}/rna_seq"

# Treatment samples
download_ena "ERR458493" "." "1"  # treat_rep1_R1
download_ena "ERR458493" "." "2"  # treat_rep1_R2
download_ena "ERR458494" "." "1"  # treat_rep2_R1
download_ena "ERR458494" "." "2"  # treat_rep2_R2

# Control samples
download_ena "ERR458495" "." "1"  # ctrl_rep1_R1
download_ena "ERR458495" "." "2"  # ctrl_rep1_R2
download_ena "ERR458496" "." "1"  # ctrl_rep2_R1
download_ena "ERR458496" "." "2"  # ctrl_rep2_R2

# Rename to standard format
[ -f "ERR458493_R1.fastq.gz" ] && mv ERR458493_R1.fastq.gz treat_rep1_R1.fastq.gz
[ -f "ERR458493_R2.fastq.gz" ] && mv ERR458493_R2.fastq.gz treat_rep1_R2.fastq.gz
[ -f "ERR458494_R1.fastq.gz" ] && mv ERR458494_R1.fastq.gz treat_rep2_R1.fastq.gz
[ -f "ERR458494_R2.fastq.gz" ] && mv ERR458494_R2.fastq.gz treat_rep2_R2.fastq.gz
[ -f "ERR458495_R1.fastq.gz" ] && mv ERR458495_R1.fastq.gz ctrl_rep1_R1.fastq.gz
[ -f "ERR458495_R2.fastq.gz" ] && mv ERR458495_R2.fastq.gz ctrl_rep1_R2.fastq.gz
[ -f "ERR458496_R1.fastq.gz" ] && mv ERR458496_R1.fastq.gz ctrl_rep2_R1.fastq.gz
[ -f "ERR458496_R2.fastq.gz" ] && mv ERR458496_R2.fastq.gz ctrl_rep2_R2.fastq.gz

echo "  ✓ RNA-seq data downloaded"

# ========================================
# 2. ChIP-seq - H3K27ac from SRA
# ========================================
echo ""
echo "[2/4] ChIP-seq test data (H3K27ac, single-end)"
echo "      Small human ChIP-seq (~150-200MB/file)"

mkdir -p "${BASE_DIR}/chip_seq"
cd "${BASE_DIR}/chip_seq"

# ChIP samples
download_ena "SRR5344681" "." "0"  # sample1
download_ena "SRR5344682" "." "0"  # sample2

# Input control
download_ena "SRR5344683" "." "0"  # input

# Rename
[ -f "SRR5344681.fastq.gz" ] && mv SRR5344681.fastq.gz sample1.fastq.gz
[ -f "SRR5344682.fastq.gz" ] && mv SRR5344682.fastq.gz sample2.fastq.gz
[ -f "SRR5344683.fastq.gz" ] && mv SRR5344683.fastq.gz input.fastq.gz

echo "  ✓ ChIP-seq data downloaded"

# ========================================
# 3. ATAC-seq - Paired-end
# ========================================
echo ""
echo "[3/4] ATAC-seq test data (paired-end)"
echo "      Human ATAC-seq (~200-300MB/file)"

mkdir -p "${BASE_DIR}/atac_seq"
cd "${BASE_DIR}/atac_seq"

# ATAC samples (paired-end)
download_ena "SRR891268" "." "1"  # sample1_R1
download_ena "SRR891268" "." "2"  # sample1_R2
download_ena "SRR891269" "." "1"  # sample2_R1
download_ena "SRR891269" "." "2"  # sample2_R2

# Rename
[ -f "SRR891268_R1.fastq.gz" ] && mv SRR891268_R1.fastq.gz sample1_R1.fastq.gz
[ -f "SRR891268_R2.fastq.gz" ] && mv SRR891268_R2.fastq.gz sample1_R2.fastq.gz
[ -f "SRR891269_R1.fastq.gz" ] && mv SRR891269_R1.fastq.gz sample2_R1.fastq.gz
[ -f "SRR891269_R2.fastq.gz" ] && mv SRR891269_R2.fastq.gz sample2_R2.fastq.gz

echo "  ✓ ATAC-seq data downloaded"

# ========================================
# 4. DNA-seq - Keep existing if good
# ========================================
echo ""
echo "[4/4] DNA-seq data"

if [ -f "${BASE_DIR}/dna_seq/sample1_R1.fastq.gz" ]; then
    echo "  ✓ DNA-seq data already exists"
else
    echo "  ℹ DNA-seq: Use existing large files or download separately"
fi

# ========================================
# Summary
# ========================================
echo ""
echo "========================================="
echo "✓ Download Complete!"
echo "========================================="
echo ""
echo "📊 Downloaded datasets:"
echo ""

for dir in rna_seq chip_seq atac_seq; do
    if [ -d "${BASE_DIR}/${dir}" ]; then
        echo "$dir:"
        ls -lh "${BASE_DIR}/${dir}"/*.fastq.gz 2>/dev/null | awk '{print "  " $9 ": " $5}' || echo "  (no files)"
        echo ""
    fi
done

echo "📦 Total size:"
du -sh "${BASE_DIR}"

echo ""
echo "🎯 Next steps:"
echo "  1. Update pipeline configs with new sample names"
echo "  2. Run pipelines: sbatch scripts/submit_*.sh"
echo "  3. Monitor with: squeue --me"
echo ""
echo "💡 Note: These are smaller test datasets for validation"
echo "   For production analysis, download full-size datasets"
echo ""
