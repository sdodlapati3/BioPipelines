# BioPipelines GCP HPC Setup Tutorial

Complete step-by-step guide for deploying BioPipelines on GCP HPC Slurm cluster.

---

## Prerequisites

- GCP account with HPC cluster access
- `gcloud` CLI installed and configured
- Local BioPipelines repository cloned

---

## Step 1: Connect to GCP HPC Cluster

```bash
# Connect using IAP tunneling with OS Login
gcloud compute ssh hpcslurm-slurm-login-001 \
  --project=rcc-hpc \
  --zone=us-central1-a \
  --tunnel-through-iap
```

---

## Step 2: Install Miniconda

```bash
# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install to ~/miniconda3
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3

# Initialize conda
~/miniconda3/bin/conda init

# Activate base environment
source ~/.bashrc
```

---

## Step 3: Transfer BioPipelines Repository

```bash
# From your local machine, transfer entire repository
gcloud compute scp --recurse \
  --project=rcc-hpc \
  --zone=us-central1-a \
  --tunnel-through-iap \
  /path/to/local/BioPipelines \
  username@hpcslurm-slurm-login-001:~/
```

---

## Step 4: Create BioPipelines Conda Environment

### Fix environment.yml (if needed)

Update `environment.yml` to fix dependency conflicts:

```yaml
# Change numpy version for deeptools compatibility
- numpy>=1.21,<1.24  # was: numpy>=1.24

# Comment out unavailable packages
# - vep=110  # Not available in current channels
```

### Install mamba (required for Snakemake)

```bash
source ~/miniconda3/bin/activate base
conda install -n base -c conda-forge mamba -y
```

### Create environment

```bash
cd ~/BioPipelines
conda env create -f environment.yml -p ~/envs/biopipelines -y
```

**Expected time**: 30-60 minutes  
**Components installed**: GATK, BWA, STAR, MACS2, DESeq2, Snakemake, and 80+ tools

---

## Step 5: Download Reference Genomes

### Create download script

Transfer `scripts/download_references.sh` to cluster, or create it:

```bash
chmod +x ~/BioPipelines/scripts/download_references.sh

# Start download in background
cd ~/BioPipelines
nohup bash scripts/download_references.sh > reference_download.log 2>&1 &
```

### What gets downloaded:
- hg38 reference genome (3.1 GB)
- BWA index files (4.8 GB)
- dbSNP known sites (28 GB)
- GENCODE gene annotations (1.5 GB)

**Total size**: ~38 GB  
**Expected time**: 1-3 hours  
**Location**: `~/references/`

### Monitor progress:

```bash
tail -f ~/BioPipelines/reference_download.log
```

---

## Step 6: Download Test Data

### DNA-seq test data (NA12878 exome)

```bash
mkdir -p ~/BioPipelines/data/raw/dna_seq
cd ~/BioPipelines/data/raw/dna_seq

# Download paired-end exome data
wget -c 'https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/NA12878/Garvan_NA12878_HG001_HiSeq_Exome/NIST7035_TAAGGCGA_L001_R1_001.fastq.gz' \
  -O sample1_R1.fastq.gz

wget -c 'https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/NA12878/Garvan_NA12878_HG001_HiSeq_Exome/NIST7035_TAAGGCGA_L001_R2_001.fastq.gz' \
  -O sample1_R2.fastq.gz

# Move to correct location
mv ~/BioPipelines/data/raw/dna_seq/*.fastq.gz ~/BioPipelines/data/raw/
```

**Size**: 3.7 GB (paired-end)

---

## Step 7: Configure DNA-seq Pipeline

### Update config file

Edit `~/BioPipelines/pipelines/dna_seq/variant_calling/config.yaml`:

```yaml
samples:
  - sample1

reference:
  genome: "/home/YOUR_USERNAME/references/genomes/hg38/hg38.fa"
  known_sites: "/home/YOUR_USERNAME/references/known_sites/dbsnp_155.hg38.vcf.gz"

snpeff_genome: "hg38"

align:
  threads: 16  # Use all available CPUs

variant_calling:
  threads: 8
```

---

## Step 8: Create Slurm Submission Script

Create `~/BioPipelines/scripts/submit_dna_seq.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=dna_seq_pipeline
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

echo "BioPipelines DNA-seq Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

# Activate conda environment
source ~/miniconda3/bin/activate ~/envs/biopipelines

# Run pipeline
cd ~/BioPipelines/pipelines/dna_seq/variant_calling

snakemake \
    --cores $SLURM_CPUS_PER_TASK \
    --use-conda \
    --conda-frontend conda \
    --printshellcmds \
    --keep-going \
    --rerun-incomplete

echo "Pipeline complete!"
```

```bash
chmod +x ~/BioPipelines/scripts/submit_dna_seq.sh
```

---

## Step 9: Submit Pipeline to Slurm

### Validate workflow (dry-run)

```bash
cd ~/BioPipelines/pipelines/dna_seq/variant_calling
source ~/miniconda3/bin/activate ~/envs/biopipelines
snakemake --dry-run --cores 4
```

**Expected output**: Lists 11 jobs to be executed

### Clean Snakemake cache (if rerunning)

```bash
rm -rf ~/BioPipelines/pipelines/dna_seq/variant_calling/.snakemake
```

### Submit job

```bash
sbatch ~/BioPipelines/scripts/submit_dna_seq.sh
```

### Monitor job

```bash
# Check queue
squeue -u $USER

# Check job details
squeue -j JOB_ID

# View output
tail -f ~/BioPipelines/pipelines/dna_seq/variant_calling/slurm_JOB_ID.out

# Check for errors
tail -f ~/BioPipelines/pipelines/dna_seq/variant_calling/slurm_JOB_ID.err

# View job history
sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS
```

---

## Step 10: Verify Results

### Check output files

```bash
# Results directory
ls -lh ~/BioPipelines/data/results/

# VCF files
ls -lh ~/BioPipelines/data/results/vcf/

# QC reports
ls -lh ~/BioPipelines/data/results/multiqc_report.html
```

**Expected outputs**:
- Annotated VCF file with variants
- MultiQC HTML report
- Processing logs in `data/results/qc/logs/`

---

## Pipeline Steps Executed

The DNA-seq pipeline runs 11 jobs:

1. **fastqc_raw**: Quality control on raw reads
2. **trim_reads**: Adapter trimming with fastp
3. **align_reads**: BWA-MEM alignment to hg38
4. **mark_duplicates**: PCR duplicate marking (Picard)
5. **base_recalibration**: GATK BaseRecalibrator
6. **apply_bqsr**: Apply base quality score recalibration
7. **call_variants**: GATK HaplotypeCaller
8. **filter_variants**: Hard filtering
9. **annotate_variants**: SnpEff annotation
10. **multiqc**: Aggregate QC report
11. **all**: Final target rule

---

## Troubleshooting

### Conda environment issues

```bash
# If mamba fails, use conda
snakemake --use-conda --conda-frontend conda ...

# Clean corrupted conda cache
rm -rf ~/.conda/pkgs/*
conda clean --all
```

### Slurm partition issues

```bash
# Check available partitions
sinfo -O partitions,nodes,cpus,memory,timelimit

# Use different partition
#SBATCH --partition=debugspot  # For testing (2 CPUs)
#SBATCH --partition=cpuspot    # For production (16 CPUs)
```

### Out of memory

Increase memory request:
```bash
#SBATCH --mem=200G
```

### Job time limit exceeded

Increase time:
```bash
#SBATCH --time=8:00:00
```

---

## Resource Requirements

### DNA-seq Pipeline (Exome, 3.7GB)

- CPUs: 16 recommended (min: 4)
- Memory: 100GB recommended (min: 32GB)
- Time: ~30-60 minutes
- Disk space: ~50GB for outputs

### RNA-seq Pipeline (per sample pair)

- CPUs: 16 recommended
- Memory: 64GB recommended
- Time: ~45-90 minutes
- Disk space: ~30GB per sample

### ChIP-seq Pipeline (per sample)

- CPUs: 8 recommended
- Memory: 32GB recommended
- Time: ~20-40 minutes
- Disk space: ~20GB per sample

---

## Quick Reference Commands

```bash
# Connect to cluster
gcloud compute ssh hpcslurm-slurm-login-001 --project=rcc-hpc --zone=us-central1-a --tunnel-through-iap

# Activate environment
source ~/miniconda3/bin/activate ~/envs/biopipelines

# Submit job
cd ~/BioPipelines/pipelines/dna_seq/variant_calling
sbatch ~/BioPipelines/scripts/submit_dna_seq.sh

# Monitor
squeue -u $USER
tail -f slurm_*.out

# Check results
ls -lh ~/BioPipelines/data/results/
```

---

## Next Steps

After DNA-seq pipeline completes:

1. ✅ Review MultiQC report
2. ✅ Inspect variant calls (VCF file)
3. 🔄 Test RNA-seq pipeline
4. 🔄 Test ChIP-seq pipeline
5. 🔄 Test ATAC-seq pipeline
6. 📝 Create unit tests
7. 📝 Add more demo datasets

---

## Notes

- **First run**: Takes longer due to conda environment creation (5-10 min setup)
- **Subsequent runs**: Much faster as conda environments are cached
- **Storage**: Keep `~/references/` - reused across all pipelines
- **Cost**: Use `cpuspot` partition for cost-effective spot instances
- **Login node**: Never run pipelines on login node - always use Slurm!
