# BioPipelines GCP Storage Architecture

## Current Architecture Issues ❌

The current setup stores everything in the home directory:
- ❌ References: `~/references/` (~38 GB)
- ❌ Data: `~/BioPipelines/data/` (~50+ GB per run)
- ❌ Results: `~/BioPipelines/data/results/`

**Problems:**
1. Limited home directory quota
2. Data not persistent across cluster resets
3. Hard to share data between team members
4. No backup/versioning
5. Slow for large datasets

---

## Recommended GCP Architecture ✅

### Storage Strategy

```
GCS Buckets (Persistent, Shared)
├── gs://biopipelines-references/          # Reference genomes (read-only)
│   ├── genomes/hg38/
│   ├── annotations/
│   └── known_sites/
│
├── gs://biopipelines-data/                # Input datasets (read-only)
│   ├── dna_seq/
│   ├── rna_seq/
│   ├── chip_seq/
│   └── atac_seq/
│
└── gs://biopipelines-results-[PROJECT]/   # Output results (read-write)
    ├── dna_seq/
    ├── rna_seq/
    └── logs/

Compute Node Local Storage (Temporary, Fast)
├── /mnt/disks/scratch/                    # Fast local SSD
│   └── [job_id]/                          # Job-specific temp directory
│       ├── input/                         # Staged from GCS
│       ├── working/                       # Intermediate files
│       └── output/                        # Final results (copy to GCS)
│
└── ~/BioPipelines/                        # Code only
    └── pipelines/
```

---

## Implementation Plan

### Phase 1: Create GCS Buckets

```bash
# Set project
PROJECT_ID="rcc-hpc"
REGION="us-central1"

# Create buckets
gsutil mb -p ${PROJECT_ID} -l ${REGION} gs://biopipelines-references/
gsutil mb -p ${PROJECT_ID} -l ${REGION} gs://biopipelines-data/
gsutil mb -p ${PROJECT_ID} -l ${REGION} gs://biopipelines-results-${PROJECT_ID}/

# Set lifecycle policies for results bucket (auto-delete old results)
cat > lifecycle.json <<EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 90}
      }
    ]
  }
}
EOF

gsutil lifecycle set lifecycle.json gs://biopipelines-results-${PROJECT_ID}/

# Set permissions
gsutil iam ch allAuthenticatedUsers:objectViewer gs://biopipelines-references/
```

### Phase 2: Upload References to GCS

```bash
# Upload from cluster or local machine
cd ~/references
gsutil -m rsync -r genomes/ gs://biopipelines-references/genomes/
gsutil -m rsync -r annotations/ gs://biopipelines-references/annotations/
gsutil -m rsync -r known_sites/ gs://biopipelines-references/known_sites/

# Verify
gsutil du -sh gs://biopipelines-references/
```

### Phase 3: Update Scripts to Use GCS

Create `scripts/gcp_stage_data.sh`:

```bash
#!/bin/bash
# Stage data from GCS to local SSD for fast processing

JOB_ID=${SLURM_JOB_ID:-$$}
SCRATCH_DIR="/mnt/disks/scratch/${JOB_ID}"
GCS_DATA_BUCKET="gs://biopipelines-data"
GCS_REF_BUCKET="gs://biopipelines-references"

# Create working directory
mkdir -p ${SCRATCH_DIR}/{input,working,output,references}

# Stage input data
echo "Staging input data from GCS..."
gsutil -m rsync -r ${GCS_DATA_BUCKET}/dna_seq/ ${SCRATCH_DIR}/input/

# Stage references (or mount via gcsfuse)
echo "Staging references..."
gsutil -m rsync -r ${GCS_REF_BUCKET}/genomes/hg38/ ${SCRATCH_DIR}/references/genome/
gsutil -m rsync -r ${GCS_REF_BUCKET}/known_sites/ ${SCRATCH_DIR}/references/known_sites/

echo "Data staging complete!"
echo "Working directory: ${SCRATCH_DIR}"
```

### Phase 4: Update Pipelines

Update config paths to use local scratch:

```yaml
# config.yaml
samples:
  - sample1

reference:
  genome: "/mnt/disks/scratch/{job_id}/references/genome/hg38.fa"
  known_sites: "/mnt/disks/scratch/{job_id}/references/known_sites/dbsnp_155.hg38.vcf.gz"

input_dir: "/mnt/disks/scratch/{job_id}/input"
output_dir: "/mnt/disks/scratch/{job_id}/output"
```

### Phase 5: Update Submit Script

```bash
#!/bin/bash
#SBATCH --job-name=dna_seq_pipeline
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=4:00:00

# Define paths
JOB_ID=${SLURM_JOB_ID}
SCRATCH_DIR="/mnt/disks/scratch/${JOB_ID}"
GCS_RESULTS="gs://biopipelines-results-rcc-hpc/dna_seq/${JOB_ID}"

# Activate environment
source ~/miniconda3/bin/activate ~/envs/biopipelines

# Stage data from GCS
bash ~/BioPipelines/scripts/gcp_stage_data.sh

# Run pipeline
cd ~/BioPipelines/pipelines/dna_seq/variant_calling
snakemake --cores ${SLURM_CPUS_PER_TASK} --use-conda

# Copy results back to GCS
echo "Copying results to GCS..."
gsutil -m rsync -r ${SCRATCH_DIR}/output/ ${GCS_RESULTS}/

# Cleanup
rm -rf ${SCRATCH_DIR}

echo "Results available at: ${GCS_RESULTS}"
```

---

## Alternative: gcsfuse (Recommended)

Mount GCS buckets as filesystems for easier access:

```bash
# Install gcsfuse on cluster
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install gcsfuse

# Mount in job script
mkdir -p ~/gcs/{references,data,results}
gcsfuse --implicit-dirs biopipelines-references ~/gcs/references
gcsfuse --implicit-dirs biopipelines-data ~/gcs/data
gcsfuse --implicit-dirs biopipelines-results-rcc-hpc ~/gcs/results

# Use mounted paths in config
reference:
  genome: "/home/$USER/gcs/references/genomes/hg38/hg38.fa"
```

---

## Cost Optimization

### Storage Costs
- **Standard**: $0.020/GB/month (for active data)
- **Nearline**: $0.010/GB/month (accessed <1/month)
- **Coldline**: $0.004/GB/month (accessed <1/quarter)

**Strategy:**
- References → Standard (frequently accessed)
- Input data → Nearline (accessed per-project)
- Results → Coldline after 30 days (archive)

### Transfer Costs
- Ingress (upload): Free
- Egress within region: Free
- Egress to internet: $0.12/GB

**Optimization:**
- Keep cluster and buckets in same region (us-central1)
- Use gsutil -m for parallel transfers
- Stage to local SSD for compute-intensive operations

---

## Migration Checklist

- [ ] Create GCS buckets (references, data, results)
- [ ] Upload reference genomes to GCS (~38 GB)
- [ ] Upload test datasets to GCS (~10 GB)
- [ ] Create gcp_stage_data.sh script
- [ ] Update pipeline configs for GCS paths
- [ ] Update submit scripts with staging logic
- [ ] Test data staging performance
- [ ] Set up gcsfuse (optional but recommended)
- [ ] Update documentation with GCS instructions
- [ ] Set up lifecycle policies for auto-cleanup
- [ ] Configure IAM permissions for team access

---

## Performance Comparison

| Storage | Read Speed | Write Speed | Best For |
|---------|-----------|-------------|----------|
| Local SSD | 1000+ MB/s | 1000+ MB/s | Active compute |
| GCS (direct) | 200-600 MB/s | 200-600 MB/s | Storage/archive |
| gcsfuse | 50-200 MB/s | 50-200 MB/s | Convenience |
| Home dir (~/) | 100-500 MB/s | 100-500 MB/s | Small files |

**Recommendation:** Stage from GCS to local SSD for best performance.

---

## Next Steps

1. Create GCS buckets
2. Upload references once (shared by all)
3. Update download scripts to upload to GCS
4. Modify submit scripts for staging
5. Test with one sample
6. Document new workflow

