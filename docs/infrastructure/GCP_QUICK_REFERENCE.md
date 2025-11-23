# BioPipelines GCP Quick Reference

## 🚀 Quick Commands

### Connect to Cluster
```bash
gcloud compute ssh hpcslurm-slurm-login-001 \
  --project=rcc-hpc \
  --zone=us-central1-a \
  --tunnel-through-iap
```

### Create GCS Buckets (One-time)
```bash
gsutil mb -l us-central1 gs://biopipelines-data/
gsutil mb -l us-central1 gs://biopipelines-references/
gsutil mb -l us-central1 gs://biopipelines-results-rcc-hpc/
```

### Upload Test Data
```bash
# From local machine or cluster
./scripts/download_test_data.sh
```

### Submit Pipeline Job
```bash
# On cluster
cd ~/BioPipelines
sbatch scripts/submit_dna_seq.sh
```

### Monitor Job
```bash
squeue -u $USER                    # Check job status
tail -f slurm_*.out                # View output
sacct -j JOB_ID --format=...      # Job history
```

### Check Results
```bash
# List results in GCS
gsutil ls gs://biopipelines-results-rcc-hpc/dna_seq/

# Download results
gsutil -m rsync -r gs://biopipelines-results-rcc-hpc/dna_seq/[JOB_ID]/ ./results/
```

---

## 📂 Key Paths

### GCS Buckets
- **Data:** `gs://biopipelines-data/[pipeline]/test/`
- **References:** `gs://biopipelines-references/genomes/hg38/`
- **Results:** `gs://biopipelines-results-rcc-hpc/[pipeline]/[JOB_ID]/`

### Cluster Paths
- **Code:** `~/BioPipelines/`
- **Environment:** `~/envs/biopipelines`
- **Scratch:** `/mnt/disks/scratch/[JOB_ID]/`
- **Shared Cache:** `/mnt/disks/scratch/shared/references/`

---

## 🔧 Common Tasks

### Update Code on Cluster
```bash
# Option 1: Git pull
cd ~/BioPipelines
git pull

# Option 2: SCP from local
gcloud compute scp --recurse ~/path/to/BioPipelines \
  username@hpcslurm-slurm-login-001:~/ \
  --project=rcc-hpc --zone=us-central1-a --tunnel-through-iap
```

### Upload New Dataset
```bash
gsutil -m cp sample*.fastq.gz gs://biopipelines-data/dna_seq/test/
```

### Clean Up Old Results
```bash
# List old jobs
gsutil ls gs://biopipelines-results-rcc-hpc/dna_seq/

# Delete specific job
gsutil -m rm -r gs://biopipelines-results-rcc-hpc/dna_seq/12345678/
```

### Check Storage Usage
```bash
gsutil du -sh gs://biopipelines-data/
gsutil du -sh gs://biopipelines-references/
gsutil du -sh gs://biopipelines-results-rcc-hpc/
```

---

## 🐛 Troubleshooting

### Job Failed
```bash
# Check error log
cat slurm_[JOB_ID].err

# Check Snakemake log
cat /mnt/disks/scratch/[JOB_ID]/logs/*.log

# Re-run with debug
snakemake --dry-run -p
```

### Out of Memory
```bash
# Edit submit script, increase memory
#SBATCH --mem=200G
```

### Staging Failed
```bash
# Check GCS access
gsutil ls gs://biopipelines-data/

# Re-authenticate
gcloud auth login
```

### Conda Environment Issues
```bash
# Recreate environment
conda env remove -p ~/envs/biopipelines
conda env create -f environment.yml -p ~/envs/biopipelines
```

---

## 📊 Resource Recommendations

| Pipeline | CPUs | Memory | Time | Storage |
|----------|------|--------|------|---------|
| DNA-seq  | 16   | 100GB  | 4h   | 50GB    |
| RNA-seq  | 16   | 64GB   | 2h   | 30GB    |
| ChIP-seq | 8    | 32GB   | 1h   | 20GB    |
| ATAC-seq | 8    | 32GB   | 1h   | 20GB    |

---

## 📚 Documentation

- **Setup:** `docs/GCP_HPC_SETUP.md`
- **Storage:** `docs/GCP_STORAGE_ARCHITECTURE.md`
- **Architecture:** `GCP_ARCHITECTURE_CORRECTED.md`
- **Next Steps:** `NEXT_STEPS.md`
- **Tasks:** `TODO.md`

---

## 🆘 Help

**Can't connect to cluster?**
```bash
gcloud auth login
gcloud config set project rcc-hpc
```

**Can't find files in GCS?**
```bash
gsutil ls gs://biopipelines-data/
# If empty, run: ./scripts/download_test_data.sh
```

**Job stuck in queue?**
```bash
squeue -u $USER
# Check partition: sinfo
# Try different partition: #SBATCH --partition=debugspot
```

---

*Last updated: November 20, 2025*

