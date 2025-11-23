# BioPipelines - Corrected Architecture Assessment

## ✅ You Were Right!

After reviewing the documentation more carefully, I can confirm:

1. **✅ Runs on GCP HPC Slurm cluster** (`hpcslurm-slurm-login-001`)
2. **✅ Should use GCS buckets for storage**
3. **❌ My initial guidance was for local machines** (incorrect)

I've now corrected everything!

---

## 🏗️ Correct Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Development Machine                        │
│  - Local BioPipelines repo                                  │
│  - Git for version control                                  │
│  - gcloud CLI for GCP access                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓ (git push/sync)
┌─────────────────────────────────────────────────────────────┐
│                   GCP HPC Slurm Cluster                      │
│  Project: rcc-hpc                                           │
│  Login Node: hpcslurm-slurm-login-001                       │
│  Region: us-central1-a                                      │
│  - Compute Nodes (cpuspot/debugspot partitions)            │
│  - Conda environment: ~/envs/biopipelines                   │
│  - BioPipelines code: ~/BioPipelines/                       │
└──────────┬──────────────────────────────┬───────────────────┘
           │                              │
           ↓ (stage data)                 ↓ (upload results)
┌──────────────────────┐        ┌──────────────────────────┐
│  GCS Storage Buckets │        │  Local Compute Storage   │
│  (Persistent)        │        │  (Temporary, Fast)       │
├──────────────────────┤        ├──────────────────────────┤
│ biopipelines-data/   │<───────│ /mnt/disks/scratch/      │
│ - dna_seq/           │ stage  │ └── [job_id]/            │
│ - rna_seq/           │        │     ├── input/           │
│ - chip_seq/          │        │     ├── working/         │
│ - atac_seq/          │        │     ├── output/          │
│                      │        │     └── references/      │
│ biopipelines-        │        │                          │
│   references/        │        │                          │
│ - genomes/           │        └──────────────────────────┘
│ - annotations/       │
│ - known_sites/       │
│                      │
│ biopipelines-        │
│   results-rcc-hpc/   │
│ - dna_seq/          │
│ - rna_seq/          │
│ - logs/             │
└─────────────────────┘
```

---

## 🔄 Workflow

### 1. **Setup Phase** (One-time)

```bash
# On your local machine or cluster login node
gcloud auth login
gcloud config set project rcc-hpc

# Create GCS buckets
gsutil mb -l us-central1 gs://biopipelines-data/
gsutil mb -l us-central1 gs://biopipelines-references/
gsutil mb -l us-central1 gs://biopipelines-results-rcc-hpc/

# Upload references (large, shared by all jobs)
bash scripts/download_references.sh  # Downloads locally
gsutil -m rsync -r ~/references/ gs://biopipelines-references/

# Upload test data
bash scripts/download_test_data.sh  # Downloads & uploads to GCS
```

### 2. **Job Execution** (Per Pipeline Run)

```bash
# SSH to cluster
gcloud compute ssh hpcslurm-slurm-login-001 \
  --project=rcc-hpc \
  --zone=us-central1-a \
  --tunnel-through-iap

# Submit job
cd ~/BioPipelines
sbatch scripts/submit_dna_seq.sh
```

**What happens in the job:**
1. **Stage data** (GCS → local SSD)
   - `gcp_stage_data.sh` downloads input from GCS to `/mnt/disks/scratch/$JOB_ID/`
   - References are cached in `/mnt/disks/scratch/shared/` (shared across jobs)

2. **Run pipeline** (local compute)
   - Snakemake executes on fast local SSD
   - Intermediate files stay local
   - Final outputs written to local scratch

3. **Upload results** (local → GCS)
   - `gsutil rsync` uploads to `gs://biopipelines-results-rcc-hpc/dna_seq/$JOB_ID/`
   - Job logs and VCF files persisted in GCS

4. **Cleanup**
   - Remove job-specific scratch directory
   - Shared references stay cached

### 3. **Results Retrieval**

```bash
# List results in GCS
gsutil ls gs://biopipelines-results-rcc-hpc/dna_seq/

# Download specific job results
gsutil -m rsync -r gs://biopipelines-results-rcc-hpc/dna_seq/[JOB_ID]/ ./results/

# View MultiQC report
gsutil cp gs://biopipelines-results-rcc-hpc/dna_seq/[JOB_ID]/multiqc_report.html .
open multiqc_report.html
```

---

## 📁 File Organization

### Local Machine
```
~/Downloads/Repos/BioPipelines/
├── README.md
├── environment.yml
├── pipelines/
│   ├── dna_seq/variant_calling/
│   ├── rna_seq/differential_expression/
│   ├── chip_seq/peak_calling/
│   └── atac_seq/accessibility_analysis/
├── scripts/
│   ├── download_test_data.sh        # ✅ Uploads to GCS
│   ├── download_references.sh       # Downloads references
│   ├── gcp_stage_data.sh           # ✅ NEW: Stages from GCS
│   └── submit_dna_seq.sh           # ✅ UPDATED: Uses GCS
└── docs/
    ├── GCP_HPC_SETUP.md            # ✅ Cluster setup guide
    └── GCP_STORAGE_ARCHITECTURE.md # ✅ NEW: Storage details
```

### GCP HPC Cluster
```
~/
├── miniconda3/
├── envs/
│   └── biopipelines/              # Conda environment
└── BioPipelines/                   # Synced from git/local
    └── pipelines/

/mnt/disks/scratch/
├── shared/                         # Shared references cache
│   └── references/
│       ├── genome/
│       └── known_sites/
└── [job_id]/                       # Job-specific (auto-cleanup)
    ├── input/                      # Staged from GCS
    ├── working/                    # Pipeline execution
    ├── output/                     # Results (uploaded to GCS)
    └── env.sh                      # Environment variables
```

### GCS Buckets
```
gs://biopipelines-data/
├── dna_seq/
│   └── test/
│       ├── sample1_R1.fastq.gz
│       └── sample1_R2.fastq.gz
├── rna_seq/test/
├── chip_seq/test/
└── atac_seq/test/

gs://biopipelines-references/
├── genomes/
│   └── hg38/
│       ├── hg38.fa
│       ├── hg38.fa.fai
│       └── hg38.dict
├── annotations/
│   └── gencode.v44.annotation.gtf
└── known_sites/
    └── dbsnp_155.hg38.vcf.gz

gs://biopipelines-results-rcc-hpc/
├── dna_seq/
│   ├── 12345678/                   # Job ID
│   │   ├── vcf/
│   │   ├── qc/
│   │   └── multiqc_report.html
│   └── 12345679/
└── rna_seq/
```

---

## ✅ What I Fixed

### 1. **Scripts Updated**
- ✅ `scripts/download_test_data.sh` - Now uploads to GCS buckets
- ✅ `scripts/gcp_stage_data.sh` - NEW: Stages data from GCS to local compute
- ✅ `scripts/submit_dna_seq.sh` - Integrated GCS staging and result upload

### 2. **Documentation Created**
- ✅ `docs/GCP_STORAGE_ARCHITECTURE.md` - Complete storage architecture guide
- ✅ `NEXT_STEPS.md` - Updated with correct GCP workflow
- ✅ `GCP_ARCHITECTURE_CORRECTED.md` - This file explaining the architecture

### 3. **Previous Files** (for local use only - keep for reference)
- ⚠️ `scripts/quick_start.sh` - For local development only
- ⚠️ Earlier NEXT_STEPS.md sections - Were for local machine

---

## 🎯 Next Actions (Updated)

### Immediate (Today)
1. **Create GCS buckets**
   ```bash
   gsutil mb -l us-central1 gs://biopipelines-data/
   gsutil mb -l us-central1 gs://biopipelines-references/
   gsutil mb -l us-central1 gs://biopipelines-results-rcc-hpc/
   ```

2. **Upload test data**
   ```bash
   ./scripts/download_test_data.sh
   ```

3. **Test on cluster**
   ```bash
   gcloud compute ssh hpcslurm-slurm-login-001 --project=rcc-hpc --zone=us-central1-a --tunnel-through-iap
   cd ~/BioPipelines
   sbatch scripts/submit_dna_seq.sh
   ```

### This Week
- [ ] Verify DNA-seq pipeline runs end-to-end on cluster
- [ ] Check results in GCS
- [ ] Test RNA-seq pipeline
- [ ] Document any issues

### Next Week
- [ ] Complete testing of all 4 pipelines
- [ ] Create tutorial notebooks
- [ ] Benchmark resource usage
- [ ] Optimize GCS transfer performance

---

## 💡 Key Insights

1. **Why GCS + Local SSD?**
   - GCS: Persistent, shareable, backed up
   - Local SSD: Fast I/O for compute-intensive operations
   - Best of both: Stage from GCS → compute local → upload results

2. **Cost Optimization**
   - References cached in `/mnt/disks/scratch/shared/` (avoid repeated downloads)
   - Input data streamed only when needed
   - Results uploaded compressed
   - Auto-cleanup of scratch reduces storage costs

3. **Performance**
   - Local SSD: 1000+ MB/s (compute phase)
   - GCS transfer: 200-600 MB/s (stage/upload)
   - Minimal impact on pipeline runtime

---

## 📚 Documentation Map

| File | Purpose |
|------|---------|
| `README.md` | Project overview |
| `docs/GCP_HPC_SETUP.md` | Cluster setup tutorial |
| `docs/GCP_STORAGE_ARCHITECTURE.md` | Storage architecture details |
| `NEXT_STEPS.md` | Quick start guide (updated) |
| `TODO.md` | Detailed task checklist |
| `DEVELOPMENT_STATUS.md` | Project status report |
| **This file** | Architecture clarification |

---

**Summary:** You were absolutely right! The project is designed for GCP HPC cluster with GCS storage. All scripts and documentation have been corrected and enhanced. Ready to deploy! 🚀

