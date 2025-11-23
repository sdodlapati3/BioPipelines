# BioPipelines TODO List

## 🔥 Immediate Actions (This Week)

### Day 1-2: Setup & Legal
- [ ] Add MIT LICENSE file
- [ ] Update README.md (replace YOUR_USERNAME with actual GitHub username)
- [ ] Create CONTRIBUTING.md
- [ ] Create .github/ISSUE_TEMPLATE/

### Day 3-5: Validation Infrastructure
- [ ] Download test datasets:
  - [ ] DNA-seq: Small human WGS sample (chr22 only)
  - [ ] RNA-seq: Small RNA-seq sample (subset)
  - [ ] ChIP-seq: Small ChIP-seq sample
  - [ ] ATAC-seq: Small ATAC-seq sample
- [ ] Create `data/test/README.md` with dataset descriptions
- [ ] Write `scripts/download_test_data.sh` script

### Day 6-7: First Pipeline Test
- [ ] Test DNA-seq variant calling pipeline end-to-end
- [ ] Document any errors encountered
- [ ] Fix bugs found during testing
- [ ] Create example output in `data/results/example/`

---

## 📅 Week 2: Core Testing

### DNA-seq Pipeline
- [ ] Write unit tests for variant calling utilities
- [ ] Test with different reference genomes
- [ ] Benchmark resource usage (time/memory)
- [ ] Document in `docs/pipelines/dna_seq.md`

### RNA-seq Pipeline
- [ ] Test differential expression pipeline
- [ ] Validate DESeq2 results
- [ ] Test with different conditions
- [ ] Document in `docs/pipelines/rna_seq.md`

### Python Utilities
- [ ] Write tests for `src/biopipelines/alignment/`
- [ ] Write tests for `src/biopipelines/preprocessing/`
- [ ] Write tests for `src/biopipelines/variant_calling/`
- [ ] Set up pytest configuration

---

## 📅 Week 3: Documentation

### Pipeline Docs
- [ ] Create `docs/pipelines/dna_seq_variant_calling.md`
- [ ] Create `docs/pipelines/rna_seq_differential_expression.md`
- [ ] Create `docs/pipelines/chip_seq_peak_calling.md`
- [ ] Create `docs/pipelines/atac_seq_accessibility.md`

### Tutorials
- [ ] Create `notebooks/tutorials/01_dna_seq_variant_calling.ipynb`
- [ ] Create `notebooks/tutorials/02_rna_seq_analysis.ipynb`
- [ ] Create `notebooks/tutorials/03_chip_seq_analysis.ipynb`
- [ ] Create installation guide: `docs/INSTALLATION.md`

### API Docs
- [ ] Add docstrings to all Python functions
- [ ] Set up Sphinx documentation
- [ ] Generate HTML docs
- [ ] Create `docs/api/index.md`

---

## 📅 Week 4-5: Missing Pipelines

### Metagenomics Pipeline
- [ ] Create `pipelines/metagenomics/taxonomic_profiling/`
- [ ] Implement Kraken2/MetaPhlAn workflow
- [ ] Add assembly step (MEGAHIT)
- [ ] Add functional annotation (HUMAnN3)
- [ ] Create config and environment files
- [ ] Test with mock community data

### RNA-seq Isoform Analysis
- [ ] Create `pipelines/rna_seq/isoform_analysis/Snakefile`
- [ ] Implement StringTie/Salmon workflow
- [ ] Add differential isoform usage analysis
- [ ] Create environment file
- [ ] Test with known isoform switching

### Standalone QC Workflows
- [ ] Create `pipelines/dna_seq/quality_control/Snakefile`
- [ ] Create `pipelines/rna_seq/quality_control/Snakefile`
- [ ] Modular QC-only workflows

---

## 📅 Week 6: Polish & CI/CD

### GitHub Actions
- [ ] Create `.github/workflows/tests.yml`
- [ ] Set up automated testing
- [ ] Add linting (black, flake8)
- [ ] Add coverage reporting

### Benchmarking
- [ ] Run performance benchmarks on all pipelines
- [ ] Compare with published workflows
- [ ] Document in `benchmarks/results/`

### Final Polish
- [ ] Update README with badges
- [ ] Add citation information
- [ ] Create CHANGELOG.md
- [ ] Tag v1.0.0 release

---

## 🎯 Critical Path (Minimum Viable Product)

If time is limited, focus on these essentials:

1. ✅ **Add LICENSE** (5 min)
2. ✅ **Test DNA-seq pipeline** (1 day)
3. ✅ **Test RNA-seq pipeline** (1 day)
4. ✅ **Write README installation instructions** (2 hours)
5. ✅ **Create one tutorial notebook** (4 hours)
6. ✅ **Write basic unit tests** (1 day)

**Total MVP Time:** ~3 days

---

## 🐛 Known Issues

- [ ] DNA-seq Snakefile has uncommitted changes
- [ ] `docs/submit_dna_seq.sh` is untracked - review and commit/remove
- [ ] Empty `__init__.py` files need proper imports
- [ ] Need to verify all conda environments install correctly
- [ ] Path issues in Snakefiles may need adjustment for different systems

---

## 💡 Nice-to-Have Features

### Future Enhancements
- [ ] Docker containers for each pipeline
- [ ] Nextflow version of pipelines
- [ ] Cloud deployment scripts (AWS/GCP)
- [ ] Web dashboard for results
- [ ] Parameter optimization tool
- [ ] Automatic report generation
- [ ] Integration with public databases

### Advanced Analysis
- [ ] Single-cell RNA-seq pipeline
- [ ] Spatial transcriptomics
- [ ] Long-read sequencing (PacBio/Nanopore)
- [ ] Multi-omics integration
- [ ] Machine learning QC predictions

---

## 📞 Support Needed

### Questions to Resolve
- [ ] Which GitHub username to use?
- [ ] Which reference genomes to support? (hg38, mm10, others?)
- [ ] What compute environment? (HPC, cloud, local?)
- [ ] Citation preferences?
- [ ] Target audience? (researchers, core facilities, students?)

### External Resources Needed
- [ ] Access to test datasets
- [ ] Compute resources for testing
- [ ] Code review partners
- [ ] Beta testers

---

## 🎓 Learning Resources

If you need to learn more about components:

- **Snakemake:** https://snakemake.readthedocs.io/
- **GATK Best Practices:** https://gatk.broadinstitute.org/
- **RNA-seq Analysis:** https://www.bioconductor.org/packages/release/workflows/html/rnaseqGene.html
- **Conda/Bioconda:** https://bioconda.github.io/
- **pytest:** https://docs.pytest.org/

---

*Last Updated: November 20, 2025*

# 🎯 What to Do Next - Executive Summary

## 📊 Current Status

**The Good News:**
- ✅ BioPipelines is **well-designed** with excellent architecture
- ✅ 4 complete pipelines implemented (~930 lines of Snakemake code)
- ✅ ~1,400 lines of Python utilities
- ✅ Professional environment setup with 50+ bioinformatics tools
- ✅ Proper package structure
- ✅ **Configured for GCP HPC Slurm cluster deployment**

**The Gap:**
- ❌ No testing or validation yet
- ❌ No documentation
- ❌ 5 pipelines not implemented (metagenomics, QC modules, isoform analysis)
- ❌ GCS buckets not yet created
- ✅ LICENSE added
- ✅ GCS integration scripts created

**Verdict:** This is a **solid alpha release (v0.1.0)** designed for **GCP HPC cluster** that needs validation and GCS setup.

---

## 🏗️ Architecture Note

**IMPORTANT:** This project runs on **GCP HPC Slurm cluster**, not local machine!

```
Development Machine (Local)
    ↓ (develop & commit code)
GitHub Repository
    ↓ (clone/sync)
GCP HPC Cluster (hpcslurm-slurm-login-001)
    ↓ (reads data from)
GCS Buckets (gs://biopipelines-*)
```

**Storage Architecture:**
- 📦 **gs://biopipelines-data** - Input datasets
- 📦 **gs://biopipelines-references** - Reference genomes
- 📦 **gs://biopipelines-results-rcc-hpc** - Pipeline outputs
- 💾 **/mnt/disks/scratch/$JOB_ID** - Temporary compute storage

See `docs/GCP_STORAGE_ARCHITECTURE.md` for details.

---

## 🚀 Recommended Action Plan

### Option 1: Quick Win (2-3 days) ⭐ RECOMMENDED
**Goal:** Validate DNA-seq pipeline on GCP cluster

```bash
Day 1: Setup GCS & Upload Data
# On your local machine or cluster
gcloud auth login
./scripts/download_test_data.sh  # Downloads & uploads to GCS
# Creates gs://biopipelines-data/dna_seq/test/

Day 2: Test on GCP Cluster
# SSH to cluster
gcloud compute ssh hpcslurm-slurm-login-001 --project=rcc-hpc --zone=us-central1-a --tunnel-through-iap

# Submit test job
cd ~/BioPipelines
sbatch scripts/submit_dna_seq.sh

# Monitor
squeue -u $USER
tail -f slurm_*.out

Day 3: Validate & Document
# Check results in GCS
gsutil ls gs://biopipelines-results-rcc-hpc/dna_seq/
# Download and review
# Create tutorial notebook
```

**Outcome:** Working pipeline running on GCP with results in GCS!

### Option 2: Production Ready (3-4 weeks)
**Goal:** Complete all testing and documentation

- Week 1: Test all 4 existing pipelines
- Week 2: Write comprehensive docs + tests
- Week 3: Implement metagenomics pipeline
- Week 4: Polish and release v1.0.0

### Option 3: Feature Complete (6-8 weeks)
**Goal:** Implement all promised features

- All pipelines implemented
- Full test coverage
- Complete documentation
- Benchmarking results
- CI/CD setup

---

## 🎬 Start Right Now (15 minutes)

### Step 1: Create GCS Buckets (from local machine or cluster)

```bash
# Authenticate with GCP
gcloud auth login
gcloud config set project rcc-hpc

# Create buckets
gsutil mb -p rcc-hpc -l us-central1 gs://biopipelines-data/
gsutil mb -p rcc-hpc -l us-central1 gs://biopipelines-references/
gsutil mb -p rcc-hpc -l us-central1 gs://biopipelines-results-rcc-hpc/

# Verify
gsutil ls
```

### Step 2: Upload Test Data to GCS

```bash
cd /path/to/BioPipelines

# Download test data and upload to GCS
./scripts/download_test_data.sh

# This will:
# - Download chr22 reference genome
# - Download test FASTQ files
# - Upload everything to GCS buckets
```

### Step 3: Connect to GCP Cluster & Submit Job

```bash
# SSH to cluster
gcloud compute ssh hpcslurm-slurm-login-001 \
  --project=rcc-hpc \
  --zone=us-central1-a \
  --tunnel-through-iap

# On the cluster:
# Sync your code
cd ~/BioPipelines
git pull  # or transfer files

# Activate environment (if not already created)
source ~/miniconda3/bin/activate ~/envs/biopipelines

# Submit job
sbatch scripts/submit_dna_seq.sh

# Monitor
squeue -u $USER
tail -f slurm_*.out
```

---

## 📋 Immediate Actions (Today)

1. **Create GCS buckets** (5 minutes)
   ```bash
   gsutil mb -l us-central1 gs://biopipelines-data/
   gsutil mb -l us-central1 gs://biopipelines-references/
   gsutil mb -l us-central1 gs://biopipelines-results-rcc-hpc/
   ```

2. **Upload test data** (10-20 minutes)
   ```bash
   ./scripts/download_test_data.sh
   ```

3. **SSH to cluster and submit test job** (5 minutes)
   ```bash
   gcloud compute ssh hpcslurm-slurm-login-001 --project=rcc-hpc --zone=us-central1-a --tunnel-through-iap
   cd ~/BioPipelines
   sbatch scripts/submit_dna_seq.sh
   ```

4. **Monitor and verify** (1-2 hours for job to run)
   ```bash
   squeue -u $USER
   tail -f slurm_*.out
   gsutil ls gs://biopipelines-results-rcc-hpc/dna_seq/
   ```

---

## 🎯 Success Criteria

**Minimum Viable Demo (3 days):**
- [ ] 1 pipeline runs successfully end-to-end
- [ ] Installation instructions work
- [ ] 1 example output/notebook
- [ ] README updated

**Production Ready (3 weeks):**
- [ ] All 4 pipelines tested and working
- [ ] Documentation for each pipeline
- [ ] Basic unit tests
- [ ] Example datasets included

**Feature Complete (6 weeks):**
- [ ] All 9 promised pipelines implemented
- [ ] >70% test coverage
- [ ] Full documentation
- [ ] Benchmarking results

---

## 💡 Pro Tips

1. **Don't implement new pipelines yet** - validate what exists first
2. **Start with DNA-seq** - it's the most complete (281 lines)
3. **Use small test datasets** - chr22 only for DNA-seq, subset for RNA-seq
4. **Document as you go** - capture setup steps, errors, solutions
5. **One pipeline at a time** - don't parallelize until you have one working

---

## 🆘 If You Get Stuck

**Common Issues:**
- Conda environment won't create → Check channel priorities
- Snakemake fails → Check file paths in Snakefile
- Missing reference data → Use scripts/download_references.sh
- Out of memory → Use smaller test dataset

**Resources:**
- See TODO.md for detailed checklist
- See DEVELOPMENT_STATUS.md for full status report
- Check Snakemake docs: https://snakemake.readthedocs.io/

---

## 🎉 Quick Wins Already Done

I just created:
- ✅ LICENSE file (MIT)
- ✅ DEVELOPMENT_STATUS.md (full status report)
- ✅ TODO.md (detailed checklist)
- ✅ scripts/quick_start.sh (setup automation)
- ✅ This summary document

You're ready to start testing! 🚀

---

**Recommendation:** Start with Option 1 (Quick Win). Get one pipeline working this week, then decide if you want to expand or polish.

