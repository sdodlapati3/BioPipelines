# Nextflow Practice Exercises

These exercises will help you learn Nextflow DSL2 syntax step by step.

## Exercise 1: Simple Process (`test_simple.nf`)

**Concepts**: Basic process, value inputs, Channel.of()

```bash
nextflow run tests/test_simple.nf
```

**What it does**:
- Creates a channel with 3 names
- Runs a simple echo process for each name
- Submits jobs to SLURM
- Demonstrates basic input/output

**Key Learning**:
- `process` definition
- `input:` and `output:` blocks
- `script:` block with shell commands
- Using channels with the pipe operator `|`

---

## Exercise 2: File Processing (`test_files.nf`)

**Concepts**: Path inputs, file handling, tuple outputs

```bash
nextflow run tests/test_files.nf
```

**What it does**:
- Creates test files dynamically
- Counts lines in each file
- Returns file and count as tuple
- Uses `map` to format output

**Key Learning**:
- Working with `path` inputs
- Tuple outputs `tuple path(file), stdout`
- Using `tag` directive for logging
- Transforming outputs with `.map()`

---

## Exercise 3: Meta Maps (`test_meta.nf`) ⭐ IMPORTANT

**Concepts**: Meta map pattern (used in all real pipelines)

```bash
nextflow run tests/test_meta.nf
```

**What it does**:
- Creates samples with metadata (id, condition, single_end)
- Processes each sample with its metadata
- Generates output files with sample information
- Demonstrates the meta map pattern

**Key Learning**:
- Meta map structure: `[id: 'sample1', condition: 'WT', single_end: false]`
- Tuple input: `tuple val(meta), path(reads)`
- Using `$meta.id` in script and output paths
- This is **the pattern we'll use for RNA-seq and all pipelines**

**View output files**:
```bash
find /scratch/sdodl001/BioPipelines/work -name "*_processed.txt" | head -1 | xargs cat
```

---

## Exercise 4: Resource Labels (`test_slurm.nf`)

**Concepts**: Resource management, SLURM integration, process labels

```bash
nextflow run tests/test_slurm.nf -c config/base.config
```

**What it does**:
- Runs "light" tasks with 2 CPUs
- Runs "heavy" tasks with 8 CPUs
- Uses resource labels from config
- Demonstrates SLURM job submission with different resources

**Key Learning**:
- `label` directive: `label 'process_low'` or `label 'process_high'`
- Resource labels defined in `config/base.config`
- How to control CPU/memory allocation
- Monitoring SLURM jobs: `watch -n 5 squeue -u $USER`

---

## Understanding the Output

### Successful Run Output:
```
executor >  slurm (3)
[e6/e925ed] process > processWithMeta (sample2) [100%] 3 of 3 ✔
Sample sample2 (KO): /path/to/output.txt

Completed at: 24-Nov-2025 03:50:42
Duration    : 2m 15s
CPU hours   : (a few seconds)
Succeeded   : 3
```

**Key Information**:
- `executor > slurm (3)`: 3 jobs submitted to SLURM
- `[e6/e925ed]`: Work directory hash (first 2 chars = subdirectory)
- `processWithMeta (sample2)`: Process name with sample tag
- `[100%] 3 of 3 ✔`: All 3 tasks completed successfully
- `Duration: 2m 15s`: Total pipeline runtime
- Full work path: `/scratch/sdodl001/BioPipelines/work/e6/e925ed...`

---

## Monitoring Jobs

### Check SLURM queue:
```bash
squeue -u $USER
watch -n 5 squeue -u $USER  # Auto-refresh every 5 seconds
```

### Check completed jobs:
```bash
sacct -u $USER --format=JobID,JobName,State,Elapsed
```

### Resume a failed run:
```bash
nextflow run tests/test_meta.nf -resume
```

---

## Common Commands

### Run with resume (skip completed tasks):
```bash
nextflow run tests/test_simple.nf -resume
```

### Clean work directory:
```bash
nextflow clean -f
```

### View Nextflow log:
```bash
cat .nextflow.log
```

### Debug a failed task:
```bash
# Find the work directory from error message
cd /scratch/sdodl001/BioPipelines/work/e6/e925ed...

# View command executed
cat .command.sh

# View output
cat .command.out

# View errors
cat .command.err

# View exit code
cat .exitcode

# Run command interactively
bash .command.sh
```

---

## Next Steps

After completing these exercises:

1. ✅ **Understand DSL2 syntax**: processes, workflows, channels
2. ✅ **Master meta map pattern**: This is critical for real pipelines
3. ✅ **Learn resource labels**: Control CPU/memory allocation
4. 📚 **Complete Nextflow training**: https://training.nextflow.io
5. 🔬 **Study nf-core/rnaseq**: Real-world RNA-seq implementation
6. 🛠️ **Create FastQC module**: First real bioinformatics module

---

## Key Takeaways

### Meta Map Pattern (CRITICAL)
```groovy
// This is how we'll handle all samples in real pipelines
def meta = [
    id: 'sample1',              // Sample identifier
    single_end: false,          // Paired-end or single-end
    condition: 'WT'             // Experimental condition
]

// Always first in tuple
tuple val(meta), path(reads)

// Use in script
"Processing sample: ${meta.id}"

// Use in output paths
path("${meta.id}_output.txt")
```

### Resource Labels
```groovy
process STAR_ALIGN {
    label 'process_high'  // 8 CPUs, 16 GB, 12h
    // ...
}
```

Defined in `config/base.config`:
- `process_low`: 2 CPUs, 4 GB, 1h
- `process_medium`: 4 CPUs, 8 GB, 4h
- `process_high`: 8 CPUs, 16 GB, 12h

---

## Troubleshooting

### Jobs not submitting to SLURM?
Check partition name in `~/.nextflow/config`:
```bash
sinfo  # List available partitions
```

### Out of memory errors?
Increase memory for specific process:
```groovy
process MY_PROCESS {
    memory = '32 GB'  // Override default
    // ...
}
```

### Tasks running sequentially?
Nextflow parallelizes automatically. Check:
```bash
squeue -u $USER  # Should see multiple jobs
```

---

**Status**: Week 1 Day 1 - Exercises Created ✅  
**Next**: Complete Nextflow training, then study nf-core/rnaseq
