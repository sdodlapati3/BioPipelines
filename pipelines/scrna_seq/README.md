# Single-cell RNA-seq Pipeline

Comprehensive 10x Genomics single-cell RNA-seq analysis workflow using STARsolo and Scanpy.

## 📋 Overview

This pipeline performs complete single-cell RNA-seq analysis from raw FASTQ files to cell type annotation and differential expression analysis.

### Workflow Steps

1. **Quality Control** - FastQC analysis of raw reads
2. **Alignment & Counting** - STARsolo for alignment and UMI counting
3. **Cell Filtering** - Remove low-quality cells and doublets
4. **Normalization** - Total count normalization and log-transformation
5. **Feature Selection** - Identify highly variable genes (HVGs)
6. **Dimensionality Reduction** - PCA and UMAP embedding
7. **Clustering** - Leiden and Louvain clustering
8. **Cell Type Annotation** - Marker gene-based annotation
9. **Differential Expression** - Between cell types
10. **Reporting** - Comprehensive HTML report with visualizations

## 🚀 Quick Start

### 1. Download Test Data

```bash
# Download 10x PBMC test dataset
python scripts/download_scrna_test_data.py

# Or download specific FASTQ files from SRA
fastq-dump --split-files --gzip SRR8206317
```

### 2. Prepare Reference

```bash
# Build STAR index for your genome (if not already done)
STAR \
    --runMode genomeGenerate \
    --genomeDir data/references/star_index \
    --genomeFastaFiles data/references/hg38.fa \
    --sjdbGTFfile data/references/Homo_sapiens.GRCh38.110.gtf \
    --runThreadN 16 \
    --sjdbOverhang 100
```

### 3. Place Data

Ensure your FASTQ files are named correctly:
- `data/raw/scrna_seq/sample1_R1.fastq.gz` (Read 1: Cell barcode + UMI)
- `data/raw/scrna_seq/sample1_R2.fastq.gz` (Read 2: cDNA)

### 4. Configure Analysis

Edit `pipelines/scrna_seq/config.yaml`:
- Chemistry version (v2 or v3)
- Filtering thresholds
- Clustering parameters
- Marker genes for annotation

### 5. Run Pipeline

```bash
sbatch scripts/submit_scrna_seq.sh
```

## 📊 Expected Outputs

### Directory Structure

```
data/results/scrna_seq/
├── qc/                           # FastQC reports
├── starsolo/                     # STARsolo outputs
│   └── sample1/
│       └── Solo.out/
│           └── Gene/
│               └── filtered/      # Filtered count matrix
├── filtered/                     # Processed data
│   ├── sample1_raw.h5ad
│   ├── sample1_filtered.h5ad
│   └── sample1_pca.h5ad
├── clustering/                   # Clustering results
│   └── sample1_clustered.h5ad
├── annotation/                   # Cell type annotations
│   ├── sample1_annotated.h5ad
│   └── sample1_cluster_annotations.csv
├── differential_expression/      # DEG analysis
│   ├── sample1_deg.csv
│   └── sample1_rank_genes.csv
├── plots/                        # All visualizations
│   ├── sample1_umap_clusters.png
│   ├── sample1_marker_genes.png
│   └── ...
├── scrna_seq_report.html        # Main report
└── multiqc_report.html          # MultiQC report
```

### Key Files

- **AnnData objects** (`.h5ad`): Contain expression data, metadata, embeddings
- **DEG results** (`.csv`): Differentially expressed genes per cell type
- **HTML reports**: Comprehensive analysis summary
- **Plots**: UMAP, violin, dotplot, heatmap visualizations

## 🔧 Configuration

### Key Parameters

**Filtering Thresholds:**
```yaml
filtering:
  min_counts: 500          # Minimum UMI counts per cell
  max_counts: 30000        # Maximum UMI counts (doublet threshold)
  min_genes: 250           # Minimum genes per cell
  max_mito_percent: 20     # Maximum mitochondrial content (%)
```

**Feature Selection:**
```yaml
feature_selection:
  n_top_genes: 2000        # Number of highly variable genes
  flavor: "seurat"         # Method: seurat, cell_ranger
```

**Clustering:**
```yaml
clustering:
  neighbors:
    n_neighbors: 15
    n_pcs: 40
  leiden:
    resolution: [0.4, 0.6, 0.8, 1.0]  # Multiple resolutions
```

**Cell Type Annotation:**
```yaml
annotation:
  marker_genes:
    "CD4 T cells": ["CD3D", "CD3E", "CD4", "IL7R"]
    "CD8 T cells": ["CD3D", "CD3E", "CD8A", "CD8B"]
    "B cells": ["MS4A1", "CD79A", "CD79B"]
    "NK cells": ["GNLY", "NKG7", "NCAM1"]
    # Add your own markers
```

## 🧬 10x Genomics Chemistry

### Version 3 (default)
- Cell barcode: 16 bp (positions 1-16)
- UMI: 12 bp (positions 17-28)
- Whitelist: `3M-february-2018.txt`

### Version 2
- Cell barcode: 16 bp
- UMI: 10 bp
- Whitelist: `737K-august-2016.txt`

Update `config.yaml` accordingly.

## 📈 Quality Metrics

### Expected Values (PBMC dataset)

| Metric | Good | Warning | Poor |
|--------|------|---------|------|
| Cells detected | >1000 | 500-1000 | <500 |
| Median UMI/cell | >1000 | 500-1000 | <500 |
| Median genes/cell | >500 | 250-500 | <250 |
| Mitochondrial % | <10% | 10-20% | >20% |
| Doublet rate | <6% | 6-10% | >10% |

## 🛠️ Troubleshooting

### Low Cell Count
- Check STARsolo output: `Solo.out/Gene/Summary.csv`
- Verify whitelist matches chemistry version
- Adjust `--soloCellFilter` parameters

### High Mitochondrial Content
- May indicate cell stress or death
- Adjust `max_mito_percent` threshold
- Check sample quality

### Poor Clustering
- Try different resolution parameters
- Check if batch effects present (use batch correction)
- Verify sufficient highly variable genes selected

### Memory Issues
- Reduce `n_pcs` in PCA
- Process fewer cells (subsample)
- Increase job memory allocation

## 📚 References

### Software
- **STARsolo**: Kaminow et al. (2021) Genome Biology
- **Scanpy**: Wolf et al. (2018) Genome Biology
- **Scrublet**: Wolock et al. (2019) Cell Systems

### Datasets
- 10x Genomics: https://www.10xgenomics.com/resources/datasets
- PanglaoDB: https://panglaodb.se/
- CellMarker: http://biocc.hrbmu.edu.cn/CellMarker/

### Cell Type Markers
- Human Cell Atlas: https://www.humancellatlas.org/
- CellMarker database: http://biocc.hrbmu.edu.cn/CellMarker/
- PanglaoDB markers: https://panglaodb.se/markers.html

## 💡 Tips

1. **Start with filtered matrix** - Test pipeline with pre-filtered data before running full alignment
2. **Use known datasets** - Validate pipeline with PBMC data before your samples
3. **Check QC plots** - Review violin plots and scatter plots before proceeding
4. **Iterate on clustering** - Try multiple resolutions to find optimal granularity
5. **Validate annotations** - Check marker gene expression in annotated cell types

## 🎯 Performance

**Resource Requirements:**

| Step | CPUs | Memory | Time (3k cells) |
|------|------|--------|-----------------|
| FastQC | 2 | 2GB | 5 min |
| STARsolo | 8 | 32GB | 30 min |
| Filtering/QC | 1 | 8GB | 5 min |
| Clustering | 1 | 16GB | 10 min |
| Full pipeline | 16 | 64GB | 1-2 hours |

**Scaling:**
- 10k cells: ~2-3 hours
- 50k cells: ~4-6 hours  
- 100k+ cells: Consider downsampling or using Seurat

## 🔄 Integration with Other Pipelines

This pipeline generates outputs compatible with:
- **Seurat** (R): Export to RDS format
- **CellPhoneDB**: Cell-cell interaction analysis
- **Monocle3**: Trajectory inference
- **Velocyto**: RNA velocity analysis
- **scVI**: Deep learning integration

## 📝 Citation

If you use this pipeline, please cite:

```
BioPipelines: Comprehensive multi-omics analysis workflows
https://github.com/yourusername/BioPipelines
```

Plus individual tools (STARsolo, Scanpy, Scrublet, etc.)

## 📧 Support

For issues or questions:
- Open GitHub issue
- Check documentation: `docs/tutorials/scrna_seq_tutorial.md`
- Review example outputs

---

**Pipeline Version**: 1.0.0  
**Last Updated**: 2024-11-21  
**Maintainer**: BioPipelines Team
