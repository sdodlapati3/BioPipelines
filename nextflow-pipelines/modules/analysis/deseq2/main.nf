/*
 * DESeq2 Module
 * 
 * DESeq2 - Differential gene expression analysis (R-based)
 * Statistical analysis of RNA-seq count data
 * Uses existing rna-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * DESeq2 Differential Expression
 */
process DESEQ2_DIFFERENTIAL {
    tag "deseq2_analysis"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/differential_expression/deseq2", mode: 'copy'
    
    cpus params.deseq2?.cpus ?: 4
    memory params.deseq2?.memory ?: '16.GB'
    
    input:
    path count_matrix
    path sample_info
    val design_formula
    val contrast
    
    output:
    path "deseq2_results.csv", emit: results
    path "deseq2_normalized_counts.csv", emit: normalized_counts
    path "deseq2_rlog_counts.csv", emit: rlog_counts
    path "ma_plot.pdf", emit: ma_plot
    path "pca_plot.pdf", emit: pca_plot
    path "volcano_plot.pdf", emit: volcano_plot
    path "heatmap.pdf", emit: heatmap
    
    script:
    """
    #!/usr/bin/env Rscript
    
    library(DESeq2)
    library(ggplot2)
    library(pheatmap)
    
    # Load count matrix and sample info
    counts <- read.csv("${count_matrix}", row.names=1)
    coldata <- read.csv("${sample_info}", row.names=1)
    
    # Ensure count matrix columns match sample info rows
    coldata <- coldata[colnames(counts), , drop=FALSE]
    
    # Create DESeq2 dataset
    dds <- DESeqDataSetFromMatrix(
        countData = counts,
        colData = coldata,
        design = as.formula("${design_formula}")
    )
    
    # Run differential expression analysis
    dds <- DESeq(dds)
    
    # Extract results
    res <- results(dds, contrast=c("${contrast}"))
    res_df <- as.data.frame(res)
    write.csv(res_df, "deseq2_results.csv")
    
    # Normalized counts
    norm_counts <- counts(dds, normalized=TRUE)
    write.csv(norm_counts, "deseq2_normalized_counts.csv")
    
    # rlog transformation for visualization
    rld <- rlog(dds, blind=FALSE)
    rlog_counts <- assay(rld)
    write.csv(rlog_counts, "deseq2_rlog_counts.csv")
    
    # MA plot
    pdf("ma_plot.pdf")
    plotMA(res, ylim=c(-5,5))
    dev.off()
    
    # PCA plot
    pdf("pca_plot.pdf")
    plotPCA(rld, intgroup=colnames(coldata)[1])
    dev.off()
    
    # Volcano plot
    pdf("volcano_plot.pdf")
    res_df\$significant <- ifelse(res_df\$padj < 0.05 & abs(res_df\$log2FoldChange) > 1, "Yes", "No")
    ggplot(res_df, aes(x=log2FoldChange, y=-log10(padj), color=significant)) +
        geom_point(alpha=0.5) +
        scale_color_manual(values=c("gray", "red")) +
        theme_minimal() +
        labs(title="Volcano Plot", x="log2 Fold Change", y="-log10(padj)")
    dev.off()
    
    # Heatmap of top genes
    pdf("heatmap.pdf", width=10, height=12)
    top_genes <- head(order(res\$padj), 50)
    pheatmap(rlog_counts[top_genes,], 
             cluster_rows=TRUE, 
             cluster_cols=TRUE,
             show_rownames=TRUE,
             scale="row")
    dev.off()
    """
}

/*
 * Workflow: DESeq2 analysis
 */
workflow DESEQ2_PIPELINE {
    take:
    count_matrix   // path: gene count matrix (genes x samples)
    sample_info    // path: sample metadata CSV
    design_formula // val: design formula (e.g., "~ condition")
    contrast       // val: contrast specification (e.g., "condition,treated,control")
    
    main:
    DESEQ2_DIFFERENTIAL(
        count_matrix,
        sample_info,
        design_formula,
        contrast
    )
    
    emit:
    results = DESEQ2_DIFFERENTIAL.out.results
    normalized_counts = DESEQ2_DIFFERENTIAL.out.normalized_counts
    ma_plot = DESEQ2_DIFFERENTIAL.out.ma_plot
    pca_plot = DESEQ2_DIFFERENTIAL.out.pca_plot
    volcano_plot = DESEQ2_DIFFERENTIAL.out.volcano_plot
}
