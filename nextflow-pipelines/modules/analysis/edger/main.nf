/*
 * edgeR Module
 * 
 * edgeR - Differential expression analysis (R-based)
 * Alternative to DESeq2 for RNA-seq differential expression
 * Uses existing rna-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * edgeR Differential Expression
 */
process EDGER_DIFFERENTIAL {
    tag "edger_analysis"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/differential_expression/edger", mode: 'copy'
    
    cpus params.edger?.cpus ?: 4
    memory params.edger?.memory ?: '16.GB'
    
    input:
    path count_matrix
    path sample_info
    val design_formula
    val contrast
    
    output:
    path "edger_results.csv", emit: results
    path "edger_normalized_counts.csv", emit: normalized_counts
    path "md_plot.pdf", emit: md_plot
    path "mds_plot.pdf", emit: mds_plot
    path "bcv_plot.pdf", emit: bcv_plot
    path "volcano_plot.pdf", emit: volcano_plot
    
    script:
    """
    #!/usr/bin/env Rscript
    
    library(edgeR)
    library(ggplot2)
    
    # Load count matrix and sample info
    counts <- read.csv("${count_matrix}", row.names=1)
    samples <- read.csv("${sample_info}", row.names=1)
    
    # Ensure count matrix columns match sample info rows
    samples <- samples[colnames(counts), , drop=FALSE]
    
    # Create DGEList object
    dge <- DGEList(counts=counts, group=samples\$condition)
    
    # Filter low-expressed genes
    keep <- filterByExpr(dge)
    dge <- dge[keep, , keep.lib.sizes=FALSE]
    
    # Calculate normalization factors
    dge <- calcNormFactors(dge)
    
    # Extract normalized counts
    norm_counts <- cpm(dge, log=FALSE)
    write.csv(norm_counts, "edger_normalized_counts.csv")
    
    # Create design matrix
    design <- model.matrix(as.formula("${design_formula}"), data=samples)
    
    # Estimate dispersion
    dge <- estimateDisp(dge, design)
    
    # Fit model
    fit <- glmQLFit(dge, design)
    
    # Test for differential expression
    contrast_vector <- makeContrasts("${contrast}", levels=design)
    qlf <- glmQLFTest(fit, contrast=contrast_vector)
    
    # Extract results
    results <- topTags(qlf, n=Inf)
    write.csv(results\$table, "edger_results.csv")
    
    # MD plot
    pdf("md_plot.pdf")
    plotMD(qlf)
    dev.off()
    
    # MDS plot
    pdf("mds_plot.pdf")
    plotMDS(dge, col=as.numeric(samples\$condition))
    dev.off()
    
    # BCV plot
    pdf("bcv_plot.pdf")
    plotBCV(dge)
    dev.off()
    
    # Volcano plot
    pdf("volcano_plot.pdf")
    res_df <- results\$table
    res_df\$significant <- ifelse(res_df\$FDR < 0.05 & abs(res_df\$logFC) > 1, "Yes", "No")
    ggplot(res_df, aes(x=logFC, y=-log10(FDR), color=significant)) +
        geom_point(alpha=0.5) +
        scale_color_manual(values=c("gray", "red")) +
        theme_minimal() +
        labs(title="Volcano Plot", x="log2 Fold Change", y="-log10(FDR)")
    dev.off()
    """
}

/*
 * Workflow: edgeR analysis
 */
workflow EDGER_PIPELINE {
    take:
    count_matrix   // path: gene count matrix (genes x samples)
    sample_info    // path: sample metadata CSV
    design_formula // val: design formula (e.g., "~ condition")
    contrast       // val: contrast specification
    
    main:
    EDGER_DIFFERENTIAL(
        count_matrix,
        sample_info,
        design_formula,
        contrast
    )
    
    emit:
    results = EDGER_DIFFERENTIAL.out.results
    normalized_counts = EDGER_DIFFERENTIAL.out.normalized_counts
    md_plot = EDGER_DIFFERENTIAL.out.md_plot
    mds_plot = EDGER_DIFFERENTIAL.out.mds_plot
    bcv_plot = EDGER_DIFFERENTIAL.out.bcv_plot
    volcano_plot = EDGER_DIFFERENTIAL.out.volcano_plot
}
