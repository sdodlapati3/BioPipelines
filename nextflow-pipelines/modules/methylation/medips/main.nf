/*
 * MEDIPS Module
 * 
 * MEDIPS - MeDIP-seq data analysis in R
 * Methylated DNA immunoprecipitation sequencing analysis
 * Uses existing methylation container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * MEDIPS coverage and saturation analysis
 */
process MEDIPS_COVERAGE {
    tag "medips_${sample_id}"
    container "${params.containers.methylation}"
    
    publishDir "${params.outdir}/methylation/medips", mode: 'copy'
    
    cpus params.medips?.cpus ?: 8
    memory params.medips?.memory ?: '32.GB'
    
    input:
    tuple val(sample_id), path(bam), path(bai)
    path genome
    val window_size
    
    output:
    path "${sample_id}_medips.rds", emit: medips_object
    path "${sample_id}_saturation.pdf", emit: saturation_plot
    path "${sample_id}_coverage.pdf", emit: coverage_plot
    path "${sample_id}_CpG_enrichment.pdf", emit: cpg_plot
    
    script:
    def extend = params.medips?.extend ?: 200
    def shift = params.medips?.shift ?: 0
    def uniq = params.medips?.uniq ?: 1e-3
    
    """
    #!/usr/bin/env Rscript
    
    library(MEDIPS)
    library(BSgenome)
    
    # Create MEDIPS set
    medips_set <- MEDIPS.createSet(
        file = "${bam}",
        BSgenome = "${genome}",
        extend = ${extend},
        shift = ${shift},
        uniq = ${uniq},
        window_size = ${window_size}
    )
    
    # Saturation analysis
    sr <- MEDIPS.saturation(medips_set, nit = 10, nrit = 1)
    pdf("${sample_id}_saturation.pdf")
    MEDIPS.plotSaturation(sr)
    dev.off()
    
    # Coverage analysis
    cr <- MEDIPS.seqCoverage(medips_set, type = "hist", plotType = "both")
    pdf("${sample_id}_coverage.pdf")
    MEDIPS.plotSeqCoverage(cr)
    dev.off()
    
    # CpG enrichment
    er <- MEDIPS.CpGenrich(medips_set)
    pdf("${sample_id}_CpG_enrichment.pdf")
    MEDIPS.plotCpGEnrich(er)
    dev.off()
    
    # Save object
    saveRDS(medips_set, "${sample_id}_medips.rds")
    """
}

/*
 * MEDIPS differential methylation
 */
process MEDIPS_DMR {
    tag "medips_dmr_${condition1}_vs_${condition2}"
    container "${params.containers.methylation}"
    
    publishDir "${params.outdir}/methylation/medips/dmr", mode: 'copy'
    
    cpus params.medips?.cpus ?: 8
    memory params.medips?.memory ?: '32.GB'
    
    input:
    tuple val(condition1), path(medips1)
    tuple val(condition2), path(medips2)
    val fdr
    
    output:
    path "${condition1}_vs_${condition2}_DMRs.txt", emit: dmr_table
    path "${condition1}_vs_${condition2}_MA_plot.pdf", emit: ma_plot
    path "${condition1}_vs_${condition2}_volcano.pdf", emit: volcano_plot
    
    script:
    def diff_method = params.medips?.diff_method ?: "edgeR"
    
    """
    #!/usr/bin/env Rscript
    
    library(MEDIPS)
    library(ggplot2)
    
    # Load MEDIPS sets
    set1 <- readRDS("${medips1}")
    set2 <- readRDS("${medips2}")
    
    # Differential methylation
    meth_diff <- MEDIPS.meth(
        MSet1 = set1,
        MSet2 = set2,
        diff.method = "${diff_method}",
        prob.method = "poisson",
        MeDIP = TRUE
    )
    
    # Filter by FDR
    dmr <- meth_diff[meth_diff\$edgeR.adj.p.value < ${fdr}, ]
    write.table(dmr, "${condition1}_vs_${condition2}_DMRs.txt", sep = "\\t", quote = FALSE, row.names = FALSE)
    
    # MA plot
    pdf("${condition1}_vs_${condition2}_MA_plot.pdf")
    plot(meth_diff\$edgeR.logCPM, meth_diff\$edgeR.logFC,
         xlab = "Average log CPM", ylab = "log2 Fold Change",
         main = "MA Plot", pch = 20, cex = 0.5, col = "gray")
    points(dmr\$edgeR.logCPM, dmr\$edgeR.logFC, col = "red", pch = 20, cex = 0.5)
    dev.off()
    
    # Volcano plot
    pdf("${condition1}_vs_${condition2}_volcano.pdf")
    plot(meth_diff\$edgeR.logFC, -log10(meth_diff\$edgeR.adj.p.value),
         xlab = "log2 Fold Change", ylab = "-log10(FDR)",
         main = "Volcano Plot", pch = 20, cex = 0.5, col = "gray")
    points(dmr\$edgeR.logFC, -log10(dmr\$edgeR.adj.p.value), col = "red", pch = 20, cex = 0.5)
    abline(h = -log10(${fdr}), lty = 2, col = "blue")
    dev.off()
    """
}

/*
 * Workflow: MEDIPS MeDIP-seq analysis
 */
workflow MEDIPS_PIPELINE {
    take:
    bam_ch         // channel: [ val(sample_id), path(bam), path(bai) ]
    genome         // val: BSgenome package name
    window_size    // val: window size for analysis
    
    main:
    MEDIPS_COVERAGE(bam_ch, genome, window_size)
    
    emit:
    medips_object = MEDIPS_COVERAGE.out.medips_object
    saturation = MEDIPS_COVERAGE.out.saturation_plot
    coverage = MEDIPS_COVERAGE.out.coverage_plot
}
