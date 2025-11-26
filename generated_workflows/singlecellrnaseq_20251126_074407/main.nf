#!/usr/bin/env nextflow

/*
 * Single-Cell RNA-seq Analysis Pipeline
 * =====================================
 * 10x Genomics data processing with STARsolo
 */

nextflow.enable.dsl = 2

// Import modules
include { STARSOLO } from '../modules/scrna/starsolo/main'

/*
 * Define the workflow
 */
workflow {
    // Sample input - 10x Genomics format
    // R1 = barcode + UMI, R2 = cDNA read
    Channel.of(
        [
            [id: 'sample1', single_end: false, chemistry: 'v3'],
            [
                file("/scratch/sdodl001/BioPipelines/data/raw/scrna_seq/sample1_R1.fastq.gz"),
                file("/scratch/sdodl001/BioPipelines/data/raw/scrna_seq/sample1_R2.fastq.gz")
            ]
        ]
    ).set { samples_ch }

    // Reference files
    star_index = file("/scratch/sdodl001/BioPipelines/data/references/star_index_hg38")
    whitelist = file("/scratch/sdodl001/BioPipelines/data/references/10x_whitelist_v3.txt")

    // Run STARsolo for alignment and quantification
    STARSOLO(samples_ch, star_index, whitelist)
}

/*
 * Workflow completion notification
 */
workflow.onComplete {
    println """
    Pipeline completed at: $workflow.complete
    Execution status: ${workflow.success ? 'SUCCESS' : 'FAILED'}
    Execution duration: $workflow.duration
    """
}
