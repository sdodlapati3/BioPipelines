#!/usr/bin/env nextflow

/*
 * Long-Read Sequencing Analysis Pipeline
 * ======================================
 * Nanopore/PacBio data processing with Minimap2
 */

nextflow.enable.dsl = 2

// Import modules
include { MINIMAP2 } from '../modules/alignment/minimap2/main'

/*
 * Define the workflow
 */
workflow {
    // Sample input
    Channel.of(
        [
            [id: 'sample1', single_end: true, platform: 'nanopore'],
            file("/scratch/sdodl001/BioPipelines/data/raw/long_read/sample1.fastq.gz")
        ]
    ).set { samples_ch }

    // Reference genome
    reference = file("/scratch/sdodl001/BioPipelines/data/references/hg38.fa")

    // Alignment with Minimap2
    MINIMAP2(samples_ch, reference)
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
