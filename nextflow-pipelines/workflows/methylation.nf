#!/usr/bin/env nextflow

/*
 * DNA Methylation (WGBS/RRBS) Analysis Pipeline
 * =============================================
 * Bisulfite sequencing analysis with Bismark
 */

nextflow.enable.dsl = 2

// Import modules
include { FASTQC } from '../modules/qc/fastqc/main'
include { TRIM_GALORE } from '../modules/trimming/trim_galore/main'
include { BISMARK_ALIGN } from '../modules/alignment/bismark/main'
include { BISMARK_METHYLATION_EXTRACTOR } from '../modules/methylation/bismark_extractor/main'

/*
 * Define the workflow
 */
workflow {
    // Sample input
    Channel.of(
        [
            [id: 'sample1', single_end: false],
            [
                file("/scratch/sdodl001/BioPipelines/data/raw/methylation/sample1_R1.fastq.gz"),
                file("/scratch/sdodl001/BioPipelines/data/raw/methylation/sample1_R2.fastq.gz")
            ]
        ]
    ).set { samples_ch }

    // Reference index
    bismark_index = file("/scratch/sdodl001/BioPipelines/data/references/bismark_index")

    // Step 1: Quality Control
    FASTQC(samples_ch)

    // Step 2: Adapter and quality trimming
    TRIM_GALORE(samples_ch)

    // Step 3: Bismark alignment
    BISMARK_ALIGN(TRIM_GALORE.out.reads, bismark_index)

    // Step 4: Methylation extraction
    BISMARK_METHYLATION_EXTRACTOR(BISMARK_ALIGN.out.bam)
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
