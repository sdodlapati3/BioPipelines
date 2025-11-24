#!/usr/bin/env nextflow

/*
 * Metagenomics Analysis Pipeline
 * ==============================
 * Taxonomic classification with Kraken2 and abundance estimation with Bracken
 */

nextflow.enable.dsl = 2

// Import modules
include { FASTQC } from '../modules/qc/fastqc/main'
include { KRAKEN2 } from '../modules/metagenomics/kraken2/main'
include { BRACKEN } from '../modules/metagenomics/bracken/main'

/*
 * Define the workflow
 */
workflow {
    // Sample input
    Channel.of(
        [
            [id: 'sample1', single_end: false],
            [
                file("/scratch/sdodl001/BioPipelines/data/raw/metagenomics/sample1_R1.fastq.gz"),
                file("/scratch/sdodl001/BioPipelines/data/raw/metagenomics/sample1_R2.fastq.gz")
            ]
        ]
    ).set { samples_ch }

    // Kraken2 database
    kraken_db = file("/scratch/sdodl001/BioPipelines/data/references/kraken2_db")

    // Step 1: Quality Control
    FASTQC(samples_ch)

    // Step 2: Taxonomic classification with Kraken2
    KRAKEN2(samples_ch, kraken_db)

    // Step 3: Abundance estimation with Bracken
    BRACKEN(KRAKEN2.out.report, kraken_db)
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
