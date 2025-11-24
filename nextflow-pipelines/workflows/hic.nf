#!/usr/bin/env nextflow

/*
 * Hi-C Contact Analysis Pipeline
 * ==============================
 * 3D genome organization analysis
 */

nextflow.enable.dsl = 2

// Import modules
include { FASTQC } from '../modules/qc/fastqc/main'
include { BOWTIE2_ALIGN } from '../modules/alignment/bowtie2/main'
include { PAIRTOOLS_PARSE } from '../modules/hic/pairtools_parse/main'
include { COOLER_CLOAD } from '../modules/hic/cooler_cload/main'

/*
 * Define the workflow
 */
workflow {
    // Sample input - Hi-C paired-end reads
    Channel.of(
        [
            [id: 'sample1', single_end: false],
            [
                file("/scratch/sdodl001/BioPipelines/data/processed/hic/sample1_R1.trimmed.fastq.gz"),
                file("/scratch/sdodl001/BioPipelines/data/processed/hic/sample1_R2.trimmed.fastq.gz")
            ]
        ]
    ).set { samples_ch }

    // Reference files
    bowtie2_index = file("/scratch/sdodl001/BioPipelines/data/references/bowtie2_index_hg38")
    chrom_sizes = file("/scratch/sdodl001/BioPipelines/data/references/hg38.chrom.sizes")
    
    // Resolution for contact matrix (10kb by default)
    resolution = 10000

    // Step 1: Quality Control
    FASTQC(samples_ch)

    // Step 2: Alignment with Bowtie2
    BOWTIE2_ALIGN(samples_ch, bowtie2_index)

    // Step 3: Parse pairs from aligned reads
    PAIRTOOLS_PARSE(BOWTIE2_ALIGN.out.bam, chrom_sizes)

    // Step 4: Create contact matrix
    COOLER_CLOAD(PAIRTOOLS_PARSE.out.pairs, chrom_sizes, resolution)
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
