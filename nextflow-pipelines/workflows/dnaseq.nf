#!/usr/bin/env nextflow

/*
 * DNA-seq Variant Calling Pipeline
 * =================================
 * GATK best practices for germline variant calling
 */

nextflow.enable.dsl = 2

// Import modules
include { FASTQC } from '../modules/qc/fastqc/main'
include { BWAMEM_ALIGN } from '../modules/alignment/bwamem/main'
include { PICARD_MARKDUPLICATES } from '../modules/processing/markduplicates/main'
include { GATK_HAPLOTYPECALLER } from '../modules/variant_calling/gatk_haplotypecaller/main'

/*
 * Define the workflow
 */
workflow {
    // Sample input
    Channel.of(
        [
            [id: 'sample1', single_end: false],
            [
                file("/scratch/sdodl001/BioPipelines/data/raw/dna_seq/sample1_R1.fastq.gz"),
                file("/scratch/sdodl001/BioPipelines/data/raw/dna_seq/sample1_R2.fastq.gz")
            ]
        ]
    ).set { samples_ch }

    // Reference files - BWA needs all index files staged
    bwa_index_fa = file("/scratch/sdodl001/BioPipelines/data/references/hg38.fa")
    bwa_index_files = Channel.fromPath("/scratch/sdodl001/BioPipelines/data/references/hg38.fa.{amb,ann,bwt,pac,sa}").collect()
    genome_fasta = file("/scratch/sdodl001/BioPipelines/data/references/hg38.fa")
    genome_fai = file("/scratch/sdodl001/BioPipelines/data/references/hg38.fa.fai")
    genome_dict = file("/scratch/sdodl001/BioPipelines/data/references/hg38.dict")

    // Step 1: Quality Control
    FASTQC(samples_ch)

    // Step 2: Alignment with BWA-MEM
    BWAMEM_ALIGN(samples_ch, [bwa_index_fa, bwa_index_files])

    // Step 3: Mark Duplicates
    PICARD_MARKDUPLICATES(BWAMEM_ALIGN.out.bam)

    // Step 4: Variant Calling
    GATK_HAPLOTYPECALLER(
        PICARD_MARKDUPLICATES.out.bam,
        genome_fasta,
        genome_fai,
        genome_dict
    )
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
