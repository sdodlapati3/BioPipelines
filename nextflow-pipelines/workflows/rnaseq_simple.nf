#!/usr/bin/env nextflow

/*
 * Simple RNA-seq Pipeline
 * 
 * Workflow: FastQC -> STAR -> featureCounts
 * Tests basic module composition with real data
 */

nextflow.enable.dsl=2

// Import modules
include { FASTQC        } from '../modules/qc/fastqc/main'
include { STAR_ALIGN    } from '../modules/alignment/star/main'
include { FEATURECOUNTS } from '../modules/quantification/featurecounts/main'

workflow {
    // Input: Single sample (mut_rep1) - paired-end RNA-seq
    def meta = [
        id: 'mut_rep1',
        single_end: false,
        strandedness: 'reverse'  // Standard Illumina TruSeq
    ]
    
    // Define input files
    reads = [
        file('/scratch/sdodl001/BioPipelines/data/processed/rna_seq/mut_rep1_R1.trimmed.fastq.gz'),
        file('/scratch/sdodl001/BioPipelines/data/processed/rna_seq/mut_rep1_R2.trimmed.fastq.gz')
    ]
    
    // Reference files
    star_index = file('/scratch/sdodl001/BioPipelines/data/references/star_index_hg38')
    gtf_file = file('/scratch/sdodl001/BioPipelines/data/references/gencode.v45.primary_assembly.annotation.gtf')
    
    // Create input channel
    input_ch = Channel.of([meta, reads])
    
    // Step 1: Quality Control
    FASTQC(input_ch)
    
    // Step 2: Alignment with STAR
    STAR_ALIGN(
        input_ch,
        star_index
    )
    
    // Step 3: Quantification with featureCounts
    FEATURECOUNTS(
        STAR_ALIGN.out.bam,
        gtf_file
    )
    
    // Display results
    FASTQC.out.html
        | map { m, html -> "FastQC: ${m.id} - ${html.size()} HTML files" }
        | view
    
    STAR_ALIGN.out.bam
        | map { m, bam -> "STAR: ${m.id} - ${bam.name}" }
        | view
    
    FEATURECOUNTS.out.counts
        | map { m, counts -> "Counts: ${m.id} - ${counts.name}" }
        | view
    
    // Summary
    println """
    ============================================================
    Simple RNA-seq Workflow Complete
    ============================================================
    Sample:  ${meta.id}
    Mode:    Paired-end
    Strand:  ${meta.strandedness}
    
    💡 Check outputs in work directory:
       FastQC:        find work -name '*.html'
       Alignments:    find work -name '*.bam'
       Gene counts:   find work -name '*.featureCounts.txt'
    ============================================================
    """.stripIndent()
}
