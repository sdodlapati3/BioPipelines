#!/usr/bin/env nextflow

/*
 * ATAC-seq Pipeline
 * 
 * Workflow: FastQC -> Bowtie2 -> MACS2 peak calling
 */

nextflow.enable.dsl=2

include { FASTQC         } from '../modules/qc/fastqc/main'
include { BOWTIE2_ALIGN  } from '../modules/alignment/bowtie2/main'
include { MACS2_CALLPEAK } from '../modules/peakcalling/macs2/main'

workflow {
    // ATAC samples (paired-end)
    Channel
        .of(
            [
                [id: 'new_sample1', single_end: false, assay: 'ATAC-seq'],
                [
                    file('/scratch/sdodl001/BioPipelines/data/processed/atac_seq/new_sample1_R1.trimmed.fastq.gz'),
                    file('/scratch/sdodl001/BioPipelines/data/processed/atac_seq/new_sample1_R2.trimmed.fastq.gz')
                ]
            ],
            [
                [id: 'new_sample2', single_end: false, assay: 'ATAC-seq'],
                [
                    file('/scratch/sdodl001/BioPipelines/data/processed/atac_seq/new_sample2_R1.trimmed.fastq.gz'),
                    file('/scratch/sdodl001/BioPipelines/data/processed/atac_seq/new_sample2_R2.trimmed.fastq.gz')
                ]
            ]
        )
        .set { atac_samples }
    
    // Reference - Bowtie2 needs directory and all index files
    bowtie2_index_dir = file('/scratch/sdodl001/BioPipelines/data/references/bowtie2_index')
    bowtie2_index_files = Channel.fromPath('/scratch/sdodl001/BioPipelines/data/references/bowtie2_index/hg38.*.bt2*').collect()
    
    // QC
    FASTQC(atac_samples)
    
    // Align
    BOWTIE2_ALIGN(atac_samples, [bowtie2_index_dir, bowtie2_index_files])
    
    // Peak calling (no control for ATAC-seq)
    MACS2_CALLPEAK(
        BOWTIE2_ALIGN.out.bam,
        []
    )
    
    // Display results
    MACS2_CALLPEAK.out.peaks
        .map { m, peaks -> "${m.id}: ${peaks.name}" }
        .collect()
        .view { results -> "\n=== ATAC-seq Peaks ===\n" + results.join('\n') }
    
    workflow.onComplete {
        println """
        ============================================================
        ATAC-seq Pipeline Complete
        ============================================================
        Assay:    ATAC-seq (chromatin accessibility)
        Samples:  2 biological samples
        
        Pipeline Steps:
        - FastQC: Quality control
        - Bowtie2: Genome alignment
        - MACS2: Peak calling (open chromatin regions)
        
        Status:   ${workflow.success ? '✅ SUCCESS' : '❌ FAILED'}
        Duration: ${workflow.duration}
        
        💡 Check outputs:
           Peaks: find work -name '*_peaks.narrowPeak'
           BAMs:  find work -name '*.bam'
        ============================================================
        """.stripIndent()
    }
}
