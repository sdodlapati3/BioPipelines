#!/usr/bin/env nextflow

/*
 * Multi-Sample RNA-seq Pipeline
 * 
 * Workflow: FastQC -> STAR -> featureCounts
 * Processes multiple samples in parallel
 */

nextflow.enable.dsl=2

// Import modules
include { FASTQC        } from '../modules/qc/fastqc/main'
include { STAR_ALIGN    } from '../modules/alignment/star/main'
include { FEATURECOUNTS } from '../modules/quantification/featurecounts/main'

workflow {
    // Define all RNA-seq samples
    Channel
        .of(
            [
                [id: 'wt_rep1', single_end: false, strandedness: 'reverse', condition: 'wildtype', replicate: 1],
                [
                    file('/scratch/sdodl001/BioPipelines/data/processed/rna_seq/wt_rep1_R1.trimmed.fastq.gz'),
                    file('/scratch/sdodl001/BioPipelines/data/processed/rna_seq/wt_rep1_R2.trimmed.fastq.gz')
                ]
            ],
            [
                [id: 'wt_rep2', single_end: false, strandedness: 'reverse', condition: 'wildtype', replicate: 2],
                [
                    file('/scratch/sdodl001/BioPipelines/data/processed/rna_seq/wt_rep2_R1.trimmed.fastq.gz'),
                    file('/scratch/sdodl001/BioPipelines/data/processed/rna_seq/wt_rep2_R2.trimmed.fastq.gz')
                ]
            ],
            [
                [id: 'mut_rep1', single_end: false, strandedness: 'reverse', condition: 'mutant', replicate: 1],
                [
                    file('/scratch/sdodl001/BioPipelines/data/processed/rna_seq/mut_rep1_R1.trimmed.fastq.gz'),
                    file('/scratch/sdodl001/BioPipelines/data/processed/rna_seq/mut_rep1_R2.trimmed.fastq.gz')
                ]
            ],
            [
                [id: 'mut_rep2', single_end: false, strandedness: 'reverse', condition: 'mutant', replicate: 2],
                [
                    file('/scratch/sdodl001/BioPipelines/data/processed/rna_seq/mut_rep2_R1.trimmed.fastq.gz'),
                    file('/scratch/sdodl001/BioPipelines/data/processed/rna_seq/mut_rep2_R2.trimmed.fastq.gz')
                ]
            ]
        )
        .set { samples_ch }
    
    // Reference files
    star_index = file('/scratch/sdodl001/BioPipelines/data/references/star_index_hg38')
    gtf_file = file('/scratch/sdodl001/BioPipelines/data/references/genes_GRCh38.gtf')
    
    // Run QC on all samples (parallel)
    FASTQC(samples_ch)
    
    // Align all samples (parallel)
    STAR_ALIGN(
        samples_ch,
        star_index
    )
    
    // Quantify all samples (parallel)
    FEATURECOUNTS(
        STAR_ALIGN.out.bam,
        gtf_file
    )
    
    // Collect and display results
    FASTQC.out.html
        .map { m, html -> "${m.id}: ${html.size()} HTML files" }
        .collect()
        .view { results -> "\n=== FastQC Results ===\n" + results.join('\n') }
    
    STAR_ALIGN.out.bam
        .map { m, bam -> "${m.id}: ${bam.name}" }
        .collect()
        .view { results -> "\n=== STAR Alignments ===\n" + results.join('\n') }
    
    FEATURECOUNTS.out.counts
        .map { m, counts -> "${m.id}: ${counts.name}" }
        .collect()
        .view { results -> "\n=== Gene Counts ===\n" + results.join('\n') }
    
    // Summary
    workflow.onComplete {
        println """
        ============================================================
        Multi-Sample RNA-seq Workflow Complete
        ============================================================
        Samples:  4 (wt_rep1, wt_rep2, mut_rep1, mut_rep2)
        Mode:     Paired-end
        Strand:   Reverse
        
        Pipeline Steps:
        - FastQC: Quality control for all samples
        - STAR: Genome alignment with hg38
        - featureCounts: Gene expression quantification
        
        Status:   ${workflow.success ? '✅ SUCCESS' : '❌ FAILED'}
        Duration: ${workflow.duration}
        
        💡 Check outputs:
           FastQC:      find work -name '*_fastqc.html'
           Alignments:  find work -name '*.bam'
           Counts:      find work -name '*.featureCounts.txt'
        ============================================================
        """.stripIndent()
    }
}
