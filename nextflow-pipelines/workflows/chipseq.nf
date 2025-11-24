#!/usr/bin/env nextflow

/*
 * ChIP-seq Pipeline
 * 
 * Workflow: FastQC -> Bowtie2 -> MACS2 peak calling
 */

nextflow.enable.dsl=2

include { FASTQC         } from '../modules/qc/fastqc/main'
include { BOWTIE2_ALIGN  } from '../modules/alignment/bowtie2/main'
include { MACS2_CALLPEAK } from '../modules/peakcalling/macs2/main'

workflow {
    // ChIP samples
    Channel
        .of(
            [
                [id: 'h3k4me3_rep1', single_end: true, antibody: 'H3K4me3', replicate: 1],
                file('/scratch/sdodl001/BioPipelines/data/processed/chip_seq/h3k4me3_rep1.trimmed.fastq.gz')
            ],
            [
                [id: 'h3k4me3_rep2', single_end: true, antibody: 'H3K4me3', replicate: 2],
                file('/scratch/sdodl001/BioPipelines/data/processed/chip_seq/h3k4me3_rep2.trimmed.fastq.gz')
            ]
        )
        .set { chip_samples }
    
    // Control sample
    control_meta = [id: 'input_control', single_end: true, type: 'input']
    control_reads = file('/scratch/sdodl001/BioPipelines/data/processed/chip_seq/input_control.trimmed.fastq.gz')
    
    // Reference
    bowtie2_index = file('/scratch/sdodl001/BioPipelines/data/references/bowtie2_index_hg38')
    
    // QC
    FASTQC(chip_samples)
    FASTQC([[control_meta, control_reads]])
    
    // Align ChIP samples
    BOWTIE2_ALIGN(chip_samples, bowtie2_index)
    
    // Align control
    control_align = BOWTIE2_ALIGN([[control_meta, control_reads]], bowtie2_index)
    
    // Peak calling with control
    MACS2_CALLPEAK(
        BOWTIE2_ALIGN.out.bam,
        control_align.bam.map { meta, bam -> bam }.first()
    )
    
    // Display results
    MACS2_CALLPEAK.out.peaks
        .map { m, peaks -> "${m.id}: ${peaks.name}" }
        .collect()
        .view { results -> "\n=== ChIP-seq Peaks ===\n" + results.join('\n') }
    
    workflow.onComplete {
        println """
        ============================================================
        ChIP-seq Pipeline Complete
        ============================================================
        Antibody: H3K4me3 (active promoter mark)
        Samples:  2 replicates + input control
        
        Pipeline Steps:
        - FastQC: Quality control
        - Bowtie2: Genome alignment
        - MACS2: Peak calling
        
        Status:   ${workflow.success ? '✅ SUCCESS' : '❌ FAILED'}
        Duration: ${workflow.duration}
        
        💡 Check outputs:
           Peaks: find work -name '*_peaks.narrowPeak'
           BAMs:  find work -name '*.bam'
        ============================================================
        """.stripIndent()
    }
}
