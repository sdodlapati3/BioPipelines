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
    
    // Reference - Bowtie2 needs directory and all index files
    bowtie2_index_dir = file('/scratch/sdodl001/BioPipelines/data/references/bowtie2_index')
    bowtie2_index_files = Channel.fromPath('/scratch/sdodl001/BioPipelines/data/references/bowtie2_index/hg38.*.bt2*').collect()
    
    // Combine all samples (ChIP + control) for QC and alignment
    all_samples = chip_samples.mix(Channel.of([control_meta, control_reads]))
    
    // QC on all samples
    FASTQC(all_samples)
    
    // Align all samples
    BOWTIE2_ALIGN(all_samples, [bowtie2_index_dir, bowtie2_index_files])
    
    // Separate control from ChIP samples for peak calling
    control_bam = BOWTIE2_ALIGN.out.bam
        .filter { meta, bam, bai -> meta.id == 'input_control' }
        .map { meta, bam, bai -> bam }
        .first()
    
    chip_bams = BOWTIE2_ALIGN.out.bam
        .filter { meta, bam, bai -> meta.id != 'input_control' }
    
    // Peak calling with control
    MACS2_CALLPEAK(
        chip_bams,
        control_bam
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
