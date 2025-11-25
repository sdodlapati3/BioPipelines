/*
 * Trim Galore Module
 * 
 * Trim Galore - Wrapper around Cutadapt and FastQC
 * Automatic quality and adapter trimming with QC
 * Uses existing rna-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * Trim Galore paired-end
 */
process TRIMGALORE_PE {
    tag "trimgalore_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/trimming/trimgalore", mode: 'copy'
    
    cpus params.trimgalore?.cpus ?: 4
    memory params.trimgalore?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(read1), path(read2)
    
    output:
    tuple val(sample_id), path("*_val_1.fq.gz"), path("*_val_2.fq.gz"), emit: reads
    path "*_trimming_report.txt", emit: report
    path "*_fastqc.{zip,html}", emit: fastqc
    
    script:
    def quality = params.trimgalore?.quality ?: 20
    def min_length = params.trimgalore?.min_length ?: 20
    def adapter1 = params.trimgalore?.adapter1 ?: ""
    def adapter2 = params.trimgalore?.adapter2 ?: ""
    def adapter1_opt = adapter1 ? "--adapter ${adapter1}" : ""
    def adapter2_opt = adapter2 ? "--adapter2 ${adapter2}" : ""
    def clip_r1 = params.trimgalore?.clip_r1 ?: 0
    def clip_r2 = params.trimgalore?.clip_r2 ?: 0
    def three_prime_clip_r1 = params.trimgalore?.three_prime_clip_r1 ?: 0
    def three_prime_clip_r2 = params.trimgalore?.three_prime_clip_r2 ?: 0
    
    """
    trim_galore \\
        --paired \\
        --quality ${quality} \\
        --length ${min_length} \\
        ${adapter1_opt} \\
        ${adapter2_opt} \\
        --clip_R1 ${clip_r1} \\
        --clip_R2 ${clip_r2} \\
        --three_prime_clip_R1 ${three_prime_clip_r1} \\
        --three_prime_clip_R2 ${three_prime_clip_r2} \\
        --cores ${task.cpus} \\
        --fastqc \\
        --gzip \\
        ${read1} ${read2}
    """
}

/*
 * Trim Galore single-end
 */
process TRIMGALORE_SE {
    tag "trimgalore_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/trimming/trimgalore", mode: 'copy'
    
    cpus params.trimgalore?.cpus ?: 4
    memory params.trimgalore?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(reads)
    
    output:
    tuple val(sample_id), path("*_trimmed.fq.gz"), emit: reads
    path "*_trimming_report.txt", emit: report
    path "*_fastqc.{zip,html}", emit: fastqc
    
    script:
    def quality = params.trimgalore?.quality ?: 20
    def min_length = params.trimgalore?.min_length ?: 20
    def adapter = params.trimgalore?.adapter ?: ""
    def adapter_opt = adapter ? "--adapter ${adapter}" : ""
    def clip_r1 = params.trimgalore?.clip_r1 ?: 0
    def three_prime_clip_r1 = params.trimgalore?.three_prime_clip_r1 ?: 0
    
    """
    trim_galore \\
        --quality ${quality} \\
        --length ${min_length} \\
        ${adapter_opt} \\
        --clip_R1 ${clip_r1} \\
        --three_prime_clip_R1 ${three_prime_clip_r1} \\
        --cores ${task.cpus} \\
        --fastqc \\
        --gzip \\
        ${reads}
    """
}

/*
 * Workflow: Trim Galore trimming
 */
workflow TRIMGALORE_PIPELINE {
    take:
    reads_ch      // channel: [ val(sample_id), path(reads) ]
    
    main:
    
    // Split into PE and SE
    pe_ch = reads_ch.filter { it[1] instanceof List && it[1].size() == 2 }
        .map { sample_id, reads -> tuple(sample_id, reads[0], reads[1]) }
    
    se_ch = reads_ch.filter { !(it[1] instanceof List) || it[1].size() == 1 }
        .map { sample_id, reads -> tuple(sample_id, reads instanceof List ? reads[0] : reads) }
    
    TRIMGALORE_PE(pe_ch)
    TRIMGALORE_SE(se_ch)
    
    // Combine outputs
    all_reads = TRIMGALORE_PE.out.reads.mix(TRIMGALORE_SE.out.reads)
    all_reports = TRIMGALORE_PE.out.report.mix(TRIMGALORE_SE.out.report)
    all_fastqc = TRIMGALORE_PE.out.fastqc.mix(TRIMGALORE_SE.out.fastqc)
    
    emit:
    reads = all_reads
    report = all_reports
    fastqc = all_fastqc
}
