/*
 * Trimmomatic Module
 * 
 * Trimmomatic - Flexible read trimming tool
 * Removes adapters and low-quality bases
 * Available in all pipeline containers
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * Trimmomatic - Paired-end trimming
 */
process TRIMMOMATIC_PE {
    tag "trim_pe_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/trimmed_reads", mode: 'copy',
        pattern: "*_R{1,2}_paired.fastq.gz"
    publishDir "${params.outdir}/trimmed_reads/unpaired", mode: 'copy',
        pattern: "*_unpaired.fastq.gz"
    publishDir "${params.outdir}/trimmed_reads/logs", mode: 'copy',
        pattern: "*.log"
    
    cpus params.trimmomatic?.cpus ?: 4
    memory params.trimmomatic?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(reads)
    path adapters
    
    output:
    tuple val(sample_id), path("${sample_id}_R{1,2}_paired.fastq.gz"), emit: paired_reads
    tuple val(sample_id), path("${sample_id}_R{1,2}_unpaired.fastq.gz"), emit: unpaired_reads
    path "${sample_id}_trimmomatic.log", emit: log
    
    script:
    def leading = params.trimmomatic?.leading ?: 3
    def trailing = params.trimmomatic?.trailing ?: 3
    def sliding_window = params.trimmomatic?.sliding_window ?: "4:15"
    def minlen = params.trimmomatic?.minlen ?: 36
    
    """
    trimmomatic PE \\
        -threads ${task.cpus} \\
        -phred33 \\
        ${reads[0]} ${reads[1]} \\
        ${sample_id}_R1_paired.fastq.gz ${sample_id}_R1_unpaired.fastq.gz \\
        ${sample_id}_R2_paired.fastq.gz ${sample_id}_R2_unpaired.fastq.gz \\
        ILLUMINACLIP:${adapters}:2:30:10 \\
        LEADING:${leading} \\
        TRAILING:${trailing} \\
        SLIDINGWINDOW:${sliding_window} \\
        MINLEN:${minlen} \\
        2> ${sample_id}_trimmomatic.log
    """
}

/*
 * Trimmomatic - Single-end trimming
 */
process TRIMMOMATIC_SE {
    tag "trim_se_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/trimmed_reads", mode: 'copy'
    publishDir "${params.outdir}/trimmed_reads/logs", mode: 'copy',
        pattern: "*.log"
    
    cpus params.trimmomatic?.cpus ?: 4
    memory params.trimmomatic?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(reads)
    path adapters
    
    output:
    tuple val(sample_id), path("${sample_id}_trimmed.fastq.gz"), emit: trimmed_reads
    path "${sample_id}_trimmomatic.log", emit: log
    
    script:
    def leading = params.trimmomatic?.leading ?: 3
    def trailing = params.trimmomatic?.trailing ?: 3
    def sliding_window = params.trimmomatic?.sliding_window ?: "4:15"
    def minlen = params.trimmomatic?.minlen ?: 36
    
    """
    trimmomatic SE \\
        -threads ${task.cpus} \\
        -phred33 \\
        ${reads} \\
        ${sample_id}_trimmed.fastq.gz \\
        ILLUMINACLIP:${adapters}:2:30:10 \\
        LEADING:${leading} \\
        TRAILING:${trailing} \\
        SLIDINGWINDOW:${sliding_window} \\
        MINLEN:${minlen} \\
        2> ${sample_id}_trimmomatic.log
    """
}

/*
 * Trimmomatic - Custom parameters
 */
process TRIMMOMATIC_CUSTOM {
    tag "trim_custom_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/trimmed_reads", mode: 'copy'
    
    cpus params.trimmomatic?.cpus ?: 4
    memory params.trimmomatic?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(reads)
    path adapters
    val trimming_params
    
    output:
    tuple val(sample_id), path("${sample_id}_*_paired.fastq.gz"), emit: paired_reads optional true
    tuple val(sample_id), path("${sample_id}_trimmed.fastq.gz"), emit: trimmed_reads optional true
    path "${sample_id}_trimmomatic.log", emit: log
    
    script:
    if (reads instanceof List) {
        // Paired-end
        """
        trimmomatic PE \\
            -threads ${task.cpus} \\
            -phred33 \\
            ${reads[0]} ${reads[1]} \\
            ${sample_id}_R1_paired.fastq.gz ${sample_id}_R1_unpaired.fastq.gz \\
            ${sample_id}_R2_paired.fastq.gz ${sample_id}_R2_unpaired.fastq.gz \\
            ${trimming_params} \\
            2> ${sample_id}_trimmomatic.log
        """
    } else {
        // Single-end
        """
        trimmomatic SE \\
            -threads ${task.cpus} \\
            -phred33 \\
            ${reads} \\
            ${sample_id}_trimmed.fastq.gz \\
            ${trimming_params} \\
            2> ${sample_id}_trimmomatic.log
        """
    }
}

/*
 * Workflow: Standard Trimmomatic pipeline
 */
workflow TRIMMOMATIC_WORKFLOW {
    take:
    reads_ch   // channel: [ val(sample_id), path(reads) ]
    adapters   // path: adapter fasta file
    
    main:
    // Auto-detect paired vs single-end
    reads_ch.branch {
        paired: it[1] instanceof List
        single: true
    }.set { branched }
    
    TRIMMOMATIC_PE(branched.paired, adapters)
    TRIMMOMATIC_SE(branched.single, adapters)
    
    // Combine outputs
    trimmed_reads = TRIMMOMATIC_PE.out.paired_reads.mix(TRIMMOMATIC_SE.out.trimmed_reads)
    
    emit:
    reads = trimmed_reads
    logs = TRIMMOMATIC_PE.out.log.mix(TRIMMOMATIC_SE.out.log)
}
