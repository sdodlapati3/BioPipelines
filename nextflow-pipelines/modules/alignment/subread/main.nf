/*
 * Subread Module
 * 
 * Subread - Complete RNA-seq analysis suite
 * Includes Subjunc (aligner) and featureCounts (quantification)
 * Uses existing rna-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * Subread index
 */
process SUBREAD_INDEX {
    tag "subread_index"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/reference/subread", mode: 'copy'
    
    cpus params.subread?.cpus ?: 8
    memory params.subread?.memory ?: '32.GB'
    
    input:
    path reference
    
    output:
    path "subread_index", emit: index
    
    script:
    """
    mkdir subread_index
    
    subread-buildindex \\
        -o subread_index/genome \\
        ${reference}
    """
}

/*
 * Subjunc alignment (splice-aware)
 */
process SUBJUNC_ALIGN {
    tag "subjunc_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/alignment/subjunc", mode: 'copy'
    
    cpus params.subjunc?.cpus ?: 16
    memory params.subjunc?.memory ?: '32.GB'
    
    input:
    tuple val(sample_id), path(reads)
    path index
    
    output:
    tuple val(sample_id), path("${sample_id}.bam"), emit: bam
    path "${sample_id}.bam.indel.vcf", emit: indels
    path "${sample_id}.log", emit: log
    
    script:
    def max_mismatch = params.subjunc?.max_mismatch ?: 3
    def max_indel = params.subjunc?.max_indel ?: 5
    
    if (reads instanceof List) {
        """
        subjunc \\
            -i ${index}/genome \\
            -r ${reads[0]} \\
            -R ${reads[1]} \\
            -T ${task.cpus} \\
            -M ${max_mismatch} \\
            -I ${max_indel} \\
            --allJunctions \\
            -o ${sample_id}.bam \\
            > ${sample_id}.log 2>&1
        """
    } else {
        """
        subjunc \\
            -i ${index}/genome \\
            -r ${reads} \\
            -T ${task.cpus} \\
            -M ${max_mismatch} \\
            -I ${max_indel} \\
            --allJunctions \\
            -o ${sample_id}.bam \\
            > ${sample_id}.log 2>&1
        """
    }
}

/*
 * Subread-align (DNA-seq aligner)
 */
process SUBREAD_ALIGN {
    tag "subread_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/alignment/subread", mode: 'copy'
    
    cpus params.subread?.cpus ?: 16
    memory params.subread?.memory ?: '32.GB'
    
    input:
    tuple val(sample_id), path(reads)
    path index
    
    output:
    tuple val(sample_id), path("${sample_id}.bam"), emit: bam
    path "${sample_id}.bam.indel.vcf", emit: indels
    path "${sample_id}.log", emit: log
    
    script:
    def max_mismatch = params.subread?.max_mismatch ?: 3
    
    if (reads instanceof List) {
        """
        subread-align \\
            -i ${index}/genome \\
            -r ${reads[0]} \\
            -R ${reads[1]} \\
            -T ${task.cpus} \\
            -M ${max_mismatch} \\
            -o ${sample_id}.bam \\
            > ${sample_id}.log 2>&1
        """
    } else {
        """
        subread-align \\
            -i ${index}/genome \\
            -r ${reads} \\
            -T ${task.cpus} \\
            -M ${max_mismatch} \\
            -o ${sample_id}.bam \\
            > ${sample_id}.log 2>&1
        """
    }
}

/*
 * Workflow: Subread alignment pipeline
 */
workflow SUBREAD_PIPELINE {
    take:
    reads_ch      // channel: [ val(sample_id), path(reads) ]
    reference     // path: reference genome
    aligner       // val: 'subjunc' or 'subread'
    
    main:
    SUBREAD_INDEX(reference)
    
    if (aligner == 'subjunc') {
        SUBJUNC_ALIGN(reads_ch, SUBREAD_INDEX.out.index)
        bam_out = SUBJUNC_ALIGN.out.bam
        log_out = SUBJUNC_ALIGN.out.log
    } else {
        SUBREAD_ALIGN(reads_ch, SUBREAD_INDEX.out.index)
        bam_out = SUBREAD_ALIGN.out.bam
        log_out = SUBREAD_ALIGN.out.log
    }
    
    emit:
    bam = bam_out
    log = log_out
    index = SUBREAD_INDEX.out.index
}
