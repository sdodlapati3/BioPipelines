/*
 * GSNAP Module
 * 
 * GSNAP - Genomic Short-read Nucleotide Alignment Program
 * Fast alignment with SNP-tolerant mapping
 * Uses existing rna-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * GMAP/GSNAP build index
 */
process GSNAP_BUILD {
    tag "gsnap_build"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/reference/gsnap", mode: 'copy'
    
    cpus params.gsnap?.cpus ?: 8
    memory params.gsnap?.memory ?: '64.GB'
    
    input:
    path reference
    val genome_name
    
    output:
    path "${genome_name}", emit: index
    
    script:
    """
    mkdir -p ${genome_name}
    
    gmap_build \\
        -d ${genome_name} \\
        -D . \\
        ${reference}
    """
}

/*
 * GSNAP alignment
 */
process GSNAP_ALIGN {
    tag "gsnap_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/alignment/gsnap", mode: 'copy'
    
    cpus params.gsnap?.cpus ?: 16
    memory params.gsnap?.memory ?: '32.GB'
    
    input:
    tuple val(sample_id), path(reads)
    path index
    val genome_name
    
    output:
    tuple val(sample_id), path("${sample_id}.bam"), emit: bam
    tuple val(sample_id), path("${sample_id}.bam.bai"), emit: bai
    path "${sample_id}.log", emit: log
    
    script:
    def max_mismatches = params.gsnap?.max_mismatches ?: 5
    def novelsplicing = params.gsnap?.novelsplicing ? "--novelsplicing=1" : "--novelsplicing=0"
    def nofails = params.gsnap?.nofails ? "--nofails" : ""
    
    if (reads instanceof List) {
        """
        gsnap \\
            -d ${genome_name} \\
            -D ${index}/.. \\
            -t ${task.cpus} \\
            -m ${max_mismatches} \\
            -A sam \\
            ${novelsplicing} \\
            ${nofails} \\
            ${reads[0]} ${reads[1]} \\
            2> ${sample_id}.log \\
            | samtools view -bS - \\
            | samtools sort -@ ${task.cpus} -o ${sample_id}.bam -
        
        samtools index ${sample_id}.bam
        """
    } else {
        """
        gsnap \\
            -d ${genome_name} \\
            -D ${index}/.. \\
            -t ${task.cpus} \\
            -m ${max_mismatches} \\
            -A sam \\
            ${novelsplicing} \\
            ${nofails} \\
            ${reads} \\
            2> ${sample_id}.log \\
            | samtools view -bS - \\
            | samtools sort -@ ${task.cpus} -o ${sample_id}.bam -
        
        samtools index ${sample_id}.bam
        """
    }
}

/*
 * Workflow: GSNAP alignment pipeline
 */
workflow GSNAP_PIPELINE {
    take:
    reads_ch      // channel: [ val(sample_id), path(reads) ]
    reference     // path: reference genome
    genome_name   // val: genome identifier
    
    main:
    GSNAP_BUILD(reference, genome_name)
    GSNAP_ALIGN(reads_ch, GSNAP_BUILD.out.index, genome_name)
    
    emit:
    bam = GSNAP_ALIGN.out.bam
    log = GSNAP_ALIGN.out.log
    index = GSNAP_BUILD.out.index
}
