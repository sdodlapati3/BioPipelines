/*
 * Bowtie Module
 * 
 * Bowtie - Ultrafast short-read aligner (original version)
 * Specialized for aligning short reads without gaps
 * Uses existing rna-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * Bowtie build index
 */
process BOWTIE_BUILD {
    tag "bowtie_build"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/reference/bowtie", mode: 'copy'
    
    cpus params.bowtie?.cpus ?: 8
    memory params.bowtie?.memory ?: '32.GB'
    
    input:
    path reference
    
    output:
    path "bowtie_index", emit: index
    
    script:
    """
    mkdir bowtie_index
    
    bowtie-build \\
        --threads ${task.cpus} \\
        ${reference} \\
        bowtie_index/genome
    """
}

/*
 * Bowtie alignment
 */
process BOWTIE_ALIGN {
    tag "bowtie_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/alignment/bowtie", mode: 'copy'
    
    cpus params.bowtie?.cpus ?: 16
    memory params.bowtie?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(reads)
    path index
    
    output:
    tuple val(sample_id), path("${sample_id}.bam"), emit: bam
    path "${sample_id}.log", emit: log
    
    script:
    def max_mismatch = params.bowtie?.max_mismatch ?: 2
    def best = params.bowtie?.best ? "--best" : ""
    def strata = params.bowtie?.strata ? "--strata" : ""
    def suppress = params.bowtie?.suppress ?: 5
    
    if (reads instanceof List) {
        """
        bowtie \\
            -p ${task.cpus} \\
            -v ${max_mismatch} \\
            ${best} \\
            ${strata} \\
            -m ${suppress} \\
            --sam \\
            ${index}/genome \\
            -1 ${reads[0]} \\
            -2 ${reads[1]} \\
            2> ${sample_id}.log \\
            | samtools view -bS - \\
            | samtools sort -@ ${task.cpus} -o ${sample_id}.bam -
        
        samtools index ${sample_id}.bam
        """
    } else {
        """
        bowtie \\
            -p ${task.cpus} \\
            -v ${max_mismatch} \\
            ${best} \\
            ${strata} \\
            -m ${suppress} \\
            --sam \\
            ${index}/genome \\
            ${reads} \\
            2> ${sample_id}.log \\
            | samtools view -bS - \\
            | samtools sort -@ ${task.cpus} -o ${sample_id}.bam -
        
        samtools index ${sample_id}.bam
        """
    }
}

/*
 * Workflow: Bowtie alignment pipeline
 */
workflow BOWTIE_PIPELINE {
    take:
    reads_ch      // channel: [ val(sample_id), path(reads) ]
    reference     // path: reference genome
    
    main:
    BOWTIE_BUILD(reference)
    BOWTIE_ALIGN(reads_ch, BOWTIE_BUILD.out.index)
    
    emit:
    bam = BOWTIE_ALIGN.out.bam
    log = BOWTIE_ALIGN.out.log
    index = BOWTIE_BUILD.out.index
}
