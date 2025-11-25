/*
 * Mosdepth Module
 * 
 * Mosdepth - Fast BAM/CRAM depth calculation
 * Per-base and per-region coverage analysis
 * Uses existing dna-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * Mosdepth - Calculate coverage depth
 */
process MOSDEPTH {
    tag "mosdepth_${sample_id}"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/qc/mosdepth", mode: 'copy'
    
    cpus params.mosdepth?.cpus ?: 4
    memory params.mosdepth?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(bam), path(bai)
    path bed
    
    output:
    tuple val(sample_id), path("${sample_id}.mosdepth.global.dist.txt"), emit: global_dist
    tuple val(sample_id), path("${sample_id}.mosdepth.region.dist.txt"), emit: region_dist
    tuple val(sample_id), path("${sample_id}.mosdepth.summary.txt"), emit: summary
    tuple val(sample_id), path("${sample_id}.per-base.bed.gz"), emit: per_base, optional: true
    tuple val(sample_id), path("${sample_id}.regions.bed.gz"), emit: regions, optional: true
    
    script:
    def bed_opt = bed ? "--by ${bed}" : ""
    def no_per_base = params.mosdepth?.no_per_base ? "--no-per-base" : ""
    
    """
    mosdepth \\
        --threads ${task.cpus} \\
        ${bed_opt} \\
        ${no_per_base} \\
        ${sample_id} \\
        ${bam}
    """
}

/*
 * Mosdepth with quantization (for fast coverage bins)
 */
process MOSDEPTH_QUANTIZED {
    tag "mosdepth_quant_${sample_id}"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/qc/mosdepth", mode: 'copy'
    
    cpus params.mosdepth?.cpus ?: 4
    memory params.mosdepth?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(bam), path(bai)
    val quantize
    
    output:
    tuple val(sample_id), path("${sample_id}.mosdepth.global.dist.txt"), emit: global_dist
    tuple val(sample_id), path("${sample_id}.mosdepth.summary.txt"), emit: summary
    tuple val(sample_id), path("${sample_id}.quantized.bed.gz"), emit: quantized
    
    script:
    """
    mosdepth \\
        --threads ${task.cpus} \\
        --quantize ${quantize} \\
        --no-per-base \\
        ${sample_id} \\
        ${bam}
    """
}

/*
 * Workflow: Mosdepth coverage analysis
 */
workflow MOSDEPTH_PIPELINE {
    take:
    bam_ch    // channel: [ val(sample_id), path(bam), path(bai) ]
    bed       // path: optional BED file with regions
    
    main:
    MOSDEPTH(bam_ch, bed)
    
    emit:
    global_dist = MOSDEPTH.out.global_dist
    region_dist = MOSDEPTH.out.region_dist
    summary = MOSDEPTH.out.summary
}
