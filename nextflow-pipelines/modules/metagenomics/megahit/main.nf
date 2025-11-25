/*
 * MEGAHIT Module
 * 
 * MEGAHIT - Ultra-fast and memory-efficient metagenomic assembler
 * Optimized for large and complex metagenomics datasets
 * Uses existing metagenomics container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * MEGAHIT assembly
 */
process MEGAHIT_ASSEMBLE {
    tag "megahit_${sample_id}"
    container "${params.containers.metagenomics}"
    
    publishDir "${params.outdir}/metagenomics/megahit", mode: 'copy'
    
    cpus params.megahit?.cpus ?: 32
    memory params.megahit?.memory ?: '128.GB'
    
    input:
    tuple val(sample_id), path(reads)
    
    output:
    tuple val(sample_id), path("${sample_id}/final.contigs.fa"), emit: contigs
    tuple val(sample_id), path("${sample_id}"), emit: assembly_dir
    path "${sample_id}/log", emit: log
    
    script:
    def min_contig_len = params.megahit?.min_contig_len ?: 200
    def k_min = params.megahit?.k_min ?: 21
    def k_max = params.megahit?.k_max ?: 141
    def k_step = params.megahit?.k_step ?: 12
    def preset = params.megahit?.preset ?: ""
    def preset_opt = preset ? "--presets ${preset}" : ""
    
    if (reads instanceof List) {
        """
        megahit \\
            -1 ${reads[0]} \\
            -2 ${reads[1]} \\
            -o ${sample_id} \\
            --min-contig-len ${min_contig_len} \\
            --k-min ${k_min} \\
            --k-max ${k_max} \\
            --k-step ${k_step} \\
            ${preset_opt} \\
            --num-cpu-threads ${task.cpus} \\
            --memory ${task.memory.toGiga()}000000000
        """
    } else {
        """
        megahit \\
            -r ${reads} \\
            -o ${sample_id} \\
            --min-contig-len ${min_contig_len} \\
            --k-min ${k_min} \\
            --k-max ${k_max} \\
            --k-step ${k_step} \\
            ${preset_opt} \\
            --num-cpu-threads ${task.cpus} \\
            --memory ${task.memory.toGiga()}000000000
        """
    }
}

/*
 * Workflow: MEGAHIT assembly
 */
workflow MEGAHIT_PIPELINE {
    take:
    reads_ch      // channel: [ val(sample_id), path(reads) ]
    
    main:
    MEGAHIT_ASSEMBLE(reads_ch)
    
    emit:
    contigs = MEGAHIT_ASSEMBLE.out.contigs
    assembly_dir = MEGAHIT_ASSEMBLE.out.assembly_dir
}
