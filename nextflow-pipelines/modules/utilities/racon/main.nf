/*
 * Racon Module
 * 
 * Racon - Consensus module for raw de novo genome assembly
 * Polishes genome assemblies using long reads
 * Uses existing long-read container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * Racon polish
 */
process RACON_POLISH {
    tag "racon_${sample_id}_round${round}"
    container "${params.containers.longread}"
    
    publishDir "${params.outdir}/assembly/racon", mode: 'copy'
    
    cpus params.racon?.cpus ?: 16
    memory params.racon?.memory ?: '64.GB'
    
    input:
    tuple val(sample_id), path(reads), path(assembly), path(alignments)
    val round
    
    output:
    tuple val(sample_id), path("${sample_id}_racon_round${round}.fasta"), emit: polished
    
    script:
    """
    racon \\
        -t ${task.cpus} \\
        ${reads} \\
        ${alignments} \\
        ${assembly} \\
        > ${sample_id}_racon_round${round}.fasta
    """
}

/*
 * Minimap2 alignment for Racon
 */
process RACON_ALIGN {
    tag "racon_align_${sample_id}_round${round}"
    container "${params.containers.longread}"
    
    cpus params.racon?.cpus ?: 16
    memory params.racon?.memory ?: '64.GB'
    
    input:
    tuple val(sample_id), path(reads), path(assembly)
    val round
    val preset
    
    output:
    tuple val(sample_id), path(reads), path(assembly), path("${sample_id}_round${round}.paf"), emit: alignment
    
    script:
    """
    minimap2 \\
        -x ${preset} \\
        -t ${task.cpus} \\
        ${assembly} \\
        ${reads} \\
        > ${sample_id}_round${round}.paf
    """
}

/*
 * Workflow: Racon polishing pipeline (multiple rounds)
 */
workflow RACON_PIPELINE {
    take:
    reads_ch       // channel: [ val(sample_id), path(reads), path(assembly) ]
    preset         // val: "map-pb", "map-ont", etc.
    polish_rounds  // val: number of polishing rounds (default: 4)
    
    main:
    def rounds = polish_rounds ?: 4
    def current_ch = reads_ch
    
    // First round
    RACON_ALIGN(current_ch, 1, preset)
    RACON_POLISH(RACON_ALIGN.out.alignment, 1)
    
    // Subsequent rounds
    if (rounds > 1) {
        def polished_ch = RACON_POLISH.out.polished.map { 
            sample_id, polished -> 
            [sample_id, current_ch.reads, polished] 
        }
        
        (2..rounds).each { round ->
            RACON_ALIGN(polished_ch, round, preset)
            RACON_POLISH(RACON_ALIGN.out.alignment, round)
            polished_ch = RACON_POLISH.out.polished.map { 
                sample_id, polished -> 
                [sample_id, current_ch.reads, polished] 
            }
        }
    }
    
    emit:
    polished = RACON_POLISH.out.polished
}
