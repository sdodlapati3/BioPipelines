/*
 * Flye Module
 * 
 * Flye - De novo assembler for long reads
 * Fast assembler for PacBio and Oxford Nanopore reads
 * Uses existing long-read container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * Flye assembly
 */
process FLYE_ASSEMBLE {
    tag "flye_${sample_id}"
    container "${params.containers.longread}"
    
    publishDir "${params.outdir}/assembly/flye", mode: 'copy'
    
    cpus params.flye?.cpus ?: 32
    memory params.flye?.memory ?: '128.GB'
    
    input:
    tuple val(sample_id), path(reads)
    val genome_size
    val read_type
    
    output:
    path "${sample_id}_flye/*", emit: results
    path "${sample_id}_flye/assembly.fasta", emit: assembly
    path "${sample_id}_flye/assembly_graph.gfa", emit: graph
    path "${sample_id}_flye/assembly_info.txt", emit: info
    
    script:
    def type_opt = ""
    if (read_type == "pacbio-raw") {
        type_opt = "--pacbio-raw"
    } else if (read_type == "pacbio-corr") {
        type_opt = "--pacbio-corr"
    } else if (read_type == "pacbio-hifi") {
        type_opt = "--pacbio-hifi"
    } else if (read_type == "nano-raw") {
        type_opt = "--nano-raw"
    } else if (read_type == "nano-corr") {
        type_opt = "--nano-corr"
    } else if (read_type == "nano-hq") {
        type_opt = "--nano-hq"
    }
    
    def meta = params.flye?.meta ? "--meta" : ""
    def polish = params.flye?.polish_iterations ?: 1
    
    """
    flye \\
        ${type_opt} ${reads} \\
        --out-dir ${sample_id}_flye \\
        --genome-size ${genome_size} \\
        --threads ${task.cpus} \\
        --iterations ${polish} \\
        ${meta}
    """
}

/*
 * Workflow: Flye assembly pipeline
 */
workflow FLYE_PIPELINE {
    take:
    reads_ch     // channel: [ val(sample_id), path(reads) ]
    genome_size  // val: estimated genome size (e.g., "4.6m", "100m", "3g")
    read_type    // val: "pacbio-raw", "pacbio-corr", "pacbio-hifi", "nano-raw", "nano-corr", "nano-hq"
    
    main:
    FLYE_ASSEMBLE(reads_ch, genome_size, read_type)
    
    emit:
    assembly = FLYE_ASSEMBLE.out.assembly
    graph = FLYE_ASSEMBLE.out.graph
    info = FLYE_ASSEMBLE.out.info
}
