/*
 * Canu Module
 * 
 * Canu - Long-read genome assembler
 * Assembler for PacBio and Oxford Nanopore reads
 * Uses existing long-read container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * Canu assembly
 */
process CANU_ASSEMBLE {
    tag "canu_${sample_id}"
    container "${params.containers.longread}"
    
    publishDir "${params.outdir}/assembly/canu", mode: 'copy'
    
    cpus params.canu?.cpus ?: 32
    memory params.canu?.memory ?: '128.GB'
    
    input:
    tuple val(sample_id), path(reads)
    val genome_size
    val sequencing_type
    
    output:
    path "${sample_id}_canu/*", emit: results
    path "${sample_id}_canu/${sample_id}.contigs.fasta", emit: contigs
    path "${sample_id}_canu/${sample_id}.correctedReads.fasta.gz", emit: corrected_reads
    path "${sample_id}_canu/${sample_id}.report", emit: report
    
    script:
    def read_type = sequencing_type == "pacbio" ? "-pacbio" : "-nanopore"
    
    """
    canu \\
        -p ${sample_id} \\
        -d ${sample_id}_canu \\
        genomeSize=${genome_size} \\
        maxThreads=${task.cpus} \\
        maxMemory=${task.memory.toGiga()} \\
        ${read_type} ${reads}
    """
}

/*
 * Canu correction only
 */
process CANU_CORRECT {
    tag "canu_correct_${sample_id}"
    container "${params.containers.longread}"
    
    publishDir "${params.outdir}/corrected_reads/canu", mode: 'copy'
    
    cpus params.canu?.cpus ?: 32
    memory params.canu?.memory ?: '128.GB'
    
    input:
    tuple val(sample_id), path(reads)
    val genome_size
    val sequencing_type
    
    output:
    path "${sample_id}_corrected/*", emit: results
    path "${sample_id}_corrected/${sample_id}.correctedReads.fasta.gz", emit: corrected_reads
    
    script:
    def read_type = sequencing_type == "pacbio" ? "-pacbio" : "-nanopore"
    
    """
    canu \\
        -correct \\
        -p ${sample_id} \\
        -d ${sample_id}_corrected \\
        genomeSize=${genome_size} \\
        maxThreads=${task.cpus} \\
        maxMemory=${task.memory.toGiga()} \\
        ${read_type} ${reads}
    """
}

/*
 * Workflow: Canu assembly pipeline
 */
workflow CANU_PIPELINE {
    take:
    reads_ch         // channel: [ val(sample_id), path(reads) ]
    genome_size      // val: estimated genome size (e.g., "4.6m", "100m", "3g")
    sequencing_type  // val: "pacbio" or "nanopore"
    
    main:
    CANU_ASSEMBLE(reads_ch, genome_size, sequencing_type)
    
    emit:
    contigs = CANU_ASSEMBLE.out.contigs
    corrected_reads = CANU_ASSEMBLE.out.corrected_reads
    report = CANU_ASSEMBLE.out.report
}
