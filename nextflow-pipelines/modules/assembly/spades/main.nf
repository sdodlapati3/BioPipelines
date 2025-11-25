/*
 * SPAdes Module
 * 
 * SPAdes - Genome assembler for bacteria, fungi, and isolates
 * De Bruijn graph-based assembly
 * Uses existing metagenomics container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * SPAdes assembly
 */
process SPADES_ASSEMBLE {
    tag "spades_${sample_id}"
    container "${params.containers.metagenomics}"
    
    publishDir "${params.outdir}/assembly/spades", mode: 'copy'
    
    cpus params.spades?.cpus ?: 16
    memory params.spades?.memory ?: '64.GB'
    
    input:
    tuple val(sample_id), path(reads)
    
    output:
    path "${sample_id}_spades/*", emit: results
    path "${sample_id}_spades/scaffolds.fasta", emit: scaffolds
    path "${sample_id}_spades/contigs.fasta", emit: contigs
    path "${sample_id}_spades/assembly_graph.fastg", emit: graph
    
    script:
    if (reads instanceof List) {
        """
        spades.py \\
            -1 ${reads[0]} \\
            -2 ${reads[1]} \\
            -o ${sample_id}_spades \\
            -t ${task.cpus} \\
            -m ${task.memory.toGiga()}
        """
    } else {
        """
        spades.py \\
            -s ${reads} \\
            -o ${sample_id}_spades \\
            -t ${task.cpus} \\
            -m ${task.memory.toGiga()}
        """
    }
}

/*
 * metaSPAdes - Metagenomic assembly
 */
process METASPADES {
    tag "metaspades_${sample_id}"
    container "${params.containers.metagenomics}"
    
    publishDir "${params.outdir}/assembly/metaspades", mode: 'copy'
    
    cpus params.spades?.cpus ?: 16
    memory params.spades?.memory ?: '128.GB'
    
    input:
    tuple val(sample_id), path(reads)
    
    output:
    path "${sample_id}_metaspades/*", emit: results
    path "${sample_id}_metaspades/scaffolds.fasta", emit: scaffolds
    path "${sample_id}_metaspades/contigs.fasta", emit: contigs
    
    script:
    if (reads instanceof List) {
        """
        metaspades.py \\
            -1 ${reads[0]} \\
            -2 ${reads[1]} \\
            -o ${sample_id}_metaspades \\
            -t ${task.cpus} \\
            -m ${task.memory.toGiga()}
        """
    } else {
        """
        metaspades.py \\
            -s ${reads} \\
            -o ${sample_id}_metaspades \\
            -t ${task.cpus} \\
            -m ${task.memory.toGiga()}
        """
    }
}

/*
 * rnaSPAdes - Transcriptome assembly
 */
process RNASPADES {
    tag "rnaspades_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/assembly/rnaspades", mode: 'copy'
    
    cpus params.spades?.cpus ?: 16
    memory params.spades?.memory ?: '64.GB'
    
    input:
    tuple val(sample_id), path(reads)
    
    output:
    path "${sample_id}_rnaspades/*", emit: results
    path "${sample_id}_rnaspades/transcripts.fasta", emit: transcripts
    
    script:
    if (reads instanceof List) {
        """
        rnaspades.py \\
            -1 ${reads[0]} \\
            -2 ${reads[1]} \\
            -o ${sample_id}_rnaspades \\
            -t ${task.cpus} \\
            -m ${task.memory.toGiga()}
        """
    } else {
        """
        rnaspades.py \\
            -s ${reads} \\
            -o ${sample_id}_rnaspades \\
            -t ${task.cpus} \\
            -m ${task.memory.toGiga()}
        """
    }
}

/*
 * Workflow: SPAdes assembly pipeline
 */
workflow SPADES_PIPELINE {
    take:
    reads_ch     // channel: [ val(sample_id), path(reads) ]
    mode         // val: "isolate", "meta", "rna"
    
    main:
    if (mode == "meta") {
        METASPADES(reads_ch)
        assembly_out = METASPADES.out.scaffolds
    } else if (mode == "rna") {
        RNASPADES(reads_ch)
        assembly_out = RNASPADES.out.transcripts
    } else {
        SPADES_ASSEMBLE(reads_ch)
        assembly_out = SPADES_ASSEMBLE.out.scaffolds
    }
    
    emit:
    assembly = assembly_out
}
