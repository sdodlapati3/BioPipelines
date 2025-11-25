/*
 * Trinity Module
 * 
 * Trinity - De novo RNA-seq assembly
 * Reconstructs transcriptomes from RNA-seq data
 * Uses existing rna-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * Trinity de novo assembly
 */
process TRINITY_DENOVO {
    tag "trinity_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/assembly/trinity", mode: 'copy'
    
    cpus params.trinity?.cpus ?: 16
    memory params.trinity?.memory ?: '64.GB'
    
    input:
    tuple val(sample_id), path(reads)
    val strandedness
    
    output:
    path "${sample_id}_trinity/*", emit: results
    path "${sample_id}_trinity/Trinity.fasta", emit: assembly
    path "${sample_id}_trinity/Trinity.fasta.gene_trans_map", emit: gene_map
    
    script:
    def strand_opt = ""
    if (strandedness == "forward") {
        strand_opt = "--SS_lib_type RF"
    } else if (strandedness == "reverse") {
        strand_opt = "--SS_lib_type FR"
    }
    
    if (reads instanceof List) {
        """
        Trinity \\
            --seqType fq \\
            --left ${reads[0]} \\
            --right ${reads[1]} \\
            ${strand_opt} \\
            --CPU ${task.cpus} \\
            --max_memory ${task.memory.toGiga()}G \\
            --output ${sample_id}_trinity
        """
    } else {
        """
        Trinity \\
            --seqType fq \\
            --single ${reads} \\
            ${strand_opt} \\
            --CPU ${task.cpus} \\
            --max_memory ${task.memory.toGiga()}G \\
            --output ${sample_id}_trinity
        """
    }
}

/*
 * Trinity genome-guided assembly
 */
process TRINITY_GENOME_GUIDED {
    tag "trinity_gg_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/assembly/trinity", mode: 'copy'
    
    cpus params.trinity?.cpus ?: 16
    memory params.trinity?.memory ?: '64.GB'
    
    input:
    tuple val(sample_id), path(bam)
    
    output:
    path "${sample_id}_trinity_gg/*", emit: results
    path "${sample_id}_trinity_gg/Trinity-GG.fasta", emit: assembly
    
    script:
    """
    Trinity \\
        --genome_guided_bam ${bam} \\
        --genome_guided_max_intron 10000 \\
        --CPU ${task.cpus} \\
        --max_memory ${task.memory.toGiga()}G \\
        --output ${sample_id}_trinity_gg
    """
}

/*
 * Workflow: Trinity assembly pipeline
 */
workflow TRINITY_PIPELINE {
    take:
    reads_ch      // channel: [ val(sample_id), path(reads) ]
    strandedness  // val: "none", "forward", "reverse"
    
    main:
    TRINITY_DENOVO(reads_ch, strandedness)
    
    emit:
    assembly = TRINITY_DENOVO.out.assembly
    gene_map = TRINITY_DENOVO.out.gene_map
}
