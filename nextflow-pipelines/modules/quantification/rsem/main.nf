/*
 * RSEM Module
 * 
 * RSEM (RNA-Seq by Expectation-Maximization)
 * Accurate transcript quantification from RNA-seq data
 * Uses existing rna-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * RSEM Prepare Reference - Build reference index
 */
process RSEM_PREPARE_REFERENCE {
    tag "rsem_reference_${reference.baseName}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/reference/rsem", mode: 'copy'
    
    cpus params.rsem?.cpus ?: 8
    memory params.rsem?.memory ?: '32.GB'
    
    input:
    path reference
    path gtf
    
    output:
    path "rsem_reference/*", emit: index
    
    script:
    """
    mkdir -p rsem_reference
    
    rsem-prepare-reference \\
        --gtf ${gtf} \\
        --num-threads ${task.cpus} \\
        ${reference} \\
        rsem_reference/rsem_ref
    """
}

/*
 * RSEM Calculate Expression
 */
process RSEM_CALCULATE_EXPRESSION {
    tag "rsem_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/quantification/rsem", mode: 'copy'
    
    cpus params.rsem?.cpus ?: 8
    memory params.rsem?.memory ?: '32.GB'
    
    input:
    tuple val(sample_id), path(reads)
    path reference_index
    val strandedness  // "none", "forward", "reverse"
    
    output:
    tuple val(sample_id), path("${sample_id}.genes.results"), emit: genes
    tuple val(sample_id), path("${sample_id}.isoforms.results"), emit: isoforms
    path "${sample_id}.stat", emit: stats_dir
    
    script:
    def strand_opt = ""
    if (strandedness == "forward") {
        strand_opt = "--strandedness forward"
    } else if (strandedness == "reverse") {
        strand_opt = "--strandedness reverse"
    }
    
    def ref_prefix = reference_index[0].toString().replaceAll(/\.[^.]+$/, '')
    
    if (reads instanceof List) {
        // Paired-end
        """
        rsem-calculate-expression \\
            --paired-end \\
            --num-threads ${task.cpus} \\
            ${strand_opt} \\
            --estimate-rspd \\
            --append-names \\
            ${reads[0]} ${reads[1]} \\
            ${ref_prefix} \\
            ${sample_id}
        """
    } else {
        // Single-end
        """
        rsem-calculate-expression \\
            --num-threads ${task.cpus} \\
            ${strand_opt} \\
            --estimate-rspd \\
            --append-names \\
            ${reads} \\
            ${ref_prefix} \\
            ${sample_id}
        """
    }
}

/*
 * RSEM from BAM
 */
process RSEM_CALCULATE_EXPRESSION_BAM {
    tag "rsem_bam_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/quantification/rsem", mode: 'copy'
    
    cpus params.rsem?.cpus ?: 8
    memory params.rsem?.memory ?: '32.GB'
    
    input:
    tuple val(sample_id), path(bam)
    path reference_index
    val paired_end
    
    output:
    tuple val(sample_id), path("${sample_id}.genes.results"), emit: genes
    tuple val(sample_id), path("${sample_id}.isoforms.results"), emit: isoforms
    path "${sample_id}.stat", emit: stats_dir
    
    script:
    def paired_opt = paired_end ? "--paired-end" : ""
    def ref_prefix = reference_index[0].toString().replaceAll(/\.[^.]+$/, '')
    
    """
    rsem-calculate-expression \\
        --bam \\
        ${paired_opt} \\
        --num-threads ${task.cpus} \\
        --estimate-rspd \\
        --append-names \\
        ${bam} \\
        ${ref_prefix} \\
        ${sample_id}
    """
}

/*
 * RSEM with STAR alignment
 */
process RSEM_STAR {
    tag "rsem_star_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/quantification/rsem", mode: 'copy'
    publishDir "${params.outdir}/alignments", mode: 'copy',
        pattern: "*.bam"
    
    cpus params.rsem?.cpus ?: 16
    memory params.rsem?.memory ?: '64.GB'
    
    input:
    tuple val(sample_id), path(reads)
    path reference_index
    val strandedness
    
    output:
    tuple val(sample_id), path("${sample_id}.genes.results"), emit: genes
    tuple val(sample_id), path("${sample_id}.isoforms.results"), emit: isoforms
    tuple val(sample_id), path("${sample_id}.transcript.bam"), emit: bam
    path "${sample_id}.stat", emit: stats_dir
    
    script:
    def strand_opt = ""
    if (strandedness == "forward") {
        strand_opt = "--strandedness forward"
    } else if (strandedness == "reverse") {
        strand_opt = "--strandedness reverse"
    }
    
    def ref_prefix = reference_index[0].toString().replaceAll(/\.[^.]+$/, '')
    
    if (reads instanceof List) {
        """
        rsem-calculate-expression \\
            --paired-end \\
            --star \\
            --star-gzipped-read-file \\
            --num-threads ${task.cpus} \\
            ${strand_opt} \\
            --estimate-rspd \\
            --append-names \\
            ${reads[0]} ${reads[1]} \\
            ${ref_prefix} \\
            ${sample_id}
        """
    } else {
        """
        rsem-calculate-expression \\
            --star \\
            --star-gzipped-read-file \\
            --num-threads ${task.cpus} \\
            ${strand_opt} \\
            --estimate-rspd \\
            --append-names \\
            ${reads} \\
            ${ref_prefix} \\
            ${sample_id}
        """
    }
}

/*
 * RSEM Generate Data Matrix - Combine multiple samples
 */
process RSEM_GENERATE_DATA_MATRIX {
    tag "rsem_matrix"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/quantification/rsem", mode: 'copy'
    
    input:
    path gene_results
    
    output:
    path "genes_expression_matrix.txt", emit: gene_matrix
    
    script:
    def results_list = gene_results.collect { it.toString() }.join(' ')
    
    """
    rsem-generate-data-matrix ${results_list} > genes_expression_matrix.txt
    """
}

/*
 * Workflow: Complete RSEM pipeline with reference preparation
 */
workflow RSEM_PIPELINE {
    take:
    reads_ch       // channel: [ val(sample_id), path(reads) ]
    reference      // path: reference genome fasta
    gtf            // path: annotation GTF
    strandedness   // val: "none", "forward", "reverse"
    
    main:
    // Prepare reference
    RSEM_PREPARE_REFERENCE(reference, gtf)
    
    // Quantify expression
    RSEM_CALCULATE_EXPRESSION(
        reads_ch,
        RSEM_PREPARE_REFERENCE.out.index,
        strandedness
    )
    
    emit:
    genes = RSEM_CALCULATE_EXPRESSION.out.genes
    isoforms = RSEM_CALCULATE_EXPRESSION.out.isoforms
    stats = RSEM_CALCULATE_EXPRESSION.out.stats_dir
}
