/*
 * HTSeq Module
 * 
 * HTSeq-count - Gene expression quantification
 * Counts reads mapping to genomic features
 * Uses existing rna-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * HTSeq-count - Gene-level quantification
 */
process HTSEQ_COUNT {
    tag "htseq_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/quantification", mode: 'copy'
    
    cpus params.htseq?.cpus ?: 2
    memory params.htseq?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(bam)
    path gtf
    val strand_mode  // "yes", "no", "reverse"
    val feature_type  // "exon" (default), "gene"
    val id_attribute  // "gene_id" (default), "gene_name"
    
    output:
    tuple val(sample_id), path("${sample_id}.counts.txt"), emit: counts
    path "${sample_id}.counts.summary.txt", emit: summary
    
    script:
    def order = params.htseq?.order ?: "name"  // "name" or "pos"
    def feature = feature_type ?: "exon"
    def id_attr = id_attribute ?: "gene_id"
    def mode = params.htseq?.mode ?: "union"  // "union", "intersection-strict", "intersection-nonempty"
    def nonunique = params.htseq?.nonunique ?: "none"  // "none", "all", "fraction", "random"
    
    """
    htseq-count \\
        --format bam \\
        --order ${order} \\
        --stranded ${strand_mode} \\
        --type ${feature} \\
        --idattr ${id_attr} \\
        --mode ${mode} \\
        --nonunique ${nonunique} \\
        ${bam} \\
        ${gtf} \\
        > ${sample_id}.counts.txt 2> ${sample_id}.counts.summary.txt
    """
}

/*
 * HTSeq-count for sorted BAMs
 */
process HTSEQ_COUNT_SORTED {
    tag "htseq_sorted_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/quantification", mode: 'copy'
    
    cpus params.htseq?.cpus ?: 2
    memory params.htseq?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(bam)
    tuple val(sample_id), path(bai)
    path gtf
    val strand_mode
    
    output:
    tuple val(sample_id), path("${sample_id}.counts.txt"), emit: counts
    path "${sample_id}.counts.summary.txt", emit: summary
    
    script:
    """
    htseq-count \\
        --format bam \\
        --order pos \\
        --stranded ${strand_mode} \\
        --type exon \\
        --idattr gene_id \\
        --mode union \\
        ${bam} \\
        ${gtf} \\
        > ${sample_id}.counts.txt 2> ${sample_id}.counts.summary.txt
    """
}

/*
 * HTSeq-count with custom parameters
 */
process HTSEQ_COUNT_CUSTOM {
    tag "htseq_custom_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/quantification", mode: 'copy'
    
    cpus params.htseq?.cpus ?: 2
    memory params.htseq?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(bam)
    path gtf
    val extra_params
    
    output:
    tuple val(sample_id), path("${sample_id}.counts.txt"), emit: counts
    path "${sample_id}.counts.summary.txt", emit: summary
    
    script:
    """
    htseq-count \\
        --format bam \\
        ${extra_params} \\
        ${bam} \\
        ${gtf} \\
        > ${sample_id}.counts.txt 2> ${sample_id}.counts.summary.txt
    """
}

/*
 * HTSeq-count for multiple BAMs
 */
process HTSEQ_COUNT_MULTI {
    tag "htseq_multi"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/quantification", mode: 'copy'
    
    cpus params.htseq?.cpus ?: 4
    memory params.htseq?.memory ?: '16.GB'
    
    input:
    path bams
    path gtf
    val strand_mode
    
    output:
    path "multi_sample.counts.txt", emit: counts
    path "multi_sample.counts.summary.txt", emit: summary
    
    script:
    def bam_list = bams.collect { it.toString() }.join(' ')
    
    """
    htseq-count \\
        --format bam \\
        --order name \\
        --stranded ${strand_mode} \\
        --type exon \\
        --idattr gene_id \\
        --mode union \\
        ${bam_list} \\
        ${gtf} \\
        > multi_sample.counts.txt 2> multi_sample.counts.summary.txt
    """
}

/*
 * Workflow: Standard HTSeq pipeline
 */
workflow HTSEQ_WORKFLOW {
    take:
    bam_ch        // channel: [ val(sample_id), path(bam) ]
    gtf           // path: annotation GTF
    strand_mode   // val: "yes", "no", "reverse"
    
    main:
    HTSEQ_COUNT(
        bam_ch,
        gtf,
        strand_mode,
        "exon",
        "gene_id"
    )
    
    emit:
    counts = HTSEQ_COUNT.out.counts
    summary = HTSEQ_COUNT.out.summary
}
