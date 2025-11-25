/*
 * TopHat2 Module
 * 
 * TopHat2 - Splice-aware RNA-seq aligner
 * Alignment of RNA-seq reads to reference genome
 * Uses existing rna-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * TopHat2 Align - Align RNA-seq reads
 */
process TOPHAT2_ALIGN {
    tag "tophat2_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/alignments/tophat2", mode: 'copy'
    
    cpus params.tophat2?.cpus ?: 8
    memory params.tophat2?.memory ?: '32.GB'
    
    input:
    tuple val(sample_id), path(reads)
    path index
    path gtf
    val strandedness
    
    output:
    tuple val(sample_id), path("${sample_id}/accepted_hits.bam"), emit: bam
    path "${sample_id}/align_summary.txt", emit: summary
    path "${sample_id}/junctions.bed", emit: junctions
    path "${sample_id}/insertions.bed", emit: insertions
    path "${sample_id}/deletions.bed", emit: deletions
    
    script:
    def library_type = ""
    if (strandedness == "forward") {
        library_type = "--library-type fr-firststrand"
    } else if (strandedness == "reverse") {
        library_type = "--library-type fr-secondstrand"
    } else {
        library_type = "--library-type fr-unstranded"
    }
    
    def gtf_opt = gtf ? "-G ${gtf}" : ""
    
    // Get base name of index (remove .1.bt2 extension)
    def index_base = index[0].toString().replaceAll(/\.1\.bt2.*$/, '')
    
    if (reads instanceof List) {
        // Paired-end
        """
        tophat2 \\
            -o ${sample_id} \\
            -p ${task.cpus} \\
            ${library_type} \\
            ${gtf_opt} \\
            --no-novel-juncs \\
            ${index_base} \\
            ${reads[0]} ${reads[1]}
        """
    } else {
        // Single-end
        """
        tophat2 \\
            -o ${sample_id} \\
            -p ${task.cpus} \\
            ${library_type} \\
            ${gtf_opt} \\
            --no-novel-juncs \\
            ${index_base} \\
            ${reads}
        """
    }
}

/*
 * TopHat2 with novel junction discovery
 */
process TOPHAT2_DENOVO {
    tag "tophat2_denovo_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/alignments/tophat2_denovo", mode: 'copy'
    
    cpus params.tophat2?.cpus ?: 8
    memory params.tophat2?.memory ?: '32.GB'
    
    input:
    tuple val(sample_id), path(reads)
    path index
    path gtf
    val strandedness
    
    output:
    tuple val(sample_id), path("${sample_id}/accepted_hits.bam"), emit: bam
    path "${sample_id}/align_summary.txt", emit: summary
    path "${sample_id}/junctions.bed", emit: junctions
    path "${sample_id}/insertions.bed", emit: insertions
    path "${sample_id}/deletions.bed", emit: deletions
    
    script:
    def library_type = ""
    if (strandedness == "forward") {
        library_type = "--library-type fr-firststrand"
    } else if (strandedness == "reverse") {
        library_type = "--library-type fr-secondstrand"
    } else {
        library_type = "--library-type fr-unstranded"
    }
    
    def gtf_opt = gtf ? "-G ${gtf}" : ""
    def index_base = index[0].toString().replaceAll(/\.1\.bt2.*$/, '')
    
    if (reads instanceof List) {
        """
        tophat2 \\
            -o ${sample_id} \\
            -p ${task.cpus} \\
            ${library_type} \\
            ${gtf_opt} \\
            ${index_base} \\
            ${reads[0]} ${reads[1]}
        """
    } else {
        """
        tophat2 \\
            -o ${sample_id} \\
            -p ${task.cpus} \\
            ${library_type} \\
            ${gtf_opt} \\
            ${index_base} \\
            ${reads}
        """
    }
}

/*
 * Workflow: TopHat2 alignment pipeline
 */
workflow TOPHAT2_PIPELINE {
    take:
    reads_ch       // channel: [ val(sample_id), path(reads) ]
    index          // path: Bowtie2 index files
    gtf            // path: annotation GTF
    strandedness   // val: "none", "forward", "reverse"
    allow_novel    // val: true/false for novel junction discovery
    
    main:
    if (allow_novel) {
        TOPHAT2_DENOVO(reads_ch, index, gtf, strandedness)
        bam_out = TOPHAT2_DENOVO.out.bam
        summary_out = TOPHAT2_DENOVO.out.summary
        junctions_out = TOPHAT2_DENOVO.out.junctions
    } else {
        TOPHAT2_ALIGN(reads_ch, index, gtf, strandedness)
        bam_out = TOPHAT2_ALIGN.out.bam
        summary_out = TOPHAT2_ALIGN.out.summary
        junctions_out = TOPHAT2_ALIGN.out.junctions
    }
    
    emit:
    bam = bam_out
    summary = summary_out
    junctions = junctions_out
}
