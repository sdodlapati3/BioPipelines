/*
 * Qualimap Module
 * 
 * Qualimap - Quality control of alignment sequencing data
 * Comprehensive BAM quality metrics
 * Uses existing rna-seq or dna-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * Qualimap bamqc - BAM quality control
 */
process QUALIMAP_BAMQC {
    tag "qualimap_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/qc/qualimap", mode: 'copy'
    
    cpus params.qualimap?.cpus ?: 8
    memory params.qualimap?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(bam)
    path gtf
    
    output:
    path "${sample_id}_qualimap/*", emit: results
    path "${sample_id}_qualimap/qualimapReport.html", emit: report
    path "${sample_id}_qualimap/genome_results.txt", emit: stats
    
    script:
    def gtf_opt = gtf ? "-gff ${gtf}" : ""
    def java_mem = task.memory ? "-Xmx${task.memory.toGiga()}G" : "-Xmx16G"
    
    """
    qualimap bamqc \\
        -bam ${bam} \\
        ${gtf_opt} \\
        -outdir ${sample_id}_qualimap \\
        -nt ${task.cpus} \\
        --java-mem-size=${java_mem}
    """
}

/*
 * Qualimap rnaseq - RNA-seq specific QC
 */
process QUALIMAP_RNASEQ {
    tag "qualimap_rnaseq_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/qc/qualimap", mode: 'copy'
    
    cpus params.qualimap?.cpus ?: 8
    memory params.qualimap?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(bam)
    path gtf
    
    output:
    path "${sample_id}_qualimap_rnaseq/*", emit: results
    path "${sample_id}_qualimap_rnaseq/qualimapReport.html", emit: report
    path "${sample_id}_qualimap_rnaseq/rnaseq_qc_results.txt", emit: stats
    
    script:
    def java_mem = task.memory ? "-Xmx${task.memory.toGiga()}G" : "-Xmx16G"
    
    """
    qualimap rnaseq \\
        -bam ${bam} \\
        -gtf ${gtf} \\
        -outdir ${sample_id}_qualimap_rnaseq \\
        --java-mem-size=${java_mem}
    """
}

/*
 * Qualimap multi-bamqc - Compare multiple samples
 */
process QUALIMAP_MULTIBAMQC {
    tag "qualimap_multi"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/qc/qualimap", mode: 'copy'
    
    cpus params.qualimap?.cpus ?: 8
    memory params.qualimap?.memory ?: '32.GB'
    
    input:
    path bam_files
    path gtf
    
    output:
    path "multisample_qualimap/*", emit: results
    path "multisample_qualimap/qualimapReport.html", emit: report
    
    script:
    def java_mem = task.memory ? "-Xmx${task.memory.toGiga()}G" : "-Xmx32G"
    def gtf_opt = gtf ? "-gff ${gtf}" : ""
    
    """
    # Create input list file
    echo "${bam_files.collect { it.toString() }.join('\\n')}" > bam_list.txt
    
    qualimap multi-bamqc \\
        -d bam_list.txt \\
        ${gtf_opt} \\
        -outdir multisample_qualimap \\
        --java-mem-size=${java_mem}
    """
}

/*
 * Workflow: Qualimap QC pipeline
 */
workflow QUALIMAP_PIPELINE {
    take:
    bam_ch     // channel: [ val(sample_id), path(bam) ]
    gtf        // path: annotation GTF
    analysis_type // val: "bam" or "rnaseq"
    
    main:
    if (analysis_type == "rnaseq") {
        QUALIMAP_RNASEQ(bam_ch, gtf)
        results_out = QUALIMAP_RNASEQ.out.results
        report_out = QUALIMAP_RNASEQ.out.report
        stats_out = QUALIMAP_RNASEQ.out.stats
    } else {
        QUALIMAP_BAMQC(bam_ch, gtf)
        results_out = QUALIMAP_BAMQC.out.results
        report_out = QUALIMAP_BAMQC.out.report
        stats_out = QUALIMAP_BAMQC.out.stats
    }
    
    emit:
    results = results_out
    report = report_out
    stats = stats_out
}
