/*
 * Cell Ranger Module
 * 
 * Cell Ranger - 10x Genomics single-cell analysis
 * Complete pipeline for scRNA-seq, scATAC-seq, and multiome
 * Uses existing scrna-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * Cell Ranger count - scRNA-seq gene expression
 */
process CELLRANGER_COUNT {
    tag "cellranger_${sample_id}"
    container "${params.containers.scrnaseq}"
    
    publishDir "${params.outdir}/scrna/cellranger", mode: 'copy'
    
    cpus params.cellranger?.cpus ?: 32
    memory params.cellranger?.memory ?: '64.GB'
    
    input:
    tuple val(sample_id), path(fastq_dir)
    path transcriptome
    
    output:
    path "${sample_id}/outs", emit: outs_dir
    path "${sample_id}/outs/filtered_feature_bc_matrix", emit: filtered_matrix
    path "${sample_id}/outs/raw_feature_bc_matrix", emit: raw_matrix
    path "${sample_id}/outs/web_summary.html", emit: web_summary
    path "${sample_id}/outs/metrics_summary.csv", emit: metrics
    path "${sample_id}/outs/molecule_info.h5", emit: molecule_info
    path "${sample_id}/outs/possorted_genome_bam.bam", emit: bam
    
    script:
    def expect_cells = params.cellranger?.expect_cells ?: ""
    def force_cells = params.cellranger?.force_cells ?: ""
    def chemistry = params.cellranger?.chemistry ?: "auto"
    def expect_opt = expect_cells ? "--expect-cells=${expect_cells}" : ""
    def force_opt = force_cells ? "--force-cells=${force_cells}" : ""
    
    """
    cellranger count \\
        --id=${sample_id} \\
        --transcriptome=${transcriptome} \\
        --fastqs=${fastq_dir} \\
        --sample=${sample_id} \\
        --localcores=${task.cpus} \\
        --localmem=${task.memory.toGiga()} \\
        --chemistry=${chemistry} \\
        ${expect_opt} \\
        ${force_opt}
    """
}

/*
 * Cell Ranger aggr - Aggregate multiple samples
 */
process CELLRANGER_AGGR {
    tag "cellranger_aggr"
    container "${params.containers.scrnaseq}"
    
    publishDir "${params.outdir}/scrna/cellranger/aggregated", mode: 'copy'
    
    cpus params.cellranger?.cpus ?: 32
    memory params.cellranger?.memory ?: '64.GB'
    
    input:
    path molecule_info_files
    path aggr_csv
    
    output:
    path "aggregated/outs", emit: outs_dir
    path "aggregated/outs/filtered_feature_bc_matrix", emit: filtered_matrix
    path "aggregated/outs/web_summary.html", emit: web_summary
    path "aggregated/outs/aggregation.csv", emit: aggregation_csv
    
    script:
    def normalize = params.cellranger?.normalize ?: "mapped"
    
    """
    cellranger aggr \\
        --id=aggregated \\
        --csv=${aggr_csv} \\
        --normalize=${normalize} \\
        --localcores=${task.cpus} \\
        --localmem=${task.memory.toGiga()}
    """
}

/*
 * Cell Ranger reanalyze - Secondary analysis with custom parameters
 */
process CELLRANGER_REANALYZE {
    tag "cellranger_reanalyze_${sample_id}"
    container "${params.containers.scrnaseq}"
    
    publishDir "${params.outdir}/scrna/cellranger/reanalysis", mode: 'copy'
    
    cpus params.cellranger?.cpus ?: 16
    memory params.cellranger?.memory ?: '32.GB'
    
    input:
    tuple val(sample_id), path(filtered_matrix)
    path params_csv
    
    output:
    path "${sample_id}_reanalysis/outs", emit: outs_dir
    path "${sample_id}_reanalysis/outs/analysis", emit: analysis_dir
    
    script:
    """
    cellranger reanalyze \\
        --id=${sample_id}_reanalysis \\
        --matrix=${filtered_matrix} \\
        --params=${params_csv} \\
        --localcores=${task.cpus} \\
        --localmem=${task.memory.toGiga()}
    """
}

/*
 * Cell Ranger ATAC count - scATAC-seq
 */
process CELLRANGER_ATAC_COUNT {
    tag "cellranger_atac_${sample_id}"
    container "${params.containers.scrnaseq}"
    
    publishDir "${params.outdir}/scatac/cellranger", mode: 'copy'
    
    cpus params.cellranger?.cpus ?: 32
    memory params.cellranger?.memory ?: '64.GB'
    
    input:
    tuple val(sample_id), path(fastq_dir)
    path reference
    
    output:
    path "${sample_id}/outs", emit: outs_dir
    path "${sample_id}/outs/filtered_peak_bc_matrix", emit: filtered_matrix
    path "${sample_id}/outs/raw_peak_bc_matrix", emit: raw_matrix
    path "${sample_id}/outs/web_summary.html", emit: web_summary
    path "${sample_id}/outs/peaks.bed", emit: peaks
    path "${sample_id}/outs/fragments.tsv.gz", emit: fragments
    
    script:
    """
    cellranger-atac count \\
        --id=${sample_id} \\
        --reference=${reference} \\
        --fastqs=${fastq_dir} \\
        --sample=${sample_id} \\
        --localcores=${task.cpus} \\
        --localmem=${task.memory.toGiga()}
    """
}

/*
 * Workflow: Cell Ranger scRNA-seq pipeline
 */
workflow CELLRANGER_PIPELINE {
    take:
    fastq_ch       // channel: [ val(sample_id), path(fastq_dir) ]
    transcriptome  // path: Cell Ranger transcriptome reference
    
    main:
    CELLRANGER_COUNT(fastq_ch, transcriptome)
    
    emit:
    filtered_matrix = CELLRANGER_COUNT.out.filtered_matrix
    web_summary = CELLRANGER_COUNT.out.web_summary
    metrics = CELLRANGER_COUNT.out.metrics
    bam = CELLRANGER_COUNT.out.bam
}
