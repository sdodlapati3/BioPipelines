/*
 * HiC-Pro Module
 * 
 * HiC-Pro - Hi-C data processing pipeline
 * Mapping, quality control, and contact matrix generation
 * Uses existing hic container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * HiC-Pro full pipeline
 */
process HICPRO_PROCESS {
    tag "hicpro_${sample_id}"
    container "${params.containers.hic}"
    
    publishDir "${params.outdir}/hic/hicpro", mode: 'copy'
    
    cpus params.hicpro?.cpus ?: 32
    memory params.hicpro?.memory ?: '64.GB'
    
    input:
    tuple val(sample_id), path(read1), path(read2)
    path config_file
    path genome_index
    
    output:
    tuple val(sample_id), path("${sample_id}/hic_results/data/*/*.validPairs"), emit: valid_pairs
    tuple val(sample_id), path("${sample_id}/hic_results/matrix/*/*.matrix"), emit: matrices
    path "${sample_id}/hic_results/stats", emit: stats
    path "${sample_id}/hic_results/pic", emit: plots
    
    script:
    """
    # Create input directory structure
    mkdir -p ${sample_id}_input
    ln -s ${read1} ${sample_id}_input/${sample_id}_R1.fastq.gz
    ln -s ${read2} ${sample_id}_input/${sample_id}_R2.fastq.gz
    
    # Run HiC-Pro
    HiC-Pro \\
        -i ${sample_id}_input \\
        -o ${sample_id} \\
        -c ${config_file} \\
        -p
    """
}

/*
 * HiC-Pro matrix building only
 */
process HICPRO_MATRIX {
    tag "hicpro_matrix_${sample_id}"
    container "${params.containers.hic}"
    
    publishDir "${params.outdir}/hic/hicpro/matrices", mode: 'copy'
    
    cpus params.hicpro?.cpus ?: 8
    memory params.hicpro?.memory ?: '32.GB'
    
    input:
    tuple val(sample_id), path(valid_pairs)
    path config_file
    val bin_sizes
    
    output:
    tuple val(sample_id), path("${sample_id}_matrices/*"), emit: matrices
    
    script:
    def bins = bin_sizes instanceof List ? bin_sizes.join(',') : bin_sizes
    
    """
    # Build matrices at different resolutions
    build_matrix \\
        --matrix-format upper \\
        --binsize ${bins} \\
        --chrsizes ${config_file} \\
        --ifile ${valid_pairs} \\
        --oprefix ${sample_id}
    
    mkdir -p ${sample_id}_matrices
    mv ${sample_id}*.matrix ${sample_id}_matrices/
    """
}

/*
 * HiC-Pro quality control
 */
process HICPRO_QC {
    tag "hicpro_qc_${sample_id}"
    container "${params.containers.hic}"
    
    publishDir "${params.outdir}/hic/hicpro/qc", mode: 'copy'
    
    input:
    tuple val(sample_id), path(stats_dir)
    
    output:
    path "${sample_id}_qc_report.html", emit: report
    path "${sample_id}_plots/*.pdf", emit: plots
    
    script:
    """
    # Generate QC plots
    mkdir -p ${sample_id}_plots
    
    # Create summary report
    make_viewpoints \\
        -i ${stats_dir} \\
        -o ${sample_id}_qc_report.html
    """
}

/*
 * Workflow: HiC-Pro pipeline
 */
workflow HICPRO_PIPELINE {
    take:
    reads_ch      // channel: [ val(sample_id), path(read1), path(read2) ]
    config_file   // path: HiC-Pro configuration
    genome_index  // path: genome index
    bin_sizes     // val: bin sizes for matrices
    
    main:
    HICPRO_PROCESS(reads_ch, config_file, genome_index)
    HICPRO_MATRIX(HICPRO_PROCESS.out.valid_pairs, config_file, bin_sizes)
    HICPRO_QC(HICPRO_PROCESS.out.stats)
    
    emit:
    valid_pairs = HICPRO_PROCESS.out.valid_pairs
    matrices = HICPRO_MATRIX.out.matrices
    qc_report = HICPRO_QC.out.report
}
