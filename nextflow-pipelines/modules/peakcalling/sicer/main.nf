/*
 * SICER Module
 * 
 * SICER - Spatial clustering for identification of ChIP-enriched regions
 * Specialized for broad histone marks
 * Uses existing chip-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * SICER broad peak calling
 */
process SICER_CALL {
    tag "sicer_${sample_id}"
    container "${params.containers.chipseq}"
    
    publishDir "${params.outdir}/peaks/sicer", mode: 'copy'
    
    cpus params.sicer?.cpus ?: 4
    memory params.sicer?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(treatment_bed)
    path control_bed
    val genome
    val fragment_size
    val window_size
    val gap_size
    
    output:
    tuple val(sample_id), path("${sample_id}-W${window_size}-G${gap_size}-islands-summary"), emit: peaks
    tuple val(sample_id), path("${sample_id}-W${window_size}-G${gap_size}-islands-summary-FDR*"), emit: filtered_peaks
    path "${sample_id}*.scoreisland", emit: score_islands
    
    script:
    def fdr = params.sicer?.fdr ?: 0.01
    
    """
    # SICER requires BED files in specific directory structure
    mkdir -p input_dir
    cp ${treatment_bed} input_dir/${sample_id}.bed
    cp ${control_bed} input_dir/control.bed
    
    # Run SICER
    sicer \\
        -t input_dir/${sample_id}.bed \\
        -c input_dir/control.bed \\
        -s ${genome} \\
        -w ${window_size} \\
        -g ${gap_size} \\
        -f ${fragment_size} \\
        -fdr ${fdr} \\
        -o ./
    
    # Move output files
    mv *-W${window_size}-G${gap_size}* .
    """
}

/*
 * SICER differential analysis
 */
process SICER_DIFF {
    tag "sicer_diff_${condition1}_vs_${condition2}"
    container "${params.containers.chipseq}"
    
    publishDir "${params.outdir}/peaks/sicer/differential", mode: 'copy'
    
    cpus params.sicer?.cpus ?: 4
    memory params.sicer?.memory ?: '16.GB'
    
    input:
    tuple val(condition1), path(bed1)
    tuple val(condition2), path(bed2)
    val genome
    val fragment_size
    val window_size
    val gap_size
    
    output:
    path "${condition1}_vs_${condition2}-W${window_size}-G${gap_size}-increased-islands-summary", emit: increased_peaks
    path "${condition1}_vs_${condition2}-W${window_size}-G${gap_size}-decreased-islands-summary", emit: decreased_peaks
    
    script:
    """
    mkdir -p input_dir
    cp ${bed1} input_dir/${condition1}.bed
    cp ${bed2} input_dir/${condition2}.bed
    
    # Run SICER differential
    sicer_df \\
        -t1 input_dir/${condition1}.bed \\
        -t2 input_dir/${condition2}.bed \\
        -s ${genome} \\
        -w ${window_size} \\
        -g ${gap_size} \\
        -f ${fragment_size} \\
        -o ./
    
    # Rename outputs
    mv *increased-islands-summary ${condition1}_vs_${condition2}-W${window_size}-G${gap_size}-increased-islands-summary
    mv *decreased-islands-summary ${condition1}_vs_${condition2}-W${window_size}-G${gap_size}-decreased-islands-summary
    """
}

/*
 * Workflow: SICER broad peak calling
 */
workflow SICER_PIPELINE {
    take:
    treatment_ch   // channel: [ val(sample_id), path(bed) ]
    control_bed    // path: control BED file
    genome         // val: genome identifier
    fragment_size  // val: fragment size
    window_size    // val: window size
    gap_size       // val: gap size
    
    main:
    SICER_CALL(treatment_ch, control_bed, genome, fragment_size, window_size, gap_size)
    
    emit:
    peaks = SICER_CALL.out.peaks
    filtered_peaks = SICER_CALL.out.filtered_peaks
}
