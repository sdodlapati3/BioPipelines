/*
 * RSeQC Module
 * 
 * RSeQC - RNA-seq Quality Control package
 * Comprehensive RNA-seq QC metrics
 * Uses existing rna-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * RSeQC read_distribution - Read distribution across genomic features
 */
process RSEQC_READDISTRIBUTION {
    tag "rseqc_readdist_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/qc/rseqc", mode: 'copy'
    
    memory params.rseqc?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(bam), path(bai)
    path bed
    
    output:
    path "${sample_id}_read_distribution.txt", emit: stats
    
    script:
    """
    read_distribution.py \\
        -i ${bam} \\
        -r ${bed} \\
        > ${sample_id}_read_distribution.txt
    """
}

/*
 * RSeQC gene_body_coverage - Gene body coverage
 */
process RSEQC_GENEBODYCOVERAGE {
    tag "rseqc_genebody_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/qc/rseqc", mode: 'copy'
    
    memory params.rseqc?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(bam), path(bai)
    path bed
    
    output:
    path "${sample_id}.geneBodyCoverage.*", emit: results
    
    script:
    """
    geneBody_coverage.py \\
        -i ${bam} \\
        -r ${bed} \\
        -o ${sample_id}
    """
}

/*
 * RSeQC inner_distance - Inner distance between paired reads
 */
process RSEQC_INNERDISTANCE {
    tag "rseqc_innerdist_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/qc/rseqc", mode: 'copy'
    
    memory params.rseqc?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(bam), path(bai)
    path bed
    
    output:
    path "${sample_id}.inner_distance.*", emit: results
    path "${sample_id}.inner_distance_freq.txt", emit: freq
    
    script:
    """
    inner_distance.py \\
        -i ${bam} \\
        -r ${bed} \\
        -o ${sample_id}
    """
}

/*
 * RSeQC infer_experiment - Infer strand specificity
 */
process RSEQC_INFEREXPERIMENT {
    tag "rseqc_infer_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/qc/rseqc", mode: 'copy'
    
    memory params.rseqc?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(bam), path(bai)
    path bed
    
    output:
    path "${sample_id}_infer_experiment.txt", emit: stats
    
    script:
    """
    infer_experiment.py \\
        -i ${bam} \\
        -r ${bed} \\
        > ${sample_id}_infer_experiment.txt
    """
}

/*
 * RSeQC junction_annotation - Annotate splice junctions
 */
process RSEQC_JUNCTIONANNOTATION {
    tag "rseqc_junction_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/qc/rseqc", mode: 'copy'
    
    memory params.rseqc?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(bam), path(bai)
    path bed
    
    output:
    path "${sample_id}.junction.*", emit: results
    path "${sample_id}.junction_annotation.txt", emit: stats
    
    script:
    """
    junction_annotation.py \\
        -i ${bam} \\
        -r ${bed} \\
        -o ${sample_id} \\
        > ${sample_id}.junction_annotation.txt
    """
}

/*
 * RSeQC read_duplication - Read duplication rate
 */
process RSEQC_READDUPLICATION {
    tag "rseqc_readdup_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/qc/rseqc", mode: 'copy'
    
    memory params.rseqc?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(bam), path(bai)
    
    output:
    path "${sample_id}.read_duplication.*", emit: results
    
    script:
    """
    read_duplication.py \\
        -i ${bam} \\
        -o ${sample_id}
    """
}

/*
 * RSeQC bam_stat - Basic BAM statistics
 */
process RSEQC_BAMSTAT {
    tag "rseqc_bamstat_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/qc/rseqc", mode: 'copy'
    
    memory params.rseqc?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(bam), path(bai)
    
    output:
    path "${sample_id}_bam_stat.txt", emit: stats
    
    script:
    """
    bam_stat.py \\
        -i ${bam} \\
        > ${sample_id}_bam_stat.txt
    """
}

/*
 * Workflow: Complete RSeQC QC pipeline
 */
workflow RSEQC_PIPELINE {
    take:
    bam_ch    // channel: [ val(sample_id), path(bam), path(bai) ]
    gene_bed  // path: gene model in BED format
    
    main:
    // Basic BAM statistics
    RSEQC_BAMSTAT(bam_ch)
    
    // Read distribution
    RSEQC_READDISTRIBUTION(bam_ch, gene_bed)
    
    // Gene body coverage
    RSEQC_GENEBODYCOVERAGE(bam_ch, gene_bed)
    
    // Infer experiment (strand specificity)
    RSEQC_INFEREXPERIMENT(bam_ch, gene_bed)
    
    // Junction annotation
    RSEQC_JUNCTIONANNOTATION(bam_ch, gene_bed)
    
    // Read duplication
    RSEQC_READDUPLICATION(bam_ch)
    
    // Inner distance (for paired-end)
    RSEQC_INNERDISTANCE(bam_ch, gene_bed)
    
    emit:
    bam_stats = RSEQC_BAMSTAT.out.stats
    read_dist = RSEQC_READDISTRIBUTION.out.stats
    gene_coverage = RSEQC_GENEBODYCOVERAGE.out.results
    strand_info = RSEQC_INFEREXPERIMENT.out.stats
    junctions = RSEQC_JUNCTIONANNOTATION.out.stats
    duplication = RSEQC_READDUPLICATION.out.results
    inner_dist = RSEQC_INNERDISTANCE.out.freq
}
