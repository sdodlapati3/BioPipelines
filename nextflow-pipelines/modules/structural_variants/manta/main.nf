/*
 * Manta Module
 * 
 * Manta - Structural variant and indel caller
 * Rapid discovery and genotyping of structural variants
 * Uses existing structural-variants container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * Manta germline SV calling
 */
process MANTA_GERMLINE {
    tag "manta_germline_${sample_id}"
    container "${params.containers.structuralvariants}"
    
    publishDir "${params.outdir}/sv/manta", mode: 'copy'
    
    cpus params.manta?.cpus ?: 16
    memory params.manta?.memory ?: '32.GB'
    
    input:
    tuple val(sample_id), path(bam), path(bai)
    path reference
    path reference_fai
    path target_bed
    
    output:
    tuple val(sample_id), path("${sample_id}/results/variants/diploidSV.vcf.gz"), emit: vcf
    tuple val(sample_id), path("${sample_id}/results/variants/diploidSV.vcf.gz.tbi"), emit: vcf_index
    path "${sample_id}/results/variants/candidateSV.vcf.gz", emit: candidate_vcf
    path "${sample_id}/results/stats", emit: stats
    
    script:
    def bed_opt = target_bed ? "--callRegions ${target_bed}" : ""
    
    """
    # Configure Manta
    configManta.py \\
        --bam ${bam} \\
        --referenceFasta ${reference} \\
        ${bed_opt} \\
        --runDir ${sample_id}
    
    # Run Manta
    ${sample_id}/runWorkflow.py \\
        -m local \\
        -j ${task.cpus}
    """
}

/*
 * Manta somatic SV calling
 */
process MANTA_SOMATIC {
    tag "manta_somatic_${tumor_id}_${normal_id}"
    container "${params.containers.structuralvariants}"
    
    publishDir "${params.outdir}/sv/manta", mode: 'copy'
    
    cpus params.manta?.cpus ?: 16
    memory params.manta?.memory ?: '32.GB'
    
    input:
    tuple val(tumor_id), path(tumor_bam), path(tumor_bai), val(normal_id), path(normal_bam), path(normal_bai)
    path reference
    path reference_fai
    path target_bed
    
    output:
    tuple val(tumor_id), path("${tumor_id}_vs_${normal_id}/results/variants/somaticSV.vcf.gz"), emit: somatic_vcf
    tuple val(tumor_id), path("${tumor_id}_vs_${normal_id}/results/variants/diploidSV.vcf.gz"), emit: diploid_vcf
    path "${tumor_id}_vs_${normal_id}/results/variants/candidateSV.vcf.gz", emit: candidate_vcf
    path "${tumor_id}_vs_${normal_id}/results/stats", emit: stats
    
    script:
    def bed_opt = target_bed ? "--callRegions ${target_bed}" : ""
    
    """
    # Configure Manta
    configManta.py \\
        --tumorBam ${tumor_bam} \\
        --normalBam ${normal_bam} \\
        --referenceFasta ${reference} \\
        ${bed_opt} \\
        --runDir ${tumor_id}_vs_${normal_id}
    
    # Run Manta
    ${tumor_id}_vs_${normal_id}/runWorkflow.py \\
        -m local \\
        -j ${task.cpus}
    """
}

/*
 * Workflow: Manta SV calling
 */
workflow MANTA_PIPELINE {
    take:
    bam_ch        // channel: [ val(sample_id), path(bam), path(bai) ]
    reference     // path: reference genome
    reference_fai // path: reference index
    target_bed    // path: target regions (optional)
    mode          // val: 'germline' or 'somatic'
    
    main:
    if (mode == 'germline') {
        MANTA_GERMLINE(bam_ch, reference, reference_fai, target_bed)
        vcf_out = MANTA_GERMLINE.out.vcf
        stats_out = MANTA_GERMLINE.out.stats
    } else {
        MANTA_SOMATIC(bam_ch, reference, reference_fai, target_bed)
        vcf_out = MANTA_SOMATIC.out.somatic_vcf
        stats_out = MANTA_SOMATIC.out.stats
    }
    
    emit:
    vcf = vcf_out
    stats = stats_out
}
