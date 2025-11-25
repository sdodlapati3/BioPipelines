/*
 * VarScan Module
 * 
 * VarScan - Variant detection in massively parallel sequencing data
 * Somatic and germline variant calling, CNV detection
 * Uses existing dna-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * VarScan mpileup2snp - Call SNPs from mpileup
 */
process VARSCAN_MPILEUP2SNP {
    tag "varscan_snp_${sample_id}"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/variants/varscan", mode: 'copy'
    
    memory params.varscan?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(bam), path(bai)
    path reference
    
    output:
    tuple val(sample_id), path("${sample_id}.snp.vcf"), emit: vcf
    
    script:
    def min_coverage = params.varscan?.min_coverage ?: 8
    def min_var_freq = params.varscan?.min_var_freq ?: 0.2
    def p_value = params.varscan?.p_value ?: 0.05
    
    """
    samtools mpileup -f ${reference} ${bam} | \\
    varscan mpileup2snp \\
        --min-coverage ${min_coverage} \\
        --min-var-freq ${min_var_freq} \\
        --p-value ${p_value} \\
        --output-vcf 1 \\
        > ${sample_id}.snp.vcf
    """
}

/*
 * VarScan mpileup2indel - Call indels from mpileup
 */
process VARSCAN_MPILEUP2INDEL {
    tag "varscan_indel_${sample_id}"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/variants/varscan", mode: 'copy'
    
    memory params.varscan?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(bam), path(bai)
    path reference
    
    output:
    tuple val(sample_id), path("${sample_id}.indel.vcf"), emit: vcf
    
    script:
    def min_coverage = params.varscan?.min_coverage ?: 8
    def min_var_freq = params.varscan?.min_var_freq ?: 0.2
    def p_value = params.varscan?.p_value ?: 0.05
    
    """
    samtools mpileup -f ${reference} ${bam} | \\
    varscan mpileup2indel \\
        --min-coverage ${min_coverage} \\
        --min-var-freq ${min_var_freq} \\
        --p-value ${p_value} \\
        --output-vcf 1 \\
        > ${sample_id}.indel.vcf
    """
}

/*
 * VarScan somatic - Call somatic variants
 */
process VARSCAN_SOMATIC {
    tag "varscan_somatic_${tumor_id}"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/variants/varscan/somatic", mode: 'copy'
    
    memory params.varscan?.memory ?: '16.GB'
    
    input:
    tuple val(tumor_id), path(tumor_bam), path(tumor_bai)
    tuple val(normal_id), path(normal_bam), path(normal_bai)
    path reference
    
    output:
    path "${tumor_id}.somatic.snp.vcf", emit: snp
    path "${tumor_id}.somatic.indel.vcf", emit: indel
    path "${tumor_id}.somatic.hc.vcf", emit: high_confidence
    
    script:
    def min_coverage = params.varscan?.min_coverage ?: 8
    def min_var_freq = params.varscan?.somatic_min_var_freq ?: 0.10
    def p_value = params.varscan?.p_value ?: 0.05
    
    """
    samtools mpileup -f ${reference} ${normal_bam} ${tumor_bam} | \\
    varscan somatic \\
        --mpileup 1 \\
        --min-coverage ${min_coverage} \\
        --min-var-freq ${min_var_freq} \\
        --p-value ${p_value} \\
        --output-vcf 1 \\
        --output-snp ${tumor_id}.somatic.snp.vcf \\
        --output-indel ${tumor_id}.somatic.indel.vcf
    
    # Filter for high confidence
    varscan processSomatic ${tumor_id}.somatic.snp.vcf
    mv ${tumor_id}.somatic.snp.Somatic.hc.vcf ${tumor_id}.somatic.hc.vcf
    """
}

/*
 * VarScan copynumber - CNV detection
 */
process VARSCAN_COPYNUMBER {
    tag "varscan_cnv_${tumor_id}"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/cnv/varscan", mode: 'copy'
    
    memory params.varscan?.memory ?: '16.GB'
    
    input:
    tuple val(tumor_id), path(tumor_bam), path(tumor_bai)
    tuple val(normal_id), path(normal_bam), path(normal_bai)
    path reference
    
    output:
    path "${tumor_id}.copynumber", emit: copynumber
    path "${tumor_id}.copynumber.called", emit: called
    
    script:
    """
    samtools mpileup -f ${reference} ${normal_bam} ${tumor_bam} | \\
    varscan copynumber \\
        --mpileup 1 \\
        --output-file ${tumor_id}.copynumber
    
    varscan copyCaller ${tumor_id}.copynumber \\
        --output-file ${tumor_id}.copynumber.called
    """
}

/*
 * Workflow: VarScan somatic variant calling
 */
workflow VARSCAN_PIPELINE {
    take:
    tumor_bam_ch     // channel: [ val(tumor_id), path(bam), path(bai) ]
    normal_bam_ch    // channel: [ val(normal_id), path(bam), path(bai) ]
    reference        // path: reference genome
    
    main:
    VARSCAN_SOMATIC(tumor_bam_ch, normal_bam_ch, reference)
    VARSCAN_COPYNUMBER(tumor_bam_ch, normal_bam_ch, reference)
    
    emit:
    somatic_snp = VARSCAN_SOMATIC.out.snp
    somatic_indel = VARSCAN_SOMATIC.out.indel
    high_confidence = VARSCAN_SOMATIC.out.high_confidence
    copynumber = VARSCAN_COPYNUMBER.out.called
}
