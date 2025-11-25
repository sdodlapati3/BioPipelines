/*
 * GATK Module
 * 
 * GATK (Genome Analysis Toolkit)
 * Variant discovery in high-throughput sequencing data
 * Uses existing dna-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * GATK HaplotypeCaller - Call germline SNPs and indels
 */
process GATK_HAPLOTYPECALLER {
    tag "gatk_hc_${sample_id}"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/variants/gatk", mode: 'copy'
    
    cpus params.gatk?.cpus ?: 4
    memory params.gatk?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(bam), path(bai)
    path reference
    path reference_index
    path reference_dict
    path intervals
    
    output:
    tuple val(sample_id), path("${sample_id}.g.vcf.gz"), path("${sample_id}.g.vcf.gz.tbi"), emit: gvcf
    
    script:
    def intervals_opt = intervals ? "-L ${intervals}" : ""
    
    """
    gatk HaplotypeCaller \\
        -R ${reference} \\
        -I ${bam} \\
        -O ${sample_id}.g.vcf.gz \\
        -ERC GVCF \\
        ${intervals_opt} \\
        --native-pair-hmm-threads ${task.cpus}
    """
}

/*
 * GATK GenotypeGVCFs - Joint genotyping
 */
process GATK_GENOTYPEGVCFS {
    tag "gatk_genotype"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/variants/gatk", mode: 'copy'
    
    cpus params.gatk?.cpus ?: 4
    memory params.gatk?.memory ?: '32.GB'
    
    input:
    path gvcf_files
    path reference
    path reference_index
    path reference_dict
    
    output:
    path "cohort.vcf.gz", emit: vcf
    path "cohort.vcf.gz.tbi", emit: vcf_index
    
    script:
    def gvcf_args = gvcf_files.collect { "-V ${it}" }.join(' ')
    
    """
    gatk GenotypeGVCFs \\
        -R ${reference} \\
        ${gvcf_args} \\
        -O cohort.vcf.gz
    """
}

/*
 * GATK MarkDuplicates - Mark duplicate reads
 */
process GATK_MARKDUPLICATES {
    tag "gatk_markdup_${sample_id}"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/alignments/marked", mode: 'copy'
    
    cpus params.gatk?.cpus ?: 4
    memory params.gatk?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(bam)
    
    output:
    tuple val(sample_id), path("${sample_id}.marked.bam"), path("${sample_id}.marked.bam.bai"), emit: bam
    path "${sample_id}.metrics.txt", emit: metrics
    
    script:
    """
    gatk MarkDuplicates \\
        -I ${bam} \\
        -O ${sample_id}.marked.bam \\
        -M ${sample_id}.metrics.txt \\
        --CREATE_INDEX true
    """
}

/*
 * GATK BaseRecalibrator - Calculate base quality score recalibration
 */
process GATK_BASERECALIBRATOR {
    tag "gatk_bqsr_${sample_id}"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/recal", mode: 'copy'
    
    cpus params.gatk?.cpus ?: 4
    memory params.gatk?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(bam), path(bai)
    path reference
    path reference_index
    path reference_dict
    path known_sites
    path known_sites_index
    
    output:
    tuple val(sample_id), path("${sample_id}.recal_data.table"), emit: table
    
    script:
    def known_sites_args = known_sites.collect { "--known-sites ${it}" }.join(' ')
    
    """
    gatk BaseRecalibrator \\
        -R ${reference} \\
        -I ${bam} \\
        -O ${sample_id}.recal_data.table \\
        ${known_sites_args}
    """
}

/*
 * GATK ApplyBQSR - Apply base quality score recalibration
 */
process GATK_APPLYBQSR {
    tag "gatk_applybqsr_${sample_id}"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/alignments/recalibrated", mode: 'copy'
    
    cpus params.gatk?.cpus ?: 4
    memory params.gatk?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(bam), path(bai), path(recal_table)
    path reference
    path reference_index
    path reference_dict
    
    output:
    tuple val(sample_id), path("${sample_id}.recal.bam"), path("${sample_id}.recal.bam.bai"), emit: bam
    
    script:
    """
    gatk ApplyBQSR \\
        -R ${reference} \\
        -I ${bam} \\
        --bqsr-recal-file ${recal_table} \\
        -O ${sample_id}.recal.bam \\
        --create-output-bam-index true
    """
}

/*
 * GATK VariantFiltration - Filter variants
 */
process GATK_VARIANTFILTRATION {
    tag "gatk_filter"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/variants/gatk/filtered", mode: 'copy'
    
    cpus params.gatk?.cpus ?: 2
    memory params.gatk?.memory ?: '8.GB'
    
    input:
    path vcf
    path reference
    path reference_index
    path reference_dict
    
    output:
    path "filtered.vcf.gz", emit: vcf
    path "filtered.vcf.gz.tbi", emit: vcf_index
    
    script:
    """
    gatk VariantFiltration \\
        -R ${reference} \\
        -V ${vcf} \\
        -O filtered.vcf.gz \\
        --filter-expression "QD < 2.0" --filter-name "QD2" \\
        --filter-expression "FS > 60.0" --filter-name "FS60" \\
        --filter-expression "MQ < 40.0" --filter-name "MQ40" \\
        --filter-expression "MQRankSum < -12.5" --filter-name "MQRankSum-12.5" \\
        --filter-expression "ReadPosRankSum < -8.0" --filter-name "ReadPosRankSum-8"
    """
}

/*
 * GATK SelectVariants - Select SNPs or INDELs
 */
process GATK_SELECTVARIANTS {
    tag "gatk_select_${variant_type}"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/variants/gatk/selected", mode: 'copy'
    
    cpus params.gatk?.cpus ?: 2
    memory params.gatk?.memory ?: '8.GB'
    
    input:
    path vcf
    path reference
    path reference_index
    path reference_dict
    val variant_type  // "SNP" or "INDEL"
    
    output:
    path "${variant_type.toLowerCase()}.vcf.gz", emit: vcf
    path "${variant_type.toLowerCase()}.vcf.gz.tbi", emit: vcf_index
    
    script:
    """
    gatk SelectVariants \\
        -R ${reference} \\
        -V ${vcf} \\
        -O ${variant_type.toLowerCase()}.vcf.gz \\
        --select-type-to-include ${variant_type}
    """
}

/*
 * GATK VariantEval - Evaluate variants
 */
process GATK_VARIANTEVAL {
    tag "gatk_eval"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/variants/gatk/evaluation", mode: 'copy'
    
    cpus params.gatk?.cpus ?: 2
    memory params.gatk?.memory ?: '8.GB'
    
    input:
    path vcf
    path reference
    path reference_index
    path reference_dict
    path dbsnp
    path dbsnp_index
    
    output:
    path "variant_eval.table", emit: eval_table
    
    script:
    def dbsnp_opt = dbsnp ? "--dbsnp ${dbsnp}" : ""
    
    """
    gatk VariantEval \\
        -R ${reference} \\
        -eval ${vcf} \\
        ${dbsnp_opt} \\
        -O variant_eval.table
    """
}

/*
 * Workflow: Complete GATK variant calling pipeline
 */
workflow GATK_PIPELINE {
    take:
    bam_ch            // channel: [ val(sample_id), path(bam), path(bai) ]
    reference         // path: reference genome
    reference_index   // path: reference .fai
    reference_dict    // path: reference .dict
    known_sites       // path: known variants for BQSR
    known_sites_index // path: known variants index
    intervals         // path: optional intervals file
    
    main:
    // Mark duplicates
    GATK_MARKDUPLICATES(bam_ch.map { [it[0], it[1]] })
    
    // Base quality score recalibration
    GATK_BASERECALIBRATOR(
        GATK_MARKDUPLICATES.out.bam,
        reference,
        reference_index,
        reference_dict,
        known_sites,
        known_sites_index
    )
    
    // Apply BQSR
    bqsr_input = GATK_MARKDUPLICATES.out.bam.join(GATK_BASERECALIBRATOR.out.table)
    GATK_APPLYBQSR(
        bqsr_input,
        reference,
        reference_index,
        reference_dict
    )
    
    // Call variants
    GATK_HAPLOTYPECALLER(
        GATK_APPLYBQSR.out.bam,
        reference,
        reference_index,
        reference_dict,
        intervals
    )
    
    // Joint genotyping
    all_gvcfs = GATK_HAPLOTYPECALLER.out.gvcf.map { it[1..2] }.collect()
    GATK_GENOTYPEGVCFS(
        all_gvcfs,
        reference,
        reference_index,
        reference_dict
    )
    
    // Filter variants
    GATK_VARIANTFILTRATION(
        GATK_GENOTYPEGVCFS.out.vcf,
        reference,
        reference_index,
        reference_dict
    )
    
    emit:
    vcf = GATK_GENOTYPEGVCFS.out.vcf
    filtered_vcf = GATK_VARIANTFILTRATION.out.vcf
    recalibrated_bams = GATK_APPLYBQSR.out.bam
}
