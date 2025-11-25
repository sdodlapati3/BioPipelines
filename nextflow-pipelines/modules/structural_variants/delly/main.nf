/*
 * DELLY Module
 * 
 * DELLY - Integrated structural variant prediction
 * Discovery and genotyping of deletions, duplications, inversions, translocations
 * Uses existing structural-variants container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * DELLY call SVs
 */
process DELLY_CALL {
    tag "delly_${sample_id}"
    container "${params.containers.structuralvariants}"
    
    publishDir "${params.outdir}/sv/delly", mode: 'copy'
    
    cpus params.delly?.cpus ?: 4
    memory params.delly?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(bam), path(bai)
    path reference
    path exclude_regions
    
    output:
    tuple val(sample_id), path("${sample_id}.bcf"), emit: bcf
    tuple val(sample_id), path("${sample_id}.vcf.gz"), emit: vcf
    
    script:
    def sv_type = params.delly?.sv_type ?: "ALL"
    def exclude_opt = exclude_regions ? "-x ${exclude_regions}" : ""
    
    """
    # Call SVs
    delly call \\
        -t ${sv_type} \\
        -g ${reference} \\
        ${exclude_opt} \\
        -o ${sample_id}.bcf \\
        ${bam}
    
    # Convert to VCF
    bcftools view ${sample_id}.bcf | bgzip > ${sample_id}.vcf.gz
    tabix -p vcf ${sample_id}.vcf.gz
    """
}

/*
 * DELLY filter
 */
process DELLY_FILTER {
    tag "delly_filter_${sample_id}"
    container "${params.containers.structuralvariants}"
    
    publishDir "${params.outdir}/sv/delly", mode: 'copy'
    
    memory params.delly?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(bcf)
    val filter_type
    
    output:
    tuple val(sample_id), path("${sample_id}.filtered.bcf"), emit: bcf
    tuple val(sample_id), path("${sample_id}.filtered.vcf.gz"), emit: vcf
    
    script:
    def filter = filter_type == "somatic" ? "somatic" : "germline"
    
    """
    # Filter SVs
    delly filter \\
        -f ${filter} \\
        -o ${sample_id}.filtered.bcf \\
        ${bcf}
    
    # Convert to VCF
    bcftools view ${sample_id}.filtered.bcf | bgzip > ${sample_id}.filtered.vcf.gz
    tabix -p vcf ${sample_id}.filtered.vcf.gz
    """
}

/*
 * DELLY somatic mode
 */
process DELLY_SOMATIC {
    tag "delly_somatic_${tumor_id}_${normal_id}"
    container "${params.containers.structuralvariants}"
    
    publishDir "${params.outdir}/sv/delly", mode: 'copy'
    
    cpus params.delly?.cpus ?: 4
    memory params.delly?.memory ?: '16.GB'
    
    input:
    tuple val(tumor_id), path(tumor_bam), path(tumor_bai), val(normal_id), path(normal_bam), path(normal_bai)
    path reference
    path exclude_regions
    
    output:
    tuple val(tumor_id), path("${tumor_id}_vs_${normal_id}.bcf"), emit: bcf
    tuple val(tumor_id), path("${tumor_id}_vs_${normal_id}.somatic.vcf.gz"), emit: somatic_vcf
    
    script:
    def sv_type = params.delly?.sv_type ?: "ALL"
    def exclude_opt = exclude_regions ? "-x ${exclude_regions}" : ""
    
    """
    # Create sample file
    echo -e "${tumor_id}\\ttumor" > samples.tsv
    echo -e "${normal_id}\\tcontrol" >> samples.tsv
    
    # Call SVs
    delly call \\
        -t ${sv_type} \\
        -g ${reference} \\
        ${exclude_opt} \\
        -o ${tumor_id}_vs_${normal_id}.bcf \\
        ${tumor_bam} ${normal_bam}
    
    # Filter somatic
    delly filter \\
        -f somatic \\
        -s samples.tsv \\
        -o ${tumor_id}_vs_${normal_id}.filtered.bcf \\
        ${tumor_id}_vs_${normal_id}.bcf
    
    # Convert to VCF
    bcftools view ${tumor_id}_vs_${normal_id}.filtered.bcf | bgzip > ${tumor_id}_vs_${normal_id}.somatic.vcf.gz
    tabix -p vcf ${tumor_id}_vs_${normal_id}.somatic.vcf.gz
    """
}

/*
 * Workflow: DELLY SV calling
 */
workflow DELLY_PIPELINE {
    take:
    bam_ch          // channel: [ val(sample_id), path(bam), path(bai) ]
    reference       // path: reference genome
    exclude_regions // path: exclude regions (optional)
    mode            // val: 'germline' or 'somatic'
    
    main:
    if (mode == 'germline') {
        DELLY_CALL(bam_ch, reference, exclude_regions)
        DELLY_FILTER(DELLY_CALL.out.bcf, 'germline')
        vcf_out = DELLY_FILTER.out.vcf
    } else {
        DELLY_SOMATIC(bam_ch, reference, exclude_regions)
        vcf_out = DELLY_SOMATIC.out.somatic_vcf
    }
    
    emit:
    vcf = vcf_out
}
