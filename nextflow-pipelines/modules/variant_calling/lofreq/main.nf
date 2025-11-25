/*
 * LoFreq Module
 * 
 * LoFreq - Fast and sensitive variant calling from high-throughput sequencing data
 * Low-frequency variant detection (down to 0.5%)
 * Uses existing dna-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * LoFreq call - Call variants
 */
process LOFREQ_CALL {
    tag "lofreq_${sample_id}"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/variants/lofreq", mode: 'copy'
    
    cpus params.lofreq?.cpus ?: 4
    memory params.lofreq?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(bam), path(bai)
    path reference
    
    output:
    tuple val(sample_id), path("${sample_id}.vcf.gz"), emit: vcf
    
    script:
    """
    lofreq call-parallel \\
        --pp-threads ${task.cpus} \\
        -f ${reference} \\
        -o ${sample_id}.vcf.gz \\
        ${bam}
    """
}

/*
 * LoFreq somatic - Call somatic variants
 */
process LOFREQ_SOMATIC {
    tag "lofreq_somatic_${tumor_id}"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/variants/lofreq/somatic", mode: 'copy'
    
    cpus params.lofreq?.cpus ?: 8
    memory params.lofreq?.memory ?: '16.GB'
    
    input:
    tuple val(tumor_id), path(tumor_bam), path(tumor_bai)
    tuple val(normal_id), path(normal_bam), path(normal_bai)
    path reference
    
    output:
    path "${tumor_id}_somatic_final.snvs.vcf.gz", emit: snvs
    path "${tumor_id}_somatic_final.indels.vcf.gz", emit: indels
    
    script:
    """
    lofreq somatic \\
        --threads ${task.cpus} \\
        -n ${normal_bam} \\
        -t ${tumor_bam} \\
        -f ${reference} \\
        -o ${tumor_id}_
    """
}

/*
 * LoFreq indelqual - Add indel qualities
 */
process LOFREQ_INDELQUAL {
    tag "lofreq_indelqual_${sample_id}"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/alignments/lofreq", mode: 'copy'
    
    memory params.lofreq?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(bam), path(bai)
    path reference
    
    output:
    tuple val(sample_id), path("${sample_id}.indelqual.bam"), path("${sample_id}.indelqual.bam.bai"), emit: bam
    
    script:
    """
    lofreq indelqual \\
        --dindel \\
        -f ${reference} \\
        -o ${sample_id}.indelqual.bam \\
        ${bam}
    
    samtools index ${sample_id}.indelqual.bam
    """
}

/*
 * Workflow: LoFreq variant calling
 */
workflow LOFREQ_PIPELINE {
    take:
    bam_ch       // channel: [ val(sample_id), path(bam), path(bai) ]
    reference    // path: reference genome
    
    main:
    LOFREQ_INDELQUAL(bam_ch, reference)
    LOFREQ_CALL(LOFREQ_INDELQUAL.out.bam, reference)
    
    emit:
    vcf = LOFREQ_CALL.out.vcf
}
