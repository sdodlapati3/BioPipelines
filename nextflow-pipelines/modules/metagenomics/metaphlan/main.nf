/*
 * MetaPhlAn Module
 * 
 * MetaPhlAn - Metagenomic Phylogenetic Analysis
 * Profiling composition of microbial communities
 * Uses existing metagenomics container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * MetaPhlAn profiling
 */
process METAPHLAN_PROFILE {
    tag "metaphlan_${sample_id}"
    container "${params.containers.metagenomics}"
    
    publishDir "${params.outdir}/metagenomics/metaphlan", mode: 'copy'
    
    cpus params.metaphlan?.cpus ?: 8
    memory params.metaphlan?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(reads)
    path metaphlan_db
    
    output:
    tuple val(sample_id), path("${sample_id}_profile.txt"), emit: profile
    tuple val(sample_id), path("${sample_id}.bowtie2.bz2"), emit: bowtie2_output
    
    script:
    def input_type = params.metaphlan?.input_type ?: "fastq"
    
    if (reads instanceof List) {
        """
        metaphlan \\
            ${reads[0]},${reads[1]} \\
            --input_type ${input_type} \\
            --nproc ${task.cpus} \\
            --bowtie2db ${metaphlan_db} \\
            --bowtie2out ${sample_id}.bowtie2.bz2 \\
            -o ${sample_id}_profile.txt
        """
    } else {
        """
        metaphlan \\
            ${reads} \\
            --input_type ${input_type} \\
            --nproc ${task.cpus} \\
            --bowtie2db ${metaphlan_db} \\
            --bowtie2out ${sample_id}.bowtie2.bz2 \\
            -o ${sample_id}_profile.txt
        """
    }
}

/*
 * MetaPhlAn merge profiles
 */
process METAPHLAN_MERGE {
    tag "metaphlan_merge"
    container "${params.containers.metagenomics}"
    
    publishDir "${params.outdir}/metagenomics/metaphlan", mode: 'copy'
    
    input:
    path profiles
    
    output:
    path "merged_abundance_table.txt", emit: merged_table
    
    script:
    def profile_list = profiles.collect { it.toString() }.join(' ')
    
    """
    merge_metaphlan_tables.py \\
        ${profile_list} \\
        > merged_abundance_table.txt
    """
}

/*
 * Workflow: MetaPhlAn profiling pipeline
 */
workflow METAPHLAN_PIPELINE {
    take:
    reads_ch      // channel: [ val(sample_id), path(reads) ]
    metaphlan_db  // path: MetaPhlAn database
    
    main:
    METAPHLAN_PROFILE(reads_ch, metaphlan_db)
    
    all_profiles = METAPHLAN_PROFILE.out.profile.map { it[1] }.collect()
    METAPHLAN_MERGE(all_profiles)
    
    emit:
    profiles = METAPHLAN_PROFILE.out.profile
    merged_table = METAPHLAN_MERGE.out.merged_table
}
