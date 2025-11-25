/*
 * Kallisto Module
 * 
 * Kallisto - Fast transcript quantification
 * Pseudoalignment-based RNA-seq quantification
 * Uses existing rna-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * Kallisto Index - Build transcriptome index
 */
process KALLISTO_INDEX {
    tag "kallisto_index_${transcriptome.baseName}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/reference/kallisto", mode: 'copy'
    
    memory params.kallisto?.index_memory ?: '8.GB'
    
    input:
    path transcriptome
    
    output:
    path "kallisto.idx", emit: index
    
    script:
    def kmer_size = params.kallisto?.kmer_size ?: 31
    
    """
    kallisto index \\
        -i kallisto.idx \\
        -k ${kmer_size} \\
        ${transcriptome}
    """
}

/*
 * Kallisto Quantification - Standard mode
 */
process KALLISTO_QUANT {
    tag "kallisto_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/quantification/kallisto", mode: 'copy'
    
    cpus params.kallisto?.cpus ?: 4
    memory params.kallisto?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(reads)
    path index
    val fragment_length   // for single-end only
    val fragment_sd       // for single-end only
    
    output:
    tuple val(sample_id), path("${sample_id}"), emit: results
    tuple val(sample_id), path("${sample_id}/abundance.tsv"), emit: abundance
    tuple val(sample_id), path("${sample_id}/abundance.h5"), emit: h5
    path "${sample_id}/run_info.json", emit: run_info
    
    script:
    def bootstrap = params.kallisto?.bootstrap ?: 0
    
    if (reads instanceof List) {
        // Paired-end
        """
        kallisto quant \\
            -i ${index} \\
            -o ${sample_id} \\
            -t ${task.cpus} \\
            -b ${bootstrap} \\
            ${reads[0]} ${reads[1]}
        """
    } else {
        // Single-end
        def frag_len = fragment_length ?: 200
        def frag_sd = fragment_sd ?: 80
        
        """
        kallisto quant \\
            -i ${index} \\
            -o ${sample_id} \\
            -t ${task.cpus} \\
            -b ${bootstrap} \\
            --single \\
            -l ${frag_len} \\
            -s ${frag_sd} \\
            ${reads}
        """
    }
}

/*
 * Kallisto with Bootstrapping - For uncertainty estimation
 */
process KALLISTO_QUANT_BOOTSTRAP {
    tag "kallisto_boot_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/quantification/kallisto", mode: 'copy'
    
    cpus params.kallisto?.cpus ?: 8
    memory params.kallisto?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(reads)
    path index
    
    output:
    tuple val(sample_id), path("${sample_id}"), emit: results
    tuple val(sample_id), path("${sample_id}/abundance.tsv"), emit: abundance
    path "${sample_id}/run_info.json", emit: run_info
    
    script:
    def bootstrap = params.kallisto?.bootstrap ?: 100
    
    if (reads instanceof List) {
        """
        kallisto quant \\
            -i ${index} \\
            -o ${sample_id} \\
            -t ${task.cpus} \\
            -b ${bootstrap} \\
            ${reads[0]} ${reads[1]}
        """
    } else {
        """
        kallisto quant \\
            -i ${index} \\
            -o ${sample_id} \\
            -t ${task.cpus} \\
            -b ${bootstrap} \\
            --single \\
            -l 200 \\
            -s 80 \\
            ${reads}
        """
    }
}

/*
 * Kallisto Bus - For scRNA-seq
 */
process KALLISTO_BUS {
    tag "kallisto_bus_${sample_id}"
    container "${params.containers.scrnaseq}"
    
    publishDir "${params.outdir}/quantification/kallisto_bus", mode: 'copy'
    
    cpus params.kallisto?.cpus ?: 8
    memory params.kallisto?.memory ?: '32.GB'
    
    input:
    tuple val(sample_id), path(reads)
    path index
    path transcripts_to_genes
    val technology  // "10xv2", "10xv3", "dropseq", etc.
    
    output:
    tuple val(sample_id), path("${sample_id}"), emit: results
    path "${sample_id}/output.bus", emit: bus
    path "${sample_id}/run_info.json", emit: run_info
    
    script:
    """
    kallisto bus \\
        -i ${index} \\
        -o ${sample_id} \\
        -x ${technology} \\
        -t ${task.cpus} \\
        ${reads[0]} ${reads[1]}
    """
}

/*
 * Kallisto Pseudobam - Generate pseudoalignments in BAM format
 */
process KALLISTO_PSEUDOBAM {
    tag "kallisto_pseudobam_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/quantification/kallisto", mode: 'copy'
    
    cpus params.kallisto?.cpus ?: 4
    memory params.kallisto?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(reads)
    path index
    
    output:
    tuple val(sample_id), path("${sample_id}"), emit: results
    tuple val(sample_id), path("${sample_id}/pseudoalignments.bam"), emit: bam
    path "${sample_id}/run_info.json", emit: run_info
    
    script:
    if (reads instanceof List) {
        """
        kallisto quant \\
            -i ${index} \\
            -o ${sample_id} \\
            -t ${task.cpus} \\
            --pseudobam \\
            ${reads[0]} ${reads[1]} \\
            | samtools view -bS - > ${sample_id}/pseudoalignments.bam
        """
    } else {
        """
        kallisto quant \\
            -i ${index} \\
            -o ${sample_id} \\
            -t ${task.cpus} \\
            --pseudobam \\
            --single \\
            -l 200 \\
            -s 80 \\
            ${reads} \\
            | samtools view -bS - > ${sample_id}/pseudoalignments.bam
        """
    }
}

/*
 * Workflow: Complete Kallisto pipeline with indexing
 */
workflow KALLISTO_PIPELINE {
    take:
    reads_ch       // channel: [ val(sample_id), path(reads) ]
    transcriptome  // path: transcriptome fasta
    
    main:
    // Build index
    KALLISTO_INDEX(transcriptome)
    
    // Quantify reads
    KALLISTO_QUANT(reads_ch, KALLISTO_INDEX.out.index, null, null)
    
    emit:
    results = KALLISTO_QUANT.out.results
    abundance = KALLISTO_QUANT.out.abundance
    run_info = KALLISTO_QUANT.out.run_info
}
