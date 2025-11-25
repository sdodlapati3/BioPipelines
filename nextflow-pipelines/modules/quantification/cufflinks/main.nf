/*
 * Cufflinks Module
 * 
 * Cufflinks - Transcript assembly and differential expression
 * Reference-based transcript assembly from RNA-seq alignments
 * Uses existing rna-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * Cufflinks - Assemble transcripts from BAM
 */
process CUFFLINKS_ASSEMBLE {
    tag "cufflinks_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/assembly/cufflinks", mode: 'copy'
    
    cpus params.cufflinks?.cpus ?: 8
    memory params.cufflinks?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(bam)
    path gtf
    val strandedness
    
    output:
    tuple val(sample_id), path("${sample_id}/transcripts.gtf"), emit: gtf
    path "${sample_id}/genes.fpkm_tracking", emit: genes_fpkm
    path "${sample_id}/isoforms.fpkm_tracking", emit: isoforms_fpkm
    
    script:
    def library_type = ""
    if (strandedness == "forward") {
        library_type = "--library-type fr-firststrand"
    } else if (strandedness == "reverse") {
        library_type = "--library-type fr-secondstrand"
    } else {
        library_type = "--library-type fr-unstranded"
    }
    
    def gtf_opt = gtf ? "-g ${gtf}" : ""
    
    """
    cufflinks \\
        -o ${sample_id} \\
        -p ${task.cpus} \\
        ${library_type} \\
        ${gtf_opt} \\
        --multi-read-correct \\
        ${bam}
    """
}

/*
 * Cuffmerge - Merge assemblies from multiple samples
 */
process CUFFMERGE {
    tag "cuffmerge"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/assembly/cufflinks", mode: 'copy'
    
    cpus params.cufflinks?.cpus ?: 8
    memory params.cufflinks?.memory ?: '16.GB'
    
    input:
    path gtf_files
    path reference_gtf
    path reference_fasta
    
    output:
    path "merged/merged.gtf", emit: merged_gtf
    
    script:
    """
    # Create assembly list file
    ls -1 ${gtf_files} > assembly_list.txt
    
    cuffmerge \\
        -o merged \\
        -g ${reference_gtf} \\
        -s ${reference_fasta} \\
        -p ${task.cpus} \\
        assembly_list.txt
    """
}

/*
 * Cuffquant - Precompute gene expression levels
 */
process CUFFQUANT {
    tag "cuffquant_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/quantification/cufflinks", mode: 'copy'
    
    cpus params.cufflinks?.cpus ?: 8
    memory params.cufflinks?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(bam)
    path merged_gtf
    val strandedness
    
    output:
    tuple val(sample_id), path("${sample_id}/abundances.cxb"), emit: abundances
    
    script:
    def library_type = ""
    if (strandedness == "forward") {
        library_type = "--library-type fr-firststrand"
    } else if (strandedness == "reverse") {
        library_type = "--library-type fr-secondstrand"
    } else {
        library_type = "--library-type fr-unstranded"
    }
    
    """
    cuffquant \\
        -o ${sample_id} \\
        -p ${task.cpus} \\
        ${library_type} \\
        --multi-read-correct \\
        ${merged_gtf} \\
        ${bam}
    """
}

/*
 * Cuffnorm - Normalize expression across samples
 */
process CUFFNORM {
    tag "cuffnorm"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/quantification/cufflinks/normalized", mode: 'copy'
    
    cpus params.cufflinks?.cpus ?: 8
    memory params.cufflinks?.memory ?: '16.GB'
    
    input:
    path merged_gtf
    path abundances
    val labels
    
    output:
    path "cuffnorm_output/*", emit: results
    path "cuffnorm_output/genes.fpkm_table", emit: genes_fpkm_table
    path "cuffnorm_output/isoforms.fpkm_table", emit: isoforms_fpkm_table
    
    script:
    def abundance_files = abundances.collect { it.toString() }.join(' ')
    def label_string = labels.join(',')
    
    """
    cuffnorm \\
        -o cuffnorm_output \\
        -p ${task.cpus} \\
        -L ${label_string} \\
        ${merged_gtf} \\
        ${abundance_files}
    """
}

/*
 * Cuffdiff - Differential expression analysis
 */
process CUFFDIFF {
    tag "cuffdiff"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/differential_expression/cufflinks", mode: 'copy'
    
    cpus params.cuffdiff?.cpus ?: 12
    memory params.cuffdiff?.memory ?: '32.GB'
    
    input:
    path merged_gtf
    path group1_abundances
    path group2_abundances
    val group1_label
    val group2_label
    
    output:
    path "cuffdiff_output/*", emit: results
    path "cuffdiff_output/gene_exp.diff", emit: gene_diff
    path "cuffdiff_output/isoform_exp.diff", emit: isoform_diff
    
    script:
    def group1_files = group1_abundances.collect { it.toString() }.join(',')
    def group2_files = group2_abundances.collect { it.toString() }.join(',')
    
    """
    cuffdiff \\
        -o cuffdiff_output \\
        -p ${task.cpus} \\
        -L ${group1_label},${group2_label} \\
        --multi-read-correct \\
        ${merged_gtf} \\
        ${group1_files} \\
        ${group2_files}
    """
}

/*
 * Workflow: Complete Cufflinks pipeline
 */
workflow CUFFLINKS_PIPELINE {
    take:
    bam_ch             // channel: [ val(sample_id), path(bam) ]
    reference_gtf      // path: reference annotation
    reference_fasta    // path: reference genome
    strandedness       // val: "none", "forward", "reverse"
    
    main:
    // Assemble transcripts for each sample
    CUFFLINKS_ASSEMBLE(bam_ch, reference_gtf, strandedness)
    
    // Merge all assemblies
    all_gtfs = CUFFLINKS_ASSEMBLE.out.gtf.map { it[1] }.collect()
    CUFFMERGE(all_gtfs, reference_gtf, reference_fasta)
    
    // Quantify with merged assembly
    CUFFQUANT(
        bam_ch,
        CUFFMERGE.out.merged_gtf,
        strandedness
    )
    
    emit:
    merged_gtf = CUFFMERGE.out.merged_gtf
    abundances = CUFFQUANT.out.abundances
    genes_fpkm = CUFFLINKS_ASSEMBLE.out.genes_fpkm
    isoforms_fpkm = CUFFLINKS_ASSEMBLE.out.isoforms_fpkm
}

/*
 * Workflow: Cuffdiff differential expression
 */
workflow CUFFDIFF_WORKFLOW {
    take:
    merged_gtf         // path: merged assembly
    group1_abundances  // channel: abundances for group 1
    group2_abundances  // channel: abundances for group 2
    group1_label       // val: label for group 1
    group2_label       // val: label for group 2
    
    main:
    CUFFDIFF(
        merged_gtf,
        group1_abundances,
        group2_abundances,
        group1_label,
        group2_label
    )
    
    emit:
    gene_diff = CUFFDIFF.out.gene_diff
    isoform_diff = CUFFDIFF.out.isoform_diff
    results = CUFFDIFF.out.results
}
