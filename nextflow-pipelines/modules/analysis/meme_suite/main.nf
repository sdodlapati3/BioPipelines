/*
 * MEME Suite Module
 * 
 * MEME Suite - Motif discovery and analysis
 * MEME, DREME, FIMO, TOMTOM for comprehensive motif analysis
 * Uses existing chip-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * MEME - Motif discovery
 */
process MEME {
    tag "meme_${sample_id}"
    container "${params.containers.chipseq}"
    
    publishDir "${params.outdir}/motifs/meme", mode: 'copy'
    
    cpus params.meme?.cpus ?: 8
    memory params.meme?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(sequences)
    val nmotifs
    val minw
    val maxw
    
    output:
    path "${sample_id}_meme", emit: results_dir
    path "${sample_id}_meme/meme.html", emit: html
    path "${sample_id}_meme/meme.txt", emit: txt
    path "${sample_id}_meme/meme.xml", emit: xml
    
    script:
    def mod = params.meme?.mod ?: "zoops"
    
    """
    meme ${sequences} \\
        -dna \\
        -oc ${sample_id}_meme \\
        -mod ${mod} \\
        -nmotifs ${nmotifs} \\
        -minw ${minw} \\
        -maxw ${maxw} \\
        -p ${task.cpus}
    """
}

/*
 * DREME - Discriminative motif discovery
 */
process DREME {
    tag "dreme_${sample_id}"
    container "${params.containers.chipseq}"
    
    publishDir "${params.outdir}/motifs/dreme", mode: 'copy'
    
    cpus params.dreme?.cpus ?: 8
    memory params.dreme?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(positive_sequences)
    path negative_sequences
    
    output:
    path "${sample_id}_dreme", emit: results_dir
    path "${sample_id}_dreme/dreme.html", emit: html
    path "${sample_id}_dreme/dreme.txt", emit: txt
    path "${sample_id}_dreme/dreme.xml", emit: xml
    
    script:
    def neg_opt = negative_sequences ? "-n ${negative_sequences}" : ""
    def evalue = params.dreme?.evalue ?: 0.05
    
    """
    dreme \\
        -p ${positive_sequences} \\
        ${neg_opt} \\
        -oc ${sample_id}_dreme \\
        -e ${evalue} \\
        -dna \\
        -p ${task.cpus}
    """
}

/*
 * FIMO - Find individual motif occurrences
 */
process FIMO {
    tag "fimo_${sample_id}"
    container "${params.containers.chipseq}"
    
    publishDir "${params.outdir}/motifs/fimo", mode: 'copy'
    
    cpus params.fimo?.cpus ?: 4
    memory params.fimo?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(motifs), path(sequences)
    
    output:
    path "${sample_id}_fimo", emit: results_dir
    path "${sample_id}_fimo/fimo.tsv", emit: tsv
    path "${sample_id}_fimo/fimo.html", emit: html
    path "${sample_id}_fimo/fimo.gff", emit: gff
    
    script:
    def thresh = params.fimo?.thresh ?: 1e-4
    
    """
    fimo \\
        --oc ${sample_id}_fimo \\
        --thresh ${thresh} \\
        ${motifs} \\
        ${sequences}
    """
}

/*
 * TOMTOM - Motif comparison
 */
process TOMTOM {
    tag "tomtom_${sample_id}"
    container "${params.containers.chipseq}"
    
    publishDir "${params.outdir}/motifs/tomtom", mode: 'copy'
    
    memory params.tomtom?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(query_motifs)
    path target_motifs
    
    output:
    path "${sample_id}_tomtom", emit: results_dir
    path "${sample_id}_tomtom/tomtom.html", emit: html
    path "${sample_id}_tomtom/tomtom.tsv", emit: tsv
    path "${sample_id}_tomtom/tomtom.xml", emit: xml
    
    script:
    def dist = params.tomtom?.dist ?: "pearson"
    def thresh = params.tomtom?.thresh ?: 0.5
    
    """
    tomtom \\
        -oc ${sample_id}_tomtom \\
        -dist ${dist} \\
        -thresh ${thresh} \\
        ${query_motifs} \\
        ${target_motifs}
    """
}

/*
 * AME - Analysis of Motif Enrichment
 */
process AME {
    tag "ame_${sample_id}"
    container "${params.containers.chipseq}"
    
    publishDir "${params.outdir}/motifs/ame", mode: 'copy'
    
    cpus params.ame?.cpus ?: 4
    memory params.ame?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(sequences)
    path motif_database
    path control_sequences
    
    output:
    path "${sample_id}_ame", emit: results_dir
    path "${sample_id}_ame/ame.html", emit: html
    path "${sample_id}_ame/ame.tsv", emit: tsv
    
    script:
    def control_opt = control_sequences ? "--control ${control_sequences}" : ""
    def method = params.ame?.method ?: "ranksum"
    
    """
    ame \\
        --oc ${sample_id}_ame \\
        --method ${method} \\
        ${control_opt} \\
        ${sequences} \\
        ${motif_database}
    """
}

/*
 * Workflow: MEME Suite motif analysis
 */
workflow MEME_PIPELINE {
    take:
    sequences_ch      // channel: [ val(sample_id), path(sequences) ]
    nmotifs           // val: number of motifs to find
    minw              // val: minimum motif width
    maxw              // val: maximum motif width
    
    main:
    MEME(sequences_ch, nmotifs, minw, maxw)
    DREME(sequences_ch, null)
    
    emit:
    meme_results = MEME.out.results_dir
    dreme_results = DREME.out.results_dir
}
