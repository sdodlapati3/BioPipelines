/*
 * Augustus Module
 * 
 * Augustus - Gene prediction in eukaryotic genomes
 * Ab initio and evidence-based gene structure prediction
 * Uses existing base container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * Augustus gene prediction
 */
process AUGUSTUS_PREDICT {
    tag "augustus_${sample_id}"
    container "${params.containers.base}"
    
    publishDir "${params.outdir}/annotation/augustus", mode: 'copy'
    
    cpus params.augustus?.cpus ?: 4
    memory params.augustus?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(genome)
    val species
    
    output:
    tuple val(sample_id), path("${sample_id}.gff"), emit: gff
    tuple val(sample_id), path("${sample_id}.gtf"), emit: gtf
    tuple val(sample_id), path("${sample_id}.aa"), emit: proteins
    tuple val(sample_id), path("${sample_id}.codingseq"), emit: cds
    
    script:
    def utr = params.augustus?.utr ? "--UTR=on" : "--UTR=off"
    def strand = params.augustus?.strand ?: "both"
    
    """
    augustus \\
        --species=${species} \\
        ${utr} \\
        --strand=${strand} \\
        --gff3=on \\
        --protein=on \\
        --codingseq=on \\
        --outfile=${sample_id}.gff \\
        ${genome}
    
    # Convert to GTF
    gtf2gff.pl < ${sample_id}.gff --printExon --out=${sample_id}.gtf
    
    # Extract sequences
    getAnnoFasta.pl ${sample_id}.gff
    mv ${sample_id}.gff.aa ${sample_id}.aa
    mv ${sample_id}.gff.codingseq ${sample_id}.codingseq
    """
}

/*
 * Augustus with hints (evidence-based)
 */
process AUGUSTUS_HINTS {
    tag "augustus_hints_${sample_id}"
    container "${params.containers.base}"
    
    publishDir "${params.outdir}/annotation/augustus", mode: 'copy'
    
    cpus params.augustus?.cpus ?: 4
    memory params.augustus?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(genome), path(hints)
    val species
    
    output:
    tuple val(sample_id), path("${sample_id}_hints.gff"), emit: gff
    tuple val(sample_id), path("${sample_id}_hints.gtf"), emit: gtf
    tuple val(sample_id), path("${sample_id}_hints.aa"), emit: proteins
    
    script:
    def utr = params.augustus?.utr ? "--UTR=on" : "--UTR=off"
    def extrinsic_cfg = params.augustus?.extrinsic_cfg ?: "/usr/share/augustus/config/extrinsic/extrinsic.M.RM.E.W.cfg"
    
    """
    augustus \\
        --species=${species} \\
        ${utr} \\
        --hintsfile=${hints} \\
        --extrinsicCfgFile=${extrinsic_cfg} \\
        --gff3=on \\
        --protein=on \\
        --outfile=${sample_id}_hints.gff \\
        ${genome}
    
    # Convert to GTF
    gtf2gff.pl < ${sample_id}_hints.gff --printExon --out=${sample_id}_hints.gtf
    
    # Extract proteins
    getAnnoFasta.pl ${sample_id}_hints.gff
    mv ${sample_id}_hints.gff.aa ${sample_id}_hints.aa
    """
}

/*
 * Workflow: Augustus gene prediction
 */
workflow AUGUSTUS_PIPELINE {
    take:
    genome_ch     // channel: [ val(sample_id), path(genome) ]
    species       // val: species model
    hints_ch      // channel: [ val(sample_id), path(hints) ] (optional)
    
    main:
    AUGUSTUS_PREDICT(genome_ch, species)
    
    if (hints_ch) {
        genome_hints_ch = genome_ch.join(hints_ch)
        AUGUSTUS_HINTS(genome_hints_ch, species)
    }
    
    emit:
    gff = AUGUSTUS_PREDICT.out.gff
    gtf = AUGUSTUS_PREDICT.out.gtf
    proteins = AUGUSTUS_PREDICT.out.proteins
}
