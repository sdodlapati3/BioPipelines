/*
 * Prokka Module
 * 
 * Prokka - Rapid prokaryotic genome annotation
 * Identifies genes, rRNA, tRNA, and other features
 * Uses existing metagenomics container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * Prokka annotation
 */
process PROKKA_ANNOTATE {
    tag "prokka_${sample_id}"
    container "${params.containers.metagenomics}"
    
    publishDir "${params.outdir}/annotation/prokka", mode: 'copy'
    
    cpus params.prokka?.cpus ?: 8
    memory params.prokka?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(assembly)
    val kingdom
    
    output:
    path "${sample_id}_prokka/*", emit: results
    path "${sample_id}_prokka/${sample_id}.gff", emit: gff
    path "${sample_id}_prokka/${sample_id}.gbk", emit: gbk
    path "${sample_id}_prokka/${sample_id}.faa", emit: proteins
    path "${sample_id}_prokka/${sample_id}.ffn", emit: genes
    path "${sample_id}_prokka/${sample_id}.fna", emit: contigs
    path "${sample_id}_prokka/${sample_id}.txt", emit: stats
    
    script:
    def genus = params.prokka?.genus ?: ""
    def species = params.prokka?.species ?: ""
    def strain = params.prokka?.strain ?: ""
    def plasmid = params.prokka?.plasmid ? "--plasmid ${params.prokka.plasmid}" : ""
    
    def genus_opt = genus ? "--genus ${genus}" : ""
    def species_opt = species ? "--species ${species}" : ""
    def strain_opt = strain ? "--strain ${strain}" : ""
    
    """
    prokka \\
        --outdir ${sample_id}_prokka \\
        --prefix ${sample_id} \\
        --kingdom ${kingdom} \\
        ${genus_opt} \\
        ${species_opt} \\
        ${strain_opt} \\
        ${plasmid} \\
        --cpus ${task.cpus} \\
        --force \\
        ${assembly}
    """
}

/*
 * Workflow: Prokka annotation pipeline
 */
workflow PROKKA_PIPELINE {
    take:
    assembly_ch  // channel: [ val(sample_id), path(assembly) ]
    kingdom      // val: "Bacteria", "Archaea", "Viruses"
    
    main:
    PROKKA_ANNOTATE(assembly_ch, kingdom)
    
    emit:
    gff = PROKKA_ANNOTATE.out.gff
    proteins = PROKKA_ANNOTATE.out.proteins
    genes = PROKKA_ANNOTATE.out.genes
}
