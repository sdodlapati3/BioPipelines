/*
 * BLAST Module
 * 
 * BLAST - Basic Local Alignment Search Tool
 * Sequence similarity searches (blastn, blastp, blastx, tblastn, tblastx)
 * Uses existing base container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * Make BLAST database
 */
process MAKEBLASTDB {
    tag "makeblastdb_${db_name}"
    container "${params.containers.base}"
    
    publishDir "${params.outdir}/reference/blast", mode: 'copy'
    
    memory params.blast?.memory ?: '16.GB'
    
    input:
    path fasta
    val db_name
    val dbtype
    
    output:
    path "${db_name}*", emit: blast_db
    
    script:
    """
    makeblastdb \\
        -in ${fasta} \\
        -dbtype ${dbtype} \\
        -out ${db_name} \\
        -parse_seqids
    """
}

/*
 * BLASTn - Nucleotide-nucleotide BLAST
 */
process BLASTN {
    tag "blastn_${sample_id}"
    container "${params.containers.base}"
    
    publishDir "${params.outdir}/blast/blastn", mode: 'copy'
    
    cpus params.blastn?.cpus ?: 8
    memory params.blastn?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(query)
    path blast_db
    
    output:
    tuple val(sample_id), path("${sample_id}_blastn.txt"), emit: results
    tuple val(sample_id), path("${sample_id}_blastn.xml"), emit: xml
    
    script:
    def evalue = params.blastn?.evalue ?: "1e-5"
    def max_target_seqs = params.blastn?.max_target_seqs ?: 10
    def outfmt = params.blastn?.outfmt ?: 6
    
    """
    blastn \\
        -query ${query} \\
        -db ${blast_db.toString().replaceAll(/\\..*/, '')} \\
        -out ${sample_id}_blastn.txt \\
        -outfmt ${outfmt} \\
        -evalue ${evalue} \\
        -max_target_seqs ${max_target_seqs} \\
        -num_threads ${task.cpus}
    
    # Generate XML output
    blastn \\
        -query ${query} \\
        -db ${blast_db.toString().replaceAll(/\\..*/, '')} \\
        -out ${sample_id}_blastn.xml \\
        -outfmt 5 \\
        -evalue ${evalue} \\
        -max_target_seqs ${max_target_seqs} \\
        -num_threads ${task.cpus}
    """
}

/*
 * BLASTp - Protein-protein BLAST
 */
process BLASTP {
    tag "blastp_${sample_id}"
    container "${params.containers.base}"
    
    publishDir "${params.outdir}/blast/blastp", mode: 'copy'
    
    cpus params.blastp?.cpus ?: 8
    memory params.blastp?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(query)
    path blast_db
    
    output:
    tuple val(sample_id), path("${sample_id}_blastp.txt"), emit: results
    tuple val(sample_id), path("${sample_id}_blastp.xml"), emit: xml
    
    script:
    def evalue = params.blastp?.evalue ?: "1e-5"
    def max_target_seqs = params.blastp?.max_target_seqs ?: 10
    def outfmt = params.blastp?.outfmt ?: 6
    
    """
    blastp \\
        -query ${query} \\
        -db ${blast_db.toString().replaceAll(/\\..*/, '')} \\
        -out ${sample_id}_blastp.txt \\
        -outfmt ${outfmt} \\
        -evalue ${evalue} \\
        -max_target_seqs ${max_target_seqs} \\
        -num_threads ${task.cpus}
    
    blastp \\
        -query ${query} \\
        -db ${blast_db.toString().replaceAll(/\\..*/, '')} \\
        -out ${sample_id}_blastp.xml \\
        -outfmt 5 \\
        -evalue ${evalue} \\
        -max_target_seqs ${max_target_seqs} \\
        -num_threads ${task.cpus}
    """
}

/*
 * BLASTx - Nucleotide query vs protein database
 */
process BLASTX {
    tag "blastx_${sample_id}"
    container "${params.containers.base}"
    
    publishDir "${params.outdir}/blast/blastx", mode: 'copy'
    
    cpus params.blastx?.cpus ?: 8
    memory params.blastx?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(query)
    path blast_db
    
    output:
    tuple val(sample_id), path("${sample_id}_blastx.txt"), emit: results
    
    script:
    def evalue = params.blastx?.evalue ?: "1e-5"
    def max_target_seqs = params.blastx?.max_target_seqs ?: 10
    def outfmt = params.blastx?.outfmt ?: 6
    
    """
    blastx \\
        -query ${query} \\
        -db ${blast_db.toString().replaceAll(/\\..*/, '')} \\
        -out ${sample_id}_blastx.txt \\
        -outfmt ${outfmt} \\
        -evalue ${evalue} \\
        -max_target_seqs ${max_target_seqs} \\
        -num_threads ${task.cpus}
    """
}

/*
 * tBLASTn - Protein query vs nucleotide database
 */
process TBLASTN {
    tag "tblastn_${sample_id}"
    container "${params.containers.base}"
    
    publishDir "${params.outdir}/blast/tblastn", mode: 'copy'
    
    cpus params.tblastn?.cpus ?: 8
    memory params.tblastn?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(query)
    path blast_db
    
    output:
    tuple val(sample_id), path("${sample_id}_tblastn.txt"), emit: results
    
    script:
    def evalue = params.tblastn?.evalue ?: "1e-5"
    def max_target_seqs = params.tblastn?.max_target_seqs ?: 10
    def outfmt = params.tblastn?.outfmt ?: 6
    
    """
    tblastn \\
        -query ${query} \\
        -db ${blast_db.toString().replaceAll(/\\..*/, '')} \\
        -out ${sample_id}_tblastn.txt \\
        -outfmt ${outfmt} \\
        -evalue ${evalue} \\
        -max_target_seqs ${max_target_seqs} \\
        -num_threads ${task.cpus}
    """
}

/*
 * Workflow: BLAST sequence similarity search
 */
workflow BLAST_PIPELINE {
    take:
    query_ch      // channel: [ val(sample_id), path(query) ]
    blast_db      // path: BLAST database files
    blast_type    // val: 'blastn', 'blastp', 'blastx', or 'tblastn'
    
    main:
    if (blast_type == 'blastn') {
        BLASTN(query_ch, blast_db)
        results = BLASTN.out.results
    } else if (blast_type == 'blastp') {
        BLASTP(query_ch, blast_db)
        results = BLASTP.out.results
    } else if (blast_type == 'blastx') {
        BLASTX(query_ch, blast_db)
        results = BLASTX.out.results
    } else if (blast_type == 'tblastn') {
        TBLASTN(query_ch, blast_db)
        results = TBLASTN.out.results
    }
    
    emit:
    results = results
}
