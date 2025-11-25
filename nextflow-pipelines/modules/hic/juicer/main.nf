/*
 * Juicer Module
 * 
 * Juicer - Hi-C analysis pipeline
 * Complete pipeline for Hi-C data processing and visualization
 * Uses existing hic container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * Juicer preprocessing
 */
process JUICER_PRE {
    tag "juicer_${sample_id}"
    container "${params.containers.hic}"
    
    publishDir "${params.outdir}/hic/juicer", mode: 'copy'
    
    cpus params.juicer?.cpus ?: 32
    memory params.juicer?.memory ?: '128.GB'
    
    input:
    tuple val(sample_id), path(read1), path(read2)
    path reference
    path chrom_sizes
    path restriction_site
    
    output:
    tuple val(sample_id), path("${sample_id}/aligned/merged_nodups.txt"), emit: merged_nodups
    path "${sample_id}/aligned/inter.txt", emit: inter_stats
    path "${sample_id}/aligned/collisions.txt", emit: collisions
    
    script:
    def enzyme = params.juicer?.enzyme ?: "MboI"
    
    """
    # Create directory structure
    mkdir -p ${sample_id}/fastq
    mkdir -p ${sample_id}/splits
    mkdir -p ${sample_id}/aligned
    
    # Link FASTQ files
    ln -s ${read1} ${sample_id}/fastq/${sample_id}_R1.fastq.gz
    ln -s ${read2} ${sample_id}/fastq/${sample_id}_R2.fastq.gz
    
    # Run Juicer
    juicer.sh \\
        -d ${sample_id} \\
        -p ${chrom_sizes} \\
        -y ${restriction_site} \\
        -z ${reference} \\
        -t ${task.cpus} \\
        -s ${enzyme}
    """
}

/*
 * Juicer tools - Create .hic file
 */
process JUICER_TOOLS_HIC {
    tag "juicer_hic_${sample_id}"
    container "${params.containers.hic}"
    
    publishDir "${params.outdir}/hic/juicer", mode: 'copy'
    
    memory params.juicer?.memory ?: '64.GB'
    
    input:
    tuple val(sample_id), path(merged_nodups)
    path chrom_sizes
    val resolutions
    
    output:
    tuple val(sample_id), path("${sample_id}.hic"), emit: hic_file
    
    script:
    def res_list = resolutions instanceof List ? resolutions.join(',') : resolutions
    def mapq = params.juicer?.mapq ?: 30
    
    """
    java -Xmx${task.memory.toGiga()}g -jar \$JUICER_TOOLS_JAR pre \\
        -r ${res_list} \\
        -q ${mapq} \\
        ${merged_nodups} \\
        ${sample_id}.hic \\
        ${chrom_sizes}
    """
}

/*
 * Juicer tools - Extract matrix
 */
process JUICER_TOOLS_DUMP {
    tag "juicer_dump_${sample_id}_${chr}"
    container "${params.containers.hic}"
    
    publishDir "${params.outdir}/hic/juicer/matrices", mode: 'copy'
    
    memory params.juicer?.memory ?: '32.GB'
    
    input:
    tuple val(sample_id), path(hic_file), val(chr)
    val resolution
    val normalization
    
    output:
    tuple val(sample_id), path("${sample_id}_${chr}_${resolution}.txt"), emit: matrix
    
    script:
    """
    java -Xmx${task.memory.toGiga()}g -jar \$JUICER_TOOLS_JAR dump \\
        observed \\
        ${normalization} \\
        ${hic_file} \\
        ${chr} \\
        ${chr} \\
        BP \\
        ${resolution} \\
        ${sample_id}_${chr}_${resolution}.txt
    """
}

/*
 * Juicer tools - Arrowhead (TAD calling)
 */
process JUICER_ARROWHEAD {
    tag "arrowhead_${sample_id}"
    container "${params.containers.hic}"
    
    publishDir "${params.outdir}/hic/juicer/tads", mode: 'copy'
    
    cpus params.juicer?.cpus ?: 8
    memory params.juicer?.memory ?: '64.GB'
    
    input:
    tuple val(sample_id), path(hic_file)
    val resolution
    
    output:
    path "${sample_id}_arrowhead", emit: tad_dir
    path "${sample_id}_arrowhead/*_blocks.bedpe", emit: tad_blocks
    
    script:
    """
    java -Xmx${task.memory.toGiga()}g -jar \$JUICER_TOOLS_JAR arrowhead \\
        --threads ${task.cpus} \\
        -r ${resolution} \\
        ${hic_file} \\
        ${sample_id}_arrowhead
    """
}

/*
 * Juicer tools - HiCCUPS (loop calling)
 */
process JUICER_HICCUPS {
    tag "hiccups_${sample_id}"
    container "${params.containers.hic}"
    
    publishDir "${params.outdir}/hic/juicer/loops", mode: 'copy'
    
    cpus params.juicer?.cpus ?: 8
    memory params.juicer?.memory ?: '64.GB'
    
    input:
    tuple val(sample_id), path(hic_file)
    val resolution
    
    output:
    path "${sample_id}_hiccups", emit: loop_dir
    path "${sample_id}_hiccups/merged_loops.bedpe", emit: loops
    
    script:
    """
    java -Xmx${task.memory.toGiga()}g -jar \$JUICER_TOOLS_JAR hiccups \\
        --threads ${task.cpus} \\
        -r ${resolution} \\
        ${hic_file} \\
        ${sample_id}_hiccups
    """
}

/*
 * Workflow: Juicer Hi-C analysis
 */
workflow JUICER_PIPELINE {
    take:
    reads_ch         // channel: [ val(sample_id), path(read1), path(read2) ]
    reference        // path: reference genome
    chrom_sizes      // path: chromosome sizes
    restriction_site // path: restriction site file
    resolutions      // val: list of resolutions
    
    main:
    JUICER_PRE(reads_ch, reference, chrom_sizes, restriction_site)
    JUICER_TOOLS_HIC(JUICER_PRE.out.merged_nodups, chrom_sizes, resolutions)
    JUICER_ARROWHEAD(JUICER_TOOLS_HIC.out.hic_file, resolutions[0])
    JUICER_HICCUPS(JUICER_TOOLS_HIC.out.hic_file, resolutions[0])
    
    emit:
    hic_file = JUICER_TOOLS_HIC.out.hic_file
    tads = JUICER_ARROWHEAD.out.tad_blocks
    loops = JUICER_HICCUPS.out.loops
}
