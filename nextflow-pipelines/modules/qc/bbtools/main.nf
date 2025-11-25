/*
 * BBTools Module
 * 
 * BBTools - Comprehensive suite for NGS data manipulation
 * BBDuk (filtering), BBMap (aligner), repair, reformat, stats
 * Uses existing base container with BBTools
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * BBDuk - Quality and adapter trimming
 */
process BBDUK_TRIM {
    tag "bbduk_${sample_id}"
    container "${params.containers.base}"
    
    publishDir "${params.outdir}/qc/bbduk", mode: 'copy'
    
    cpus params.bbduk?.cpus ?: 8
    memory params.bbduk?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(reads)
    path adapters
    
    output:
    tuple val(sample_id), path("${sample_id}_trimmed_*.fastq.gz"), emit: reads
    path "${sample_id}_bbduk.stats", emit: stats
    
    script:
    def ktrim = params.bbduk?.ktrim ?: "r"
    def k = params.bbduk?.k ?: 23
    def mink = params.bbduk?.mink ?: 11
    def hdist = params.bbduk?.hdist ?: 1
    def qtrim = params.bbduk?.qtrim ?: "rl"
    def trimq = params.bbduk?.trimq ?: 20
    def minlen = params.bbduk?.minlen ?: 50
    
    if (reads instanceof List) {
        """
        bbduk.sh \\
            in1=${reads[0]} \\
            in2=${reads[1]} \\
            out1=${sample_id}_trimmed_R1.fastq.gz \\
            out2=${sample_id}_trimmed_R2.fastq.gz \\
            ref=${adapters} \\
            ktrim=${ktrim} \\
            k=${k} \\
            mink=${mink} \\
            hdist=${hdist} \\
            qtrim=${qtrim} \\
            trimq=${trimq} \\
            minlen=${minlen} \\
            threads=${task.cpus} \\
            stats=${sample_id}_bbduk.stats
        """
    } else {
        """
        bbduk.sh \\
            in=${reads} \\
            out=${sample_id}_trimmed_SE.fastq.gz \\
            ref=${adapters} \\
            ktrim=${ktrim} \\
            k=${k} \\
            mink=${mink} \\
            hdist=${hdist} \\
            qtrim=${qtrim} \\
            trimq=${trimq} \\
            minlen=${minlen} \\
            threads=${task.cpus} \\
            stats=${sample_id}_bbduk.stats
        """
    }
}

/*
 * BBMap - Fast splice-aware aligner
 */
process BBMAP_ALIGN {
    tag "bbmap_${sample_id}"
    container "${params.containers.base}"
    
    publishDir "${params.outdir}/alignment/bbmap", mode: 'copy'
    
    cpus params.bbmap?.cpus ?: 16
    memory params.bbmap?.memory ?: '32.GB'
    
    input:
    tuple val(sample_id), path(reads)
    path reference
    
    output:
    tuple val(sample_id), path("${sample_id}.bam"), emit: bam
    path "${sample_id}_bbmap.stats", emit: stats
    
    script:
    def maxindel = params.bbmap?.maxindel ?: 200000
    
    if (reads instanceof List) {
        """
        bbmap.sh \\
            in1=${reads[0]} \\
            in2=${reads[1]} \\
            ref=${reference} \\
            out=${sample_id}.bam \\
            maxindel=${maxindel} \\
            threads=${task.cpus} \\
            statsfile=${sample_id}_bbmap.stats \\
            -Xmx${task.memory.toGiga()}g
        """
    } else {
        """
        bbmap.sh \\
            in=${reads} \\
            ref=${reference} \\
            out=${sample_id}.bam \\
            maxindel=${maxindel} \\
            threads=${task.cpus} \\
            statsfile=${sample_id}_bbmap.stats \\
            -Xmx${task.memory.toGiga()}g
        """
    }
}

/*
 * Repair - Fix broken paired-end files
 */
process BBTOOLS_REPAIR {
    tag "repair_${sample_id}"
    container "${params.containers.base}"
    
    publishDir "${params.outdir}/qc/repair", mode: 'copy'
    
    memory params.bbtools?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(read1), path(read2)
    
    output:
    tuple val(sample_id), path("${sample_id}_repaired_R1.fastq.gz"), path("${sample_id}_repaired_R2.fastq.gz"), emit: reads
    path "${sample_id}_singletons.fastq.gz", emit: singletons
    
    script:
    """
    repair.sh \\
        in1=${read1} \\
        in2=${read2} \\
        out1=${sample_id}_repaired_R1.fastq.gz \\
        out2=${sample_id}_repaired_R2.fastq.gz \\
        outs=${sample_id}_singletons.fastq.gz \\
        -Xmx${task.memory.toGiga()}g
    """
}

/*
 * Reformat - Convert and manipulate sequence files
 */
process BBTOOLS_REFORMAT {
    tag "reformat_${sample_id}"
    container "${params.containers.base}"
    
    publishDir "${params.outdir}/utilities/reformat", mode: 'copy'
    
    memory params.bbtools?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(reads)
    val out_format
    
    output:
    tuple val(sample_id), path("${sample_id}_reformatted.*"), emit: reads
    
    script:
    def ext = out_format == "fasta" ? "fasta" : "fastq"
    
    """
    reformat.sh \\
        in=${reads} \\
        out=${sample_id}_reformatted.${ext} \\
        -Xmx${task.memory.toGiga()}g
    """
}

/*
 * Workflow: BBTools QC and preprocessing
 */
workflow BBTOOLS_PIPELINE {
    take:
    reads_ch      // channel: [ val(sample_id), path(reads) ]
    adapters      // path: adapter sequences
    
    main:
    BBDUK_TRIM(reads_ch, adapters)
    
    emit:
    trimmed_reads = BBDUK_TRIM.out.reads
    stats = BBDUK_TRIM.out.stats
}
