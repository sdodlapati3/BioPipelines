/*
 * BWA Alignment Module
 * 
 * BWA (Burrows-Wheeler Aligner) for DNA-seq alignment
 * Supports both single-end and paired-end reads
 * Uses existing dna-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * BWA Index - Build reference index
 */
process BWA_INDEX {
    tag "bwa_index_${reference.baseName}"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/reference/bwa", mode: 'copy'
    
    input:
    path reference
    
    output:
    path "${reference}*", emit: index
    
    script:
    """
    bwa index ${reference}
    """
}

/*
 * BWA Alignment - Align reads to reference (MEM algorithm)
 */
process BWA_ALIGN {
    tag "bwa_${sample_id}"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/alignments", mode: 'copy',
        pattern: "*.bam"
    publishDir "${params.outdir}/alignments/stats", mode: 'copy',
        pattern: "*.{txt,flagstat}"
    
    cpus params.bwa?.cpus ?: 8
    memory params.bwa?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(reads)
    path reference
    path index
    
    output:
    tuple val(sample_id), path("${sample_id}.bam"), emit: bam
    tuple val(sample_id), path("${sample_id}.bam.bai"), emit: bai
    path "${sample_id}.flagstat.txt", emit: stats
    
    script:
    def read_group = "@RG\\tID:${sample_id}\\tSM:${sample_id}\\tPL:ILLUMINA"
    
    if (reads instanceof List) {
        // Paired-end
        """
        bwa mem \\
            -t ${task.cpus} \\
            -R "${read_group}" \\
            ${reference} \\
            ${reads[0]} ${reads[1]} \\
            | samtools view -bS - \\
            | samtools sort -@ ${task.cpus} -o ${sample_id}.bam -
        
        samtools index ${sample_id}.bam
        samtools flagstat ${sample_id}.bam > ${sample_id}.flagstat.txt
        """
    } else {
        // Single-end
        """
        bwa mem \\
            -t ${task.cpus} \\
            -R "${read_group}" \\
            ${reference} \\
            ${reads} \\
            | samtools view -bS - \\
            | samtools sort -@ ${task.cpus} -o ${sample_id}.bam -
        
        samtools index ${sample_id}.bam
        samtools flagstat ${sample_id}.bam > ${sample_id}.flagstat.txt
        """
    }
}

/*
 * BWA ALN - Alternative algorithm for shorter reads (<70bp)
 */
process BWA_ALN {
    tag "bwa_aln_${sample_id}"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/alignments", mode: 'copy',
        pattern: "*.bam"
    
    cpus params.bwa?.cpus ?: 8
    memory params.bwa?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(reads)
    path reference
    path index
    
    output:
    tuple val(sample_id), path("${sample_id}.bam"), emit: bam
    tuple val(sample_id), path("${sample_id}.bam.bai"), emit: bai
    
    script:
    if (reads instanceof List) {
        // Paired-end
        """
        bwa aln -t ${task.cpus} ${reference} ${reads[0]} > ${sample_id}_R1.sai
        bwa aln -t ${task.cpus} ${reference} ${reads[1]} > ${sample_id}_R2.sai
        
        bwa sampe ${reference} \\
            ${sample_id}_R1.sai ${sample_id}_R2.sai \\
            ${reads[0]} ${reads[1]} \\
            | samtools view -bS - \\
            | samtools sort -@ ${task.cpus} -o ${sample_id}.bam -
        
        samtools index ${sample_id}.bam
        """
    } else {
        // Single-end
        """
        bwa aln -t ${task.cpus} ${reference} ${reads} > ${sample_id}.sai
        
        bwa samse ${reference} ${sample_id}.sai ${reads} \\
            | samtools view -bS - \\
            | samtools sort -@ ${task.cpus} -o ${sample_id}.bam -
        
        samtools index ${sample_id}.bam
        """
    }
}

/*
 * Workflow: Complete BWA pipeline with indexing
 */
workflow BWA_PIPELINE {
    take:
    reads_ch      // channel: [ val(sample_id), path(reads) ]
    reference     // path: reference genome fasta
    
    main:
    // Build index if not provided
    BWA_INDEX(reference)
    
    // Align reads
    BWA_ALIGN(reads_ch, reference, BWA_INDEX.out.index)
    
    emit:
    bam = BWA_ALIGN.out.bam
    bai = BWA_ALIGN.out.bai
    stats = BWA_ALIGN.out.stats
}
