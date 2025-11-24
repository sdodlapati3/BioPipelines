#!/usr/bin/env nextflow

nextflow.enable.dsl=2

/*
 * Bowtie2 Alignment Module (Tier 2 Container)
 * ============================================
 * Performs fast alignment for ChIP-seq, ATAC-seq, DNA-seq
 * 
 * Container: alignment_short_read.sif (Tier 2)
 * Tool: Bowtie2 2.5.3
 */

process BOWTIE2_ALIGN {
    tag "${sample_id}"
    container "/scratch/sdodl001/BioPipelines/containers/tier2/alignment_short_read.sif"
    
    cpus 8
    memory '16 GB'
    time '2h'
    
    publishDir "${params.outdir}/bowtie2/${sample_id}", mode: 'copy'
    
    input:
    tuple val(sample_id), path(reads)
    path index
    
    output:
    tuple val(sample_id), path("${sample_id}.bam"), emit: bam
    tuple val(sample_id), path("${sample_id}.bam.bai"), emit: bai
    path "${sample_id}.bowtie2.log", emit: log
    
    script:
    def read1 = reads[0]
    def read2 = reads.size() > 1 ? reads[1] : ""
    def input_reads = read2 ? "-1 ${read1} -2 ${read2}" : "-U ${read1}"
    
    """
    # Extract index basename (assumes index files like hg38.*.bt2)
    INDEX_BASE=\$(basename ${index}/*.1.bt2 .1.bt2)
    INDEX_DIR=\$(dirname ${index}/*.1.bt2)
    
    # Run Bowtie2 alignment
    bowtie2 \
        -x \${INDEX_DIR}/\${INDEX_BASE} \
        ${input_reads} \
        -p ${task.cpus} \
        --very-sensitive \
        2> ${sample_id}.bowtie2.log \
        | samtools view -@ ${task.cpus} -bS - \
        | samtools sort -@ ${task.cpus} -o ${sample_id}.bam -
    
    # Index BAM
    samtools index -@ ${task.cpus} ${sample_id}.bam
    """
}

/*
 * Bowtie2 Index Generation
 * =========================
 * Build Bowtie2 index from reference FASTA
 */

process BOWTIE2_INDEX {
    tag "${genome_name}"
    container "/scratch/sdodl001/BioPipelines/containers/tier2/alignment_short_read.sif"
    
    cpus 8
    memory '16 GB'
    time '2h'
    
    publishDir "${params.outdir}/bowtie2_index", mode: 'copy'
    
    input:
    path genome_fasta
    val genome_name
    
    output:
    path "bowtie2_index_${genome_name}", emit: index
    
    script:
    """
    mkdir -p bowtie2_index_${genome_name}
    
    bowtie2-build \
        --threads ${task.cpus} \
        ${genome_fasta} \
        bowtie2_index_${genome_name}/${genome_name}
    """
}

/*
 * Bowtie2 Alignment with Quality Filtering
 * =========================================
 * Aligned with additional filtering for high-quality reads
 */

process BOWTIE2_ALIGN_FILTERED {
    tag "${sample_id}"
    container "/scratch/sdodl001/BioPipelines/containers/tier2/alignment_short_read.sif"
    
    cpus 8
    memory '16 GB'
    time '2h'
    
    publishDir "${params.outdir}/bowtie2/${sample_id}", mode: 'copy'
    
    input:
    tuple val(sample_id), path(reads)
    path index
    val mapq_threshold
    
    output:
    tuple val(sample_id), path("${sample_id}.filtered.bam"), emit: bam
    tuple val(sample_id), path("${sample_id}.filtered.bam.bai"), emit: bai
    path "${sample_id}.bowtie2.log", emit: log
    path "${sample_id}.stats.txt", emit: stats
    
    script:
    def read1 = reads[0]
    def read2 = reads.size() > 1 ? reads[1] : ""
    def input_reads = read2 ? "-1 ${read1} -2 ${read2}" : "-U ${read1}"
    
    """
    # Extract index basename
    INDEX_BASE=\$(basename ${index}/*.1.bt2 .1.bt2)
    INDEX_DIR=\$(dirname ${index}/*.1.bt2)
    
    # Align and filter
    bowtie2 \
        -x \${INDEX_DIR}/\${INDEX_BASE} \
        ${input_reads} \
        -p ${task.cpus} \
        --very-sensitive \
        2> ${sample_id}.bowtie2.log \
        | samtools view -@ ${task.cpus} -bS -q ${mapq_threshold} -F 4 - \
        | samtools sort -@ ${task.cpus} -o ${sample_id}.filtered.bam -
    
    # Index and stats
    samtools index -@ ${task.cpus} ${sample_id}.filtered.bam
    samtools flagstat ${sample_id}.filtered.bam > ${sample_id}.stats.txt
    """
}
