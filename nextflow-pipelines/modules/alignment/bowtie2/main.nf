#!/usr/bin/env nextflow

/*
 * Bowtie2 - Fast alignment for short reads
 *
 * Aligns ChIP-seq/ATAC-seq reads to reference genome.
 */

process BOWTIE2_ALIGN {
    tag "${meta.id}"
    label 'process_high'
    
    container "${params.containers.chipseq}"
    
    input:
    tuple val(meta), path(reads)
    path index
    
    output:
    tuple val(meta), path('*.bam'), emit: bam
    tuple val(meta), path('*.bai'), emit: bai
    path "versions.yml"           , emit: versions
    
    when:
    task.ext.when == null || task.ext.when
    
    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    def reads_cmd = meta.single_end ? "-U ${reads}" : "-1 ${reads[0]} -2 ${reads[1]}"
    
    """
    # Fix libfreetype symlink (same as other modules)
    ln -sf /opt/conda/lib/libfreetype.so.6.20.4 libfreetype.so.6 2>/dev/null || true
    export LD_LIBRARY_PATH=\$PWD:/opt/conda/lib:\${LD_LIBRARY_PATH:-}
    
    bowtie2 \\
        -x $index/genome \\
        $reads_cmd \\
        --threads ${task.cpus} \\
        $args \\
        2> ${prefix}.bowtie2.log \\
        | samtools view -@ ${task.cpus} -bS - \\
        | samtools sort -@ ${task.cpus} -o ${prefix}.sorted.bam -
    
    samtools index ${prefix}.sorted.bam
    mv ${prefix}.sorted.bam ${prefix}.bam
    mv ${prefix}.sorted.bam.bai ${prefix}.bai
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bowtie2: \$(echo \$(bowtie2 --version 2>&1) | head -n1 | sed 's/^.*version //; s/ .*\$//')
        samtools: \$(echo \$(samtools --version 2>&1) | head -n1 | sed 's/^.*samtools //; s/ .*\$//')
    END_VERSIONS
    """
    
    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.bam
    touch ${prefix}.bai
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bowtie2: 2.5.0
        samtools: 1.17
    END_VERSIONS
    """
}
