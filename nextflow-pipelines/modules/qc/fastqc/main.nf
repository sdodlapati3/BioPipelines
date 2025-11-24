#!/usr/bin/env nextflow

/*
 * FastQC - Quality control for high-throughput sequence data
 *
 * This module runs FastQC on FASTQ files to assess sequencing quality.
 * Adapted from nf-core/fastqc with simplified container handling.
 */

process FASTQC {
    tag "${meta.id}"
    label 'process_medium'
    
    // Use our existing RNA-seq container (contains FastQC)
    container "${params.containers.rnaseq}"
    
    input:
    tuple val(meta), path(reads)
    
    output:
    tuple val(meta), path("*.html"), emit: html
    tuple val(meta), path("*.zip") , emit: zip
    path "versions.yml"            , emit: versions
    
    when:
    task.ext.when == null || task.ext.when
    
    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    
    // Calculate memory per thread for FastQC
    // FastQC allocates (threads * memory) total, so divide by cpus
    def memory_in_mb = task.memory ? task.memory.toUnit('MB') / task.cpus : 512
    def fastqc_memory = memory_in_mb > 10000 ? 10000 : (memory_in_mb < 100 ? 100 : memory_in_mb)
    
    """
    # Fix missing libfreetype.so.6 symlink for Java font rendering
    # Create symlink in work directory (writable)
    ln -sf /opt/conda/lib/libfreetype.so.6.20.4 libfreetype.so.6
    export LD_LIBRARY_PATH=\$PWD:/opt/conda/lib:\${LD_LIBRARY_PATH:-}
    
    fastqc \\
        ${args} \\
        --threads ${task.cpus} \\
        --memory ${fastqc_memory} \\
        --quiet \\
        ${reads}
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        fastqc: \$( fastqc --version | sed '/FastQC v/!d; s/.*v//' )
    END_VERSIONS
    """
    
    stub:
    // For testing without running actual FastQC
    """
    touch ${prefix}_fastqc.html
    touch ${prefix}_fastqc.zip
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        fastqc: \$( fastqc --version 2>&1 | sed '/FastQC v/!d; s/.*v//' || echo "0.12.1" )
    END_VERSIONS
    """
}
