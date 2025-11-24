#!/usr/bin/env nextflow

/*
 * MACS2 - Model-based Analysis of ChIP-Seq
 *
 * Peak calling for ChIP-seq and ATAC-seq data.
 */

process MACS2_CALLPEAK {
    tag "${meta.id}"
    label 'process_medium'
    
    container "${params.containers.chipseq}"
    
    input:
    tuple val(meta), path(bam)
    path control_bam
    
    output:
    tuple val(meta), path('*.narrowPeak'), emit: peaks
    tuple val(meta), path('*.xls')       , emit: xls
    path "versions.yml"                  , emit: versions
    
    when:
    task.ext.when == null || task.ext.when
    
    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    def control = control_bam ? "-c $control_bam" : ''
    def format = meta.single_end ? 'BAM' : 'BAMPE'
    
    """
    macs2 callpeak \\
        -t $bam \\
        $control \\
        -f $format \\
        -g hs \\
        -n $prefix \\
        $args
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        macs2: \$(macs2 --version 2>&1 | sed 's/^macs2 //')
    END_VERSIONS
    """
    
    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}_peaks.narrowPeak
    touch ${prefix}_peaks.xls
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        macs2: 2.2.7
    END_VERSIONS
    """
}
