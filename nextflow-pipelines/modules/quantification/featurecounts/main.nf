#!/usr/bin/env nextflow

/*
 * featureCounts - Count reads mapping to genomic features
 *
 * Quantifies gene expression from aligned BAM files.
 * Part of the Subread package.
 */

process FEATURECOUNTS {
    tag "${meta.id}"
    label 'process_medium'
    
    // Use RNA-seq container (contains Subread/featureCounts)
    container "${params.containers.rnaseq}"
    
    input:
    tuple val(meta), path(bam)
    path gtf
    
    output:
    tuple val(meta), path("*.featureCounts.txt")        , emit: counts
    tuple val(meta), path("*.featureCounts.txt.summary"), emit: summary
    path "versions.yml"                                 , emit: versions
    
    when:
    task.ext.when == null || task.ext.when
    
    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    def paired_end = meta.single_end ? '' : '-p'
    
    // Strandedness: 0=unstranded, 1=forward, 2=reverse
    def strandedness = 0
    if (meta.strandedness == 'forward') {
        strandedness = 1
    } else if (meta.strandedness == 'reverse') {
        strandedness = 2
    }
    
    """
    featureCounts \\
        -T ${task.cpus} \\
        -a $gtf \\
        -o ${prefix}.featureCounts.txt \\
        -s $strandedness \\
        $paired_end \\
        $args \\
        $bam
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        subread: \$(echo \$(featureCounts -v 2>&1) | sed -e "s/featureCounts v//g")
    END_VERSIONS
    """
    
    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.featureCounts.txt
    touch ${prefix}.featureCounts.txt.summary
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        subread: \$(echo \$(featureCounts -v 2>&1) | sed -e "s/featureCounts v//g" || echo "2.0.3")
    END_VERSIONS
    """
}
