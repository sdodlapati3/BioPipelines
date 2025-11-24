process BISMARK_METHYLATION_EXTRACTOR {
    tag "$meta.id"
    label 'process_medium'

    container params.containers.methylation

    input:
    tuple val(meta), path(bam), path(bai)

    output:
    tuple val(meta), path("*.bedGraph.gz"), emit: bedgraph
    tuple val(meta), path("*.bismark.cov.gz"), emit: coverage
    path "*.txt"                             , emit: report
    path "versions.yml"                      , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    
    """
    bismark_methylation_extractor \\
        $args \\
        --paired-end \\
        --parallel ${task.cpus} \\
        --bedGraph \\
        --counts \\
        --gzip \\
        --buffer_size 10G \\
        ${bam}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bismark: \$(echo \$(bismark_methylation_extractor --version 2>&1) | sed 's/^.*Bismark Extractor Version: v//; s/ .*\$//')
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.bedGraph.gz
    touch ${prefix}.bismark.cov.gz
    touch ${prefix}_splitting_report.txt
    touch versions.yml
    """
}
