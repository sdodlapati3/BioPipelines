process PAIRTOOLS_PARSE {
    tag "$meta.id"
    label 'process_medium'

    container params.containers.hic

    input:
    tuple val(meta), path(bam), path(bai)
    path chrom_sizes

    output:
    tuple val(meta), path("*.pairs.gz"), emit: pairs
    path "*.stats.txt"                  , emit: stats
    path "versions.yml"                 , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    
    """
    pairtools parse \\
        $args \\
        --chroms-path ${chrom_sizes} \\
        --output ${prefix}.pairs.gz \\
        --output-stats ${prefix}.stats.txt \\
        ${bam}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        pairtools: \$(pairtools --version 2>&1 | sed 's/pairtools, version //')
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.pairs.gz
    touch ${prefix}.stats.txt
    touch versions.yml
    """
}
