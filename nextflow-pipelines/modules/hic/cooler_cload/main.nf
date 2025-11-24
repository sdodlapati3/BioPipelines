process COOLER_CLOAD {
    tag "$meta.id"
    label 'process_medium'

    container params.containers.hic

    input:
    tuple val(meta), path(pairs)
    path chrom_sizes
    val resolution

    output:
    tuple val(meta), path("*.cool"), emit: cool
    path "versions.yml"            , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    
    """
    cooler cload pairs \\
        $args \\
        --assembly hg38 \\
        ${chrom_sizes}:${resolution} \\
        ${pairs} \\
        ${prefix}_${resolution}.cool

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        cooler: \$(cooler --version 2>&1)
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}_${resolution}.cool
    touch versions.yml
    """
}
