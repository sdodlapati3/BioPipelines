process BRACKEN {
    tag "$meta.id"
    label 'process_low'

    container params.containers.metagenomics

    input:
    tuple val(meta), path(report)
    path db

    output:
    tuple val(meta), path("*.bracken"), emit: output
    path "versions.yml"                , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    def read_len = task.ext.read_len ?: 150
    def level = task.ext.level ?: 'S'  // Species level by default
    
    """
    bracken \\
        -d ${db} \\
        -i ${report} \\
        -o ${prefix}.bracken \\
        -r ${read_len} \\
        -l ${level} \\
        $args

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bracken: \$(echo \$(bracken -v 2>&1) | sed 's/^.*Bracken version //; s/ .*\$//')
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.bracken
    touch versions.yml
    """
}
