process KRAKEN2 {
    tag "$meta.id"
    label 'process_high'

    container params.containers.metagenomics

    input:
    tuple val(meta), path(reads)
    path db

    output:
    tuple val(meta), path("*.kraken2.report"), emit: report
    tuple val(meta), path("*.kraken2.output"), emit: output
    path "versions.yml"                       , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    def paired = meta.single_end ? "" : "--paired"
    
    """
    kraken2 \\
        --db ${db} \\
        --threads ${task.cpus} \\
        $paired \\
        --report ${prefix}.kraken2.report \\
        --output ${prefix}.kraken2.output \\
        $args \\
        ${reads.join(' ')}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        kraken2: \$(echo \$(kraken2 --version 2>&1) | sed 's/^.*Kraken version //; s/ .*\$//')
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.kraken2.report
    touch ${prefix}.kraken2.output
    touch versions.yml
    """
}
