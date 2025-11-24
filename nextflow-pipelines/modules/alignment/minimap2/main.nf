process MINIMAP2 {
    tag "$meta.id"
    label 'process_high'

    container params.containers.longread

    input:
    tuple val(meta), path(reads)
    path reference

    output:
    tuple val(meta), path("*.bam"), path("*.bam.bai"), emit: bam
    path "versions.yml"                               , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: '-ax map-ont'  // Oxford Nanopore by default
    def prefix = task.ext.prefix ?: "${meta.id}"
    
    """
    minimap2 \\
        -t ${task.cpus} \\
        $args \\
        ${reference} \\
        ${reads} \\
        | samtools sort -@ ${task.cpus} -o ${prefix}.bam -

    samtools index ${prefix}.bam

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        minimap2: \$(minimap2 --version 2>&1)
        samtools: \$(echo \$(samtools --version 2>&1) | sed 's/^.*samtools //; s/ .*\$//')
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.bam
    touch ${prefix}.bam.bai
    touch versions.yml
    """
}
