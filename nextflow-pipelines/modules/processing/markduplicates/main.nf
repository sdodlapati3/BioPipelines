process PICARD_MARKDUPLICATES {
    tag "$meta.id"
    label 'process_medium'

    container params.containers.dnaseq

    input:
    tuple val(meta), path(bam), path(bai)

    output:
    tuple val(meta), path("*.marked.bam"), path("*.marked.bam.bai"), emit: bam
    path "*.metrics.txt"                                            , emit: metrics
    path "versions.yml"                                             , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    def avail_mem = task.memory ? "-Xmx${task.memory.toGiga()}g" : ''
    
    """
    picard ${avail_mem} MarkDuplicates \\
        $args \\
        INPUT=${bam} \\
        OUTPUT=${prefix}.marked.bam \\
        METRICS_FILE=${prefix}.marked.metrics.txt \\
        CREATE_INDEX=true \\
        VALIDATION_STRINGENCY=LENIENT \\
        TMP_DIR=.

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        picard: \$(picard MarkDuplicates --version 2>&1 | grep -o 'Version:.*' | sed 's/Version://g')
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.marked.bam
    touch ${prefix}.marked.bam.bai
    touch ${prefix}.marked.metrics.txt
    touch versions.yml
    """
}
