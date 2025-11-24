process BISMARK_ALIGN {
    tag "$meta.id"
    label 'process_high'

    container params.containers.methylation

    input:
    tuple val(meta), path(reads)
    path index

    output:
    tuple val(meta), path("*.bam"), path("*.bam.bai"), emit: bam
    path "*_report.txt"                               , emit: report
    path "versions.yml"                               , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    
    """
    bismark \\
        $args \\
        --parallel ${task.cpus} \\
        --genome ${index} \\
        -1 ${reads[0]} \\
        -2 ${reads[1]}

    # Rename Bismark's default output to our expected name
    samtools sort -@ ${task.cpus} -o ${prefix}.bam *_pe.bam
    samtools index ${prefix}.bam
    rm *_pe.bam

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bismark: \$(echo \$(bismark --version 2>&1) | sed 's/^.*Bismark Version: v//; s/ .*\$//')
        samtools: \$(echo \$(samtools --version 2>&1) | sed 's/^.*samtools //; s/ .*\$//')
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.bam
    touch ${prefix}.bam.bai
    touch ${prefix}_report.txt
    touch versions.yml
    """
}
