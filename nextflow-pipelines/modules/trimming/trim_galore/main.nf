process TRIM_GALORE {
    tag "$meta.id"
    label 'process_low'

    container params.containers.methylation

    input:
    tuple val(meta), path(reads)

    output:
    tuple val(meta), path("*_trimmed.fq.gz"), emit: reads
    path "*_trimming_report.txt"            , emit: report
    path "versions.yml"                     , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    
    // Fix libfreetype issue
    """
    ln -sf /opt/conda/lib/libfreetype.so.6.20.4 libfreetype.so.6 || true
    export LD_LIBRARY_PATH=\$PWD:/opt/conda/lib:\$LD_LIBRARY_PATH

    trim_galore \\
        $args \\
        --cores ${task.cpus} \\
        --paired \\
        --basename ${prefix} \\
        ${reads[0]} \\
        ${reads[1]}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        trimgalore: \$(echo \$(trim_galore --version 2>&1) | sed 's/^.*version //; s/ .*\$//')
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}_val_1.fq.gz
    touch ${prefix}_val_2.fq.gz
    touch ${prefix}_R1_trimming_report.txt
    touch ${prefix}_R2_trimming_report.txt
    touch versions.yml
    """
}
