process BWAMEM_ALIGN {
    tag "$meta.id"
    label 'process_high'

    container params.containers.dnaseq

    input:
    tuple val(meta), path(reads)
    tuple path(index), path(index_files)

    output:
    tuple val(meta), path("*.bam"), path("*.bai"), emit: bam
    path "versions.yml"                           , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    def read_group = "@RG\\tID:${meta.id}\\tSM:${meta.id}\\tPL:ILLUMINA\\tLB:${meta.id}"
    
    // Fix libfreetype issue
    """
    ln -sf /opt/conda/lib/libfreetype.so.6.20.4 libfreetype.so.6 || true
    export LD_LIBRARY_PATH=\$PWD:/opt/conda/lib:\$LD_LIBRARY_PATH

    bwa mem \\
        -t ${task.cpus} \\
        -R '${read_group}' \\
        $args \\
        ${index} \\
        ${reads[0]} \\
        ${reads[1]} \\
        | samtools sort -@ ${task.cpus} -o ${prefix}.bam -

    samtools index ${prefix}.bam

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bwa: \$(echo \$(bwa 2>&1) | sed 's/^.*Version: //; s/ .*\$//')
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
