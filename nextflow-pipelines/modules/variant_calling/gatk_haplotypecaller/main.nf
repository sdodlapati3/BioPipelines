process GATK_HAPLOTYPECALLER {
    tag "$meta.id"
    label 'process_high'

    container params.containers.dnaseq

    input:
    tuple val(meta), path(bam), path(bai)
    path fasta
    path fasta_fai
    path dict

    output:
    tuple val(meta), path("*.vcf.gz"), path("*.vcf.gz.tbi"), emit: vcf
    path "versions.yml"                                    , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    def avail_mem = task.memory ? "-Xmx${task.memory.toGiga()}g" : ''
    
    """
    gatk ${avail_mem} HaplotypeCaller \\
        $args \\
        --reference ${fasta} \\
        --input ${bam} \\
        --output ${prefix}.vcf.gz \\
        --native-pair-hmm-threads ${task.cpus} \\
        --tmp-dir .

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        gatk: \$(echo \$(gatk --version 2>&1) | sed 's/^.*(GATK) v//; s/ .*\$//')
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.vcf.gz
    touch ${prefix}.vcf.gz.tbi
    touch versions.yml
    """
}
