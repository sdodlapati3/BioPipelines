process STARSOLO {
    tag "$meta.id"
    label 'process_high'

    container params.containers.scrnaseq

    input:
    tuple val(meta), path(reads)
    path index
    path whitelist

    output:
    tuple val(meta), path("*Solo.out"), emit: counts
    path "*Log.final.out"             , emit: log_final
    path "versions.yml"               , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    
    // Fix libfreetype issue
    """
    ln -sf /opt/conda/lib/libfreetype.so.6.20.4 libfreetype.so.6 || true
    export LD_LIBRARY_PATH=\$PWD:/opt/conda/lib:\$LD_LIBRARY_PATH

    STAR \\
        --runMode alignReads \\
        --runThreadN ${task.cpus} \\
        --genomeDir ${index} \\
        --readFilesIn ${reads[1]} ${reads[0]} \\
        --readFilesCommand zcat \\
        --outSAMtype BAM SortedByCoordinate \\
        --outFileNamePrefix ${prefix}_ \\
        --soloType CB_UMI_Simple \\
        --soloCBwhitelist ${whitelist} \\
        --soloCBlen 16 \\
        --soloUMIlen 12 \\
        --soloFeatures Gene GeneFull \\
        --soloOutFileNames Solo.out/ genes.tsv barcodes.tsv matrix.mtx \\
        $args

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        star: \$(STAR --version | sed -e "s/STAR_//g")
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    mkdir -p ${prefix}_Solo.out/Gene/filtered
    touch ${prefix}_Solo.out/Gene/filtered/matrix.mtx
    touch ${prefix}_Log.final.out
    touch versions.yml
    """
}
