/*
 * HOMER Module
 * ============
 * Hypergeometric Optimization of Motif EnRichment
 * 
 * Container: atac-seq, chip-seq
 */

process HOMER_MAKETAGDIRECTORY {
    tag "$meta.id"
    label 'process_medium'
    
    container "${params.containers.atacseq}"
    
    input:
    tuple val(meta), path(bam)
    path genome_fasta
    
    output:
    tuple val(meta), path("*_tagdir"), emit: tagdir
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    makeTagDirectory \\
        ${prefix}_tagdir \\
        $bam \\
        -genome $genome_fasta \\
        $args
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        homer: \$(analyzeRepeats.pl --version 2>&1 | head -1 | sed 's/.*v//')
    END_VERSIONS
    """
}

process HOMER_FINDPEAKS {
    tag "$meta.id"
    label 'process_medium'
    
    container "${params.containers.atacseq}"
    
    input:
    tuple val(meta), path(tagdir)
    path control_tagdir
    
    output:
    tuple val(meta), path("*.peaks.txt"), emit: peaks
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: "-style factor"
    def control = control_tagdir ? "-i $control_tagdir" : ""
    """
    findPeaks \\
        $tagdir \\
        $control \\
        -o ${prefix}.peaks.txt \\
        $args
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        homer: \$(analyzeRepeats.pl --version 2>&1 | head -1 | sed 's/.*v//')
    END_VERSIONS
    """
}

process HOMER_ANNOTATEPEAKS {
    tag "$meta.id"
    label 'process_medium'
    
    container "${params.containers.atacseq}"
    
    input:
    tuple val(meta), path(peaks)
    path gtf
    path genome_fasta
    
    output:
    tuple val(meta), path("*.annotated.txt"), emit: annotated
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    annotatePeaks.pl \\
        $peaks \\
        $genome_fasta \\
        -gtf $gtf \\
        $args \\
        > ${prefix}.annotated.txt
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        homer: \$(analyzeRepeats.pl --version 2>&1 | head -1 | sed 's/.*v//')
    END_VERSIONS
    """
}

process HOMER_FINDMOTIFSGENOME {
    tag "$meta.id"
    label 'process_high'
    
    container "${params.containers.atacseq}"
    
    input:
    tuple val(meta), path(peaks)
    path genome_fasta
    
    output:
    tuple val(meta), path("*_motifs"), emit: motifs
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: "-size 200 -mask"
    """
    findMotifsGenome.pl \\
        $peaks \\
        $genome_fasta \\
        ${prefix}_motifs \\
        -p $task.cpus \\
        $args
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        homer: \$(analyzeRepeats.pl --version 2>&1 | head -1 | sed 's/.*v//')
    END_VERSIONS
    """
}

process HOMER_POS2BED {
    tag "$meta.id"
    label 'process_low'
    
    container "${params.containers.atacseq}"
    
    input:
    tuple val(meta), path(peaks)
    
    output:
    tuple val(meta), path("*.bed"), emit: bed
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    pos2bed.pl $peaks > ${prefix}.bed
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        homer: \$(analyzeRepeats.pl --version 2>&1 | head -1 | sed 's/.*v//')
    END_VERSIONS
    """
}

process HOMER_MAKEUCSCFILE {
    tag "$meta.id"
    label 'process_medium'
    
    container "${params.containers.atacseq}"
    
    input:
    tuple val(meta), path(tagdir)
    
    output:
    tuple val(meta), path("*.bedGraph.gz"), emit: bedgraph
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    makeUCSCfile \\
        $tagdir \\
        -o ${prefix}.bedGraph \\
        $args
    
    gzip ${prefix}.bedGraph
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        homer: \$(analyzeRepeats.pl --version 2>&1 | head -1 | sed 's/.*v//')
    END_VERSIONS
    """
}

process HOMER_GETDIFFERENTIALPEAKS {
    tag "$meta.id"
    label 'process_medium'
    
    container "${params.containers.atacseq}"
    
    input:
    tuple val(meta), path(tagdir_target)
    path tagdir_background
    
    output:
    tuple val(meta), path("*.differential.txt"), emit: peaks
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    getDifferentialPeaks \\
        $tagdir_target \\
        $tagdir_background \\
        $args \\
        > ${prefix}.differential.txt
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        homer: \$(analyzeRepeats.pl --version 2>&1 | head -1 | sed 's/.*v//')
    END_VERSIONS
    """
}
