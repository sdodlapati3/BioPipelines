#!/usr/bin/env nextflow

/*
 * STAR - Spliced Transcripts Alignment to a Reference
 *
 * Aligns RNA-seq reads to reference genome using STAR.
 * Simplified version based on nf-core/star/align module.
 */

process STAR_ALIGN {
    tag "${meta.id}"
    label 'process_high'
    
    // Use RNA-seq container (contains STAR and samtools)
    container "${params.containers.rnaseq}"
    
    input:
    tuple val(meta), path(reads)
    path index
    
    output:
    tuple val(meta), path('*.Aligned.sortedByCoord.out.bam'), emit: bam
    tuple val(meta), path('*.Log.final.out')                , emit: log_final
    tuple val(meta), path('*.Log.out')                      , emit: log_out
    tuple val(meta), path('*.SJ.out.tab')                   , emit: spl_junc_tab
    path "versions.yml"                                     , emit: versions
    
    when:
    task.ext.when == null || task.ext.when
    
    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    def reads_cmd = meta.single_end ? reads : "${reads[0]} ${reads[1]}"
    
    """
    # Fix missing libfreetype.so.6 symlink (same as FastQC)
    ln -sf /opt/conda/lib/libfreetype.so.6.20.4 libfreetype.so.6
    export LD_LIBRARY_PATH=\$PWD:/opt/conda/lib:\${LD_LIBRARY_PATH:-}
    
    STAR \\
        --genomeDir $index \\
        --readFilesIn $reads_cmd \\
        --readFilesCommand zcat \\
        --runThreadN ${task.cpus} \\
        --outFileNamePrefix ${prefix}. \\
        --outSAMtype BAM SortedByCoordinate \\
        --outSAMattributes NH HI AS NM MD \\
        --outSAMattrRGline ID:${prefix} SM:${prefix} PL:ILLUMINA \\
        --quantMode GeneCounts \\
        $args
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        star: \$(STAR --version | sed -e "s/STAR_//g")
    END_VERSIONS
    """
    
    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.Aligned.sortedByCoord.out.bam
    touch ${prefix}.Log.final.out
    touch ${prefix}.Log.out
    touch ${prefix}.SJ.out.tab
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        star: \$(STAR --version 2>&1 | sed -e "s/STAR_//g" || echo "2.7.11a")
    END_VERSIONS
    """
}
