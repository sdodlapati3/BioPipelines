/*
 * HISAT2 Alignment Module
 * 
 * HISAT2 - Graph-based alignment for RNA-seq
 * Fast and memory-efficient splice-aware aligner
 * Uses existing rna-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * HISAT2 Index - Build reference index
 */
process HISAT2_INDEX {
    tag "hisat2_index_${reference.baseName}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/reference/hisat2", mode: 'copy'
    
    cpus params.hisat2?.index_cpus ?: 8
    memory params.hisat2?.index_memory ?: '32.GB'
    
    input:
    path reference
    
    output:
    path "hisat2_index.*", emit: index
    
    script:
    """
    hisat2-build \\
        -p ${task.cpus} \\
        ${reference} \\
        hisat2_index
    """
}

/*
 * HISAT2 Index with Splice Sites - Build index with known splice sites
 */
process HISAT2_INDEX_SPLICED {
    tag "hisat2_spliced_${reference.baseName}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/reference/hisat2", mode: 'copy'
    
    cpus params.hisat2?.index_cpus ?: 8
    memory params.hisat2?.index_memory ?: '32.GB'
    
    input:
    path reference
    path gtf
    
    output:
    path "hisat2_index.*", emit: index
    path "splice_sites.txt", emit: splice_sites
    path "exons.txt", emit: exons
    
    script:
    """
    # Extract splice sites and exons from GTF
    hisat2_extract_splice_sites.py ${gtf} > splice_sites.txt
    hisat2_extract_exons.py ${gtf} > exons.txt
    
    # Build index with splice site information
    hisat2-build \\
        -p ${task.cpus} \\
        --ss splice_sites.txt \\
        --exon exons.txt \\
        ${reference} \\
        hisat2_index
    """
}

/*
 * HISAT2 Alignment
 */
process HISAT2_ALIGN {
    tag "hisat2_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/alignments", mode: 'copy',
        pattern: "*.bam"
    publishDir "${params.outdir}/alignments/stats", mode: 'copy',
        pattern: "*.{txt,summary}"
    
    cpus params.hisat2?.cpus ?: 8
    memory params.hisat2?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(reads)
    path index
    val strand_specific  // optional: "FR", "RF", or "unstranded"
    
    output:
    tuple val(sample_id), path("${sample_id}.bam"), emit: bam
    tuple val(sample_id), path("${sample_id}.bam.bai"), emit: bai
    path "${sample_id}.hisat2.summary.txt", emit: summary
    path "${sample_id}.flagstat.txt", emit: stats
    
    script:
    def strandedness = ""
    if (strand_specific == "FR") {
        strandedness = "--rna-strandness FR"
    } else if (strand_specific == "RF") {
        strandedness = "--rna-strandness RF"
    }
    
    def index_base = index[0].toString().replaceAll(/\..*$/, '')
    
    if (reads instanceof List) {
        // Paired-end
        """
        hisat2 \\
            -p ${task.cpus} \\
            -x ${index_base} \\
            -1 ${reads[0]} \\
            -2 ${reads[1]} \\
            ${strandedness} \\
            --summary-file ${sample_id}.hisat2.summary.txt \\
            | samtools view -bS - \\
            | samtools sort -@ ${task.cpus} -o ${sample_id}.bam -
        
        samtools index ${sample_id}.bam
        samtools flagstat ${sample_id}.bam > ${sample_id}.flagstat.txt
        """
    } else {
        // Single-end
        """
        hisat2 \\
            -p ${task.cpus} \\
            -x ${index_base} \\
            -U ${reads} \\
            ${strandedness} \\
            --summary-file ${sample_id}.hisat2.summary.txt \\
            | samtools view -bS - \\
            | samtools sort -@ ${task.cpus} -o ${sample_id}.bam -
        
        samtools index ${sample_id}.bam
        samtools flagstat ${sample_id}.bam > ${sample_id}.flagstat.txt
        """
    }
}

/*
 * HISAT2 with Novel Splice Sites - Align and discover novel splice junctions
 */
process HISAT2_ALIGN_NOVEL {
    tag "hisat2_novel_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/alignments", mode: 'copy',
        pattern: "*.bam"
    publishDir "${params.outdir}/alignments/novel_junctions", mode: 'copy',
        pattern: "*.txt"
    
    cpus params.hisat2?.cpus ?: 8
    memory params.hisat2?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(reads)
    path index
    val strand_specific
    
    output:
    tuple val(sample_id), path("${sample_id}.bam"), emit: bam
    tuple val(sample_id), path("${sample_id}.bam.bai"), emit: bai
    path "${sample_id}.novel_splicesite.txt", emit: novel_junctions
    path "${sample_id}.hisat2.summary.txt", emit: summary
    
    script:
    def strandedness = ""
    if (strand_specific == "FR") {
        strandedness = "--rna-strandness FR"
    } else if (strand_specific == "RF") {
        strandedness = "--rna-strandness RF"
    }
    
    def index_base = index[0].toString().replaceAll(/\..*$/, '')
    
    if (reads instanceof List) {
        """
        hisat2 \\
            -p ${task.cpus} \\
            -x ${index_base} \\
            -1 ${reads[0]} \\
            -2 ${reads[1]} \\
            ${strandedness} \\
            --novel-splicesite-outfile ${sample_id}.novel_splicesite.txt \\
            --summary-file ${sample_id}.hisat2.summary.txt \\
            | samtools view -bS - \\
            | samtools sort -@ ${task.cpus} -o ${sample_id}.bam -
        
        samtools index ${sample_id}.bam
        """
    } else {
        """
        hisat2 \\
            -p ${task.cpus} \\
            -x ${index_base} \\
            -U ${reads} \\
            ${strandedness} \\
            --novel-splicesite-outfile ${sample_id}.novel_splicesite.txt \\
            --summary-file ${sample_id}.hisat2.summary.txt \\
            | samtools view -bS - \\
            | samtools sort -@ ${task.cpus} -o ${sample_id}.bam -
        
        samtools index ${sample_id}.bam
        """
    }
}

/*
 * Workflow: Complete HISAT2 pipeline with indexing
 */
workflow HISAT2_PIPELINE {
    take:
    reads_ch          // channel: [ val(sample_id), path(reads) ]
    reference         // path: reference genome fasta
    gtf              // path: annotation GTF (optional)
    strand_specific   // val: "FR", "RF", or "unstranded"
    
    main:
    // Build index with splice site information if GTF provided
    if (gtf) {
        HISAT2_INDEX_SPLICED(reference, gtf)
        index_ch = HISAT2_INDEX_SPLICED.out.index
    } else {
        HISAT2_INDEX(reference)
        index_ch = HISAT2_INDEX.out.index
    }
    
    // Align reads
    HISAT2_ALIGN(reads_ch, index_ch, strand_specific)
    
    emit:
    bam = HISAT2_ALIGN.out.bam
    bai = HISAT2_ALIGN.out.bai
    summary = HISAT2_ALIGN.out.summary
    stats = HISAT2_ALIGN.out.stats
}
