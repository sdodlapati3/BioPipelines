#!/usr/bin/env nextflow

nextflow.enable.dsl=2

/*
 * STAR Alignment Module (Tier 2 Container)
 * ==========================================
 * Performs splice-aware RNA-seq alignment using STAR
 * 
 * Container: alignment_short_read.sif (Tier 2)
 * Tool: STAR 2.7.11a
 */

process STAR_ALIGN {
    tag "${sample_id}"
    container "/scratch/sdodl001/BioPipelines/containers/tier2/alignment_short_read.sif"
    
    // Resource allocation
    cpus 8
    memory '32 GB'
    time '2h'
    
    // Publish results
    publishDir "${params.outdir}/star/${sample_id}", mode: 'copy', pattern: "*.{bam,bai,log,out,tab}"
    
    input:
    tuple val(sample_id), path(reads)
    path genome_index
    
    output:
    tuple val(sample_id), path("${sample_id}_Aligned.sortedByCoord.out.bam"), emit: bam
    tuple val(sample_id), path("${sample_id}_Aligned.sortedByCoord.out.bam.bai"), emit: bai
    path "${sample_id}_Log.final.out", emit: log
    path "${sample_id}_Log.out", emit: log_out
    path "${sample_id}_Log.progress.out", emit: log_progress
    path "${sample_id}_SJ.out.tab", emit: junctions, optional: true
    
    script:
    def read1 = reads[0]
    def read2 = reads.size() > 1 ? reads[1] : ""
    def read_files = read2 ? "--readFilesIn ${read1} ${read2}" : "--readFilesIn ${read1}"
    
    """
    # Set number of threads
    export OMP_NUM_THREADS=${task.cpus}
    
    # Run STAR alignment
    STAR \
        --runMode alignReads \
        --genomeDir ${genome_index} \
        ${read_files} \
        --readFilesCommand zcat \
        --outFileNamePrefix ${sample_id}_ \
        --outSAMtype BAM SortedByCoordinate \
        --outSAMunmapped Within \
        --outSAMattributes Standard \
        --runThreadN ${task.cpus} \
        --limitBAMsortRAM ${task.memory.toBytes()}
    
    # Index BAM file
    samtools index -@ ${task.cpus} ${sample_id}_Aligned.sortedByCoord.out.bam
    """
}

/*
 * STAR Index Generation
 * ======================
 * Generate STAR genome index from FASTA + GTF
 */

process STAR_INDEX {
    tag "${genome_name}"
    container "/scratch/sdodl001/BioPipelines/containers/tier2/alignment_short_read.sif"
    
    cpus 8
    memory '32 GB'
    time '2h'
    
    publishDir "${params.outdir}/star_index", mode: 'copy'
    
    input:
    path genome_fasta
    path genome_gtf
    val genome_name
    
    output:
    path "star_index_${genome_name}", emit: index
    
    script:
    """
    mkdir -p star_index_${genome_name}
    
    STAR \
        --runMode genomeGenerate \
        --genomeDir star_index_${genome_name} \
        --genomeFastaFiles ${genome_fasta} \
        --sjdbGTFfile ${genome_gtf} \
        --sjdbOverhang 100 \
        --runThreadN ${task.cpus} \
        --limitGenomeGenerateRAM ${task.memory.toBytes()}
    """
}

/*
 * STAR Solo (scRNA-seq)
 * ======================
 * Process 10x Genomics scRNA-seq data
 */

process STARSOLO {
    tag "${sample_id}"
    container "/scratch/sdodl001/BioPipelines/containers/tier2/alignment_short_read.sif"
    
    cpus 8
    memory '32 GB'
    time '3h'
    
    publishDir "${params.outdir}/starsolo/${sample_id}", mode: 'copy'
    
    input:
    tuple val(sample_id), path(read1), path(read2)
    path genome_index
    path whitelist
    
    output:
    tuple val(sample_id), path("${sample_id}_Solo.out"), emit: solo_out
    path "${sample_id}_Aligned.sortedByCoord.out.bam", emit: bam
    path "${sample_id}_Log.final.out", emit: log
    
    script:
    """
    STAR \
        --runMode alignReads \
        --genomeDir ${genome_index} \
        --readFilesIn ${read2} ${read1} \
        --readFilesCommand zcat \
        --outFileNamePrefix ${sample_id}_ \
        --outSAMtype BAM SortedByCoordinate \
        --runThreadN ${task.cpus} \
        --soloType CB_UMI_Simple \
        --soloCBwhitelist ${whitelist} \
        --soloCBstart 1 \
        --soloCBlen 16 \
        --soloUMIstart 17 \
        --soloUMIlen 12 \
        --soloFeatures Gene GeneFull \
        --soloOutFileNames ${sample_id}_Solo.out/ features.tsv barcodes.tsv matrix.mtx
    """
}
