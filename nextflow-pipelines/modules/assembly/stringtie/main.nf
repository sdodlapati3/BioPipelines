/*
 * StringTie Module
 * 
 * StringTie - Transcript assembly and quantification
 * Reference-guided or de novo transcript assembly from RNA-seq alignments
 * Uses existing rna-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * StringTie Assembly - Assemble transcripts from alignments
 */
process STRINGTIE_ASSEMBLE {
    tag "stringtie_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/assembly/stringtie", mode: 'copy'
    
    cpus params.stringtie?.cpus ?: 8
    memory params.stringtie?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(bam)
    path gtf
    val strandedness  // "none", "forward", "reverse"
    
    output:
    tuple val(sample_id), path("${sample_id}.gtf"), emit: gtf
    tuple val(sample_id), path("${sample_id}_abundance.txt"), emit: abundance
    path "${sample_id}_gene_abundance.txt", emit: gene_abundance
    
    script:
    def strand_opt = ""
    if (strandedness == "forward") {
        strand_opt = "--fr"
    } else if (strandedness == "reverse") {
        strand_opt = "--rf"
    }
    
    def guide_opt = gtf ? "-G ${gtf}" : ""
    
    """
    stringtie \\
        ${bam} \\
        ${guide_opt} \\
        -o ${sample_id}.gtf \\
        -A ${sample_id}_gene_abundance.txt \\
        -p ${task.cpus} \\
        ${strand_opt} \\
        -l ${sample_id} \\
        -e
    
    # Extract abundance for downstream use
    grep -v "^#" ${sample_id}.gtf | \\
        awk -F'\t' 'BEGIN{OFS="\t"}{print \$1,\$4,\$5,\$7,\$9}' > ${sample_id}_abundance.txt
    """
}

/*
 * StringTie Merge - Merge transcripts from multiple samples
 */
process STRINGTIE_MERGE {
    tag "stringtie_merge"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/assembly/stringtie", mode: 'copy'
    
    cpus params.stringtie?.cpus ?: 8
    memory params.stringtie?.memory ?: '16.GB'
    
    input:
    path gtf_files
    path reference_gtf
    
    output:
    path "merged.gtf", emit: merged_gtf
    
    script:
    """
    # Create list of GTF files
    ls -1 *.gtf | grep -v "^merged.gtf" > gtf_list.txt
    
    stringtie --merge \\
        -G ${reference_gtf} \\
        -o merged.gtf \\
        -p ${task.cpus} \\
        gtf_list.txt
    """
}

/*
 * StringTie Quantify - Re-quantify with merged assembly
 */
process STRINGTIE_QUANTIFY {
    tag "stringtie_quant_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/quantification/stringtie", mode: 'copy'
    
    cpus params.stringtie?.cpus ?: 8
    memory params.stringtie?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(bam)
    path merged_gtf
    val strandedness
    
    output:
    tuple val(sample_id), path("${sample_id}.gtf"), emit: gtf
    tuple val(sample_id), path("${sample_id}_abundance.txt"), emit: abundance
    path "${sample_id}_gene_abundance.txt", emit: gene_abundance
    path "${sample_id}.ballgown", emit: ballgown
    
    script:
    def strand_opt = ""
    if (strandedness == "forward") {
        strand_opt = "--fr"
    } else if (strandedness == "reverse") {
        strand_opt = "--rf"
    }
    
    """
    stringtie \\
        ${bam} \\
        -G ${merged_gtf} \\
        -o ${sample_id}.gtf \\
        -A ${sample_id}_gene_abundance.txt \\
        -p ${task.cpus} \\
        ${strand_opt} \\
        -l ${sample_id} \\
        -e \\
        -B \\
        -b ${sample_id}.ballgown
    
    # Extract abundance
    grep -v "^#" ${sample_id}.gtf | \\
        awk -F'\t' 'BEGIN{OFS="\t"}{print \$1,\$4,\$5,\$7,\$9}' > ${sample_id}_abundance.txt
    """
}

/*
 * StringTie Abundance to Count Matrix
 */
process STRINGTIE_MATRIX {
    tag "stringtie_matrix"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/quantification/stringtie", mode: 'copy'
    
    input:
    path abundance_files
    
    output:
    path "gene_count_matrix.csv", emit: gene_matrix
    path "transcript_count_matrix.csv", emit: transcript_matrix
    
    script:
    """
    #!/usr/bin/env python3
    
    import pandas as pd
    import glob
    
    # Collect all gene abundance files
    files = glob.glob("*_gene_abundance.txt")
    
    # Read and merge
    dfs = []
    for f in files:
        sample_id = f.replace("_gene_abundance.txt", "")
        df = pd.read_csv(f, sep='\t')
        df = df[['Gene ID', 'TPM', 'FPKM', 'Coverage']]
        df = df.rename(columns={'TPM': f'{sample_id}_TPM', 
                                 'FPKM': f'{sample_id}_FPKM',
                                 'Coverage': f'{sample_id}_Coverage'})
        if len(dfs) == 0:
            dfs.append(df)
        else:
            dfs.append(df[df.columns[1:]])
    
    # Combine
    result = pd.concat([dfs[0]] + [df for df in dfs[1:]], axis=1)
    result.to_csv("gene_count_matrix.csv", index=False)
    
    # Create TPM-only matrix for downstream analysis
    tpm_cols = [col for col in result.columns if '_TPM' in col]
    tpm_matrix = result[['Gene ID'] + tpm_cols]
    tpm_matrix.to_csv("gene_tpm_matrix.csv", index=False)
    
    # Placeholder for transcript matrix (would need transcript abundance files)
    result.to_csv("transcript_count_matrix.csv", index=False)
    """
}

/*
 * Workflow: Complete StringTie pipeline
 */
workflow STRINGTIE_PIPELINE {
    take:
    bam_ch         // channel: [ val(sample_id), path(bam) ]
    reference_gtf  // path: reference annotation GTF
    strandedness   // val: "none", "forward", "reverse"
    
    main:
    // Initial assembly for each sample
    STRINGTIE_ASSEMBLE(bam_ch, reference_gtf, strandedness)
    
    // Merge all assemblies
    all_gtfs = STRINGTIE_ASSEMBLE.out.gtf.map { it[1] }.collect()
    STRINGTIE_MERGE(all_gtfs, reference_gtf)
    
    // Re-quantify with merged assembly
    STRINGTIE_QUANTIFY(
        bam_ch,
        STRINGTIE_MERGE.out.merged_gtf,
        strandedness
    )
    
    // Generate count matrix
    all_abundances = STRINGTIE_QUANTIFY.out.gene_abundance.collect()
    STRINGTIE_MATRIX(all_abundances)
    
    emit:
    merged_gtf = STRINGTIE_MERGE.out.merged_gtf
    gene_matrix = STRINGTIE_MATRIX.out.gene_matrix
    transcript_matrix = STRINGTIE_MATRIX.out.transcript_matrix
    abundances = STRINGTIE_QUANTIFY.out.abundance
}
