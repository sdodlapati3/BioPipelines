/*
 * GSEA Module
 * 
 * GSEA - Gene Set Enrichment Analysis
 * Determines whether predefined gene sets show significant differences
 * Uses existing base container with R/Python
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * GSEA analysis (using GSEApy Python implementation)
 */
process GSEA_ANALYSIS {
    tag "gsea_${comparison}"
    container "${params.containers.base}"
    
    publishDir "${params.outdir}/analysis/gsea", mode: 'copy'
    
    cpus params.gsea?.cpus ?: 4
    memory params.gsea?.memory ?: '16.GB'
    
    input:
    tuple val(comparison), path(expression_data)
    path gene_sets
    val permutation_num
    
    output:
    path "${comparison}_gsea", emit: results_dir
    path "${comparison}_gsea/gseapy.gene_sets.report.csv", emit: report
    path "${comparison}_gsea/*.pdf", emit: plots
    
    script:
    def permutations = permutation_num ?: 1000
    def method = params.gsea?.method ?: "signal_to_noise"
    
    """
    #!/usr/bin/env python3
    
    import gseapy as gp
    import pandas as pd
    
    # Load expression data
    data = pd.read_csv('${expression_data}', sep='\\t', index_col=0)
    
    # Run GSEA
    gs_res = gp.gsea(
        data=data,
        gene_sets='${gene_sets}',
        cls='${comparison}',
        permutation_num=${permutations},
        outdir='${comparison}_gsea',
        method='${method}',
        processes=${task.cpus},
        format='pdf'
    )
    """
}

/*
 * Gene Set Enrichment using preranked list
 */
process GSEA_PRERANKED {
    tag "gsea_preranked_${comparison}"
    container "${params.containers.base}"
    
    publishDir "${params.outdir}/analysis/gsea", mode: 'copy'
    
    cpus params.gsea?.cpus ?: 4
    memory params.gsea?.memory ?: '16.GB'
    
    input:
    tuple val(comparison), path(ranked_genes)
    path gene_sets
    val permutation_num
    
    output:
    path "${comparison}_preranked", emit: results_dir
    path "${comparison}_preranked/gseapy.gene_sets.report.csv", emit: report
    path "${comparison}_preranked/*.pdf", emit: plots
    
    script:
    def permutations = permutation_num ?: 1000
    
    """
    #!/usr/bin/env python3
    
    import gseapy as gp
    import pandas as pd
    
    # Load ranked gene list
    rnk = pd.read_csv('${ranked_genes}', sep='\\t', header=None, names=['gene', 'rank'])
    
    # Run preranked GSEA
    pre_res = gp.prerank(
        rnk=rnk,
        gene_sets='${gene_sets}',
        permutation_num=${permutations},
        outdir='${comparison}_preranked',
        processes=${task.cpus},
        format='pdf'
    )
    """
}

/*
 * Enrichr - Over-representation analysis
 */
process ENRICHR {
    tag "enrichr_${sample_id}"
    container "${params.containers.base}"
    
    publishDir "${params.outdir}/analysis/enrichr", mode: 'copy'
    
    memory params.enrichr?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(gene_list)
    val gene_sets
    
    output:
    tuple val(sample_id), path("${sample_id}_enrichr.csv"), emit: results
    path "${sample_id}_enrichr_*.pdf", emit: plots
    
    script:
    def libraries = gene_sets instanceof List ? gene_sets : [gene_sets]
    def lib_str = libraries.collect { "'${it}'" }.join(', ')
    
    """
    #!/usr/bin/env python3
    
    import gseapy as gp
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Load gene list
    with open('${gene_list}', 'r') as f:
        genes = [line.strip() for line in f]
    
    # Run Enrichr
    enr = gp.enrichr(
        gene_list=genes,
        gene_sets=[${lib_str}],
        organism='Human',
        outdir='${sample_id}_enrichr'
    )
    
    # Save results
    results = enr.results
    results.to_csv('${sample_id}_enrichr.csv', index=False)
    
    # Create plots
    for library in [${lib_str}]:
        lib_results = results[results['Gene_set'].str.contains(library)]
        if len(lib_results) > 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            top_results = lib_results.head(20)
            ax.barh(range(len(top_results)), -top_results['Adjusted P-value'].apply(lambda x: -np.log10(x)))
            ax.set_yticks(range(len(top_results)))
            ax.set_yticklabels(top_results['Term'])
            ax.set_xlabel('-log10(Adjusted P-value)')
            ax.set_title(f'Enrichr Results: {library}')
            plt.tight_layout()
            plt.savefig(f'${sample_id}_enrichr_{library}.pdf')
            plt.close()
    """
}

/*
 * Workflow: GSEA enrichment analysis
 */
workflow GSEA_PIPELINE {
    take:
    expression_ch  // channel: [ val(comparison), path(expression_data) ]
    gene_sets      // path: gene set database
    permutations   // val: number of permutations
    
    main:
    GSEA_ANALYSIS(expression_ch, gene_sets, permutations)
    
    emit:
    results = GSEA_ANALYSIS.out.results_dir
    report = GSEA_ANALYSIS.out.report
}
