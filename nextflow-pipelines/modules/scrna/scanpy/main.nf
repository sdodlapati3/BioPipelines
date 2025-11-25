/*
 * Scanpy Module
 * 
 * Scanpy - Single-cell analysis in Python
 * Alternative to Seurat for scRNA-seq analysis
 * Uses existing scrna-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * Scanpy preprocessing and clustering
 */
process SCANPY_ANALYSIS {
    tag "scanpy_analysis"
    container "${params.containers.scrnaseq}"
    
    publishDir "${params.outdir}/scrna/scanpy", mode: 'copy'
    
    cpus params.scanpy?.cpus ?: 8
    memory params.scanpy?.memory ?: '32.GB'
    
    input:
    path count_matrix
    val min_genes
    val min_cells
    val max_mito_percent
    val resolution
    
    output:
    path "adata.h5ad", emit: adata
    path "qc_metrics.pdf", emit: qc_plot
    path "umap.pdf", emit: umap
    path "marker_genes.csv", emit: markers
    path "rank_genes.pdf", emit: rank_genes
    
    script:
    """
    #!/usr/bin/env python3
    
    import scanpy as sc
    import pandas as pd
    import matplotlib.pyplot as plt
    
    sc.settings.verbosity = 3
    sc.settings.set_figure_params(dpi=80, facecolor='white')
    
    # Load data
    adata = sc.read_10x_mtx('${count_matrix}', var_names='gene_symbols', cache=True)
    
    # Basic filtering
    sc.pp.filter_cells(adata, min_genes=${min_genes})
    sc.pp.filter_genes(adata, min_cells=${min_cells})
    
    # Calculate mitochondrial genes
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    # QC plots
    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True, save='_qc.pdf')
    
    # Filter
    adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    adata = adata[adata.obs.pct_counts_mt < ${max_mito_percent}, :]
    
    # Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Find variable genes
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    
    # Regress out
    sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
    sc.pp.scale(adata, max_value=10)
    
    # PCA
    sc.tl.pca(adata, svd_solver='arpack')
    
    # Neighbors and clustering
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.leiden(adata, resolution=${resolution})
    
    # UMAP
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=['leiden'], save='_clusters.pdf')
    
    # Find marker genes
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
    sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, save='_markers.pdf')
    
    # Export markers
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    markers_df = pd.DataFrame({
        group + '_' + key[:1]: result[key][group]
        for group in groups for key in ['names', 'scores', 'pvals', 'pvals_adj', 'logfoldchanges']
    })
    markers_df.to_csv('marker_genes.csv')
    
    # Save
    adata.write('adata.h5ad')
    
    # Rename plots
    import os
    os.rename('figures/violin_qc.pdf', 'qc_metrics.pdf')
    os.rename('figures/umap_clusters.pdf', 'umap.pdf')
    os.rename('figures/rank_genes_groups_leiden_markers.pdf', 'rank_genes.pdf')
    """
}

/*
 * Workflow: Scanpy scRNA-seq analysis
 */
workflow SCANPY_PIPELINE {
    take:
    count_matrix       // path: 10X count matrix directory
    min_genes          // val: minimum genes per cell
    min_cells          // val: minimum cells per gene
    max_mito_percent   // val: maximum mitochondrial percentage
    resolution         // val: clustering resolution
    
    main:
    SCANPY_ANALYSIS(count_matrix, min_genes, min_cells, max_mito_percent, resolution)
    
    emit:
    adata = SCANPY_ANALYSIS.out.adata
    umap = SCANPY_ANALYSIS.out.umap
    markers = SCANPY_ANALYSIS.out.markers
}
