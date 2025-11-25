/*
 * Seurat Module
 * 
 * Seurat - Single-cell RNA-seq analysis (R-based)
 * QC, normalization, clustering, and visualization
 * Uses existing scrna-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * Seurat QC and normalization
 */
process SEURAT_QC_NORMALIZE {
    tag "seurat_qc"
    container "${params.containers.scrnaseq}"
    
    publishDir "${params.outdir}/scrna/seurat", mode: 'copy'
    
    cpus params.seurat?.cpus ?: 8
    memory params.seurat?.memory ?: '32.GB'
    
    input:
    path count_matrix
    val min_features
    val max_features
    val max_mito_percent
    
    output:
    path "seurat_qc.rds", emit: seurat_object
    path "qc_violin.pdf", emit: qc_violin
    path "feature_scatter.pdf", emit: feature_scatter
    
    script:
    """
    #!/usr/bin/env Rscript
    
    library(Seurat)
    library(ggplot2)
    
    # Load data
    counts <- Read10X(data.dir = "${count_matrix}")
    seurat_obj <- CreateSeuratObject(counts = counts, project = "scRNA", min.cells = 3, min.features = 200)
    
    # Calculate mitochondrial percentage
    seurat_obj[["percent.mt"]] <- PercentageFeatureSet(seurat_obj, pattern = "^MT-")
    
    # QC plots
    pdf("qc_violin.pdf")
    VlnPlot(seurat_obj, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3)
    dev.off()
    
    pdf("feature_scatter.pdf")
    plot1 <- FeatureScatter(seurat_obj, feature1 = "nCount_RNA", feature2 = "percent.mt")
    plot2 <- FeatureScatter(seurat_obj, feature1 = "nCount_RNA", feature2 = "nFeature_RNA")
    print(plot1 + plot2)
    dev.off()
    
    # Filter cells
    seurat_obj <- subset(seurat_obj, subset = nFeature_RNA > ${min_features} & nFeature_RNA < ${max_features} & percent.mt < ${max_mito_percent})
    
    # Normalize
    seurat_obj <- NormalizeData(seurat_obj, normalization.method = "LogNormalize", scale.factor = 10000)
    
    # Find variable features
    seurat_obj <- FindVariableFeatures(seurat_obj, selection.method = "vst", nfeatures = 2000)
    
    # Scale data
    seurat_obj <- ScaleData(seurat_obj)
    
    # Save
    saveRDS(seurat_obj, "seurat_qc.rds")
    """
}

/*
 * Seurat clustering
 */
process SEURAT_CLUSTER {
    tag "seurat_cluster"
    container "${params.containers.scrnaseq}"
    
    publishDir "${params.outdir}/scrna/seurat", mode: 'copy'
    
    cpus params.seurat?.cpus ?: 8
    memory params.seurat?.memory ?: '32.GB'
    
    input:
    path seurat_object
    val resolution
    
    output:
    path "seurat_clustered.rds", emit: seurat_object
    path "umap_clusters.pdf", emit: umap
    path "cluster_markers.csv", emit: markers
    
    script:
    """
    #!/usr/bin/env Rscript
    
    library(Seurat)
    library(ggplot2)
    
    # Load object
    seurat_obj <- readRDS("${seurat_object}")
    
    # PCA
    seurat_obj <- RunPCA(seurat_obj, features = VariableFeatures(object = seurat_obj))
    
    # Clustering
    seurat_obj <- FindNeighbors(seurat_obj, dims = 1:10)
    seurat_obj <- FindClusters(seurat_obj, resolution = ${resolution})
    
    # UMAP
    seurat_obj <- RunUMAP(seurat_obj, dims = 1:10)
    
    # Plot
    pdf("umap_clusters.pdf")
    print(DimPlot(seurat_obj, reduction = "umap"))
    dev.off()
    
    # Find markers
    markers <- FindAllMarkers(seurat_obj, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)
    write.csv(markers, "cluster_markers.csv")
    
    # Save
    saveRDS(seurat_obj, "seurat_clustered.rds")
    """
}

/*
 * Workflow: Seurat scRNA-seq analysis
 */
workflow SEURAT_PIPELINE {
    take:
    count_matrix       // path: 10X count matrix directory
    min_features       // val: minimum features per cell
    max_features       // val: maximum features per cell
    max_mito_percent   // val: maximum mitochondrial percentage
    resolution         // val: clustering resolution
    
    main:
    SEURAT_QC_NORMALIZE(count_matrix, min_features, max_features, max_mito_percent)
    SEURAT_CLUSTER(SEURAT_QC_NORMALIZE.out.seurat_object, resolution)
    
    emit:
    seurat_object = SEURAT_CLUSTER.out.seurat_object
    umap = SEURAT_CLUSTER.out.umap
    markers = SEURAT_CLUSTER.out.markers
}
