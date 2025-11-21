#!/usr/bin/env python3
"""
Generate comprehensive HTML report
"""

import sys
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Snakemake inputs/outputs
input_h5ad = snakemake.input.h5ad
qc_metrics = snakemake.input.qc_metrics
deg_results = snakemake.input.deg
output_html = snakemake.output.html

print("Generating comprehensive scRNA-seq analysis report...")

# Load data
adata_list = [sc.read_h5ad(f) for f in input_h5ad]
qc_list = [pd.read_csv(f) for f in qc_metrics]
deg_list = [pd.read_csv(f) for f in deg_results]

# Start HTML
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Single-cell RNA-seq Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 30px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        h3 {{
            color: #764ba2;
            margin-top: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric-box {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            margin: 10px;
            border-radius: 8px;
            min-width: 200px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metric-box h3 {{
            margin: 0 0 10px 0;
            color: white;
            font-size: 0.9em;
            text-transform: uppercase;
            opacity: 0.9;
        }}
        .metric-box .value {{
            font-size: 2em;
            font-weight: bold;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            color: #666;
        }}
        .warning {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }}
        .success {{
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧬 Single-cell RNA-seq Analysis Report</h1>
        <p>Comprehensive analysis of single-cell transcriptomics data</p>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
"""

# Summary section
total_cells = sum([adata.n_obs for adata in adata_list])
total_genes = adata_list[0].n_vars  # Assuming same genes across samples

html_content += f"""
    <div class="section">
        <h2>📊 Dataset Summary</h2>
        <div class="stats-grid">
            <div class="metric-box">
                <h3>Total Samples</h3>
                <div class="value">{len(adata_list)}</div>
            </div>
            <div class="metric-box">
                <h3>Total Cells</h3>
                <div class="value">{total_cells:,}</div>
            </div>
            <div class="metric-box">
                <h3>Total Genes</h3>
                <div class="value">{total_genes:,}</div>
            </div>
"""

# Cell type statistics
if 'celltype' in adata_list[0].obs.columns:
    n_celltypes = adata_list[0].obs['celltype'].nunique()
    html_content += f"""
            <div class="metric-box">
                <h3>Cell Types</h3>
                <div class="value">{n_celltypes}</div>
            </div>
"""

html_content += """
        </div>
    </div>
"""

# Sample-level statistics
html_content += """
    <div class="section">
        <h2>🔬 Sample Statistics</h2>
        <table>
            <tr>
                <th>Sample</th>
                <th>Cells</th>
                <th>Median UMI</th>
                <th>Median Genes</th>
                <th>Median Mito %</th>
                <th>Cell Types</th>
            </tr>
"""

for idx, (adata, qc_df) in enumerate(zip(adata_list, qc_list)):
    sample_name = f"sample{idx+1}"
    n_cells = adata.n_obs
    median_umi = qc_df['total_counts'].median()
    median_genes = qc_df['n_genes_by_counts'].median()
    median_mito = qc_df['pct_counts_mt'].median()
    
    if 'celltype' in adata.obs.columns:
        cell_types = ', '.join(adata.obs['celltype'].unique())
    else:
        cell_types = "N/A"
    
    html_content += f"""
            <tr>
                <td><strong>{sample_name}</strong></td>
                <td>{n_cells:,}</td>
                <td>{median_umi:.0f}</td>
                <td>{median_genes:.0f}</td>
                <td>{median_mito:.2f}%</td>
                <td>{cell_types}</td>
            </tr>
"""

html_content += """
        </table>
    </div>
"""

# Cell type distribution
if 'celltype' in adata_list[0].obs.columns:
    html_content += """
    <div class="section">
        <h2>🧫 Cell Type Distribution</h2>
        <table>
            <tr>
                <th>Cell Type</th>
                <th>Number of Cells</th>
                <th>Percentage</th>
                <th>Mean UMI</th>
                <th>Mean Genes</th>
            </tr>
"""
    
    for adata in adata_list:
        celltype_stats = adata.obs.groupby('celltype').agg({
            'total_counts': ['count', 'mean'],
            'n_genes_by_counts': 'mean'
        }).round(2)
        
        for celltype in celltype_stats.index:
            n_cells = int(celltype_stats.loc[celltype, ('total_counts', 'count')])
            pct = n_cells / adata.n_obs * 100
            mean_umi = celltype_stats.loc[celltype, ('total_counts', 'mean')]
            mean_genes = celltype_stats.loc[celltype, ('n_genes_by_counts', 'mean')]
            
            html_content += f"""
            <tr>
                <td><strong>{celltype}</strong></td>
                <td>{n_cells:,}</td>
                <td>{pct:.2f}%</td>
                <td>{mean_umi:.0f}</td>
                <td>{mean_genes:.0f}</td>
            </tr>
"""
    
    html_content += """
        </table>
    </div>
"""

# Differential expression summary
html_content += """
    <div class="section">
        <h2>📈 Differential Expression Summary</h2>
"""

total_degs = 0
for deg_df in deg_list:
    sig_degs = deg_df[
        (deg_df['pvals_adj'] < 0.05) &
        (np.abs(deg_df['logfoldchanges']) > 0.5)
    ]
    total_degs += len(sig_degs)

html_content += f"""
        <div class="success">
            <strong>Total Significant DEGs:</strong> {total_degs:,} genes (padj < 0.05, |logFC| > 0.5)
        </div>
        
        <h3>Top Differentially Expressed Genes per Cell Type</h3>
        <table>
            <tr>
                <th>Cell Type</th>
                <th>Gene</th>
                <th>Log2 FC</th>
                <th>Adj. P-value</th>
            </tr>
"""

for deg_df in deg_list:
    if 'celltype' in deg_df.columns:
        for celltype in deg_df['celltype'].unique():
            ct_degs = deg_df[deg_df['celltype'] == celltype].nsmallest(5, 'pvals_adj')
            
            for _, gene in ct_degs.iterrows():
                html_content += f"""
            <tr>
                <td><strong>{celltype}</strong></td>
                <td><code>{gene['names']}</code></td>
                <td>{gene['logfoldchanges']:.2f}</td>
                <td>{gene['pvals_adj']:.2e}</td>
            </tr>
"""

html_content += """
        </table>
    </div>
"""

# Analysis pipeline info
html_content += f"""
    <div class="section">
        <h2>⚙️ Analysis Pipeline</h2>
        <h3>Processing Steps</h3>
        <ol>
            <li><strong>Quality Control:</strong> Cell filtering based on UMI counts, gene counts, and mitochondrial content</li>
            <li><strong>Doublet Detection:</strong> Identification of potential doublets using Scrublet</li>
            <li><strong>Normalization:</strong> Total count normalization and log-transformation</li>
            <li><strong>Feature Selection:</strong> Identification of highly variable genes (HVGs)</li>
            <li><strong>Dimensionality Reduction:</strong> PCA and UMAP embedding</li>
            <li><strong>Clustering:</strong> Leiden and Louvain clustering algorithms</li>
            <li><strong>Cell Type Annotation:</strong> Marker gene-based annotation</li>
            <li><strong>Differential Expression:</strong> Wilcoxon rank-sum test between cell types</li>
        </ol>
        
        <h3>Software Tools</h3>
        <ul>
            <li><strong>STARsolo:</strong> Alignment and UMI counting</li>
            <li><strong>Scanpy:</strong> Single-cell analysis (v{sc.__version__})</li>
            <li><strong>Scrublet:</strong> Doublet detection</li>
        </ul>
    </div>
"""

# Footer
html_content += f"""
    <div class="footer">
        <p>Report generated by BioPipelines scRNA-seq workflow</p>
        <p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>
"""

# Write HTML file
with open(output_html, 'w') as f:
    f.write(html_content)

print(f"\nReport saved to {output_html}")
print("Report generation complete!")
