#!/usr/bin/env python3
"""
Generate SV summary plots from VCF files
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
from collections import defaultdict

def parse_vcf(vcf_file):
    """Parse VCF and extract SV information"""
    svs = []
    with gzip.open(vcf_file, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            chrom, pos, _, ref, alt, _, _, info = fields[:8]
            
            # Extract SV type and length from INFO
            sv_type = 'UNK'
            sv_len = 0
            for item in info.split(';'):
                if item.startswith('SVTYPE='):
                    sv_type = item.split('=')[1]
                elif item.startswith('SVLEN='):
                    sv_len = abs(int(item.split('=')[1]))
            
            svs.append({
                'chrom': chrom,
                'pos': int(pos),
                'type': sv_type,
                'length': sv_len
            })
    return pd.DataFrame(svs)

def plot_sv_summary(vcf_files, output_html):
    """Generate summary plots"""
    # Combine all samples
    all_svs = []
    for vcf in vcf_files:
        sample = vcf.split('/')[-1].replace('.annotated.vcf.gz', '')
        df = parse_vcf(vcf)
        df['sample'] = sample
        all_svs.append(df)
    
    df_all = pd.concat(all_svs, ignore_index=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Structural Variant Summary', fontsize=16)
    
    # Plot 1: SV types
    sv_counts = df_all['type'].value_counts()
    axes[0, 0].bar(sv_counts.index, sv_counts.values)
    axes[0, 0].set_xlabel('SV Type')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('SV Type Distribution')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: SV length distribution
    df_len = df_all[df_all['length'] > 0]
    axes[0, 1].hist(df_len['length'], bins=50, edgecolor='black')
    axes[0, 1].set_xlabel('SV Length (bp)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('SV Size Distribution')
    axes[0, 1].set_xscale('log')
    
    # Plot 3: SVs per chromosome
    chrom_counts = df_all['chrom'].value_counts()
    axes[1, 0].bar(range(len(chrom_counts)), chrom_counts.values)
    axes[1, 0].set_xlabel('Chromosome')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('SVs per Chromosome')
    axes[1, 0].set_xticks(range(len(chrom_counts)))
    axes[1, 0].set_xticklabels(chrom_counts.index, rotation=45, ha='right')
    
    # Plot 4: Summary statistics
    axes[1, 1].axis('off')
    stats_text = f"""
    Total SVs: {len(df_all)}
    Samples: {df_all['sample'].nunique()}
    
    SV Types:
    {sv_counts.to_string()}
    
    Size Stats (bp):
    Mean: {df_len['length'].mean():.0f}
    Median: {df_len['length'].median():.0f}
    Max: {df_len['length'].max():.0f}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                    verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_html.replace('.html', '.png'), dpi=150, bbox_inches='tight')
    
    # Create simple HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head><title>SV Summary</title></head>
    <body>
        <h1>Structural Variant Analysis Summary</h1>
        <img src="sv_summary.png" width="100%">
        <h2>Statistics</h2>
        <pre>{stats_text}</pre>
    </body>
    </html>
    """
    
    with open(output_html, 'w') as f:
        f.write(html_content)
    
    print(f"Summary plots saved to {output_html}")

if __name__ == '__main__':
    import sys
    vcf_files = snakemake.input
    output_html = snakemake.output[0]
    plot_sv_summary(vcf_files, output_html)
