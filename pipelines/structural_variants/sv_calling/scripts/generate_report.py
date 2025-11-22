#!/usr/bin/env python3
"""
Generate HTML report for SV calling pipeline
"""
import os
from datetime import datetime

def generate_report(vcf_files, plot_html, output_html):
    """Generate comprehensive HTML report"""
    
    # Count SVs per sample
    sv_counts = {}
    for vcf in vcf_files:
        sample = os.path.basename(vcf).replace('.annotated.vcf.gz', '')
        count = 0
        import gzip
        with gzip.open(vcf, 'rt') as f:
            for line in f:
                if not line.startswith('#'):
                    count += 1
        sv_counts[sample] = count
    
    # Read plot HTML
    plot_content = ""
    if os.path.exists(plot_html):
        with open(plot_html, 'r') as f:
            plot_content = f.read()
    
    # Generate report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Structural Variants Detection Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .summary {{ background-color: #ecf0f1; padding: 20px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Structural Variants Detection Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <h2>Summary</h2>
            <p>Total Samples: {len(sv_counts)}</p>
            <p>Total SVs Detected: {sum(sv_counts.values())}</p>
        </div>
        
        <h2>SVs per Sample</h2>
        <table>
            <tr><th>Sample</th><th>SV Count</th></tr>
            {''.join(f'<tr><td>{s}</td><td>{c}</td></tr>' for s, c in sv_counts.items())}
        </table>
        
        <h2>Visualization</h2>
        {plot_content}
        
        <h2>Pipeline Information</h2>
        <ul>
            <li>Callers used: DELLY, Manta, LUMPY</li>
            <li>Consensus method: SURVIVOR (min 2 callers)</li>
            <li>Min SV size: 50 bp</li>
            <li>Max SV size: 1,000,000 bp</li>
        </ul>
    </body>
    </html>
    """
    
    with open(output_html, 'w') as f:
        f.write(html)
    
    print(f"Report saved to {output_html}")

if __name__ == '__main__':
    vcf_files = snakemake.input[:-1]  # All except last (plot HTML)
    plot_html = snakemake.input[-1]
    output_html = snakemake.output[0]
    generate_report(vcf_files, plot_html, output_html)
