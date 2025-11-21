#!/usr/bin/env python3
"""
BioPipelines Data Download CLI
===============================

Command-line interface for downloading genomics datasets.

Usage examples:
--------------
# Download from SRA
biopipes-download sra SRR891268 --type atac_seq

# Download from ENCODE
biopipes-download encode ENCFF001NQP --type chip_seq

# Search datasets
biopipes-download search --source encode --query "H3K4me3" --organism human

# Download entire experiment
biopipes-download encode-experiment ENCSR000AED --type rna_seq
"""

import argparse
import sys
import logging
from pathlib import Path

from biopipelines.data_download import DataDownloader, DatasetType, DataSource


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def cmd_download_sra(args):
    """Download from SRA"""
    downloader = DataDownloader(args.output_dir)
    files = downloader.download_sra(
        args.accession,
        args.type,
        use_aspera=args.aspera
    )
    
    print(f"\n✓ Downloaded {len(files)} files:")
    for f in files:
        print(f"  - {f}")


def cmd_download_encode(args):
    """Download from ENCODE"""
    downloader = DataDownloader(args.output_dir)
    file_path = downloader.download_encode(
        args.file_id,
        args.type
    )
    
    print(f"\n✓ Downloaded: {file_path}")


def cmd_download_encode_experiment(args):
    """Download entire ENCODE experiment"""
    from biopipelines.data_download.encode_downloader import ENCODEDownloader
    
    downloader = ENCODEDownloader(Path(args.output_dir))
    files = downloader.download_experiment(
        args.experiment_id,
        args.type,
        file_format=args.format
    )
    
    print(f"\n✓ Downloaded {len(files)} files:")
    for f in files:
        print(f"  - {f}")


def cmd_search(args):
    """Search for datasets"""
    downloader = DataDownloader()
    results = downloader.search_datasets(
        query=args.query,
        source=args.source,
        dataset_type=args.type,
        organism=args.organism,
        limit=args.limit
    )
    
    print(f"\n📊 Found {len(results)} datasets:\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.get('accession', 'N/A')}")
        print(f"   Title: {result.get('title', result.get('assay_title', 'N/A'))}")
        print(f"   Organism: {result.get('organism', 'N/A')}")
        print(f"   Type: {result.get('experiment_type', result.get('assay_title', 'N/A'))}")
        if 'url' in result:
            print(f"   URL: {result['url']}")
        print()


def cmd_list_sources(args):
    """List available data sources"""
    print("\n📚 Available Data Sources:\n")
    
    sources = [
        ("ENCODE", "ENCODE Project", "https://www.encodeproject.org/"),
        ("SRA", "NCBI Sequence Read Archive", "https://www.ncbi.nlm.nih.gov/sra"),
        ("ENA", "European Nucleotide Archive", "https://www.ebi.ac.uk/ena"),
        ("GEO", "Gene Expression Omnibus", "https://www.ncbi.nlm.nih.gov/geo/"),
        ("1000 Genomes", "1000 Genomes Project", "http://www.internationalgenome.org/")
    ]
    
    for name, full_name, url in sources:
        print(f"  • {name:15} - {full_name}")
        print(f"    {url}\n")


def main():
    parser = argparse.ArgumentParser(
        description="BioPipelines Data Download Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        default='data/raw',
        help='Output directory (default: data/raw)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # SRA download
    sra_parser = subparsers.add_parser('sra', help='Download from SRA/ENA')
    sra_parser.add_argument('accession', help='SRA accession (SRR, ERR, DRR)')
    sra_parser.add_argument('--type', required=True, 
                           choices=[t.value for t in DatasetType],
                           help='Dataset type')
    sra_parser.add_argument('--aspera', action='store_true',
                           help='Use Aspera for faster download')
    sra_parser.set_defaults(func=cmd_download_sra)
    
    # ENCODE download
    encode_parser = subparsers.add_parser('encode', help='Download from ENCODE')
    encode_parser.add_argument('file_id', help='ENCODE file ID (e.g., ENCFF001NQP)')
    encode_parser.add_argument('--type', required=True,
                              choices=[t.value for t in DatasetType],
                              help='Dataset type')
    encode_parser.set_defaults(func=cmd_download_encode)
    
    # ENCODE experiment download
    exp_parser = subparsers.add_parser('encode-experiment', 
                                       help='Download entire ENCODE experiment')
    exp_parser.add_argument('experiment_id', help='ENCODE experiment ID (e.g., ENCSR000AED)')
    exp_parser.add_argument('--type', required=True,
                           choices=[t.value for t in DatasetType],
                           help='Dataset type')
    exp_parser.add_argument('--format', default='fastq',
                           help='File format to download (default: fastq)')
    exp_parser.set_defaults(func=cmd_download_encode_experiment)
    
    # Search
    search_parser = subparsers.add_parser('search', help='Search for datasets')
    search_parser.add_argument('--query', required=True, help='Search query')
    search_parser.add_argument('--source', required=True,
                              choices=[s.value for s in DataSource],
                              help='Data source to search')
    search_parser.add_argument('--type',
                              choices=[t.value for t in DatasetType],
                              help='Filter by dataset type')
    search_parser.add_argument('--organism', default='human',
                              help='Organism (default: human)')
    search_parser.add_argument('--limit', type=int, default=10,
                              help='Maximum results (default: 10)')
    search_parser.set_defaults(func=cmd_search)
    
    # List sources
    list_parser = subparsers.add_parser('list-sources', help='List available data sources')
    list_parser.set_defaults(func=cmd_list_sources)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    setup_logging(args.verbose)
    
    try:
        args.func(args)
    except Exception as e:
        logging.error(f"Error: {e}")
        if args.verbose:
            raise
        sys.exit(1)


if __name__ == '__main__':
    main()
