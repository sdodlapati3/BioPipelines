#!/bin/bash
# Smart entrypoint for RNA-seq container
# Supports: direct execution, AI agent mode, Snakemake mode

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print usage
usage() {
    cat << EOF
${GREEN}BioPipelines RNA-seq Container v1.0.0${NC}

${YELLOW}Usage Modes:${NC}

1. ${GREEN}Full Pipeline Mode${NC} (Snakemake-based):
   singularity run rna-seq.sif \\
     --input /data/fastq \\
     --output /data/results \\
     --genome hg38 \\
     --threads 8

2. ${GREEN}AI Agent Mode${NC} (JSON config):
   singularity run rna-seq.sif --config config.json
   
   Example config.json:
   {
     "input_dir": "/data/fastq",
     "output_dir": "/data/results",
     "genome": "hg38",
     "strandedness": "reverse",
     "threads": 8,
     "samples": ["sample1", "sample2"]
   }

3. ${GREEN}Direct Tool Execution${NC}:
   singularity exec rna-seq.sif fastqc sample.fastq.gz
   singularity exec rna-seq.sif salmon quant ...

4. ${GREEN}Interactive Mode${NC}:
   singularity shell rna-seq.sif

${YELLOW}Parameters:${NC}
  --input DIR        Input directory with FASTQ files
  --output DIR       Output directory (will be created)
  --genome STR       Reference genome (hg38, mm10, etc.)
  --annotation FILE  GTF annotation file
  --threads INT      Number of threads (default: 8)
  --strandedness     unstranded|forward|reverse (default: unstranded)
  --config FILE      JSON configuration for AI agents
  --help            Show this help message

${YELLOW}Environment Variables:${NC}
  BIOPIPELINES_GENOME_DIR   Directory containing reference genomes
  BIOPIPELINES_THREADS      Default thread count

${YELLOW}Examples:${NC}
  # Quick QC only
  singularity run rna-seq.sif --input /data/fastq --output /data/qc --genome hg38 --only-qc

  # Full analysis with custom annotation
  singularity run rna-seq.sif \\
    --input /data/fastq \\
    --output /data/results \\
    --genome hg38 \\
    --annotation /refs/gencode.v44.gtf \\
    --threads 16

  # AI agent mode
  singularity run rna-seq.sif --config analysis_config.json
EOF
    exit 0
}

# Parse arguments
INPUT_DIR=""
OUTPUT_DIR=""
GENOME=""
ANNOTATION=""
THREADS="${BIOPIPELINES_THREADS:-8}"
STRANDEDNESS="unstranded"
CONFIG_FILE=""
ONLY_QC=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            usage
            ;;
        --input)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --genome)
            GENOME="$2"
            shift 2
            ;;
        --annotation)
            ANNOTATION="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --strandedness)
            STRANDEDNESS="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --only-qc)
            ONLY_QC=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# AI Agent mode
if [[ -n "$CONFIG_FILE" ]]; then
    echo -e "${GREEN}Running in AI Agent mode${NC}"
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
        exit 1
    fi
    
    # Parse JSON config (requires jq or python)
    if command -v python3 &> /dev/null; then
        INPUT_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['input_dir'])")
        OUTPUT_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['output_dir'])")
        GENOME=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['genome'])")
        THREADS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('threads', 8))")
    else
        echo -e "${RED}Error: python3 required for AI agent mode${NC}"
        exit 1
    fi
fi

# Validate required parameters
if [[ -z "$INPUT_DIR" ]] || [[ -z "$OUTPUT_DIR" ]] || [[ -z "$GENOME" ]]; then
    echo -e "${RED}Error: Missing required parameters${NC}"
    echo "Required: --input, --output, --genome"
    echo "Run with --help for usage information"
    exit 1
fi

# Validate input directory
if [[ ! -d "$INPUT_DIR" ]]; then
    echo -e "${RED}Error: Input directory not found: $INPUT_DIR${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}BioPipelines RNA-seq Analysis${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo "Input:        $INPUT_DIR"
echo "Output:       $OUTPUT_DIR"
echo "Genome:       $GENOME"
echo "Annotation:   ${ANNOTATION:-auto-detect}"
echo "Threads:      $THREADS"
echo "Strandedness: $STRANDEDNESS"
echo -e "${GREEN}═══════════════════════════════════════${NC}"

# Export variables for Snakemake
export INPUT_DIR OUTPUT_DIR GENOME ANNOTATION THREADS STRANDEDNESS

# Run pipeline
cd /analysis
exec /opt/biopipelines/run_pipeline.sh
