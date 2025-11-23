FROM biopipelines/base:1.0.0

# Container metadata
LABEL org.opencontainers.image.title="BioPipelines RNA-seq Pipeline"
LABEL org.opencontainers.image.description="Complete RNA-seq analysis from FASTQ to differential expression"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.authors="BioPipelines Team <sdodlapa@github>"
LABEL org.opencontainers.image.source="https://github.com/sdodlapa/BioPipelines"
LABEL org.opencontainers.image.licenses="MIT"

# AI Agent metadata
LABEL ai.agent.category="transcriptomics"
LABEL ai.agent.tier="pipeline"
LABEL ai.agent.capabilities="qc,trimming,alignment,quantification,differential_expression"
LABEL ai.agent.input="fastq,fastq.gz"
LABEL ai.agent.output="bam,counts,deseq2_results"

USER root

# Install RNA-seq specific tools
RUN micromamba install -y -n base -c conda-forge -c bioconda \
    # Preprocessing
    fastp=0.23.4 \
    trimmomatic=0.39 \
    cutadapt=4.4 \
    # Alignment
    star=2.7.10b \
    hisat2=2.2.1 \
    # Quantification
    salmon=1.10.1 \
    subread=2.0.3 \
    htseq=2.0.2 \
    # R and Bioconductor for DE analysis
    r-base=4.2 \
    bioconductor-deseq2 \
    bioconductor-edger \
    bioconductor-limma \
    r-ggplot2 \
    r-pheatmap \
    r-rcolorbrewer \
    r-tidyverse \
    && micromamba clean --all --yes

# Copy pipeline scripts
COPY entrypoint.sh /opt/biopipelines/entrypoint.sh
COPY run_pipeline.sh /opt/biopipelines/run_pipeline.sh
RUN chmod +x /opt/biopipelines/*.sh

# Copy Snakefile and config
RUN mkdir -p /opt/biopipelines/rna_seq
COPY ../../pipelines/rna_seq/Snakefile /opt/biopipelines/rna_seq/ || echo "Snakefile will be mounted at runtime"
COPY ../../pipelines/rna_seq/config.yaml /opt/biopipelines/rna_seq/ || echo "Config will be mounted at runtime"

# Environment variables
ENV PIPELINE_NAME="rna-seq"
ENV PIPELINE_VERSION="1.0.0"

WORKDIR /analysis

# Entrypoint for AI agent compatibility
ENTRYPOINT ["/opt/biopipelines/entrypoint.sh"]
CMD ["--help"]
