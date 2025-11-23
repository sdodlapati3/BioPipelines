FROM mambaorg/micromamba:1.5.8

# Container metadata (OCI standard)
LABEL org.opencontainers.image.title="BioPipelines Base Container"
LABEL org.opencontainers.image.description="Foundation container with common bioinformatics tools"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.authors="BioPipelines Team <sdodlapa@github>"
LABEL org.opencontainers.image.source="https://github.com/sdodlapa/BioPipelines"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.created="2025-11-23"

# AI Agent metadata (custom labels)
LABEL ai.agent.category="bioinformatics"
LABEL ai.agent.tier="base"
LABEL ai.agent.capabilities="samtools,bcftools,bedtools,fastqc"

USER root

# Install common tools shared across all pipelines
RUN micromamba install -y -n base -c conda-forge -c bioconda \
    # Core sequence manipulation
    samtools=1.17 \
    bcftools=1.17 \
    htslib=1.17 \
    bedtools=2.30.0 \
    # Universal QC
    fastqc=0.12.1 \
    multiqc=1.14 \
    # Python essentials
    python=3.11 \
    numpy \
    pandas \
    scipy \
    matplotlib \
    seaborn \
    biopython \
    pysam \
    pybedtools \
    # Utilities
    parallel \
    pigz \
    && micromamba clean --all --yes

# Configure environment
ENV PATH="/opt/conda/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Create working directories
RUN mkdir -p /analysis /data /results && \
    chmod -R 777 /analysis /data /results

WORKDIR /analysis

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD samtools --version || exit 1

# Default command
CMD ["/bin/bash"]
