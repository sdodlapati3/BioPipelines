#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Import modules
include { FASTQC } from './nextflow-modules/fastqc/main.nf'
include { BISMARK_ALIGN } from './nextflow-modules/bismark/main.nf'
include { MULTIQC; MULTIQC_CUSTOM } from './nextflow-modules/multiqc/main.nf'
include { TRIM_GALORE } from './nextflow-modules/trim_galore/main.nf'

// Define parameters
params.reads = './data/*.fastq'
params.genome = './genome/'
params.outdir = './results'

// Define input channel
Channel.fromPath(params.reads)
    .set { reads_ch }

// Quality control with FastQC
reads_ch
    | FASTQC

// Trim reads with Trim Galore
reads_ch
    | TRIM_GALORE
    | map { file -> file[0] }
    | set { trimmed_reads_ch }

// Align reads with Bismark
trimmed_reads_ch
    | BISMARK_ALIGN {
        genome = params.genome
    }
    | set { aligned_reads_ch }

// Aggregate results with MultiQC
aligned_reads_ch
    | MULTIQC {
        outdir = params.outdir
    }