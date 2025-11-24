#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Exercise 3: Using meta maps (the pattern we'll use for real pipelines)
process processWithMeta {
    tag "$meta.id"
    
    input:
    tuple val(meta), path(reads)

    output:
    tuple val(meta), path("${meta.id}_processed.txt")

    script:
    """
    echo "Processing sample: ${meta.id}" > ${meta.id}_processed.txt
    echo "Condition: ${meta.condition}" >> ${meta.id}_processed.txt
    echo "Single-end: ${meta.single_end}" >> ${meta.id}_processed.txt
    echo "Read count: \$(wc -l < $reads)" >> ${meta.id}_processed.txt
    """
}

workflow {
    // Create sample data with metadata (this is the pattern for real workflows)
    Channel.of(
        [
            [id: 'sample1', condition: 'WT', single_end: false],
            'Sample1 data line 1\nSample1 data line 2'
        ],
        [
            [id: 'sample2', condition: 'KO', single_end: false],
            'Sample2 data line 1\nSample2 data line 2\nSample2 data line 3'
        ],
        [
            [id: 'sample3', condition: 'WT', single_end: true],
            'Sample3 data line 1'
        ]
    )
    .map { meta, data ->
        // Create a file from the data string
        def file = file("${workDir}/input_${meta.id}.txt")
        file.text = data
        return [meta, file]
    }
    .set { input_ch }
    
    // Process with metadata
    processWithMeta(input_ch)
    
    // View results
    processWithMeta.out
        | map { meta, file -> "Sample ${meta.id} (${meta.condition}): ${file}" }
        | view
}
